"""
app/services/document_loader.py  (v2 — RSS-based, reliable)
=============================================================
Menggunakan RSS Feed resmi — jauh lebih reliable dari scraping HTML
karena RSS memang dibuat untuk dibaca programatically.

Sources:
  1. Kontan Investasi RSS  → https://investasi.kontan.co.id/rss
  2. Kontan.co.id RSS      → https://www.kontan.co.id/rss/investasi
  3. CNBC Indonesia RSS    → https://www.cnbcindonesia.com/RSS (market)
  4. Yahoo Finance RSS     → search berita per ticker (gratis)
  5. PDF parser            → PyMuPDF untuk laporan keuangan

Install tambahan:
  pip install feedparser
"""

import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import feedparser
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/rss+xml, application/xml, text/xml, */*",
}

CHUNK_SIZE    = 400
CHUNK_OVERLAP = 80

# ─── RSS FEEDS ───────────────────────────────
RSS_FEEDS = {
    "kontan_investasi": "https://investasi.kontan.co.id/rss",
    "kontan_main":      "https://www.kontan.co.id/rss/investasi",
    "cnbc_market":      "https://www.cnbcindonesia.com/RSS/pasar-modal.xml",
    "cnbc_news":        "https://www.cnbcindonesia.com/RSS/news.xml",
}

# Yahoo Finance RSS per ticker (selalu jalan, gratis)
def yahoo_rss_url(ticker: str) -> str:
    """Yahoo Finance RSS feed untuk satu ticker."""
    return f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"


# ─────────────────────────────────────────────
# DATA CLASS
# ─────────────────────────────────────────────

@dataclass
class Document:
    content:  str
    source:   str
    ticker:   Optional[str]
    doc_type: str
    title:    str = ""
    date:     str = ""

    def to_metadata(self) -> dict:
        return {
            "source":   self.source,
            "ticker":   self.ticker or "GENERAL",
            "doc_type": self.doc_type,
            "title":    self.title[:200],
            "date":     self.date,
        }


# ─────────────────────────────────────────────
# TEXT UTILITIES
# ─────────────────────────────────────────────

def clean_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)          # strip HTML tags
    text = re.sub(r"&[a-z]+;", " ", text)          # strip HTML entities
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    if len(text) <= chunk_size:
        return [text] if len(text) > 50 else []

    chunks = []
    start  = 0
    while start < len(text):
        end = start + chunk_size
        if end < len(text):
            space_idx = text.rfind(" ", start, end)
            if space_idx > start:
                end = space_idx
        chunks.append(text[start:end].strip())
        start = end - overlap

    return [c for c in chunks if len(c) > 50]


# ─────────────────────────────────────────────
# RSS FETCHER (universal)
# ─────────────────────────────────────────────

def fetch_rss(url: str, keyword: str = None, max_items: int = 10) -> list[dict]:
    """
    Fetch dan parse RSS feed. Filter by keyword jika diberikan.

    Returns list of dicts: {title, summary, link, published}
    """
    try:
        feed = feedparser.parse(url, request_headers=HEADERS)

        if feed.bozo and not feed.entries:
            logger.warning(f"[loader] RSS parse warning for {url}: {feed.bozo_exception}")
            return []

        items = []
        for entry in feed.entries[:max_items * 3]:  # ambil lebih banyak sebelum filter
            title   = entry.get("title", "")
            summary = entry.get("summary", "") or entry.get("description", "")
            link    = entry.get("link", "")
            pub     = entry.get("published", "") or entry.get("updated", "")

            # Bersihkan HTML dari summary
            summary = clean_text(summary)
            title   = clean_text(title)

            # Filter by keyword kalau ada
            if keyword:
                kw_lower = keyword.lower()
                combined = (title + " " + summary).lower()
                if kw_lower not in combined:
                    continue

            if len(title) < 5:
                continue

            items.append({
                "title":     title,
                "summary":   summary,
                "link":      link,
                "published": pub,
            })

            if len(items) >= max_items:
                break

        logger.info(f"[loader] RSS {url[:50]}: {len(items)} items (keyword='{keyword}')")
        return items

    except Exception as e:
        logger.warning(f"[loader] RSS fetch error {url}: {e}")
        return []


def fetch_article_content(url: str) -> str:
    """
    Fetch full article content dari URL.
    Dipakai untuk memperkaya summary RSS yang biasanya pendek.
    """
    try:
        resp = requests.get(url, headers=HEADERS, timeout=8)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # Coba berbagai selector umum untuk konten artikel
        for selector in [
            "div.detail-text", "div.content-text", "div.article-body",
            "div.detail", "article", "div.post-content",
            "div[class*='content']", "div[class*='article']",
        ]:
            tag = soup.select_one(selector)
            if tag:
                paragraphs = tag.find_all("p")
                text = " ".join(p.get_text(" ", strip=True) for p in paragraphs if len(p.get_text()) > 30)
                if len(text) > 200:
                    return clean_text(text)

        # Fallback: ambil semua <p> di halaman
        paragraphs = soup.find_all("p")
        text = " ".join(p.get_text(" ", strip=True) for p in paragraphs if len(p.get_text()) > 40)
        return clean_text(text)[:3000]  # cap 3000 char

    except Exception as e:
        logger.debug(f"[loader] Article fetch error {url}: {e}")
        return ""


# ─────────────────────────────────────────────
# SOURCE 1: KONTAN RSS
# ─────────────────────────────────────────────

def scrape_kontan_rss(ticker: str, max_items: int = 5) -> list[Document]:
    """
    Ambil berita dari Kontan via RSS feed resmi.
    Filter by ticker name / company name.
    """
    ticker_clean = ticker.replace(".JK", "").upper()
    docs = []

    # Coba beberapa RSS Kontan
    rss_urls = [
        "https://investasi.kontan.co.id/rss",
        "https://www.kontan.co.id/rss/investasi",
        "https://www.kontan.co.id/rss",
    ]

    for rss_url in rss_urls:
        items = fetch_rss(rss_url, keyword=ticker_clean, max_items=max_items)
        for item in items:
            # Gabungkan title + summary sebagai konten dasar
            content = f"{item['title']}. {item['summary']}"

            # Coba fetch full article (optional, boleh gagal)
            if item.get("link") and len(content) < 300:
                full = fetch_article_content(item["link"])
                if full:
                    content = f"{item['title']}. {full}"
                time.sleep(0.3)

            content = clean_text(content)
            if len(content) < 80:
                continue

            docs.append(Document(
                content  = content,
                source   = item.get("link", rss_url),
                ticker   = ticker_clean,
                doc_type = "news",
                title    = item["title"][:200],
                date     = item.get("published", ""),
            ))

        if docs:
            break  # cukup dari satu sumber

    logger.info(f"[loader] Kontan RSS: {len(docs)} berita untuk {ticker_clean}")
    return docs


# ─────────────────────────────────────────────
# SOURCE 2: CNBC INDONESIA RSS
# ─────────────────────────────────────────────

def scrape_cnbc_rss(ticker: str, max_items: int = 5) -> list[Document]:
    """Ambil berita dari CNBC Indonesia via RSS."""
    ticker_clean = ticker.replace(".JK", "").upper()
    docs = []

    rss_urls = [
        "https://www.cnbcindonesia.com/RSS/pasar-modal.xml",
        "https://www.cnbcindonesia.com/RSS/market.xml",
        "https://www.cnbcindonesia.com/RSS/news.xml",
    ]

    for rss_url in rss_urls:
        items = fetch_rss(rss_url, keyword=ticker_clean, max_items=max_items)
        for item in items:
            content = clean_text(f"{item['title']}. {item['summary']}")
            if len(content) < 80:
                continue
            docs.append(Document(
                content  = content,
                source   = item.get("link", rss_url),
                ticker   = ticker_clean,
                doc_type = "news",
                title    = item["title"][:200],
                date     = item.get("published", ""),
            ))

    logger.info(f"[loader] CNBC RSS: {len(docs)} berita untuk {ticker_clean}")
    return docs


# ─────────────────────────────────────────────
# SOURCE 3: YAHOO FINANCE RSS (paling reliable)
# ─────────────────────────────────────────────

def scrape_yahoo_rss(ticker: str, max_items: int = 5) -> list[Document]:
    """
    Yahoo Finance RSS — sangat reliable, selalu jalan.
    Berita dalam Bahasa Inggris tapi konteksnya relevan.
    """
    ticker_clean = ticker.replace(".JK", "").upper()
    ticker_jk    = f"{ticker_clean}.JK"

    url   = yahoo_rss_url(ticker_jk)
    items = fetch_rss(url, max_items=max_items)
    docs  = []

    for item in items:
        content = clean_text(f"{item['title']}. {item['summary']}")
        if len(content) < 50:
            content = item["title"]
        if len(content) < 20:
            continue

        docs.append(Document(
            content  = content,
            source   = item.get("link", url),
            ticker   = ticker_clean,
            doc_type = "news_en",
            title    = item["title"][:200],
            date     = item.get("published", ""),
        ))

    logger.info(f"[loader] Yahoo Finance RSS: {len(docs)} berita untuk {ticker_clean}")
    return docs


# ─────────────────────────────────────────────
# SOURCE 4: PDF PARSER
# ─────────────────────────────────────────────

def parse_pdf(pdf_path: str, ticker: str = "UNKNOWN") -> list[Document]:
    """Parse PDF laporan keuangan menggunakan PyMuPDF."""
    try:
        import fitz
    except ImportError:
        logger.error("[loader] PyMuPDF belum terinstall: pip install pymupdf")
        return []

    path = Path(pdf_path)
    if not path.exists():
        logger.error(f"[loader] PDF tidak ditemukan: {pdf_path}")
        return []

    docs = []
    try:
        pdf = fitz.open(str(path))
        for page_num, page in enumerate(pdf, 1):
            text = clean_text(page.get_text("text"))
            if len(text) < 100:
                continue
            docs.append(Document(
                content  = text,
                source   = f"{path.name}::page_{page_num}",
                ticker   = ticker.replace(".JK", "").upper(),
                doc_type = "pdf",
                title    = f"{path.stem} — Hal. {page_num}",
                date     = "",
            ))
        pdf.close()
        logger.info(f"[loader] PDF '{path.name}': {len(docs)} halaman")
    except Exception as e:
        logger.error(f"[loader] PDF error: {e}")

    return docs


# ─────────────────────────────────────────────
# MAIN LOADER
# ─────────────────────────────────────────────

def load_all_documents(
    ticker: str,
    pdf_paths: list[str] = None,
    max_news: int = 5,
) -> list[Document]:
    """
    Load semua dokumen untuk satu ticker dari semua sumber:
    Kontan RSS + CNBC RSS + Yahoo Finance RSS + PDF (opsional).
    """
    ticker_clean = ticker.replace(".JK", "").upper()
    all_docs: list[Document] = []

    # 1. Kontan RSS
    all_docs.extend(scrape_kontan_rss(ticker_clean, max_items=max_news))

    # 2. CNBC Indonesia RSS
    all_docs.extend(scrape_cnbc_rss(ticker_clean, max_items=max_news))

    # 3. Yahoo Finance RSS (fallback paling reliable)
    all_docs.extend(scrape_yahoo_rss(ticker_clean, max_items=max_news))

    # 4. PDF opsional
    for pdf_path in (pdf_paths or []):
        all_docs.extend(parse_pdf(pdf_path, ticker=ticker_clean))

    logger.info(f"[loader] Total {len(all_docs)} dokumen untuk {ticker_clean}")
    return all_docs


def chunk_documents(docs: list[Document]) -> list[tuple[str, dict]]:
    """Chunk semua dokumen → list of (text, metadata) siap ChromaDB."""
    result = []
    for doc in docs:
        for i, chunk in enumerate(chunk_text(doc.content)):
            meta = doc.to_metadata()
            meta["chunk_index"] = i
            result.append((chunk, meta))
    logger.info(f"[loader] {len(result)} chunks siap di-embed")
    return result