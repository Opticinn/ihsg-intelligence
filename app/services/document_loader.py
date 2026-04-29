"""
app/services/document_loader.py  (v4 — TradingView Ideas + Google News)
=========================================================================
Sumber data untuk RAG:
  1. TradingView Ideas  — analisis teknikal komunitas trader IDX
  2. Google News RSS    — berita terbaru per ticker
  3. Yahoo Finance RSS  — backup
  4. PDF parser         — laporan keuangan lokal
"""

import logging
import re
import urllib.parse
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
    "Accept-Language": "id-ID,id;q=0.9,en;q=0.8",
}

CHUNK_SIZE    = 400
CHUNK_OVERLAP = 80

COMPANY_NAMES = {
    "BBCA": "Bank Central Asia BBCA saham",
    "BBRI": "Bank Rakyat Indonesia BBRI saham",
    "BMRI": "Bank Mandiri BMRI saham",
    "BBNI": "Bank Negara Indonesia BBNI saham",
    "ARTO": "Bank Jago ARTO saham",
    "TLKM": "Telkom Indonesia TLKM saham",
    "GOTO": "GoTo Gojek Tokopedia GOTO saham",
    "BUKA": "Bukalapak BUKA saham",
    "ISAT": "Indosat ISAT saham",
    "UNVR": "Unilever Indonesia UNVR saham",
    "ICBP": "Indofood CBP ICBP saham",
    "INDF": "Indofood INDF saham",
    "AMRT": "Alfamart AMRT saham",
    "ADRO": "Adaro Energy ADRO saham",
    "ITMG": "Indo Tambangraya ITMG saham",
    "PTBA": "Bukit Asam PTBA saham",
    "ANTM": "Antam ANTM saham",
    "PGAS": "Perusahaan Gas Negara PGAS saham",
    "ASII": "Astra International ASII saham",
    "JSMR": "Jasa Marga JSMR saham",
    "CPIN": "Charoen Pokphand CPIN saham",
    "KLBF": "Kalbe Farma KLBF saham",
}


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


def clean_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&[a-z]+;", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    if len(text) <= chunk_size:
        return [text] if len(text) > 50 else []
    chunks, start = [], 0
    while start < len(text):
        end = start + chunk_size
        if end < len(text):
            space_idx = text.rfind(" ", start, end)
            if space_idx > start:
                end = space_idx
        chunks.append(text[start:end].strip())
        start = end - overlap
    return [c for c in chunks if len(c) > 50]


# ── SOURCE 1: TRADINGVIEW IDEAS ──

def scrape_tradingview_ideas(ticker: str, max_items: int = 6) -> list[Document]:
    """Scrape analisis komunitas trader dari TradingView Ideas."""
    ticker_clean = ticker.replace(".JK", "").upper()
    url = f"https://id.tradingview.com/symbols/IDX-{ticker_clean}/ideas/"
    docs = []

    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        if resp.status_code != 200:
            logger.warning(f"[loader] TradingView {ticker_clean}: HTTP {resp.status_code}")
            return []

        soup = BeautifulSoup(resp.text, "html.parser")

        # Hapus elemen noise
        for tag in soup.find_all(["nav", "footer", "script", "style"]):
            tag.decompose()

        # Ambil semua text block panjang
        all_text = soup.get_text(separator="\n")
        lines    = [l.strip() for l in all_text.split("\n") if len(l.strip()) > 80]

        NOISE = [
            "Bahasa Indonesia", "Pilih data pasar", "Lebih dari sekadar",
            "TradingView", "oleh ", "Diupdate", "Tampilkan lebih",
            "Pembelian", "Penjualan", "Ide-Ide", "Skrip Pine",
        ]

        current_block = []
        blocks = []

        for line in lines:
            if any(noise in line for noise in NOISE):
                if current_block and len(" ".join(current_block)) > 100:
                    blocks.append(" ".join(current_block))
                current_block = []
                continue
            current_block.append(line)
            if len(" ".join(current_block)) > 800:
                blocks.append(" ".join(current_block))
                current_block = []

        if current_block and len(" ".join(current_block)) > 100:
            blocks.append(" ".join(current_block))

        for i, block in enumerate(blocks[:max_items]):
            content = clean_text(block)
            if len(content) < 80:
                continue
            title = content[:100].split(".")[0]
            docs.append(Document(
                content  = content[:2000],
                source   = url,
                ticker   = ticker_clean,
                doc_type = "tradingview_idea",
                title    = f"Analisis Trader {ticker_clean}: {title}",
                date     = "",
            ))

        logger.info(f"[loader] TradingView Ideas: {len(docs)} analisis untuk {ticker_clean}")

    except Exception as e:
        logger.warning(f"[loader] TradingView Ideas error {ticker_clean}: {e}")

    return docs


# ── SOURCE 2: GOOGLE NEWS RSS ──

def scrape_google_news(ticker: str, max_items: int = 5) -> list[Document]:
    ticker_clean  = ticker.replace(".JK", "").upper()
    query         = COMPANY_NAMES.get(ticker_clean, f"{ticker_clean} saham IDX")
    query_encoded = urllib.parse.quote(query)
    url           = f"https://news.google.com/rss/search?q={query_encoded}&hl=id&gl=ID&ceid=ID:id"
    docs = []
    try:
        feed = feedparser.parse(url, request_headers=HEADERS)
        for entry in feed.entries[:max_items]:
            title   = clean_text(entry.get("title", ""))
            summary = clean_text(entry.get("summary", "") or entry.get("description", ""))
            content = f"{title}. {summary}" if summary else title
            if len(content) < 30:
                continue
            docs.append(Document(
                content  = clean_text(content),
                source   = entry.get("link", url),
                ticker   = ticker_clean,
                doc_type = "news",
                title    = title[:200],
                date     = entry.get("published", ""),
            ))
    except Exception as e:
        logger.warning(f"[loader] Google News error {ticker_clean}: {e}")
    logger.info(f"[loader] Google News: {len(docs)} berita untuk {ticker_clean}")
    return docs


# ── SOURCE 3: YAHOO FINANCE RSS ──

def scrape_yahoo_rss(ticker: str, max_items: int = 3) -> list[Document]:
    ticker_clean = ticker.replace(".JK", "").upper()
    url  = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker_clean}.JK&region=US&lang=en-US"
    docs = []
    try:
        feed = feedparser.parse(url, request_headers=HEADERS)
        for entry in feed.entries[:max_items]:
            title   = clean_text(entry.get("title", ""))
            summary = clean_text(entry.get("summary", ""))
            content = f"{title}. {summary}" if summary else title
            if len(content) < 20:
                continue
            docs.append(Document(
                content  = content,
                source   = entry.get("link", url),
                ticker   = ticker_clean,
                doc_type = "news_en",
                title    = title[:200],
                date     = entry.get("published", ""),
            ))
    except Exception as e:
        logger.warning(f"[loader] Yahoo RSS error {ticker_clean}: {e}")
    return docs


# ── SOURCE 4: PDF ──

def parse_pdf(pdf_path: str, ticker: str = "UNKNOWN") -> list[Document]:
    try:
        import fitz
    except ImportError:
        return []
    path = Path(pdf_path)
    if not path.exists():
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
    except Exception as e:
        logger.error(f"[loader] PDF error: {e}")
    return docs


# ── MAIN ──

def load_all_documents(ticker: str, pdf_paths: list[str] = None, max_news: int = 5) -> list[Document]:
    ticker_clean = ticker.replace(".JK", "").upper()
    all_docs: list[Document] = []

    # 1. TradingView Ideas — analisis paling relevan untuk RAG
    all_docs.extend(scrape_tradingview_ideas(ticker_clean, max_items=6))

    # 2. Google News RSS — berita terbaru
    all_docs.extend(scrape_google_news(ticker_clean, max_items=max_news))

    # 3. Yahoo Finance backup
    if len(all_docs) < 4:
        all_docs.extend(scrape_yahoo_rss(ticker_clean, max_items=3))

    # 4. PDF opsional
    for pdf_path in (pdf_paths or []):
        all_docs.extend(parse_pdf(pdf_path, ticker=ticker_clean))

    logger.info(f"[loader] Total {len(all_docs)} dokumen untuk {ticker_clean}")
    return all_docs


def chunk_documents(docs: list[Document]) -> list[tuple[str, dict]]:
    result = []
    for doc in docs:
        for i, chunk in enumerate(chunk_text(doc.content)):
            meta = doc.to_metadata()
            meta["chunk_index"] = i
            result.append((chunk, meta))
    logger.info(f"[loader] {len(result)} chunks siap di-embed")
    return result