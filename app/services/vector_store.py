"""
app/services/vector_store.py
============================
Manajemen ChromaDB — embed dokumen dan retrieve konteks untuk RAG.

Stack:
  - ChromaDB      : vector database lokal (gratis, no cloud)
  - HuggingFace   : paraphrase-multilingual-MiniLM-L12-v2
                    (support Bahasa Indonesia, ringan ~120MB)

Alur:
  ingest → embed → simpan ke ChromaDB
  query  → embed query → similarity search → return chunks relevan
"""

import logging
import os
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings
from langchain_huggingface import HuggingFaceEmbeddings

from app.services.document_loader import Document, chunk_documents, load_all_documents

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

CHROMA_PATH     = os.getenv("CHROMA_PATH", "chroma_db")        # folder lokal
COLLECTION_NAME = "ihsg_documents"
EMBED_MODEL     = "paraphrase-multilingual-MiniLM-L12-v2"      # support Bahasa Indonesia
TOP_K           = 4    # jumlah chunk yang diambil per query


# ─────────────────────────────────────────────
# SINGLETON CLIENT
# ─────────────────────────────────────────────

_chroma_client:     Optional[chromadb.PersistentClient] = None
_collection:        Optional[chromadb.Collection]       = None
_embedding_model:   Optional[HuggingFaceEmbeddings]     = None


def _get_embedding_model() -> HuggingFaceEmbeddings:
    """Load embedding model sekali, cache di memory."""
    global _embedding_model
    if _embedding_model is None:
        logger.info(f"[vector_store] Loading embedding model: {EMBED_MODEL}...")
        _embedding_model = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        logger.info("[vector_store] Embedding model siap.")
    return _embedding_model


def _get_collection() -> chromadb.Collection:
    """Get atau buat ChromaDB collection."""
    global _chroma_client, _collection

    if _collection is not None:
        return _collection

    Path(CHROMA_PATH).mkdir(parents=True, exist_ok=True)

    _chroma_client = chromadb.PersistentClient(
        path=CHROMA_PATH,
        settings=Settings(anonymized_telemetry=False),
    )

    _collection = _chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},   # cosine similarity untuk teks
    )

    count = _collection.count()
    logger.info(f"[vector_store] Collection '{COLLECTION_NAME}' siap — {count} dokumen tersimpan.")
    return _collection


# ─────────────────────────────────────────────
# INGEST
# ─────────────────────────────────────────────

def ingest_documents(docs: list[Document]) -> int:
    """
    Embed semua dokumen dan simpan ke ChromaDB.

    Args:
        docs: list Document dari document_loader

    Returns:
        Jumlah chunk yang berhasil disimpan.
    """
    if not docs:
        logger.warning("[vector_store] Tidak ada dokumen untuk di-ingest.")
        return 0

    collection = _get_collection()
    embedder   = _get_embedding_model()

    # Chunk semua dokumen
    chunks_and_meta = chunk_documents(docs)

    if not chunks_and_meta:
        logger.warning("[vector_store] Tidak ada chunks yang dihasilkan.")
        return 0

    texts     = [c[0] for c in chunks_and_meta]
    metadatas = [c[1] for c in chunks_and_meta]

    # Buat ID unik per chunk agar tidak duplikat saat ingest ulang
    ids = [
        f"{meta['ticker']}_{meta['doc_type']}_{meta['source'][-30:].replace('/', '_')}_{i}"
        for i, meta in enumerate(metadatas)
    ]
    # Sanitasi ID — ChromaDB tidak suka karakter khusus
    ids = [re.sub(r"[^a-zA-Z0-9_\-]", "_", id_)[:100] for id_ in ids]

    # Embed dalam batch (lebih efisien)
    logger.info(f"[vector_store] Embedding {len(texts)} chunks...")
    embeddings = embedder.embed_documents(texts)

    # Upsert ke ChromaDB (insert atau update jika ID sudah ada)
    collection.upsert(
        ids        = ids,
        documents  = texts,
        embeddings = embeddings,
        metadatas  = metadatas,
    )

    count_after = collection.count()
    logger.info(f"[vector_store] Ingest selesai — total di DB: {count_after} chunks")
    return len(texts)


def ingest_ticker(
    ticker: str,
    pdf_paths: list[str] = None,
    max_news: int = 5,
) -> int:
    """
    Load + embed + simpan semua dokumen untuk satu ticker.
    Shortcut: load_all_documents → ingest_documents.
    """
    docs = load_all_documents(ticker, pdf_paths=pdf_paths, max_news=max_news)
    return ingest_documents(docs)


def ingest_all_tickers(
    tickers: list[str],
    max_news_per_ticker: int = 3,
) -> dict:
    """Ingest dokumen untuk semua ticker sekaligus."""
    results = {}
    for ticker in tickers:
        try:
            n = ingest_ticker(ticker, max_news=max_news_per_ticker)
            results[ticker] = {"status": "ok", "chunks": n}
        except Exception as e:
            logger.error(f"[vector_store] Error ingest {ticker}: {e}")
            results[ticker] = {"status": "error", "message": str(e)}
    return results


# ─────────────────────────────────────────────
# RETRIEVE
# ─────────────────────────────────────────────

def retrieve_context(
    query: str,
    ticker: Optional[str] = None,
    doc_type: Optional[str] = None,
    top_k: int = TOP_K,
) -> list[dict]:
    """
    Cari chunk paling relevan dengan query.

    Args:
        query:    pertanyaan user, e.g. "Bagaimana prospek BBCA?"
        ticker:   filter hanya dokumen ticker ini (opsional)
        doc_type: filter "news" | "pdf" | "idx_announcement" (opsional)
        top_k:    jumlah chunk yang dikembalikan

    Returns:
        List dict, setiap dict berisi: content, source, ticker, score
    """
    collection = _get_collection()
    embedder   = _get_embedding_model()

    if collection.count() == 0:
        logger.warning("[vector_store] ChromaDB kosong! Jalankan ingest dulu.")
        return []

    # Bangun where filter
    where = {}
    if ticker and doc_type:
        ticker_clean = ticker.replace(".JK", "").upper()
        where = {"$and": [{"ticker": ticker_clean}, {"doc_type": doc_type}]}
    elif ticker:
        where = {"ticker": ticker.replace(".JK", "").upper()}
    elif doc_type:
        where = {"doc_type": doc_type}

    # Embed query
    query_embedding = embedder.embed_query(query)

    # Query ChromaDB
    query_kwargs = {
        "query_embeddings": [query_embedding],
        "n_results":        min(top_k, collection.count()),
        "include":          ["documents", "metadatas", "distances"],
    }
    if where:
        query_kwargs["where"] = where

    try:
        results = collection.query(**query_kwargs)
    except Exception as e:
        logger.error(f"[vector_store] Query error: {e}")
        return []

    # Format hasil
    output = []
    docs_list  = results.get("documents",  [[]])[0]
    metas_list = results.get("metadatas",  [[]])[0]
    dists_list = results.get("distances",  [[]])[0]

    for doc, meta, dist in zip(docs_list, metas_list, dists_list):
        output.append({
            "content":  doc,
            "source":   meta.get("source", ""),
            "ticker":   meta.get("ticker", ""),
            "doc_type": meta.get("doc_type", ""),
            "title":    meta.get("title", ""),
            "date":     meta.get("date", ""),
            "score":    round(1 - dist, 4),   # ubah distance → similarity (1 = paling relevan)
        })

    logger.info(f"[vector_store] Query '{query[:50]}...' → {len(output)} chunks ditemukan")
    return output


def get_collection_stats() -> dict:
    """Statistik isi ChromaDB."""
    collection = _get_collection()
    count      = collection.count()

    if count == 0:
        return {"total_chunks": 0, "tickers": [], "doc_types": []}

    # Sample untuk statistik (ChromaDB tidak support GROUP BY langsung)
    sample = collection.get(limit=min(count, 1000), include=["metadatas"])
    metas  = sample.get("metadatas", [])

    tickers   = list(set(m.get("ticker", "")   for m in metas))
    doc_types = list(set(m.get("doc_type", "") for m in metas))

    return {
        "total_chunks": count,
        "tickers":      sorted(tickers),
        "doc_types":    doc_types,
    }


# fix: import re yang lupa di atas
import re
