"""
app/services/rag_chat.py
========================
RAG Chat Engine — otak dari sistem tanya-jawab saham.

Alur per request:
  1. Load/buat conversation session di PostgreSQL
  2. Ambil history percakapan (max 6 pesan terakhir — context window)
  3. Retrieve chunks relevan dari ChromaDB
  4. Juga ambil data prediksi ML terbaru dari DB (enrichment)
  5. Bangun prompt lengkap: history + konteks dokumen + pertanyaan
  6. Panggil Llama 3.2 via Ollama
  7. Simpan pasangan (user_msg, assistant_msg) ke PostgreSQL
  8. Return jawaban + sumber dokumen
"""

import json
import logging
import uuid
from datetime import datetime
from typing import Optional

from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from sqlalchemy.orm import Session

from app.core.database import SessionLocal
from app.models.conversation import Conversation, Message
from app.models.prediction import Prediction
from app.services.vector_store import retrieve_context

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

OLLAMA_BASE_URL  = "http://localhost:11434"
OLLAMA_MODEL     = "llama3.2"
TEMPERATURE      = 0.3
MAX_HISTORY_MSGS = 6    # jumlah pesan history yang dimasukkan ke prompt
MAX_CONTEXT_DOCS = 4    # jumlah chunk ChromaDB per query


# ─────────────────────────────────────────────
# LLM SINGLETON
# ─────────────────────────────────────────────

_llm: Optional[OllamaLLM] = None


def _get_llm() -> OllamaLLM:
    global _llm
    if _llm is None:
        logger.info(f"[rag_chat] Connecting to Ollama ({OLLAMA_MODEL})...")
        _llm = OllamaLLM(
            base_url    = OLLAMA_BASE_URL,
            model       = OLLAMA_MODEL,
            temperature = TEMPERATURE,
        )
        logger.info("[rag_chat] Ollama siap.")
    return _llm


# ─────────────────────────────────────────────
# PROMPT TEMPLATE
# ─────────────────────────────────────────────

RAG_PROMPT = PromptTemplate(
    input_variables=["history", "context_docs", "ml_context", "question"],
    template="""Kamu adalah IHSG AI Analyst — asisten analisis saham Bursa Efek Indonesia yang cerdas, akurat, dan berbicara dalam Bahasa Indonesia.

PANDUAN PENTING:
- Jawab HANYA berdasarkan informasi yang tersedia di bawah ini (Konteks Dokumen & Data ML).
- Jika informasi tidak cukup, katakan "Saya tidak memiliki data yang cukup untuk menjawab ini."
- Jangan membuat angka atau fakta yang tidak ada di konteks.
- Selalu akhiri jawaban dengan disclaimer singkat tentang risiko investasi.

TEORI INDIKATOR (gunakan sebagai referensi interpretasi):
- RSI > 70 = Overbought (harga mungkin sudah terlalu tinggi, potensi koreksi)
- RSI < 30 = Oversold (harga mungkin sudah terlalu rendah, potensi rebound)
- RSI 30-70 = Netral
- MACD Histogram positif = Momentum bullish (harga cenderung naik)
- MACD Histogram negatif = Momentum bearish (harga cenderung turun)
- Bollinger Band %B > 1 = Harga di atas upper band (overbought area)
- Bollinger Band %B < 0 = Harga di bawah lower band (oversold area)
- Volume Ratio > 2x = Ada aktivitas beli/jual yang tidak biasa

RIWAYAT PERCAKAPAN:
{history}

KONTEKS DOKUMEN (berita & laporan keuangan terbaru):
{context_docs}

DATA ML & TEKNIKAL TERKINI:
{ml_context}

PERTANYAAN USER:
{question}

JAWABAN (dalam Bahasa Indonesia, terstruktur dan mudah dipahami):"""
)


# ─────────────────────────────────────────────
# DB HELPERS
# ─────────────────────────────────────────────

def _get_or_create_session(session_id: Optional[str], ticker: Optional[str]) -> tuple[Conversation, Session]:
    """
    Get conversation session yang sudah ada, atau buat baru.
    Returns (conversation_obj, db_session)
    """
    db = SessionLocal()

    if session_id:
        conv = db.query(Conversation).filter(Conversation.session_id == session_id).first()
        if conv:
            return conv, db

    # Buat session baru
    new_session_id = session_id or str(uuid.uuid4())[:16]
    conv = Conversation(session_id=new_session_id, ticker=ticker)
    db.add(conv)
    db.commit()
    db.refresh(conv)
    return conv, db


def _get_history_text(conv: Conversation, db: Session, max_msgs: int = MAX_HISTORY_MSGS) -> str:
    """Ambil riwayat pesan dan format jadi string untuk prompt."""
    messages = (
        db.query(Message)
        .filter(Message.conversation_id == conv.id)
        .order_by(Message.created_at.desc())
        .limit(max_msgs)
        .all()
    )
    messages = list(reversed(messages))  # urutan kronologis

    if not messages:
        return "Belum ada riwayat percakapan."

    history_lines = []
    for msg in messages:
        role = "User" if msg.role == "user" else "Assistant"
        history_lines.append(f"{role}: {msg.content}")

    return "\n".join(history_lines)


def _get_ml_context(ticker: Optional[str], db: Session) -> str:
    """Ambil prediksi ML terbaru untuk ticker dari DB."""
    if not ticker:
        return "Tidak ada data ML spesifik untuk query ini."

    ticker_clean = ticker.replace(".JK", "").upper() + ".JK"
    pred = (
        db.query(Prediction)
        .filter(Prediction.ticker == ticker_clean)
        .order_by(Prediction.created_at.desc())
        .first()
    )

    if not pred:
        return f"Belum ada data prediksi ML untuk {ticker_clean}. Jalankan /ml/predict/{ticker} terlebih dahulu."

    try:
        features = json.loads(pred.features_snapshot) if pred.features_snapshot else {}
    except Exception:
        features = {}

    lines = [
        f"Ticker: {pred.ticker}",
        f"Signal ML: {pred.signal} (Confidence: {pred.confidence:.1%})",
        f"Harga terakhir: Rp {features.get('close_price', 'N/A'):,}",
        f"RSI: {features.get('rsi', 'N/A')}",
        f"MACD Histogram: {features.get('macd_hist', 'N/A')}",
        f"Bollinger %B: {features.get('bb_pct_b', 'N/A')}",
        f"Volume Ratio: {features.get('volume_ratio', 'N/A')}x",
        f"Return 1 hari: {float(features.get('return_1d', 0))*100:.2f}%",
        f"Prediksi dibuat: {pred.created_at.strftime('%d %b %Y %H:%M') if pred.created_at else 'N/A'}",
    ]
    return "\n".join(lines)


def _save_messages(conv: Conversation, db: Session, user_msg: str, assistant_msg: str, sources: list) -> None:
    """Simpan pasangan pesan user + assistant ke DB."""
    db.add(Message(
        conversation_id = conv.id,
        role            = "user",
        content         = user_msg,
    ))
    db.add(Message(
        conversation_id = conv.id,
        role            = "assistant",
        content         = assistant_msg,
        sources         = json.dumps(sources),
    ))
    # Update timestamp session
    conv.updated_at = datetime.utcnow()
    db.commit()


# ─────────────────────────────────────────────
# MAIN CHAT FUNCTION
# ─────────────────────────────────────────────

def chat(
    question:   str,
    session_id: Optional[str] = None,
    ticker:     Optional[str] = None,
) -> dict:
    """
    Proses satu pertanyaan user dengan full RAG pipeline.

    Args:
        question:   pertanyaan user, e.g. "Bagaimana prospek BBCA bulan ini?"
        session_id: ID sesi untuk multi-turn (None = buat sesi baru)
        ticker:     filter konteks ke satu ticker (opsional)

    Returns:
        dict: answer, session_id, sources, ml_signal
    """
    logger.info(f"[rag_chat] Question: '{question[:60]}...' | session={session_id} | ticker={ticker}")

    # ── 1. Get/create session ──
    conv, db = _get_or_create_session(session_id, ticker)

    try:
        # ── 2. History percakapan ──
        history_text = _get_history_text(conv, db)

        # ── 3. Retrieve dari ChromaDB ──
        ticker_filter = ticker.replace(".JK", "").upper() if ticker else None
        chunks = retrieve_context(
            query    = question,
            ticker   = ticker_filter,
            top_k    = MAX_CONTEXT_DOCS,
        )

        if chunks:
            context_text = "\n\n---\n".join([
                f"[Sumber: {c['title'] or c['source']} | {c['date']}]\n{c['content']}"
                for c in chunks
            ])
        else:
            context_text = (
                "Tidak ada dokumen yang relevan ditemukan di database. "
                "Jawaban akan berdasarkan data ML dan pengetahuan umum."
            )

        # ── 4. ML context ──
        ml_context = _get_ml_context(ticker, db)

        # ── 5. Build prompt ──
        prompt = RAG_PROMPT.format(
            history      = history_text,
            context_docs = context_text,
            ml_context   = ml_context,
            question     = question,
        )

        # ── 6. Panggil Ollama ──
        llm    = _get_llm()
        answer = llm.invoke(prompt)
        answer = answer.strip()

        # ── 7. Siapkan sumber dokumen untuk response ──
        sources = [
            {
                "title":    c.get("title", ""),
                "source":   c.get("source", ""),
                "ticker":   c.get("ticker", ""),
                "doc_type": c.get("doc_type", ""),
                "score":    c.get("score", 0),
            }
            for c in chunks
        ]

        # ── 8. Simpan ke DB ──
        _save_messages(conv, db, question, answer, sources)

        logger.info(f"[rag_chat] Answer generated ({len(answer)} chars) | {len(sources)} sources")

        return {
            "answer":     answer,
            "session_id": conv.session_id,
            "sources":    sources,
            "ml_signal":  _extract_ml_signal(ticker, db),
        }

    except Exception as e:
        logger.error(f"[rag_chat] Error: {e}", exc_info=True)
        raise
    finally:
        db.close()


def _extract_ml_signal(ticker: Optional[str], db: Session) -> Optional[dict]:
    """Ambil signal ML ringkas untuk ditampilkan di response."""
    if not ticker:
        return None
    ticker_clean = ticker.replace(".JK", "").upper() + ".JK"
    pred = (
        db.query(Prediction)
        .filter(Prediction.ticker == ticker_clean)
        .order_by(Prediction.created_at.desc())
        .first()
    )
    if not pred:
        return None
    return {
        "ticker":     pred.ticker,
        "signal":     pred.signal,
        "confidence": pred.confidence,
    }


# ─────────────────────────────────────────────
# CONVERSATION HISTORY HELPERS
# ─────────────────────────────────────────────

def get_conversation_history(session_id: str) -> dict:
    """Ambil seluruh riwayat percakapan satu session."""
    db = SessionLocal()
    try:
        conv = db.query(Conversation).filter(Conversation.session_id == session_id).first()
        if not conv:
            return {"error": f"Session '{session_id}' tidak ditemukan."}

        messages = (
            db.query(Message)
            .filter(Message.conversation_id == conv.id)
            .order_by(Message.created_at.asc())
            .all()
        )

        return {
            "session_id": conv.session_id,
            "ticker":     conv.ticker,
            "created_at": conv.created_at.isoformat() if conv.created_at else None,
            "messages": [
                {
                    "role":       m.role,
                    "content":    m.content,
                    "created_at": m.created_at.isoformat() if m.created_at else None,
                    "sources":    json.loads(m.sources) if m.sources else [],
                }
                for m in messages
            ],
        }
    finally:
        db.close()


def delete_conversation(session_id: str) -> bool:
    """Hapus satu sesi percakapan beserta semua pesannya."""
    db = SessionLocal()
    try:
        conv = db.query(Conversation).filter(Conversation.session_id == session_id).first()
        if not conv:
            return False
        db.delete(conv)
        db.commit()
        return True
    finally:
        db.close()
