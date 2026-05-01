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
from app.services.ingestion import IHSG_TICKERS
import json
import logging
import uuid
from datetime import datetime
from typing import Optional

from app.models.prediction import Prediction
from sqlalchemy import desc

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
MAX_HISTORY_MSGS = 0    # jumlah pesan history yang dimasukkan ke prompt
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

VALID_TICKERS = [t.replace(".JK", "") for t in IHSG_TICKERS]
TICKER_LIST   = ", ".join(VALID_TICKERS)


# ─────────────────────────────────────────────
# PROMPT TEMPLATE
# ─────────────────────────────────────────────

def _build_prompt() -> PromptTemplate:
    today = datetime.now().strftime("%d %B %Y")
    return PromptTemplate(
        input_variables=["history", "context_docs", "ml_context", "question"],
        template=f"""Kamu adalah Andi, asisten analisis saham Indonesia yang jujur dan tidak mengarang data.
Hari ini: {today}.

SAHAM YANG KAMU DUKUNG (HANYA INI):
{TICKER_LIST}

ATURAN WAJIB — JANGAN DILANGGAR:
1. JANGAN pernah mengarang sinyal, RSI, MACD, atau harga jika DATA SAHAM kosong.
2. Semua analisis merujuk kondisi HARI INI, {today}.
3. Sinyal SELL = prediksi TURUN (jangan sebut "siap naik"). Sinyal BUY = prediksi NAIK.
4. RSI < 30 = oversold (harga murah, potensi naik). RSI 30-70 = netral. RSI > 70 = overbought.
5. Tulis SATU kalimat pengingat risiko di akhir — tidak diulang.
7. Jawab pertanyaan user SECARA LANGSUNG. Jika user tanya, 
   berikan pendapat berdasarkan sinyal AI dan data — LANGSUNG BERIKAN KESIMPULAN JAWABANNYA!.
8. Jangan ulangi kalimat yang sama. Maksimal 3 paragraf pendek.
9. Confidence model berkisar 33-45% — ini NORMAL untuk prediksi saham 3-class.
   Confidence > 35% sudah BAGUS karena baseline random adalah 33%.
   JANGAN sebut confidence rendah atau tidak yakin kalau nilainya di atas 35%.
   Gunakan framing: "AI cukup yakin" (35-40%), "AI cukup percaya diri" (40-45%), "AI sangat yakin" (>45%).
10. JANGAN pernah tulis kalimat "Saya tidak dapat memberikan saran investasi" atau 
   "Saya tidak bisa memberikan rekomendasi spesifik".
   Gantinya, langsung jawab berdasarkan data dan SELALU akhiri dengan SATU kalimat:
   "Ingat, ini bukan saran investasi resmi — selalu lakukan riset sendiri."

RIWAYAT CHAT SEBELUMNYA:
{{history}}

BERITA & ANALISIS TERKAIT (hari ini, {today}):
{{context_docs}}

DATA SAHAM HARI INI ({today}):
{{ml_context}}

PERTANYAAN USER:
{{question}}

Jawab 2-3 paragraf pendek, jelaskan dengan bahasa Indonesia yang mudah dipahami oleh orang awam dan berikan jawaban akhirnya. Hanya berdasarkan data di atas."""
    )

 
 
# ══════════════════════════════════════════════════
# BAGIAN 2: Ganti _get_ml_context
# ══════════════════════════════════════════════════
 
def _get_ml_context(ticker, db) -> str:
    """Versi baru — output human-friendly, signal jelas terbaca."""
    import json
    from app.models.prediction import Prediction
 
    if not ticker:
        return "Tidak ada data saham spesifik."
 
    ticker_clean = ticker.replace(".JK", "").upper() + ".JK"
    pred = (
        db.query(Prediction)
        .filter(Prediction.ticker == ticker_clean)
        .order_by(Prediction.created_at.desc())
        .first()
    )
 
    if not pred:
        return f"Belum ada data prediksi untuk {ticker_clean}. Coba akses /ml/predict/{ticker} dulu."
 
    try:
        features = json.loads(pred.features_snapshot) if pred.features_snapshot else {}
    except Exception:
        features = {}
 
    rsi       = features.get("rsi", 50)
    macd_hist = features.get("macd_hist", 0)
    vol_ratio = features.get("volume_ratio", 1)
    close     = features.get("close_price", 0)
    ret_1d    = features.get("return_1d", 0)
 
    # Terjemahkan RSI
    if rsi < 30:
        rsi_desc = f"RSI {rsi:.1f} — harga sudah cukup murah, berpotensi rebound"
    elif rsi > 70:
        rsi_desc = f"RSI {rsi:.1f} — harga sudah cukup mahal, ada risiko koreksi"
    else:
        rsi_desc = f"RSI {rsi:.1f} — kondisi normal"
 
    # Terjemahkan MACD
    macd_desc = (
        f"Momentum sedang menguat" if macd_hist > 0
        else f"Momentum sedang melemah"
    )
 
    # Terjemahkan volume
    if vol_ratio > 2:
        vol_desc = f"Ada lonjakan transaksi ({vol_ratio:.1f}x dari biasanya)"
    elif vol_ratio < 0.5:
        vol_desc = f"Transaksi sedang sepi ({vol_ratio:.1f}x dari biasanya)"
    else:
        vol_desc = f"Volume transaksi normal ({vol_ratio:.1f}x rata-rata)"
 
    # Signal dengan wording yang jelas
    signal_wording = {
        "BUY":  f"BELI — AI memprediksi harga {ticker_clean} berpotensi NAIK dalam 1-3 hari ke depan",
        "SELL": f"JUAL — AI memprediksi harga {ticker_clean} berpotensi TURUN dalam 1-3 hari ke depan",
        "HOLD": f"TAHAN — AI memprediksi harga {ticker_clean} relatif flat dalam 1-3 hari ke depan",
    }
 
    return f"""Ticker: {ticker_clean}
Harga saat ini: Rp {close:,.0f}
Perubahan kemarin: {ret_1d*100:+.2f}%
 
SINYAL AI: {signal_wording.get(pred.signal, pred.signal)}
Tingkat keyakinan model: {pred.confidence:.0%}
 
Kondisi teknikal (untuk referensi):
- {rsi_desc}
- {macd_desc}
- {vol_desc}"""




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


def _ml_context(ticker: Optional[str], db: Session) -> str:
    "bil prediksi ML terbaru untuk ticker dari DB."""
    if not ticker:
        return "Tidak ada data ML spesifik untuk query ini."

    # 1. Standarisa.join(lines)


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
    logger.info(f"[rag_chat] Question: '{question[:60]}...' | session={session_id} | ticker={ticker}")

    # ── 0. AUTO-DETECT TICKER ──
    if not ticker:
        from app.services.ingestion import IHSG_TICKERS
        for t in IHSG_TICKERS:
            if t.replace(".JK", "").lower() in question.lower():
                ticker = t
                logger.info(f"🔍 [Auto-Detect] Menemukan ticker {ticker}")
                break

    # ── 1. VALIDASI TICKER ──
    if ticker:
        ticker_clean_check = ticker.replace(".JK", "").upper()
        if ticker_clean_check not in VALID_TICKERS:
            return {
                "answer": f"Maaf, saham **{ticker_clean_check}** tidak ada dalam daftar kami.\n\nSaham yang tersedia: {TICKER_LIST}",
                "session_id": session_id or "invalid",
                "sources": [],
                "ml_signal": None,
            }

    # ── 2. GET/CREATE SESSION ──
    conv, db = _get_or_create_session(session_id, ticker)
    # ... sisa kode lama ...
    
    if ticker and conv.ticker != ticker:
        conv.ticker = ticker
        db.commit()

    try:
        # ── 2. History percakapan ──
        history_text = _get_history_text(conv, db)
        
        # ── AUTO-TRIGGER: predict + ingest kalau belum ada data ──
        if ticker:
            ticker_jk = ticker.replace(".JK", "").upper() + ".JK"
            
            # Cek apakah sudah ada prediksi ML
            from datetime import datetime, timedelta
            today = datetime.utcnow() - timedelta(hours=12)
            pred_exists = db.query(Prediction)\
                .filter(
                    Prediction.ticker == ticker_jk,
                    Prediction.created_at >= today
                ).first()
            
            if not pred_exists:
                logger.info(f"[rag_chat] Auto-predict {ticker_jk}...")
                try:
                    from app.services.predictor import predict_ticker
                    predict_ticker(ticker_jk)
                except Exception as e:
                    logger.warning(f"[rag_chat] Auto-predict gagal: {e}")

            # Cek apakah sudah ada chunks di ChromaDB untuk ticker ini
            from app.services.vector_store import get_collection_stats, ingest_ticker
            stats = get_collection_stats()
            ticker_clean = ticker.replace(".JK", "").upper()
            
            from app.services.vector_store import _get_collection
            col      = _get_collection()
            existing = col.get(limit=200, include=["metadatas"])
            has_external = any(
                m.get("ticker") == ticker_clean and 
                m.get("source", "").startswith("http")
                for m in existing.get("metadatas", [])
            )

            if not has_external:
                logger.info(f"[rag_chat] Auto-ingest {ticker_clean}...")
                try:
                    ingest_ticker(ticker_clean, max_news=3)
                    import time; time.sleep(1)  # tunggu ChromaDB selesai index
                    logger.info(f"[rag_chat] Auto-ingest {ticker_clean} selesai.")
                except Exception as e:
                    logger.warning(f"[rag_chat] Auto-ingest gagal: {e}")
        # ── END AUTO-TRIGGER ──

        # ── 3. Retrieve dari ChromaDB ──
        # Gunakan ticker yang sudah terdeteksi (BBCA.JK -> BBCA)
        ticker_filter = ticker.replace(".JK", "").upper() if ticker else None
        
        chunks = retrieve_context(
            query    = question,
            ticker   = ticker_filter,
            top_k    = 6,
        )
        
        external = [c for c in chunks if c.get("source", "").startswith("http")]
        internal = [c for c in chunks if not c.get("source", "").startswith("http")]
        chunks   = external[:3] + internal[:2]  # max 3 external + 2 internal

        if chunks:
            context_text = "\n\n---\n".join([
                f"[Sumber: {c.get('title') or c.get('source')} | {c.get('date')}]\n{c['content']}"
                for c in chunks
            ])
        else:
            context_text = "Tidak ada dokumen berita spesifik. Fokus pada data ML."

        # ── 4. ML context (Sekarang sudah punya ticker!) ──
        ml_context = _get_ml_context(ticker, db)
        
        # Log untuk debugging (Cek di terminal nanti!)
        logger.info(f"📊 [ML Context Content]: {ml_context[:100]}...")

        # ── 5. Build prompt ──
        prompt = _build_prompt().format(
            history      = history_text,
            context_docs = context_text,
            ml_context   = ml_context,
            question     = question,
        )

        # ── 6. Panggil Ollama ──
        llm    = _get_llm()
        answer = llm.invoke(prompt)
        answer = answer.strip()

        # ── 7. Siapkan sumber ──
        sources = [
            {
                "title":    c.get("title", ""),
                "source":   c.get("source", ""),
                "ticker":   c.get("ticker", ""),
                "doc_type": c.get("doc_type", ""),
            }
            for c in chunks
        ]

        # ── 8. Simpan & Return ──
        _save_messages(conv, db, question, answer, sources)
        return {
            "answer":     answer,
            "session_id": conv.session_id,
            "sources":     sources,
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
