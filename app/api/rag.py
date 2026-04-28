"""
app/api/rag.py
==============
Router FastAPI untuk semua endpoint RAG & Chat.

Endpoints:
  POST /rag/chat/ask              → Q&A RAG (main endpoint)
  GET  /rag/chat/history/{sid}    → riwayat percakapan
  DELETE /rag/chat/{sid}          → hapus session
  POST /rag/ingest/{ticker}       → ingest dokumen satu ticker
  POST /rag/ingest/batch          → ingest semua ticker
  GET  /rag/stats                 → statistik ChromaDB
  GET  /rag/health                → cek koneksi Ollama
"""

import logging
from typing import Optional

import httpx
from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from starlette.requests import Request

from app.core.limiter import limiter
from app.services.ingestion import IHSG_TICKERS
from app.services.rag_chat import (
    chat,
    delete_conversation,
    get_conversation_history,
)
from app.services.vector_store import (
    get_collection_stats,
    ingest_all_tickers,
    ingest_ticker,
)

router = APIRouter(prefix="/rag", tags=["RAG & Chat"])
logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = "http://localhost:11434"


# ─────────────────────────────────────────────
# SCHEMAS
# ─────────────────────────────────────────────

class ChatRequest(BaseModel):
    question:   str             = Field(..., min_length=3, max_length=1000, description="Pertanyaan kamu")
    session_id: Optional[str]  = Field(None, description="ID sesi untuk multi-turn. Kosongkan untuk sesi baru.")
    ticker:     Optional[str]  = Field(None, description="Filter konteks ke satu ticker, e.g. 'BBCA'")

class ChatResponse(BaseModel):
    answer:     str
    session_id: str
    sources:    list
    ml_signal:  Optional[dict]

class IngestRequest(BaseModel):
    pdf_paths: Optional[list[str]] = Field(None, description="Path ke file PDF (opsional)")
    max_news:  int                 = Field(5, ge=1, le=20)

class BatchIngestRequest(BaseModel):
    tickers:           Optional[list[str]] = None    # None = semua IHSG_TICKERS
    max_news_per_ticker: int               = Field(3, ge=1, le=10)


# ─────────────────────────────────────────────
# BACKGROUND INGEST TRACKER
# ─────────────────────────────────────────────

_ingest_status = {"is_running": False, "last_result": None}


def _run_batch_ingest(tickers: list[str], max_news: int):
    global _ingest_status
    _ingest_status["is_running"] = True
    try:
        result = ingest_all_tickers(tickers, max_news_per_ticker=max_news)
        _ingest_status["last_result"] = result
    except Exception as e:
        _ingest_status["last_result"] = {"error": str(e)}
    finally:
        _ingest_status["is_running"] = False


# ─────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────

@router.post(
    "/chat/ask",
    response_model=ChatResponse,
    summary="Tanya AI Analyst tentang saham IHSG",
)
@limiter.limit("20/minute")
async def ask_chat(request: Request, body: ChatRequest):
    """
    **Endpoint utama RAG Chat.**

    Cara pakai:
    - **Pertama kali:** Kirim `question` saja, `session_id` akan dibuat otomatis.
    - **Multi-turn:** Gunakan `session_id` dari response sebelumnya untuk melanjutkan percakapan.
    - **Filter ticker:** Isi `ticker` untuk fokuskan konteks ke satu saham.

    Contoh pertanyaan:
    - "Bagaimana prospek BBCA berdasarkan berita terbaru?"
    - "Apa yang dimaksud RSI overbought?"
    - "Bandingkan BBCA dan BMRI dari sisi teknikal"

    ⚠️ Pastikan Ollama berjalan di background: `ollama serve`
    """
    try:
        result = chat(
            question   = body.question,
            session_id = body.session_id,
            ticker     = body.ticker,
        )
        return ChatResponse(**result)
    except ConnectionError:
        raise HTTPException(
            status_code=503,
            detail="Ollama tidak bisa dihubungi. Pastikan Ollama berjalan: 'ollama serve'",
        )
    except Exception as e:
        logger.error(f"[rag_api] chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/chat/history/{session_id}",
    summary="Riwayat percakapan satu sesi",
)
async def get_history(session_id: str):
    """Ambil seluruh riwayat percakapan berdasarkan session_id."""
    result = get_conversation_history(session_id)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@router.delete(
    "/chat/{session_id}",
    summary="Hapus sesi percakapan",
)
async def delete_session(session_id: str):
    """Hapus satu sesi beserta semua riwayat pesannya."""
    success = delete_conversation(session_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' tidak ditemukan.")
    return {"status": "deleted", "session_id": session_id}


@router.post(
    "/ingest/{ticker}",
    summary="Ingest dokumen untuk satu ticker",
)
@limiter.limit("10/hour")
async def ingest_one(request: Request, ticker: str, body: IngestRequest = IngestRequest()):
    """
    Scrape berita + parse PDF untuk satu ticker dan simpan ke ChromaDB.

    Proses ini membutuhkan ~30–60 detik tergantung jumlah berita.
    """
    ticker = ticker.upper()
    try:
        n_chunks = ingest_ticker(
            ticker     = ticker,
            pdf_paths  = body.pdf_paths,
            max_news   = body.max_news,
        )
        return {
            "status":  "success",
            "ticker":  ticker,
            "chunks_ingested": n_chunks,
            "message": f"{n_chunks} chunks berhasil disimpan ke ChromaDB untuk {ticker}.",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/ingest/batch/all",
    summary="Ingest semua ticker IHSG (background)",
)
@limiter.limit("2/hour")
async def ingest_batch(
    request:          Request,
    body:             BatchIngestRequest = BatchIngestRequest(),
    background_tasks: BackgroundTasks = None,
):
    """
    Ingest dokumen untuk semua ticker IHSG sekaligus, dijalankan di background.

    Proses ini membutuhkan ~10–30 menit. Response langsung dikembalikan,
    cek progress di `GET /rag/ingest/status`.
    """
    if _ingest_status["is_running"]:
        raise HTTPException(status_code=409, detail="Ingest sedang berjalan. Tunggu selesai.")

    tickers = body.tickers or IHSG_TICKERS
    background_tasks.add_task(_run_batch_ingest, tickers, body.max_news_per_ticker)

    return {
        "status":  "started",
        "tickers": tickers,
        "message": f"Ingest {len(tickers)} ticker dimulai di background.",
    }


@router.get("/ingest/status", summary="Status ingest terakhir")
async def get_ingest_status():
    return {
        "is_running":  _ingest_status["is_running"],
        "last_result": _ingest_status["last_result"],
    }


@router.get("/stats", summary="Statistik ChromaDB")
async def get_stats():
    """Tampilkan jumlah total chunks, ticker, dan tipe dokumen di ChromaDB."""
    stats = get_collection_stats()
    return {
        "chroma_stats": stats,
        "message": (
            "ChromaDB kosong. Jalankan POST /rag/ingest/{ticker} terlebih dahulu."
            if stats["total_chunks"] == 0
            else f"{stats['total_chunks']} chunks tersimpan untuk {len(stats['tickers'])} ticker."
        ),
    }


@router.get("/health", summary="Cek koneksi Ollama")
async def health_check():
    """Verifikasi apakah Ollama berjalan dan model tersedia."""
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            models = resp.json().get("models", [])
            model_names = [m.get("name", "") for m in models]

            llama_ready = any("llama3.2" in n for n in model_names)
            return {
                "ollama_status": "online",
                "available_models": model_names,
                "llama3_2_ready":   llama_ready,
                "message": (
                    "✅ Ollama + Llama 3.2 siap."
                    if llama_ready
                    else "⚠️ Ollama online tapi Llama 3.2 belum di-pull. Jalankan: ollama pull llama3.2"
                ),
            }
    except Exception as e:
        return {
            "ollama_status": "offline",
            "error":         str(e),
            "message":       "❌ Ollama tidak bisa dihubungi. Jalankan: ollama serve",
        }
