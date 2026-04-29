import logging
from typing import Optional
import httpx
from fastapi import APIRouter, BackgroundTasks, HTTPException, Depends # Tambah Depends
from pydantic import BaseModel, Field
from starlette.requests import Request
from sqlalchemy.orm import Session # Tambah Session
from sqlalchemy import desc



# Core & Database
from app.core.limiter import limiter
from app.core.database import SessionLocal # Import SessionLocal kita
from app.services.ingestion import IHSG_TICKERS

# Services
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

# Models & Langchain
from app.models.prediction import Prediction 


router = APIRouter(prefix="/rag", tags=["RAG & Chat"])
logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = "http://localhost:11434"

# ─── DEPENDENCY DB ───
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ─── SCHEMAS ───
class ChatRequest(BaseModel):
    question:   str             = Field(..., min_length=3, max_length=1000)
    session_id: Optional[str]   = Field(None)
    ticker:     Optional[str]   = Field(None)

class ChatResponse(BaseModel):
    answer:     str
    session_id: str
    sources:     list
    ml_signal:  Optional[dict]

class IngestRequest(BaseModel):
    pdf_paths: Optional[list[str]] = None
    max_news:  int                 = Field(5, ge=1, le=20)

class BatchIngestRequest(BaseModel):
    tickers:             Optional[list[str]] = None
    max_news_per_ticker: int                 = Field(3, ge=1, le=10)

# ─── BACKGROUND INGEST TRACKER ───
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


# ─── ENDPOINTS ───

@router.post("/chat/ask", response_model=ChatResponse)
@limiter.limit("20/minute")
async def ask_chat(request: Request, body: ChatRequest):
    try:
        result = chat(
            question   = body.question,
            session_id = body.session_id,
            ticker     = body.ticker,
        )
        return ChatResponse(**result)
    except Exception as e:
        logger.error(f"[rag_api] chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ingest/{ticker}", summary="Ingest dokumen + Hasil ML")
@limiter.limit("10/hour")
async def ingest_one(
    request: Request, 
    ticker: str, 
    body: IngestRequest = IngestRequest(),
    db: Session = Depends(get_db) # KUNCI: Menambahkan akses database
):
    ticker = ticker.upper()
    if not ticker.endswith(".JK"):
        ticker = f"{ticker}.JK"

    try:
        # 1. AMBIL PREDIKSI TERBARU DARI POSTGRESQL
        last_pred = db.query(Prediction).filter(
            Prediction.ticker == ticker
        ).order_by(desc(Prediction.created_at)).first()

        ml_text = ""
        if last_pred:
            ml_text = (
                f"HASIL ANALISIS MACHINE LEARNING TERBARU:\n"
                f"Ticker: {ticker}\n"
                f"Sinyal Trading: {last_pred.signal}\n"
                f"Confidence Score: {last_pred.confidence * 100:.2f}%\n"
                f"Waktu Prediksi: {last_pred.created_at}\n"
            )
            logger.info(f"✅ Menemukan data ML untuk {ticker}, siap di-ingest.")

        # 2. JALANKAN INGEST (Kirim ml_text ke service)
        # Pastikan fungsi ingest_ticker di app/services/vector_store.py 
        # bisa menerima parameter tambahan 'extra_text' jika ada.
        n_chunks = ingest_ticker(
            ticker    = ticker,
            pdf_paths = body.pdf_paths,
            max_news  = body.max_news,
            extra_text = ml_text # Kita titipkan hasil ML di sini
        )

        return {
            "status":  "success",
            "ticker":  ticker,
            "chunks_ingested": n_chunks,
            "has_ml_data": last_pred is not None,
            "message": f"Data {ticker} (termasuk hasil ML) berhasil disimpan ke ChromaDB."
        }
    except Exception as e:
        logger.error(f"Error saat ingest {ticker}: {e}")
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
