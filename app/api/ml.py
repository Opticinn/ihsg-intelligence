"""
app/api/ml.py
=============
Router FastAPI untuk semua endpoint Machine Learning.

Endpoints:
  POST /ml/train              → trigger training via Celery Worker
  GET  /ml/predict/{ticker}   → prediksi sinyal satu ticker (dengan Redis Cache)
  GET  /ml/predict/batch      → prediksi semua ticker
  GET  /ml/predictions/history → riwayat prediksi dari DB
  GET  /ml/model/info         → info model yang sedang aktif
  POST /ml/model/reload       → force reload model dari lokal
"""
from app.services.llm_service import generate_stock_analysis
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Optional

import redis
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session
from starlette.requests import Request

from app.core.database import SessionLocal
from app.core.limiter import limiter
from app.models.prediction import Prediction
from app.services.ingestion import IHSG_TICKERS
from app.services.predictor import (
    _cached_model_version,
    load_model,
    predict_all_tickers,
    predict_ticker,
    reload_model,
)

# Import Celery task
from app.worker import task_train_model

router = APIRouter(prefix="/ml", tags=["Machine Learning"])
logger = logging.getLogger(__name__)

# Inisialisasi Redis Client
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
try:
    redis_client = redis.from_url(REDIS_URL)
except Exception as e:
    logger.error(f"Gagal koneksi ke Redis saat startup: {e}")
    redis_client = None


# ─────────────────────────────────────────────
# DEPENDENCY
# ─────────────────────────────────────────────

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ─────────────────────────────────────────────
# SCHEMAS
# ─────────────────────────────────────────────

class PredictResponse(BaseModel):
    ticker:         str
    signal:         str           # BUY | HOLD | SELL
    confidence:     float
    confidence_pct: str
    model_version:  Optional[str]
    interpretation: str
    features:       dict
    prediction_id:  Optional[int]

class TrainResponse(BaseModel):
    status:     str
    message:    str

class PredictionHistoryItem(BaseModel):
    id:           int
    ticker:       str
    signal:       str
    confidence:   float
    model_version:Optional[str]
    created_at:   datetime

    class Config:
        from_attributes = True


# ─────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────

@router.post("/train", response_model=TrainResponse, summary="Trigger model training via Celery")
@limiter.limit("3/hour")
async def trigger_training(request: Request):
    """
    Memulai training model XGBoost di background menggunakan Celery Worker.
    API langsung merespons sukses, proses berat berjalan terpisah.
    """
    try:
        # Kirim tugas ke papan Celery
        task = task_train_model.delay()
        return TrainResponse(
            status="started",
            message=f"Training dilempar ke dapur Celery (Task ID: {task.id}).",
        )
    except Exception as e:
        logger.error(f"[ml_api] Gagal mengirim task ke Celery: {e}")
        raise HTTPException(status_code=500, detail="Gagal menghubungi Celery worker.")


@router.get(
    "/predict/{ticker}",
    response_model=PredictResponse,
    summary="Prediksi sinyal satu saham (Cached)",
)
@limiter.limit("60/minute")
async def predict_single(
    request: Request,
    ticker: str,
):
    """
    Prediksi sinyal trading. Menggunakan Redis cache agar response <100ms.
    """
    ticker = ticker.upper()
    if not ticker.endswith(".JK"):
        ticker = f"{ticker}.JK"

    cache_key = f"predict_cache_{ticker}"

    # 1. CEK KULKAS (REDIS) DULU
    if redis_client:
        try:
            cached_result = redis_client.get(cache_key)
            if cached_result:
                logger.info(f"⚡ [CACHE HIT] {ticker} diambil dari Redis.")
                return PredictResponse(**json.loads(cached_result))
        except Exception as e:
            logger.warning(f"⚠️ Redis read error: {e}")

    # 2. JIKA KOSONG, MINTA MODEL XG BOOST MENGHITUNG
    try:
        logger.info(f"🧠 [CACHE MISS] Menghitung prediksi {ticker}...")
        result = predict_ticker(ticker)
        
        # 3. SIMPAN HASIL KE KULKAS SELAMA 1 JAM
        if redis_client:
            try:
                redis_client.setex(cache_key, 3600, json.dumps(result))
            except Exception as e:
                logger.warning(f"⚠️ Redis write error: {e}")
                
        return PredictResponse(**result)
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"[ml_api] Predict error untuk {ticker}: {e}")
        raise HTTPException(status_code=500, detail=f"Error prediksi: {str(e)}")


@router.get("/predict/batch/all", summary="Prediksi semua ticker IHSG")
@limiter.limit("5/hour")
async def predict_batch(request: Request):
    """Prediksi batch tidak di-cache di endpoint ini untuk memastikan data segar."""
    results = predict_all_tickers(IHSG_TICKERS)

    success = [r for r in results if r.get("signal") != "ERROR"]
    errors  = [r for r in results if r.get("signal") == "ERROR"]

    return {
        "total":   len(results),
        "success": len(success),
        "errors":  len(errors),
        "results": results,
    }


@router.get(
    "/predictions/history",
    summary="Riwayat prediksi dari database",
)
async def get_prediction_history(
    db: Session = Depends(get_db),
    ticker: Optional[str] = Query(None, description="Filter by ticker, e.g. BBCA.JK"),
    signal: Optional[str] = Query(None, description="Filter by signal: BUY, HOLD, SELL"),
    days:   int           = Query(7, ge=1, le=90, description="Riwayat N hari terakhir"),
    limit:  int           = Query(50, ge=1, le=500),
):
    since = datetime.utcnow() - timedelta(days=days)
    query = db.query(Prediction).filter(Prediction.created_at >= since)

    if ticker:
        ticker = ticker.upper()
        if not ticker.endswith(".JK"):
            ticker = f"{ticker}.JK"
        query = query.filter(Prediction.ticker == ticker)

    if signal:
        query = query.filter(Prediction.signal == signal.upper())

    records = (query.order_by(Prediction.created_at.desc()).limit(limit).all())

    return {
        "count":   len(records),
        "filters": {"ticker": ticker, "signal": signal, "days": days},
        "data": [
            {
                "id":            r.id,
                "ticker":        r.ticker,
                "signal":        r.signal,
                "confidence":    r.confidence,
                "model_version": r.model_version,
                "created_at":    r.created_at.isoformat() if r.created_at else None,
            }
            for r in records
        ],
    }


@router.get("/model/info", summary="Info model yang aktif")
async def get_model_info():
    global _cached_model_version
    try:
        model, feature_cols, version = load_model()
        return {
            "status":        "loaded",
            "model_name":    "ihsg-xgb-classifier",
            "version":       version,
            "n_features":    len(feature_cols),
        }
    except RuntimeError as e:
        return {"status": "not_loaded", "message": str(e)}


@router.post("/model/reload", summary="Reload model dari file pickle")
@limiter.limit("10/hour")
async def force_reload_model(request: Request):
    try:
        model, feature_cols, version = reload_model()
        return {
            "status":  "reloaded",
            "version": version,
            "message": f"Model berhasil di-reload: {version}",
        }
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    
@router.get("/analyze/{ticker}", summary="AI Text Analysis (Llama 3.2)")
@limiter.limit("10/minute")
async def analyze_stock_with_ai(request: Request, ticker: str):
    """
    Meminta AI (Llama 3.2) untuk membaca hasil Machine Learning 
    dan memberikan analisis berbentuk bahasa manusia.
    """
    ticker = ticker.upper()
    if not ticker.endswith(".JK"):
        ticker = f"{ticker}.JK"

    # Panggil layanan LLM kita
    analisis_teks = generate_stock_analysis(ticker)

    return {
        "ticker": ticker,
        "ai_analysis": analisis_teks
    }