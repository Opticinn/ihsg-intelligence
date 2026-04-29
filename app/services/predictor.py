"""
app/services/predictor.py  (FIXED — Phase 2)
=============================================
Load model dari file pickle yang disimpan trainer_fixed.py.
Tidak pakai mlflow.xgboost.load_model (ada bug di XGBoost 2.x).

Signal mapping:
  0 → SELL  |  1 → HOLD  |  2 → BUY
"""

_cached_model_version = "v1-pickle"

import json
import logging
import os
import pickle
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from app.core.database import SessionLocal
from app.models.prediction import Prediction
from app.models.stock import StockHistory
from app.services.features import FEATURE_COLS, build_features

logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "mlruns")
LOCAL_MODEL_PATH    = Path("mlruns") / "saved_models" / "ihsg_xgb_latest.pkl"

SIGNAL_MAP = {0: "SELL", 1: "HOLD", 2: "BUY"}

# Cache in-memory
_cache: dict = {}   # keys: "model", "feature_cols", "run_id", "accuracy"


# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────

def load_model() -> tuple:
    """
    Load model dari file pickle lokal.
    Di-cache di memory setelah pertama kali load.

    Returns:
        (model, feature_cols, version_str)
    """
    global _cache

    if _cache.get("model") is not None:
        return _cache["model"], _cache["feature_cols"], _cache.get("run_id", "cached")[:8]

    if not LOCAL_MODEL_PATH.exists():
        raise RuntimeError(
            f"File model tidak ditemukan: {LOCAL_MODEL_PATH}\n"
            "Jalankan training terlebih dahulu:\n"
            "  python -m app.services.trainer"
        )

    logger.info(f"[predictor] Loading model dari {LOCAL_MODEL_PATH}...")
    with open(LOCAL_MODEL_PATH, "rb") as f:
        payload = pickle.load(f)

    _cache["model"]        = payload["model"]
    _cache["feature_cols"] = payload["feature_cols"]
    _cache["run_id"]       = payload.get("run_id", "unknown")
    _cache["accuracy"]     = payload.get("test_accuracy", payload.get("accuracy", 0))

    version = _cache["run_id"][:8]
    logger.info(
        f"[predictor] Model loaded — run_id: {version} | "
        f"accuracy: {_cache['accuracy']:.4f} | "
        f"features: {len(_cache['feature_cols'])}"
    )
    return _cache["model"], _cache["feature_cols"], version


def reload_model() -> tuple:
    """Force reload dari disk."""
    global _cache
    _cache = {}
    return load_model()


# ─────────────────────────────────────────────
# PREDICT
# ─────────────────────────────────────────────

def predict_ticker(ticker: str) -> dict:
    """
    Prediksi sinyal untuk satu ticker.

    Returns dict: signal, confidence, features snapshot, interpretasi.
    """
    # ── 1. Build fitur dari DB ──
    df = build_features(ticker)
    if df is None or df.empty:
        raise ValueError(
            f"Tidak cukup data historis untuk {ticker}. "
            "Pastikan ingestion pipeline sudah berjalan."
        )

    # ── 2. Siapkan input baris terakhir ──
    df["ticker_encoded"] = hash(ticker) % 100

    # Ghost row sudah difilter di features.py & ingestion.py
    # close_price tidak ada di df features (sudah dinormalisasi)
    # Ambil langsung baris terakhir
    last_row = df.iloc[-1]
    

    model, feature_cols, version = load_model()

    # Pastikan semua kolom ada (isi NaN jika ada kolom yang kurang)
    X = pd.DataFrame([last_row.reindex(feature_cols).astype(float)])

    # ── 3. Predict ──
    pred_class  = int(model.predict(X)[0])
    pred_proba  = model.predict_proba(X)[0]   # [prob_SELL, prob_HOLD, prob_BUY]

    signal      = SIGNAL_MAP[pred_class]
    confidence  = float(round(pred_proba[pred_class], 4))

    # Ambil close_price langsung dari DB (tidak ada di features df)
    try:
        db_temp = SessionLocal()
        last_stock = (
            db_temp.query(StockHistory)
            .filter(StockHistory.ticker == ticker, StockHistory.close_price > 0)
            .order_by(StockHistory.date.desc())
            .first()
        )
        close_price = float(last_stock.close_price) if last_stock else 0.0
        db_temp.close()
    except Exception:
        print(f"DEBUG close_price error: {e}")
        close_price = 0.0

    features_snapshot = {
        "close_price":  round(close_price, 2),
        "rsi":          round(float(last_row["rsi"]), 2),
        "macd_line":    round(float(last_row["macd_line"]), 4),
        "macd_hist":    round(float(last_row["macd_hist"]), 4),
        "bb_pct_b":     round(float(last_row["bb_pct_b"]), 4),
        "volume_ratio": round(float(last_row["volume_ratio"]), 2),
        "return_1d":    round(float(last_row["return_1d"]), 4),
        "prob_sell":    round(float(pred_proba[0]), 4),
        "prob_hold":    round(float(pred_proba[1]), 4),
        "prob_buy":     round(float(pred_proba[2]), 4),
        "date_data":    str(last_row["date"]),
    }

    # ── 4. Simpan ke DB ──
    db = SessionLocal()
    try:
        rec = Prediction(
            ticker            = ticker,
            signal            = signal,
            confidence        = confidence,
            model_version     = version,
            features_snapshot = json.dumps(features_snapshot),
        )
        db.add(rec)
        db.commit()
        db.refresh(rec)
        pred_id = rec.id
    except Exception as e:
        db.rollback()
        logger.error(f"[predictor] Gagal simpan ke DB: {e}")
        pred_id = None
    finally:
        db.close()

    logger.info(f"[predictor] {ticker} → {signal} (conf={confidence:.2%}, model={version})")

    return {
        "ticker":         ticker,
        "signal":         signal,
        "confidence":     confidence,
        "confidence_pct": f"{confidence:.1%}",
        "model_version":  version,
        "features":       features_snapshot,
        "interpretation": _interpret(signal, features_snapshot),
        "prediction_id":  pred_id,
    }


def _interpret(signal: str, f: dict) -> str:
    """Buat kalimat interpretasi dari nilai indikator."""
    notes = []

    rsi = f.get("rsi", 50)
    if rsi < 30:
        notes.append(f"RSI {rsi:.1f} — oversold")
    elif rsi > 70:
        notes.append(f"RSI {rsi:.1f} — overbought")
    else:
        notes.append(f"RSI {rsi:.1f} — netral")

    if f.get("macd_hist", 0) > 0:
        notes.append("MACD positif (momentum naik)")
    else:
        notes.append("MACD negatif (momentum turun)")

    vr = f.get("volume_ratio", 1)
    if vr > 2:
        notes.append(f"Volume spike {vr:.1f}x")
    elif vr < 0.5:
        notes.append(f"Volume sepi {vr:.1f}x")

    bb = f.get("bb_pct_b", 0.5)
    if bb > 0.9:
        notes.append("Mendekati upper Bollinger Band")
    elif bb < 0.1:
        notes.append("Mendekati lower Bollinger Band")

    prefix = {
        "BUY":  "Model prediksi potensi NAIK 3 hari ke depan.",
        "SELL": "Model prediksi potensi TURUN 3 hari ke depan.",
        "HOLD": "Model prediksi pergerakan FLAT 3 hari ke depan.",
    }
    return f"{prefix[signal]} {' | '.join(notes)}."


# ─────────────────────────────────────────────
# BATCH
# ─────────────────────────────────────────────

def predict_all_tickers(tickers: list[str]) -> list[dict]:
    results = []
    for ticker in tickers:
        try:
            results.append(predict_ticker(ticker))
        except Exception as e:
            logger.warning(f"[predictor] Skip {ticker}: {e}")
            results.append({"ticker": ticker, "signal": "ERROR", "message": str(e)})
    return results