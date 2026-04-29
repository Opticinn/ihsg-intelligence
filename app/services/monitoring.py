"""
app/services/monitoring.py  (v3 — Unified Report)
===================================================
Semua ticker dalam SATU report per kategori:
  1. data_quality.html     — kualitas data semua 22 ticker
  2. data_drift.html       — drift fitur teknikal semua ticker
  3. model_performance.html — accuracy, F1, distribusi sinyal
  4. monitoring_summary.json — ringkasan untuk Grafana/API
"""

import json
import logging
import os
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

warnings.filterwarnings("ignore")
load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

DATABASE_URL = os.getenv("DATABASE_URL")
engine       = create_engine(DATABASE_URL)
REPORTS_DIR  = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)

from app.services.ingestion import IHSG_TICKERS


# ─────────────────────────────────────────────
# 1. DATA QUALITY — semua ticker, satu report
# ─────────────────────────────────────────────

def run_data_quality_all(tickers: list[str] = None) -> dict:
    """
    Gabungkan semua ticker jadi satu DataFrame,
    lalu generate SATU report kualitas data.
    """
    from evidently.report import Report
    from evidently.metric_preset import DataQualityPreset

    if tickers is None:
        tickers = IHSG_TICKERS

    all_ref  = []
    all_curr = []
    ticker_stats = {}

    for ticker in tickers:
        try:
            query = f"SELECT * FROM stock_history WHERE ticker = '{ticker}' ORDER BY date ASC"
            df    = pd.read_sql(query, engine)

            if len(df) < 14:
                logger.warning(f"[monitoring] {ticker}: data kurang ({len(df)} baris)")
                ticker_stats[ticker] = {"status": "insufficient_data", "rows": len(df)}
                continue

            ghost_count = int(((df["close_price"] == 0) | (df["close_price"].isna())).sum())

            split = int(len(df) * 0.8)
            all_ref.append(df.iloc[:split])
            all_curr.append(df.iloc[split:])

            ticker_stats[ticker] = {
                "status":     "ok",
                "rows":       len(df),
                "ghost_rows": ghost_count,
            }
            logger.info(f"[monitoring] ✅ {ticker}: {len(df)} baris, {ghost_count} ghost rows")

        except Exception as e:
            logger.error(f"[monitoring] ❌ {ticker}: {e}")
            ticker_stats[ticker] = {"status": "error", "message": str(e)}

    if not all_ref:
        return {"status": "error", "message": "Tidak ada data cukup"}

    ref_df  = pd.concat(all_ref,  ignore_index=True)
    curr_df = pd.concat(all_curr, ignore_index=True)

    # Pilih kolom numerik saja untuk Evidently
    num_cols = ["open_price", "high_price", "low_price", "close_price", "volume"]
    num_cols = [c for c in num_cols if c in ref_df.columns]

    report = Report(metrics=[DataQualityPreset()])
    report.run(
        reference_data = ref_df[num_cols],
        current_data   = curr_df[num_cols],
    )

    path = REPORTS_DIR / "data_quality.html"
    report.save_html(str(path))

    ok_count = sum(1 for v in ticker_stats.values() if v.get("status") == "ok")
    logger.info(f"[monitoring] Data Quality report → {path} ({ok_count}/{len(tickers)} ticker OK)")

    return {
        "status":       "ok",
        "report_path":  str(path),
        "tickers_ok":   ok_count,
        "tickers_total":len(tickers),
        "per_ticker":   ticker_stats,
    }


# ─────────────────────────────────────────────
# 2. DATA DRIFT — semua ticker, satu report
# ─────────────────────────────────────────────

def run_data_drift(reference_days: int = 60, current_days: int = 14) -> dict:
    """
    Bandingkan distribusi fitur teknikal semua ticker:
    - Reference: data lama (baseline normal)
    - Current:   data 14 hari terakhir
    Kalau >30% fitur drift → trigger retrain.
    """
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset
    from app.services.features import build_features

    all_ref  = []
    all_curr = []

    for ticker in IHSG_TICKERS:
        try:
            df = build_features(ticker)
            if df is None or len(df) < reference_days + current_days:
                continue

            all_ref.append(df.iloc[-(reference_days + current_days):-current_days].copy())
            all_curr.append(df.iloc[-current_days:].copy())
        except Exception as e:
            logger.warning(f"[monitoring] Drift skip {ticker}: {e}")

    if not all_ref:
        return {"status": "error", "message": "Tidak cukup data untuk drift check"}

    ref_df  = pd.concat(all_ref,  ignore_index=True)
    curr_df = pd.concat(all_curr, ignore_index=True)

    num_cols = [c for c in ref_df.columns if ref_df[c].dtype in [np.float64, np.int64]]

    report = Report(metrics=[DataDriftPreset()])
    report.run(
        reference_data = ref_df[num_cols],
        current_data   = curr_df[num_cols],
    )

    path = REPORTS_DIR / "data_drift.html"
    report.save_html(str(path))

    report_dict  = report.as_dict()
    drift_result = report_dict.get("metrics", [{}])[0].get("result", {})
    drift_share  = drift_result.get("share_of_drifted_columns", 0)
    n_drifted    = drift_result.get("number_of_drifted_columns", 0)
    # Hanya retrain kalau drift ekstrem DAN sudah waktunya
    today_wib = datetime.utcnow().hour + 7  # approximate WIB
    weekday   = datetime.utcnow().weekday() # 0=Senin, 3=Kamis

    is_scheduled_day = weekday in (0, 3)  # Senin atau Kamis

    should_retrain = (
        is_scheduled_day and drift_share > 0.30  # jadwal rutin
    ) or (
        drift_share > 0.80  # ekstrem banget, kapanpun
    )

    logger.info(
        f"[monitoring] Drift: {n_drifted} kolom ({drift_share:.1%}) "
        f"{'→ ⚠️ RETRAIN' if should_retrain else '→ ✅ OK'}"
    )

    return {
        "status":         "ok",
        "drift_share":    round(drift_share, 4),
        "n_drifted":      n_drifted,
        "should_retrain": should_retrain,
        "report_path":    str(path),
    }


# ─────────────────────────────────────────────
# 3. MODEL PERFORMANCE — satu report
# ─────────────────────────────────────────────

def run_model_performance() -> dict:
    """
    Evaluasi performa model XGBoost pada data semua ticker.
    Generate satu report performa model.
    """
    from evidently.report import Report
    from evidently.metric_preset import ClassificationPreset
    from evidently.metrics import ColumnDistributionMetric
    from app.services.features import build_features, FEATURE_COLS, LABEL_COL
    from sklearn.metrics import accuracy_score, f1_score
    import pickle

    model_path = Path("mlruns") / "saved_models" / "ihsg_xgb_latest.pkl"
    if not model_path.exists():
        return {"status": "error", "message": "Model belum ditraining"}

    with open(model_path, "rb") as f:
        payload = pickle.load(f)

    model        = payload["model"]
    feature_cols = payload["feature_cols"]
    baseline_acc = payload.get("test_accuracy", payload.get("accuracy", 0))

    all_dfs = []
    for ticker in IHSG_TICKERS:
        try:
            df = build_features(ticker)
            if df is not None and len(df) > 20:
                all_dfs.append(df)
        except Exception:
            pass

    if not all_dfs:
        return {"status": "error", "message": "Tidak ada data fitur"}

    full_df = pd.concat(all_dfs, ignore_index=True)
    eval_df = full_df.dropna(subset=feature_cols + [LABEL_COL])

    X       = eval_df[feature_cols].astype(float)
    y_true  = eval_df[LABEL_COL].astype(int)
    y_pred  = model.predict(X).astype(int)
    y_proba = model.predict_proba(X)

    signal_map = {0: "SELL", 1: "HOLD", 2: "BUY"}
    perf_df = pd.DataFrame({
        "target":     y_true.map(signal_map).values,
        "prediction": pd.Series(y_pred).map(signal_map).values,
        "prob_SELL":  y_proba[:, 0],
        "prob_HOLD":  y_proba[:, 1],
        "prob_BUY":   y_proba[:, 2],
    })

    split   = int(len(perf_df) * 0.8)
    ref_df  = perf_df.iloc[:split]
    curr_df = perf_df.iloc[split:]

    report = Report(metrics=[
        ClassificationPreset(),
        ColumnDistributionMetric(column_name="prediction"),
    ])
    report.run(reference_data=ref_df, current_data=curr_df)

    path = REPORTS_DIR / "model_performance.html"
    report.save_html(str(path))

    current_acc = accuracy_score(y_true, y_pred)
    f1_macro    = f1_score(y_true, y_pred, average="macro")
    acc_drift   = baseline_acc - current_acc
    signal_dist = pd.Series(y_pred).map(signal_map).value_counts().to_dict()

    logger.info(
        f"[monitoring] Model: acc={current_acc:.4f} | f1={f1_macro:.4f} | "
        f"baseline={baseline_acc:.4f} | drift={acc_drift:+.4f} "
        f"{'→ ⚠️ RETRAIN' if acc_drift > 0.10 else '→ ✅ OK'}"
    )

    return {
        "status":         "ok",
        "current_acc":    round(current_acc, 4),
        "baseline_acc":   round(baseline_acc, 4),
        "acc_drift":      round(acc_drift, 4),
        "f1_macro":       round(f1_macro, 4),
        "signal_dist":    signal_dist,
        "should_retrain": acc_drift > 0.10,
        "report_path":    str(path),
    }


def save_to_db(summary: dict):
    """Simpan hasil monitoring ke tabel monitoring_logs di PostgreSQL."""
    try:
        from sqlalchemy import text
        drift = summary.get("data_drift", {})
        model = summary.get("model_performance", {})
        dq    = summary.get("data_quality", {})
        sig   = model.get("signal_dist", {})

        with engine.connect() as conn:
            conn.execute(text("""
                INSERT INTO monitoring_logs (
                    run_at, tickers_ok, tickers_total,
                    drift_share, n_drifted, should_retrain,
                    model_acc, model_baseline, acc_drift, f1_macro,
                    signal_buy, signal_hold, signal_sell
                ) VALUES (
                    NOW(), :tickers_ok, :tickers_total,
                    :drift_share, :n_drifted, :should_retrain,
                    :model_acc, :model_baseline, :acc_drift, :f1_macro,
                    :signal_buy, :signal_hold, :signal_sell
                )
            """), {
                "tickers_ok":    dq.get("tickers_ok", 0),
                "tickers_total": dq.get("tickers_total", 0),
                "drift_share":   drift.get("drift_share", 0),
                "n_drifted":     drift.get("n_drifted", 0),
                "should_retrain":drift.get("should_retrain", False),
                "model_acc":     model.get("current_acc", 0),
                "model_baseline":model.get("baseline_acc", 0),
                "acc_drift":     model.get("acc_drift", 0),
                "f1_macro":      model.get("f1_macro", 0),
                "signal_buy":    sig.get("BUY", 0),
                "signal_hold":   sig.get("HOLD", 0),
                "signal_sell":   sig.get("SELL", 0),
            })
            conn.commit()
        logger.info("[monitoring] ✅ Hasil disimpan ke tabel monitoring_logs")
    except Exception as e:
        logger.error(f"[monitoring] ❌ Gagal simpan ke DB: {e}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def run_full_monitoring() -> dict:
    """Jalankan semua monitoring, simpan summary JSON untuk Grafana."""
    logger.info("=" * 60)
    logger.info("🔍 IHSG Intelligence — Full Monitoring Pipeline")
    logger.info(f"   {datetime.utcnow().strftime('%d %B %Y %H:%M UTC')}")
    logger.info("=" * 60)

    summary = {
        "run_at":            datetime.utcnow().isoformat(),
        "data_quality":      {},
        "data_drift":        {},
        "model_performance": {},
    }

    logger.info("\n📊 [1/3] Data Quality — semua ticker dalam 1 report...")
    summary["data_quality"] = run_data_quality_all()

    logger.info("\n📈 [2/3] Data Drift Detection...")
    summary["data_drift"] = run_data_drift()

    logger.info("\n🤖 [3/3] Model Performance Evaluation...")
    summary["model_performance"] = run_model_performance()

    # Simpan JSON untuk Grafana
    summary_path = REPORTS_DIR / "monitoring_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    drift = summary["data_drift"]
    model = summary["model_performance"]
    dq    = summary["data_quality"]

    logger.info("\n" + "=" * 60)
    logger.info("📋 RINGKASAN MONITORING:")
    logger.info(f"  Data Quality  : {dq.get('tickers_ok', 0)}/{dq.get('tickers_total', 0)} ticker OK → {dq.get('report_path', '')}")
    logger.info(f"  Data Drift    : {drift.get('drift_share', 0):.1%} drift {'⚠️ RETRAIN' if drift.get('should_retrain') else '✅ OK'} → {drift.get('report_path', '')}")
    logger.info(f"  Model Acc     : {model.get('current_acc', 0):.4f} (drift: {model.get('acc_drift', 0):+.4f}) {'⚠️ RETRAIN' if model.get('should_retrain') else '✅ OK'} → {model.get('report_path', '')}")
    logger.info(f"  Summary JSON  : {summary_path}")
    logger.info("=" * 60)


    save_to_db(summary)
    
    # Kirim notifikasi Telegram
    from app.services.telegram import notify_monitoring_summary, notify_drift_alert
    notify_monitoring_summary(summary)
    notify_drift_alert(
        drift_share=summary["data_drift"].get("drift_share", 0),
        n_drifted=summary["data_drift"].get("n_drifted", 0)
    )
    
    weekday        = datetime.utcnow().weekday()
    is_sched_day   = weekday in (0, 3)  # Senin atau Kamis
    drift_share    = summary["data_drift"].get("drift_share", 0)
    acc_drift_val  = summary["model_performance"].get("acc_drift", 0)

    combined_retrain = (
        acc_drift_val > 0.10 or                          # accuracy drop parah
        (is_sched_day and drift_share > 0.30) or         # jadwal rutin
        (drift_share > 0.80 and acc_drift_val > 0.03)    # ekstrem + mulai turun
    )

    if combined_retrain:
        from app.services.telegram import send_message
        send_message(
            "🔁 *Retrain Dijadwalkan — IHSG Intelligence*\n\n"
            f"📈 Drift: `{drift_share:.1%}`\n"
            f"📉 Acc drift: `{acc_drift_val:+.4f}`\n"
            f"📅 Hari jadwal: `{'Ya' if is_sched_day else 'Tidak'}`\n\n"
            "_Retrain akan berjalan otomatis via GitHub Actions._"
        )
    
    return summary


if __name__ == "__main__":
    run_full_monitoring()