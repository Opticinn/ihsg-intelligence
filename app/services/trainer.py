"""
app/services/trainer.py  (v2 — anti-overfitting)
=================================================
Perubahan:
  1. XGBoost params diperketat untuk kurangi overfitting:
     - max_depth turun: 4 → 3
     - min_child_weight naik: 3 → 10
     - subsample: 0.8 → 0.7
     - colsample_bytree: 0.8 → 0.6
     - reg_alpha & reg_lambda dinaikkan
  2. Tidak pakai ticker_encoded (noise, tidak informatif)
  3. MIN_ACCURACY disesuaikan ke target realistis
"""

import json
import logging
import os
import pickle
import tempfile
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from app.services.features import (
    FEATURE_COLS,
    LABEL_COL,
    build_features_all_tickers,
)
from app.services.ingestion import IHSG_TICKERS

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "mlruns")
EXPERIMENT_NAME     = "ihsg-price-direction"
MODEL_NAME          = "ihsg-xgb-classifier"
TEST_RATIO          = 0.2
MIN_ACCURACY        = 0.42   # realistis: random=33%, target 42%+
RANDOM_STATE        = 42

# ── XGBoost — aggressively regularized ──
XGB_PARAMS = {
    "n_estimators":       500,
    "max_depth":          3,       # ↓ dari 4 — cegah kompleksitas berlebih
    "learning_rate":      0.02,    # ↓ dari 0.03 — belajar perlahan tapi stabil
    "subsample":          0.7,     # ↓ dari 0.8 — lebih banyak randomness
    "colsample_bytree":   0.6,     # ↓ dari 0.8 — paksa pakai subset fitur
    "min_child_weight":   10,      # ↑ dari 3 — leaf harus punya lebih banyak sampel
    "gamma":              0.3,     # ↑ dari 0.1 — lebih konservatif split
    "reg_alpha":          1.0,     # ↑ — L1 regularization lebih kuat
    "reg_lambda":         5.0,     # ↑ — L2 regularization lebih kuat
    "eval_metric":        "mlogloss",
    "random_state":       RANDOM_STATE,
    "n_jobs":             -1,
    "early_stopping_rounds": 30,
}


def _time_series_split(df: pd.DataFrame, test_ratio: float = TEST_RATIO):
    split_idx = int(len(df) * (1 - test_ratio))
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def train_model(tickers: list[str] = None) -> dict:
    if tickers is None:
        tickers = IHSG_TICKERS

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    logger.info(f"[trainer] Mulai training untuk {len(tickers)} ticker...")

    # ── 1. Build fitur ──
    ticker_dfs = build_features_all_tickers(tickers)
    if not ticker_dfs:
        return {"status": "error", "message": "No feature data available"}

    logger.info(f"[trainer] Berhasil build fitur untuk {len(ticker_dfs)} ticker.")

    # ── 2. Gabungkan (tanpa ticker_encoded — terlalu noisy) ──
    combined = (
        pd.concat(ticker_dfs.values(), ignore_index=True)
        .sort_values("date")
        .reset_index(drop=True)
    )
    logger.info(f"[trainer] Total dataset: {len(combined)} baris")

    # ── 3. Split ──
    train_df, test_df = _time_series_split(combined)

    X_train = train_df[FEATURE_COLS].astype(float)
    y_train = train_df[LABEL_COL].astype(int)
    X_test  = test_df[FEATURE_COLS].astype(float)
    y_test  = test_df[LABEL_COL].astype(int)

    logger.info(f"[trainer] Train: {len(X_train)} | Test: {len(X_test)}")
    logger.info(f"[trainer] Label dist train: {y_train.value_counts().sort_index().to_dict()}")
    logger.info(f"[trainer] Label dist test : {y_test.value_counts().sort_index().to_dict()}")

    # ── 4. Balanced sample weights ──
    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        logger.info(f"[trainer] MLflow run_id: {run_id}")

        mlflow.log_params({
            **{k: v for k, v in XGB_PARAMS.items()},
            "test_ratio":    TEST_RATIO,
            "n_tickers":     len(ticker_dfs),
            "total_rows":    len(combined),
            "train_rows":    len(X_train),
            "test_rows":     len(X_test),
            "label_lookahead": 1,
            "label_threshold": 0.005,
            "class_weight":  "balanced",
        })
        mlflow.log_dict({"features": FEATURE_COLS, "label": LABEL_COL}, "feature_info.json")

        # ── 5. Train ──
        model = XGBClassifier(**XGB_PARAMS)
        model.fit(
            X_train, y_train,
            sample_weight=sample_weights,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        # ── 6. Evaluasi test ──
        y_pred      = model.predict(X_test)
        y_pred_train= model.predict(X_train)

        train_acc   = accuracy_score(y_train, y_pred_train)
        test_acc    = accuracy_score(y_test,  y_pred)
        f1_macro    = f1_score(y_test, y_pred, average="macro",    zero_division=0)
        f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        overfit_gap = train_acc - test_acc
        report      = classification_report(
            y_test, y_pred,
            target_names=["SELL", "HOLD", "BUY"],
            zero_division=0,
        )
        cm = confusion_matrix(y_test, y_pred)

        logger.info(f"\n[trainer] Classification Report:\n{report}")
        logger.info(
            f"[trainer] Train acc: {train_acc:.4f} | "
            f"Test acc: {test_acc:.4f} | "
            f"Gap (overfit): {overfit_gap:.4f}"
        )

        # Peringatan overfitting
        if overfit_gap > 0.15:
            logger.warning(f"[trainer] ⚠️  Gap {overfit_gap:.4f} — masih overfitting, pertimbangkan regularisasi lebih.")
        else:
            logger.info(f"[trainer] ✅ Gap {overfit_gap:.4f} — generalisasi bagus.")

        mlflow.log_metrics({
            "train_accuracy": round(train_acc, 4),
            "test_accuracy":  round(test_acc, 4),
            "overfit_gap":    round(overfit_gap, 4),
            "f1_macro":       round(f1_macro, 4),
            "f1_weighted":    round(f1_weighted, 4),
        })
        mlflow.log_dict(
            {"confusion_matrix": cm.tolist(), "labels": ["SELL", "HOLD", "BUY"]},
            "confusion_matrix.json"
        )
        mlflow.log_text(report, "classification_report.txt")

        # ── 7. Simpan model via pickle ──
        local_model_dir  = Path("mlruns") / "saved_models"
        local_model_dir.mkdir(parents=True, exist_ok=True)
        local_model_path = local_model_dir / "ihsg_xgb_latest.pkl"

        with open(local_model_path, "wb") as f:
            pickle.dump({
                "model":        model,
                "feature_cols": FEATURE_COLS,
                "run_id":       run_id,
                "train_accuracy": round(train_acc, 4),
                "test_accuracy":  round(test_acc, 4),
                "overfit_gap":    round(overfit_gap, 4),
            }, f)

        mlflow.log_artifact(str(local_model_path), artifact_path="model")
        mlflow.log_dict(
            {"feature_cols": FEATURE_COLS, "n_features": len(FEATURE_COLS)},
            "model/feature_cols.json"
        )

        logger.info(f"[trainer] Model disimpan ke: {local_model_path}")

        registered = test_acc >= MIN_ACCURACY
        if registered:
            logger.info(f"[trainer] ✅ Test acc {test_acc:.4f} >= {MIN_ACCURACY} — threshold terpenuhi!")
        else:
            logger.warning(f"[trainer] ⚠️  Test acc {test_acc:.4f} < {MIN_ACCURACY} — belum capai threshold.")

        result = {
            "status":         "success",
            "run_id":         run_id,
            "train_accuracy": round(train_acc, 4),
            "test_accuracy":  round(test_acc, 4),
            "overfit_gap":    round(overfit_gap, 4),
            "f1_macro":       round(f1_macro, 4),
            "f1_weighted":    round(f1_weighted, 4),
            "train_rows":     len(X_train),
            "test_rows":      len(X_test),
            "n_tickers":      len(ticker_dfs),
            "model_path":     str(local_model_path),
            "registered":     registered,
        }
        logger.info(f"[trainer] Selesai: {result}")
        return result


if __name__ == "__main__":
    result = train_model()
    print("\n" + "=" * 55)
    print("HASIL TRAINING v2:")
    print(f"  Train accuracy : {result.get('train_accuracy')}")
    print(f"  Test accuracy  : {result.get('test_accuracy')}")
    print(f"  Overfit gap    : {result.get('overfit_gap')} (target: < 0.10)")
    print(f"  F1 macro       : {result.get('f1_macro')}")
    print(f"  Registered     : {result.get('registered')}")
    print("=" * 55)
    print("\nBuka MLflow UI: mlflow ui --backend-store-uri mlruns")