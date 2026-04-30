"""
app/services/retrain_trigger.py
================================
Adaptive Retraining Logic — "The 3-Day Sweet Spot"

Retraining Protocol:
  Base Schedule  : Monday & Thursday (every 3 trading days)
  Emergency Override:
    1. IHSG moves > ±2.5% in a single day
    2. Realized volatility > 90th percentile of 30-day rolling window
    3. Model accuracy drift > 10% from last baseline

Usage:
  from app.services.retrain_trigger import should_retrain
  result = should_retrain()
  if result["should"]:
      train_model()
"""

import logging
import os
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

from app.core.database import SessionLocal
from app.models.stock import StockHistory

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

IHSG_TICKER         = "^JKSE"
IHSG_MOVE_THRESHOLD = 0.025   # 2.5% emergency trigger
VOL_PERCENTILE      = 90      # 90th percentile of 30-day window
VOL_LOOKBACK_DAYS   = 30
MIN_HOURS_BETWEEN   = 18      # minimum cooldown between retrains (hours)
STATE_FILE          = Path("mlruns") / "retrain_state.pkl"

WATCH_TICKERS = ["BBCA.JK", "GOTO.JK", "ARTO.JK", "BBRI.JK"]


# ─────────────────────────────────────────────
# STATE MANAGEMENT
# ─────────────────────────────────────────────

def _load_state() -> dict:
    """Load retrain state from disk."""
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass
    return {
        "last_retrain_at":  None,
        "last_accuracy":    None,
        "retrain_count":    0,
        "last_trigger":     None,
    }


def _save_state(state: dict):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "wb") as f:
        pickle.dump(state, f)


def record_retrain(accuracy: float, trigger: str = "manual"):
    """
    Call this after training completes to update state.
    Should be called in trainer.py after train_model() succeeds.
    """
    state = _load_state()
    state["last_retrain_at"] = datetime.utcnow()
    state["last_accuracy"]   = accuracy
    state["retrain_count"]  += 1
    state["last_trigger"]    = trigger
    _save_state(state)
    logger.info(
        f"[retrain] State updated — accuracy={accuracy:.4f} | "
        f"trigger={trigger} | total_retrains={state['retrain_count']}"
    )


# ─────────────────────────────────────────────
# CHECK 1: SCHEDULE (Mon / Thu)
# ─────────────────────────────────────────────

def _check_schedule() -> tuple[bool, str]:
    """Check if today is a scheduled retrain day (Mon=0, Thu=3)."""
    today_wib = datetime.utcnow() + timedelta(hours=7)
    weekday   = today_wib.weekday()

    if weekday in (0, 3):
        return True, f"Scheduled retrain day (WIB weekday={weekday})"
    return False, f"Not a retrain day (weekday={weekday}, retrain on 0=Mon & 3=Thu)"


# ─────────────────────────────────────────────
# CHECK 2: IHSG MOVE > 2.5%
# ─────────────────────────────────────────────

def _check_ihsg_move() -> tuple[bool, str]:
    """Check if IHSG moved > ±2.5% today."""
    try:
        df = yf.Ticker(IHSG_TICKER).history(period="3d")
        if df.empty or len(df) < 2:
            return False, "Could not fetch IHSG data"

        closes       = df["Close"].dropna()
        daily_return = (closes.iloc[-1] - closes.iloc[-2]) / closes.iloc[-2]
        abs_move     = abs(daily_return)

        if abs_move >= IHSG_MOVE_THRESHOLD:
            direction = "up" if daily_return > 0 else "down"
            return True, (
                f"EMERGENCY: IHSG moved {direction} {daily_return:.2%} today "
                f"(threshold: ±{IHSG_MOVE_THRESHOLD:.1%})"
            )

        return False, f"IHSG move today: {daily_return:.2%} (below threshold)"

    except Exception as e:
        logger.warning(f"[retrain] Failed IHSG move check: {e}")
        return False, f"IHSG fetch error: {e}"


# ─────────────────────────────────────────────
# CHECK 3: REALIZED VOLATILITY > 90th PERCENTILE
# ─────────────────────────────────────────────

def _check_volatility() -> tuple[bool, str]:
    """Check if today's volatility exceeds 90th percentile of 30-day window."""
    db = SessionLocal()
    try:
        rows = (
            db.query(StockHistory)
            .filter(
                StockHistory.ticker == "BBCA.JK",
                StockHistory.close_price > 0,
            )
            .order_by(StockHistory.date.desc())
            .limit(VOL_LOOKBACK_DAYS + 5)
            .all()
        )

        if len(rows) < 10:
            return False, "Insufficient data for volatility check"

        closes        = pd.Series([r.close_price for r in reversed(rows)], dtype=float)
        returns       = closes.pct_change().dropna().abs()
        today_vol     = float(returns.iloc[-1])
        historical    = returns.iloc[:-1]
        threshold_vol = float(np.percentile(historical, VOL_PERCENTILE))

        if today_vol >= threshold_vol:
            return True, (
                f"EMERGENCY: Volatility today {today_vol:.2%} "
                f">= {VOL_PERCENTILE}th percentile ({threshold_vol:.2%})"
            )

        return False, (
            f"Normal volatility: {today_vol:.2%} "
            f"(p{VOL_PERCENTILE} threshold: {threshold_vol:.2%})"
        )

    except Exception as e:
        logger.warning(f"[retrain] Volatility check failed: {e}")
        return False, f"Volatility check error: {e}"
    finally:
        db.close()


# ─────────────────────────────────────────────
# CHECK 4: MODEL ACCURACY DRIFT
# ─────────────────────────────────────────────

def _check_accuracy_drift() -> tuple[bool, str]:
    """Check if model accuracy dropped > 10% from last baseline."""
    state        = _load_state()
    baseline_acc = state.get("last_accuracy")

    if baseline_acc is None:
        return False, "No baseline accuracy found (model never trained via trigger)"

    try:
        from app.services.features import build_features, FEATURE_COLS, LABEL_COL
        from sklearn.metrics import accuracy_score
        import pickle as pkl

        model_path = Path("mlruns") / "saved_models" / "ihsg_xgb_latest.pkl"
        if not model_path.exists():
            return False, "Model file not found"

        with open(model_path, "rb") as f:
            payload = pkl.load(f)

        model        = payload["model"]
        feature_cols = payload["feature_cols"]

        df = build_features("BBCA.JK")
        if df is None or len(df) < 20:
            return False, "Insufficient data for drift check"

        eval_df     = df.iloc[-int(len(df) * 0.2):]
        X_eval      = eval_df[feature_cols].astype(float)
        y_eval      = eval_df[LABEL_COL].astype(int)
        current_acc = accuracy_score(y_eval, model.predict(X_eval))

        drift = baseline_acc - current_acc
        if drift >= 0.10:
            return True, (
                f"DRIFT DETECTED: Accuracy dropped {drift:.2%} "
                f"(baseline: {baseline_acc:.4f} → current: {current_acc:.4f})"
            )

        return False, (
            f"Accuracy stable: baseline={baseline_acc:.4f} | "
            f"current={current_acc:.4f} | drift={drift:.2%}"
        )

    except Exception as e:
        logger.warning(f"[retrain] Drift check failed: {e}")
        return False, f"Drift check error: {e}"


# ─────────────────────────────────────────────
# MAIN FUNCTION
# ─────────────────────────────────────────────

def should_retrain(force: bool = False) -> dict:
    """
    Evaluate all conditions and determine if retraining is needed.

    Args:
        force: skip all checks and retrain immediately

    Returns:
        dict {
            "should": bool,
            "trigger": str,
            "reason": str,
            "checks": dict,
            "last_retrain": str,
            "retrain_count": int
        }
    """
    state   = _load_state()
    last_at = state.get("last_retrain_at")

    # Hard cooldown check
    if not force and last_at:
        elapsed = (datetime.utcnow() - last_at).total_seconds() / 3600
        if elapsed < MIN_HOURS_BETWEEN:
            return {
                "should":        False,
                "trigger":       "cooldown",
                "reason":        f"Last retrain {elapsed:.1f}h ago (cooldown: {MIN_HOURS_BETWEEN}h)",
                "checks":        {},
                "last_retrain":  last_at.strftime("%Y-%m-%d %H:%M UTC"),
                "retrain_count": state.get("retrain_count", 0),
            }

    if force:
        return {
            "should":        True,
            "trigger":       "force",
            "reason":        "Force retrain requested",
            "checks":        {},
            "last_retrain":  last_at.strftime("%Y-%m-%d %H:%M UTC") if last_at else "Never",
            "retrain_count": state.get("retrain_count", 0),
        }

    logger.info("[retrain] Evaluating retrain conditions...")

    sched_ok, sched_reason = _check_schedule()
    ihsg_ok,  ihsg_reason  = _check_ihsg_move()
    vol_ok,   vol_reason   = _check_volatility()
    drift_ok, drift_reason = _check_accuracy_drift()

    checks = {
        "schedule":   {"triggered": sched_ok,  "detail": sched_reason},
        "ihsg_move":  {"triggered": ihsg_ok,   "detail": ihsg_reason},
        "volatility": {"triggered": vol_ok,    "detail": vol_reason},
        "drift":      {"triggered": drift_ok,  "detail": drift_reason},
    }

    for name, result in checks.items():
        icon = "✅" if result["triggered"] else "⬜"
        logger.info(f"[retrain] {icon} {name}: {result['detail']}")

    # Priority: emergency > schedule
    if ihsg_ok:
        trigger, reason, should = "ihsg_move",  ihsg_reason,  True
    elif vol_ok:
        trigger, reason, should = "volatility", vol_reason,   True
    elif drift_ok:
        trigger, reason, should = "drift",      drift_reason, True
    elif sched_ok:
        trigger, reason, should = "schedule",   sched_reason, True
    else:
        trigger, reason, should = "none", "No retrain condition met", False

    if should:
        logger.info(f"[retrain] RETRAIN NEEDED — trigger: {trigger}")
    else:
        logger.info(f"[retrain] Skip retrain — {reason}")

    return {
        "should":        should,
        "trigger":       trigger,
        "reason":        reason,
        "checks":        checks,
        "last_retrain":  last_at.strftime("%Y-%m-%d %H:%M UTC") if last_at else "Never",
        "last_accuracy": state.get("last_accuracy"),
        "retrain_count": state.get("retrain_count", 0),
    }


def get_retrain_status() -> dict:
    """Return current retrain status without re-evaluating."""
    state   = _load_state()
    last_at = state.get("last_retrain_at")
    return {
        "last_retrain":    last_at.strftime("%Y-%m-%d %H:%M UTC") if last_at else "Never",
        "last_accuracy":   state.get("last_accuracy"),
        "retrain_count":   state.get("retrain_count", 0),
        "last_trigger":    state.get("last_trigger"),
        "next_schedule":   "Every Monday & Thursday (WIB)",
        "emergency_triggers": {
            "ihsg_move_threshold":      f"±{IHSG_MOVE_THRESHOLD:.1%}",
            "volatility_percentile":    VOL_PERCENTILE,
            "accuracy_drift_threshold": "10%",
            "cooldown_hours":           MIN_HOURS_BETWEEN,
        },
    }