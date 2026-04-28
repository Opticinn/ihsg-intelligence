"""
app/services/features.py  (v2 — FIXED)
=======================================
Perubahan dari v1:
  1. Label: lookahead=1, threshold=0.005 (dari data diagnosa)
     → distribusi SELL 41% / HOLD 26% / BUY 33% — jauh lebih seimbang
  2. Hapus fitur raw price (open/high/low/close/bb_upper/bb_lower/ema_12/ema_26)
     → ini penyebab overfitting — nilainya tidak stabil antar ticker
  3. Ganti dengan fitur NORMALIZED & RELATIVE:
     - price_vs_sma20   : harga relatif terhadap SMA20 (%)
     - price_vs_sma50   : harga relatif terhadap SMA50 (%)
     - high_low_range   : (high-low)/close — volatilitas intraday
     - close_vs_open    : (close-open)/open — candle body
  4. Tambah lag features (return kemarin, RSI kemarin)
     → beri model "memori" singkat tanpa overfitting
"""

import logging
import numpy as np
import pandas as pd

from app.core.database import SessionLocal
from app.models.stock import StockHistory

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# PARAMETER  ← hasil diagnosa
# ─────────────────────────────────────────────
RSI_PERIOD  = 14
MACD_FAST   = 12
MACD_SLOW   = 26
MACD_SIGNAL = 9
BB_PERIOD   = 20
BB_STD      = 2
SMA_PERIODS = [7, 20, 50]
MIN_ROWS    = 60

# ← KUNCI PERBAIKAN: lookahead=1, threshold=0.005
LABEL_LOOKAHEAD = 1
LABEL_THRESHOLD = 0.005


# ─────────────────────────────────────────────
# INDIKATOR
# ─────────────────────────────────────────────

def compute_rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    delta    = close.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).rename("rsi")


def compute_macd(close: pd.Series) -> pd.DataFrame:
    ema_fast    = close.ewm(span=MACD_FAST,   adjust=False).mean()
    ema_slow    = close.ewm(span=MACD_SLOW,   adjust=False).mean()
    macd_line   = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=MACD_SIGNAL, adjust=False).mean()
    return pd.DataFrame({
        "macd_line":   macd_line,
        "macd_signal": signal_line,
        "macd_hist":   macd_line - signal_line,
    })


def compute_bollinger(close: pd.Series) -> pd.DataFrame:
    sma   = close.rolling(BB_PERIOD).mean()
    std   = close.rolling(BB_PERIOD).std()
    upper = sma + BB_STD * std
    lower = sma - BB_STD * std
    denom = (upper - lower).replace(0, np.nan)
    return pd.DataFrame({
        "bb_width": (upper - lower) / sma.replace(0, np.nan),
        "bb_pct_b": (close - lower) / denom,
    })


def compute_normalized_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fitur harga yang NORMALIZED — tidak bergantung pada skala harga absolut.
    Ini menggantikan raw price (open/high/low/close) yang menyebabkan overfitting.
    """
    close = df["close_price"]
    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()

    return pd.DataFrame({
        # Seberapa jauh harga dari moving average (dalam %)
        "price_vs_sma20":  (close - sma20) / sma20.replace(0, np.nan),
        "price_vs_sma50":  (close - sma50) / sma50.replace(0, np.nan),
        # Volatilitas intraday: range high-low dibagi close
        "high_low_range":  (df["high_price"] - df["low_price"]) / close.replace(0, np.nan),
        # Candle body: close vs open
        "close_vs_open":   (df["close_price"] - df["open_price"]) / df["open_price"].replace(0, np.nan),
    })


def compute_volume_ratio(volume: pd.Series, period: int = 20) -> pd.Series:
    avg = volume.rolling(period).mean()
    return (volume / avg.replace(0, np.nan)).rename("volume_ratio")


def compute_returns(close: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({
        "return_1d": close.pct_change(1),
        "return_3d": close.pct_change(3),
        "return_5d": close.pct_change(5),
    })


def compute_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lag features — nilai indikator 1 dan 2 hari lalu.
    Memberi model "memori" tanpa data leakage.
    """
    return pd.DataFrame({
        "rsi_lag1":        df["rsi"].shift(1),
        "rsi_lag2":        df["rsi"].shift(2),
        "macd_hist_lag1":  df["macd_hist"].shift(1),
        "return_1d_lag1":  df["return_1d"].shift(1),
        "volume_ratio_lag1": df["volume_ratio"].shift(1),
        "bb_pct_b_lag1":   df["bb_pct_b"].shift(1),
    })


# ─────────────────────────────────────────────
# LABEL
# ─────────────────────────────────────────────

def compute_label(
    close: pd.Series,
    lookahead: int   = LABEL_LOOKAHEAD,
    threshold: float = LABEL_THRESHOLD,
) -> pd.Series:
    """
    Label berdasarkan return N hari ke depan.
      return > +threshold → BUY  (2)
      return < -threshold → SELL (0)
      sisanya             → HOLD (1)
    """
    future_return = close.shift(-lookahead) / close - 1
    label = np.where(
        future_return >  threshold, 2,
        np.where(future_return < -threshold, 0, 1)
    )
    return pd.Series(label, index=close.index, name="label")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def build_features(ticker: str, period: str = "1y") -> pd.DataFrame | None:
    db = SessionLocal()
    try:
        rows = (
            db.query(StockHistory)
            .filter(StockHistory.ticker == ticker)
            .order_by(StockHistory.date.asc())
            .all()
        )
    finally:
        db.close()

    if not rows:
        logger.warning(f"[features] Tidak ada data untuk {ticker}")
        return None

    df = pd.DataFrame([{
        "date":        r.date,
        "ticker":      r.ticker,
        "open_price":  r.open_price,
        "high_price":  r.high_price,
        "low_price":   r.low_price,
        "close_price": r.close_price,
        "volume":      float(r.volume),
    } for r in rows]).sort_values("date").reset_index(drop=True)

    if len(df) < MIN_ROWS:
        logger.warning(f"[features] {ticker}: {len(df)} baris < {MIN_ROWS}. Skip.")
        return None

    close  = df["close_price"]
    volume = df["volume"]

    # ── Hitung semua indikator ──
    rsi_series  = compute_rsi(close)
    macd_df     = compute_macd(close)
    bb_df       = compute_bollinger(close)
    ret_df      = compute_returns(close)
    vol_ratio   = compute_volume_ratio(volume)
    norm_price  = compute_normalized_price_features(df)

    # Gabungkan ke df utama
    df = pd.concat([
        df[["date", "ticker"]],   # identitas saja, tidak masuk fitur
        rsi_series,
        macd_df,
        bb_df,
        ret_df,
        vol_ratio,
        norm_price,
    ], axis=1)

    # Tambah label sebelum lag (supaya shift tidak kena label)
    df["label"] = compute_label(close)

    # Tambah lag features (harus setelah kolom dasar ada)
    lag_df = compute_lag_features(df)
    df     = pd.concat([df, lag_df], axis=1)

    # Buang baris NaN (awal window & akhir karena lookahead)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    logger.info(f"[features] {ticker}: {len(df)} baris siap")
    return df


def build_features_all_tickers(tickers: list[str]) -> dict[str, pd.DataFrame]:
    result = {}
    for ticker in tickers:
        df = build_features(ticker)
        if df is not None:
            result[ticker] = df
    return result


# ─────────────────────────────────────────────
# FEATURE COLUMNS (tidak ada raw price!)
# ─────────────────────────────────────────────

FEATURE_COLS = [
    # Momentum & trend
    "rsi", "macd_line", "macd_signal", "macd_hist",
    # Volatilitas & posisi harga
    "bb_width", "bb_pct_b",
    # Harga normalized (bukan absolut)
    "price_vs_sma20", "price_vs_sma50",
    "high_low_range", "close_vs_open",
    # Return
    "return_1d", "return_3d", "return_5d",
    # Volume
    "volume_ratio",
    # Lag features
    "rsi_lag1", "rsi_lag2",
    "macd_hist_lag1",
    "return_1d_lag1",
    "volume_ratio_lag1",
    "bb_pct_b_lag1",
]

LABEL_COL = "label"