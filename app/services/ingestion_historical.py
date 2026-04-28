"""
app/services/ingestion_historical.py
=====================================
Script sekali jalan untuk mengisi data historis 1 tahun ke belakang.
Jalankan SATU KALI sebelum training model.

Usage:
    python -m app.services.ingestion_historical

Setelah selesai, cek hasilnya:
    python -m app.services.ingestion_historical --check
"""

import sys
import logging
import time

import yfinance as yf
import pandas as pd

from app.core.database import SessionLocal
from app.models.stock import StockHistory
from app.services.ingestion import IHSG_TICKERS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

PERIOD = "1y"   # 1 tahun — cukup untuk training & semua indikator teknikal


def backfill_ticker(ticker: str, period: str = PERIOD) -> int:
    """
    Fetch data historis satu ticker dari yfinance dan simpan ke DB.
    Menghapus data lama ticker tsb sebelum insert (upsert sederhana).

    Returns:
        Jumlah baris yang berhasil disimpan.
    """
    logger.info(f"  Fetching {ticker} ({period})...")

    try:
        raw = yf.Ticker(ticker).history(period=period)
    except Exception as e:
        logger.error(f"  ❌ yfinance error {ticker}: {e}")
        return 0

    if raw.empty:
        logger.warning(f"  ⚠️  Data kosong untuk {ticker} — mungkin ticker salah atau delisted.")
        return 0

    df = raw.reset_index().copy()
    df["Ticker"] = ticker

    # Normalise kolom
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)

    df.rename(columns={
        "Date":   "date",
        "Open":   "open_price",
        "High":   "high_price",
        "Low":    "low_price",
        "Close":  "close_price",
        "Volume": "volume",
        "Ticker": "ticker",
    }, inplace=True)

    cols = ["date", "ticker", "open_price", "high_price", "low_price", "close_price", "volume"]
    df = df[cols].dropna(subset=["close_price"])

    db = SessionLocal()
    try:
        # Hapus data lama ticker ini
        deleted = db.query(StockHistory).filter(StockHistory.ticker == ticker).delete()
        if deleted:
            logger.info(f"  🗑️  Hapus {deleted} baris lama {ticker}")

        records = df.to_dict(orient="records")
        db.bulk_save_objects([StockHistory(**r) for r in records])
        db.commit()
        logger.info(f"  ✅ {ticker}: {len(records)} baris disimpan.")
        return len(records)

    except Exception as e:
        db.rollback()
        logger.error(f"  ❌ DB error {ticker}: {e}")
        return 0
    finally:
        db.close()


def check_db_counts():
    """Tampilkan jumlah baris per ticker di DB."""
    db = SessionLocal()
    try:
        from sqlalchemy import func
        rows = (
            db.query(StockHistory.ticker, func.count(StockHistory.id).label("n"))
            .group_by(StockHistory.ticker)
            .order_by(StockHistory.ticker)
            .all()
        )
        print("\n📊 Data di database saat ini:")
        print(f"  {'Ticker':<15} {'Baris':>6}  {'Status'}")
        print("  " + "-" * 40)
        total = 0
        for ticker, n in rows:
            status = "✅ siap training" if n >= 60 else f"⚠️  kurang ({n}<60)"
            print(f"  {ticker:<15} {n:>6}  {status}")
            total += n
        print("  " + "-" * 40)
        print(f"  {'TOTAL':<15} {total:>6} baris\n")
    finally:
        db.close()


def run_backfill():
    """Backfill semua ticker sekaligus."""
    logger.info(f"🚀 Mulai historical backfill — {len(IHSG_TICKERS)} ticker, period={PERIOD}")
    logger.info("   (Proses ini ~2–5 menit, sabar ya...)\n")

    total_saved = 0
    failed = []

    for i, ticker in enumerate(IHSG_TICKERS, 1):
        logger.info(f"[{i}/{len(IHSG_TICKERS)}] {ticker}")
        n = backfill_ticker(ticker)
        total_saved += n
        if n == 0:
            failed.append(ticker)
        # Jeda kecil agar tidak kena rate-limit yfinance
        time.sleep(0.5)

    print("\n" + "=" * 50)
    print("📋 HASIL BACKFILL:")
    print(f"  Total baris disimpan : {total_saved}")
    print(f"  Ticker berhasil      : {len(IHSG_TICKERS) - len(failed)}/{len(IHSG_TICKERS)}")
    if failed:
        print(f"  Ticker gagal/kosong  : {', '.join(failed)}")
    print("=" * 50)

    check_db_counts()

    if total_saved > 0:
        print("✅ Sekarang jalankan training:")
        print("   python -m app.services.trainer\n")
    else:
        print("❌ Tidak ada data yang tersimpan. Cek koneksi DB dan yfinance.\n")


if __name__ == "__main__":
    if "--check" in sys.argv:
        check_db_counts()
    else:
        run_backfill()