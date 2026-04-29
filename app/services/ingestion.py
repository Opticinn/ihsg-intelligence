import yfinance as yf
import pandas as pd
import logging

from app.core.database import SessionLocal
from app.models.stock import StockHistory

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

IHSG_TICKERS = [
    "BBCA.JK", "BBRI.JK", "BMRI.JK", "BBNI.JK", "ARTO.JK",
    "TLKM.JK", "GOTO.JK", "BUKA.JK", "ISAT.JK",
    "UNVR.JK", "ICBP.JK", "INDF.JK", "AMRT.JK",
    "ADRO.JK", "ITMG.JK", "PTBA.JK", "ANTM.JK", "PGAS.JK",
    "ASII.JK", "JSMR.JK", "CPIN.JK", "KLBF.JK"
]


def fetch_and_save_data(ticker: str, period: str = "1mo"):
    logging.info(f"Memproses {ticker}...")
    try:
        stock = yf.Ticker(ticker)
        df    = stock.history(period=period)

        if df.empty:
            logging.warning(f"Data kosong untuk {ticker}.")
            return

        df.reset_index(inplace=True)
        df['Ticker'] = ticker

        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)

        df.rename(columns={
            'Date':   'date',
            'Open':   'open_price',
            'High':   'high_price',
            'Low':    'low_price',
            'Close':  'close_price',
            'Volume': 'volume',
            'Ticker': 'ticker'
        }, inplace=True)

        cols_to_keep = ['date', 'ticker', 'open_price', 'high_price',
                        'low_price', 'close_price', 'volume']
        df = df[cols_to_keep]

        # ── GHOST ROW FIX ──
        # yfinance menyisipkan baris kosong (close=0/NaN) pada malam hari
        # untuk persiapan sesi perdagangan berikutnya.
        # Filter SEBELUM disimpan ke DB.
        before = len(df)
        df = df[
            (df["close_price"] > 0) &
            (df["close_price"].notna()) &
            (df["open_price"]  > 0) &
            (df["high_price"]  > 0) &
            (df["low_price"]   > 0) &
            (df["volume"]      > 0)
        ].copy()

        removed = before - len(df)
        if removed > 0:
            logging.warning(
                f"[ingestion] {ticker}: {removed} ghost row dihapus "
                f"(close=0 atau NaN). Sisa: {len(df)} baris valid."
            )
        if df.empty:
            logging.warning(f"[ingestion] {ticker}: semua baris tidak valid. Skip.")
            return
        # ── END GHOST ROW FIX ──

        db = SessionLocal()
        try:
            # Ambil tanggal yang sudah ada di DB
            existing_dates = set(
                row.date for row in
                db.query(StockHistory.date)
                .filter(StockHistory.ticker == ticker)
                .all()
            )

            # Hanya insert baris yang belum ada
            new_records = [
                r for r in df.to_dict(orient='records')
                if pd.Timestamp(r['date']) not in existing_dates
            ]

            if new_records:
                db.bulk_save_objects([StockHistory(**r) for r in new_records])
                db.commit()
                logging.info(f"✅ {ticker}: {len(new_records)} baris baru disimpan.")
            else:
                logging.info(f"⏭️ {ticker}: tidak ada data baru.")
        except Exception as db_err:
            db.rollback()
            logging.error(f"DB error {ticker}: {db_err}")
        finally:
            db.close()

    except Exception as e:
        logging.error(f"Error fetch {ticker}: {e}")


if __name__ == "__main__":
    for ticker in IHSG_TICKERS:
        fetch_and_save_data(ticker)