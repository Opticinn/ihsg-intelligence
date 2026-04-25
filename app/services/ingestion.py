import yfinance as yf
import pandas as pd
import logging

# Import komponen database yang sudah kita buat
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
    """Mengambil data dari yfinance dan menyimpannya ke PostgreSQL"""
    logging.info(f"Memproses {ticker}...")
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        
        if df.empty:
            logging.warning(f"Data kosong untuk {ticker}.")
            return
            
        df.reset_index(inplace=True)
        df['Ticker'] = ticker
        
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
            
        # Sesuaikan nama kolom DataFrame dengan blueprint di app/models/stock.py
        df.rename(columns={
            'Date': 'date', 
            'Open': 'open_price',
            'High': 'high_price', 
            'Low': 'low_price',
            'Close': 'close_price', 
            'Volume': 'volume',
            'Ticker': 'ticker'
        }, inplace=True)
        
        # Buang kolom yang tidak kita perlukan di database saat ini
        cols_to_keep = ['date', 'ticker', 'open_price', 'high_price', 'low_price', 'close_price', 'volume']
        df = df[cols_to_keep]

        # Buka sesi database
        db = SessionLocal()
        try:
            # Opsional & Praktis untuk tahap development: 
            # Hapus data lama untuk ticker ini agar tidak menumpuk/duplikat saat ditest berkali-kali
            db.query(StockHistory).filter(StockHistory.ticker == ticker).delete()
            
            # Ubah DataFrame menjadi list of dictionaries, lalu masukkan ke database
            records = df.to_dict(orient='records')
            db_objects = [StockHistory(**record) for record in records]
            
            db.bulk_save_objects(db_objects)
            db.commit() # Simpan permanen!
            
            logging.info(f"✅ Berhasil menyimpan {len(records)} baris data {ticker} ke Database.")
        except Exception as db_err:
            db.rollback() # Batalkan transaksi jika ada error
            logging.error(f"❌ Error database saat menyimpan {ticker}: {str(db_err)}")
        finally:
            db.close() # Selalu tutup sesi database
            
    except Exception as e:
        logging.error(f"❌ Gagal mengambil data {ticker}: {str(e)}")

if __name__ == "__main__":
    logging.info("=== Memulai Pipeline Ingestion Database ===")
    
    # Kita eksekusi 5 ticker pertama dulu sebagai test
    test_tickers = IHSG_TICKERS[:5]
    
    for t in test_tickers:
        fetch_and_save_data(t, period="1mo")
        
    logging.info("=== Pipeline Selesai ===")