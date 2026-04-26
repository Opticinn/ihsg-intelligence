import pandas as pd
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv
from evidently.report import Report
from evidently.metric_preset import DataQualityPreset
import warnings

# Mengabaikan warning agar terminal tetap bersih
warnings.filterwarnings('ignore')

# Load koneksi database dari .env
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)

def run_data_quality_check(ticker="BBCA.JK"):
    print(f"🔍 Mengambil data historis {ticker} dari Supabase...")
    
    # Menarik data dari database menggunakan pandas
    query = f"SELECT * FROM stock_history WHERE ticker = '{ticker}' ORDER BY date ASC"
    df = pd.read_sql(query, engine)
    
    # Pastikan data cukup untuk dianalisis
    if len(df) < 14:
        print(f"⚠️ Data {ticker} kurang dari 30 hari. Evidently butuh lebih banyak data untuk perbandingan.")
        return

    # Skenario: Kita jadikan 7 hari terakhir sebagai "Current Data" (data yang mau dicek)
    # Dan sisa data sebelumnya sebagai "Reference Data" (data normal/patokan)
    reference_data = df.iloc[:-3]
    current_data = df.iloc[-3:]

    print("📊 Menganalisis kualitas data dengan Evidently AI...")
    
    # Memanggil Preset Kualitas Data dari Evidently
    data_quality_report = Report(metrics=[
        DataQualityPreset(),
    ])
    
    # Menjalankan analisis
    data_quality_report.run(reference_data=reference_data, current_data=current_data)
    
    # Menyimpan laporan ke dalam bentuk file HTML
    report_filename = f"report_{ticker}_data_quality.html"
    data_quality_report.save_html(report_filename)
    
    print(f"✅ Analisis Selesai! Laporan berhasil disimpan sebagai '{report_filename}'")
    print("👉 Silakan buka file tersebut di browser (Chrome/Edge/Safari) untuk melihat hasilnya.")

if __name__ == "__main__":
    # Kamu bisa mengganti ticker ini dengan BBRI.JK, GOTO.JK, dll sesuai data yang ada
    run_data_quality_check("BBCA.JK")