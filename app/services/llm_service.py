"""
app/services/llm_service.py
===========================
Layanan untuk menggabungkan data ML dengan AI Generatif (Llama 3.2).
"""

from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from app.services.predictor import predict_ticker

# 1. Inisialisasi Llama 3.2
# Tambahkan temperature=0.3 agar jawaban AI lebih logis dan tidak terlalu berhalusinasi
llm = OllamaLLM(model="llama3.2", temperature=0.3)

# 2. Skenario Prompt (Diperkaya dengan "Cheat Sheet" teori)
prompt_template = PromptTemplate.from_template(
    """Kamu adalah analis saham profesional dari Indonesia yang cerdas. 
    Tugasmu adalah menganalisis saham berdasarkan data teknikal dan prediksi Machine Learning (XGBoost) di bawah ini.

    [DATA SAHAM]
    - Ticker: {ticker}
    - Harga Terakhir: Rp {close_price}
    - Sinyal Machine Learning: {signal} (Tingkat Keyakinan: {confidence})
    - RSI (Momentum): {rsi}
    - MACD Histogram (Tren): {macd}

    [PANDUAN TEORI UNTUKMU]
    - Sinyal XGBoost BUY berarti algoritma memprediksi harga akan naik 3 hari ke depan.
    - RSI < 30 berarti saham sedang Oversold (murah/jenuh jual). RSI > 70 berarti Overbought (mahal). RSI 30-70 adalah netral.
    - MACD Histogram positif berarti momentum sedang naik (Bullish). MACD negatif berarti momentum turun (Bearish).

    Instruksi:
    1. Berikan analisis 2 paragraf saja mengenai perpaduan angka-angka di atas.
    2. Berikan kesimpulan akhir yang tegas: (BELI, JUAL, atau PANTAU DULU).
    
    Jawablah dengan gaya bahasa yang profesional namun mudah dipahami.
    """
)

def generate_stock_analysis(ticker: str) -> str:
    """Mengambil data prediksi ML lalu menyuruh Llama membuat analisis teks."""
    try:
        # Panggil fungsi XGBoost (Ini akan mengambil data dari Redis jika ada!)
        ml_data = predict_ticker(ticker)
        features = ml_data["features"]

        # Perbaikan ambil harga (Bisa close_price atau close)
        harga_terakhir = features.get("close_price", features.get("close", 0))

        # Gabungkan data ke dalam templat
        prompt_text = prompt_template.format(
            ticker=ml_data["ticker"],
            signal=ml_data["signal"],
            confidence=ml_data["confidence_pct"],
            close_price=harga_terakhir,
            rsi=features.get("rsi", 0),
            macd=features.get("macd_hist", 0)
        )

        # Minta Llama menganalisis
        analisis = llm.invoke(prompt_text)

        return analisis
    except Exception as e:
        return f"Maaf, sistem AI gagal menganalisis saham {ticker}. Error: {str(e)}"