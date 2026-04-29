"""
app/services/llm_service.py
============================
Layanan AI Analyst — menggabungkan data ML dengan Llama 3.2.
Output dirancang untuk orang awam, bukan trader profesional.
"""

import logging
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from app.services.predictor import predict_ticker

logger = logging.getLogger(__name__)

llm = OllamaLLM(model="llama3.2", temperature=0.3)

# ─────────────────────────────────────────────
# PROMPT — dirancang untuk output natural, bukan bocor instruksi
# ─────────────────────────────────────────────

prompt_template = PromptTemplate.from_template(
"""Kamu adalah Andi, asisten investasi saham yang ramah dan bicara seperti teman.
Kamu menjelaskan saham dengan bahasa yang mudah dipahami orang yang belum ahli investasi.

Data saham {ticker} hari ini:
- Harga saat ini: Rp {close_price:,}
- Sinyal AI: {signal} (keyakinan model: {confidence})
- RSI: {rsi:.1f} {rsi_label}
- MACD: {macd_label}

Tulis analisis singkat dalam 3 bagian — gunakan bahasa sehari-hari, hindari jargon teknikal:

1. KONDISI SEKARANG (1-2 kalimat): Jelaskan kondisi saham ini seperti menjelaskan ke teman.
2. APA YANG DIKATAKAN AI (1-2 kalimat): Jelaskan sinyal {signal} dengan kata-kata sederhana, sebutkan tingkat keyakinannya.
3. SARAN (1 kalimat): Berikan saran praktis yang netral. Selalu ingatkan bahwa ini bukan nasihat investasi resmi.

PENTING: Tulis langsung isinya saja. Jangan tulis "1.", "2.", "3." — tulis mengalir seperti orang ngobrol."""
)


def _interpret_rsi(rsi: float) -> str:
    """Terjemahkan nilai RSI ke bahasa manusia."""
    if rsi < 30:
        return "(harga sudah cukup murah, banyak yang mau beli)"
    elif rsi < 50:
        return "(momentum sedang melemah)"
    elif rsi < 70:
        return "(kondisi normal)"
    else:
        return "(harga sudah cukup mahal, banyak yang mau jual)"


def _interpret_macd(macd_hist: float) -> str:
    """Terjemahkan MACD histogram ke bahasa manusia."""
    if macd_hist > 0.5:
        return "menunjukkan tren sedang menguat"
    elif macd_hist > 0:
        return "menunjukkan tren mulai membaik"
    elif macd_hist > -0.5:
        return "menunjukkan tren mulai melemah"
    else:
        return "menunjukkan tren sedang melemah"


def _interpret_signal(signal: str, confidence: float) -> str:
    """Format signal + confidence jadi bahasa manusia."""
    conf_pct = f"{confidence:.0%}"
    if signal == "BUY":
        return f"BUY — AI memprediksi harga berpotensi naik (keyakinan {conf_pct})"
    elif signal == "SELL":
        return f"SELL — AI memprediksi harga berpotensi turun (keyakinan {conf_pct})"
    else:
        return f"HOLD — AI memprediksi harga relatif flat (keyakinan {conf_pct})"


def generate_stock_analysis(ticker: str) -> str:
    """
    Ambil data prediksi ML, lalu minta Llama buat analisis
    yang mudah dipahami orang awam.
    """
    try:
        ml_data  = predict_ticker(ticker)
        features = ml_data["features"]

        close_price = features.get("close_price", 0)
        rsi         = features.get("rsi", 50)
        macd_hist   = features.get("macd_hist", 0)
        signal      = ml_data["signal"]
        confidence  = ml_data["confidence"]

        prompt_text = prompt_template.format(
            ticker      = ml_data["ticker"],
            close_price = int(close_price),
            signal      = _interpret_signal(signal, confidence),
            confidence  = f"{confidence:.0%}",
            rsi         = rsi,
            rsi_label   = _interpret_rsi(rsi),
            macd_label  = _interpret_macd(macd_hist),
        )

        analisis = llm.invoke(prompt_text).strip()

        # Bersihkan kalau ada sisa format aneh
        analisis = analisis.replace("1. KONDISI SEKARANG:", "").strip()
        analisis = analisis.replace("2. APA YANG DIKATAKAN AI:", "\n\n").strip()
        analisis = analisis.replace("3. SARAN:", "\n\n").strip()

        return analisis

    except Exception as e:
        logger.error(f"[llm_service] Error analisis {ticker}: {e}")
        return (
            f"Maaf, sistem AI sedang tidak bisa menganalisis {ticker} saat ini. "
            f"Coba beberapa saat lagi."
        )