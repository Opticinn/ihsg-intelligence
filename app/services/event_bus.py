"""
app/services/event_bus.py
=========================
Menggunakan Redis Pub/Sub sebagai pengganti Kafka.
Juga berfungsi sebagai pintu gerbang notifikasi Telegram.
"""

import os
import json
import ssl
import redis.asyncio as redis
import httpx  # Library asinkron bawaan FastAPI untuk request HTTP
from dotenv import load_dotenv

load_dotenv()

# Konfigurasi Telegram
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
ALERT_THRESHOLD = 0.38 # Ambang batas 40% sesuai kesepakatan

# Konfigurasi Redis
RAW_REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
CLEAN_REDIS_URL = RAW_REDIS_URL.split("?")[0]

if CLEAN_REDIS_URL.startswith("rediss://"):
    redis_client = redis.from_url(CLEAN_REDIS_URL, decode_responses=True, ssl_cert_reqs=ssl.CERT_NONE)
else:
    redis_client = redis.from_url(CLEAN_REDIS_URL, decode_responses=True)

async def send_telegram_alert(ticker: str, signal: str, confidence: float):
    """Fungsi rahasia untuk menembak pesan ke Telegram"""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ Telegram token/chat ID belum diatur di .env!")
        return

    # Emoji dinamis agar keren
    icon = "🟢" if signal == "BUY" else "🔴" if signal == "SELL" else "🟡"
    conf_pct = confidence * 100

    pesan = (
        f"🚨 **SINYAL ALERT IHSG** 🚨\n\n"
        f"Saham: **{ticker}**\n"
        f"Rekomendasi: {icon} **{signal}**\n"
        f"Confidence: **{conf_pct:.1f}%**\n\n"
        f"_(Sistem mendeteksi confidence melampaui batas {ALERT_THRESHOLD*100}%!)_"
    )

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": pesan,
        "parse_mode": "Markdown"
    }

    # Kirim ke Telegram tanpa memblokir proses FastAPI
    async with httpx.AsyncClient() as client:
        try:
            await client.post(url, json=payload)
            print(f"📱 [Telegram] Alert {ticker} terkirim ke HP!")
        except Exception as e:
            print(f"⚠️ [Telegram] Gagal mengirim pesan: {e}")


async def publish_signal(ticker: str, signal: str, confidence: float):
    """
    Meneriakkan hasil ML ke channel 'ihsg_signals' (Radar WebSocket).
    Jika confidence >= 40%, kirim juga alert ke HP via Telegram.
    """
    payload = {
        "event_type": "ml_signal",
        "ticker": ticker,
        "signal": signal,
        "confidence": confidence
    }
    
    # 1. Siarkan ke Radar Browser (selalu dikirim)
    await redis_client.publish("ihsg_signals", json.dumps(payload))
    print(f"📡 [Event Bus] Sinyal {ticker} ({signal}) disiarkan ke udara!")

    # 2. Filter Ketat untuk Telegram
    # Hanya kirim jika sinyalnya kuat dan bukan HOLD (HOLD kurang menarik untuk di-alert)
    if confidence >= ALERT_THRESHOLD and signal in ["BUY", "SELL"]:
        await send_telegram_alert(ticker, signal, confidence)