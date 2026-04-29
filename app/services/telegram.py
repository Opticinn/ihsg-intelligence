"""
app/services/telegram.py
=========================
Telegram notification service untuk IHSG Intelligence Platform.

Notifikasi yang dikirim:
  1. Sinyal trading (confidence > 75%)
  2. Monitoring alert (drift, accuracy drop)
  3. Pipeline status (ingestion, retrain)
  4. Ghost row alert
"""

import logging
import os
import requests
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")
BASE_URL  = f"https://api.telegram.org/bot{BOT_TOKEN}"

CONFIDENCE_THRESHOLD = 0.75  # hanya kirim kalau confidence > 75%


# ─────────────────────────────────────────────
# CORE SENDER
# ─────────────────────────────────────────────

def send_message(text: str, parse_mode: str = "Markdown") -> bool:
    """Kirim pesan ke Telegram."""
    if not BOT_TOKEN or not CHAT_ID:
        logger.warning("[telegram] BOT_TOKEN atau CHAT_ID belum dikonfigurasi")
        return False

    try:
        resp = requests.post(
            f"{BASE_URL}/sendMessage",
            json={
                "chat_id":    CHAT_ID,
                "text":       text,
                "parse_mode": parse_mode,
            },
            timeout=10,
        )
        if resp.status_code == 200:
            logger.info(f"[telegram] ✅ Pesan terkirim")
            return True
        else:
            logger.error(f"[telegram] ❌ Error {resp.status_code}: {resp.text}")
            return False
    except Exception as e:
        logger.error(f"[telegram] ❌ Exception: {e}")
        return False


# ─────────────────────────────────────────────
# 1. SINYAL TRADING
# ─────────────────────────────────────────────

def notify_signal(ticker: str, signal: str, confidence: float, close_price: float = 0):
    """
    Kirim notifikasi sinyal trading.
    Hanya kirim kalau confidence > CONFIDENCE_THRESHOLD.
    """
    if confidence < CONFIDENCE_THRESHOLD:
        logger.info(f"[telegram] Skip {ticker} — confidence {confidence:.1%} < {CONFIDENCE_THRESHOLD:.0%}")
        return

    emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}.get(signal, "⚪")
    label = {"BUY": "BELI", "SELL": "JUAL", "HOLD": "TAHAN"}.get(signal, signal)
    harga = f"Rp {close_price:,.0f}" if close_price > 0 else "N/A"
    now   = datetime.now().strftime("%d %b %Y %H:%M WIB")

    msg = (
        f"{emoji} *SINYAL {label} — {ticker}*\n\n"
        f"💰 Harga: `{harga}`\n"
        f"🎯 Confidence: `{confidence:.1%}`\n"
        f"📅 Waktu: `{now}`\n\n"
        f"_⚠️ Ini bukan saran investasi. Selalu lakukan riset sendiri._"
    )
    send_message(msg)


# ─────────────────────────────────────────────
# 2. MONITORING ALERTS
# ─────────────────────────────────────────────

def notify_monitoring_summary(summary: dict):
    """Kirim ringkasan hasil monitoring ke Telegram."""
    drift = summary.get("data_drift", {})
    model = summary.get("model_performance", {})
    dq    = summary.get("data_quality", {})
    now   = datetime.now().strftime("%d %b %Y %H:%M WIB")

    drift_share    = drift.get("drift_share", 0)
    model_acc      = model.get("current_acc", 0)
    acc_drift      = model.get("acc_drift", 0)
    should_retrain = drift.get("should_retrain", False) or model.get("should_retrain", False)
    tickers_ok     = dq.get("tickers_ok", 0)
    tickers_total  = dq.get("tickers_total", 0)

    drift_icon = "⚠️" if drift_share > 0.3 else "✅"
    model_icon = "⚠️" if abs(acc_drift) > 0.10 else "✅"
    dq_icon    = "⚠️" if tickers_ok < tickers_total else "✅"

    retrain_line = (
        "\n🔁 *ACTION REQUIRED: Retrain disarankan!*"
        if should_retrain else
        "\n✅ Model dalam kondisi baik."
    )

    msg = (
        f"📊 *IHSG Intelligence — Monitoring Report*\n"
        f"📅 `{now}`\n\n"
        f"{dq_icon} Data Quality: `{tickers_ok}/{tickers_total}` ticker OK\n"
        f"{drift_icon} Data Drift: `{drift_share:.1%}` fitur berubah\n"
        f"{model_icon} Model Accuracy: `{model_acc:.4f}` (drift: `{acc_drift:+.4f}`)\n"
        f"{retrain_line}"
    )
    send_message(msg)


def notify_drift_alert(drift_share: float, n_drifted: int):
    """Alert khusus kalau drift sangat tinggi (> 50%)."""
    if drift_share < 0.5:
        return

    msg = (
        f"🚨 *DRIFT ALERT — IHSG Intelligence*\n\n"
        f"Data drift terdeteksi sangat tinggi!\n"
        f"📈 Drift: `{drift_share:.1%}` ({n_drifted} fitur)\n\n"
        f"_Sistem akan mencoba retrain otomatis._"
    )
    send_message(msg)


def notify_accuracy_drop(current_acc: float, baseline_acc: float):
    """Alert kalau accuracy model turun signifikan."""
    drop = baseline_acc - current_acc
    if drop < 0.38:
        return

    msg = (
        f"🔴 *MODEL DEGRADED — IHSG Intelligence*\n\n"
        f"Accuracy model turun signifikan!\n"
        f"📉 Baseline: `{baseline_acc:.4f}`\n"
        f"📉 Current: `{current_acc:.4f}`\n"
        f"📉 Drop: `{drop:.4f}`\n\n"
        f"_Retrain otomatis dijadwalkan._"
    )
    send_message(msg)


# ─────────────────────────────────────────────
# 3. PIPELINE STATUS
# ─────────────────────────────────────────────

def notify_ingestion_done(tickers_ok: int, tickers_total: int, rows_inserted: int = 0):
    """Notifikasi setelah ingestion harian selesai."""
    icon = "✅" if tickers_ok == tickers_total else "⚠️"
    now  = datetime.now().strftime("%d %b %Y %H:%M WIB")

    msg = (
        f"{icon} *Daily Ingestion Selesai*\n"
        f"📅 `{now}`\n\n"
        f"📥 Ticker: `{tickers_ok}/{tickers_total}` berhasil\n"
        f"🗄️ Rows: `{rows_inserted:,}` baris baru\n"
    )
    send_message(msg)


def notify_retrain_done(train_acc: float, test_acc: float, trigger: str = "scheduled"):
    """Notifikasi setelah model selesai diretrain."""
    icon  = "✅" if test_acc >= 0.38 else "⚠️"
    now   = datetime.now().strftime("%d %b %Y %H:%M WIB")

    msg = (
        f"{icon} *Model Retrain Selesai*\n"
        f"📅 `{now}`\n\n"
        f"🎯 Train Acc: `{train_acc:.4f}`\n"
        f"🎯 Test Acc: `{test_acc:.4f}`\n"
        f"🔁 Trigger: `{trigger}`\n"
    )
    send_message(msg)


def notify_pipeline_error(job: str, error: str):
    """Notifikasi kalau ada error di pipeline."""
    now = datetime.now().strftime("%d %b %Y %H:%M WIB")
    msg = (
        f"❌ *Pipeline Error — IHSG Intelligence*\n"
        f"📅 `{now}`\n\n"
        f"🔧 Job: `{job}`\n"
        f"💥 Error: `{error[:200]}`\n"
    )
    send_message(msg)


# ─────────────────────────────────────────────
# TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing Telegram notification...")
    ok = send_message(
        "🧪 *Test — IHSG Intelligence Platform*\n\n"
        "Telegram notification berhasil dikonfigurasi!\n"
        "Semua alert akan dikirim ke sini. ✅"
    )
    print("Berhasil!" if ok else "Gagal — cek BOT_TOKEN dan CHAT_ID")