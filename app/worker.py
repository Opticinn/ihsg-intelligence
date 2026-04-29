"""
app/worker.py
=============
Celery Worker menggunakan Redis untuk antrean tugas asinkron.
Dijalankan via terminal terpisah (menggunakan gevent untuk Windows).
"""

import os
from celery import Celery
from dotenv import load_dotenv

load_dotenv()

# Ambil URL Redis dari .env (Otomatis membaca ?ssl_cert_reqs=CERT_NONE jika ada)
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Inisialisasi Celery murni dengan Redis (Tanpa Kafka/RabbitMQ)
celery_app = Celery(
    "ihsg_worker",
    broker=REDIS_URL,
    backend=REDIS_URL,
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="Asia/Jakarta",
    enable_utc=True,
)

# ─── DAFTAR TUGAS BERAT (BACKGROUND TASKS) ───

@celery_app.task(bind=True, max_retries=3)
def task_train_model(self):
    """
    Melatih ulang model XGBoost di background.
    Tidak akan membuat API FastAPI menjadi lambat/hang.
    """
    try:
        from app.services.trainer import run_training
        result = run_training()
        return {"status": "success", "result": result}
    except Exception as e:
        self.retry(countdown=60, exc=e)