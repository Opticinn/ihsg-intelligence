import os
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.core.database import engine, Base
from app.core.limiter import limiter
from app.api import retrain

# ── Models ──
from app.models import stock, user, prediction, conversation

# ── Routers ──
from app.api import stocks, auth, ml, rag, websocket

# Buat semua tabel di DB saat startup
Base.metadata.create_all(bind=engine)

# CUKUP SATU DEKLARASI APP SAJA
app = FastAPI(
    title="IHSG Intelligence API",
    description=(
        "Platform AI untuk analisis saham IHSG — 100% Free Stack\n\n"
        "**Phase 1:** Data ✅ | **Phase 2:** ML ✅ | **Phase 3:** RAG ✅ | **Phase 4:** Real-time ✅"
    ),
    version="4.0.0",
)

# ── Rate Limiting ──
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ── Endpoints Khusus UI (Radar) ──
@app.get("/ui", response_class=HTMLResponse, tags=["UI"])
async def get_ui():
    """Menampilkan Radar Saham Real-time"""
    # Mencoba mencari file test_ws.html di root directory
    file_path = "test_ws.html"
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            return f.read()
    return HTMLResponse(content="<h1>File test_ws.html tidak ditemukan di root!</h1>", status_code=404)

# ── Include Semua Routers ──
app.include_router(auth.router)
app.include_router(stocks.router)
app.include_router(ml.router)
app.include_router(rag.router)
app.include_router(websocket.router)
app.include_router(retrain.router)

@app.get("/chat", response_class=HTMLResponse, tags=["UI"])
async def get_chat_ui():
    """Halaman Antarmuka Chat RAG"""
    file_path = "test_chat.html"
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    return HTMLResponse(content="<h1>File test_chat.html belum dibuat!</h1>", status_code=404)

@app.get("/", tags=["Root"])
def read_root():
    return {
        "status":  "online",
        "version": "4.0.0 — Phase 4 WebSocket Active",
        "ui_radar": "/ui",
        "docs":    "/docs"
    }