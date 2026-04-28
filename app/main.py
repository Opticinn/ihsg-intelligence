from fastapi import FastAPI
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.core.database import engine, Base
from app.core.limiter import limiter

# ── Models (semua harus diimport agar tabel ter-create) ──
from app.models import stock, user
from app.models import prediction       # Phase 2
from app.models import conversation     # Phase 3 🆕

# ── Routers ──
from app.api import stocks, auth
from app.api import ml                  # Phase 2
from app.api import rag                 # Phase 3 🆕

# Buat semua tabel di DB (termasuk conversations & messages)
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="IHSG Intelligence API",
    description=(
        "Platform AI untuk analisis saham IHSG — 100% Free Stack\n\n"
        "**Phase 1:** Data ingestion + JWT Auth ✅\n"
        "**Phase 2:** ML Pipeline + MLflow + Prediksi ✅\n"
        "**Phase 3:** LLM + RAG Chat System ✅\n"
    ),
    version="3.0.0",
)

# ── Rate Limiting ──
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ── Routers ──
app.include_router(auth.router)
app.include_router(stocks.router)
app.include_router(ml.router)
app.include_router(rag.router)          # Phase 3 🆕

@app.get("/", tags=["Root"])
def read_root():
    return {
        "status":  "online",
        "version": "3.0.0 — Phase 3 Active",
        "endpoints": {
            "auth":        ["/auth/register", "/auth/token"],
            "stocks":      "/stocks/{ticker}/history",
            "ml_predict":  "/ml/predict/{ticker}",
            "ml_train":    "POST /ml/train",
            "ml_history":  "/ml/predictions/history",
            "rag_chat":    "POST /rag/chat/ask",
            "rag_ingest":  "POST /rag/ingest/{ticker}",
            "rag_history": "GET /rag/chat/history/{session_id}",
            "rag_stats":   "GET /rag/stats",
            "rag_health":  "GET /rag/health",
            "docs":        "/docs",
        },
    }