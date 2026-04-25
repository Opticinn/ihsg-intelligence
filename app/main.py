from fastapi import FastAPI
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.core.database import engine, Base
from app.core.limiter import limiter # <-- Import limiter kita
from app.models import stock, user 
from app.api import stocks, auth 

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="IHSG Intelligence API",
    description="REST API untuk platform data saham dan MLOps (100% Free Stack)",
    version="1.0.0"
)

# --- KONFIGURASI RATE LIMITING ---
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
# ---------------------------------

app.include_router(auth.router)   
app.include_router(stocks.router)

@app.get("/")
def read_root():
    return {
        "status": "online",
        "message": "Sistem IHSG Intelligence aktif! 🚀",
        "docs_url": "/docs"
    }