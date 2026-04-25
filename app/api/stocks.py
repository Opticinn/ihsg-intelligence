from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.models.stock import StockHistory

# Import fungsi satpam dan cetakan user
from app.api.auth import get_current_user
from app.models.user import User

from app.core.limiter import limiter

router = APIRouter(prefix="/stocks", tags=["Stocks"])

@router.get("/{ticker}/history")
@limiter.limit("5/minute")  # <-- BATASAN: 5 request per menit
def get_stock_history(
    request: Request, # <-- WAJIB ADA agar IP pengguna bisa dibaca oleh limiter
    ticker: str, 
    limit: int = 30, 
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user) 
):
    """
    Mengambil data historis harga saham berdasarkan Ticker.
    [🔒 TERKUNCI: Membutuhkan Token JWT valid]
    [⏱️ DIBATASI: 5 Request / Menit]
    """
    stocks = db.query(StockHistory)\
               .filter(StockHistory.ticker == ticker.upper())\
               .order_by(StockHistory.date.desc())\
               .limit(limit)\
               .all()
    
    if not stocks:
        raise HTTPException(status_code=404, detail=f"Data untuk ticker {ticker} tidak ditemukan.")
        
    return {
        "requested_by": current_user.username, 
        "ticker": ticker.upper(), 
        "data": stocks
    }