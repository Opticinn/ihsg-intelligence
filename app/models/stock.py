from sqlalchemy import Column, Integer, String, Float, DateTime
from app.core.database import Base

class StockHistory(Base):
    __tablename__ = "stock_history" # Nama tabel di dalam PostgreSQL nanti

    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String(10), index=True, nullable=False)
    date = Column(DateTime, index=True, nullable=False)
    open_price = Column(Float)
    high_price = Column(Float)
    low_price = Column(Float)
    close_price = Column(Float)
    volume = Column(Integer)