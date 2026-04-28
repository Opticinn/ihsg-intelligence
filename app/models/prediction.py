from sqlalchemy import Column, Integer, String, Float, DateTime, Text
from sqlalchemy.sql import func
from app.core.database import Base


class Prediction(Base):
    """
    Tabel untuk menyimpan hasil prediksi model ML.
    Setiap kali /predict/{ticker} dipanggil, hasilnya disimpan di sini.
    """
    __tablename__ = "predictions"

    id          = Column(Integer, primary_key=True, index=True)
    ticker      = Column(String(10), index=True, nullable=False)
    signal      = Column(String(10), nullable=False)   # "BUY" | "HOLD" | "SELL"
    confidence  = Column(Float, nullable=False)         # 0.0 – 1.0
    model_version = Column(String(50), nullable=True)  # e.g. "xgb-v1"
    features_snapshot = Column(Text, nullable=True)    # JSON snapshot fitur saat prediksi
    created_at  = Column(DateTime(timezone=True), server_default=func.now())

    def __repr__(self):
        return (
            f"<Prediction ticker={self.ticker} signal={self.signal} "
            f"confidence={self.confidence:.2f} at={self.created_at}>"
        )
