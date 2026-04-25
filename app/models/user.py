from sqlalchemy import Column, Integer, String
from app.core.database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    # Sangat Penting: Kita TIDAK PERNAH menyimpan password asli, hanya hasil acakannya (hash)
    hashed_password = Column(String(255), nullable=False)