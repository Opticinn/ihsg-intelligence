"""
app/models/conversation.py
===========================
Model SQLAlchemy untuk menyimpan riwayat percakapan RAG.
Mendukung multi-turn conversation (ingat konteks sebelumnya).
"""

from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

from app.core.database import Base


class Conversation(Base):
    """
    Satu sesi percakapan.
    Satu user bisa punya banyak conversation (session terpisah).
    """
    __tablename__ = "conversations"

    id         = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(64), unique=True, index=True, nullable=False)
    ticker     = Column(String(10), nullable=True)   # konteks ticker (opsional)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relasi ke messages
    messages   = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Conversation session={self.session_id} ticker={self.ticker}>"


class Message(Base):
    """
    Satu pesan dalam conversation (user atau assistant).
    """
    __tablename__ = "messages"

    id              = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"), nullable=False)
    role            = Column(String(10), nullable=False)   # "user" | "assistant"
    content         = Column(Text, nullable=False)
    sources         = Column(Text, nullable=True)          # JSON: list sumber dokumen RAG
    created_at      = Column(DateTime(timezone=True), server_default=func.now())

    conversation    = relationship("Conversation", back_populates="messages")

    def __repr__(self):
        return f"<Message role={self.role} conv={self.conversation_id}>"
