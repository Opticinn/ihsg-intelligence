import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from dotenv import load_dotenv

load_dotenv()

# Di production, kita ambil dari file .env. 
# Untuk sekarang, kita hardcode sesuai docker-compose.yml yang kamu buat sebelumnya.
DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    "postgresql://user_admin:password_rahasia@127.0.0.1:5433/ihsg_db"
)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()