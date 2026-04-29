"""
clean_ghost_rows.py
====================
Script sekali jalan — hapus ghost row yang sudah terlanjur masuk DB.

Jalankan dari root project:
    python clean_ghost_rows.py
"""

import logging
from app.core.database import SessionLocal
from app.models.stock import StockHistory

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def clean_ghost_rows():
    db = SessionLocal()
    try:
        ghost_rows = db.query(StockHistory).filter(
            (StockHistory.close_price <= 0) |
            (StockHistory.close_price == None) |
            (StockHistory.open_price  <= 0) |
            (StockHistory.volume      <= 0)
        ).all()

        if not ghost_rows:
            logger.info("✅ Tidak ada ghost row di database. Data sudah bersih!")
            return 0

        logger.info(f"🔍 Ditemukan {len(ghost_rows)} ghost row:")
        for row in ghost_rows:
            logger.info(
                f"   ID={row.id} | {row.ticker} | "
                f"date={row.date} | close={row.close_price} | vol={row.volume}"
            )

        for row in ghost_rows:
            db.delete(row)
        db.commit()

        logger.info(f"✅ {len(ghost_rows)} ghost row berhasil dihapus.")
        return len(ghost_rows)

    except Exception as e:
        db.rollback()
        logger.error(f"❌ Error: {e}")
        return -1
    finally:
        db.close()


if __name__ == "__main__":
    clean_ghost_rows()