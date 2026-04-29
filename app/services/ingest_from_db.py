"""
app/services/ingest_from_db.py
================================
Build dokumen ChromaDB dari data yang sudah ada di PostgreSQL.
Tidak perlu internet, tidak perlu scraping.

Jalankan:
  python -m app.services.ingest_from_db
"""

import logging
import sys

from app.core.database import SessionLocal
from app.models.stock import StockHistory
from app.models.prediction import Prediction
from app.services.document_loader import Document, chunk_documents
from app.services.vector_store import ingest_documents
from app.services.ingestion import IHSG_TICKERS

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

COMPANY_NAMES = {
    "BBCA": "Bank Central Asia",   "BBRI": "Bank Rakyat Indonesia",
    "BMRI": "Bank Mandiri",        "BBNI": "Bank Negara Indonesia",
    "ARTO": "Bank Jago",           "TLKM": "Telkom Indonesia",
    "GOTO": "GoTo Gojek Tokopedia","BUKA": "Bukalapak",
    "ISAT": "Indosat Ooredoo",     "UNVR": "Unilever Indonesia",
    "ICBP": "Indofood CBP",        "INDF": "Indofood Sukses Makmur",
    "AMRT": "Alfamart",            "ADRO": "Adaro Energy",
    "ITMG": "Indo Tambangraya",    "PTBA": "Bukit Asam",
    "ANTM": "Antam",               "PGAS": "Perusahaan Gas Negara",
    "ASII": "Astra International", "JSMR": "Jasa Marga",
    "CPIN": "Charoen Pokphand",    "KLBF": "Kalbe Farma",
}

SECTOR_INFO = {
    "BBCA": ("Perbankan", "Bank swasta terbesar Indonesia, dikenal stabil dan likuid tinggi."),
    "BBRI": ("Perbankan", "Bank BUMN terbesar, fokus kredit mikro dan segmen pedesaan."),
    "BMRI": ("Perbankan", "Bank BUMN dengan aset terbesar, kuat di segmen korporasi."),
    "BBNI": ("Perbankan", "Bank BUMN fokus pembiayaan infrastruktur dan ekspor-impor."),
    "ARTO": ("Perbankan Digital", "Bank digital milik GoTo, valuasi growth, sensitif sentimen tech."),
    "TLKM": ("Telekomunikasi", "BUMN telekomunikasi, pendapatan defensif dari langganan bulanan."),
    "GOTO": ("Teknologi", "Super-app Indonesia, model bisnis berbasis ekosistem digital."),
    "BUKA": ("E-commerce", "Marketplace online, bersaing ketat di segmen UMKM."),
    "ISAT": ("Telekomunikasi", "Hasil merger Indosat-Hutchison, fokus ekspansi 5G."),
    "UNVR": ("Consumer Goods", "Produk kebutuhan sehari-hari, defensif tapi sensitif inflasi bahan baku."),
    "ICBP": ("Consumer Goods", "Produk makanan olahan, dikenal lewat merek Indomie."),
    "INDF": ("Consumer Goods", "Induk ICBP, bisnis terintegrasi dari hulu ke hilir."),
    "AMRT": ("Ritel", "Operator Alfamart, pertumbuhan dari ekspansi gerai minimarket."),
    "ADRO": ("Batu Bara", "Produsen batu bara termal, sangat sensitif harga komoditas global."),
    "ITMG": ("Batu Bara", "Batu bara kalori tinggi, dividend yield tinggi secara historis."),
    "PTBA": ("Batu Bara", "BUMN batu bara, pasokan utama ke PLN."),
    "ANTM": ("Tambang Mineral", "Nikel dan emas, prospek positif dari demand baterai EV."),
    "PGAS": ("Energi Gas", "Distributor gas bumi, harga diregulasi pemerintah."),
    "ASII": ("Konglomerasi", "Otomotif dan industri, barometer ekonomi Indonesia."),
    "JSMR": ("Infrastruktur", "Operator jalan tol, pendapatan stabil dari volume lalu lintas."),
    "CPIN": ("Peternakan", "Pakan ternak dan ayam, sensitif harga jagung."),
    "KLBF": ("Farmasi", "Produk kesehatan, diuntungkan penetrasi JKN/BPJS."),
}


def build_price_doc(ticker, db):
    ticker_jk = f"{ticker}.JK"
    rows = (
        db.query(StockHistory)
        .filter(StockHistory.ticker == ticker_jk, StockHistory.close_price > 0)
        .order_by(StockHistory.date.desc())
        .limit(30).all()
    )
    if not rows:
        return None

    latest = rows[0]
    oldest = rows[-1]
    high   = max(r.high_price for r in rows)
    low    = min(r.low_price  for r in rows)
    avg_vol= sum(r.volume for r in rows if r.volume) / len(rows)
    chg30  = (latest.close_price - oldest.close_price) / oldest.close_price * 100
    trend  = "naik" if chg30 > 0 else "turun"
    company= COMPANY_NAMES.get(ticker, ticker)

    text = f"""Ringkasan Harga Saham {ticker} ({company})
Periode: {oldest.date.strftime('%d %b %Y')} sampai {latest.date.strftime('%d %b %Y')}

Harga terkini: Rp {latest.close_price:,.0f}
Perubahan 30 hari: {chg30:+.2f}% ({trend})
Harga tertinggi 30 hari: Rp {high:,.0f}
Harga terendah 30 hari: Rp {low:,.0f}
Rata-rata volume harian: {avg_vol:,.0f} lot

Saham {ticker} dalam 30 hari terakhir bergerak {trend} sebesar {abs(chg30):.1f}%. Rentang harga antara Rp {low:,.0f} hingga Rp {high:,.0f}."""

    return Document(
        content=text, source=f"db://stock_history/{ticker_jk}",
        ticker=ticker, doc_type="price_summary",
        title=f"Harga {ticker} 30 Hari Terakhir",
        date=latest.date.strftime("%Y-%m-%d"),
    )


def build_prediction_doc(ticker, db):
    import json
    ticker_jk = f"{ticker}.JK"
    pred = (
        db.query(Prediction)
        .filter(Prediction.ticker == ticker_jk)
        .order_by(Prediction.created_at.desc())
        .first()
    )
    if not pred:
        return None

    try:
        f = json.loads(pred.features_snapshot) if pred.features_snapshot else {}
    except Exception:
        f = {}

    rsi   = f.get("rsi", 50)
    macd  = f.get("macd_hist", 0)
    vol   = f.get("volume_ratio", 1)
    close = f.get("close_price", 0)
    ret1d = f.get("return_1d", 0)
    company = COMPANY_NAMES.get(ticker, ticker)

    rsi_desc  = "oversold — harga murah, potensi rebound" if rsi < 30 else "overbought — harga mahal, waspadai koreksi" if rsi > 70 else "netral"
    macd_desc = "momentum menguat (bullish)" if macd > 0 else "momentum melemah (bearish)"
    vol_desc  = f"volume spike {vol:.1f}x" if vol > 2 else f"volume sepi {vol:.1f}x" if vol < 0.5 else f"volume normal {vol:.1f}x"
    signal_map = {
        "BUY":  "AI memprediksi harga berpotensi NAIK dalam 1-3 hari",
        "SELL": "AI memprediksi harga berpotensi TURUN dalam 1-3 hari",
        "HOLD": "AI memprediksi harga relatif flat",
    }

    text = f"""Analisis Teknikal dan Sinyal AI: {ticker} ({company})
Tanggal: {pred.created_at.strftime('%d %B %Y') if pred.created_at else 'N/A'}

Harga terakhir: Rp {close:,.0f}
Perubahan 1 hari: {ret1d*100:+.2f}%

SINYAL AI: {pred.signal} — {signal_map.get(pred.signal, pred.signal)}
Keyakinan model: {pred.confidence:.0%}

Kondisi teknikal:
- RSI {rsi:.1f}: {rsi_desc}
- MACD: {macd_desc}
- Volume: {vol_desc}"""

    return Document(
        content=text, source=f"db://predictions/{ticker_jk}",
        ticker=ticker, doc_type="ml_analysis",
        title=f"Sinyal AI {ticker} — {pred.signal} ({pred.confidence:.0%})",
        date=pred.created_at.strftime("%Y-%m-%d") if pred.created_at else "",
    )


def build_sector_doc(ticker):
    company = COMPANY_NAMES.get(ticker, ticker)
    sector, desc = SECTOR_INFO.get(ticker, ("Tidak diketahui", ""))
    text = f"""Profil Perusahaan: {ticker} — {company}
Sektor: {sector}

{desc}

Untuk menganalisis {ticker}, perhatikan laporan keuangan kuartalan, berita korporasi IDX, dan kondisi makro seperti suku bunga BI dan kurs rupiah."""

    return Document(
        content=text, source=f"static://sector/{ticker}",
        ticker=ticker, doc_type="sector_context",
        title=f"Profil {ticker} — {company}", date="",
    )


def ingest_from_db(tickers=None):
    if tickers is None:
        tickers = [t.replace(".JK", "") for t in IHSG_TICKERS]

    db   = SessionLocal()
    docs = []
    stats = {"price": 0, "ml": 0, "sector": 0, "skipped": 0}

    try:
        for ticker in tickers:
            doc = build_price_doc(ticker, db)
            if doc:
                docs.append(doc)
                stats["price"] += 1
            else:
                stats["skipped"] += 1

            doc = build_prediction_doc(ticker, db)
            if doc:
                docs.append(doc)
                stats["ml"] += 1

            docs.append(build_sector_doc(ticker))
            stats["sector"] += 1
    finally:
        db.close()

    logger.info(f"[ingest_db] {len(docs)} dokumen siap di-ingest")
    n_chunks = ingest_documents(docs)

    return {
        "status": "success",
        "total_docs": len(docs),
        "chunks_ingested": n_chunks,
        "breakdown": stats,
    }


if __name__ == "__main__":
    tickers = [a.replace(".JK","").upper() for a in sys.argv[1:]] or None
    result  = ingest_from_db(tickers)
    print(f"\nDokumen: {result['total_docs']} | Chunks: {result['chunks_ingested']}")
    print(f"Price: {result['breakdown']['price']} | ML: {result['breakdown']['ml']} | Sector: {result['breakdown']['sector']}")