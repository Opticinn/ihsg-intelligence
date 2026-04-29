# 🧠 IHSG Intelligence Platform

> **An end-to-end MLOps + RAG system for Indonesian Stock Exchange (IHSG) short-term signal prediction**  
> Built with Zero-Dollar Architecture — no cloud fees, no paid APIs.

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?style=flat-square&logo=fastapi)
![XGBoost](https://img.shields.io/badge/XGBoost-2.x-orange?style=flat-square)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_DB-purple?style=flat-square)
![Llama](https://img.shields.io/badge/Llama_3.2-3B_Local-red?style=flat-square)
![Evidently](https://img.shields.io/badge/Evidently_AI-Monitoring-yellow?style=flat-square)
![Grafana](https://img.shields.io/badge/Grafana-Dashboard-orange?style=flat-square&logo=grafana)
![Telegram](https://img.shields.io/badge/Telegram-Alerts-blue?style=flat-square&logo=telegram)

---

## 📌 Overview

IHSG Intelligence Platform is a production-grade ML system that predicts short-term trading signals (BUY / HOLD / SELL) for 22 high-cap Indonesian stocks, enriched with real-time news analysis via a RAG (Retrieval-Augmented Generation) pipeline.

Users can interact through a natural language chat interface powered by a locally-running LLM (Llama 3.2), which answers stock-related questions using a combination of ML predictions and live data fetched from TradingView Ideas and Google News.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                   IHSG Intelligence Platform                     │
├──────────────┬──────────────┬───────────────┬───────────────────┤
│  Phase 1     │  Phase 2     │   Phase 3     │     Phase 4       │
│  Data & API  │  ML Pipeline │   RAG + LLM   │    Monitoring     │
├──────────────┼──────────────┼───────────────┼───────────────────┤
│ FastAPI      │ XGBoost      │ Llama 3.2     │ Evidently AI      │
│ PostgreSQL   │ MLflow       │ ChromaDB      │ Grafana Cloud     │
│ Supabase     │ Feature Eng  │ TradingView   │ Telegram Bot      │
│ JWT Auth     │ Predictor    │ Google News   │ GitHub Actions    │
│ Redis Cache  │ Celery       │ LangChain     │ Adaptive Retrain  │
└──────────────┴──────────────┴───────────────┴───────────────────┘
```

---

## ✨ Key Features

### 🤖 ML Signal Prediction
- XGBoost multiclass classifier predicting BUY / HOLD / SELL
- 20 engineered features: RSI, MACD, Bollinger Bands, Volume Ratio, lag features
- MLflow experiment tracking with pickle-based model persistence
- Redis caching — predictions cached for 1 hour per ticker
- Signal threshold: return > +0.5% → BUY | < -0.5% → SELL | else → HOLD

### 💬 RAG Chat System
- Natural language Q&A about IHSG stocks in Bahasa Indonesia
- Llama 3.2 3B running 100% locally via Ollama (zero cost)
- ChromaDB vector database for semantic retrieval
- Real-time data sources (priority order):
  - **TradingView Ideas** — Indonesian trader community technical analysis
  - **Google News RSS** — latest news per ticker (free, no API key)
  - **Internal DB** — price summary, ML signals, sector profiles
- Auto-detects ticker from user question
- Auto-triggers predict + ingest on first mention of a new ticker

### 📊 Monitoring & Observability
- **Evidently AI** — unified reports across all 22 tickers:
  - Data Quality (ghost row detection, missing values, outliers)
  - Data Drift (feature distribution shift vs 60-day baseline)
  - Model Performance (accuracy, F1, signal distribution)
- **Grafana Cloud** — 9-panel MLOps dashboard, 5-minute auto-refresh
- **Telegram Bot** — real-time alerts for:
  - Trading signals with confidence > 38%
  - Data drift alerts > 50%
  - Model accuracy degradation
  - Daily pipeline status

### ⚙️ MLOps Pipeline (GitHub Actions)
```
Mon–Fri at 16:30 WIB (09:30 UTC):
  Ingest → Feature Check → Monitor → Retrain (conditional) → Notify
```

Adaptive retraining logic — retrain only when:
- Model accuracy drops > 10% from baseline, OR
- Scheduled day (Mon/Thu) AND drift > 30%, OR
- Drift > 80% AND accuracy starts declining > 3%

---

## 🎯 Supported Tickers (22 Stocks)

| Sector | Tickers |
|--------|---------|
| 🏦 Banking | BBCA, BBRI, BMRI, BBNI, ARTO |
| 📡 Technology & Telco | TLKM, GOTO, BUKA, ISAT |
| 🛒 Consumer & Retail | UNVR, ICBP, INDF, AMRT |
| ⛏️ Mining & Energy | ADRO, ITMG, PTBA, ANTM, PGAS |
| 🏭 Industry & Others | ASII, JSMR, CPIN, KLBF |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| API | FastAPI, Uvicorn, SlowAPI (rate limiting) |
| ML | XGBoost 2.x, scikit-learn, MLflow |
| Feature Engineering | pandas, numpy, ta (technical analysis library) |
| LLM | Llama 3.2 3B via Ollama (100% local) |
| RAG | LangChain, ChromaDB, HuggingFace Embeddings |
| Embeddings | paraphrase-multilingual-MiniLM-L12-v2 |
| Database | PostgreSQL via Supabase (Transaction Pooler port 6543) |
| Cache | Redis |
| Task Queue | Celery + gevent (Windows compatible) |
| Monitoring | Evidently AI 0.4.38, Grafana Cloud |
| Alerting | Telegram Bot API |
| CI/CD | GitHub Actions (5-job adaptive pipeline) |
| Data Sources | yfinance, TradingView scraper, Google News RSS |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11
- Redis (local)
- [Ollama](https://ollama.com) with `llama3.2` model
- Supabase account (PostgreSQL)

### Installation

```bash
# Clone the repository
git clone https://github.com/username/ihsg-intelligence.git
cd ihsg-intelligence

# Install dependencies
pip install -r requirements.txt

# Pull the LLM model
ollama pull llama3.2

# Set up environment variables
cp .env.example .env
# Edit .env with your credentials
```

### Environment Variables

```env
DATABASE_URL=postgresql://postgres.[PROJECT-ID]:[PASSWORD]@[HOST].pooler.supabase.com:6543/postgres
REDIS_URL=redis://localhost:6379/0
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

### Running the System

```bash
# Terminal 1 — FastAPI server
uvicorn app.main:app --reload

# Terminal 2 — Celery worker
celery -A app.worker.celery_app worker --loglevel=info -P gevent

# Terminal 3 — Ollama LLM server
ollama serve
```

### First Time Setup

```bash
# 1. Backfill 1 year of historical data (239 rows per ticker)
python -m app.services.ingestion_historical

# 2. Train the XGBoost model
python -m app.services.trainer

# 3. Populate ChromaDB from internal database
python -m app.services.ingest_from_db

# 4. Run full monitoring pipeline
python -m app.services.monitoring
```

Open `http://localhost:8000/chat` for the chat interface.  
Open `http://localhost:8000/docs` for the Swagger API documentation.

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/ml/predict/{ticker}` | Get trading signal (Redis cached) |
| `GET` | `/ml/predict/batch/all` | Predict all 22 tickers |
| `POST` | `/ml/train` | Trigger model training via Celery |
| `GET` | `/ml/model/info` | Active model information |
| `GET` | `/ml/predictions/history` | Prediction history from DB |
| `POST` | `/rag/chat/ask` | Chat with AI analyst |
| `POST` | `/rag/ingest/{ticker}` | Ingest documents to ChromaDB |
| `GET` | `/rag/stats` | ChromaDB statistics |
| `GET` | `/retrain/status` | Adaptive retraining status |
| `GET` | `/stocks/{ticker}/history` | OHLCV historical data |
| `GET` | `/chat` | Chat UI (HTML) |
| `GET` | `/docs` | Swagger UI |

---

## 📈 Model Performance

| Metric | Value |
|--------|-------|
| Algorithm | XGBoost Multiclass Classifier |
| Train Accuracy | 47.4% |
| Test Accuracy | 38.4% → 45.6% (live evaluation) |
| Overfit Gap | 0.09 (healthy, target < 0.10) |
| F1 Macro | 0.35 |
| Random Baseline | 33.3% (3 balanced classes) |
| Improvement | +5.1% above random baseline |
| Features | 20 technical indicators |
| Training Data | 4,180 rows across 22 tickers |

> Stock price prediction is inherently noisy. A test accuracy of 38-45% above a 33% random baseline is a realistic and honest result. This model is an **analytical aid, not investment advice.**

---

## 🗂️ Project Structure

```
ihsg-intelligence/
├── app/
│   ├── api/
│   │   ├── auth.py              # JWT authentication
│   │   ├── ml.py                # ML prediction endpoints
│   │   ├── rag.py               # RAG chat endpoints
│   │   ├── stocks.py            # Stock history endpoints
│   │   └── websocket.py         # WebSocket for live signals
│   ├── core/
│   │   ├── database.py          # SQLAlchemy + Supabase connection
│   │   ├── security.py          # JWT utilities
│   │   └── limiter.py           # Rate limiting
│   ├── models/
│   │   ├── stock.py             # StockHistory ORM (BigInteger volume)
│   │   ├── prediction.py        # Prediction ORM
│   │   ├── user.py              # User ORM
│   │   └── conversation.py      # Chat session ORM
│   ├── services/
│   │   ├── ingestion.py             # Daily OHLCV ingestion (UPSERT)
│   │   ├── ingestion_historical.py  # 1-year historical backfill
│   │   ├── ingest_from_db.py        # Build ChromaDB from PostgreSQL
│   │   ├── features.py              # Feature engineering (20 features)
│   │   ├── trainer.py               # XGBoost training pipeline
│   │   ├── predictor.py             # Model inference + caching
│   │   ├── rag_chat.py              # RAG chat engine
│   │   ├── vector_store.py          # ChromaDB management
│   │   ├── document_loader.py       # TradingView + Google News scraper
│   │   ├── llm_service.py           # Llama 3.2 integration
│   │   ├── monitoring.py            # Evidently AI pipeline
│   │   ├── telegram.py              # Telegram notification service
│   │   └── event_bus.py             # WebSocket signal broadcasting
│   ├── main.py
│   └── worker.py                # Celery task definitions
├── .github/
│   └── workflows/
│       └── daily_pipeline.yml   # 5-job adaptive CI/CD pipeline
├── docs/
│   ├── model_card_v1.0.pdf      # Model specs, evaluation, deployment
│   └── incident_report.pdf      # Production bugs & post-mortem
├── mlruns/                      # MLflow artifacts (gitignored)
├── chroma_db/                   # ChromaDB local store (gitignored)
├── reports/                     # Evidently HTML reports (gitignored)
├── test_chat.html               # Chat UI
├── requirements.txt
└── README.md
```

---

## 🔧 Known Limitations & Workarounds

| Issue | Workaround |
|-------|-----------|
| Supabase connection | Use Transaction Pooler port **6543**, not 5432 |
| XGBoost 2.x compatibility | Remove `use_label_encoder`, serialize via `pickle` not `mlflow.xgboost` |
| Evidently breaking changes | Pinned to `evidently==0.4.38` |
| Celery on Windows | Run with `-P gevent` flag |
| Redis SSL | Append `?ssl_cert_reqs=CERT_NONE` to REDIS_URL |
| GOTO.JK volume overflow | Column `volume` changed from `Integer` to `BigInteger` |
| yfinance ghost rows | 3-layer filter: ingestion → features → predictor |
| LLM multi-turn context | `MAX_HISTORY_MSGS = 0` — Llama 3.2 3B is too small for reliable multi-turn |
| Ingestion data loss | Replaced DELETE+INSERT with UPSERT pattern |

---

## 📄 Documentation

| Document | Description |
|----------|-------------|
| [Model Card v1.0](docs/model_card_v1.0.pdf) | Full model specs, features, training, evaluation & deployment details |
| [Incident Report](docs/incident_report.pdf) | 7 production bugs — timeline, root cause analysis & prevention |

---

## 🗺️ Roadmap

- [x] Phase 1 — Data ingestion, REST API, JWT authentication
- [x] Phase 2 — XGBoost ML pipeline, MLflow tracking, Redis caching
- [x] Phase 3 — RAG system, Llama 3.2, TradingView scraper, auto-ingest
- [x] Phase 4 — Evidently monitoring, Grafana dashboard, Telegram alerts, adaptive CI/CD
- [ ] Phase 5 — Full adaptive retraining implementation
- [ ] Deploy to Hugging Face Spaces
- [ ] Upgrade LLM to larger model (Llama 3.1 70B or Mistral)
- [ ] PDF parser for IDX financial reports (PyMuPDF — foundation ready)

---

## ⚠️ Disclaimer

This project is a **research and portfolio project**. The signals generated are **not official investment advice**. Always conduct your own research before making any financial decisions. Investing in stocks always carries risk.

---

## 👤 Author

**Rafli Opticinn**  
Built with ❤️ and way too much debugging — April 2026
