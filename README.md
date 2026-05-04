# 🧠💵📈 IHSG Intelligence Platform

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
![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-CI%2FCD-black?style=flat-square&logo=githubactions)

---

## 📸 Screenshots

<table>
  <tr>
    <td align="center">
      <img width="480" height="755" alt="image" src="https://github.com/user-attachments/assets/150cce67-0cf6-4465-bdec-c562d6a9970f" />
      <br/><sub><b>AI Chat Interface</b></sub>
    </td>
    <td align="center">
      <img width="480" height="238" alt="image" src="https://github.com/user-attachments/assets/7cdb655b-a793-495d-9421-0f83c977f595" />
      <br/><sub><b>Grafana MLOps Dashboard</b></sub>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img width="1280" height="849" alt="image" src="https://github.com/user-attachments/assets/2323c27e-609f-4900-bab2-7e2ef67639e8" />
      <br/><sub><b>REST API — Swagger UI</b></sub>
    </td>
    <td align="center">
      <img width="480" height="1068" alt="image" src="https://github.com/user-attachments/assets/bafbd69b-5574-426b-98b2-d2ca2d79bb7f" />
      <br/><sub><b>Telegram Signal Alert</b></sub>
    </td>
  </tr>
</table>

---

## 📌 Overview

IHSG Intelligence Platform is a production-grade ML system that predicts short-term trading signals **(BUY / HOLD / SELL)** for 22 high-cap Indonesian stocks, enriched with real-time news analysis via a RAG (Retrieval-Augmented Generation) pipeline.

Users interact through a natural language chat interface powered by a locally-running LLM (Llama 3.2), which answers stock-related questions using a combination of ML predictions and live data fetched from **TradingView Ideas** and **Google News**.

### What makes this different?
- **Zero-Dollar Stack** — runs entirely on free-tier services and local hardware
- **Production-grade MLOps** — monitoring, drift detection, adaptive retraining, CI/CD pipeline
- **RAG over financial data** — LLM grounded in real market data, not hallucination
- **End-to-end** — from raw OHLCV ingestion to Telegram alerts in one cohesive system

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     IHSG Intelligence Platform                          │
├─────────────┬─────────────┬──────────────┬───────────────┬──────────────┤ 
│  Phase 1    │  Phase 2    │   Phase 3    │   Phase 4     │    Phase 5   │
│  Data & API │ ML Pipeline │   RAG + LLM  │  Monitoring   │    Adaptive  │
│             │             │              │  & Alerting   │    Retrain   │
├─────────────┼─────────────┼──────────────┼───────────────┼──────────────┤
│ FastAPI     │ XGBoost     │ Llama 3.2    │ Evidently AI  │   Retrain    │
│ PostgreSQL  │ MLflow      │ ChromaDB     │ Grafana Cloud │   Trigger    │
│ Supabase    │ Feature Eng │ TradingView  │ Telegram Bot  │   State Mgmt │
│ JWT Auth    │ Predictor   │ Google News  │ GitHub Actions│  API Ctrl    │
│ Redis Cache │ Celery      │ LangChain    │ Auto Retrain  │   Dispatch   │
└─────────────┴─────────────┴──────────────┴───────────────┴──────────────┘
```

**Data Flow:**
```
yfinance → PostgreSQL → Feature Engineering → XGBoost → Redis Cache
                                                    ↓
TradingView + Google News → ChromaDB → Llama 3.2 → Chat Response
                                                    ↓
Evidently AI → monitoring_logs → Grafana Dashboard + Telegram Alert
                                                    ↓
              GitHub Actions → Adaptive Retrain → New Model
```

---

## ✨ Key Features

### 🤖 ML Signal Prediction
- XGBoost multiclass classifier — BUY / HOLD / SELL
- 20 engineered features: RSI, MACD, Bollinger Bands, Volume Ratio, lag features
- MLflow experiment tracking with pickle-based model persistence
- Redis caching — predictions cached 1 hour per ticker
- Signal label threshold: `return > +0.5%` → BUY | `< -0.5%` → SELL

### 💬 RAG Chat System
- Natural language Q&A about IHSG stocks in Bahasa Indonesia
- Llama 3.2 3B running 100% locally via Ollama (zero cost)
- ChromaDB vector database for semantic retrieval
- Real-time data sources (priority order):
  - **TradingView Ideas** — community trader technical analysis
  - **Google News RSS** — latest news per ticker (free, no API key)
  - **Internal DB** — price summary, ML signals, sector profiles
- Auto-detects ticker from natural language question
- Auto-triggers predict + ingest on first mention of a new ticker

### 📊 Monitoring & Observability
- **Evidently AI** — unified HTML reports across all 22 tickers:
  - Data Quality (ghost row detection, missing values)
  - Data Drift (feature distribution shift vs 60-day baseline)
  - Model Performance (accuracy, F1, signal distribution)
- **Grafana Cloud** — 9-panel MLOps dashboard, 5-minute auto-refresh
- **Telegram Bot** — real-time push alerts for:
  - Trading signals (confidence > 38%)
  - Data drift (> 50%)
  - Model accuracy degradation
  - Daily pipeline completion

### 🔁 Adaptive Retraining (Phase 5)
- `retrain_trigger.py` — evaluates 4 conditions every run:
  - Scheduled day (Monday / Thursday)
  - IHSG market move > ±2.5%
  - Realized volatility > 90th percentile (30-day window)
  - Model accuracy drift > 10% from baseline
- 18-hour cooldown between retrains (prevents over-retraining)
- State persistence via `retrain_state.pkl`
- API endpoints: `GET /retrain/status`, `GET /retrain/check`, `POST /retrain/trigger`
- Emergency dispatch to GitHub Actions via REST API

### ⚙️ MLOps Pipeline (GitHub Actions)
```
Mon–Fri at 16:30 WIB:
  Ingest → Feature Check → Monitor → Retrain (conditional) → Notify

On-Demand (adaptive_retrain.yml):
  Validate → Ingest → Train → Verify → Notify
```

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
| Feature Engineering | pandas, numpy, ta (technical analysis) |
| LLM | Llama 3.2 3B via Ollama (100% local) |
| RAG | LangChain, ChromaDB, HuggingFace Embeddings |
| Embeddings | `paraphrase-multilingual-MiniLM-L12-v2` |
| Database | PostgreSQL via Supabase (Transaction Pooler port 6543) |
| Cache | Redis |
| Task Queue | Celery + gevent (Windows compatible) |
| Monitoring | Evidently AI 0.4.38 |
| Dashboard | Grafana Cloud |
| Alerting | Telegram Bot API |
| CI/CD | GitHub Actions (5-job + on-demand pipeline) |
| Data Sources | yfinance, TradingView scraper, Google News RSS |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11
- Redis (local)
- [Ollama](https://ollama.com) with `llama3.2` model
- Supabase account (free tier works)

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
# Fill in your credentials
```

### Environment Variables

```env
DATABASE_URL=postgresql://postgres.[PROJECT-ID]:[PASSWORD]@[HOST].pooler.supabase.com:6543/postgres
REDIS_URL=redis://localhost:6379/0
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
GITHUB_TOKEN=your_github_pat          # for /retrain/trigger endpoint
GITHUB_REPO=username/ihsg-intelligence
```

### Running the System

```bash
# Terminal 1 — FastAPI server
uvicorn app.main:app --reload

# Terminal 2 — Celery worker
celery -A app.worker.celery_app worker --loglevel=info -P gevent

# Terminal 3 — Ollama
ollama serve
```

### First Time Setup

```bash
# 1. Backfill 1 year of historical data (~239 rows per ticker)
python -m app.services.ingestion_historical

# 2. Train the XGBoost model
python -m app.services.trainer

# 3. Populate ChromaDB from internal database
python -m app.services.ingest_from_db

# 4. Run full monitoring pipeline
python -m app.services.monitoring
```

Open `http://localhost:8000/chat` for the chat interface.  
Open `http://localhost:8000/docs` for the Swagger API docs.

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/ml/predict/{ticker}` | Trading signal (Redis cached) |
| `GET` | `/ml/predict/batch/all` | Predict all 22 tickers |
| `POST` | `/ml/train` | Trigger training via Celery |
| `GET` | `/ml/model/info` | Active model info |
| `GET` | `/ml/predictions/history` | Prediction history |
| `POST` | `/rag/chat/ask` | Chat with AI analyst |
| `POST` | `/rag/ingest/{ticker}` | Ingest documents to ChromaDB |
| `GET` | `/rag/stats` | ChromaDB statistics |
| `GET` | `/retrain/status` | Retrain state & config |
| `GET` | `/retrain/check` | Evaluate retrain conditions |
| `POST` | `/retrain/trigger` | Emergency retrain via GitHub API |
| `GET` | `/stocks/{ticker}/history` | OHLCV historical data |
| `GET` | `/chat` | Chat UI |
| `GET` | `/docs` | Swagger UI |

---

## 📈 Model Performance

| Metric | Value |
|--------|-------|
| Algorithm | XGBoost Multiclass Classifier |
| Train Accuracy | 47.4% |
| Test Accuracy | 38.4% → **45.6% on live data** |
| Overfit Gap | 0.09 (healthy — target < 0.10) |
| F1 Macro | 0.35 |
| Random Baseline | 33.3% (3 balanced classes) |
| Improvement vs Baseline | **+5.1%** |
| Features | 20 technical indicators |
| Training Data | 4,180 rows — 22 tickers × ~190 rows |

> Stock price prediction is inherently noisy. 38-45% above a 33% random baseline is a realistic and honest result for daily OHLCV data. This model is an **analytical aid, not investment advice.**

---

## 🗂️ Project Structure

```
ihsg-intelligence/
├── app/
│   ├── api/
│   │   ├── auth.py              # JWT authentication
│   │   ├── ml.py                # ML prediction endpoints
│   │   ├── rag.py               # RAG chat endpoints
│   │   ├── retrain.py           # Adaptive retrain control endpoints
│   │   ├── stocks.py            # Stock history endpoints
│   │   └── websocket.py         # WebSocket live signals
│   ├── core/
│   │   ├── database.py          # SQLAlchemy + Supabase
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
│   │   ├── ingest_from_db.py        # ChromaDB from PostgreSQL
│   │   ├── features.py              # Feature engineering (20 features)
│   │   ├── trainer.py               # XGBoost training pipeline
│   │   ├── predictor.py             # Inference + Redis caching
│   │   ├── rag_chat.py              # RAG chat engine
│   │   ├── vector_store.py          # ChromaDB management
│   │   ├── document_loader.py       # TradingView + Google News scraper
│   │   ├── llm_service.py           # Llama 3.2 integration
│   │   ├── monitoring.py            # Evidently AI pipeline
│   │   ├── retrain_trigger.py       # Adaptive retrain logic (Phase 5)
│   │   ├── telegram.py              # Telegram notifications
│   │   └── event_bus.py             # WebSocket broadcasting
│   ├── main.py
│   └── worker.py
├── .github/
│   └── workflows/
│       ├── daily_pipeline.yml       # Scheduled 5-job pipeline
│       └── adaptive_retrain.yml     # On-demand retrain workflow
├── docs/
│   ├── screenshots/                 # Project screenshots (for README)
│   │   ├── chat_ui.png
│   │   ├── grafana_dashboard.png
│   │   ├── swagger_ui.png
│   │   └── telegram_alert.png
│   ├── model_card_v1.0.pdf
│   └── incident_report.pdf
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
| XGBoost 2.x | Remove `use_label_encoder`, serialize via `pickle` not `mlflow.xgboost` |
| Evidently API changes | Pinned to `evidently==0.4.38` |
| Celery on Windows | Run with `-P gevent` flag |
| Redis SSL | Append `?ssl_cert_reqs=CERT_NONE` to `REDIS_URL` |
| GOTO.JK volume overflow | Column `volume` → `BigInteger` + `ALTER TABLE` in Supabase |
| yfinance ghost rows | 3-layer filter: ingestion → features → predictor |
| LLM multi-turn confusion | `MAX_HISTORY_MSGS = 0` — Llama 3.2 3B too small for reliable multi-turn |
| Ingestion data loss | Replaced DELETE+INSERT with UPSERT pattern |
| pywin32 in CI | Removed from `requirements.txt` — Windows-only, breaks Ubuntu runner |

---

## 📄 Documentation

| Document | Description |
|----------|-------------|
| [Model Card v1.0](docs/model_card_v1.0.pdf) | Model specs, features, training config, evaluation & deployment |
| [Incident Report](docs/incident_report.pdf) | 7 production bugs — timeline, root cause & prevention |

---

## 🗺️ Roadmap

- [x] Phase 1 — Data ingestion, REST API, JWT authentication
- [x] Phase 2 — XGBoost ML pipeline, MLflow tracking, Redis caching
- [x] Phase 3 — RAG system, Llama 3.2, TradingView scraper, auto-ingest
- [x] Phase 4 — Evidently monitoring, Grafana dashboard, Telegram alerts, CI/CD
- [x] Phase 5 — Adaptive retraining: trigger logic, state persistence, API control, emergency dispatch
- [ ] HF Spaces deployment *(blocked: Ollama RAM > free tier limit)*
- [ ] Larger LLM upgrade *(blocked: local hardware — 1-line config change when ready)*
- [ ] PDF upload endpoint for IDX financial reports *(PyMuPDF foundation ready)*

---

## ⚠️ Disclaimer

This project is a **research and portfolio project**. The signals generated are **not official investment advice**. Always conduct your own research before making any financial decisions. Investing in stocks always carries risk.

---

## 👤 Author

**Rafli Opticinn**  
Built with ❤️ and way too much debugging — April 2026

---

<div align="center">
  <sub>If you find this project useful, consider giving it a ⭐</sub>
</div>
