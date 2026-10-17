# TimesFM Stock Prediction Platform

> A clean, modernistic stock forecasting dashboard powered by Google's **TimesFM 2.5 200M** time-series foundation model. Built for zero-shot inference, multi-horizon quantile uncertainty estimation, out-of-sample backtesting, and collaborative multi-agent coordination.

---

## ⚡ Features

- **Google TimesFM 2.5 200M Foundation Model**:
  - Zero-shot time-series forecasting.
  - Multi-horizon forecasts (7, 14, 30, 60, and 90 trading days).
  - 10-channel quantile predictions covering the 10th, 25th, 50th (median), 75th, and 90th percentiles for probabilistic uncertainty cones.
- **Modern Trading Dashboard UI**:
  - High-density dark mode inspired by TradingView and Linear.
  - Real-time interactive charts with crosshairs and tooltips (Chart.js).
  - Multi-view canvas: TimesFM Forecast Cone, Out-of-Sample Backtesting, Technical Indicators, and Multi-Asset Comparative Projections.
  - Live condition gauges: RSI-14 meter, MACD histogram, Bollinger Band positioning, and Risk Tier assessment.
- **Value-at-Risk (VaR) & Downside Risk Engine**:
  - Parametric & Historical VaR at 95% and 99% confidence horizons.
  - Expected Shortfall (CVaR) and TimesFM Q10 downside trajectory evaluation.
  - Annualized Volatility and Projected Sharpe Ratio calculation.
- **Multi-Ticker Comparative Forecast & Ranking**:
  - Normalized percentage-gain trajectory charts comparing multiple tickers simultaneously.
  - Automated return-based performance leaderboard ranking.
- **Quantitative Export Engine**:
  - Instant CSV & JSON data export containing historical prices, TimesFM point forecasts, and 10th-90th quantile spreads.
- **Out-of-Sample Backtesting Engine**:
  - Evaluates predictions against actual market ground truth.
  - Metrics: Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), Mean Absolute Percentage Error (MAPE), and Directional Accuracy %.
- **Vercel Cloud Deployment & Serverless Integration**:
  - Production-ready `vercel.json` routing configuration.
  - Dedicated lightweight serverless function entrypoint in `api/index.py` with zero-overhead dependency packaging (`api/requirements.txt`).
- **Multi-Agent Coordination (`TASKS.txt`)**:
  - Dedicated asynchronous task queue and communication board for Developers, Testers, and Project Managers.
  - Interactive UI panel allowing team members to view, add tasks, or log messages directly to `TASKS.txt`.

---

## 🏗 Architecture

```
stockPredictionProgram/
├── TASKS.txt              # Asynchronous agent coordination and task queue
├── vercel.json            # Vercel cloud deployment & routing configuration
├── requirements.txt       # Full local Python dependencies (PyTorch + TimesFM)
├── run.sh                 # Single-click local launch script
├── run_tests.py           # Automated test suite runner (29 tests)
├── api/
│   ├── index.py           # Vercel serverless function entrypoint
│   └── requirements.txt   # Lightweight serverless dependencies for cloud hosting
├── engine/
│   ├── data_loader.py     # yfinance fetcher, indicators, caching, and resilient simulation
│   ├── model_wrapper.py   # TimesFM 2.5 200M wrapper, quantile extraction & backtesting
│   └── risk_engine.py     # Value-at-Risk (VaR), CVaR, Sharpe & downside volatility
├── server/
│   └── main.py            # FastAPI REST API serving endpoints and dashboard static files
├── frontend/
│   ├── index.html         # Modernistic reactive single-page dashboard
│   ├── styles.css         # Glassmorphism dark styling & animation rules
│   └── app.js             # Client state, Chart.js multi-view renderer & live sync
└── tests/
    ├── test_data_loader.py
    ├── test_model.py
    ├── test_engine.py
    ├── test_risk_engine.py
    └── test_api.py
```


---

## 🚀 Quickstart

### 1. Prerequisites
- Python 3.10+ (PyTorch, TimesFM)
- Git

### 2. Setup Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Launch Platform
```bash
./run.sh
```
Open **[http://localhost:8000](http://localhost:8000)** in your browser.

---

## 🧪 Running Automated Tests

```bash
python3 -m unittest discover -s tests -v
```

---

## 🤖 Multi-Agent Communication Protocol (`TASKS.txt`)

All communication between the Lead Developer, Tester, and Project Manager/Orchestrator is coordinated through `TASKS.txt` in the root directory:
- **Task Assignment**: Tasks can be appended under `[2. ACTIVE TASK QUEUE]` with format:
  `[TASK-ID] | YYYY-MM-DD HH:MM | From -> To | PENDING | HIGH/MED/LOW | Description`
- **Communication Log**: Updates and questions are posted under `[4. AGENT COMMUNICATION LOG]`.
- **Architectural RFCs**: Major updates must be registered under `[5. ARCHITECTURAL DECISIONS & COORDINATION]`.
- **API Access**: The dashboard also provides endpoints `/api/tasks`, `/api/tasks/create`, and `/api/tasks/message` for programmatic coordination.
