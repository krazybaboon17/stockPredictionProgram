"""
FastAPI Server for TimesFM Stock Prediction Platform.
Exposes endpoints for stock data, TimesFM forecasts, backtesting,
and real-time agent task coordination.
"""

import os
import sys
import logging
import datetime
from typing import Optional, List
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, Response

# Add parent dir to sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from engine.data_loader import StockDataEngine, POPULAR_TICKERS
from engine.model_wrapper import TimesFMForecaster
from engine.risk_engine import RiskEngine
from engine.regime_engine import RegimeEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("server")

app = FastAPI(
    title="TimesFM Stock Forecasting Platform",
    description="Clean, modernistic dashboard powered by Google's TimesFM time-series foundation model.",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Engine Singletons
data_engine = StockDataEngine()
forecaster = TimesFMForecaster()

TASKS_FILE = os.path.join(BASE_DIR, "TASKS.txt")
_TASKS_CACHE_OVERRIDE: Optional[str] = None


# --- Pydantic Schemas ---
class PredictRequest(BaseModel):
    ticker: str = Field(..., example="NVDA")
    horizon: int = Field(30, ge=1, le=90, description="Forecast horizon in trading days")
    period: str = Field("2y", description="Historical period for context window")

class CompareRequest(BaseModel):
    tickers: List[str] = Field(..., min_length=1, max_length=8, description="List of tickers to compare")
    horizon: int = Field(30, ge=1, le=90, description="Forecast horizon in trading days")
    period: str = Field("2y", description="Historical period")

class BacktestRequest(BaseModel):
    ticker: str = Field(..., example="NVDA")
    test_days: int = Field(30, ge=5, le=90, description="Days to hold out for evaluation")
    period: str = Field("2y", description="Historical period")

class NewTaskRequest(BaseModel):
    task_id: Optional[str] = None
    from_role: str = Field("Tester", description="Role creating the task")
    to_role: str = Field("Lead Developer", description="Target role")
    priority: str = Field("HIGH", description="HIGH, MED, LOW")
    description: str = Field(..., description="Details of the task")

class ChatMessageRequest(BaseModel):
    author: str = Field(..., example="Tester")
    message: str = Field(..., example="Verified model quantiles pass unit checks.")


# --- API Endpoints ---
@app.get("/api/health")
async def health_check():
    return {
        "status": "online",
        "model": forecaster.model_status,
        "is_real_timesfm": forecaster.is_real_timesfm,
        "tasks_file_exists": os.path.exists(TASKS_FILE),
        "cached_tickers_count": len(POPULAR_TICKERS)
    }

@app.get("/api/tickers/search")
async def search_tickers(q: str = Query("", description="Ticker search query")):
    return data_engine.search_tickers(q)

@app.get("/api/stock/{ticker}")
async def get_stock_data(ticker: str, period: str = "2y", force_refresh: bool = False):
    cleaned = (ticker or "").strip()
    if not cleaned:
        raise HTTPException(status_code=400, detail="Ticker symbol cannot be empty.")
    try:
        data = data_engine.get_stock_data(cleaned, period=period, force_refresh=force_refresh)
        return data
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching stock data for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/predict")
async def predict_stock(req: PredictRequest):
    cleaned = (req.ticker or "").strip()
    if not cleaned:
        raise HTTPException(status_code=400, detail="Ticker symbol cannot be empty.")
    try:
        stock_data = data_engine.get_stock_data(cleaned, period=req.period)
        history = stock_data.get("history", [])
        if not history:
            raise HTTPException(status_code=404, detail=f"No historical data found for {req.ticker}")

        prices = [h["close"] for h in history]
        last_date = history[-1]["date"]

        forecast = forecaster.forecast_series(
            prices=prices,
            horizon=req.horizon,
            start_date_str=last_date
        )

        return {
            "ticker": cleaned.upper(),
            "name": stock_data.get("name", cleaned),
            "current_price": stock_data.get("current_price"),
            "forecast": forecast
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error for {req.ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/backtest")
async def backtest_stock(req: BacktestRequest):
    cleaned = (req.ticker or "").strip()
    if not cleaned:
        raise HTTPException(status_code=400, detail="Ticker symbol cannot be empty.")
    try:
        stock_data = data_engine.get_stock_data(cleaned, period=req.period)
        history = stock_data.get("history", [])
        if not history or len(history) < req.test_days + 30:
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient history ({len(history)} pts) for a {req.test_days}-day backtest."
            )

        prices = [h["close"] for h in history]
        dates = [h["date"] for h in history]

        result = forecaster.run_backtest(prices, dates, test_days=req.test_days)
        return {
            "ticker": cleaned.upper(),
            "name": stock_data.get("name", cleaned),
            "backtest": result
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Backtest error for {req.ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/predict/compare")
async def compare_predictions(req: CompareRequest):
    """Generates comparative multi-ticker TimesFM forecasts with normalized % return trajectories."""
    if not req.tickers:
        raise HTTPException(status_code=400, detail="Must supply at least one ticker.")

    results = []
    dates = []
    for ticker in req.tickers:
        cleaned = ticker.strip().upper()
        if not cleaned:
            continue
        try:
            stock_data = data_engine.get_stock_data(cleaned, period=req.period)
            history = stock_data.get("history", [])
            if not history:
                continue
            prices = [h["close"] for h in history]
            last_date = history[-1]["date"]
            start_price = prices[-1]

            forecast = forecaster.forecast_series(
                prices=prices,
                horizon=req.horizon,
                start_date_str=last_date
            )
            if not dates and forecast.get("predictions"):
                dates = [p["date"] for p in forecast["predictions"]]

            normalized_pct = []
            for p in forecast.get("predictions", []):
                pct = round(((p["predicted"] - start_price) / start_price) * 100, 2)
                normalized_pct.append(pct)

            target_price = forecast.get("predicted_end_price", start_price)
            exp_return = round(((target_price - start_price) / start_price) * 100, 2)

            results.append({
                "ticker": cleaned,
                "name": stock_data.get("name", cleaned),
                "sector": stock_data.get("sector", "Equity"),
                "start_price": round(float(start_price), 2),
                "predicted_end_price": round(float(target_price), 2),
                "expected_return_pct": exp_return,
                "trend": forecast.get("trend", "NEUTRAL"),
                "normalized_pct_trajectory": normalized_pct,
                "predictions": forecast.get("predictions", [])
            })
        except Exception as e:
            logger.warning(f"Error computing comparative forecast for {cleaned}: {e}")

    # Sort descending by expected return
    results.sort(key=lambda x: x["expected_return_pct"], reverse=True)

    return {
        "horizon": req.horizon,
        "dates": dates,
        "comparisons": results
    }

@app.get("/api/export/{ticker}")
async def export_forecast(
    ticker: str,
    format: str = Query("csv", pattern="^(csv|json)$"),
    horizon: int = Query(30, ge=1, le=90),
    period: str = Query("2y")
):
    """Exports TimesFM historical and forecast data as CSV or JSON."""
    cleaned = (ticker or "").strip().upper()
    if not cleaned:
        raise HTTPException(status_code=400, detail="Ticker symbol cannot be empty.")

    stock_data = data_engine.get_stock_data(cleaned, period=period)
    history = stock_data.get("history", [])
    if not history:
        raise HTTPException(status_code=404, detail=f"No data for {cleaned}")

    prices = [h["close"] for h in history]
    last_date = history[-1]["date"]
    forecast = forecaster.forecast_series(prices, horizon=horizon, start_date_str=last_date)

    if format == "json":
        payload = {
            "ticker": cleaned,
            "name": stock_data.get("name", cleaned),
            "exported_at": str(datetime.datetime.now().isoformat()) if "datetime" in globals() else "",
            "model": forecast.get("model"),
            "current_price": stock_data.get("current_price"),
            "horizon_days": horizon,
            "forecast": forecast.get("predictions", []),
            "history_last_60": history[-60:]
        }
        return JSONResponse(
            content=payload,
            headers={"Content-Disposition": f"attachment; filename={cleaned}_timesfm_forecast.json"}
        )

    # CSV Format
    import csv
    import io
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Date", "Ticker", "Type", "Close_or_Predicted", "Q10", "Q25", "Q50_Median", "Q75", "Q90", "SMA_20", "RSI"])

    # Write recent historical points
    for h in history[-60:]:
        writer.writerow([
            h["date"], cleaned, "HISTORICAL", h["close"],
            "", "", "", "", "",
            h.get("sma_20", ""), h.get("rsi", "")
        ])

    # Write forecast rows
    for p in forecast.get("predictions", []):
        writer.writerow([
            p["date"], cleaned, "TIMESFM_FORECAST", p["predicted"],
            p["q10"], p["q25"], p["q50"], p["q75"], p["q90"],
            "", ""
        ])

    csv_data = output.getvalue()
    return Response(
        content=csv_data,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={cleaned}_timesfm_forecast.csv"}
    )


@app.get("/api/risk/{ticker}")
async def get_risk_evaluation(
    ticker: str,
    horizon: int = Query(30, ge=1, le=90),
    period: str = Query("2y")
):
    """Calculates Value-at-Risk (VaR), Expected Shortfall, and downside metrics."""
    cleaned = (ticker or "").strip().upper()
    if not cleaned:
        raise HTTPException(status_code=400, detail="Ticker symbol cannot be empty.")

    stock_data = data_engine.get_stock_data(cleaned, period=period)
    history = stock_data.get("history", [])
    if not history:
        raise HTTPException(status_code=404, detail=f"No data for {cleaned}")

    prices = [h["close"] for h in history]
    last_date = history[-1]["date"]
    current_price = float(stock_data.get("current_price", prices[-1]))

    forecast = forecaster.forecast_series(prices, horizon=horizon, start_date_str=last_date)
    risk_metrics = RiskEngine.calculate_risk_metrics(
        current_price=current_price,
        historical_prices=prices,
        forecast_predictions=forecast.get("predictions", [])
    )

    return {
        "ticker": cleaned,
        "name": stock_data.get("name", cleaned),
        "current_price": current_price,
        "risk_metrics": risk_metrics
    }


@app.get("/api/regime/{ticker}")
async def get_market_regime(ticker: str):
    """
    Automated Market Regime & Momentum Screener endpoint.
    Categorizes ticker market condition into Bullish Expansion, Bearish Breakdown,
    Overbought Exhaustion, or Squeeze Consolidation with conviction score and metrics.
    """
    cleaned = ticker.strip().upper()
    if not cleaned:
        raise HTTPException(status_code=400, detail="Ticker symbol cannot be empty")

    try:
        stock_data = data_engine.get_stock_data(cleaned, period="1y")
    except Exception as e:
        logger.error(f"Error fetching data for regime screener {cleaned}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    if not stock_data or not stock_data.get("history"):
        raise HTTPException(status_code=404, detail=f"No market data available for ticker '{cleaned}'")

    regime_data = RegimeEngine.classify_regime(stock_data)
    return regime_data


@app.get("/api/tasks")
async def get_tasks():
    """Reads TASKS.txt for real-time agent coordination."""
    global _TASKS_CACHE_OVERRIDE
    if _TASKS_CACHE_OVERRIDE is not None:
        return {"content": _TASKS_CACHE_OVERRIDE}

    if not os.path.exists(TASKS_FILE):
        return {"content": "", "raw": "TASKS.txt not found."}
    
    try:
        with open(TASKS_FILE, "r", encoding="utf-8") as f:
            content = f.read()
        return {"content": content}
    except Exception as e:
        return {"content": f"TASKS.txt read error: {e}"}


@app.post("/api/tasks/create")
async def create_task(req: NewTaskRequest):
    """Allows testers and PMs to append a task into TASKS.txt."""
    global _TASKS_CACHE_OVERRIDE
    import datetime
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    task_id = req.task_id or f"TASK-{int(datetime.datetime.now().timestamp()) % 10000:04d}"
    entry_line = f"[{task_id}] | {now_str} | {req.from_role} -> {req.to_role} | PENDING | {req.priority} | {req.description}\n"

    if _TASKS_CACHE_OVERRIDE is not None:
        text = _TASKS_CACHE_OVERRIDE
    elif os.path.exists(TASKS_FILE):
        try:
            with open(TASKS_FILE, "r", encoding="utf-8") as f:
                text = f.read()
        except Exception:
            text = ""
    else:
        text = ""

    # Append under ACTIVE TASK QUEUE
    marker = "2. ACTIVE TASK QUEUE\n--------------------------------------------------------------------------------\n[Format: Task ID | Date | From -> To | Status | Priority | Description]\n"
    if marker in text:
        parts = text.split(marker)
        new_text = parts[0] + marker + entry_line + parts[1]
    else:
        new_text = text + f"\n\n[NEW TASK - {now_str}]\n" + entry_line

    _TASKS_CACHE_OVERRIDE = new_text

    try:
        with open(TASKS_FILE, "w", encoding="utf-8") as f:
            f.write(new_text)
    except (OSError, PermissionError) as e:
        logger.info(f"TASKS.txt disk write skipped (read-only filesystem): {e}")

    return {"success": True, "task_id": task_id, "message": "Task queued successfully."}


@app.post("/api/tasks/message")
async def post_message(req: ChatMessageRequest):
    """Allows agents to post a communication log message into TASKS.txt."""
    global _TASKS_CACHE_OVERRIDE
    import datetime
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"\n[{now_str}] [{req.author}]:\n{req.message}\n"

    if _TASKS_CACHE_OVERRIDE is not None:
        text = _TASKS_CACHE_OVERRIDE
    elif os.path.exists(TASKS_FILE):
        try:
            with open(TASKS_FILE, "r", encoding="utf-8") as f:
                text = f.read()
        except Exception:
            text = ""
    else:
        text = ""

    # Append under section 4
    marker = "4. AGENT COMMUNICATION LOG\n--------------------------------------------------------------------------------"
    if marker in text:
        parts = text.split(marker)
        new_text = parts[0] + marker + entry + parts[1]
    else:
        new_text = text + entry

    _TASKS_CACHE_OVERRIDE = new_text

    try:
        with open(TASKS_FILE, "w", encoding="utf-8") as f:
            f.write(new_text)
    except (OSError, PermissionError) as e:
        logger.info(f"TASKS.txt disk write skipped (read-only filesystem): {e}")

    return {"success": True, "message": "Message logged to TASKS.txt."}


# --- Static Frontend Mount ---
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    @app.get("/")
    async def serve_index():
        index_file = os.path.join(FRONTEND_DIR, "index.html")
        if os.path.exists(index_file):
            return FileResponse(index_file)
        return JSONResponse({"status": "Frontend building..."})

    @app.get("/styles.css")
    async def serve_styles():
        css_file = os.path.join(FRONTEND_DIR, "styles.css")
        if os.path.exists(css_file):
            return FileResponse(css_file, media_type="text/css")
        raise HTTPException(status_code=404, detail="styles.css not found")

    @app.get("/app.js")
    async def serve_js():
        js_file = os.path.join(FRONTEND_DIR, "app.js")
        if os.path.exists(js_file):
            return FileResponse(js_file, media_type="application/javascript")
        raise HTTPException(status_code=404, detail="app.js not found")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
