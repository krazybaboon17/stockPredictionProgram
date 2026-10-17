"""
FastAPI Server for TimesFM Stock Prediction Platform.
Exposes endpoints for stock data, TimesFM forecasts, backtesting,
and real-time agent task coordination.
"""

import os
import sys
import logging
from typing import Optional, List
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

# Add parent dir to sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from engine.data_loader import StockDataEngine, POPULAR_TICKERS
from engine.model_wrapper import TimesFMForecaster

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


# --- Pydantic Schemas ---
class PredictRequest(BaseModel):
    ticker: str = Field(..., example="NVDA")
    horizon: int = Field(30, ge=1, le=90, description="Forecast horizon in trading days")
    period: str = Field("2y", description="Historical period for context window")

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

@app.get("/api/tasks")
async def get_tasks():
    """Reads TASKS.txt for real-time agent coordination."""
    if not os.path.exists(TASKS_FILE):
        return {"content": "", "raw": "TASKS.txt not found."}
    
    with open(TASKS_FILE, "r", encoding="utf-8") as f:
        content = f.read()

    return {"content": content}

@app.post("/api/tasks/create")
async def create_task(req: NewTaskRequest):
    """Allows testers and PMs to append a task into TASKS.txt."""
    if not os.path.exists(TASKS_FILE):
        raise HTTPException(status_code=404, detail="TASKS.txt not found")

    import datetime
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    task_id = req.task_id or f"TASK-{int(datetime.datetime.now().timestamp()) % 10000:04d}"
    entry_line = f"[{task_id}] | {now_str} | {req.from_role} -> {req.to_role} | PENDING | {req.priority} | {req.description}\n"

    with open(TASKS_FILE, "r", encoding="utf-8") as f:
        text = f.read()

    # Append under ACTIVE TASK QUEUE
    marker = "2. ACTIVE TASK QUEUE\n--------------------------------------------------------------------------------\n[Format: Task ID | Date | From -> To | Status | Priority | Description]\n"
    if marker in text:
        parts = text.split(marker)
        new_text = parts[0] + marker + entry_line + parts[1]
    else:
        new_text = text + f"\n\n[NEW TASK - {now_str}]\n" + entry_line

    with open(TASKS_FILE, "w", encoding="utf-8") as f:
        f.write(new_text)

    return {"success": True, "task_id": task_id, "message": "Task queued successfully."}

@app.post("/api/tasks/message")
async def post_message(req: ChatMessageRequest):
    """Allows agents to post a communication log message into TASKS.txt."""
    if not os.path.exists(TASKS_FILE):
        raise HTTPException(status_code=404, detail="TASKS.txt not found")

    import datetime
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"\n[{now_str}] [{req.author}]:\n{req.message}\n"

    with open(TASKS_FILE, "r", encoding="utf-8") as f:
        text = f.read()

    # Append under section 4
    marker = "4. AGENT COMMUNICATION LOG\n--------------------------------------------------------------------------------"
    if marker in text:
        parts = text.split(marker)
        # Find next section or append
        new_text = parts[0] + marker + entry + parts[1]
    else:
        new_text = text + entry

    with open(TASKS_FILE, "w", encoding="utf-8") as f:
        f.write(new_text)

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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
