#!/usr/bin/env bash
# Startup script for TimesFM Stock Prediction Platform
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================================="
echo "    Launching TimesFM Stock Prediction Platform           "
echo "=========================================================="

if [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
fi

PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"

echo "Starting FastAPI server on http://localhost:${PORT}"
exec python3 -m uvicorn server.main:app --host "$HOST" --port "$PORT" --reload
