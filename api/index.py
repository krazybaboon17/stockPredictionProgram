"""
Vercel Serverless Function entrypoint for TimesFM Stock Prediction Platform.
Routes all /api/* requests to the FastAPI application instance.
"""

import os
import sys

# Ensure root directory is in sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from server.main import app
