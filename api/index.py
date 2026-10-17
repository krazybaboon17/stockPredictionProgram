"""
Vercel Serverless Function entrypoint for TimesFM Stock Prediction Platform.
Routes all requests to the FastAPI application instance and normalizes paths.
"""

import os
import sys

# Ensure root directory is in sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from server.main import app as fastapi_app


class VercelASGIApp:
    """ASGI wrapper that normalizes rewritten paths for Vercel Serverless Functions."""

    def __init__(self, inner_app):
        self.inner_app = inner_app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            path = scope.get("path", "")
            if path in ("/api/index.py", "/api/index.py/"):
                scope["path"] = "/"
            elif path.startswith("/api/index.py/"):
                scope["path"] = path[len("/api/index.py"):]
        await self.inner_app(scope, receive, send)


app = VercelASGIApp(fastapi_app)
