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
    """ASGI wrapper that normalizes rewritten paths using Vercel edge headers."""

    def __init__(self, inner_app):
        self.inner_app = inner_app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            headers = dict(scope.get("headers", []))
            # Check for original request path from Vercel edge proxy
            matched = headers.get(b"x-matched-path", b"").decode("utf-8")
            if not matched:
                matched = headers.get(b"x-vercel-matched-path", b"").decode("utf-8")

            if matched:
                # Remove query parameters if present in matched path
                scope["path"] = matched.split("?")[0]
            else:
                path = scope.get("path", "")
                if path in ("/api/index.py", "/api/index.py/"):
                    scope["path"] = "/"
                elif path.startswith("/api/index.py/"):
                    scope["path"] = path[len("/api/index.py"):]

        await self.inner_app(scope, receive, send)


app = VercelASGIApp(fastapi_app)
