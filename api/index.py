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


import urllib.parse


class VercelASGIApp:
    """ASGI wrapper that normalizes rewritten paths for Vercel Serverless Functions."""

    def __init__(self, inner_app):
        self.inner_app = inner_app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            query_bytes = scope.get("query_string", b"")
            query_str = query_bytes.decode("utf-8", errors="ignore")
            params = urllib.parse.parse_qs(query_str)

            if "__path" in params and params["__path"]:
                orig_path = params["__path"][0]
                scope["path"] = "/" + orig_path.lstrip("/")
                # Clean __path query param from query_string
                clean_params = {k: v for k, v in params.items() if k != "__path"}
                scope["query_string"] = urllib.parse.urlencode(clean_params, doseq=True).encode("utf-8")
            else:
                headers = dict(scope.get("headers", []))
                matched = headers.get(b"x-matched-path", b"").decode("utf-8", errors="ignore")
                if matched and not matched.startswith("/api/index.py"):
                    scope["path"] = matched.split("?")[0]
                else:
                    path = scope.get("path", "")
                    if path in ("/api/index.py", "/api/index.py/"):
                        scope["path"] = "/"
                    elif path.startswith("/api/index.py/"):
                        scope["path"] = path[len("/api/index.py"):]

        await self.inner_app(scope, receive, send)


app = VercelASGIApp(fastapi_app)
