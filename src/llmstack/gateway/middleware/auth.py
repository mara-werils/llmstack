"""API key authentication middleware."""

from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse


class AuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, api_keys: list[str]):
        super().__init__(app)
        self.api_keys = set(k.strip() for k in api_keys if k.strip())

    async def dispatch(self, request: Request, call_next):
        # Skip auth for health checks and docs
        if request.url.path in ("/healthz", "/metrics", "/docs", "/openapi.json"):
            return await call_next(request)

        if not self.api_keys:
            return await call_next(request)

        auth = request.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            token = auth[7:]
            if token in self.api_keys:
                return await call_next(request)

        return JSONResponse(
            status_code=401,
            content={"error": {"message": "Invalid API key", "type": "auth_error"}},
        )
