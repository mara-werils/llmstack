"""API key authentication middleware."""

from __future__ import annotations

import hashlib
import hmac

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse


def _hash_key(key: str) -> str:
    """Return the hex SHA-256 digest of an API key."""
    return hashlib.sha256(key.encode()).hexdigest()


class AuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, api_keys: list[str]):
        super().__init__(app)
        # Store mapping of hash -> identity (first 8 chars) for audit logging
        self._key_map: dict[str, str] = {}
        for k in api_keys:
            k = k.strip()
            if k:
                self._key_map[_hash_key(k)] = k[:8]

    async def dispatch(self, request: Request, call_next):
        # Skip auth for health checks, docs, and UI
        skip_paths = (
            "/healthz",
            "/metrics",
            "/docs",
            "/redoc",
            "/openapi.json",
            "/",
        )
        if request.url.path in skip_paths or request.url.path.startswith("/ui"):
            return await call_next(request)

        if not self._key_map:
            return await call_next(request)

        auth = request.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            token = auth[7:]
            token_hash = _hash_key(token)
            for stored_hash, identity in self._key_map.items():
                if hmac.compare_digest(token_hash, stored_hash):
                    request.state.api_key_identity = identity
                    return await call_next(request)

        return JSONResponse(
            status_code=401,
            content={
                "error": {
                    "message": "Invalid API key",
                    "type": "auth_error",
                }
            },
        )
