"""Request size limit middleware — reject oversized payloads early."""

from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

# Default: 10MB
DEFAULT_MAX_REQUEST_BYTES = 10 * 1024 * 1024


class RequestSizeMiddleware(BaseHTTPMiddleware):
    """Reject requests exceeding a configurable body size limit."""

    def __init__(self, app, max_bytes: int = DEFAULT_MAX_REQUEST_BYTES) -> None:
        super().__init__(app)
        self.max_bytes = max_bytes

    async def dispatch(self, request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > self.max_bytes:
            return JSONResponse(
                status_code=413,
                content={
                    "error": {
                        "message": f"Request body too large. Maximum: {self.max_bytes} bytes",
                        "type": "payload_too_large",
                    }
                },
            )
        return await call_next(request)
