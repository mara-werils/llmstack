"""Structured request logging middleware with correlation IDs.

Adds X-Request-ID to every request/response and logs structured JSON
for each request including method, path, status, duration, and tokens.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request


LOG_LEVEL = os.getenv("LLMSTACK_LOG_LEVEL", "INFO").upper()
LOG_FORMAT = os.getenv("LLMSTACK_LOG_FORMAT", "json")  # json | text

# Configure structured logger
logger = logging.getLogger("llmstack.gateway")


def _setup_logger() -> None:
    """Configure the gateway logger with structured JSON output."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

    if LOG_FORMAT == "json":
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s [%(request_id)s] %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        ))

    logger.addHandler(handler)
    logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    logger.propagate = False


class _JsonFormatter(logging.Formatter):
    """Emit log records as single-line JSON."""

    def format(self, record: logging.LogRecord) -> str:
        log_dict = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "msg": record.getMessage(),
        }
        # Merge extra fields
        for key in (
            "request_id", "method", "path", "status", "duration_ms",
            "client_ip", "tokens_in", "tokens_out", "cache_hit", "model",
            "provider", "cost_usd", "user_agent", "content_length",
            "correlation_id", "error", "tier",
        ):
            val = getattr(record, key, None)
            if val is not None:
                log_dict[key] = val

        return json.dumps(log_dict, default=str)


class LoggingMiddleware(BaseHTTPMiddleware):
    """Adds correlation ID and structured logging to every request."""

    SKIP_PATHS = {"/metrics", "/healthz"}

    def __init__(self, app):
        super().__init__(app)
        _setup_logger()

    async def dispatch(self, request: Request, call_next):
        if request.url.path in self.SKIP_PATHS:
            return await call_next(request)

        # Generate or reuse correlation ID
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4())[:8])

        # Store in request state for downstream use
        request.state.request_id = request_id

        start = time.monotonic()
        response = await call_next(request)
        duration_ms = round((time.monotonic() - start) * 1000, 1)

        # Add correlation headers
        response.headers["X-Request-ID"] = request_id

        # Extract client IP
        client = request.client
        client_ip = client.host if client else "unknown"
        forwarded = request.headers.get("X-Forwarded-For", "")
        if forwarded:
            client_ip = forwarded.split(",")[0].strip()

        # Log the request
        logger.info(
            "%s %s %d %.1fms",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "duration_ms": duration_ms,
                "client_ip": client_ip,
                "user_agent": request.headers.get("User-Agent", ""),
                "content_length": request.headers.get(
                    "Content-Length", ""
                ),
            },
        )

        return response
