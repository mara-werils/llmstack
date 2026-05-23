"""Request correlation ID middleware.

Assigns a unique correlation ID to each request for distributed tracing.
Propagates existing IDs from upstream services via X-Request-ID header.
"""

from __future__ import annotations

import uuid
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

CORRELATION_HEADER = "X-Request-ID"


class CorrelationMiddleware(BaseHTTPMiddleware):
    """Adds correlation IDs to every request/response for tracing."""

    async def dispatch(self, request: Request, call_next) -> Response:
        # Use existing header or generate new ID
        correlation_id = request.headers.get(CORRELATION_HEADER) or str(uuid.uuid4())

        # Store in request state for downstream use
        request.state.correlation_id = correlation_id

        response = await call_next(request)

        # Add to response headers
        response.headers[CORRELATION_HEADER] = correlation_id

        return response


def get_correlation_id(request: Request) -> str:
    """Get the correlation ID from the current request."""
    return getattr(request.state, "correlation_id", "unknown")
