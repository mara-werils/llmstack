"""Request correlation ID middleware.

Assigns a unique correlation ID (UUID-4) to each inbound request for
distributed tracing and structured logging.  If the client already
provides an ``X-Request-ID`` header the value is reused, which allows
correlation across reverse-proxies and upstream services.

The ID is:
* stored in ``request.state.correlation_id`` for route handlers,
* echoed back in the ``X-Request-ID`` response header,
* injected into the Python logging context so that every structured log
  line automatically includes the request ID.
"""

from __future__ import annotations

import contextvars
import logging
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

CORRELATION_HEADER = "X-Request-ID"

# Context-var for use outside of request-state (e.g. background tasks).
_current_request_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id",
    default="unknown",
)


class _RequestIDLogFilter(logging.Filter):
    """Inject the current request ID into every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = _current_request_id.get("unknown")  # type: ignore[attr-defined]
        return True


# Install the filter on the root gateway logger so all child loggers
# automatically get the request_id field.
_filter_installed = False


def _ensure_log_filter() -> None:
    global _filter_installed  # noqa: PLW0603
    if _filter_installed:
        return
    gateway_logger = logging.getLogger("llmstack.gateway")
    gateway_logger.addFilter(_RequestIDLogFilter())
    _filter_installed = True


class CorrelationMiddleware(BaseHTTPMiddleware):
    """Adds correlation IDs to every request/response for tracing."""

    def __init__(self, app) -> None:  # type: ignore[override]
        super().__init__(app)
        _ensure_log_filter()

    async def dispatch(self, request: Request, call_next) -> Response:
        # Use existing header or generate new ID
        correlation_id = request.headers.get(CORRELATION_HEADER) or str(uuid.uuid4())

        # Store in request state for downstream use
        request.state.correlation_id = correlation_id

        # Store in context-var so structured logs include the ID
        token = _current_request_id.set(correlation_id)
        try:
            response = await call_next(request)
        finally:
            _current_request_id.reset(token)

        # Add to response headers
        response.headers[CORRELATION_HEADER] = correlation_id

        return response


def get_correlation_id(request: Request | None = None) -> str:
    """Get the correlation ID from the current request or context.

    When called from a route handler, pass the ``request`` object.
    When called from background work (no request), falls back to the
    context-var set by the middleware.
    """
    if request is not None:
        return getattr(request.state, "correlation_id", "unknown")
    return _current_request_id.get("unknown")
