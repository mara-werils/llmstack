"""Simple metrics middleware — tracks request counts and latencies."""

from __future__ import annotations

import time
from collections import defaultdict

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

_request_count: dict[str, int] = defaultdict(int)
_request_latency: dict[str, list[float]] = defaultdict(list)
_error_count: dict[str, int] = defaultdict(int)


class MetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        start = time.monotonic()

        response = await call_next(request)

        duration = time.monotonic() - start
        _request_count[path] += 1
        _request_latency[path].append(duration)
        if response.status_code >= 400:
            _error_count[path] += 1

        return response


def get_metrics() -> dict:
    """Return current metrics as a dict."""
    result = {}
    for path in _request_count:
        latencies = _request_latency.get(path, [])
        sorted_lat = sorted(latencies)
        p50 = sorted_lat[len(sorted_lat) // 2] if sorted_lat else 0
        p99 = sorted_lat[int(len(sorted_lat) * 0.99)] if sorted_lat else 0

        result[path] = {
            "requests": _request_count[path],
            "errors": _error_count.get(path, 0),
            "latency_p50_ms": round(p50 * 1000, 1),
            "latency_p99_ms": round(p99 * 1000, 1),
        }
    return result
