"""Metrics middleware — tracks request counts, latencies, tokens in Prometheus format."""

from __future__ import annotations

import time
from collections import defaultdict
from threading import Lock

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

_lock = Lock()
_request_count: dict[str, int] = defaultdict(int)
_error_count: dict[str, int] = defaultdict(int)
_tokens_in: int = 0
_tokens_out: int = 0
_active_requests: int = 0

# Per-model token tracking
_tokens_by_model: dict[str, dict[str, int]] = defaultdict(lambda: {"input": 0, "output": 0})
_cost_by_model: dict[str, float] = defaultdict(float)
_cache_hits: int = 0
_cache_misses: int = 0

# Per-model request counts
_model_request_count: dict[str, int] = defaultdict(int)

# Histogram buckets for latency
_BUCKETS = [0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0]
_latency_buckets: dict[str, list[int]] = defaultdict(lambda: [0] * (len(_BUCKETS) + 1))
_latency_sum: dict[str, float] = defaultdict(float)
_latency_count: dict[str, int] = defaultdict(int)


class MetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):  # noqa: ANN001
        global _active_requests
        path = request.url.path
        if path in ("/metrics", "/healthz"):
            return await call_next(request)

        with _lock:
            _active_requests += 1
        start = time.monotonic()
        response = None
        try:
            response = await call_next(request)
            return response
        finally:
            duration = time.monotonic() - start
            with _lock:
                _active_requests -= 1
                _request_count[path] += 1
                _latency_sum[path] += duration
                _latency_count[path] += 1

                # Bucket assignment
                buckets = _latency_buckets[path]
                for i, bound in enumerate(_BUCKETS):
                    if duration <= bound:
                        buckets[i] += 1
                buckets[-1] += 1  # +Inf

                if response is not None and response.status_code >= 400:
                    _error_count[path] += 1


def get_active_requests() -> int:
    """Return the number of currently in-flight requests."""
    with _lock:
        return _active_requests


def record_tokens(input_tokens: int = 0, output_tokens: int = 0, model: str = "") -> None:
    """Record token usage from a chat completion response."""
    global _tokens_in, _tokens_out
    with _lock:
        _tokens_in += input_tokens
        _tokens_out += output_tokens
        if model:
            _tokens_by_model[model]["input"] += input_tokens
            _tokens_by_model[model]["output"] += output_tokens


def record_cost(model: str, cost_usd: float) -> None:
    """Record cost incurred for a model invocation."""
    with _lock:
        _cost_by_model[model] += cost_usd


def record_cache(hit: bool) -> None:
    """Record a cache hit or miss."""
    global _cache_hits, _cache_misses
    with _lock:
        if hit:
            _cache_hits += 1
        else:
            _cache_misses += 1


def record_model_request(model: str) -> None:
    """Record a request served by a specific model."""
    with _lock:
        _model_request_count[model] += 1


def get_metrics() -> dict:
    """Return metrics as JSON (for /metrics JSON endpoint)."""
    with _lock:
        result: dict = {}
        for path in _request_count:
            result[path] = {
                "requests": _request_count[path],
                "errors": _error_count.get(path, 0),
                "latency_avg_ms": round((_latency_sum[path] / _latency_count[path]) * 1000, 1)
                if _latency_count[path]
                else 0,
            }
        result["tokens"] = {"input": _tokens_in, "output": _tokens_out}
        result["tokens_by_model"] = {m: dict(v) for m, v in _tokens_by_model.items()}
        result["cost_by_model"] = dict(_cost_by_model)
        result["cache"] = {"hits": _cache_hits, "misses": _cache_misses}
        result["active_requests"] = _active_requests
        result["model_requests"] = dict(_model_request_count)
        return result


def get_prometheus_metrics() -> str:
    """Return metrics in Prometheus exposition format."""
    lines: list[str] = []

    with _lock:
        # Request counter
        lines.append("# HELP llmstack_requests_total Total HTTP requests")
        lines.append("# TYPE llmstack_requests_total counter")
        for path, count in _request_count.items():
            lines.append(f'llmstack_requests_total{{path="{path}"}} {count}')

        # Error counter
        lines.append("# HELP llmstack_errors_total Total HTTP errors (4xx/5xx)")
        lines.append("# TYPE llmstack_errors_total counter")
        for path, count in _error_count.items():
            lines.append(f'llmstack_errors_total{{path="{path}"}} {count}')

        # Latency histogram
        lines.append("# HELP llmstack_request_duration_seconds Request latency histogram")
        lines.append("# TYPE llmstack_request_duration_seconds histogram")
        for path in _latency_count:
            buckets = _latency_buckets[path]
            cumulative = 0
            for i, bound in enumerate(_BUCKETS):
                cumulative += buckets[i]
                lines.append(
                    f"llmstack_request_duration_seconds_bucket"
                    f'{{path="{path}",le="{bound}"}} {cumulative}'
                )
            cumulative += buckets[-1]
            lines.append(
                f'llmstack_request_duration_seconds_bucket{{path="{path}",le="+Inf"}} {cumulative}'
            )
            lines.append(
                f'llmstack_request_duration_seconds_sum{{path="{path}"}} {_latency_sum[path]:.4f}'
            )
            lines.append(
                f'llmstack_request_duration_seconds_count{{path="{path}"}} {_latency_count[path]}'
            )

        # Token counter (aggregate)
        lines.append("# HELP llmstack_tokens_total Total tokens processed")
        lines.append("# TYPE llmstack_tokens_total counter")
        lines.append(f'llmstack_tokens_total{{type="input"}} {_tokens_in}')
        lines.append(f'llmstack_tokens_total{{type="output"}} {_tokens_out}')

        # Per-model token counter
        lines.append("# HELP llmstack_model_tokens_total Tokens processed per model")
        lines.append("# TYPE llmstack_model_tokens_total counter")
        for model, counts in _tokens_by_model.items():
            lines.append(
                f'llmstack_model_tokens_total{{model="{model}",type="input"}} {counts["input"]}'
            )
            lines.append(
                f'llmstack_model_tokens_total{{model="{model}",type="output"}} {counts["output"]}'
            )

        # Active requests gauge
        lines.append("# HELP llmstack_active_requests Current in-flight requests")
        lines.append("# TYPE llmstack_active_requests gauge")
        lines.append(f"llmstack_active_requests {_active_requests}")

        # Cache hit/miss counter
        lines.append("# HELP llmstack_cache_total Cache hit and miss counts")
        lines.append("# TYPE llmstack_cache_total counter")
        lines.append(f'llmstack_cache_total{{result="hit"}} {_cache_hits}')
        lines.append(f'llmstack_cache_total{{result="miss"}} {_cache_misses}')

        # Cost per model counter
        lines.append("# HELP llmstack_cost_usd_total Cumulative cost in USD per model")
        lines.append("# TYPE llmstack_cost_usd_total counter")
        for model, cost in _cost_by_model.items():
            lines.append(f'llmstack_cost_usd_total{{model="{model}"}} {cost:.6f}')

        # Per-model request breakdown
        lines.append("# HELP llmstack_model_requests_total Requests per model")
        lines.append("# TYPE llmstack_model_requests_total counter")
        for model, count in _model_request_count.items():
            lines.append(f'llmstack_model_requests_total{{model="{model}"}} {count}')

    lines.append("")
    return "\n".join(lines)
