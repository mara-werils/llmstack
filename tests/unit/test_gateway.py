"""Tests for gateway middleware."""

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from llmstack.gateway.middleware.metrics import (
    MetricsMiddleware,
    get_active_requests,
    get_metrics,
    get_prometheus_metrics,
    record_cache,
    record_cost,
    record_model_request,
    record_tokens,
    _cost_by_model,
    _error_count,
    _latency_buckets,
    _latency_count,
    _latency_sum,
    _model_request_count,
    _request_count,
    _tokens_by_model,
)


def _reset_metrics():
    """Reset global metrics state for test isolation."""
    _request_count.clear()
    _error_count.clear()
    _latency_sum.clear()
    _latency_count.clear()
    _latency_buckets.clear()
    _tokens_by_model.clear()
    _cost_by_model.clear()
    _model_request_count.clear()
    import llmstack.gateway.middleware.metrics as m

    m._tokens_in = 0
    m._tokens_out = 0
    m._cache_hits = 0
    m._cache_misses = 0
    m._active_requests = 0


def test_get_metrics_empty():
    _reset_metrics()
    result = get_metrics()
    assert result["tokens"]["input"] == 0
    assert result["tokens"]["output"] == 0


def test_record_tokens():
    _reset_metrics()
    record_tokens(input_tokens=100, output_tokens=50)
    record_tokens(input_tokens=200, output_tokens=150)
    result = get_metrics()
    assert result["tokens"]["input"] == 300
    assert result["tokens"]["output"] == 200


def test_prometheus_format():
    _reset_metrics()
    _request_count["/v1/chat/completions"] = 10
    _error_count["/v1/chat/completions"] = 1
    _latency_sum["/v1/chat/completions"] = 5.0
    _latency_count["/v1/chat/completions"] = 10

    output = get_prometheus_metrics()
    assert "llmstack_requests_total" in output
    assert "llmstack_errors_total" in output
    assert "llmstack_request_duration_seconds" in output
    assert "llmstack_tokens_total" in output
    assert "/v1/chat/completions" in output


def test_prometheus_token_counter():
    _reset_metrics()
    record_tokens(input_tokens=42, output_tokens=13)
    output = get_prometheus_metrics()
    assert 'llmstack_tokens_total{type="input"} 42' in output
    assert 'llmstack_tokens_total{type="output"} 13' in output


def test_record_tokens_per_model():
    _reset_metrics()
    record_tokens(input_tokens=10, output_tokens=5, model="llama3.2")
    result = get_metrics()
    assert result["tokens_by_model"]["llama3.2"] == {"input": 10, "output": 5}


def test_record_cost():
    _reset_metrics()
    record_cost("llama3.2", 0.0012)
    record_cost("llama3.2", 0.0008)
    result = get_metrics()
    assert result["cost_by_model"]["llama3.2"] == pytest.approx(0.002)


def test_record_cache_hit_and_miss():
    _reset_metrics()
    record_cache(hit=True)
    record_cache(hit=True)
    record_cache(hit=False)
    result = get_metrics()
    assert result["cache"] == {"hits": 2, "misses": 1}


def test_record_model_request():
    _reset_metrics()
    record_model_request("llama3.2")
    record_model_request("llama3.2")
    record_model_request("gpt-4o")
    result = get_metrics()
    assert result["model_requests"] == {"llama3.2": 2, "gpt-4o": 1}


def test_get_active_requests_default_zero():
    _reset_metrics()
    assert get_active_requests() == 0


def test_get_metrics_includes_per_path_stats():
    _reset_metrics()
    _request_count["/v1/chat/completions"] = 5
    _error_count["/v1/chat/completions"] = 1
    _latency_sum["/v1/chat/completions"] = 2.5
    _latency_count["/v1/chat/completions"] = 5

    result = get_metrics()

    assert result["/v1/chat/completions"]["requests"] == 5
    assert result["/v1/chat/completions"]["errors"] == 1
    assert result["/v1/chat/completions"]["latency_avg_ms"] == 500.0


def test_prometheus_per_model_tokens_cost_and_requests():
    _reset_metrics()
    record_tokens(input_tokens=10, output_tokens=5, model="llama3.2")
    record_cost("llama3.2", 0.005)
    record_model_request("llama3.2")

    output = get_prometheus_metrics()

    assert 'llmstack_model_tokens_total{model="llama3.2",type="input"} 10' in output
    assert 'llmstack_model_tokens_total{model="llama3.2",type="output"} 5' in output
    assert 'llmstack_cost_usd_total{model="llama3.2"} 0.005000' in output
    assert 'llmstack_model_requests_total{model="llama3.2"} 1' in output


def _app():
    app = FastAPI()
    app.add_middleware(MetricsMiddleware)

    @app.get("/v1/ok")
    async def ok():
        return {"ok": True}

    @app.get("/v1/fails")
    async def fails():
        from starlette.responses import JSONResponse

        return JSONResponse(status_code=500, content={"error": "boom"})

    @app.get("/healthz")
    async def health():
        return {"ok": True}

    return app


class TestMetricsMiddlewareDispatch:
    def test_skips_metrics_and_healthz_paths(self):
        _reset_metrics()
        client = TestClient(_app())
        client.get("/healthz")
        assert "/healthz" not in _request_count

    def test_records_successful_request(self):
        _reset_metrics()
        client = TestClient(_app())
        client.get("/v1/ok")
        assert _request_count["/v1/ok"] == 1
        assert _error_count.get("/v1/ok", 0) == 0

    def test_records_error_response(self):
        _reset_metrics()
        client = TestClient(_app())
        client.get("/v1/fails")
        assert _request_count["/v1/fails"] == 1
        assert _error_count["/v1/fails"] == 1
