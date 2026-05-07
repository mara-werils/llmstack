"""Tests for gateway middleware."""

from llmstack.gateway.middleware.metrics import (
    get_metrics,
    get_prometheus_metrics,
    record_tokens,
    _request_count,
    _error_count,
    _latency_sum,
    _latency_count,
    _latency_buckets,
)


def _reset_metrics():
    """Reset global metrics state for test isolation."""
    _request_count.clear()
    _error_count.clear()
    _latency_sum.clear()
    _latency_count.clear()
    _latency_buckets.clear()
    import llmstack.gateway.middleware.metrics as m
    m._tokens_in = 0
    m._tokens_out = 0


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
    assert '/v1/chat/completions' in output


def test_prometheus_token_counter():
    _reset_metrics()
    record_tokens(input_tokens=42, output_tokens=13)
    output = get_prometheus_metrics()
    assert 'llmstack_tokens_total{type="input"} 42' in output
    assert 'llmstack_tokens_total{type="output"} 13' in output
