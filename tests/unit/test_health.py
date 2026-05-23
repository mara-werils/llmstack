"""Tests for health endpoint utilities."""

from __future__ import annotations


from llmstack.gateway.routes.health import _START_TIME


def test_start_time_is_set():
    """Verify that _START_TIME is initialized at import time."""
    assert _START_TIME > 0


def test_request_size_middleware():
    """Test the request size limit middleware logic."""
    from llmstack.gateway.middleware.request_size import DEFAULT_MAX_REQUEST_BYTES

    assert DEFAULT_MAX_REQUEST_BYTES == 10 * 1024 * 1024  # 10MB


class TestHealthImports:
    """Verify health-related modules import cleanly."""

    def test_health_routes_importable(self):
        from llmstack.gateway.routes.health import healthz, liveness, readiness, ping, metrics
        assert callable(healthz)
        assert callable(liveness)
        assert callable(readiness)
        assert callable(ping)
        assert callable(metrics)

    def test_observe_routes_importable(self):
        from llmstack.gateway.routes.observe import (
            list_traces, traces_summary, quality_summary, observe_stats,
        )
        assert callable(list_traces)
        assert callable(traces_summary)
        assert callable(quality_summary)
        assert callable(observe_stats)
