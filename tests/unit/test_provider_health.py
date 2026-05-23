"""Tests for provider health checker."""

from __future__ import annotations

import pytest

from llmstack.gateway.provider_health import (
    HealthCheckConfig,
    HealthStatus,
    ProviderHealthChecker,
)


@pytest.fixture
def checker():
    return ProviderHealthChecker()


class TestProviderHealthChecker:
    def test_initial_status_unknown(self, checker):
        assert checker.get_status("openai") == HealthStatus.UNKNOWN

    def test_success_marks_healthy(self, checker):
        checker.record_success("openai", latency_ms=100)
        assert checker.get_status("openai") == HealthStatus.HEALTHY

    def test_failure_marks_degraded(self, checker):
        config = HealthCheckConfig(degraded_threshold=1, failure_threshold=3)
        chk = ProviderHealthChecker(config)
        chk.record_failure("openai", error="timeout")
        assert chk.get_status("openai") == HealthStatus.DEGRADED

    def test_multiple_failures_unhealthy(self):
        config = HealthCheckConfig(failure_threshold=2)
        chk = ProviderHealthChecker(config)
        chk.record_failure("openai")
        chk.record_failure("openai")
        assert chk.get_status("openai") == HealthStatus.UNHEALTHY

    def test_success_resets_failures(self, checker):
        checker.record_failure("openai")
        checker.record_failure("openai")
        checker.record_success("openai", latency_ms=50)
        assert checker.get_status("openai") == HealthStatus.HEALTHY

    def test_high_latency_degraded(self):
        config = HealthCheckConfig(latency_threshold_ms=100)
        chk = ProviderHealthChecker(config)
        chk.record_success("openai", latency_ms=500)
        assert chk.get_status("openai") == HealthStatus.DEGRADED

    def test_get_health_single(self, checker):
        checker.record_success("openai", latency_ms=50)
        health = checker.get_health("openai")
        assert health["provider"] == "openai"
        assert health["status"] == "healthy"

    def test_get_health_all(self, checker):
        checker.record_success("openai")
        checker.record_failure("anthropic")
        health = checker.get_health()
        assert health["total_providers"] == 2
        assert "providers" in health

    def test_get_healthy_providers(self, checker):
        checker.record_success("openai")
        checker.record_success("google")
        config = HealthCheckConfig(failure_threshold=1)
        chk = ProviderHealthChecker(config)
        chk.record_success("openai")
        chk.record_failure("anthropic")
        healthy = chk.get_healthy_providers()
        assert "openai" in healthy

    def test_success_rate(self, checker):
        checker.record_success("openai")
        checker.record_success("openai")
        checker.record_failure("openai")
        health = checker.get_health("openai")
        assert abs(health["success_rate"] - 0.6667) < 0.01

    def test_unknown_provider_health(self, checker):
        health = checker.get_health("nonexistent")
        assert health["status"] == "unknown"
