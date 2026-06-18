"""Tests for circuit breaker pattern."""

from __future__ import annotations

import pytest

from llmstack.gateway.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerError,
    CircuitState,
    get_inference_breaker,
)


class TestCircuitBreakerStates:
    """Test state transitions: CLOSED → OPEN → HALF_OPEN → CLOSED."""

    def test_starts_closed(self):
        cb = CircuitBreaker(failure_threshold=3)
        assert cb.state == CircuitState.CLOSED

    def test_stays_closed_under_threshold(self):
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED

    def test_opens_on_threshold(self):
        cb = CircuitBreaker(failure_threshold=3)
        for _ in range(3):
            cb.record_failure()
        assert cb.state == CircuitState.OPEN

    def test_open_rejects_requests(self):
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=60.0)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        with pytest.raises(CircuitBreakerError) as exc_info:
            cb.check()
        assert exc_info.value.retry_after > 0

    def test_success_resets_failure_count(self):
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        # Failure count reset, need 3 more to open
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED

    def test_half_open_success_closes(self):
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.0)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        # Recovery timeout = 0, so check() transitions to half-open
        cb.check()
        assert cb.state == CircuitState.HALF_OPEN
        cb.record_success()
        assert cb.state == CircuitState.CLOSED

    def test_half_open_failure_reopens(self):
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.0)
        cb.record_failure()
        cb.record_failure()
        cb.check()  # transitions to half-open
        assert cb.state == CircuitState.HALF_OPEN
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

    def test_manual_reset(self):
        cb = CircuitBreaker(failure_threshold=1)
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        cb.reset()
        assert cb.state == CircuitState.CLOSED
        cb.check()  # should not raise

    def test_half_open_rejects_when_max_calls_reached(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.0, half_open_max_calls=1)
        cb.record_failure()
        cb.check()  # OPEN -> HALF_OPEN transition (does not consume a call slot)
        assert cb.state == CircuitState.HALF_OPEN
        cb.check()  # consumes the one allowed half-open call
        with pytest.raises(CircuitBreakerError) as exc_info:
            cb.check()  # third call while still half-open should be rejected
        assert exc_info.value.retry_after == 5.0


class TestCircuitBreakerProperties:
    def test_is_open_property(self):
        cb = CircuitBreaker(failure_threshold=1)
        assert cb.is_open is False
        cb.record_failure()
        assert cb.is_open is True

    def test_is_healthy_property(self):
        cb = CircuitBreaker(failure_threshold=1)
        assert cb.is_healthy is True
        cb.record_failure()
        assert cb.is_healthy is False

    def test_success_rate_no_requests(self):
        cb = CircuitBreaker()
        assert cb.success_rate == 1.0

    def test_success_rate_with_requests(self):
        cb = CircuitBreaker(failure_threshold=10)
        cb.record_success()
        cb.record_success()
        cb.record_failure()
        assert cb.success_rate == pytest.approx(2 / 3)

    def test_total_requests(self):
        cb = CircuitBreaker(failure_threshold=10, recovery_timeout=999.0)
        cb.record_success()
        cb.record_failure()
        cb._state = CircuitState.OPEN
        try:
            cb.check()
        except CircuitBreakerError:
            pass
        assert cb.total_requests == 3


class TestInferenceBreakerSingleton:
    def test_returns_same_instance(self):
        import llmstack.gateway.circuit_breaker as module

        module._inference_breaker = None
        first = get_inference_breaker()
        second = get_inference_breaker()
        assert first is second
        module._inference_breaker = None


class TestExponentialBackoff:
    """Test that recovery timeout increases with consecutive opens."""

    def test_backoff_increases(self):
        cb = CircuitBreaker(
            failure_threshold=1,
            recovery_timeout=10.0,
            backoff_multiplier=2.0,
            max_recovery_timeout=300.0,
        )
        # First open
        cb.record_failure()
        assert cb.current_recovery_timeout == 20.0  # 10 * 2^1

        # Reset and open again
        cb._state = CircuitState.HALF_OPEN
        cb.record_failure()
        assert cb.current_recovery_timeout == 40.0  # 10 * 2^2

    def test_backoff_capped(self):
        cb = CircuitBreaker(
            failure_threshold=1,
            recovery_timeout=100.0,
            backoff_multiplier=3.0,
            max_recovery_timeout=300.0,
        )
        cb._consecutive_opens = 10  # Would be 100 * 3^10 without cap
        assert cb.current_recovery_timeout == 300.0


class TestCircuitBreakerMetrics:
    """Test metrics reporting."""

    def test_metrics_structure(self):
        cb = CircuitBreaker()
        cb.record_success()
        cb.record_failure()
        m = cb.metrics()
        assert "state" in m
        assert "total_successes" in m
        assert "total_failures" in m
        assert "total_rejections" in m
        assert m["total_successes"] == 1
        assert m["total_failures"] == 1

    def test_rejection_counting(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=999.0)
        cb.record_failure()
        try:
            cb.check()
        except CircuitBreakerError:
            pass
        try:
            cb.check()
        except CircuitBreakerError:
            pass
        assert cb.metrics()["total_rejections"] == 2
