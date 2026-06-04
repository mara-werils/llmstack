"""Circuit breaker for inference backend resilience.

Implements the three-state circuit breaker pattern:
  CLOSED → OPEN → HALF_OPEN → CLOSED

Prevents cascading failures when the inference backend is down
by failing fast instead of timing out on every request.
"""

from __future__ import annotations

import asyncio
import time
from enum import Enum


class CircuitState(str, Enum):
    CLOSED = "closed"  # Normal operation — requests flow through
    OPEN = "open"  # Backend is down — fail fast
    HALF_OPEN = "half_open"  # Probing — allow one request to test recovery


class CircuitBreakerError(Exception):
    """Raised when the circuit is open and requests are rejected."""

    def __init__(self, retry_after: float):
        self.retry_after = retry_after
        super().__init__(f"Circuit breaker is open. Retry after {retry_after:.0f}s")


class CircuitBreaker:
    """Circuit breaker with exponential backoff and half-open probing.

    Usage:
        breaker = CircuitBreaker()

        async def call_inference(payload):
            breaker.check()  # raises CircuitBreakerError if open
            try:
                result = await make_request(payload)
                breaker.record_success()
                return result
            except Exception as e:
                breaker.record_failure()
                raise
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_max_calls: int = 1,
        max_recovery_timeout: float = 300.0,
        backoff_multiplier: float = 2.0,
    ):
        self.failure_threshold = failure_threshold
        self.base_recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        self.max_recovery_timeout = max_recovery_timeout
        self.backoff_multiplier = backoff_multiplier

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = 0.0
        self._half_open_calls = 0
        self._consecutive_opens = 0
        self._lock = asyncio.Lock()

        # Metrics
        self._total_failures = 0
        self._total_successes = 0
        self._total_rejections = 0
        self._last_state_change = time.monotonic()

    @property
    def state(self) -> CircuitState:
        return self._state

    @property
    def current_recovery_timeout(self) -> float:
        """Exponential backoff on recovery timeout."""
        timeout = self.base_recovery_timeout * (self.backoff_multiplier**self._consecutive_opens)
        return min(timeout, self.max_recovery_timeout)

    def check(self) -> None:
        """Check if a request is allowed. Raises CircuitBreakerError if not."""
        if self._state == CircuitState.CLOSED:
            return

        if self._state == CircuitState.OPEN:
            elapsed = time.monotonic() - self._last_failure_time
            if elapsed >= self.current_recovery_timeout:
                # Transition to half-open
                self._state = CircuitState.HALF_OPEN
                self._half_open_calls = 0
                self._last_state_change = time.monotonic()
                return

            retry_after = self.current_recovery_timeout - elapsed
            self._total_rejections += 1
            raise CircuitBreakerError(retry_after=retry_after)

        if self._state == CircuitState.HALF_OPEN:
            if self._half_open_calls >= self.half_open_max_calls:
                # Too many half-open calls pending, reject
                self._total_rejections += 1
                raise CircuitBreakerError(retry_after=5.0)
            self._half_open_calls += 1

    def record_success(self) -> None:
        """Record a successful request."""
        self._total_successes += 1

        if self._state == CircuitState.HALF_OPEN:
            self._success_count += 1
            # Recovered — close the circuit
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._consecutive_opens = 0
            self._last_state_change = time.monotonic()
        elif self._state == CircuitState.CLOSED:
            # Reset failure count on success
            self._failure_count = 0

    def record_failure(self) -> None:
        """Record a failed request."""
        self._total_failures += 1
        self._failure_count += 1
        self._last_failure_time = time.monotonic()

        if self._state == CircuitState.HALF_OPEN:
            # Probe failed — back to open with longer timeout
            self._state = CircuitState.OPEN
            self._consecutive_opens += 1
            self._last_state_change = time.monotonic()
        elif self._state == CircuitState.CLOSED:
            if self._failure_count >= self.failure_threshold:
                self._state = CircuitState.OPEN
                self._consecutive_opens += 1
                self._last_state_change = time.monotonic()

    def reset(self) -> None:
        """Manually reset the circuit breaker to closed state."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._half_open_calls = 0
        self._consecutive_opens = 0
        self._last_state_change = time.monotonic()

    def metrics(self) -> dict:
        """Return circuit breaker metrics."""
        return {
            "state": self._state.value,
            "failure_count": self._failure_count,
            "consecutive_opens": self._consecutive_opens,
            "current_recovery_timeout_s": round(self.current_recovery_timeout, 1),
            "total_successes": self._total_successes,
            "total_failures": self._total_failures,
            "total_rejections": self._total_rejections,
            "time_in_state_s": round(time.monotonic() - self._last_state_change, 1),
        }


# Module-level singleton for the inference backend
_inference_breaker: CircuitBreaker | None = None


def get_inference_breaker() -> CircuitBreaker:
    global _inference_breaker
    if _inference_breaker is None:
        _inference_breaker = CircuitBreaker()
    return _inference_breaker
