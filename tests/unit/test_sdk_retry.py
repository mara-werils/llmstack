"""Tests for SDK retry logic."""

from __future__ import annotations
from llmstack.sdk.retry import RetryConfig, _calculate_delay


class TestRetryConfig:
    def test_defaults(self):
        c = RetryConfig()
        assert c.max_retries == 3
        assert 429 in c.retry_on_status
        assert c.backoff_factor == 2.0


class TestCalculateDelay:
    def test_first_attempt(self):
        c = RetryConfig(jitter=False)
        delay = _calculate_delay(0, c)
        assert delay == 0.5

    def test_exponential_backoff(self):
        c = RetryConfig(jitter=False)
        d0 = _calculate_delay(0, c)
        d1 = _calculate_delay(1, c)
        d2 = _calculate_delay(2, c)
        assert d1 == d0 * 2
        assert d2 == d1 * 2

    def test_max_delay_cap(self):
        c = RetryConfig(jitter=False, max_delay=5.0)
        delay = _calculate_delay(10, c)
        assert delay <= 5.0

    def test_retry_after_override(self):
        c = RetryConfig(jitter=False)
        delay = _calculate_delay(0, c, retry_after=10.0)
        assert delay == 10.0

    def test_jitter_adds_randomness(self):
        c = RetryConfig(jitter=True)
        delays = {_calculate_delay(1, c) for _ in range(20)}
        assert len(delays) > 1  # should not all be the same
