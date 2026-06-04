"""Tests for token bucket rate limiter."""

from __future__ import annotations

import pytest

from llmstack.gateway.middleware.rate_limit import _parse_rate_limit, _InMemoryBucket


class TestRateLimitParsing:
    """Test rate limit spec parsing."""

    def test_per_minute(self):
        capacity, refill = _parse_rate_limit("100/min")
        assert capacity == 100
        assert refill == pytest.approx(100 / 60)

    def test_per_second(self):
        capacity, refill = _parse_rate_limit("10/sec")
        assert capacity == 10
        assert refill == 10.0

    def test_per_hour(self):
        capacity, refill = _parse_rate_limit("3600/hour")
        assert capacity == 3600
        assert refill == pytest.approx(1.0)

    def test_short_units(self):
        cap_m, rate_m = _parse_rate_limit("60/m")
        cap_s, rate_s = _parse_rate_limit("1/s")
        cap_h, rate_h = _parse_rate_limit("3600/h")
        assert cap_m == 60
        assert cap_s == 1
        assert cap_h == 3600

    def test_invalid_returns_default(self):
        capacity, refill = _parse_rate_limit("invalid")
        assert capacity == 100
        assert refill == pytest.approx(100 / 60)


class TestInMemoryBucket:
    """Test the in-memory fallback token bucket."""

    def test_allows_within_capacity(self):
        bucket = _InMemoryBucket(capacity=5, refill_rate=1.0)
        for _ in range(5):
            allowed, remaining, retry = bucket.try_acquire()
            assert allowed is True

    def test_rejects_over_capacity(self):
        bucket = _InMemoryBucket(capacity=2, refill_rate=0.0001)
        bucket.try_acquire()
        bucket.try_acquire()
        allowed, remaining, retry = bucket.try_acquire()
        assert allowed is False
        assert retry > 0

    def test_remaining_decrements(self):
        bucket = _InMemoryBucket(capacity=3, refill_rate=0.0)
        _, r1, _ = bucket.try_acquire()
        _, r2, _ = bucket.try_acquire()
        assert r1 > r2

    def test_refills_over_time(self):
        """Bucket should refill tokens based on elapsed time."""
        import time

        bucket = _InMemoryBucket(capacity=1, refill_rate=1000.0)  # fast refill
        bucket.try_acquire()
        time.sleep(0.01)  # allow 10ms to elapse for refill
        allowed, _, _ = bucket.try_acquire()
        assert allowed is True
