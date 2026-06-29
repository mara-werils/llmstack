"""Tests for advanced rate limiter v2."""

import time
from llmstack.gateway.rate_limiter_v2 import (
    AdvancedRateLimiter,
    RateLimitTier,
    SlidingWindowCounter,
)


def test_sliding_window_allows_within_limit():
    counter = SlidingWindowCounter(window_seconds=60, max_requests=5)
    for _ in range(5):
        result = counter.check_and_record()
        assert result.allowed


def test_sliding_window_blocks_over_limit():
    counter = SlidingWindowCounter(window_seconds=60, max_requests=3)
    for _ in range(3):
        counter.check_and_record()
    result = counter.check_and_record()
    assert not result.allowed
    assert result.retry_after > 0


def test_sliding_window_expires():
    counter = SlidingWindowCounter(window_seconds=1, max_requests=2)
    now = time.time()
    counter.check_and_record(now)
    counter.check_and_record(now)

    # Should be blocked now
    result = counter.check_and_record(now)
    assert not result.allowed

    # After window expires, should be allowed
    result = counter.check_and_record(now + 2)
    assert result.allowed


def test_rate_limiter_default_tier():
    limiter = AdvancedRateLimiter()
    assert limiter.get_tier("test-key") == RateLimitTier.FREE


def test_rate_limiter_set_tier():
    limiter = AdvancedRateLimiter()
    limiter.set_tier("premium", RateLimitTier.PRO)
    assert limiter.get_tier("premium") == RateLimitTier.PRO


def test_rate_limiter_allows_requests():
    limiter = AdvancedRateLimiter()
    limiter.set_tier("key1", RateLimitTier.STANDARD)
    result = limiter.check("key1")
    assert result.allowed


def test_rate_limiter_blocks_concurrent():
    limiter = AdvancedRateLimiter()
    limiter.set_tier("key1", RateLimitTier.FREE)

    # FREE tier allows 2 concurrent
    limiter.check("key1")
    limiter.check("key1")
    result = limiter.check("key1")
    assert not result.allowed
    assert "concurrent" in result.reason.lower()


def test_rate_limiter_release():
    limiter = AdvancedRateLimiter()
    limiter.set_tier("key1", RateLimitTier.FREE)

    limiter.check("key1")
    limiter.check("key1")
    limiter.release("key1")

    result = limiter.check("key1")
    assert result.allowed


def test_token_quota():
    limiter = AdvancedRateLimiter()
    limiter.set_tier("key1", RateLimitTier.FREE)
    # FREE tier: max 2048 tokens per request
    result = limiter.check("key1", tokens=5000)
    assert not result.allowed
    assert "token limit" in result.reason.lower()


def test_get_usage():
    limiter = AdvancedRateLimiter()
    limiter.set_tier("key1", RateLimitTier.STANDARD)
    limiter.check("key1", tokens=100)

    usage = limiter.get_usage("key1")
    assert usage["tier"] == "standard"
    assert usage["token_usage"] == 100
    assert usage["concurrent_active"] == 1


def test_per_endpoint_limits():
    limiter = AdvancedRateLimiter()
    limiter.set_tier("key1", RateLimitTier.FREE)

    # Different endpoints have separate windows
    result1 = limiter.check("key1", endpoint="chat")
    result2 = limiter.check("key1", endpoint="embeddings")
    assert result1.allowed
    assert result2.allowed


def test_minute_window_limit_blocks_without_concurrency_limit():
    limiter = AdvancedRateLimiter()
    limiter.set_tier("key1", RateLimitTier.FREE)
    # FREE: requests_per_minute=10, burst_size=5 -> minute window capacity 15.
    # Release after every check so the concurrency limit (2) never triggers.
    for _ in range(15):
        result = limiter.check("key1")
        assert result.allowed
        limiter.release("key1")

    result = limiter.check("key1")
    assert not result.allowed
    assert "rate limit exceeded" in result.reason.lower()


def test_daily_token_quota_exceeded():
    limiter = AdvancedRateLimiter()
    limiter.set_tier("key1", RateLimitTier.FREE)
    limiter._token_usage["key1"] = 49000
    limiter._token_reset["key1"] = time.time() + 1000  # prevent auto-reset

    result = limiter.check("key1", tokens=2000)

    assert not result.allowed
    assert "daily token quota" in result.reason.lower()


def test_is_rate_limited_false_when_no_history():
    limiter = AdvancedRateLimiter()
    limiter.set_tier("key1", RateLimitTier.FREE)
    assert limiter.is_rate_limited("key1") is False


def test_is_rate_limited_true_for_concurrency():
    limiter = AdvancedRateLimiter()
    limiter.set_tier("key1", RateLimitTier.FREE)
    limiter.check("key1")
    limiter.check("key1")  # FREE allows 2 concurrent
    assert limiter.is_rate_limited("key1") is True


def test_is_rate_limited_true_for_window_exhaustion():
    limiter = AdvancedRateLimiter()
    limiter.set_tier("key1", RateLimitTier.FREE)
    for _ in range(15):
        limiter.check("key1")
        limiter.release("key1")
    assert limiter.is_rate_limited("key1") is True


def test_is_rate_limited_false_when_under_limit():
    limiter = AdvancedRateLimiter()
    limiter.set_tier("key1", RateLimitTier.FREE)
    limiter.check("key1")
    limiter.release("key1")
    assert limiter.is_rate_limited("key1") is False


def test_reset_user_clears_counters():
    limiter = AdvancedRateLimiter()
    limiter.set_tier("key1", RateLimitTier.FREE)
    limiter.check("key1", tokens=100)

    limiter.reset_user("key1")

    usage = limiter.get_usage("key1")
    assert usage["token_usage"] == 0
    assert usage["concurrent_active"] == 0


def test_reset_user_clears_request_windows():
    limiter = AdvancedRateLimiter()
    limiter.set_tier("key1", RateLimitTier.FREE)
    # Exhaust the per-minute window (FREE capacity = 10 + 5 burst = 15),
    # releasing each slot so the concurrency limit never trips.
    for _ in range(15):
        assert limiter.check("key1", endpoint="chat").allowed
        limiter.release("key1")
    assert not limiter.check("key1", endpoint="chat").allowed  # window is full

    limiter.reset_user("key1")

    # The per-endpoint sliding window must be cleared, not just token/concurrent.
    assert limiter.check("key1", endpoint="chat").allowed


def test_release_on_zero_concurrent_is_noop():
    limiter = AdvancedRateLimiter()
    limiter.release("never-checked-in")  # should not raise or go negative
    assert limiter._concurrent["never-checked-in"] == 0
