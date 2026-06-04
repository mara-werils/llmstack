"""Tests for tiered rate limiting."""

import pytest

from llmstack.gateway.middleware.rate_limit_tiers import (
    TieredRateLimiter,
    TierConfig,
    DEFAULT_TIERS,
)


@pytest.fixture
def limiter():
    return TieredRateLimiter()


class TestTieredRateLimiter:
    def test_default_tiers_exist(self):
        assert len(DEFAULT_TIERS) == 4
        assert "enterprise" in DEFAULT_TIERS
        assert "free" in DEFAULT_TIERS

    def test_standard_tier_allows_requests(self, limiter):
        allowed, reason = limiter.check("key1")
        assert allowed is True

    def test_free_tier_rate_limit(self, limiter):
        limiter.set_key_tier("free-key", "free")
        # Free tier: 20 req/min
        for _ in range(20):
            limiter.record_request("free-key")

        allowed, reason = limiter.check("free-key")
        assert allowed is False
        assert "req/min" in reason

    def test_enterprise_higher_limits(self, limiter):
        limiter.set_key_tier("ent-key", "enterprise")
        for _ in range(50):
            limiter.record_request("ent-key")
            limiter.record_completion("ent-key")

        allowed, _ = limiter.check("ent-key")
        assert allowed is True  # Enterprise allows 1000/min

    def test_concurrent_limit(self, limiter):
        limiter.add_tier(
            TierConfig(
                name="test",
                concurrent_requests=2,
                requests_per_minute=1000,
            )
        )
        limiter.set_key_tier("key1", "test")

        limiter.record_request("key1")
        limiter.record_request("key1")

        allowed, reason = limiter.check("key1")
        assert allowed is False
        assert "Concurrent" in reason

    def test_record_completion_frees_slot(self, limiter):
        limiter.add_tier(
            TierConfig(
                name="test",
                concurrent_requests=1,
                requests_per_minute=1000,
            )
        )
        limiter.set_key_tier("key1", "test")

        limiter.record_request("key1")
        limiter.record_completion("key1")

        allowed, _ = limiter.check("key1")
        assert allowed is True

    def test_get_limits(self, limiter):
        limiter.set_key_tier("key1", "pro")
        limits = limiter.get_limits("key1")
        assert limits["tier"] == "pro"
        assert limits["requests_per_minute"] == 300

    def test_get_all_tiers(self, limiter):
        tiers = limiter.get_all_tiers()
        assert len(tiers) == 4
        names = {t["name"] for t in tiers}
        assert "enterprise" in names

    def test_custom_tier(self, limiter):
        limiter.add_tier(
            TierConfig(
                name="vip",
                requests_per_minute=5000,
            )
        )
        limiter.set_key_tier("vip-key", "vip")
        limits = limiter.get_limits("vip-key")
        assert limits["requests_per_minute"] == 5000

    def test_unknown_key_uses_standard(self, limiter):
        limits = limiter.get_limits("new-key")
        assert limits["tier"] == "standard"
