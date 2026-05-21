"""Tests for usage quota system."""

import pytest

from llmstack.gateway.quotas import (
    QuotaManager, QuotaLimit, QuotaPeriod, QuotaExceededError,
)


@pytest.fixture
def manager():
    return QuotaManager()


class TestQuotaManager:
    def test_no_limits_allows_all(self, manager):
        manager.check("any-key")  # Should not raise

    def test_request_limit_enforced(self, manager):
        manager.add_limit(QuotaLimit(
            api_key="key1", max_requests=2, period=QuotaPeriod.TOTAL,
        ))
        manager.record_usage("key1", tokens=10)
        manager.record_usage("key1", tokens=10)

        with pytest.raises(QuotaExceededError, match="Request quota"):
            manager.check("key1")

    def test_token_limit_enforced(self, manager):
        manager.add_limit(QuotaLimit(
            api_key="key1", max_tokens=100, period=QuotaPeriod.TOTAL,
        ))
        manager.record_usage("key1", tokens=100)

        with pytest.raises(QuotaExceededError, match="Token quota"):
            manager.check("key1")

    def test_cost_limit_enforced(self, manager):
        manager.add_limit(QuotaLimit(
            api_key="key1", max_cost_usd=1.0, period=QuotaPeriod.TOTAL,
        ))
        manager.record_usage("key1", cost_usd=1.0)

        with pytest.raises(QuotaExceededError, match="Cost quota"):
            manager.check("key1")

    def test_wildcard_applies_to_all(self, manager):
        manager.add_limit(QuotaLimit(
            api_key="*", max_requests=1, period=QuotaPeriod.TOTAL,
        ))
        manager.record_usage("any-key", tokens=10)

        with pytest.raises(QuotaExceededError):
            manager.check("any-key")

    def test_model_specific_limit(self, manager):
        manager.add_limit(QuotaLimit(
            api_key="key1", max_requests=1,
            period=QuotaPeriod.TOTAL, model="gpt-4o",
        ))
        manager.record_usage("key1", model="gpt-4o", tokens=10)

        with pytest.raises(QuotaExceededError):
            manager.check("key1", model="gpt-4o")

        # Different model should still work
        manager.check("key1", model="llama3.2")

    def test_different_key_not_affected(self, manager):
        manager.add_limit(QuotaLimit(
            api_key="key1", max_requests=1, period=QuotaPeriod.TOTAL,
        ))
        manager.record_usage("key1", tokens=10)

        # key2 should not be affected
        manager.check("key2")

    def test_get_usage(self, manager):
        manager.record_usage("key1", model="gpt-4o", tokens=100, cost_usd=0.01)
        manager.record_usage("key1", model="gpt-4o", tokens=200, cost_usd=0.02)

        usage = manager.get_usage("key1")
        assert "gpt-4o" in usage
        assert usage["gpt-4o"]["total"]["requests"] == 2
        assert usage["gpt-4o"]["total"]["tokens"] == 300

    def test_remove_limits(self, manager):
        manager.add_limit(QuotaLimit(api_key="key1", max_requests=1))
        removed = manager.remove_limits("key1")
        assert removed == 1
        assert manager.get_limits() == []

    def test_get_limits(self, manager):
        manager.add_limit(QuotaLimit(
            api_key="key1", max_requests=100, period=QuotaPeriod.DAILY,
        ))
        limits = manager.get_limits()
        assert len(limits) == 1
        assert limits[0]["api_key"] == "key1"
        assert limits[0]["period"] == "daily"
