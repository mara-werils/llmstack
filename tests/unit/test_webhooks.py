"""Tests for webhook notification system."""

import pytest

from llmstack.gateway.webhooks import (
    WebhookManager, WebhookEvent, WebhookEndpoint, _compute_signature,
)


@pytest.fixture
def manager():
    return WebhookManager()


class TestWebhookManager:
    def test_register_endpoint(self, manager):
        ep = manager.register(
            url="https://example.com/hook",
            events=[WebhookEvent.REQUEST_COMPLETED],
            description="Test hook",
        )
        assert ep.url == "https://example.com/hook"
        assert ep.active is True

    def test_list_endpoints(self, manager):
        manager.register(url="https://a.com", events=[WebhookEvent.REQUEST_ERROR])
        manager.register(url="https://b.com", events=[WebhookEvent.BUDGET_ALERT])
        assert len(manager.list_endpoints()) == 2

    def test_unregister(self, manager):
        ep = manager.register(url="https://a.com", events=[WebhookEvent.REQUEST_ERROR])
        assert manager.unregister(ep.id) is True
        assert manager.unregister("nonexistent") is False
        assert len(manager.list_endpoints()) == 0

    def test_dispatch_matching_event(self, manager):
        manager.register(
            url="https://a.com",
            events=[WebhookEvent.REQUEST_COMPLETED],
        )
        deliveries = manager.dispatch(
            WebhookEvent.REQUEST_COMPLETED,
            {"model": "gpt-4o", "latency_ms": 500},
        )
        assert len(deliveries) == 1
        assert deliveries[0].event == "request.completed"

    def test_dispatch_non_matching_event(self, manager):
        manager.register(
            url="https://a.com",
            events=[WebhookEvent.BUDGET_ALERT],
        )
        deliveries = manager.dispatch(
            WebhookEvent.REQUEST_COMPLETED,
            {"model": "test"},
        )
        assert len(deliveries) == 0

    def test_dispatch_inactive_endpoint_skipped(self, manager):
        ep = manager.register(
            url="https://a.com",
            events=[WebhookEvent.REQUEST_COMPLETED],
        )
        ep.active = False
        deliveries = manager.dispatch(
            WebhookEvent.REQUEST_COMPLETED, {},
        )
        assert len(deliveries) == 0

    def test_record_result_success(self, manager):
        manager.register(
            url="https://a.com",
            events=[WebhookEvent.REQUEST_COMPLETED],
        )
        deliveries = manager.dispatch(WebhookEvent.REQUEST_COMPLETED, {})
        manager.record_result(deliveries[0].id, 200, True, duration_ms=50.0)

        records = manager.get_deliveries()
        assert records[0].success is True
        assert records[0].status_code == 200

    def test_auto_disable_after_failures(self, manager):
        ep = manager.register(
            url="https://failing.com",
            events=[WebhookEvent.REQUEST_COMPLETED],
        )
        # Simulate consecutive failures up to max
        for _ in range(WebhookManager.MAX_FAILURES):
            deliveries = manager.dispatch(WebhookEvent.REQUEST_COMPLETED, {})
            if deliveries:
                manager.record_result(deliveries[0].id, 500, False, error="timeout")

        # Endpoint should be disabled
        fetched = manager.get_endpoint(ep.id)
        assert fetched.active is False

    def test_stats(self, manager):
        manager.register(url="https://a.com", events=[WebhookEvent.REQUEST_COMPLETED])
        manager.dispatch(WebhookEvent.REQUEST_COMPLETED, {})

        stats = manager.get_stats()
        assert stats["total_endpoints"] == 1
        assert stats["total_deliveries"] == 1
        assert "request.completed" in stats["event_counts"]

    def test_build_payload(self, manager):
        payload = manager.build_payload(
            WebhookEvent.BUDGET_ALERT,
            {"budget": "monthly", "spend": 50.0},
        )
        assert payload["event"] == "budget.alert"
        assert "timestamp" in payload
        assert payload["data"]["budget"] == "monthly"


class TestSignature:
    def test_compute_signature(self):
        sig = _compute_signature('{"test": true}', "secret123")
        assert len(sig) == 64  # SHA-256 hex digest

    def test_signature_deterministic(self):
        s1 = _compute_signature("data", "key")
        s2 = _compute_signature("data", "key")
        assert s1 == s2

    def test_signature_differs_with_key(self):
        s1 = _compute_signature("data", "key1")
        s2 = _compute_signature("data", "key2")
        assert s1 != s2
