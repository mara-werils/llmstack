"""Tests for the /router and /healthz route modules."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from llmstack.gateway.routes import health as health_route
from llmstack.gateway.routes import router as router_route
from llmstack.gateway.router import _state as router_state


@pytest.fixture
def router_client():
    app = FastAPI()
    app.include_router(router_route.router)
    return TestClient(app)


@pytest.fixture(autouse=True)
def _disable_router(monkeypatch):
    monkeypatch.setattr(router_state, "_router", None)
    monkeypatch.setattr(router_state, "_stats", None)


class TestRouterRoutes:
    def test_stats_404_when_disabled(self, router_client):
        assert router_client.get("/router/stats").status_code == 404

    def test_classify_404_when_disabled(self, router_client):
        assert router_client.post("/router/classify", json={"messages": []}).status_code == 404

    def test_stats_enabled(self, router_client, monkeypatch):
        class _Stats:
            def summary(self):
                return {"total_requests": 3, "tier_distribution": {}}

        monkeypatch.setattr(router_state, "_stats", _Stats())
        body = router_client.get("/router/stats").json()
        assert body["total_requests"] == 3

    def test_classify_enabled(self, router_client, monkeypatch):
        class _Profile:
            score = 0.7
            tier = "complex"
            factors = {"length": 1}
            suggested_model = "llama3.1:70b"

        class _Router:
            def classify_only(self, messages):
                return _Profile()

        monkeypatch.setattr(router_state, "_router", _Router())
        body = router_client.post("/router/classify", json={"messages": [{"role": "user"}]}).json()
        assert body["tier"] == "complex"
        assert body["suggested_model"] == "llama3.1:70b"

    def test_stats_exception_treated_as_disabled(self, router_client, monkeypatch):
        def _boom():
            raise RuntimeError("router state unavailable")

        monkeypatch.setattr(router_state, "get_stats", _boom)
        assert router_client.get("/router/stats").status_code == 404

    def test_classify_exception_treated_as_disabled(self, router_client, monkeypatch):
        def _boom():
            raise RuntimeError("router state unavailable")

        monkeypatch.setattr(router_state, "get_router", _boom)
        resp = router_client.post("/router/classify", json={"messages": []})
        assert resp.status_code == 404


@pytest.fixture
def health_client(monkeypatch):
    # No backends configured → checks dict stays empty → healthy.
    monkeypatch.setattr(health_route, "INFERENCE_URL", "")
    monkeypatch.setattr(health_route, "QDRANT_URL", "")
    monkeypatch.setattr(health_route, "REDIS_URL", "")
    app = FastAPI()
    app.include_router(health_route.router)
    return TestClient(app)


class TestHealthRoutes:
    def test_ping(self, health_client):
        resp = health_client.get("/ping")
        assert resp.status_code == 200
        assert resp.text == "pong"

    def test_liveness(self, health_client):
        assert health_client.get("/healthz/live").json()["status"] == "alive"

    def test_healthz_ok_with_no_backends(self, health_client):
        resp = health_client.get("/healthz")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert "version" in body
        assert body["services"] == {}

    def test_readiness_ready_with_no_backends(self, health_client):
        resp = health_client.get("/healthz/ready")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ready"

    def test_healthz_degraded_when_backend_down(self, health_client, monkeypatch):
        monkeypatch.setattr(health_route, "INFERENCE_URL", "http://localhost:1/v1")

        async def fake_check(url):
            return False

        monkeypatch.setattr(health_route, "_check_url", fake_check)
        resp = health_client.get("/healthz")
        assert resp.status_code == 503
        assert resp.json()["status"] == "degraded"

    def test_metrics_endpoint(self, health_client):
        resp = health_client.get("/metrics")
        assert resp.status_code == 200
        assert "text/plain" in resp.headers["content-type"]
