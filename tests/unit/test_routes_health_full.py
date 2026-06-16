"""Thorough unit tests for llmstack.gateway.routes.health.

Covers _check_url real branches, the /healthz extras blocks (circuit breaker,
cache, router stats, providers), and the /healthz/ready backend probes for both
healthy and unhealthy/degraded paths. All collaborators are monkeypatched —
no real network or redis.
"""

from __future__ import annotations

import sys
import types

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from llmstack.gateway.routes import health as health_route


@pytest.fixture
def make_client():
    """Build a TestClient with all backend URLs cleared by default."""

    def _make(inference="", qdrant="", redis=""):
        # Reset all module-level backend URLs to a known state.
        import llmstack.gateway.routes.health as h

        h.INFERENCE_URL = inference
        h.QDRANT_URL = qdrant
        h.REDIS_URL = redis
        app = FastAPI()
        app.include_router(h.router)
        return TestClient(app)

    return _make


# ---------------------------------------------------------------------------
# _check_url
# ---------------------------------------------------------------------------


class _FakeResp:
    def __init__(self, status_code):
        self.status_code = status_code


class _FakeAsyncClient:
    """Stand-in for httpx.AsyncClient used by _check_url."""

    status_code = 200
    raise_exc = False

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get(self, url):
        if type(self).raise_exc:
            raise RuntimeError("connection refused")
        return _FakeResp(type(self).status_code)


class TestCheckUrl:
    @pytest.mark.asyncio
    async def test_empty_url_returns_false(self):
        assert await health_route._check_url("") is False

    @pytest.mark.asyncio
    async def test_status_200_returns_true(self, monkeypatch):
        _FakeAsyncClient.status_code = 200
        _FakeAsyncClient.raise_exc = False
        monkeypatch.setattr(health_route.httpx, "AsyncClient", _FakeAsyncClient)
        assert await health_route._check_url("http://svc/health") is True

    @pytest.mark.asyncio
    async def test_status_500_returns_false(self, monkeypatch):
        _FakeAsyncClient.status_code = 500
        _FakeAsyncClient.raise_exc = False
        monkeypatch.setattr(health_route.httpx, "AsyncClient", _FakeAsyncClient)
        assert await health_route._check_url("http://svc/health") is False

    @pytest.mark.asyncio
    async def test_exception_returns_false(self, monkeypatch):
        _FakeAsyncClient.status_code = 200
        _FakeAsyncClient.raise_exc = True
        monkeypatch.setattr(health_route.httpx, "AsyncClient", _FakeAsyncClient)
        assert await health_route._check_url("http://svc/health") is False
        _FakeAsyncClient.raise_exc = False


# ---------------------------------------------------------------------------
# Helpers to install a fake redis.asyncio module
# ---------------------------------------------------------------------------


def _install_fake_redis(monkeypatch, *, ping_ok):
    class _FakeRedis:
        async def ping(self):
            if not ping_ok:
                raise RuntimeError("redis down")
            return True

        async def aclose(self):
            return None

    fake_mod = types.ModuleType("redis.asyncio")
    fake_mod.from_url = lambda *a, **k: _FakeRedis()
    # Ensure parent package exists so "import redis.asyncio as aioredis" works.
    if "redis" not in sys.modules:
        sys.modules["redis"] = types.ModuleType("redis")
    monkeypatch.setitem(sys.modules, "redis.asyncio", fake_mod)
    # `import redis.asyncio as aioredis` binds via getattr(redis, "asyncio")
    # when the parent package is already imported, so patch the attribute too.
    monkeypatch.setattr(sys.modules["redis"], "asyncio", fake_mod, raising=False)


# ---------------------------------------------------------------------------
# /healthz backend probe branches
# ---------------------------------------------------------------------------


class TestHealthzBackends:
    def test_inference_healthy(self, make_client, monkeypatch):
        async def fake_check(url):
            return True

        monkeypatch.setattr(health_route, "_check_url", fake_check)
        client = make_client(inference="http://infer/v1")
        body = client.get("/healthz").json()
        assert body["status"] == "ok"
        assert body["services"]["inference"] is True

    def test_inference_falls_back_to_health_suffix(self, make_client, monkeypatch):
        calls = []

        async def fake_check(url):
            calls.append(url)
            # First call (root) fails, second call (/health) succeeds.
            return url.endswith("/health")

        monkeypatch.setattr(health_route, "_check_url", fake_check)
        client = make_client(inference="http://infer/v1")
        resp = client.get("/healthz")
        assert resp.status_code == 200
        assert resp.json()["services"]["inference"] is True
        # /v1 stripped, then both root and /health probed.
        assert "http://infer" in calls
        assert "http://infer/health" in calls

    def test_qdrant_branch(self, make_client, monkeypatch):
        seen = []

        async def fake_check(url):
            seen.append(url)
            return True

        monkeypatch.setattr(health_route, "_check_url", fake_check)
        client = make_client(qdrant="http://qdrant:6333")
        body = client.get("/healthz").json()
        assert body["services"]["qdrant"] is True
        assert "http://qdrant:6333/healthz" in seen

    def test_redis_healthy(self, make_client, monkeypatch):
        _install_fake_redis(monkeypatch, ping_ok=True)
        client = make_client(redis="redis://localhost:6379")
        body = client.get("/healthz").json()
        assert body["services"]["redis"] is True
        assert body["status"] == "ok"

    def test_redis_unhealthy_degrades(self, make_client, monkeypatch):
        _install_fake_redis(monkeypatch, ping_ok=False)
        client = make_client(redis="redis://localhost:6379")
        resp = client.get("/healthz")
        assert resp.status_code == 503
        body = resp.json()
        assert body["services"]["redis"] is False
        assert body["status"] == "degraded"


# ---------------------------------------------------------------------------
# /healthz extras blocks
# ---------------------------------------------------------------------------


class TestHealthzExtras:
    def test_circuit_breaker_extra(self, make_client, monkeypatch):
        import llmstack.gateway.circuit_breaker as cb

        class _Breaker:
            def metrics(self):
                return {"state": "closed", "failures": 0}

        monkeypatch.setattr(cb, "get_inference_breaker", lambda: _Breaker())
        client = make_client()
        body = client.get("/healthz").json()
        assert body["circuit_breaker"] == {"state": "closed", "failures": 0}

    def test_cache_extra(self, make_client, monkeypatch):
        import llmstack.gateway.cache as cache_mod

        class _Stats:
            def to_dict(self):
                return {"hits": 5, "misses": 1}

        class _Cache:
            stats = _Stats()

        monkeypatch.setattr(cache_mod, "_cache", _Cache())
        client = make_client()
        body = client.get("/healthz").json()
        assert body["cache"] == {"hits": 5, "misses": 1}

    def test_router_stats_extra(self, make_client, monkeypatch):
        import llmstack.gateway.router._state as state

        class _Stats:
            def summary(self):
                return {
                    "total_requests": 9,
                    "tier_distribution": {"simple": 9},
                    "provider_distribution": {"ollama": 9},
                    "estimated_savings_pct": 42.0,
                    "total_cost_usd": 1.23,
                    "cost_by_provider_usd": {"ollama": 1.23},
                }

        monkeypatch.setattr(state, "get_stats", lambda: _Stats())
        client = make_client()
        body = client.get("/healthz").json()
        assert body["router"]["total_requests"] == 9
        assert body["router"]["tier_distribution"] == {"simple": 9}
        assert body["router"]["estimated_savings_pct"] == 42.0
        assert body["router"]["total_cost_usd"] == 1.23

    def test_router_stats_none_skips(self, make_client, monkeypatch):
        import llmstack.gateway.router._state as state

        monkeypatch.setattr(state, "get_stats", lambda: None)
        client = make_client()
        body = client.get("/healthz").json()
        assert "router" not in body

    def test_providers_extra(self, make_client, monkeypatch):
        import llmstack.gateway.providers.registry as reg

        class _Model:
            def __init__(self, provider):
                self.provider = provider

        class _Registry:
            def all_providers(self):
                return {"ollama": object(), "openai": object()}

            def all_models(self):
                return [_Model("ollama"), _Model("ollama"), _Model("openai")]

        monkeypatch.setattr(reg, "get_registry", lambda: _Registry())
        client = make_client()
        body = client.get("/healthz").json()
        assert body["providers"]["ollama"]["models"] == 2
        assert body["providers"]["openai"]["models"] == 1

    def test_registry_none_skips(self, make_client, monkeypatch):
        import llmstack.gateway.providers.registry as reg

        monkeypatch.setattr(reg, "get_registry", lambda: None)
        client = make_client()
        body = client.get("/healthz").json()
        assert "providers" not in body

    def test_extras_exceptions_are_swallowed(self, make_client, monkeypatch):
        import llmstack.gateway.cache as cache_mod
        import llmstack.gateway.circuit_breaker as cb
        import llmstack.gateway.providers.registry as reg
        import llmstack.gateway.router._state as state

        def _boom(*a, **k):
            raise RuntimeError("boom")

        monkeypatch.setattr(cb, "get_inference_breaker", _boom)
        monkeypatch.setattr(state, "get_stats", _boom)
        monkeypatch.setattr(reg, "get_registry", _boom)

        class _BadStats:
            def to_dict(self):
                raise RuntimeError("boom")

        class _BadCache:
            stats = _BadStats()

        monkeypatch.setattr(cache_mod, "_cache", _BadCache())
        client = make_client()
        resp = client.get("/healthz")
        # Despite every extras block raising, the endpoint still succeeds.
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        for key in ("circuit_breaker", "cache", "router", "providers"):
            assert key not in body


# ---------------------------------------------------------------------------
# /healthz/ready backend probe branches
# ---------------------------------------------------------------------------


class TestReadiness:
    def test_ready_with_no_backends(self, make_client):
        client = make_client()
        resp = client.get("/healthz/ready")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ready"

    def test_inference_ready(self, make_client, monkeypatch):
        async def fake_check(url):
            return True

        monkeypatch.setattr(health_route, "_check_url", fake_check)
        client = make_client(inference="http://infer/v1")
        resp = client.get("/healthz/ready")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ready"
        assert body["checks"]["inference"] is True

    def test_inference_fallback_health_suffix(self, make_client, monkeypatch):
        async def fake_check(url):
            return url.endswith("/health")

        monkeypatch.setattr(health_route, "_check_url", fake_check)
        client = make_client(inference="http://infer/v1")
        assert client.get("/healthz/ready").json()["checks"]["inference"] is True

    def test_qdrant_ready(self, make_client, monkeypatch):
        seen = []

        async def fake_check(url):
            seen.append(url)
            return True

        monkeypatch.setattr(health_route, "_check_url", fake_check)
        client = make_client(qdrant="http://qdrant:6333")
        assert client.get("/healthz/ready").json()["checks"]["qdrant"] is True
        assert "http://qdrant:6333/healthz" in seen

    def test_redis_ready(self, make_client, monkeypatch):
        _install_fake_redis(monkeypatch, ping_ok=True)
        client = make_client(redis="redis://localhost:6379")
        body = client.get("/healthz/ready").json()
        assert body["checks"]["redis"] is True
        assert body["status"] == "ready"

    def test_redis_not_ready(self, make_client, monkeypatch):
        _install_fake_redis(monkeypatch, ping_ok=False)
        client = make_client(redis="redis://localhost:6379")
        resp = client.get("/healthz/ready")
        assert resp.status_code == 503
        body = resp.json()
        assert body["checks"]["redis"] is False
        assert body["status"] == "not_ready"

    def test_not_ready_when_inference_down(self, make_client, monkeypatch):
        async def fake_check(url):
            return False

        monkeypatch.setattr(health_route, "_check_url", fake_check)
        client = make_client(inference="http://infer/v1")
        resp = client.get("/healthz/ready")
        assert resp.status_code == 503
        assert resp.json()["status"] == "not_ready"
