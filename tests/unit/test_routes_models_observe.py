"""Tests for the /v1/models and /observe API routes."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from llmstack.gateway.providers import registry as registry_mod
from llmstack.gateway.providers.base import ProviderModel
from llmstack.gateway.routes import models as models_route
from llmstack.gateway.routes import observe as observe_route
from llmstack.observe import _state as observe_state


# --------------------------------------------------------------------------- #
# /v1/models
# --------------------------------------------------------------------------- #
@pytest.fixture
def models_client(monkeypatch):
    async def fake_proxy_models():
        return {"object": "list", "data": [{"id": "llama3"}, {"id": "llama3"}]}

    monkeypatch.setattr(models_route, "proxy_models", fake_proxy_models)
    app = FastAPI()
    app.include_router(models_route.router, prefix="/v1")
    return app, monkeypatch


def test_models_dedupes_and_merges_providers(models_client):
    app, monkeypatch = models_client

    class _Reg:
        def all_models(self):
            return [ProviderModel(id="gpt-4o", provider="openai", context_length=128000)]

    monkeypatch.setattr(registry_mod, "get_registry", lambda: _Reg())
    data = TestClient(app).get("/v1/models").json()["data"]
    ids = [m["id"] for m in data]
    assert ids.count("llama3") == 1  # deduped
    assert "gpt-4o" in ids


def test_models_created_timestamp_is_stable(models_client):
    app, monkeypatch = models_client

    class _Reg:
        def all_models(self):
            return [ProviderModel(id="gpt-4o", provider="openai", context_length=128000)]

    monkeypatch.setattr(registry_mod, "get_registry", lambda: _Reg())
    client = TestClient(app)
    first = {m["id"]: m for m in client.get("/v1/models").json()["data"]}
    second = {m["id"]: m for m in client.get("/v1/models").json()["data"]}
    # OpenAI clients diff model lists; `created` must not move between calls.
    assert first["gpt-4o"]["created"] == second["gpt-4o"]["created"]
    assert first["gpt-4o"]["created"] == models_route._REGISTRY_MODEL_CREATED


def test_models_survives_registry_lookup_failure(models_client):
    """A broken provider registry must not take down /v1/models entirely —
    local models should still be returned."""
    app, monkeypatch = models_client

    def boom():
        raise RuntimeError("registry unavailable")

    monkeypatch.setattr(registry_mod, "get_registry", boom)
    data = TestClient(app).get("/v1/models").json()["data"]
    assert [m["id"] for m in data] == ["llama3"]


def test_models_handles_proxy_failure(monkeypatch):
    async def boom():
        raise ConnectionError("down")

    monkeypatch.setattr(models_route, "proxy_models", boom)
    monkeypatch.setattr(registry_mod, "get_registry", lambda: None)
    app = FastAPI()
    app.include_router(models_route.router, prefix="/v1")
    assert TestClient(app).get("/v1/models").json() == {"object": "list", "data": []}


# --------------------------------------------------------------------------- #
# /observe
# --------------------------------------------------------------------------- #
@pytest.fixture
def observe_client():
    app = FastAPI()
    app.include_router(observe_route.router)
    return TestClient(app)


@pytest.fixture(autouse=True)
def _reset_observe(monkeypatch):
    # Default: observe disabled (all getters return None)
    for name in ("get_trace_store", "get_tracker", "get_ab_manager"):
        monkeypatch.setattr(observe_state, name, lambda: None)


class TestObserveDisabled:
    def test_traces(self, observe_client):
        body = observe_client.get("/observe/traces").json()
        assert body["traces"] == []

    def test_traces_summary(self, observe_client):
        assert "error" in observe_client.get("/observe/traces/summary").json()

    def test_quality(self, observe_client):
        assert "error" in observe_client.get("/observe/quality").json()

    def test_alerts(self, observe_client):
        assert observe_client.get("/observe/alerts").json() == {"alerts": []}

    def test_create_ab_test_503(self, observe_client):
        resp = observe_client.post("/observe/ab-test", json={"name": "t"})
        assert resp.status_code == 503

    def test_list_ab_tests(self, observe_client):
        assert observe_client.get("/observe/ab-test").json() == {"tests": []}

    def test_get_ab_test_503(self, observe_client):
        assert observe_client.get("/observe/ab-test/x").status_code == 503

    def test_stats_503(self, observe_client):
        assert observe_client.get("/observe/stats").status_code == 503

    def test_stop_ab_test_503(self, observe_client):
        assert observe_client.delete("/observe/ab-test/x").status_code == 503


class _Trace:
    def to_dict(self):
        return {"id": "t1"}


class _FakeStore:
    total_count = 1

    def query(self, **kw):
        self.last_kwargs = kw
        return [_Trace()]

    def summary(self):
        return {"count": 1}


class _Alert:
    def to_dict(self):
        return {"msg": "drift"}


class _FakeTracker:
    def summary(self):
        return {"avg": 0.9}

    def get_alerts(self, limit=20):
        return [_Alert()]


class TestObserveEnabled:
    def test_traces_with_filters(self, observe_client, monkeypatch):
        store = _FakeStore()
        monkeypatch.setattr(observe_state, "get_trace_store", lambda: store)
        body = observe_client.get("/observe/traces?model=llama3&provider=local").json()
        assert body["total"] == 1
        assert body["traces"] == [{"id": "t1"}]
        assert store.last_kwargs == {"limit": 50, "model": "llama3", "provider": "local"}

    def test_traces_summary_enabled(self, observe_client, monkeypatch):
        monkeypatch.setattr(observe_state, "get_trace_store", lambda: _FakeStore())
        assert observe_client.get("/observe/traces/summary").json() == {"count": 1}

    def test_quality_and_alerts(self, observe_client, monkeypatch):
        monkeypatch.setattr(observe_state, "get_tracker", lambda: _FakeTracker())
        assert observe_client.get("/observe/quality").json() == {"avg": 0.9}
        assert observe_client.get("/observe/alerts").json() == {"alerts": [{"msg": "drift"}]}

    def test_stats_aggregates(self, observe_client, monkeypatch):
        monkeypatch.setattr(observe_state, "get_trace_store", lambda: _FakeStore())
        monkeypatch.setattr(observe_state, "get_tracker", lambda: _FakeTracker())

        class _Mgr:
            def list_tests(self):
                return [object(), object()]

        monkeypatch.setattr(observe_state, "get_ab_manager", lambda: _Mgr())
        body = observe_client.get("/observe/stats").json()
        assert body["traces"]["total"] == 1
        assert body["quality"]["avg"] == 0.9
        assert body["ab_tests"]["active"] == 2

    def test_ab_test_create_list_get_stop(self, observe_client, monkeypatch):
        created = []

        class _Result:
            def __init__(self, name):
                self.name = name

            def to_dict(self):
                return {"name": self.name, "winner": "a"}

        class _Mgr:
            def create_test(self, test):
                created.append(test)

            def list_tests(self):
                return created

            def get_results(self, name):
                return _Result(name) if created else None

            def stop_test(self, name):
                self.stopped = name

        mgr = _Mgr()
        monkeypatch.setattr(observe_state, "get_ab_manager", lambda: mgr)

        resp = observe_client.post(
            "/observe/ab-test",
            json={"name": "exp", "model_a": "a", "model_b": "b", "traffic_split": 0.3},
        )
        assert resp.json()["status"] == "created"
        assert created[0].name == "exp"
        assert created[0].traffic_split == 0.3

        listed = observe_client.get("/observe/ab-test").json()["tests"]
        assert listed[0]["winner"] == "a"

        assert observe_client.get("/observe/ab-test/exp").json()["name"] == "exp"
        assert observe_client.delete("/observe/ab-test/exp").json()["status"] == "stopped"

    def test_get_ab_test_not_found(self, observe_client, monkeypatch):
        class _Mgr:
            def get_results(self, name):
                return None

        monkeypatch.setattr(observe_state, "get_ab_manager", lambda: _Mgr())
        assert observe_client.get("/observe/ab-test/ghost").status_code == 404
