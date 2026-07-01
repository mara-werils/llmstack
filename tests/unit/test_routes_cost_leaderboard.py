"""Tests for the /cost, /leaderboard, and /v1/embeddings route modules."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from llmstack.gateway.cost_tracker import CostTracker
from llmstack.gateway.leaderboard import Leaderboard
from llmstack.gateway.routes import cost as cost_route
from llmstack.gateway.routes import embeddings as embeddings_route
from llmstack.gateway.routes import leaderboard as lb_route


# --------------------------------------------------------------------------- #
# /cost
# --------------------------------------------------------------------------- #
@pytest.fixture
def cost_client(monkeypatch):
    monkeypatch.setattr(cost_route, "_tracker", CostTracker())
    app = FastAPI()
    app.include_router(cost_route.router)
    return TestClient(app)


class TestCostRoutes:
    def test_summary(self, cost_client):
        assert cost_client.get("/cost/summary").status_code == 200

    def test_summary_includes_local_savings(self, cost_client, tmp_path, monkeypatch):
        from llmstack.core.savings import SavingsCalculator, SavingsLedger
        from llmstack.gateway import savings as gw_savings
        from llmstack.gateway.savings import SavingsTracker

        tracker = SavingsTracker(
            calculator=SavingsCalculator("gpt-4o"),
            ledger=SavingsLedger(path=tmp_path / "savings.json"),
        )
        tracker.record(1000, 500, timestamp=1.0)
        monkeypatch.setattr(gw_savings, "_tracker", tracker)
        body = cost_client.get("/cost/summary").json()
        assert "savings" in body
        assert body["savings"]["total_saved_usd"] > 0
        monkeypatch.setattr(gw_savings, "_tracker", None)

    def test_budget_crud(self, cost_client):
        resp = cost_client.post(
            "/cost/budgets", json={"name": "b1", "limit_usd": 10, "period": "monthly"}
        )
        assert resp.status_code == 201
        assert any(b["name"] == "b1" for b in cost_client.get("/cost/budgets").json()["budgets"])
        assert cost_client.delete("/cost/budgets/b1").json()["deleted"] is True

    def test_add_budget_invalid_period(self, cost_client):
        resp = cost_client.post(
            "/cost/budgets", json={"name": "b", "limit_usd": 1, "period": "hourly"}
        )
        assert resp.status_code == 400

    def test_remove_missing_budget_404(self, cost_client):
        assert cost_client.delete("/cost/budgets/ghost").status_code == 404

    def test_alerts(self, cost_client):
        assert cost_client.get("/cost/alerts").json() == {"alerts": []}

    def test_set_pricing(self, cost_client):
        resp = cost_client.post(
            "/cost/pricing",
            json={"model": "m", "input_per_million": 1.0, "output_per_million": 2.0},
        )
        assert resp.json()["updated"] is True


class TestCostLazySingleton:
    def test_get_tracker_lazily_creates_and_reuses(self, monkeypatch):
        monkeypatch.setattr(cost_route, "_tracker", None)
        first = cost_route.get_tracker()
        assert cost_route.get_tracker() is first

    def test_init_cost_tracker_sets_module_tracker(self, monkeypatch):
        # monkeypatch restores the pre-test `_tracker` value on teardown even
        # though init_cost_tracker() mutates the module global directly.
        monkeypatch.setattr(cost_route, "_tracker", None)
        tracker = CostTracker()
        cost_route.init_cost_tracker(tracker)
        assert cost_route.get_tracker() is tracker


# --------------------------------------------------------------------------- #
# /leaderboard
# --------------------------------------------------------------------------- #
@pytest.fixture
def lb_client(monkeypatch):
    monkeypatch.setattr(lb_route, "_leaderboard", Leaderboard())
    app = FastAPI()
    app.include_router(lb_route.router)
    return TestClient(app)


class TestLeaderboardRoutes:
    def test_rankings_empty(self, lb_client):
        assert lb_client.get("/leaderboard").json() == {"rankings": []}

    def test_summary(self, lb_client):
        assert lb_client.get("/leaderboard/summary").status_code == 200

    def test_model_not_found(self, lb_client):
        assert lb_client.get("/leaderboard/models/ghost").status_code == 404

    def test_model_found(self, lb_client, monkeypatch):
        lb = Leaderboard()
        lb.record("llama3", provider="local", latency_ms=10.0, tokens=5)
        monkeypatch.setattr(lb_route, "_leaderboard", lb)
        resp = lb_client.get("/leaderboard/models/llama3")
        assert resp.status_code == 200
        assert resp.json()["model"] == "llama3"

    def test_compare_requires_models(self, lb_client):
        assert lb_client.get("/leaderboard/compare").status_code == 400

    def test_compare_with_models(self, lb_client):
        resp = lb_client.get("/leaderboard/compare?models=a,b")
        assert resp.status_code == 200
        assert "comparison" in resp.json()


class TestLeaderboardLazySingleton:
    def test_get_leaderboard_lazily_creates_and_reuses(self, monkeypatch):
        monkeypatch.setattr(lb_route, "_leaderboard", None)
        first = lb_route.get_leaderboard()
        assert lb_route.get_leaderboard() is first

    def test_init_leaderboard_sets_module_leaderboard(self, monkeypatch):
        monkeypatch.setattr(lb_route, "_leaderboard", None)
        lb = Leaderboard()
        lb_route.init_leaderboard(lb)
        assert lb_route.get_leaderboard() is lb


# --------------------------------------------------------------------------- #
# /v1/embeddings
# --------------------------------------------------------------------------- #
@pytest.fixture
def emb_client(monkeypatch):
    async def fake_proxy_embeddings(payload):
        return {"object": "list", "data": [{"embedding": [0.1, 0.2]}], "model": payload["model"]}

    monkeypatch.setattr(embeddings_route, "proxy_embeddings", fake_proxy_embeddings)
    app = FastAPI()
    app.include_router(embeddings_route.router, prefix="/v1")
    return TestClient(app)


class TestEmbeddingsRoute:
    def test_success(self, emb_client):
        resp = emb_client.post("/v1/embeddings", json={"input": "hello", "model": "bge-m3"})
        assert resp.status_code == 200
        assert resp.json()["model"] == "bge-m3"

    def test_validation_error(self, emb_client):
        # Missing required 'model' field.
        resp = emb_client.post("/v1/embeddings", json={"input": "hello"})
        assert resp.status_code == 422
        assert resp.json()["error"]["type"] == "validation_error"
