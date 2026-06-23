"""Tests for the /v1/savings route module."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from llmstack.core.savings import SavingsCalculator, SavingsLedger
from llmstack.gateway.routes import savings as savings_route
from llmstack.gateway.savings import SavingsTracker


@pytest.fixture
def client(tmp_path, monkeypatch):
    tracker = SavingsTracker(
        calculator=SavingsCalculator("gpt-4o"),
        ledger=SavingsLedger(path=tmp_path / "savings.json"),
    )
    monkeypatch.setattr(savings_route, "_tracker", tracker)
    app = FastAPI()
    app.include_router(savings_route.router, prefix="/v1")
    return TestClient(app), tracker


def test_summary_starts_empty(client) -> None:
    c, _ = client
    body = c.get("/v1/savings/summary").json()
    assert body["total_requests"] == 0
    assert body["total_saved_usd"] == 0.0


def test_summary_reflects_recorded_savings(client) -> None:
    c, tracker = client
    tracker.record(1000, 500, timestamp=1.0)
    body = c.get("/v1/savings/summary").json()
    assert body["total_requests"] == 1
    assert body["total_saved_usd"] > 0


def test_summary_accepts_plan_param(client) -> None:
    c, tracker = client
    tracker.record(1000, 500, timestamp=1.0)
    body = c.get("/v1/savings/summary", params={"plan": "cursor-pro"}).json()
    assert body["subscription"]["key"] == "cursor-pro"


def test_pricing_endpoint_is_dated_and_sourced(client) -> None:
    c, _ = client
    body = c.get("/v1/savings/pricing").json()
    assert body["as_of"]
    assert body["baseline_model"] == "gpt-4o-mini"
    assert body["api_pricing"] and all(p["source"].startswith("http") for p in body["api_pricing"])
    assert body["subscriptions"] and all(
        s["source"].startswith("http") for s in body["subscriptions"]
    )


def test_reset_clears_totals(client) -> None:
    c, tracker = client
    tracker.record(1000, 500, timestamp=1.0)
    assert c.post("/v1/savings/reset").json()["reset"] is True
    assert c.get("/v1/savings/summary").json()["total_requests"] == 0


def test_route_falls_back_to_process_tracker(monkeypatch, tmp_path) -> None:
    # When no tracker is injected, the route uses the process-wide tracker.
    import llmstack.core.savings as core_savings
    import llmstack.gateway.savings as gw_savings

    monkeypatch.setattr(core_savings, "DEFAULT_LEDGER_PATH", tmp_path / "s.json")
    monkeypatch.setattr(core_savings, "_ledger", None)
    monkeypatch.setattr(gw_savings, "_tracker", None)
    monkeypatch.setattr(savings_route, "_tracker", None)
    app = FastAPI()
    app.include_router(savings_route.router, prefix="/v1")
    client = TestClient(app)
    assert client.get("/v1/savings/summary").status_code == 200
    monkeypatch.setattr(gw_savings, "_tracker", None)
    monkeypatch.setattr(core_savings, "_ledger", None)
