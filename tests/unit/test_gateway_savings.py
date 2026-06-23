"""Tests for the gateway savings tracker (llmstack.gateway.savings)."""

from __future__ import annotations

import pytest

from llmstack.core.savings import SavingsCalculator, SavingsLedger
from llmstack.gateway.savings import (
    SavingsTracker,
    get_savings_tracker,
    init_savings_tracker,
)


@pytest.fixture
def tracker(tmp_path):
    ledger = SavingsLedger(path=tmp_path / "savings.json")
    return SavingsTracker(calculator=SavingsCalculator("gpt-4o"), ledger=ledger)


def test_local_request_accrues_saving(tracker) -> None:
    est = tracker.record(1000, 500, cost_usd=0.0, timestamp=1.0)
    assert est is not None
    assert est.saved_usd > 0
    assert tracker.ledger.state.total_requests == 1
    assert tracker.ledger.state.total_saved_usd == pytest.approx(est.saved_usd)


def test_paid_cloud_request_is_skipped(tracker) -> None:
    assert tracker.record(1000, 500, cost_usd=0.01, timestamp=1.0) is None
    assert tracker.ledger.state.total_requests == 0


def test_empty_usage_is_skipped(tracker) -> None:
    assert tracker.record(0, 0, timestamp=1.0) is None
    assert tracker.ledger.state.total_requests == 0


def test_record_uses_wall_clock_when_no_timestamp(tracker, monkeypatch) -> None:
    monkeypatch.setattr("llmstack.gateway.savings.time.time", lambda: 12345.0)
    tracker.record(10, 10)
    assert tracker.ledger.state.last_recorded_at == 12345.0


def test_summary_passthrough(tracker) -> None:
    tracker.record(1000, 500, timestamp=1.0)
    summary = tracker.summary("cursor-pro")
    assert summary["total_requests"] == 1
    assert summary["subscription"]["key"] == "cursor-pro"


def test_reset(tracker) -> None:
    tracker.record(1000, 500, timestamp=1.0)
    tracker.reset()
    assert tracker.ledger.state.total_requests == 0


def test_process_tracker_override_and_default(tmp_path) -> None:
    import llmstack.gateway.savings as savings_mod

    custom = SavingsTracker(ledger=SavingsLedger(path=tmp_path / "s.json"))
    init_savings_tracker(custom)
    try:
        assert get_savings_tracker() is custom
    finally:
        savings_mod._tracker = None


def test_default_tracker_created_lazily(monkeypatch, tmp_path) -> None:
    import llmstack.core.savings as core_savings
    import llmstack.gateway.savings as savings_mod

    monkeypatch.setattr(core_savings, "DEFAULT_LEDGER_PATH", tmp_path / "s.json")
    monkeypatch.setattr(core_savings, "_ledger", None)
    monkeypatch.setattr(savings_mod, "_tracker", None)
    t = get_savings_tracker()
    assert t is get_savings_tracker()
    monkeypatch.setattr(savings_mod, "_tracker", None)
    monkeypatch.setattr(core_savings, "_ledger", None)
