"""Tests for the persistent savings ledger (llmstack.core.savings)."""

from __future__ import annotations

import json

import pytest

from llmstack.core.savings import (
    SavingsCalculator,
    SavingsLedger,
    SavingsState,
    get_ledger,
    set_ledger,
)


@pytest.fixture
def ledger(tmp_path):
    return SavingsLedger(path=tmp_path / "savings.json")


def _est(input_tokens=1000, output_tokens=500):
    return SavingsCalculator("gpt-4o").estimate(input_tokens, output_tokens)


def test_fresh_ledger_is_empty(ledger) -> None:
    assert ledger.state.total_requests == 0
    assert ledger.state.total_saved_usd == 0.0
    assert ledger.state.first_recorded_at is None


def test_record_accumulates_and_sets_timestamps(ledger) -> None:
    ledger.record(_est(), timestamp=100.0)
    ledger.record(_est(), timestamp=200.0)
    s = ledger.state
    assert s.total_requests == 2
    assert s.total_input_tokens == 2000
    assert s.total_output_tokens == 1000
    assert s.total_saved_usd == pytest.approx(2 * _est().saved_usd)
    assert s.first_recorded_at == 100.0
    assert s.last_recorded_at == 200.0


def test_record_persists_and_reloads(tmp_path) -> None:
    path = tmp_path / "savings.json"
    first = SavingsLedger(path=path)
    first.record(_est(), timestamp=1.0)
    # A new ledger over the same path sees the persisted totals.
    second = SavingsLedger(path=path)
    assert second.state.total_requests == 1
    assert second.state.total_saved_usd == pytest.approx(_est().saved_usd)


def test_record_without_persist_does_not_write(tmp_path) -> None:
    path = tmp_path / "savings.json"
    led = SavingsLedger(path=path)
    led.record(_est(), timestamp=1.0, persist=False)
    assert not path.exists()


def test_load_ignores_corrupt_file(tmp_path) -> None:
    path = tmp_path / "savings.json"
    path.write_text("{not valid json")
    led = SavingsLedger(path=path)
    assert led.state == SavingsState()


def test_load_ignores_unknown_keys(tmp_path) -> None:
    path = tmp_path / "savings.json"
    path.write_text(json.dumps({"total_requests": 5, "bogus_field": 1}))
    led = SavingsLedger(path=path)
    assert led.state.total_requests == 5


def test_summary_includes_subscription_equivalence(ledger) -> None:
    # Save exactly two Cursor-months worth.
    from llmstack.core.pricing import baseline_subscription

    plan = baseline_subscription("cursor-pro")
    target = plan.effective_monthly_usd * 2
    # Force the saved total directly for a clean assertion.
    ledger.state.total_saved_usd = target
    summary = ledger.summary("cursor-pro")
    assert summary["subscription"]["key"] == "cursor-pro"
    assert summary["subscription"]["months_covered"] == pytest.approx(2.0)


def test_reset_clears_totals(ledger) -> None:
    ledger.record(_est(), timestamp=1.0)
    ledger.reset()
    assert ledger.state.total_requests == 0
    assert ledger.path.exists()  # reset persists by default


def test_reset_without_persist(tmp_path) -> None:
    path = tmp_path / "savings.json"
    led = SavingsLedger(path=path)
    led.reset(persist=False)
    assert not path.exists()


def test_process_ledger_override(tmp_path) -> None:
    import llmstack.core.savings as savings_mod

    custom = SavingsLedger(path=tmp_path / "savings.json")
    set_ledger(custom)
    try:
        assert get_ledger() is custom
    finally:
        # Reset the module global so other tests get a fresh default.
        savings_mod._ledger = None


def test_get_ledger_creates_default_lazily(monkeypatch, tmp_path) -> None:
    import llmstack.core.savings as savings_mod

    monkeypatch.setattr(savings_mod, "DEFAULT_LEDGER_PATH", tmp_path / "savings.json")
    monkeypatch.setattr(savings_mod, "_ledger", None)
    led = get_ledger()
    assert led is get_ledger()  # cached after first creation
    assert led.path == tmp_path / "savings.json"
    monkeypatch.setattr(savings_mod, "_ledger", None)
