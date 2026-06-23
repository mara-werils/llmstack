"""Smoke tests for the `llmstack savings` command (llmstack.cli.commands.savings)."""

from __future__ import annotations

import pytest

import llmstack.core.savings as core_savings
from llmstack.cli.commands.savings import savings
from llmstack.core.savings import SavingsCalculator


@pytest.fixture
def ledger_path(monkeypatch, tmp_path):
    path = tmp_path / "savings.json"
    monkeypatch.setattr(core_savings, "DEFAULT_LEDGER_PATH", path)
    monkeypatch.setattr(core_savings, "_ledger", None)
    yield path
    monkeypatch.setattr(core_savings, "_ledger", None)


def _seed():
    est = SavingsCalculator("gpt-4o").estimate(1000, 500)
    core_savings.get_ledger().record(est, timestamp=1.0)


def test_empty_ledger_runs(ledger_path, capsys) -> None:
    savings()
    out = capsys.readouterr().out
    assert "No local requests recorded" in out


def test_populated_summary_renders(ledger_path, capsys) -> None:
    _seed()
    savings()
    out = capsys.readouterr().out
    assert "Saved so far" in out
    assert "month(s)" in out


def test_json_output(ledger_path, capsys) -> None:
    _seed()
    savings(as_json=True)
    out = capsys.readouterr().out
    assert "total_saved_usd" in out


def test_plan_override(ledger_path, capsys) -> None:
    _seed()
    savings(plan="cursor-pro")
    out = capsys.readouterr().out
    assert "Cursor" in out


def test_unknown_plan_is_handled(ledger_path, capsys) -> None:
    _seed()
    savings(plan="nope")
    out = capsys.readouterr().out
    assert "Unknown plan" in out


def test_reset(ledger_path, capsys) -> None:
    _seed()
    savings(reset=True)
    assert "reset" in capsys.readouterr().out.lower()
    assert core_savings.get_ledger().state.total_requests == 0
