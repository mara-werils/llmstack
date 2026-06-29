"""Tests for cost tracking and budget management."""

import time

import pytest

from llmstack.gateway.cost_tracker import (
    Budget,
    BudgetAlert,
    BudgetPeriod,
    CostEntry,
    CostTracker,
    MODEL_PRICING,
)


@pytest.fixture
def tracker():
    return CostTracker()


class TestCostCalculation:
    def test_known_model_pricing(self, tracker):
        cost = tracker.calculate_cost("gpt-4o", input_tokens=1000, output_tokens=500)
        expected = (1000 / 1_000_000) * 2.50 + (500 / 1_000_000) * 10.00
        assert abs(cost - expected) < 1e-10

    def test_local_model_free(self, tracker):
        cost = tracker.calculate_cost("llama3.2", input_tokens=10000, output_tokens=5000)
        assert cost == 0.0

    def test_unknown_model_zero(self, tracker):
        assert tracker.calculate_cost("unknown-model", 100, 100) == 0.0

    def test_custom_pricing(self, tracker):
        tracker.set_pricing("my-model", 1.0, 2.0)
        cost = tracker.calculate_cost("my-model", 1_000_000, 1_000_000)
        assert cost == 3.0

    def test_prefix_matching_pricing(self, tracker):
        cost = tracker.calculate_cost("gpt-4o-2024-08-06", input_tokens=1_000_000, output_tokens=0)
        assert cost == 2.50

    def test_prefix_matching_prefers_longest_key(self, tracker):
        # "gpt-4o-mini-2024-..." must match gpt-4o-mini ($0.15/1M), not the
        # shorter, 16x-pricier gpt-4o prefix ($2.50/1M).
        cost = tracker.calculate_cost(
            "gpt-4o-mini-2024-07-18", input_tokens=1_000_000, output_tokens=0
        )
        assert cost == 0.15

    def test_custom_pricing_prefix_match(self, tracker):
        # A versioned variant of a custom-priced model inherits the custom price.
        tracker.set_pricing("acme-llm", 1.0, 2.0)
        cost = tracker.calculate_cost("acme-llm-v2-2026", input_tokens=1_000_000, output_tokens=0)
        assert cost == 1.0

    def test_custom_pricing_overrides_builtin_prefix(self, tracker):
        # Custom pricing wins over a built-in entry sharing the same prefix.
        tracker.set_pricing("gpt-4o", 0.01, 0.02)
        cost = tracker.calculate_cost("gpt-4o-custom-tag", input_tokens=1_000_000, output_tokens=0)
        assert cost == 0.01


class TestCostRecording:
    def test_record_entry(self, tracker):
        entry = tracker.record("gpt-4o", "openai", 100, 50)
        assert entry.model == "gpt-4o"
        assert entry.cost_usd >= 0

    def test_record_with_explicit_cost(self, tracker):
        entry = tracker.record("test", "local", 100, 50, cost_usd=0.05)
        assert entry.cost_usd == 0.05

    def test_summary(self, tracker):
        tracker.record("gpt-4o", "openai", 1000, 500, cost_usd=0.01)
        tracker.record("llama3.2", "local", 1000, 500, cost_usd=0.0)

        summary = tracker.get_summary()
        assert summary["total_requests"] == 2
        assert summary["total_cost_usd"] == 0.01
        assert "gpt-4o" in summary["cost_by_model"]

    def test_empty_summary(self, tracker):
        summary = tracker.get_summary()
        assert summary["total_cost_usd"] == 0
        assert summary["total_requests"] == 0

    def test_total_cost_and_requests_properties(self, tracker):
        tracker.record("gpt-4o", "openai", 1000, 500, cost_usd=0.02)
        tracker.record("gpt-4o", "openai", 1000, 500, cost_usd=0.03)
        assert tracker.total_cost_usd == pytest.approx(0.05)
        assert tracker.total_requests == 2


class TestBudgets:
    def test_add_and_list_budget(self, tracker):
        budget = Budget(name="monthly-limit", limit_usd=100.0, period=BudgetPeriod.MONTHLY)
        tracker.add_budget(budget)
        budgets = tracker.get_budgets()
        assert len(budgets) == 1
        assert budgets[0]["name"] == "monthly-limit"

    def test_remove_budget(self, tracker):
        budget = Budget(name="temp", limit_usd=50.0)
        tracker.add_budget(budget)
        assert tracker.remove_budget("temp") is True
        assert tracker.remove_budget("nonexistent") is False

    def test_budget_alert_triggered(self, tracker):
        budget = Budget(
            name="test-budget",
            limit_usd=0.001,
            period=BudgetPeriod.TOTAL,
            alert_at_percent=50.0,
        )
        tracker.add_budget(budget)
        tracker.record("gpt-4o", "openai", 10000, 5000, cost_usd=0.001)

        alerts = tracker.get_alerts()
        assert len(alerts) >= 1
        assert alerts[0].budget_name == "test-budget"
        assert alerts[0].percent_used >= 50.0

    def test_model_specific_budget(self, tracker):
        budget = Budget(
            name="gpt4-budget",
            limit_usd=1.0,
            period=BudgetPeriod.TOTAL,
            model="gpt-4o",
        )
        tracker.add_budget(budget)
        tracker.record("llama3.2", "local", 1000, 500, cost_usd=0.5)
        # Should not trigger alert for llama3.2
        alerts = tracker.get_alerts()
        assert len(alerts) == 0

    def test_get_spend_by_period(self, tracker):
        tracker.record("gpt-4o", "openai", 100, 50, cost_usd=0.01)
        daily = tracker.get_spend(BudgetPeriod.DAILY)
        assert daily == 0.01
        total = tracker.get_spend(BudgetPeriod.TOTAL)
        assert total == 0.01

    def test_get_spend_excludes_entries_outside_period(self, tracker):
        old_entry = CostEntry(
            timestamp=time.time() - 90000,  # > 1 day old
            model="gpt-4o",
            provider="openai",
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.01,
        )
        tracker._entries.append(old_entry)
        assert tracker.get_spend(BudgetPeriod.DAILY) == 0.0

    def test_get_spend_filters_by_model_and_provider(self, tracker):
        tracker.record("gpt-4o", "openai", 100, 50, cost_usd=0.01)
        tracker.record("claude-sonnet-4-20250514", "anthropic", 100, 50, cost_usd=0.02)

        assert tracker.get_spend(BudgetPeriod.TOTAL, model="gpt-4o") == 0.01
        assert tracker.get_spend(BudgetPeriod.TOTAL, model="nonexistent") == 0.0
        assert tracker.get_spend(BudgetPeriod.TOTAL, provider="anthropic") == 0.02
        assert tracker.get_spend(BudgetPeriod.TOTAL, provider="nonexistent") == 0.0

    def test_provider_specific_budget(self, tracker):
        budget = Budget(
            name="openai-budget",
            limit_usd=1.0,
            period=BudgetPeriod.TOTAL,
            provider="openai",
        )
        tracker.add_budget(budget)
        # Different provider should be skipped by the budget-level filter
        tracker.record("claude-sonnet-4-20250514", "anthropic", 1000, 500, cost_usd=0.5)
        assert tracker.get_alerts() == []

    def test_budget_check_ignores_entries_outside_period(self, tracker):
        old_entry = CostEntry(
            timestamp=time.time() - 90000,  # > 1 day old
            model="gpt-4o",
            provider="openai",
            input_tokens=1000,
            output_tokens=500,
            cost_usd=10.0,  # would blow the budget if counted
        )
        tracker._entries.append(old_entry)

        budget = Budget(
            name="daily-budget",
            limit_usd=1.0,
            period=BudgetPeriod.DAILY,
            alert_at_percent=50.0,
        )
        tracker.add_budget(budget)

        tracker.record("gpt-4o", "openai", 100, 50, cost_usd=0.01)

        alerts = tracker.get_alerts()
        assert not any(a.budget_name == "daily-budget" for a in alerts)

    def test_budget_spend_ignores_non_matching_entries(self, tracker):
        budget = Budget(
            name="gpt4-budget",
            limit_usd=0.01,
            period=BudgetPeriod.TOTAL,
            model="gpt-4o",
            provider="openai",
            alert_at_percent=50.0,
        )
        tracker.add_budget(budget)
        # Non-matching entries (wrong model, wrong provider) recorded first.
        tracker.record("llama3.2", "local", 1000, 500, cost_usd=0.0)
        tracker.record("gpt-4o", "anthropic", 1000, 500, cost_usd=0.5)
        # Matching entry triggers the alert based only on matching spend.
        tracker.record("gpt-4o", "openai", 1000, 500, cost_usd=0.01)

        alerts = tracker.get_alerts()
        assert any(a.budget_name == "gpt4-budget" for a in alerts)


class TestBudgetAlert:
    def test_to_dict(self):
        alert = BudgetAlert(
            budget_name="test",
            current_spend=0.012345,
            limit_usd=1.0,
            percent_used=1.2345,
        )
        d = alert.to_dict()
        assert d["budget_name"] == "test"
        assert d["current_spend"] == 0.012345
        assert d["percent_used"] == 1.23
        assert d["triggered_at"] > 0


class TestModelPricing:
    def test_pricing_table_has_entries(self):
        assert len(MODEL_PRICING) >= 10

    def test_local_models_are_free(self):
        for model in ["llama3.2", "llama3.2:1b", "mistral", "codellama"]:
            assert MODEL_PRICING[model] == (0.0, 0.0)
