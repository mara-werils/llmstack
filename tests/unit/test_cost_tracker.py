"""Tests for cost tracking and budget management."""

import pytest

from llmstack.gateway.cost_tracker import (
    CostTracker,
    Budget,
    BudgetPeriod,
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


class TestModelPricing:
    def test_pricing_table_has_entries(self):
        assert len(MODEL_PRICING) >= 10

    def test_local_models_are_free(self):
        for model in ["llama3.2", "llama3.2:1b", "mistral", "codellama"]:
            assert MODEL_PRICING[model] == (0.0, 0.0)
