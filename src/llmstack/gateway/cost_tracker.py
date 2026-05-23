"""Cost tracking and budget management for LLM API usage.

Tracks per-model, per-provider cost with configurable budget limits
and alerts. Supports daily, weekly, and monthly budget periods.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from threading import RLock
from typing import Any


class BudgetPeriod(str, Enum):
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    TOTAL = "total"


# Default pricing per 1M tokens (input/output) for common models
MODEL_PRICING: dict[str, tuple[float, float]] = {
    # OpenAI
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4-turbo": (10.00, 30.00),
    "gpt-3.5-turbo": (0.50, 1.50),
    # Anthropic
    "claude-opus-4-20250514": (15.00, 75.00),
    "claude-sonnet-4-20250514": (3.00, 15.00),
    "claude-haiku-4-5-20251001": (1.00, 5.00),
    # Google
    "gemini-2.0-flash": (0.075, 0.30),
    "gemini-2.5-pro": (1.25, 10.00),
    # Groq (hosted)
    "llama-3.3-70b-versatile": (0.59, 0.79),
    "llama-3.1-8b-instant": (0.05, 0.08),
    # Local (free)
    "llama3.2": (0.0, 0.0),
    "llama3.2:1b": (0.0, 0.0),
    "llama3.1:70b": (0.0, 0.0),
    "mistral": (0.0, 0.0),
    "codellama": (0.0, 0.0),
}


@dataclass
class CostEntry:
    """A single cost record."""

    timestamp: float
    model: str
    provider: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    request_id: str = ""


@dataclass
class Budget:
    """A budget limit configuration."""

    name: str
    limit_usd: float
    period: BudgetPeriod = BudgetPeriod.MONTHLY
    model: str | None = None       # None = applies to all models
    provider: str | None = None    # None = applies to all providers
    alert_at_percent: float = 80.0  # alert when usage hits this %


@dataclass
class BudgetAlert:
    """An alert triggered when budget threshold is reached."""

    budget_name: str
    current_spend: float
    limit_usd: float
    percent_used: float
    triggered_at: float = 0.0

    def __post_init__(self):
        if not self.triggered_at:
            self.triggered_at = time.time()

    def to_dict(self) -> dict:
        return {
            "budget_name": self.budget_name,
            "current_spend": round(self.current_spend, 6),
            "limit_usd": self.limit_usd,
            "percent_used": round(self.percent_used, 2),
            "triggered_at": self.triggered_at,
        }


class CostTracker:
    """Tracks LLM API costs with budget enforcement and alerting."""

    def __init__(self):
        self._lock = RLock()
        self._entries: list[CostEntry] = []
        self._budgets: dict[str, Budget] = {}
        self._alerts: list[BudgetAlert] = []
        self._custom_pricing: dict[str, tuple[float, float]] = {}

    def set_pricing(self, model: str, input_per_m: float, output_per_m: float) -> None:
        """Set custom pricing for a model (per 1M tokens)."""
        with self._lock:
            self._custom_pricing[model] = (input_per_m, output_per_m)

    def calculate_cost(
        self, model: str, input_tokens: int, output_tokens: int,
    ) -> float:
        """Calculate cost for a request based on token counts."""
        pricing = self._custom_pricing.get(model) or MODEL_PRICING.get(model)
        if pricing is None:
            # Try prefix matching (e.g., "gpt-4o-2024-..." -> "gpt-4o")
            for key, val in MODEL_PRICING.items():
                if model.startswith(key):
                    pricing = val
                    break
        if pricing is None:
            return 0.0

        input_cost = (input_tokens / 1_000_000) * pricing[0]
        output_cost = (output_tokens / 1_000_000) * pricing[1]
        return input_cost + output_cost

    def record(
        self,
        model: str,
        provider: str,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float | None = None,
        request_id: str = "",
    ) -> CostEntry:
        """Record a cost entry and check budget limits."""
        if cost_usd is None:
            cost_usd = self.calculate_cost(model, input_tokens, output_tokens)

        entry = CostEntry(
            timestamp=time.time(),
            model=model,
            provider=provider,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            request_id=request_id,
        )

        with self._lock:
            self._entries.append(entry)
            self._check_budgets(entry)

        return entry

    def add_budget(self, budget: Budget) -> None:
        """Add or update a budget limit."""
        with self._lock:
            self._budgets[budget.name] = budget

    def remove_budget(self, name: str) -> bool:
        """Remove a budget."""
        with self._lock:
            return self._budgets.pop(name, None) is not None

    def get_spend(
        self,
        period: BudgetPeriod = BudgetPeriod.TOTAL,
        model: str | None = None,
        provider: str | None = None,
    ) -> float:
        """Get total spend for a period with optional filters."""
        cutoff = self._period_cutoff(period)
        with self._lock:
            total = 0.0
            for e in self._entries:
                if e.timestamp < cutoff:
                    continue
                if model and e.model != model:
                    continue
                if provider and e.provider != provider:
                    continue
                total += e.cost_usd
            return total

    def get_summary(self) -> dict[str, Any]:
        """Get a comprehensive cost summary."""
        with self._lock:
            if not self._entries:
                return {"total_cost_usd": 0, "total_requests": 0}

            total_cost = sum(e.cost_usd for e in self._entries)
            total_input = sum(e.input_tokens for e in self._entries)
            total_output = sum(e.output_tokens for e in self._entries)

            by_model: dict[str, float] = defaultdict(float)
            by_provider: dict[str, float] = defaultdict(float)
            for e in self._entries:
                by_model[e.model] += e.cost_usd
                by_provider[e.provider] += e.cost_usd

            return {
                "total_cost_usd": round(total_cost, 6),
                "total_requests": len(self._entries),
                "total_input_tokens": total_input,
                "total_output_tokens": total_output,
                "cost_by_model": {
                    k: round(v, 6) for k, v in sorted(
                        by_model.items(), key=lambda x: -x[1]
                    )
                },
                "cost_by_provider": {
                    k: round(v, 6) for k, v in sorted(
                        by_provider.items(), key=lambda x: -x[1]
                    )
                },
                "daily_spend": round(
                    self.get_spend(BudgetPeriod.DAILY), 6
                ),
                "weekly_spend": round(
                    self.get_spend(BudgetPeriod.WEEKLY), 6
                ),
                "monthly_spend": round(
                    self.get_spend(BudgetPeriod.MONTHLY), 6
                ),
            }

    def get_alerts(self, limit: int = 50) -> list[BudgetAlert]:
        """Get recent budget alerts."""
        with self._lock:
            return list(reversed(self._alerts[-limit:]))

    def get_budgets(self) -> list[dict]:
        """Get all configured budgets with current spend."""
        with self._lock:
            results = []
            for b in self._budgets.values():
                spend = self.get_spend(b.period, model=b.model, provider=b.provider)
                results.append({
                    "name": b.name,
                    "limit_usd": b.limit_usd,
                    "period": b.period.value,
                    "model": b.model,
                    "provider": b.provider,
                    "current_spend": round(spend, 6),
                    "percent_used": round((spend / b.limit_usd) * 100, 2) if b.limit_usd > 0 else 0,
                    "alert_at_percent": b.alert_at_percent,
                })
            return results

    def _check_budgets(self, entry: CostEntry) -> None:
        """Check all budgets against current spend (caller must hold _lock)."""
        for budget in self._budgets.values():
            if budget.model and budget.model != entry.model:
                continue
            if budget.provider and budget.provider != entry.provider:
                continue

            spend = 0.0
            cutoff = self._period_cutoff(budget.period)
            for e in self._entries:
                if e.timestamp < cutoff:
                    continue
                if budget.model and e.model != budget.model:
                    continue
                if budget.provider and e.provider != budget.provider:
                    continue
                spend += e.cost_usd

            percent = (spend / budget.limit_usd) * 100 if budget.limit_usd > 0 else 0
            if percent >= budget.alert_at_percent:
                alert = BudgetAlert(
                    budget_name=budget.name,
                    current_spend=spend,
                    limit_usd=budget.limit_usd,
                    percent_used=percent,
                )
                self._alerts.append(alert)

    @staticmethod
    def _period_cutoff(period: BudgetPeriod) -> float:
        """Get the timestamp cutoff for a budget period."""
        now = time.time()
        if period == BudgetPeriod.DAILY:
            return now - 86400
        elif period == BudgetPeriod.WEEKLY:
            return now - 604800
        elif period == BudgetPeriod.MONTHLY:
            return now - 2592000
        return 0.0  # TOTAL — no cutoff
