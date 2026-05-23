"""Usage quotas — per-API-key and per-model usage limits.

Enforces configurable quotas on requests, tokens, and cost
per API key with daily/monthly reset periods.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from threading import Lock


class QuotaPeriod(str, Enum):
    DAILY = "daily"
    MONTHLY = "monthly"
    TOTAL = "total"


@dataclass
class QuotaLimit:
    """A quota limit configuration."""

    api_key: str = "*"             # "*" applies to all keys
    max_requests: int = 0          # 0 = unlimited
    max_tokens: int = 0            # 0 = unlimited
    max_cost_usd: float = 0.0     # 0 = unlimited
    period: QuotaPeriod = QuotaPeriod.DAILY
    model: str | None = None       # None = all models


@dataclass
class QuotaUsage:
    """Current usage for a quota."""

    requests: int = 0
    tokens: int = 0
    cost_usd: float = 0.0
    last_reset: float = 0.0
    entries: list[tuple[float, int, float]] = field(default_factory=list)

    def add(self, tokens: int, cost_usd: float) -> None:
        now = time.time()
        self.requests += 1
        self.tokens += tokens
        self.cost_usd += cost_usd
        self.entries.append((now, tokens, cost_usd))

    def get_usage_in_period(self, period: QuotaPeriod) -> tuple[int, int, float]:
        """Get (requests, tokens, cost) within the given period."""
        cutoff = _period_cutoff(period)
        reqs = 0
        toks = 0
        cost = 0.0
        for ts, t, c in self.entries:
            if ts >= cutoff:
                reqs += 1
                toks += t
                cost += c
        return reqs, toks, cost


class QuotaExceededError(Exception):
    """Raised when a usage quota is exceeded."""

    def __init__(self, message: str, quota_name: str = "", retry_after: int = 0):
        super().__init__(message)
        self.quota_name = quota_name
        self.retry_after = retry_after


class QuotaManager:
    """Manages and enforces usage quotas per API key."""

    def __init__(self):
        self._lock = Lock()
        self._limits: list[QuotaLimit] = []
        self._usage: dict[str, QuotaUsage] = defaultdict(QuotaUsage)

    def add_limit(self, limit: QuotaLimit) -> None:
        """Add a quota limit."""
        with self._lock:
            self._limits.append(limit)

    def remove_limits(self, api_key: str) -> int:
        """Remove all limits for an API key."""
        with self._lock:
            before = len(self._limits)
            self._limits = [lim for lim in self._limits if lim.api_key != api_key]
            return before - len(self._limits)

    def check(self, api_key: str, model: str = "") -> None:
        """Check if a request is allowed. Raises QuotaExceededError if not."""
        with self._lock:
            for limit in self._limits:
                if limit.api_key != "*" and limit.api_key != api_key:
                    continue
                if limit.model and limit.model != model:
                    continue

                usage_key = f"{api_key}:{limit.model or '*'}"
                usage = self._usage[usage_key]
                reqs, toks, cost = usage.get_usage_in_period(limit.period)

                if limit.max_requests > 0 and reqs >= limit.max_requests:
                    raise QuotaExceededError(
                        f"Request quota exceeded: {reqs}/{limit.max_requests} "
                        f"({limit.period.value})",
                        quota_name=f"{api_key}:requests",
                    )
                if limit.max_tokens > 0 and toks >= limit.max_tokens:
                    raise QuotaExceededError(
                        f"Token quota exceeded: {toks}/{limit.max_tokens} "
                        f"({limit.period.value})",
                        quota_name=f"{api_key}:tokens",
                    )
                if limit.max_cost_usd > 0 and cost >= limit.max_cost_usd:
                    raise QuotaExceededError(
                        f"Cost quota exceeded: ${cost:.4f}/${limit.max_cost_usd} "
                        f"({limit.period.value})",
                        quota_name=f"{api_key}:cost",
                    )

    def record_usage(
        self, api_key: str, model: str = "", tokens: int = 0, cost_usd: float = 0.0,
    ) -> None:
        """Record usage after a successful request."""
        with self._lock:
            # Record for specific model
            key = f"{api_key}:{model or '*'}"
            self._usage[key].add(tokens, cost_usd)
            # Also record for wildcard
            if model:
                wildcard_key = f"{api_key}:*"
                self._usage[wildcard_key].add(tokens, cost_usd)

    def get_usage(self, api_key: str) -> dict:
        """Get current usage for an API key."""
        with self._lock:
            results = {}
            for key, usage in self._usage.items():
                if key.startswith(f"{api_key}:"):
                    model_part = key.split(":", 1)[1]
                    daily_r, daily_t, daily_c = usage.get_usage_in_period(QuotaPeriod.DAILY)
                    monthly_r, monthly_t, monthly_c = usage.get_usage_in_period(QuotaPeriod.MONTHLY)
                    results[model_part] = {
                        "daily": {"requests": daily_r, "tokens": daily_t, "cost_usd": round(daily_c, 6)},
                        "monthly": {"requests": monthly_r, "tokens": monthly_t, "cost_usd": round(monthly_c, 6)},
                        "total": {"requests": usage.requests, "tokens": usage.tokens, "cost_usd": round(usage.cost_usd, 6)},
                    }
            return results

    def get_limits(self) -> list[dict]:
        """Get all configured limits."""
        with self._lock:
            return [
                {
                    "api_key": lim.api_key,
                    "max_requests": lim.max_requests,
                    "max_tokens": lim.max_tokens,
                    "max_cost_usd": lim.max_cost_usd,
                    "period": lim.period.value,
                    "model": lim.model,
                }
                for lim in self._limits
            ]


def _period_cutoff(period: QuotaPeriod) -> float:
    now = time.time()
    if period == QuotaPeriod.DAILY:
        return now - 86400
    elif period == QuotaPeriod.MONTHLY:
        return now - 2592000
    return 0.0
