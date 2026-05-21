"""Model performance leaderboard — compare models on quality, speed, and cost.

Aggregates metrics from real usage to rank models and help users
choose the best model for their use case.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from threading import Lock
from typing import Any


@dataclass
class ModelMetrics:
    """Aggregated metrics for a single model."""

    model: str = ""
    provider: str = ""
    total_requests: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0

    # Latency
    latencies_ms: list[float] = field(default_factory=list)

    # Quality scores (0-1)
    quality_scores: list[float] = field(default_factory=list)

    # Errors
    error_count: int = 0

    # First and last seen
    first_seen: float = 0.0
    last_seen: float = 0.0

    def record(
        self,
        latency_ms: float,
        tokens: int = 0,
        cost_usd: float = 0.0,
        quality_score: float | None = None,
        error: bool = False,
    ) -> None:
        now = time.time()
        self.total_requests += 1
        self.total_tokens += tokens
        self.total_cost_usd += cost_usd
        self.latencies_ms.append(latency_ms)
        if quality_score is not None:
            self.quality_scores.append(quality_score)
        if error:
            self.error_count += 1
        if not self.first_seen:
            self.first_seen = now
        self.last_seen = now

    @property
    def avg_latency_ms(self) -> float:
        if not self.latencies_ms:
            return 0.0
        return sum(self.latencies_ms) / len(self.latencies_ms)

    @property
    def p50_latency_ms(self) -> float:
        return self._percentile(50)

    @property
    def p95_latency_ms(self) -> float:
        return self._percentile(95)

    @property
    def p99_latency_ms(self) -> float:
        return self._percentile(99)

    @property
    def avg_quality(self) -> float:
        if not self.quality_scores:
            return 0.0
        return sum(self.quality_scores) / len(self.quality_scores)

    @property
    def avg_cost_per_request(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.total_cost_usd / self.total_requests

    @property
    def error_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.error_count / self.total_requests

    @property
    def tokens_per_second(self) -> float:
        if not self.latencies_ms or self.total_tokens == 0:
            return 0.0
        total_seconds = sum(self.latencies_ms) / 1000
        return self.total_tokens / total_seconds if total_seconds > 0 else 0.0

    def _percentile(self, p: int) -> float:
        if not self.latencies_ms:
            return 0.0
        sorted_l = sorted(self.latencies_ms)
        idx = int(len(sorted_l) * p / 100)
        idx = min(idx, len(sorted_l) - 1)
        return sorted_l[idx]

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "provider": self.provider,
            "total_requests": self.total_requests,
            "total_tokens": self.total_tokens,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "p50_latency_ms": round(self.p50_latency_ms, 1),
            "p95_latency_ms": round(self.p95_latency_ms, 1),
            "p99_latency_ms": round(self.p99_latency_ms, 1),
            "avg_quality": round(self.avg_quality, 4),
            "avg_cost_per_request": round(self.avg_cost_per_request, 6),
            "error_rate": round(self.error_rate, 4),
            "tokens_per_second": round(self.tokens_per_second, 1),
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
        }


class Leaderboard:
    """Model performance leaderboard with ranking and comparison."""

    def __init__(self):
        self._lock = Lock()
        self._models: dict[str, ModelMetrics] = {}

    def record(
        self,
        model: str,
        provider: str = "local",
        latency_ms: float = 0.0,
        tokens: int = 0,
        cost_usd: float = 0.0,
        quality_score: float | None = None,
        error: bool = False,
    ) -> None:
        """Record a model usage event."""
        with self._lock:
            if model not in self._models:
                self._models[model] = ModelMetrics(model=model, provider=provider)
            self._models[model].record(
                latency_ms=latency_ms,
                tokens=tokens,
                cost_usd=cost_usd,
                quality_score=quality_score,
                error=error,
            )

    def get_rankings(
        self,
        sort_by: str = "quality",
        min_requests: int = 5,
    ) -> list[dict]:
        """Get model rankings sorted by the specified metric.

        sort_by: quality, latency, cost, speed, requests, error_rate
        """
        with self._lock:
            models = [
                m for m in self._models.values()
                if m.total_requests >= min_requests
            ]

        sort_keys = {
            "quality": lambda m: -m.avg_quality,
            "latency": lambda m: m.avg_latency_ms,
            "cost": lambda m: m.avg_cost_per_request,
            "speed": lambda m: -m.tokens_per_second,
            "requests": lambda m: -m.total_requests,
            "error_rate": lambda m: m.error_rate,
        }

        key_fn = sort_keys.get(sort_by, sort_keys["quality"])
        sorted_models = sorted(models, key=key_fn)

        return [
            {**m.to_dict(), "rank": i + 1}
            for i, m in enumerate(sorted_models)
        ]

    def compare(self, models: list[str]) -> list[dict]:
        """Compare specific models side by side."""
        with self._lock:
            results = []
            for name in models:
                m = self._models.get(name)
                if m:
                    results.append(m.to_dict())
            return results

    def get_model(self, model: str) -> dict | None:
        """Get detailed metrics for a specific model."""
        with self._lock:
            m = self._models.get(model)
            return m.to_dict() if m else None

    def get_summary(self) -> dict:
        """Get leaderboard summary."""
        with self._lock:
            return {
                "total_models": len(self._models),
                "total_requests": sum(m.total_requests for m in self._models.values()),
                "total_cost_usd": round(
                    sum(m.total_cost_usd for m in self._models.values()), 6
                ),
                "top_by_quality": self.get_rankings("quality", min_requests=1)[:3],
                "top_by_speed": self.get_rankings("speed", min_requests=1)[:3],
                "top_by_cost": self.get_rankings("cost", min_requests=1)[:3],
            }
