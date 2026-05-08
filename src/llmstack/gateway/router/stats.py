"""Router statistics — tracks routing decisions for analytics and observability."""

from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from threading import Lock

from llmstack.gateway.router.router import RoutingDecision


@dataclass
class _RequestRecord:
    model: str
    tier: str
    score: float
    latency_ms: float
    timestamp: float


class RouterStats:
    """Thread-safe routing statistics tracker.

    Tracks per-model and per-tier request counts, latencies, and maintains
    a sliding window of recent decisions for inspection.
    """

    def __init__(self, history_size: int = 500):
        self._lock = Lock()
        self._model_counts: dict[str, int] = defaultdict(int)
        self._tier_counts: dict[str, int] = defaultdict(int)
        self._model_latencies: dict[str, list[float]] = defaultdict(list)
        self._tier_latencies: dict[str, list[float]] = defaultdict(list)
        self._total_requests: int = 0
        self._largest_model_avoided: int = 0
        self._history: deque[_RequestRecord] = deque(maxlen=history_size)
        self._largest_model: str | None = None

    def set_largest_model(self, model_name: str) -> None:
        """Configure which model counts as the 'large' model for savings calc."""
        with self._lock:
            self._largest_model = model_name

    def record(self, decision: RoutingDecision, latency_ms: float) -> None:
        """Record a completed routing decision."""
        with self._lock:
            self._total_requests += 1
            self._model_counts[decision.model] += 1
            self._tier_counts[decision.profile.tier] += 1
            self._model_latencies[decision.model].append(latency_ms)
            self._tier_latencies[decision.profile.tier].append(latency_ms)

            if self._largest_model and decision.model != self._largest_model:
                self._largest_model_avoided += 1

            self._history.append(_RequestRecord(
                model=decision.model,
                tier=decision.profile.tier,
                score=decision.profile.score,
                latency_ms=latency_ms,
                timestamp=time.time(),
            ))

    def summary(self) -> dict:
        """Return a snapshot of routing statistics."""
        with self._lock:
            total = max(self._total_requests, 1)

            model_dist = {m: {"count": c, "pct": round(c / total * 100, 1)}
                          for m, c in self._model_counts.items()}
            tier_dist = {t: {"count": c, "pct": round(c / total * 100, 1)}
                         for t, c in self._tier_counts.items()}

            avg_latency_by_model = {}
            for m, lats in self._model_latencies.items():
                if lats:
                    avg_latency_by_model[m] = round(sum(lats) / len(lats), 1)

            avg_latency_by_tier = {}
            for t, lats in self._tier_latencies.items():
                if lats:
                    avg_latency_by_tier[t] = round(sum(lats) / len(lats), 1)

            savings_pct = round(self._largest_model_avoided / total * 100, 1) if total else 0.0

            recent = [
                {
                    "model": r.model,
                    "tier": r.tier,
                    "score": r.score,
                    "latency_ms": round(r.latency_ms, 1),
                    "timestamp": r.timestamp,
                }
                for r in list(self._history)[-20:]
            ]

            return {
                "total_requests": self._total_requests,
                "model_distribution": model_dist,
                "tier_distribution": tier_dist,
                "avg_latency_by_model_ms": avg_latency_by_model,
                "avg_latency_by_tier_ms": avg_latency_by_tier,
                "estimated_savings_pct": savings_pct,
                "largest_model": self._largest_model,
                "recent_decisions": recent,
            }
