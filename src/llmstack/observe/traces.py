"""Request tracing — full lifecycle trace of each LLM request.

Captures: prompt, routing decision, model, provider, response, latency,
tokens, cost, quality scores, and metadata.
"""

from __future__ import annotations

import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from threading import Lock
from typing import Any


@dataclass
class Trace:
    """A single request trace through the LLM pipeline."""

    id: str = ""
    timestamp: float = 0.0

    # Request
    model: str = ""
    provider: str = ""
    messages: list[dict] = field(default_factory=list)
    temperature: float = 0.0
    stream: bool = False

    # Routing
    routed_model: str = ""
    routed_tier: str = ""
    routing_time_ms: float = 0.0

    # Response
    response: str = ""
    finish_reason: str = ""

    # Metrics
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    cached: bool = False

    # Quality scores (filled by QualityScorer)
    quality: dict[str, float] = field(default_factory=dict)

    # Metadata
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())[:12]
        if not self.timestamp:
            self.timestamp = time.time()

    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "model": self.model,
            "provider": self.provider,
            "routed_model": self.routed_model,
            "routed_tier": self.routed_tier,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "latency_ms": round(self.latency_ms, 1),
            "cost_usd": round(self.cost_usd, 6),
            "cached": self.cached,
            "quality": self.quality,
            "finish_reason": self.finish_reason,
            "error": self.error,
        }


class TraceStore:
    """In-memory rolling store of recent traces.

    Thread-safe, bounded-size deque for fast appends and queries.
    """

    def __init__(self, max_size: int = 5000):
        self._lock = Lock()
        self._traces: deque[Trace] = deque(maxlen=max_size)
        self._total_count: int = 0

    def add(self, trace: Trace) -> None:
        with self._lock:
            self._traces.append(trace)
            self._total_count += 1

    def recent(self, limit: int = 50) -> list[Trace]:
        with self._lock:
            return list(self._traces)[-limit:]

    def query(
        self,
        model: str | None = None,
        provider: str | None = None,
        min_latency_ms: float | None = None,
        has_error: bool | None = None,
        limit: int = 100,
    ) -> list[Trace]:
        """Query traces with filters."""
        with self._lock:
            results = []
            for t in reversed(self._traces):
                if model and t.model != model:
                    continue
                if provider and t.provider != provider:
                    continue
                if min_latency_ms is not None and t.latency_ms < min_latency_ms:
                    continue
                if has_error is True and t.error is None:
                    continue
                if has_error is False and t.error is not None:
                    continue
                results.append(t)
                if len(results) >= limit:
                    break
            return results

    def summary(self) -> dict:
        """Return aggregate statistics."""
        with self._lock:
            traces = list(self._traces)

        if not traces:
            return {"total": 0, "stored": 0}

        total_latency = sum(t.latency_ms for t in traces)
        total_tokens = sum(t.total_tokens() for t in traces)
        total_cost = sum(t.cost_usd for t in traces)
        errors = sum(1 for t in traces if t.error)
        cached = sum(1 for t in traces if t.cached)

        models: dict[str, int] = {}
        providers: dict[str, int] = {}
        for t in traces:
            models[t.model] = models.get(t.model, 0) + 1
            if t.provider:
                providers[t.provider] = providers.get(t.provider, 0) + 1

        # Quality averages
        quality_sums: dict[str, float] = {}
        quality_counts: dict[str, int] = {}
        for t in traces:
            for k, v in t.quality.items():
                quality_sums[k] = quality_sums.get(k, 0.0) + v
                quality_counts[k] = quality_counts.get(k, 0) + 1

        avg_quality = {k: round(quality_sums[k] / quality_counts[k], 4) for k in quality_sums}

        n = len(traces)
        return {
            "total": self._total_count,
            "stored": n,
            "avg_latency_ms": round(total_latency / n, 1),
            "total_tokens": total_tokens,
            "total_cost_usd": round(total_cost, 4),
            "error_rate": round(errors / n, 4),
            "cache_hit_rate": round(cached / n, 4),
            "models": models,
            "providers": providers,
            "avg_quality": avg_quality,
        }

    @property
    def total_count(self) -> int:
        return self._total_count
