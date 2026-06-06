"""Streaming analytics — track and optimize SSE streaming performance.

Monitors time-to-first-token (TTFT), inter-token latency, throughput,
and streaming health for real-time performance insights.
"""

from __future__ import annotations

import statistics
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from threading import Lock


@dataclass
class StreamMetrics:
    """Metrics for a single streaming request."""

    request_id: str = ""
    model: str = ""
    provider: str = ""
    ttft_ms: float = 0.0  # Time to first token
    total_duration_ms: float = 0.0
    token_count: int = 0
    chunk_count: int = 0
    inter_token_latencies: list[float] = field(default_factory=list)

    @property
    def tokens_per_second(self) -> float:
        if self.total_duration_ms <= 0:
            return 0.0
        return (self.token_count / self.total_duration_ms) * 1000

    @property
    def avg_inter_token_ms(self) -> float:
        if not self.inter_token_latencies:
            return 0.0
        return statistics.mean(self.inter_token_latencies)

    @property
    def p50_inter_token_ms(self) -> float:
        if not self.inter_token_latencies:
            return 0.0
        return statistics.median(self.inter_token_latencies)

    @property
    def p95_inter_token_ms(self) -> float:
        if len(self.inter_token_latencies) < 2:
            return self.avg_inter_token_ms
        sorted_l = sorted(self.inter_token_latencies)
        idx = min(int(len(sorted_l) * 0.95), len(sorted_l) - 1)
        return sorted_l[idx]

    @property
    def p99_inter_token_ms(self) -> float:
        """Return the 99th percentile inter-token latency in milliseconds."""
        if len(self.inter_token_latencies) < 2:
            return self.avg_inter_token_ms
        sorted_l = sorted(self.inter_token_latencies)
        idx = min(int(len(sorted_l) * 0.99), len(sorted_l) - 1)
        return sorted_l[idx]

    def to_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "model": self.model,
            "provider": self.provider,
            "ttft_ms": round(self.ttft_ms, 1),
            "total_duration_ms": round(self.total_duration_ms, 1),
            "token_count": self.token_count,
            "chunk_count": self.chunk_count,
            "tokens_per_second": round(self.tokens_per_second, 1),
            "avg_inter_token_ms": round(self.avg_inter_token_ms, 1),
            "p95_inter_token_ms": round(self.p95_inter_token_ms, 1),
        }


class StreamingTracker:
    """Tracks streaming performance across requests and models."""

    def __init__(self, max_records: int = 2000):
        self._lock = Lock()
        self._records: deque[StreamMetrics] = deque(maxlen=max_records)
        self._active_streams: dict[str, dict] = {}  # request_id -> tracking state

    def start_stream(self, request_id: str, model: str = "", provider: str = "") -> None:
        """Mark the start of a streaming request."""
        with self._lock:
            self._active_streams[request_id] = {
                "model": model,
                "provider": provider,
                "start_time": time.monotonic(),
                "first_token_time": None,
                "last_token_time": None,
                "token_count": 0,
                "chunk_count": 0,
                "inter_token_latencies": [],
            }

    def record_chunk(self, request_id: str, token_count: int = 1) -> None:
        """Record a streaming chunk arrival."""
        now = time.monotonic()
        with self._lock:
            state = self._active_streams.get(request_id)
            if state is None:
                return

            if state["first_token_time"] is None:
                state["first_token_time"] = now
            elif state["last_token_time"] is not None:
                itl = (now - state["last_token_time"]) * 1000
                state["inter_token_latencies"].append(itl)

            state["last_token_time"] = now
            state["token_count"] += token_count
            state["chunk_count"] += 1

    def end_stream(self, request_id: str) -> StreamMetrics | None:
        """Finalize a streaming request and record metrics."""
        with self._lock:
            state = self._active_streams.pop(request_id, None)
            if state is None:
                return None

            now = time.monotonic()
            start = state["start_time"]
            first = state["first_token_time"]

            metrics = StreamMetrics(
                request_id=request_id,
                model=state["model"],
                provider=state["provider"],
                ttft_ms=(first - start) * 1000 if first else 0,
                total_duration_ms=(now - start) * 1000,
                token_count=state["token_count"],
                chunk_count=state["chunk_count"],
                inter_token_latencies=state["inter_token_latencies"],
            )
            self._records.append(metrics)
            return metrics

    def get_summary(self, model: str | None = None) -> dict:
        """Get streaming performance summary."""
        with self._lock:
            records = [r for r in self._records if model is None or r.model == model]

        if not records:
            return {"total_streams": 0, "active_streams": len(self._active_streams)}

        ttfts = [r.ttft_ms for r in records if r.ttft_ms > 0]
        tps_values = [r.tokens_per_second for r in records if r.tokens_per_second > 0]

        # Per-model breakdown
        by_model: dict[str, dict] = defaultdict(lambda: {"count": 0, "ttfts": [], "tps": []})
        for r in records:
            by_model[r.model]["count"] += 1
            if r.ttft_ms > 0:
                by_model[r.model]["ttfts"].append(r.ttft_ms)
            if r.tokens_per_second > 0:
                by_model[r.model]["tps"].append(r.tokens_per_second)

        model_stats = {}
        for m, data in by_model.items():
            model_stats[m] = {
                "streams": data["count"],
                "avg_ttft_ms": round(statistics.mean(data["ttfts"]), 1) if data["ttfts"] else 0,
                "avg_tps": round(statistics.mean(data["tps"]), 1) if data["tps"] else 0,
            }

        return {
            "total_streams": len(records),
            "active_streams": len(self._active_streams),
            "avg_ttft_ms": round(statistics.mean(ttfts), 1) if ttfts else 0,
            "p95_ttft_ms": round(
                sorted(ttfts)[int(len(ttfts) * 0.95)]
                if len(ttfts) >= 2
                else (ttfts[0] if ttfts else 0),
                1,
            ),
            "avg_tokens_per_second": round(statistics.mean(tps_values), 1) if tps_values else 0,
            "by_model": model_stats,
        }

    def get_recent(self, limit: int = 20) -> list[dict]:
        """Get recent streaming metrics."""
        with self._lock:
            records = list(self._records)[-limit:]
        return [r.to_dict() for r in reversed(records)]
