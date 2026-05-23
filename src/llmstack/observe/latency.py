"""Latency percentile tracking (p50, p95, p99).

Tracks request latencies and computes percentile distributions for
performance monitoring and SLA compliance.
"""

from __future__ import annotations

import logging
import time
import threading
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class LatencyConfig:
    """Configuration for latency tracking."""

    # Maximum samples to keep in memory
    max_samples: int = 10000

    # Window size for rolling percentiles (seconds, 0 = all time)
    window_seconds: float = 3600.0


class LatencyTracker:
    """Tracks request latencies and computes percentiles.

    Maintains a sorted list of recent latency samples and computes
    p50, p95, p99, and other percentiles on demand.
    """

    def __init__(self, config: LatencyConfig | None = None):
        self.config = config or LatencyConfig()
        self._samples: list[tuple[float, float]] = []  # (timestamp, latency_ms)
        self._lock = threading.Lock()

    def record(self, latency_ms: float, label: str = "") -> None:
        """Record a latency measurement."""
        with self._lock:
            self._samples.append((time.time(), latency_ms))
            if len(self._samples) > self.config.max_samples:
                self._samples = self._samples[-self.config.max_samples:]

    def percentile(self, p: float) -> float:
        """Compute the p-th percentile of recorded latencies.

        Args:
            p: Percentile (0-100), e.g., 50 for p50, 95 for p95.

        Returns:
            Latency value at the given percentile, or 0.0 if no data.
        """
        values = self._get_window_values()
        if not values:
            return 0.0

        values.sort()
        idx = int(len(values) * p / 100)
        idx = min(idx, len(values) - 1)
        return values[idx]

    def get_percentiles(self) -> dict[str, float]:
        """Get common percentile values."""
        return {
            "p50": round(self.percentile(50), 2),
            "p75": round(self.percentile(75), 2),
            "p90": round(self.percentile(90), 2),
            "p95": round(self.percentile(95), 2),
            "p99": round(self.percentile(99), 2),
        }

    def get_summary(self) -> dict[str, Any]:
        """Get a full latency summary."""
        values = self._get_window_values()
        if not values:
            return {
                "count": 0,
                "percentiles": {},
                "mean": 0.0,
                "min": 0.0,
                "max": 0.0,
            }

        return {
            "count": len(values),
            "percentiles": self.get_percentiles(),
            "mean": round(sum(values) / len(values), 2),
            "min": round(min(values), 2),
            "max": round(max(values), 2),
        }

    def reset(self) -> None:
        """Clear all recorded samples."""
        with self._lock:
            self._samples.clear()

    def _get_window_values(self) -> list[float]:
        """Get latency values within the configured time window."""
        with self._lock:
            if self.config.window_seconds <= 0:
                return [lat for _, lat in self._samples]
            cutoff = time.time() - self.config.window_seconds
            return [lat for ts, lat in self._samples if ts >= cutoff]
