"""Quality tracker — rolling window quality monitoring with drift detection and alerts.

Tracks quality scores over time and fires alerts when quality degrades
beyond configurable thresholds.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from threading import Lock


@dataclass
class QualityAlert:
    """An alert fired when quality degrades."""

    metric: str              # "coherence", "relevance", "overall", etc.
    model: str = ""
    provider: str = ""
    current_value: float = 0.0
    threshold: float = 0.0
    window_size: int = 0
    message: str = ""
    severity: str = "warning"  # "warning" or "critical"
    timestamp: float = 0.0

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.time()

    def to_dict(self) -> dict:
        return {
            "metric": self.metric,
            "model": self.model,
            "provider": self.provider,
            "current_value": round(self.current_value, 4),
            "threshold": round(self.threshold, 4),
            "severity": self.severity,
            "message": self.message,
            "timestamp": self.timestamp,
        }


@dataclass
class _QualityWindow:
    """Rolling window of quality scores for a single metric."""

    values: deque = field(default_factory=lambda: deque(maxlen=200))
    timestamps: deque = field(default_factory=lambda: deque(maxlen=200))

    def add(self, value: float, ts: float | None = None) -> None:
        self.values.append(value)
        self.timestamps.append(ts or time.time())

    def mean(self) -> float:
        if not self.values:
            return 0.0
        return sum(self.values) / len(self.values)

    def recent_mean(self, n: int = 20) -> float:
        """Mean of the most recent n values."""
        vals = list(self.values)[-n:]
        return sum(vals) / len(vals) if vals else 0.0

    def trend(self, window: int = 50) -> float:
        """Simple trend: recent mean - older mean. Negative = degrading."""
        vals = list(self.values)
        if len(vals) < window:
            return 0.0
        mid = len(vals) // 2
        older = sum(vals[:mid]) / mid
        recent = sum(vals[mid:]) / (len(vals) - mid)
        return recent - older

    def count(self) -> int:
        return len(self.values)


class QualityTracker:
    """Tracks quality scores per model and fires alerts on degradation.

    Monitors:
    - Overall quality score (weighted aggregate)
    - Individual metrics (coherence, relevance, refusal, toxicity, repetition)
    - Per-model and per-provider quality

    Alerts when:
    - Recent quality drops below absolute threshold
    - Quality trend is negative beyond drift threshold
    """

    def __init__(
        self,
        alert_threshold: float = 0.4,
        drift_threshold: float = -0.1,
        window_size: int = 200,
    ):
        self._lock = Lock()
        self._alert_threshold = alert_threshold
        self._drift_threshold = drift_threshold
        self._window_size = window_size

        # Global windows
        self._global: dict[str, _QualityWindow] = {}
        # Per-model windows
        self._per_model: dict[str, dict[str, _QualityWindow]] = {}

        self._alerts: deque[QualityAlert] = deque(maxlen=100)

    def record(
        self,
        scores: dict[str, float],
        model: str = "",
        provider: str = "",
    ) -> list[QualityAlert]:
        """Record quality scores and return any triggered alerts."""
        alerts: list[QualityAlert] = []

        with self._lock:
            ts = time.time()

            for metric, value in scores.items():
                # Global tracking
                if metric not in self._global:
                    self._global[metric] = _QualityWindow(
                        values=deque(maxlen=self._window_size),
                        timestamps=deque(maxlen=self._window_size),
                    )
                self._global[metric].add(value, ts)

                # Per-model tracking
                if model:
                    if model not in self._per_model:
                        self._per_model[model] = {}
                    if metric not in self._per_model[model]:
                        self._per_model[model][metric] = _QualityWindow(
                            values=deque(maxlen=self._window_size),
                            timestamps=deque(maxlen=self._window_size),
                        )
                    self._per_model[model][metric].add(value, ts)

            # Check for alerts (only after enough data)
            for metric in ["overall", "coherence", "relevance"]:
                window = self._global.get(metric)
                if window is None or window.count() < 10:
                    continue

                # Absolute threshold check
                recent = window.recent_mean(20)
                if recent < self._alert_threshold:
                    alert = QualityAlert(
                        metric=metric,
                        model=model,
                        provider=provider,
                        current_value=recent,
                        threshold=self._alert_threshold,
                        window_size=window.count(),
                        severity="critical" if recent < self._alert_threshold * 0.5 else "warning",
                        message=f"{metric} quality dropped to {recent:.3f} "
                                f"(threshold: {self._alert_threshold})",
                        timestamp=ts,
                    )
                    alerts.append(alert)
                    self._alerts.append(alert)

                # Drift detection
                trend = window.trend(50)
                if trend < self._drift_threshold and window.count() >= 50:
                    alert = QualityAlert(
                        metric=metric,
                        model=model,
                        provider=provider,
                        current_value=trend,
                        threshold=self._drift_threshold,
                        window_size=window.count(),
                        severity="warning",
                        message=f"{metric} quality drifting: {trend:+.3f} over last {window.count()} requests",
                        timestamp=ts,
                    )
                    alerts.append(alert)
                    self._alerts.append(alert)

        return alerts

    def get_alerts(self, limit: int = 20) -> list[QualityAlert]:
        """Return recent alerts."""
        with self._lock:
            return list(self._alerts)[-limit:]

    def summary(self) -> dict:
        """Return quality summary across all metrics and models."""
        with self._lock:
            global_summary = {}
            for metric, window in self._global.items():
                global_summary[metric] = {
                    "mean": round(window.mean(), 4),
                    "recent": round(window.recent_mean(20), 4),
                    "trend": round(window.trend(50), 4),
                    "count": window.count(),
                }

            model_summary = {}
            for model, metrics in self._per_model.items():
                model_summary[model] = {
                    m: {
                        "mean": round(w.mean(), 4),
                        "recent": round(w.recent_mean(20), 4),
                        "count": w.count(),
                    }
                    for m, w in metrics.items()
                }

            return {
                "global": global_summary,
                "by_model": model_summary,
                "alerts": [a.to_dict() for a in list(self._alerts)[-10:]],
            }
