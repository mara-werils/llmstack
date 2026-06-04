"""Quality regression detection — automatic rollback when quality drops.

Continuously monitors model quality and triggers rollback if a new
version performs worse than the previous one. Uses statistical tests
to distinguish real regression from noise.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from llmstack.learn.store import FeedbackStore
from llmstack.learn.versions import ModelVersionManager

logger = logging.getLogger(__name__)


class RegressionSeverity(str, Enum):
    """Severity of detected regression."""

    NONE = "none"
    MILD = "mild"  # <5% drop, monitor
    MODERATE = "moderate"  # 5-15% drop, alert
    SEVERE = "severe"  # >15% drop, auto-rollback


@dataclass
class RegressionAlert:
    """A detected quality regression event."""

    timestamp: float = field(default_factory=time.time)
    severity: RegressionSeverity = RegressionSeverity.NONE
    model_version: str = ""
    metric: str = ""
    current_value: float = 0.0
    baseline_value: float = 0.0
    drop_percent: float = 0.0
    sample_size: int = 0
    confidence: float = 0.0
    auto_rolled_back: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "severity": self.severity.value,
            "model_version": self.model_version,
            "metric": self.metric,
            "current_value": round(self.current_value, 4),
            "baseline_value": round(self.baseline_value, 4),
            "drop_percent": round(self.drop_percent, 2),
            "sample_size": self.sample_size,
            "confidence": round(self.confidence, 4),
            "auto_rolled_back": self.auto_rolled_back,
        }


@dataclass
class RegressionConfig:
    """Configuration for regression detection."""

    # Minimum samples before making a judgment
    min_samples: int = 10

    # Window size for computing moving average
    window_size: int = 20

    # Thresholds for severity levels (as proportion of baseline)
    mild_threshold: float = 0.03  # 3% drop
    moderate_threshold: float = 0.08  # 8% drop
    severe_threshold: float = 0.15  # 15% drop

    # Confidence required to trigger alert (0-1)
    min_confidence: float = 0.7

    # Auto-rollback on severe regression
    auto_rollback: bool = True

    # Metrics to monitor
    monitored_metrics: list[str] = field(
        default_factory=lambda: ["overall", "coherence", "relevance"]
    )


class RegressionDetector:
    """Monitors model quality and detects regression.

    Uses a sliding window of quality scores to detect when a model
    version is performing worse than its baseline. Applies statistical
    confidence checks to avoid false positives.
    """

    def __init__(
        self,
        store: FeedbackStore,
        version_mgr: ModelVersionManager,
        config: RegressionConfig | None = None,
    ):
        self.store = store
        self.version_mgr = version_mgr
        self.config = config or RegressionConfig()
        self._alerts: list[RegressionAlert] = []

    @property
    def alerts(self) -> list[RegressionAlert]:
        return self._alerts

    def check(self) -> list[RegressionAlert]:
        """Run regression check on the active model version.

        Returns list of alerts (empty if no regression detected).
        """
        active = self.version_mgr.get_active()
        if not active:
            return []

        alerts: list[RegressionAlert] = []

        for metric in self.config.monitored_metrics:
            alert = self._check_metric(active.version, metric, active.quality_score)
            if alert and alert.severity != RegressionSeverity.NONE:
                alerts.append(alert)

                if alert.severity == RegressionSeverity.SEVERE and self.config.auto_rollback:
                    self._handle_rollback(alert)

        self._alerts.extend(alerts)
        return alerts

    def record_quality(
        self,
        model_version: str,
        metric: str,
        value: float,
        sample_size: int = 1,
    ) -> None:
        """Record a quality measurement for the active version."""
        self.store.add_quality_snapshot(
            model_version=model_version,
            metric=metric,
            value=value,
            sample_size=sample_size,
        )

    def get_health(self) -> dict[str, Any]:
        """Get current quality health status."""
        active = self.version_mgr.get_active()
        if not active:
            return {"status": "no_active_model", "metrics": {}}

        metrics: dict[str, Any] = {}
        for metric in self.config.monitored_metrics:
            trend = self.store.get_quality_trend(active.version, metric, limit=20)
            if trend:
                values = [t["value"] for t in trend]
                metrics[metric] = {
                    "current": values[0] if values else 0.0,
                    "mean": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "samples": len(values),
                    "trend": "improving"
                    if len(values) > 1 and values[0] > values[-1]
                    else "declining"
                    if len(values) > 1 and values[0] < values[-1]
                    else "stable",
                }

        recent_alerts = [a for a in self._alerts if time.time() - a.timestamp < 3600]

        return {
            "status": "healthy" if not recent_alerts else "degraded",
            "model_version": active.version,
            "metrics": metrics,
            "recent_alerts": len(recent_alerts),
        }

    def _check_metric(self, version: str, metric: str, baseline: float) -> RegressionAlert | None:
        """Check a single metric for regression."""
        trend = self.store.get_quality_trend(version, metric, limit=self.config.window_size)

        if len(trend) < self.config.min_samples:
            return None

        values = [t["value"] for t in trend]
        current_avg = sum(values) / len(values)

        if baseline <= 0:
            return None

        drop = (baseline - current_avg) / baseline
        drop_percent = drop * 100

        # Statistical confidence via standard error
        confidence = self._compute_confidence(values, baseline)

        if confidence < self.config.min_confidence:
            return None

        # Determine severity
        if drop >= self.config.severe_threshold:
            severity = RegressionSeverity.SEVERE
        elif drop >= self.config.moderate_threshold:
            severity = RegressionSeverity.MODERATE
        elif drop >= self.config.mild_threshold:
            severity = RegressionSeverity.MILD
        else:
            severity = RegressionSeverity.NONE

        if severity == RegressionSeverity.NONE:
            return None

        return RegressionAlert(
            severity=severity,
            model_version=version,
            metric=metric,
            current_value=current_avg,
            baseline_value=baseline,
            drop_percent=drop_percent,
            sample_size=len(values),
            confidence=confidence,
        )

    def _compute_confidence(self, values: list[float], baseline: float) -> float:
        """Compute confidence that regression is real (not noise).

        Uses a simple z-test against the baseline.
        """
        n = len(values)
        if n < 2:
            return 0.0

        mean = sum(values) / n
        variance = sum((v - mean) ** 2 for v in values) / (n - 1)
        std = math.sqrt(variance) if variance > 0 else 0.001
        se = std / math.sqrt(n)

        if se == 0:
            return 1.0 if mean < baseline else 0.0

        # Z-score: how many standard errors below baseline
        z = (baseline - mean) / se

        # Approximate one-tailed p-value using normal CDF approximation
        # P(Z > z) for positive z
        if z <= 0:
            return 0.0

        # Abramowitz and Stegun approximation
        t = 1.0 / (1.0 + 0.2316419 * z)
        d = 0.3989422804014327  # 1/sqrt(2*pi)
        p = (
            d
            * math.exp(-z * z / 2)
            * (
                0.3193815 * t
                - 0.3565638 * t**2
                + 1.781478 * t**3
                - 1.821256 * t**4
                + 1.330274 * t**5
            )
        )

        return 1.0 - p  # confidence = 1 - p_value

    def _handle_rollback(self, alert: RegressionAlert) -> None:
        """Handle automatic rollback on severe regression."""
        logger.warning(
            "SEVERE regression detected on %s (%.1f%% drop). Auto-rolling back.",
            alert.metric,
            alert.drop_percent,
        )
        result = self.version_mgr.rollback()
        if result:
            alert.auto_rolled_back = True
            logger.info("Rolled back to version %s", result.version)
        else:
            logger.error("Rollback failed — no previous version available")
