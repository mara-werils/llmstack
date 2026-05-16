"""Learning analytics — tracks improvement over time with visual reports.

Provides metrics on the learning pipeline's effectiveness:
- Feedback collection rate
- Correction rate trends
- Model quality over versions
- Training efficiency
- User satisfaction trajectory
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from llmstack.learn.store import FeedbackStore
from llmstack.learn.versions import ModelVersionManager


@dataclass
class LearningMetrics:
    """Aggregate learning pipeline metrics."""

    # Feedback metrics
    total_feedback: int = 0
    positive_rate: float = 0.0
    correction_rate: float = 0.0
    feedback_per_day: float = 0.0

    # Training metrics
    total_train_runs: int = 0
    total_versions: int = 0
    avg_dataset_size: float = 0.0
    avg_train_time: float = 0.0

    # Quality metrics
    current_quality: float = 0.0
    quality_improvement: float = 0.0  # from first to current version
    best_quality: float = 0.0
    quality_trend: str = "stable"  # improving, declining, stable

    # Efficiency
    feedback_to_improvement_ratio: float = 0.0
    unused_feedback: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "feedback": {
                "total": self.total_feedback,
                "positive_rate": round(self.positive_rate, 3),
                "correction_rate": round(self.correction_rate, 3),
                "per_day": round(self.feedback_per_day, 1),
                "unused": self.unused_feedback,
            },
            "training": {
                "total_runs": self.total_train_runs,
                "total_versions": self.total_versions,
                "avg_dataset_size": round(self.avg_dataset_size, 0),
                "avg_train_time_sec": round(self.avg_train_time, 1),
            },
            "quality": {
                "current": round(self.current_quality, 4),
                "improvement": round(self.quality_improvement, 4),
                "best": round(self.best_quality, 4),
                "trend": self.quality_trend,
            },
            "efficiency": {
                "feedback_to_improvement": round(
                    self.feedback_to_improvement_ratio, 4
                ),
            },
        }


@dataclass
class TimeSeriesPoint:
    """A single point in a time series."""

    timestamp: float
    value: float
    label: str = ""


class LearningAnalytics:
    """Computes learning pipeline analytics and reports.

    Aggregates data from the feedback store and version manager
    to produce actionable insights about the learning pipeline.
    """

    def __init__(self, store: FeedbackStore, version_mgr: ModelVersionManager):
        self.store = store
        self.version_mgr = version_mgr

    def compute_metrics(self) -> LearningMetrics:
        """Compute current learning pipeline metrics."""
        stats = self.store.get_stats()
        metrics = LearningMetrics()

        # Feedback metrics
        metrics.total_feedback = stats["total_feedback"]
        metrics.unused_feedback = stats["unused_feedback"]

        by_type = stats.get("feedback_by_type", {})
        total_rated = by_type.get("thumbs_up", 0) + by_type.get("thumbs_down", 0)
        if total_rated > 0:
            metrics.positive_rate = by_type.get("thumbs_up", 0) / total_rated

        corrections = (
            by_type.get("correction", 0)
            + by_type.get("edit", 0)
        )
        if metrics.total_feedback > 0:
            metrics.correction_rate = corrections / metrics.total_feedback

        # Feedback rate
        all_feedback = self.store.get_feedback(limit=1)
        oldest = self.store.get_feedback(limit=1)
        if oldest:
            days = max(1, (time.time() - oldest[0].timestamp) / 86400)
            metrics.feedback_per_day = metrics.total_feedback / days

        # Training metrics
        metrics.total_train_runs = stats["total_train_runs"]
        metrics.total_versions = stats["total_versions"]

        # Quality metrics
        versions = self.version_mgr.list_versions(limit=50)
        if versions:
            active = next((v for v in versions if v.is_active), None)
            metrics.current_quality = active.quality_score if active else 0.0
            metrics.best_quality = max(v.quality_score for v in versions)

            if len(versions) >= 2:
                first_quality = versions[-1].quality_score
                latest_quality = versions[0].quality_score
                metrics.quality_improvement = latest_quality - first_quality

                # Trend from last 5 versions
                recent = versions[:5]
                if len(recent) >= 2:
                    recent_scores = [v.quality_score for v in recent]
                    if recent_scores[0] > recent_scores[-1] + 0.01:
                        metrics.quality_trend = "improving"
                    elif recent_scores[0] < recent_scores[-1] - 0.01:
                        metrics.quality_trend = "declining"

        # Efficiency
        if metrics.total_feedback > 0 and metrics.quality_improvement > 0:
            metrics.feedback_to_improvement_ratio = (
                metrics.quality_improvement / metrics.total_feedback
            )

        return metrics

    def get_quality_timeline(self, limit: int = 50) -> list[TimeSeriesPoint]:
        """Get quality score over time across all versions."""
        versions = self.version_mgr.list_versions(limit=limit)
        return [
            TimeSeriesPoint(
                timestamp=v.timestamp,
                value=v.quality_score,
                label=f"v{v.version}",
            )
            for v in reversed(versions)
            if v.quality_score > 0
        ]

    def get_feedback_timeline(
        self, bucket_hours: int = 24, limit: int = 30
    ) -> list[TimeSeriesPoint]:
        """Get feedback count over time in hourly buckets."""
        all_feedback = self.store.get_feedback(limit=10000)
        if not all_feedback:
            return []

        bucket_seconds = bucket_hours * 3600
        now = time.time()
        buckets: dict[int, int] = {}

        for fb in all_feedback:
            bucket_idx = int((now - fb.timestamp) / bucket_seconds)
            if bucket_idx < limit:
                buckets[bucket_idx] = buckets.get(bucket_idx, 0) + 1

        return [
            TimeSeriesPoint(
                timestamp=now - (i * bucket_seconds),
                value=float(buckets.get(i, 0)),
                label=f"-{i * bucket_hours}h",
            )
            for i in range(limit)
        ]

    def get_summary(self) -> dict[str, Any]:
        """Get a human-readable summary of learning pipeline status."""
        metrics = self.compute_metrics()
        stats = self.store.get_stats()

        summary: dict[str, Any] = {
            "status": self._compute_status(metrics),
            "metrics": metrics.to_dict(),
            "recommendations": self._get_recommendations(metrics),
        }

        return summary

    def _compute_status(self, metrics: LearningMetrics) -> str:
        """Compute overall pipeline status."""
        if metrics.total_feedback == 0:
            return "inactive"
        if metrics.total_versions == 0:
            return "collecting"
        if metrics.quality_trend == "improving":
            return "improving"
        if metrics.quality_trend == "declining":
            return "degrading"
        return "active"

    def _get_recommendations(self, metrics: LearningMetrics) -> list[str]:
        """Generate actionable recommendations."""
        recs: list[str] = []

        if metrics.total_feedback == 0:
            recs.append(
                "Start collecting feedback by using thumbs up/down "
                "or corrections in chat/ask commands"
            )

        if metrics.total_feedback > 0 and metrics.total_versions == 0:
            if metrics.unused_feedback >= 25:
                recs.append(
                    f"You have {metrics.unused_feedback} unused feedback items. "
                    "Run 'llmstack learn train' to trigger fine-tuning"
                )
            else:
                recs.append(
                    f"Collecting feedback ({metrics.unused_feedback}/25 threshold). "
                    "Keep using llmstack to build training data"
                )

        if metrics.positive_rate < 0.5 and metrics.total_feedback >= 20:
            recs.append(
                "Low satisfaction rate. Consider adding more corrections "
                "to help the model learn your preferences"
            )

        if metrics.quality_trend == "declining":
            recs.append(
                "Quality is declining. Check for data quality issues "
                "or consider rolling back to a previous version"
            )

        if metrics.correction_rate > 0.5 and metrics.total_feedback >= 10:
            recs.append(
                "High correction rate suggests the model needs more training. "
                "Corrections are being captured for the next training run"
            )

        return recs
