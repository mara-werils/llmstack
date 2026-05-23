"""Drift detection — monitors for data and concept drift over time.

Detects when user queries or feedback patterns shift significantly,
indicating the model may need retraining on new patterns. Tracks:
- Query distribution shifts
- Topic drift
- Feedback pattern changes
- Response quality drift
"""

from __future__ import annotations

import logging
import math
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from llmstack.learn.feedback import Feedback
from llmstack.learn.store import FeedbackStore

logger = logging.getLogger(__name__)


@dataclass
class DriftAlert:
    """A detected drift event."""

    drift_type: str  # query_distribution, topic, feedback_pattern, quality
    severity: str  # low, medium, high
    description: str
    timestamp: float = field(default_factory=time.time)
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class DriftConfig:
    """Configuration for drift detection."""

    # Window sizes for comparison (in seconds)
    baseline_window: float = 604800  # 7 days
    recent_window: float = 86400     # 1 day

    # Thresholds
    distribution_threshold: float = 0.3  # KL divergence threshold
    topic_threshold: float = 0.4         # topic shift threshold
    feedback_shift_threshold: float = 0.2  # feedback pattern shift

    # Minimum samples
    min_baseline_samples: int = 20
    min_recent_samples: int = 5


class DriftDetector:
    """Monitors for drift in queries, topics, and feedback patterns.

    Compares recent activity against a historical baseline to detect
    significant shifts that may require model retraining.
    """

    def __init__(self, store: FeedbackStore, config: DriftConfig | None = None):
        self.store = store
        self.config = config or DriftConfig()

    def check(self) -> list[DriftAlert]:
        """Run all drift checks and return alerts."""
        alerts: list[DriftAlert] = []

        now = time.time()
        baseline_since = now - self.config.baseline_window
        recent_since = now - self.config.recent_window

        baseline = self.store.get_feedback(since=baseline_since, limit=5000)
        recent = self.store.get_feedback(since=recent_since, limit=500)

        if len(baseline) < self.config.min_baseline_samples:
            return []
        if len(recent) < self.config.min_recent_samples:
            return []

        # Exclude recent from baseline for fair comparison
        baseline = [fb for fb in baseline if fb.timestamp < recent_since]

        # Check query distribution drift
        query_alert = self._check_query_distribution(baseline, recent)
        if query_alert:
            alerts.append(query_alert)

        # Check topic drift
        topic_alert = self._check_topic_drift(baseline, recent)
        if topic_alert:
            alerts.append(topic_alert)

        # Check feedback pattern drift
        feedback_alert = self._check_feedback_pattern(baseline, recent)
        if feedback_alert:
            alerts.append(feedback_alert)

        return alerts

    def _check_query_distribution(
        self, baseline: list[Feedback], recent: list[Feedback]
    ) -> DriftAlert | None:
        """Check if query length/complexity distribution has shifted."""
        baseline_lengths = [len(fb.query) for fb in baseline if fb.query]
        recent_lengths = [len(fb.query) for fb in recent if fb.query]

        if not baseline_lengths or not recent_lengths:
            return None

        # Compare length distributions using bins
        bins = [0, 20, 50, 100, 200, 500, float("inf")]
        baseline_dist = self._bin_distribution(baseline_lengths, bins)
        recent_dist = self._bin_distribution(recent_lengths, bins)

        kl_div = self._kl_divergence(baseline_dist, recent_dist)

        if kl_div > self.config.distribution_threshold:
            avg_baseline = sum(baseline_lengths) / len(baseline_lengths)
            avg_recent = sum(recent_lengths) / len(recent_lengths)
            return DriftAlert(
                drift_type="query_distribution",
                severity="medium" if kl_div < 0.6 else "high",
                description=(
                    f"Query complexity has shifted "
                    f"(avg length {avg_baseline:.0f} → {avg_recent:.0f})"
                ),
                details={"kl_divergence": round(kl_div, 4)},
            )

        return None

    def _check_topic_drift(
        self, baseline: list[Feedback], recent: list[Feedback]
    ) -> DriftAlert | None:
        """Check if query topics have shifted."""
        baseline_topics = self._extract_topics(baseline)
        recent_topics = self._extract_topics(recent)

        if not baseline_topics or not recent_topics:
            return None

        # Compare topic distributions
        all_topics = set(baseline_topics.keys()) | set(recent_topics.keys())
        baseline_total = sum(baseline_topics.values())
        recent_total = sum(recent_topics.values())

        baseline_dist = {t: baseline_topics.get(t, 0) / baseline_total for t in all_topics}
        recent_dist = {t: recent_topics.get(t, 0) / recent_total for t in all_topics}

        # New topics that didn't exist in baseline
        new_topics = [
            t for t in all_topics
            if recent_dist.get(t, 0) > 0.1 and baseline_dist.get(t, 0) < 0.02
        ]

        if new_topics:
            return DriftAlert(
                drift_type="topic",
                severity="medium",
                description=f"New topics emerging: {', '.join(new_topics[:5])}",
                details={"new_topics": new_topics},
            )

        return None

    def _check_feedback_pattern(
        self, baseline: list[Feedback], recent: list[Feedback]
    ) -> DriftAlert | None:
        """Check if feedback distribution has changed."""
        baseline_types = Counter(fb.feedback_type.value for fb in baseline)
        recent_types = Counter(fb.feedback_type.value for fb in recent)

        baseline_total = sum(baseline_types.values())
        recent_total = sum(recent_types.values())

        if baseline_total == 0 or recent_total == 0:
            return None

        # Check if negative feedback rate has increased
        baseline_negative = (
            baseline_types.get("thumbs_down", 0)
            + baseline_types.get("regenerate", 0)
        ) / baseline_total
        recent_negative = (
            recent_types.get("thumbs_down", 0)
            + recent_types.get("regenerate", 0)
        ) / recent_total

        shift = recent_negative - baseline_negative
        if shift > self.config.feedback_shift_threshold:
            return DriftAlert(
                drift_type="feedback_pattern",
                severity="high" if shift > 0.4 else "medium",
                description=(
                    f"Negative feedback rate increased "
                    f"({baseline_negative:.0%} → {recent_negative:.0%})"
                ),
                details={
                    "baseline_negative_rate": round(baseline_negative, 3),
                    "recent_negative_rate": round(recent_negative, 3),
                    "shift": round(shift, 3),
                },
            )

        return None

    def _extract_topics(self, feedback: list[Feedback]) -> Counter:
        """Extract topic keywords from queries."""
        topics: Counter = Counter()
        keywords = {
            "python": "python",
            "javascript": "javascript",
            "typescript": "typescript",
            "rust": "rust",
            "go": "golang",
            "api": "api",
            "database": "database",
            "test": "testing",
            "deploy": "deployment",
            "docker": "docker",
            "error": "debugging",
            "bug": "debugging",
            "performance": "performance",
            "security": "security",
            "auth": "authentication",
            "css": "frontend",
            "react": "frontend",
            "sql": "database",
        }

        for fb in feedback:
            q = fb.query.lower()
            for keyword, topic in keywords.items():
                if keyword in q:
                    topics[topic] += 1

        return topics

    def _bin_distribution(self, values: list[float], bins: list[float]) -> list[float]:
        """Create binned distribution from values."""
        counts = [0] * (len(bins) - 1)
        for v in values:
            for i in range(len(bins) - 1):
                if bins[i] <= v < bins[i + 1]:
                    counts[i] += 1
                    break

        total = sum(counts)
        if total == 0:
            return [1.0 / len(counts)] * len(counts)
        return [c / total for c in counts]

    def _kl_divergence(self, p: list[float], q: list[float]) -> float:
        """Compute KL divergence between two distributions."""
        eps = 1e-10
        kl = 0.0
        for pi, qi in zip(p, q):
            pi = max(pi, eps)
            qi = max(qi, eps)
            kl += pi * math.log(pi / qi)
        return max(0.0, kl)
