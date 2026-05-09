"""A/B testing — compare models side-by-side with statistical metrics.

Splits traffic between two models, collects per-model quality scores,
and computes a statistical comparison.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from threading import Lock


@dataclass
class ABTestResult:
    """Results of an A/B test comparison."""

    test_name: str = ""
    model_a: str = ""
    model_b: str = ""
    requests_a: int = 0
    requests_b: int = 0
    avg_quality_a: float = 0.0
    avg_quality_b: float = 0.0
    avg_latency_a_ms: float = 0.0
    avg_latency_b_ms: float = 0.0
    avg_cost_a_usd: float = 0.0
    avg_cost_b_usd: float = 0.0
    winner: str = ""
    confidence: str = ""   # "low", "medium", "high"

    def to_dict(self) -> dict:
        return {
            "test_name": self.test_name,
            "model_a": self.model_a,
            "model_b": self.model_b,
            "requests_a": self.requests_a,
            "requests_b": self.requests_b,
            "avg_quality_a": round(self.avg_quality_a, 4),
            "avg_quality_b": round(self.avg_quality_b, 4),
            "avg_latency_a_ms": round(self.avg_latency_a_ms, 1),
            "avg_latency_b_ms": round(self.avg_latency_b_ms, 1),
            "avg_cost_a_usd": round(self.avg_cost_a_usd, 6),
            "avg_cost_b_usd": round(self.avg_cost_b_usd, 6),
            "winner": self.winner,
            "confidence": self.confidence,
        }


@dataclass
class ABTest:
    """Configuration for a single A/B test."""

    name: str
    model_a: str
    model_b: str
    traffic_split: float = 0.5   # fraction of traffic going to model_b
    active: bool = True
    created_at: float = 0.0

    def __post_init__(self):
        if not self.created_at:
            self.created_at = time.time()

    def select_model(self, request_id: str = "") -> str:
        """Deterministically select A or B based on request_id hash."""
        if not request_id:
            request_id = str(time.time())
        h = hashlib.md5(request_id.encode()).hexdigest()
        bucket = int(h[:8], 16) / 0xFFFFFFFF
        return self.model_b if bucket < self.traffic_split else self.model_a


@dataclass
class _ModelMetrics:
    """Accumulated metrics for one side of an A/B test."""

    qualities: list[float] = field(default_factory=list)
    latencies: list[float] = field(default_factory=list)
    costs: list[float] = field(default_factory=list)


class ABTestManager:
    """Manages active A/B tests and collects comparison metrics."""

    def __init__(self):
        self._lock = Lock()
        self._tests: dict[str, ABTest] = {}
        self._metrics: dict[str, dict[str, _ModelMetrics]] = {}

    def create_test(self, test: ABTest) -> None:
        """Register a new A/B test."""
        with self._lock:
            self._tests[test.name] = test
            self._metrics[test.name] = {
                test.model_a: _ModelMetrics(),
                test.model_b: _ModelMetrics(),
            }

    def get_test(self, name: str) -> ABTest | None:
        with self._lock:
            return self._tests.get(name)

    def list_tests(self) -> list[ABTest]:
        with self._lock:
            return list(self._tests.values())

    def select_model(self, test_name: str, request_id: str = "") -> str | None:
        """Select a model for a request in the given A/B test."""
        with self._lock:
            test = self._tests.get(test_name)
            if test is None or not test.active:
                return None
            return test.select_model(request_id)

    def record(
        self,
        test_name: str,
        model: str,
        quality: float,
        latency_ms: float,
        cost_usd: float = 0.0,
    ) -> None:
        """Record a result for one side of an A/B test."""
        with self._lock:
            if test_name not in self._metrics:
                return
            metrics = self._metrics[test_name].get(model)
            if metrics is None:
                return
            metrics.qualities.append(quality)
            metrics.latencies.append(latency_ms)
            metrics.costs.append(cost_usd)

    def get_results(self, test_name: str) -> ABTestResult | None:
        """Compute current A/B test results."""
        with self._lock:
            test = self._tests.get(test_name)
            if test is None:
                return None

            metrics = self._metrics.get(test_name, {})
            m_a = metrics.get(test.model_a, _ModelMetrics())
            m_b = metrics.get(test.model_b, _ModelMetrics())

            avg_q_a = sum(m_a.qualities) / max(len(m_a.qualities), 1)
            avg_q_b = sum(m_b.qualities) / max(len(m_b.qualities), 1)
            avg_l_a = sum(m_a.latencies) / max(len(m_a.latencies), 1)
            avg_l_b = sum(m_b.latencies) / max(len(m_b.latencies), 1)
            avg_c_a = sum(m_a.costs) / max(len(m_a.costs), 1)
            avg_c_b = sum(m_b.costs) / max(len(m_b.costs), 1)

            # Determine winner
            winner, confidence = _determine_winner(
                m_a.qualities, m_b.qualities, test.model_a, test.model_b,
            )

            return ABTestResult(
                test_name=test_name,
                model_a=test.model_a,
                model_b=test.model_b,
                requests_a=len(m_a.qualities),
                requests_b=len(m_b.qualities),
                avg_quality_a=avg_q_a,
                avg_quality_b=avg_q_b,
                avg_latency_a_ms=avg_l_a,
                avg_latency_b_ms=avg_l_b,
                avg_cost_a_usd=avg_c_a,
                avg_cost_b_usd=avg_c_b,
                winner=winner,
                confidence=confidence,
            )

    def stop_test(self, test_name: str) -> None:
        with self._lock:
            if test_name in self._tests:
                self._tests[test_name].active = False


def _determine_winner(
    vals_a: list[float], vals_b: list[float],
    name_a: str, name_b: str,
) -> tuple[str, str]:
    """Determine the winner and confidence level.

    Uses simple mean comparison with sample-size-based confidence.
    """
    if not vals_a and not vals_b:
        return "", "low"

    n = min(len(vals_a), len(vals_b))
    if n < 5:
        mean_a = sum(vals_a) / max(len(vals_a), 1)
        mean_b = sum(vals_b) / max(len(vals_b), 1)
        winner = name_a if mean_a >= mean_b else name_b
        return winner, "low"

    mean_a = sum(vals_a) / len(vals_a)
    mean_b = sum(vals_b) / len(vals_b)

    diff = abs(mean_a - mean_b)
    winner = name_a if mean_a >= mean_b else name_b

    # Confidence based on effect size and sample count
    if n >= 100 and diff > 0.05:
        confidence = "high"
    elif n >= 30 and diff > 0.03:
        confidence = "medium"
    else:
        confidence = "low"

    return winner, confidence
