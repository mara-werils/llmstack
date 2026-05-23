"""Data quality scoring for training examples.

Assigns quality scores to feedback entries to prioritize high-quality
examples during training. Low-quality examples (too short, incoherent,
or irrelevant) are downweighted or excluded.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

from llmstack.learn.feedback import Feedback

logger = logging.getLogger(__name__)


@dataclass
class QualityScore:
    """Quality assessment for a single training example."""

    completeness: float = 0.0
    coherence: float = 0.0
    relevance: float = 0.0
    informativeness: float = 0.0
    overall: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "completeness": round(self.completeness, 4),
            "coherence": round(self.coherence, 4),
            "relevance": round(self.relevance, 4),
            "informativeness": round(self.informativeness, 4),
            "overall": round(self.overall, 4),
        }


@dataclass
class QualityScorerConfig:
    """Configuration for the quality scorer."""

    # Minimum overall score to include in training
    min_quality: float = 0.3

    # Weights for each dimension
    completeness_weight: float = 0.25
    coherence_weight: float = 0.25
    relevance_weight: float = 0.25
    informativeness_weight: float = 0.25

    # Minimum response length to be considered complete
    min_response_length: int = 10

    # Minimum query length
    min_query_length: int = 3


class DataQualityScorer:
    """Scores the quality of feedback entries for training data selection.

    Evaluates each entry on completeness, coherence, relevance, and
    informativeness to determine its value as a training example.
    """

    def __init__(self, config: QualityScorerConfig | None = None):
        self.config = config or QualityScorerConfig()

    def score(self, feedback: Feedback) -> QualityScore:
        """Score a single feedback entry."""
        completeness = self._score_completeness(feedback)
        coherence = self._score_coherence(feedback)
        relevance = self._score_relevance(feedback)
        informativeness = self._score_informativeness(feedback)

        overall = (
            self.config.completeness_weight * completeness
            + self.config.coherence_weight * coherence
            + self.config.relevance_weight * relevance
            + self.config.informativeness_weight * informativeness
        )

        return QualityScore(
            completeness=completeness,
            coherence=coherence,
            relevance=relevance,
            informativeness=informativeness,
            overall=min(1.0, overall),
        )

    def filter_quality(
        self,
        entries: list[Feedback],
        min_quality: float | None = None,
    ) -> list[tuple[Feedback, QualityScore]]:
        """Filter entries by quality threshold.

        Returns list of (feedback, score) tuples for entries above threshold.
        """
        threshold = min_quality if min_quality is not None else self.config.min_quality
        results: list[tuple[Feedback, QualityScore]] = []

        for entry in entries:
            score = self.score(entry)
            if score.overall >= threshold:
                results.append((entry, score))

        results.sort(key=lambda x: x[1].overall, reverse=True)
        return results

    def get_distribution(self, entries: list[Feedback]) -> dict[str, Any]:
        """Get quality score distribution across entries."""
        if not entries:
            return {"total": 0, "bins": {}}

        scores = [self.score(e).overall for e in entries]
        bins = {"low": 0, "medium": 0, "high": 0, "excellent": 0}
        for s in scores:
            if s < 0.3:
                bins["low"] += 1
            elif s < 0.5:
                bins["medium"] += 1
            elif s < 0.7:
                bins["high"] += 1
            else:
                bins["excellent"] += 1

        return {
            "total": len(scores),
            "mean": round(sum(scores) / len(scores), 4),
            "min": round(min(scores), 4),
            "max": round(max(scores), 4),
            "bins": bins,
        }

    def _score_completeness(self, fb: Feedback) -> float:
        """Score how complete the query-response pair is."""
        score = 1.0

        if len(fb.query) < self.config.min_query_length:
            score -= 0.5
        if len(fb.response) < self.config.min_response_length:
            score -= 0.5

        # Check for truncated responses
        if fb.response.endswith("...") or fb.response.endswith("…"):
            score -= 0.3

        # Check for empty correction when feedback is negative
        if fb.feedback_type.value == "thumbs_down" and not fb.correction:
            score -= 0.2

        return max(0.0, score)

    def _score_coherence(self, fb: Feedback) -> float:
        """Score how coherent the response is."""
        response = fb.response
        if not response:
            return 0.0

        score = 0.8  # Base score

        # Check for broken sentences
        sentences = re.split(r'[.!?]+', response)
        if sentences:
            avg_len = sum(len(s.strip()) for s in sentences) / len(sentences)
            if avg_len < 5:
                score -= 0.3

        # Check for excessive repetition
        words = response.lower().split()
        if len(words) > 10:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.3:
                score -= 0.4

        # Check for garbled text
        non_ascii = sum(1 for c in response if ord(c) > 127)
        if len(response) > 0 and non_ascii / len(response) > 0.3:
            score -= 0.3

        return max(0.0, min(1.0, score))

    def _score_relevance(self, fb: Feedback) -> float:
        """Score how relevant the response is to the query."""
        if not fb.query or not fb.response:
            return 0.0

        query_words = set(fb.query.lower().split())
        response_words = set(fb.response.lower().split())

        # Remove common stop words
        stop = {"the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "to", "for", "of"}
        query_words -= stop
        response_words -= stop

        if not query_words:
            return 0.5

        overlap = len(query_words & response_words)
        relevance = overlap / len(query_words) if query_words else 0.0

        return min(1.0, relevance + 0.3)  # Base boost since responses may use different words

    def _score_informativeness(self, fb: Feedback) -> float:
        """Score how informative the response is."""
        response = fb.response
        if not response:
            return 0.0

        score = 0.5  # Base

        # Longer responses tend to be more informative (up to a point)
        resp_len = len(response)
        if resp_len > 50:
            score += 0.1
        if resp_len > 200:
            score += 0.1
        if resp_len > 500:
            score += 0.1

        # Code blocks are informative
        if "```" in response or "def " in response:
            score += 0.2

        # Lists/structure indicate organized info
        if re.search(r'^\s*[-*•]\s', response, re.MULTILINE):
            score += 0.1

        return min(1.0, score)
