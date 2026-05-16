"""Active learning — intelligently requests feedback on uncertain responses.

Instead of randomly prompting for feedback, identifies responses where
the model is most uncertain or where feedback would be most valuable
for improving the model. Implements uncertainty-based and diversity-based
selection strategies.
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

from llmstack.learn.store import FeedbackStore

logger = logging.getLogger(__name__)


@dataclass
class ActiveLearningConfig:
    """Configuration for active learning feedback requests."""

    # Maximum feedback requests per session
    max_requests_per_session: int = 5

    # Minimum interactions before first request
    warmup_interactions: int = 3

    # Cooldown between requests (interactions)
    cooldown_interactions: int = 3

    # Uncertainty threshold for requesting feedback
    uncertainty_threshold: float = 0.4

    # Diversity: don't ask about similar queries
    diversity_threshold: float = 0.7


@dataclass
class UncertaintySignal:
    """Signals indicating model uncertainty."""

    hedging_score: float = 0.0       # presence of hedging language
    length_anomaly: float = 0.0      # unusual response length
    repetition_score: float = 0.0    # repetitive patterns
    novelty_score: float = 0.0       # query unlike training data
    overall: float = 0.0


class ActiveLearner:
    """Decides WHEN and WHAT to ask for feedback on.

    Uses uncertainty estimation and diversity criteria to select
    the most informative interactions for feedback, maximizing
    learning signal per user interruption.
    """

    def __init__(
        self,
        store: FeedbackStore,
        config: ActiveLearningConfig | None = None,
    ):
        self.store = store
        self.config = config or ActiveLearningConfig()
        self._session_requests = 0
        self._interaction_count = 0
        self._last_request_at = 0
        self._asked_queries: list[str] = []

    def should_request_feedback(
        self,
        query: str,
        response: str,
    ) -> bool:
        """Decide whether to request feedback for this interaction.

        Returns True if this interaction is a high-value feedback candidate.
        """
        self._interaction_count += 1

        # Hard limits
        if self._session_requests >= self.config.max_requests_per_session:
            return False
        if self._interaction_count < self.config.warmup_interactions:
            return False
        if (self._interaction_count - self._last_request_at) < self.config.cooldown_interactions:
            return False

        # Compute uncertainty
        uncertainty = self.estimate_uncertainty(query, response)
        if uncertainty.overall < self.config.uncertainty_threshold:
            return False

        # Diversity check — don't ask about similar queries
        if self._is_similar_to_asked(query):
            return False

        return True

    def mark_requested(self, query: str) -> None:
        """Mark that we requested feedback for this query."""
        self._session_requests += 1
        self._last_request_at = self._interaction_count
        self._asked_queries.append(query)

    def estimate_uncertainty(self, query: str, response: str) -> UncertaintySignal:
        """Estimate model uncertainty for a response.

        Uses heuristic signals since we don't have logprobs for most local models.
        """
        hedging = self._score_hedging(response)
        length_anomaly = self._score_length_anomaly(query, response)
        repetition = self._score_repetition(response)
        novelty = self._score_novelty(query)

        overall = (
            0.3 * hedging
            + 0.2 * length_anomaly
            + 0.2 * repetition
            + 0.3 * novelty
        )

        return UncertaintySignal(
            hedging_score=hedging,
            length_anomaly=length_anomaly,
            repetition_score=repetition,
            novelty_score=novelty,
            overall=min(1.0, overall),
        )

    def get_feedback_prompt(self, query: str, response: str) -> str:
        """Generate a contextual feedback prompt."""
        uncertainty = self.estimate_uncertainty(query, response)

        if uncertainty.hedging_score > 0.6:
            return (
                "\n[Learn] I'm not fully confident in this answer. "
                "Was it helpful? (y/n/c:correction): "
            )
        if uncertainty.novelty_score > 0.6:
            return (
                "\n[Learn] This seems like a new type of question for me. "
                "Did I get it right? (y/n/c:correction): "
            )
        return (
            "\n[Learn] Quick feedback? (y/n/c:correction/s=skip): "
        )

    def _score_hedging(self, response: str) -> float:
        """Score presence of hedging/uncertainty language."""
        hedges = [
            r"(?i)i('m| am) not (sure|certain|confident)",
            r"(?i)(might|may|could) (be|have)",
            r"(?i)i think",
            r"(?i)it('s| is) possible",
            r"(?i)perhaps|maybe|possibly",
            r"(?i)if i('m| am) (not )?mistaken",
            r"(?i)i believe",
            r"(?i)it seems (like|that)",
        ]
        hits = sum(1 for h in hedges if re.search(h, response))
        return min(1.0, hits / 3)

    def _score_length_anomaly(self, query: str, response: str) -> float:
        """Score unusual response length relative to query."""
        query_len = len(query)
        response_len = len(response)

        if query_len == 0:
            return 0.0

        ratio = response_len / max(query_len, 1)

        # Very short responses to long queries = uncertain
        if ratio < 0.3 and query_len > 50:
            return 0.7
        # Very long responses to short queries = potentially rambling
        if ratio > 20 and query_len < 30:
            return 0.5
        return 0.0

    def _score_repetition(self, response: str) -> float:
        """Score repetitive patterns that suggest confusion."""
        if len(response) < 100:
            return 0.0

        sentences = re.split(r'[.!?\n]+', response)
        sentences = [s.strip().lower() for s in sentences if len(s.strip()) > 15]

        if len(sentences) < 3:
            return 0.0

        unique = set(sentences)
        dup_ratio = 1.0 - (len(unique) / len(sentences))
        return min(1.0, dup_ratio * 2)

    def _score_novelty(self, query: str) -> float:
        """Score how novel this query is compared to seen queries.

        Higher novelty = more valuable feedback.
        """
        # Check against recent feedback in store
        recent = self.store.get_feedback(limit=100)
        if not recent:
            return 0.8  # no history = everything is novel

        query_words = set(query.lower().split())
        if not query_words:
            return 0.5

        # Compute max similarity to any stored query
        max_sim = 0.0
        for fb in recent:
            fb_words = set(fb.query.lower().split())
            if not fb_words:
                continue
            overlap = len(query_words & fb_words)
            sim = overlap / max(len(query_words), len(fb_words))
            max_sim = max(max_sim, sim)

        # Novelty is inverse of max similarity
        return 1.0 - max_sim

    def _is_similar_to_asked(self, query: str) -> bool:
        """Check if query is too similar to one we already asked about."""
        query_words = set(query.lower().split())
        for asked in self._asked_queries[-20:]:
            asked_words = set(asked.lower().split())
            if not asked_words:
                continue
            overlap = len(query_words & asked_words)
            sim = overlap / max(len(query_words), len(asked_words))
            if sim > self.config.diversity_threshold:
                return True
        return False
