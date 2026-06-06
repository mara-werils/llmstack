"""Feedback deduplication and normalization.

Ensures training data quality by removing duplicate feedback entries,
normalizing text, and merging related signals for the same query-response pair.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from typing import Any

from llmstack.learn.feedback import Feedback

logger = logging.getLogger(__name__)


@dataclass
class DedupConfig:
    """Configuration for feedback deduplication."""

    # Similarity threshold for considering entries duplicates (0.0-1.0)
    similarity_threshold: float = 0.85

    # Whether to normalize whitespace before comparison
    normalize_whitespace: bool = True

    # Whether to normalize case before comparison
    normalize_case: bool = True

    # Maximum entries to check for duplicates (for performance)
    max_check_window: int = 500


@dataclass
class DedupStats:
    """Statistics from a deduplication run."""

    total_input: int = 0
    duplicates_removed: int = 0
    merged: int = 0
    output_count: int = 0

    @property
    def dedup_ratio(self) -> float:
        return self.duplicates_removed / self.total_input if self.total_input > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_input": self.total_input,
            "duplicates_removed": self.duplicates_removed,
            "merged": self.merged,
            "output_count": self.output_count,
            "dedup_ratio": round(self.dedup_ratio, 4),
        }


class FeedbackDeduplicator:
    """Removes duplicate and near-duplicate feedback entries.

    Uses content hashing for exact duplicates and word-overlap
    similarity for near-duplicates.
    """

    def __init__(self, config: DedupConfig | None = None):
        self.config = config or DedupConfig()
        self._total_processed = 0
        self._total_duplicates = 0

    @property
    def total_processed(self) -> int:
        """Return total entries processed across all dedup runs."""
        return self._total_processed

    @property
    def total_duplicates(self) -> int:
        """Return total duplicates removed across all runs."""
        return self._total_duplicates

    @property
    def overall_dedup_ratio(self) -> float:
        """Return lifetime deduplication ratio."""
        return self._total_duplicates / self._total_processed if self._total_processed > 0 else 0.0

    def deduplicate(self, entries: list[Feedback]) -> tuple[list[Feedback], DedupStats]:
        """Remove duplicate feedback entries.

        Returns:
            Tuple of (deduplicated list, statistics).
        """
        stats = DedupStats(total_input=len(entries))

        if not entries:
            return [], stats

        # Phase 1: Exact dedup by content hash
        seen_hashes: set[str] = set()
        phase1: list[Feedback] = []

        for entry in entries:
            h = self._content_hash(entry)
            if h not in seen_hashes:
                seen_hashes.add(h)
                phase1.append(entry)

        exact_dups = len(entries) - len(phase1)

        # Phase 2: Near-duplicate detection
        phase2: list[Feedback] = []
        for entry in phase1:
            is_dup = False
            for existing in phase2[-self.config.max_check_window :]:
                if self._is_near_duplicate(entry, existing):
                    is_dup = True
                    stats.merged += 1
                    break
            if not is_dup:
                phase2.append(entry)

        near_dups = len(phase1) - len(phase2)
        stats.duplicates_removed = exact_dups + near_dups
        stats.output_count = len(phase2)

        self._total_processed += stats.total_input
        self._total_duplicates += stats.duplicates_removed

        return phase2, stats

    def normalize(self, text: str) -> str:
        """Normalize text for comparison."""
        if self.config.normalize_case:
            text = text.lower()
        if self.config.normalize_whitespace:
            text = re.sub(r"\s+", " ", text).strip()
        return text

    def _content_hash(self, entry: Feedback) -> str:
        """Create a hash of the feedback content."""
        content = f"{self.normalize(entry.query)}|{self.normalize(entry.response)}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _is_near_duplicate(self, a: Feedback, b: Feedback) -> bool:
        """Check if two feedback entries are near-duplicates."""
        query_sim = self._word_similarity(
            self.normalize(a.query),
            self.normalize(b.query),
        )
        if query_sim < self.config.similarity_threshold:
            return False

        response_sim = self._word_similarity(
            self.normalize(a.response),
            self.normalize(b.response),
        )
        return response_sim >= self.config.similarity_threshold

    def _word_similarity(self, text_a: str, text_b: str) -> float:
        """Compute word-level Jaccard similarity."""
        words_a = set(text_a.split())
        words_b = set(text_b.split())

        if not words_a and not words_b:
            return 1.0
        if not words_a or not words_b:
            return 0.0

        intersection = len(words_a & words_b)
        union = len(words_a | words_b)
        return intersection / union if union > 0 else 0.0
