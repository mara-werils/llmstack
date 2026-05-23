"""Tests for feedback deduplication."""

from __future__ import annotations

import pytest

from llmstack.learn.dedup import DedupConfig, FeedbackDeduplicator
from llmstack.learn.feedback import Feedback, FeedbackType


def _fb(query: str, response: str) -> Feedback:
    return Feedback(
        feedback_type=FeedbackType.THUMBS_UP,
        query=query,
        response=response,
    )


@pytest.fixture
def dedup():
    return FeedbackDeduplicator()


class TestFeedbackDeduplicator:
    def test_empty_input(self, dedup):
        result, stats = dedup.deduplicate([])
        assert result == []
        assert stats.total_input == 0

    def test_no_duplicates(self, dedup):
        entries = [_fb("q1", "r1"), _fb("q2", "r2")]
        result, stats = dedup.deduplicate(entries)
        assert len(result) == 2
        assert stats.duplicates_removed == 0

    def test_exact_duplicates_removed(self, dedup):
        entries = [_fb("hello", "world"), _fb("hello", "world"), _fb("other", "resp")]
        result, stats = dedup.deduplicate(entries)
        assert len(result) == 2
        assert stats.duplicates_removed == 1

    def test_near_duplicates_merged(self):
        dedup = FeedbackDeduplicator(DedupConfig(similarity_threshold=0.7))
        entries = [
            _fb(
                "How do I sort a list in Python efficiently?",
                "Use the sorted() function for a new list.",
            ),
            _fb(
                "How do I sort a list in Python efficiently please?",
                "Use the sorted() function for a new sorted list.",
            ),
        ]
        result, stats = dedup.deduplicate(entries)
        assert len(result) == 1

    def test_different_entries_kept(self, dedup):
        entries = [
            _fb("What is Python?", "A programming language."),
            _fb("What is Rust?", "A systems programming language."),
        ]
        result, stats = dedup.deduplicate(entries)
        assert len(result) == 2

    def test_normalize_whitespace(self, dedup):
        text = "  hello   world  "
        assert dedup.normalize(text) == "hello world"

    def test_normalize_case(self, dedup):
        text = "Hello World"
        assert dedup.normalize(text) == "hello world"

    def test_stats_structure(self, dedup):
        entries = [_fb("q", "r"), _fb("q", "r")]
        _, stats = dedup.deduplicate(entries)
        d = stats.to_dict()
        assert "total_input" in d
        assert "duplicates_removed" in d
        assert "dedup_ratio" in d
        assert d["total_input"] == 2
        assert d["duplicates_removed"] == 1

    def test_dedup_ratio(self, dedup):
        entries = [_fb("q", "r")] * 5
        _, stats = dedup.deduplicate(entries)
        assert stats.dedup_ratio == 0.8  # 4 out of 5 removed

    def test_custom_threshold(self):
        # Very strict threshold — only exact matches
        dedup = FeedbackDeduplicator(DedupConfig(similarity_threshold=1.0))
        entries = [
            _fb("How to sort?", "Use sorted."),
            _fb("How to sort?!", "Use sorted!"),
        ]
        result, stats = dedup.deduplicate(entries)
        # These are exact-hash different but near-dup check with threshold=1.0 won't merge
        assert len(result) == 2
