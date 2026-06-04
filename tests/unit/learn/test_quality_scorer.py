"""Tests for data quality scoring."""

from __future__ import annotations

import pytest

from llmstack.learn.quality_scorer import (
    DataQualityScorer,
    QualityScorerConfig,
)
from llmstack.learn.feedback import Feedback, FeedbackType


def _fb(query: str, response: str, **kwargs) -> Feedback:
    return Feedback(
        feedback_type=kwargs.get("ftype", FeedbackType.THUMBS_UP),
        query=query,
        response=response,
        correction=kwargs.get("correction", ""),
    )


@pytest.fixture
def scorer():
    return DataQualityScorer()


class TestDataQualityScorer:
    def test_good_example_high_score(self, scorer):
        fb = _fb(
            "How do I sort a list in Python?",
            "Use the sorted() function: `sorted([3, 1, 2])` returns `[1, 2, 3]`. "
            "For in-place sorting, use `list.sort()`. Both accept a `key` parameter.",
        )
        score = scorer.score(fb)
        assert score.overall > 0.5

    def test_empty_response_low_score(self, scorer):
        fb = _fb("What is X?", "")
        score = scorer.score(fb)
        assert score.overall < 0.3

    def test_short_query_penalized(self, scorer):
        fb = _fb("X", "Something about X.")
        score = scorer.score(fb)
        assert score.completeness < 0.8

    def test_truncated_response_penalized(self, scorer):
        fb = _fb("Explain Python", "Python is a programming language that...")
        score = scorer.score(fb)
        assert score.completeness < 1.0

    def test_code_response_informative(self, scorer):
        fb = _fb(
            "Write a function",
            "```python\ndef add(a, b):\n    return a + b\n```",
        )
        score = scorer.score(fb)
        assert score.informativeness > 0.5

    def test_repetitive_response_low_coherence(self, scorer):
        fb = _fb("Q", "word " * 50)
        score = scorer.score(fb)
        assert score.coherence < 0.6

    def test_filter_quality(self, scorer):
        entries = [
            _fb("Good query here", "A detailed and helpful response about the topic."),
            _fb("", ""),
            _fb("Another good query", "Another good response with useful info and code."),
        ]
        filtered = scorer.filter_quality(entries, min_quality=0.3)
        assert len(filtered) >= 1

    def test_distribution(self, scorer):
        entries = [
            _fb("Q1", "A1" * 50),
            _fb("Q2", ""),
            _fb("Q3", "A3" * 100),
        ]
        dist = scorer.get_distribution(entries)
        assert dist["total"] == 3
        assert "mean" in dist
        assert "bins" in dist

    def test_empty_distribution(self, scorer):
        dist = scorer.get_distribution([])
        assert dist["total"] == 0

    def test_score_structure(self, scorer):
        fb = _fb("Q", "A long enough response.")
        score = scorer.score(fb)
        d = score.to_dict()
        assert all(
            k in d for k in ["completeness", "coherence", "relevance", "informativeness", "overall"]
        )

    def test_custom_config(self):
        config = QualityScorerConfig(min_quality=0.8, min_response_length=50)
        scorer = DataQualityScorer(config=config)
        fb = _fb("Short", "Short resp")
        score = scorer.score(fb)
        assert score.completeness < 1.0
