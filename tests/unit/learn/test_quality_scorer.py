"""Tests for data quality scoring."""

from __future__ import annotations

import pytest

from llmstack.learn.quality_scorer import (
    DataQualityScorer,
    QualityScore,
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

    def test_thumbs_down_without_correction_penalized(self, scorer):
        fb = _fb(
            "How do I do X",
            "Here is a reasonably long but apparently wrong answer about X.",
            ftype=FeedbackType.THUMBS_DOWN,
        )
        score = scorer.score(fb)
        assert score.completeness < 1.0

    def test_broken_short_sentences_low_coherence(self, scorer):
        fb = _fb("Q", "Hi. Ok. No. Um. Eh. Hm. So. Bo. Ko. Po.")
        score = scorer.score(fb)
        assert score.coherence < 0.8

    def test_garbled_non_ascii_response_low_coherence(self, scorer):
        fb = _fb("Q", "中文中文中文中文中文 mostly garbled")
        score = scorer.score(fb)
        assert score.coherence < 0.8

    def test_relevance_falls_back_when_only_stopwords(self, scorer):
        fb = _fb("the a an is", "some unrelated response text")
        score = scorer.score(fb)
        assert score.relevance == 0.5

    def test_very_long_response_max_informativeness_boost(self, scorer):
        fb = _fb("Explain in detail", "x" * 600)
        score = scorer.score(fb)
        assert score.informativeness == pytest.approx(0.8)

    def test_list_structure_boosts_informativeness(self, scorer):
        fb = _fb("List options", "Options:\n- one\n- two\n- three")
        score = scorer.score(fb)
        assert score.informativeness > 0.5


class TestQualityScoreProperties:
    def test_passes_threshold(self):
        score = QualityScore(overall=0.5)
        assert score.passes_threshold(0.3) is True
        assert score.passes_threshold(0.6) is False

    def test_best_and_worst_dimension(self):
        score = QualityScore(completeness=0.9, coherence=0.2, relevance=0.5, informativeness=0.4)
        assert score.best_dimension == "completeness"
        assert score.worst_dimension == "coherence"


class TestDistributionBins:
    def test_all_bins_populated(self, scorer, monkeypatch):
        values = iter([0.1, 0.4, 0.6, 0.9])
        monkeypatch.setattr(scorer, "score", lambda fb: QualityScore(overall=next(values)))
        entries = [_fb("q", "r") for _ in range(4)]

        dist = scorer.get_distribution(entries)

        assert dist["bins"] == {"low": 1, "medium": 1, "high": 1, "excellent": 1}
