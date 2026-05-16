"""Tests for active learning feedback selection."""

from __future__ import annotations

import pytest

from llmstack.learn.active import ActiveLearner, ActiveLearningConfig
from llmstack.learn.feedback import Feedback, FeedbackType
from llmstack.learn.store import FeedbackStore


@pytest.fixture
def store(tmp_path):
    s = FeedbackStore(db_path=tmp_path / "test.db")
    yield s
    s.close()


@pytest.fixture
def learner(store):
    config = ActiveLearningConfig(
        max_requests_per_session=3,
        warmup_interactions=2,
        cooldown_interactions=2,
        uncertainty_threshold=0.3,
    )
    return ActiveLearner(store=store, config=config)


class TestActiveLearner:
    def test_warmup_period(self, learner):
        """No feedback requests during warmup."""
        assert not learner.should_request_feedback("Hello", "Hi there!")

    def test_requests_after_warmup(self, learner):
        """May request feedback after warmup with uncertain response."""
        # Advance past warmup
        learner._interaction_count = 5
        learner._last_request_at = 0

        # Uncertain response (hedging language)
        result = learner.should_request_feedback(
            "How does async work?",
            "I think maybe it might possibly work by perhaps using coroutines, "
            "although I'm not entirely certain about the exact mechanism.",
        )
        # Should request due to hedging
        assert result is True

    def test_max_requests_limit(self, learner):
        """Stops requesting after max per session."""
        learner._interaction_count = 100
        learner._session_requests = 3  # at max

        result = learner.should_request_feedback("query", "I'm not sure maybe...")
        assert result is False

    def test_cooldown(self, learner):
        """Respects cooldown between requests."""
        # should_request_feedback increments _interaction_count first, so:
        # after increment: count=4, last_request_at=3, diff=1 < cooldown=2
        learner._interaction_count = 3
        learner._last_request_at = 3

        result = learner.should_request_feedback("query", "I think maybe perhaps...")
        assert result is False

    def test_uncertainty_estimation(self, learner):
        """Estimates uncertainty from response signals."""
        # Confident response
        confident = learner.estimate_uncertainty(
            "What is 2+2?",
            "The answer is 4.",
        )
        assert confident.overall < 0.5

        # Hedging response
        uncertain = learner.estimate_uncertainty(
            "What is the best framework?",
            "I think it might possibly be React, although maybe Vue could "
            "perhaps be better in some cases. I'm not entirely sure.",
        )
        assert uncertain.overall > confident.overall
        assert uncertain.hedging_score > 0.3

    def test_novelty_scoring(self, learner, store):
        """Novel queries score higher for feedback value."""
        # Add some history
        for i in range(10):
            store.add_feedback(Feedback(
                feedback_type=FeedbackType.THUMBS_UP,
                query=f"How do I write Python code for task {i}?",
                response=f"response {i}",
            ))

        # Similar query — low novelty
        similar = learner.estimate_uncertainty(
            "How do I write Python code for a new task?",
            "response",
        )

        # Very different query — high novelty
        novel = learner.estimate_uncertainty(
            "Explain quantum entanglement in simple terms",
            "response",
        )

        assert novel.novelty_score > similar.novelty_score

    def test_diversity_filter(self, learner):
        """Doesn't ask about queries too similar to already-asked ones."""
        learner._interaction_count = 10
        learner._last_request_at = 0

        # Mark a query as asked
        learner._asked_queries.append("How do I sort a list in Python?")

        # Very similar query should be filtered
        result = learner.should_request_feedback(
            "How do I sort a list in Python efficiently?",
            "I think maybe you could possibly try using sorted()...",
        )
        assert result is False
