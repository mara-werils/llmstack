"""Tests for llmstack.learn.active active learning feedback selection."""

from __future__ import annotations

from llmstack.learn.active import ActiveLearner, ActiveLearningConfig, UncertaintySignal


class _FakeRecord:
    def __init__(self, query):
        self.query = query


class _FakeStore:
    def __init__(self, records=None):
        self._records = records or []

    def get_feedback(self, limit=100):
        return self._records


def _learner(**config_overrides):
    config = ActiveLearningConfig(**config_overrides)
    return ActiveLearner(store=_FakeStore(), config=config)


def test_requests_remaining_and_interaction_count():
    learner = _learner(max_requests_per_session=3)
    assert learner.requests_remaining == 3
    assert learner.interaction_count == 0
    learner._session_requests = 2
    assert learner.requests_remaining == 1
    learner._session_requests = 10
    assert learner.requests_remaining == 0  # clamped at 0


def test_should_request_feedback_blocked_by_session_limit():
    learner = _learner(max_requests_per_session=0, warmup_interactions=0, cooldown_interactions=0)
    assert learner.should_request_feedback("q", "r") is False


def test_should_request_feedback_blocked_during_warmup():
    learner = _learner(warmup_interactions=5, cooldown_interactions=0)
    assert learner.should_request_feedback("q", "r") is False
    assert learner.interaction_count == 1


def test_should_request_feedback_blocked_by_cooldown():
    learner = _learner(warmup_interactions=1, cooldown_interactions=5)
    learner.should_request_feedback("q1", "r1")
    learner.mark_requested("q1")
    result = learner.should_request_feedback("q2", "r2")
    assert result is False


def test_should_request_feedback_blocked_by_low_uncertainty():
    learner = _learner(warmup_interactions=1, cooldown_interactions=0, uncertainty_threshold=2.0)
    assert learner.should_request_feedback("hello", "a normal confident answer") is False


def test_should_request_feedback_blocked_by_similarity_to_asked():
    learner = _learner(warmup_interactions=1, cooldown_interactions=0, uncertainty_threshold=0.0)
    learner._asked_queries.append("how do i reset my password")
    result = learner.should_request_feedback("how do i reset my password", "I'm not sure")
    assert result is False


def test_should_request_feedback_returns_true_when_all_checks_pass():
    learner = _learner(warmup_interactions=1, cooldown_interactions=0, uncertainty_threshold=0.0)
    assert learner.should_request_feedback("a totally new query", "an answer") is True


def test_mark_requested_updates_state():
    learner = _learner()
    learner._interaction_count = 7
    learner.mark_requested("my query")
    assert learner._session_requests == 1
    assert learner._last_request_at == 7
    assert learner._asked_queries == ["my query"]


def test_estimate_uncertainty_combines_signals():
    learner = _learner()
    signal = learner.estimate_uncertainty("short", "I'm not sure, but maybe it's this.")
    assert isinstance(signal, UncertaintySignal)
    assert 0.0 <= signal.overall <= 1.0


def test_get_feedback_prompt_hedging_branch():
    learner = _learner()
    prompt = learner.get_feedback_prompt("q", "I'm not sure, I think maybe it could be, perhaps")
    assert "not fully confident" in prompt


def test_get_feedback_prompt_novelty_branch():
    store = _FakeStore(records=[])  # no history -> novelty 0.8 > 0.6
    learner = ActiveLearner(store=store)
    prompt = learner.get_feedback_prompt("a brand new kind of question", "a plain confident answer")
    assert "new type of question" in prompt


def test_get_feedback_prompt_default_branch():
    store = _FakeStore(records=[_FakeRecord("a brand new kind of question")])
    learner = ActiveLearner(store=store)
    prompt = learner.get_feedback_prompt("a brand new kind of question", "a plain confident answer")
    assert "Quick feedback" in prompt


class TestScoreHedging:
    def test_no_hedging_language(self):
        learner = _learner()
        assert learner._score_hedging("This is definitely correct.") == 0.0

    def test_some_hedging_language(self):
        learner = _learner()
        score = learner._score_hedging("I think it might be correct, perhaps.")
        assert score > 0.0

    def test_caps_at_one(self):
        learner = _learner()
        response = (
            "I'm not sure, it might be, I think, it's possible, perhaps, "
            "if I'm not mistaken, I believe, it seems like"
        )
        assert learner._score_hedging(response) == 1.0


class TestScoreLengthAnomaly:
    def test_empty_query_returns_zero(self):
        learner = _learner()
        assert learner._score_length_anomaly("", "some response") == 0.0

    def test_short_response_to_long_query(self):
        learner = _learner()
        query = "x" * 60
        response = "ok"
        assert learner._score_length_anomaly(query, response) == 0.7

    def test_long_response_to_short_query(self):
        learner = _learner()
        query = "hi"
        response = "y" * 50
        assert learner._score_length_anomaly(query, response) == 0.5

    def test_normal_ratio_returns_zero(self):
        learner = _learner()
        assert learner._score_length_anomaly("a normal query here", "a normal response here") == 0.0


class TestScoreRepetition:
    def test_short_response_returns_zero(self):
        learner = _learner()
        assert learner._score_repetition("too short") == 0.0

    def test_few_sentences_returns_zero(self):
        learner = _learner()
        response = "x" * 150
        assert learner._score_repetition(response) == 0.0

    def test_repetitive_sentences_score_high(self):
        learner = _learner()
        sentence = "this is a repeated sentence that is long enough to count. "
        response = sentence * 5
        score = learner._score_repetition(response)
        assert score > 0.0


class TestScoreNovelty:
    def test_no_history_is_fully_novel(self):
        learner = ActiveLearner(store=_FakeStore(records=[]))
        assert learner._score_novelty("anything") == 0.8

    def test_empty_query_words_returns_half(self):
        learner = ActiveLearner(store=_FakeStore(records=[_FakeRecord("some query")]))
        assert learner._score_novelty("   ") == 0.5

    def test_overlapping_query_reduces_novelty(self):
        learner = ActiveLearner(store=_FakeStore(records=[_FakeRecord("reset my password")]))
        score = learner._score_novelty("reset my password")
        assert score == 0.0

    def test_record_with_empty_query_is_skipped(self):
        learner = ActiveLearner(
            store=_FakeStore(records=[_FakeRecord(""), _FakeRecord("reset my password")])
        )
        score = learner._score_novelty("reset my password")
        assert score == 0.0


class TestIsSimilarToAsked:
    def test_no_asked_queries_returns_false(self):
        learner = _learner()
        assert learner._is_similar_to_asked("anything") is False

    def test_skips_blank_asked_queries(self):
        learner = _learner()
        learner._asked_queries.append("   ")
        assert learner._is_similar_to_asked("anything") is False

    def test_similar_query_returns_true(self):
        learner = _learner(diversity_threshold=0.5)
        learner._asked_queries.append("how do i reset my password")
        assert learner._is_similar_to_asked("how do i reset my password please") is True

    def test_dissimilar_query_returns_false(self):
        learner = _learner(diversity_threshold=0.9)
        learner._asked_queries.append("how do i reset my password")
        assert learner._is_similar_to_asked("completely unrelated topic here") is False
