"""Tests for the feedback collector convenience API."""

from __future__ import annotations

import pytest

from llmstack.learn.collector import FeedbackCollector
from llmstack.learn.config import LearnConfig
from llmstack.learn.feedback import Feedback, FeedbackType
from llmstack.learn.patterns import PatternLearner
from llmstack.learn.preferences import PreferenceLearner
from llmstack.learn.store import FeedbackStore


@pytest.fixture
def store(tmp_path):
    """Create a temporary, real feedback store."""
    s = FeedbackStore(db_path=tmp_path / "learning.db")
    yield s
    s.close()


@pytest.fixture
def config(tmp_path):
    """LearnConfig pointing all storage paths at tmp_path."""
    cfg = LearnConfig()
    cfg.storage.db_path = str(tmp_path / "learning.db")
    cfg.storage.preferences_path = str(tmp_path / "preferences.json")
    cfg.storage.prompts_dir = str(tmp_path / "prompts")
    return cfg


@pytest.fixture
def collector(store, config):
    """Collector backed by a shared tmp store and tmp-backed config paths."""
    c = FeedbackCollector(store=store, config=config)
    c.record_interaction(
        query="What is Python?",
        response="Python is a programming language.",
        model="llama3.2",
        command="ask",
    )
    return c


class TestInit:
    def test_default_construction(self):
        c = FeedbackCollector()
        assert isinstance(c.config, LearnConfig)
        assert c.interaction_count == 0
        assert not c.has_pending_interaction

    def test_construction_with_explicit_store_and_config(self, store, config):
        c = FeedbackCollector(store=store, config=config)
        assert c.config is config
        # the explicit store is returned by the lazy property without rebuild
        assert c.store is store


class TestLazyProperties:
    def test_store_lazy_built_from_config(self, config):
        c = FeedbackCollector(config=config)
        s = c.store
        assert isinstance(s, FeedbackStore)
        # cached on repeat access
        assert c.store is s
        s.close()

    def test_preference_learner_lazy(self, collector):
        pl = collector.preference_learner
        assert isinstance(pl, PreferenceLearner)
        # cached
        assert collector.preference_learner is pl

    def test_pattern_learner_lazy(self, collector):
        pl = collector.pattern_learner
        assert isinstance(pl, PatternLearner)
        # cached
        assert collector.pattern_learner is pl


class TestRecordInteraction:
    def test_record_interaction_sets_state(self, collector):
        assert collector.interaction_count == 1
        assert collector.has_pending_interaction

    def test_record_interaction_increments(self, collector):
        collector.record_interaction("q2", "r2")
        assert collector.interaction_count == 2

    def test_record_interaction_defaults(self, store, config):
        c = FeedbackCollector(store=store, config=config)
        c.record_interaction("q", "r")
        fb = c.thumbs_up()
        assert fb.model == ""
        assert fb.command == ""


class TestExplicitSignals:
    def test_thumbs_up(self, collector, store):
        fb = collector.thumbs_up()
        assert fb.feedback_type == FeedbackType.THUMBS_UP
        assert fb.query == "What is Python?"
        assert store.get_stats()["total_feedback"] == 1

    def test_thumbs_down(self, collector, store):
        fb = collector.thumbs_down()
        assert fb.feedback_type == FeedbackType.THUMBS_DOWN
        assert store.get_stats()["total_feedback"] == 1

    def test_correct(self, collector):
        fb = collector.correct("a better answer")
        assert fb.feedback_type == FeedbackType.CORRECTION
        assert fb.correction == "a better answer"
        assert fb.has_correction

    def test_edit(self, collector):
        fb = collector.edit("an edited answer")
        assert fb.feedback_type == FeedbackType.EDIT
        assert fb.correction == "an edited answer"
        assert fb.has_correction

    def test_prefer(self, collector):
        fb = collector.prefer(preferred="good one", rejected="bad one")
        assert fb.feedback_type == FeedbackType.PREFERENCE
        assert fb.correction == "good one"
        assert fb.preferred_over == "bad one"


class TestImplicitSignals:
    def test_on_regenerate(self, collector):
        fb = collector.on_regenerate()
        assert fb.feedback_type == FeedbackType.REGENERATE
        assert fb.is_negative

    def test_on_copy(self, collector):
        fb = collector.on_copy()
        assert fb.feedback_type == FeedbackType.COPY
        assert fb.is_positive


class TestSubmitUpdatesLearners:
    def test_correction_invokes_learners(self, collector, monkeypatch):
        pref_calls: list[Feedback] = []
        pat_calls: list[Feedback] = []
        monkeypatch.setattr(
            collector.preference_learner,
            "learn_from_feedback",
            lambda fb: pref_calls.append(fb),
        )
        monkeypatch.setattr(
            collector.pattern_learner,
            "learn_from_feedback",
            lambda fb: pat_calls.append(fb),
        )
        collector.correct("better")
        assert len(pref_calls) == 1
        assert len(pat_calls) == 1

    def test_non_correction_skips_learners(self, collector, monkeypatch):
        called: list[str] = []
        monkeypatch.setattr(
            collector.preference_learner,
            "learn_from_feedback",
            lambda fb: called.append("pref"),
        )
        monkeypatch.setattr(
            collector.pattern_learner,
            "learn_from_feedback",
            lambda fb: called.append("pat"),
        )
        collector.thumbs_up()
        assert called == []


class TestShouldPrompt:
    def test_no_prompt_when_disabled(self, store, config):
        config.enabled = False
        c = FeedbackCollector(store=store, config=config)
        c.record_interaction("q", "r")
        assert c.should_prompt() is False

    def test_no_prompt_when_interactive_off(self, store, config):
        config.feedback.interactive_feedback = False
        c = FeedbackCollector(store=store, config=config)
        c.record_interaction("q", "r")
        assert c.should_prompt() is False

    def test_no_prompt_with_zero_interactions(self, store, config):
        c = FeedbackCollector(store=store, config=config)
        assert c.should_prompt() is False

    def test_prompt_on_interval(self, store, config):
        config.feedback.prompt_interval = 3
        c = FeedbackCollector(store=store, config=config)
        for _ in range(3):
            c.record_interaction("q", "r")
        assert c.should_prompt() is True

    def test_no_prompt_off_interval(self, store, config):
        config.feedback.prompt_interval = 3
        c = FeedbackCollector(store=store, config=config)
        c.record_interaction("q", "r")
        assert c.should_prompt() is False


class TestParseFeedbackInput:
    def test_skip_inputs_return_none(self, collector):
        for raw in ("", "  ", "s", "skip", "SKIP"):
            assert collector.parse_feedback_input(raw) is None

    @pytest.mark.parametrize("raw", ["y", "yes", "+", "good", "ok", "OK"])
    def test_positive_inputs(self, collector, raw):
        fb = collector.parse_feedback_input(raw)
        assert fb is not None
        assert fb.feedback_type == FeedbackType.THUMBS_UP

    @pytest.mark.parametrize("raw", ["n", "no", "-", "bad"])
    def test_negative_inputs(self, collector, raw):
        fb = collector.parse_feedback_input(raw)
        assert fb is not None
        assert fb.feedback_type == FeedbackType.THUMBS_DOWN

    def test_edit_prefix(self, collector):
        fb = collector.parse_feedback_input("e: a much better answer")
        assert fb is not None
        assert fb.feedback_type == FeedbackType.EDIT
        assert fb.correction == "a much better answer"

    def test_edit_long_prefix(self, collector):
        fb = collector.parse_feedback_input("edit: changed it")
        assert fb is not None
        assert fb.feedback_type == FeedbackType.EDIT
        assert fb.correction == "changed it"

    def test_correct_prefix(self, collector):
        fb = collector.parse_feedback_input("c: the right answer")
        assert fb is not None
        assert fb.feedback_type == FeedbackType.CORRECTION
        assert fb.correction == "the right answer"

    def test_correct_long_prefix(self, collector):
        fb = collector.parse_feedback_input("correct: fixed")
        assert fb is not None
        assert fb.feedback_type == FeedbackType.CORRECTION
        assert fb.correction == "fixed"

    def test_edit_prefix_empty_correction_returns_none(self, collector):
        assert collector.parse_feedback_input("e:   ") is None

    def test_correct_prefix_empty_correction_returns_none(self, collector):
        assert collector.parse_feedback_input("c:") is None

    def test_unrecognized_input_returns_none(self, collector):
        assert collector.parse_feedback_input("what is this") is None


class TestGetStats:
    def test_get_stats_shape(self, collector, store):
        collector.thumbs_up()
        stats = collector.get_stats()
        assert stats["interactions"] == 1
        assert stats["current_query"] == "What is Python?"
        assert stats["model"] == "llama3.2"
        assert stats["total_stored"] == 1
        assert stats["pending"] == 1  # freshly stored feedback is unused

    def test_get_stats_truncates_long_query(self, store, config):
        c = FeedbackCollector(store=store, config=config)
        long_query = "x" * 100
        c.record_interaction(long_query, "resp")
        stats = c.get_stats()
        assert stats["current_query"] == "x" * 50

    def test_get_stats_empty_query(self, store, config):
        c = FeedbackCollector(store=store, config=config)
        stats = c.get_stats()
        assert stats["current_query"] == ""
        assert stats["interactions"] == 0


class TestClose:
    def test_close_with_store(self, store, config):
        c = FeedbackCollector(store=store, config=config)
        # accessing close should call the underlying store's close
        closed = {"flag": False}

        def fake_close():
            closed["flag"] = True

        store.close = fake_close  # type: ignore[method-assign]
        c.close()
        assert closed["flag"] is True

    def test_close_without_store_is_noop(self, config):
        c = FeedbackCollector(config=config)
        # _store is None — close should not raise
        c.close()
