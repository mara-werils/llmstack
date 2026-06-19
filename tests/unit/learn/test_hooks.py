"""Tests for the learning integration hooks."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from llmstack.learn.config import LearnConfig
from llmstack.learn.feedback import Feedback, FeedbackType
from llmstack.learn.hooks import LearningHooks, create_hooks
from llmstack.learn.preferences import PreferenceLearner
from llmstack.learn.store import FeedbackStore


@pytest.fixture
def store(tmp_path):
    s = FeedbackStore(db_path=tmp_path / "test.db")
    yield s
    s.close()


@pytest.fixture
def preference_learner(store, tmp_path):
    return PreferenceLearner(store=store, preferences_path=tmp_path / "prefs.json")


@pytest.fixture
def config():
    return LearnConfig()


@pytest.fixture
def hooks(store, preference_learner, config):
    return LearningHooks(
        store=store,
        preference_learner=preference_learner,
        regression_detector=None,
        config=config,
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestInit:
    def test_defaults(self, store, preference_learner):
        h = LearningHooks(store=store, preference_learner=preference_learner)
        assert h.store is store
        assert h.preference_learner is preference_learner
        assert h.regression_detector is None
        assert isinstance(h.config, LearnConfig)
        assert h._interaction_count == 0
        assert h._current_query == ""
        assert h._current_response == ""
        assert h._current_model == ""

    def test_config_default_created_when_none(self, store, preference_learner):
        h = LearningHooks(
            store=store,
            preference_learner=preference_learner,
            config=None,
        )
        assert isinstance(h.config, LearnConfig)

    def test_custom_config_kept(self, store, preference_learner):
        cfg = LearnConfig()
        cfg.enabled = False
        h = LearningHooks(
            store=store,
            preference_learner=preference_learner,
            config=cfg,
        )
        assert h.config is cfg


# ---------------------------------------------------------------------------
# pre_generate
# ---------------------------------------------------------------------------


class TestPreGenerate:
    def test_disabled_returns_unchanged(self, hooks):
        hooks.config.enabled = False
        messages = [{"role": "user", "content": "hi"}]
        result = hooks.pre_generate(messages, model="gpt")
        assert result is messages

    def test_inject_disabled_returns_unchanged(self, hooks):
        hooks.config.preferences.inject_into_prompts = False
        messages = [{"role": "user", "content": "hi"}]
        result = hooks.pre_generate(messages)
        assert result is messages

    def test_no_additions_returns_unchanged(self, hooks, monkeypatch):
        monkeypatch.setattr(
            hooks.preference_learner,
            "get_system_prompt_additions",
            lambda: "",
        )
        messages = [{"role": "user", "content": "hi"}]
        result = hooks.pre_generate(messages)
        assert result is messages

    def test_injects_into_existing_system_message(self, hooks, monkeypatch):
        monkeypatch.setattr(
            hooks.preference_learner,
            "get_system_prompt_additions",
            lambda: "BE CONCISE",
        )
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "hi"},
        ]
        result = hooks.pre_generate(messages, model="gpt-4")
        # original not mutated
        assert messages[0]["content"] == "You are helpful."
        assert result is not messages
        assert result[0]["role"] == "system"
        assert "You are helpful." in result[0]["content"]
        assert "BE CONCISE" in result[0]["content"]
        assert hooks._current_model == "gpt-4"

    def test_inserts_system_message_when_absent(self, hooks, monkeypatch):
        monkeypatch.setattr(
            hooks.preference_learner,
            "get_system_prompt_additions",
            lambda: "BE CONCISE",
        )
        messages = [{"role": "user", "content": "hi"}]
        result = hooks.pre_generate(messages)
        assert result[0] == {"role": "system", "content": "BE CONCISE"}
        assert result[1]["role"] == "user"
        # original list untouched
        assert len(messages) == 1

    def test_inserts_when_messages_empty(self, hooks, monkeypatch):
        monkeypatch.setattr(
            hooks.preference_learner,
            "get_system_prompt_additions",
            lambda: "BE CONCISE",
        )
        result = hooks.pre_generate([])
        assert result == [{"role": "system", "content": "BE CONCISE"}]


# ---------------------------------------------------------------------------
# post_generate
# ---------------------------------------------------------------------------


class TestPostGenerate:
    def test_disabled_noop(self, hooks):
        hooks.config.enabled = False
        hooks.post_generate("q", "r", model="m", quality_score=0.9)
        assert hooks._interaction_count == 0
        assert hooks._current_query == ""

    def test_records_interaction_state(self, hooks):
        hooks.post_generate("the query", "the response", model="gpt")
        assert hooks._current_query == "the query"
        assert hooks._current_response == "the response"
        assert hooks._current_model == "gpt"
        assert hooks._interaction_count == 1

    def test_model_fallback_to_current(self, hooks):
        hooks._current_model = "previous-model"
        hooks.post_generate("q", "r", model="")
        assert hooks._current_model == "previous-model"

    def test_no_regression_detector_skips_quality(self, hooks):
        # regression_detector is None — should not error even with score
        hooks.post_generate("q", "r", quality_score=0.8)
        assert hooks._interaction_count == 1

    def test_records_quality_when_detector_and_active(self, store, preference_learner):
        detector = MagicMock()
        active = MagicMock()
        active.version = "v2"
        detector.version_mgr.get_active.return_value = active
        h = LearningHooks(
            store=store,
            preference_learner=preference_learner,
            regression_detector=detector,
        )
        h.post_generate("q", "r", model="m", quality_score=0.75)
        detector.record_quality.assert_called_once_with(
            model_version="v2",
            metric="overall",
            value=0.75,
        )

    def test_no_quality_record_when_score_zero(self, store, preference_learner):
        detector = MagicMock()
        h = LearningHooks(
            store=store,
            preference_learner=preference_learner,
            regression_detector=detector,
        )
        h.post_generate("q", "r", quality_score=0.0)
        detector.record_quality.assert_not_called()

    def test_no_quality_record_when_no_active_version(self, store, preference_learner):
        detector = MagicMock()
        detector.version_mgr.get_active.return_value = None
        h = LearningHooks(
            store=store,
            preference_learner=preference_learner,
            regression_detector=detector,
        )
        h.post_generate("q", "r", quality_score=0.9)
        detector.record_quality.assert_not_called()


# ---------------------------------------------------------------------------
# on_feedback
# ---------------------------------------------------------------------------


class TestOnFeedback:
    def test_disabled_noop(self, hooks):
        hooks.config.enabled = False
        hooks.on_feedback(FeedbackType.THUMBS_UP)
        assert hooks.store.get_feedback() == []

    def test_stores_feedback(self, hooks):
        hooks.post_generate("the query", "the response", model="gpt")
        hooks.on_feedback(FeedbackType.THUMBS_UP, command="ask")
        stored = hooks.store.get_feedback()
        assert len(stored) == 1
        fb = stored[0]
        assert fb.feedback_type == FeedbackType.THUMBS_UP
        assert fb.query == "the query"
        assert fb.response == "the response"
        assert fb.model == "gpt"
        assert fb.command == "ask"

    def test_kwargs_become_context(self, hooks):
        hooks.on_feedback(FeedbackType.THUMBS_UP, session="abc", turn=3)
        fb = hooks.store.get_feedback()[0]
        assert fb.context == {"session": "abc", "turn": 3}

    def test_correction_triggers_preference_learning(self, hooks):
        called = {}

        def fake_learn(feedback):
            called["fb"] = feedback

        hooks.preference_learner.learn_from_feedback = fake_learn
        hooks.post_generate("q", "original response", model="m")
        hooks.on_feedback(
            FeedbackType.CORRECTION,
            correction="better response",
        )
        assert "fb" in called
        assert isinstance(called["fb"], Feedback)
        assert called["fb"].correction == "better response"

    def test_no_correction_skips_preference_learning(self, hooks):
        hooks.preference_learner.learn_from_feedback = MagicMock()
        hooks.on_feedback(FeedbackType.THUMBS_DOWN, rating=1)
        hooks.preference_learner.learn_from_feedback.assert_not_called()


# ---------------------------------------------------------------------------
# Implicit signals: on_copy / on_regenerate / on_abandon
# ---------------------------------------------------------------------------


class TestImplicitSignals:
    def test_on_copy_records(self, hooks):
        hooks.post_generate("q", "r")
        hooks.on_copy()
        stored = hooks.store.get_feedback()
        assert len(stored) == 1
        assert stored[0].feedback_type == FeedbackType.COPY

    def test_on_copy_disabled_globally(self, hooks):
        hooks.config.enabled = False
        hooks.on_copy()
        assert hooks.store.get_feedback() == []

    def test_on_copy_implicit_disabled(self, hooks):
        hooks.config.feedback.implicit_signals = False
        hooks.on_copy()
        assert hooks.store.get_feedback() == []

    def test_on_regenerate_records(self, hooks):
        hooks.post_generate("q", "r")
        hooks.on_regenerate()
        stored = hooks.store.get_feedback()
        assert len(stored) == 1
        assert stored[0].feedback_type == FeedbackType.REGENERATE

    def test_on_regenerate_implicit_disabled(self, hooks):
        hooks.config.feedback.implicit_signals = False
        hooks.on_regenerate()
        assert hooks.store.get_feedback() == []

    def test_on_abandon_records_when_response_present(self, hooks):
        hooks.post_generate("q", "some response")
        hooks.on_abandon()
        stored = hooks.store.get_feedback()
        assert len(stored) == 1
        assert stored[0].feedback_type == FeedbackType.ABANDON

    def test_on_abandon_noop_without_response(self, hooks):
        # no post_generate; _current_response is ""
        hooks.on_abandon()
        assert hooks.store.get_feedback() == []

    def test_on_abandon_implicit_disabled(self, hooks):
        hooks.config.feedback.implicit_signals = False
        hooks.post_generate("q", "r")
        hooks.on_abandon()
        assert hooks.store.get_feedback() == []


# ---------------------------------------------------------------------------
# should_prompt_feedback / get_feedback_prompt
# ---------------------------------------------------------------------------


class TestShouldPromptFeedback:
    def test_false_when_disabled(self, hooks):
        hooks.config.enabled = False
        hooks._interaction_count = 5
        assert hooks.should_prompt_feedback() is False

    def test_false_when_interactive_disabled(self, hooks):
        hooks.config.feedback.interactive_feedback = False
        hooks._interaction_count = 5
        assert hooks.should_prompt_feedback() is False

    def test_false_at_zero_interactions(self, hooks):
        hooks._interaction_count = 0
        assert hooks.should_prompt_feedback() is False

    def test_true_at_interval(self, hooks):
        hooks.config.feedback.prompt_interval = 5
        hooks._interaction_count = 5
        assert hooks.should_prompt_feedback() is True

    def test_true_at_multiple_of_interval(self, hooks):
        hooks.config.feedback.prompt_interval = 3
        hooks._interaction_count = 9
        assert hooks.should_prompt_feedback() is True

    def test_false_between_intervals(self, hooks):
        hooks.config.feedback.prompt_interval = 5
        hooks._interaction_count = 6
        assert hooks.should_prompt_feedback() is False

    def test_integration_via_post_generate(self, hooks):
        hooks.config.feedback.prompt_interval = 2
        hooks.post_generate("q", "r")
        assert hooks.should_prompt_feedback() is False
        hooks.post_generate("q", "r")
        assert hooks.should_prompt_feedback() is True


class TestGetFeedbackPrompt:
    def test_returns_text(self, hooks):
        prompt = hooks.get_feedback_prompt()
        assert isinstance(prompt, str)
        assert "Learning" in prompt
        assert "y/n" in prompt


# ---------------------------------------------------------------------------
# create_hooks factory
# ---------------------------------------------------------------------------


class TestCreateHooks:
    def _make_config(self, tmp_path: Path, *, quality_enabled: bool) -> LearnConfig:
        cfg = LearnConfig()
        cfg.storage.db_path = str(tmp_path / "learning.db")
        cfg.storage.preferences_path = str(tmp_path / "prefs.json")
        cfg.storage.versions_dir = str(tmp_path / "versions")
        cfg.quality.enabled = quality_enabled
        return cfg

    def test_creates_hooks_with_regression(self, tmp_path):
        cfg = self._make_config(tmp_path, quality_enabled=True)
        h = create_hooks(cfg)
        try:
            assert isinstance(h, LearningHooks)
            assert h.config is cfg
            assert isinstance(h.store, FeedbackStore)
            assert isinstance(h.preference_learner, PreferenceLearner)
            assert h.regression_detector is not None
        finally:
            h.store.close()

    def test_no_regression_when_quality_disabled(self, tmp_path):
        cfg = self._make_config(tmp_path, quality_enabled=False)
        h = create_hooks(cfg)
        try:
            assert h.regression_detector is None
        finally:
            h.store.close()

    def test_default_config_when_none(self, tmp_path, monkeypatch):
        # Avoid touching the real home directory by patching LearnConfig default.
        cfg = self._make_config(tmp_path, quality_enabled=True)
        monkeypatch.setattr("llmstack.learn.hooks.LearnConfig", lambda: cfg)
        h = create_hooks()
        try:
            assert isinstance(h, LearningHooks)
            assert h.config is cfg
        finally:
            h.store.close()
