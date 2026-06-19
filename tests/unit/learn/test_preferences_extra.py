"""Additional coverage for user preference learning.

Targets branches/lines not exercised by test_preferences.py:
- LengthPreference.has_signal / length_ratio / neutral tendency
- FormatPreference & TonePreference "decrease" / increase branches
- UserPreferences.to_system_prompt_additions (detailed/bullets/headers/formal/casual)
- PreferenceLearner.learn_from_feedback EDIT & PREFERENCE paths
- PreferenceLearner.rebuild_from_history
- PreferenceLearner._load malformed-JSON fallback
"""

from __future__ import annotations

import pytest

from llmstack.learn.feedback import Feedback, FeedbackType
from llmstack.learn.preferences import (
    FormatPreference,
    LengthPreference,
    PreferenceLearner,
    TonePreference,
    UserPreferences,
)
from llmstack.learn.store import FeedbackStore


@pytest.fixture
def store(tmp_path):
    s = FeedbackStore(db_path=tmp_path / "test.db")
    yield s
    s.close()


@pytest.fixture
def learner(store, tmp_path):
    return PreferenceLearner(
        store=store,
        preferences_path=tmp_path / "prefs.json",
    )


class TestLengthPreference:
    def test_has_signal_thresholds(self):
        lp = LengthPreference()
        assert lp.has_signal is False
        lp.samples = 3
        assert lp.has_signal is True

    def test_length_ratio_zero_rejected(self):
        lp = LengthPreference(avg_rejected_length=0.0)
        assert lp.length_ratio == 1.0

    def test_length_ratio_value(self):
        lp = LengthPreference(avg_preferred_length=200.0, avg_rejected_length=100.0)
        assert lp.length_ratio == pytest.approx(2.0)

    def test_tendency_neutral_when_similar(self):
        """preferred/rejected close together -> neutral even with enough samples."""
        lp = LengthPreference()
        for _ in range(5):
            lp.update(preferred_len=100.0, rejected_len=100.0)
        assert lp.samples >= 3
        assert lp.tendency == "neutral"


class TestFormatPreferenceDecrease:
    def test_all_features_decrease(self):
        """correction strips code blocks/bullets/headers/markdown vs original."""
        fmt = FormatPreference()
        before = (
            fmt.prefers_code_blocks,
            fmt.prefers_bullet_lists,
            fmt.prefers_headers,
            fmt.prefers_markdown,
        )
        original = "```py\nx\n```\n- a\n- b\n# Head\n**bold** `c` [x](y)"
        correction = "plain text answer"
        fmt.update(correction, original)
        assert fmt.prefers_code_blocks < before[0]
        assert fmt.prefers_bullet_lists < before[1]
        assert fmt.prefers_headers < before[2]
        assert fmt.prefers_markdown < before[3]

    def test_star_bullets_increase(self):
        fmt = FormatPreference()
        original = "plain"
        correction = "\n* one\n* two"
        fmt.update(correction, original)
        assert fmt.prefers_bullet_lists > 0.5

    def test_headers_increase(self):
        fmt = FormatPreference()
        original = "plain answer"
        correction = "\n# Heading\nbody"
        fmt.update(correction, original)
        assert fmt.prefers_headers > 0.5


class TestTonePreference:
    def test_directness_decrease_when_more_hedging(self):
        """correction adds hedging -> directness drops."""
        tone = TonePreference()
        original = "The answer is X."
        correction = "I think perhaps maybe it seems X."
        tone.update(correction, original)
        assert tone.directness < 0.5

    def test_formality_increase_when_contractions_removed(self):
        tone = TonePreference()
        original = "don't won't can't it's"
        correction = "do not will not cannot it is"
        tone.update(correction, original)
        assert tone.formality > 0.5

    def test_formality_decrease_when_contractions_added(self):
        tone = TonePreference()
        original = "do not"
        correction = "don't won't can't"
        tone.update(correction, original)
        assert tone.formality < 0.5


class TestSystemPromptAdditions:
    def test_detailed_length_addition(self):
        prefs = UserPreferences()
        prefs.length.samples = 5
        prefs.length.tendency = "detailed"
        out = prefs.to_system_prompt_additions()
        assert "detailed" in out.lower()

    def test_formatting_additions(self):
        prefs = UserPreferences()
        prefs.formatting.samples = 5
        prefs.formatting.prefers_code_blocks = 0.9
        prefs.formatting.prefers_bullet_lists = 0.9
        prefs.formatting.prefers_headers = 0.9
        out = prefs.to_system_prompt_additions()
        assert "code blocks" in out
        assert "bullet points" in out
        assert "headers" in out

    def test_formal_tone_addition(self):
        prefs = UserPreferences()
        prefs.tone.samples = 5
        prefs.tone.directness = 0.9
        prefs.tone.formality = 0.9
        out = prefs.to_system_prompt_additions()
        assert "direct" in out.lower()
        assert "formal" in out.lower()

    def test_casual_tone_addition(self):
        prefs = UserPreferences()
        prefs.tone.samples = 5
        prefs.tone.formality = 0.1
        out = prefs.to_system_prompt_additions()
        assert "casual" in out.lower()

    def test_no_additions_below_thresholds(self):
        prefs = UserPreferences()
        assert prefs.to_system_prompt_additions() == ""


class TestLearnFromFeedbackBranches:
    def test_edit_feedback(self, learner):
        fb = Feedback(
            feedback_type=FeedbackType.EDIT,
            query="q",
            response="original long winded response",
            correction="short",
        )
        learner.learn_from_feedback(fb)
        assert learner.preferences.total_signals == 1
        assert learner.preferences.length.samples == 1

    def test_preference_feedback(self, learner):
        fb = Feedback(
            feedback_type=FeedbackType.PREFERENCE,
            query="q",
            response="",
            correction="the chosen winner response",
            preferred_over="the losing response that was rejected",
        )
        learner.learn_from_feedback(fb)
        assert learner.preferences.total_signals == 1
        assert learner.preferences.length.samples == 1

    def test_preference_feedback_missing_preferred_over_no_learn(self, learner):
        """PREFERENCE without preferred_over still bumps signals but learns nothing."""
        fb = Feedback(
            feedback_type=FeedbackType.PREFERENCE,
            query="q",
            response="",
            correction="winner",
            preferred_over="",
        )
        learner.learn_from_feedback(fb)
        assert learner.preferences.total_signals == 1
        assert learner.preferences.length.samples == 0

    def test_non_learning_feedback_type(self, learner):
        """THUMBS_UP doesn't learn style but still records a signal + timestamp."""
        fb = Feedback(feedback_type=FeedbackType.THUMBS_UP, query="q", response="r")
        learner.learn_from_feedback(fb)
        assert learner.preferences.total_signals == 1
        assert learner.preferences.last_updated > 0
        assert learner.preferences.length.samples == 0


class TestRebuildFromHistory:
    def test_rebuild_from_history(self, learner, store):
        for _ in range(4):
            store.add_feedback(
                Feedback(
                    feedback_type=FeedbackType.CORRECTION,
                    query="q",
                    response="a very long original response with lots of words",
                    correction="short",
                )
            )
        for _ in range(2):
            store.add_feedback(
                Feedback(
                    feedback_type=FeedbackType.EDIT,
                    query="q",
                    response="another wordy original answer here please",
                    correction="terse",
                )
            )

        learner.rebuild_from_history(limit=100)

        assert learner.preferences.total_signals == 6
        assert learner.preferences.length.samples == 6
        assert learner.preferences.last_updated > 0

    def test_rebuild_skips_corrections_without_text(self, learner, store):
        # correction == "" should be skipped inside the loop
        store.add_feedback(
            Feedback(
                feedback_type=FeedbackType.CORRECTION,
                query="q",
                response="resp",
                correction="",
            )
        )
        learner.rebuild_from_history()
        assert learner.preferences.total_signals == 0
        assert learner.preferences.length.samples == 0

    def test_rebuild_resets_existing(self, learner, store):
        learner.preferences.total_signals = 99
        learner.rebuild_from_history()
        assert learner.preferences.total_signals == 0


class TestLoad:
    def test_load_malformed_json_returns_defaults(self, store, tmp_path):
        path = tmp_path / "prefs.json"
        path.write_text("{ this is not valid json ]")
        learner = PreferenceLearner(store=store, preferences_path=path)
        assert learner.preferences.total_signals == 0
        assert learner.preferences.length.samples == 0

    def test_load_missing_file_returns_defaults(self, store, tmp_path):
        learner = PreferenceLearner(store=store, preferences_path=tmp_path / "does_not_exist.json")
        assert isinstance(learner.preferences, UserPreferences)
        assert learner.preferences.total_signals == 0

    def test_load_roundtrips_saved_profile(self, store, tmp_path):
        path = tmp_path / "prefs.json"
        l1 = PreferenceLearner(store=store, preferences_path=path)
        l1.preferences.formatting.samples = 7
        l1.preferences.formatting.prefers_code_blocks = 0.8
        l1.preferences.tone.samples = 4
        l1.preferences.tone.formality = 0.3
        l1.preferences.length.samples = 6
        l1.preferences.length.tendency = "concise"
        l1.preferences.total_signals = 12
        l1._save()

        l2 = PreferenceLearner(store=store, preferences_path=path)
        assert l2.preferences.formatting.samples == 7
        assert l2.preferences.formatting.prefers_code_blocks == pytest.approx(0.8)
        assert l2.preferences.tone.samples == 4
        assert l2.preferences.tone.formality == pytest.approx(0.3)
        assert l2.preferences.length.samples == 6
        assert l2.preferences.length.tendency == "concise"
        assert l2.preferences.total_signals == 12

    def test_get_profile_shape(self, learner):
        profile = learner.get_profile()
        assert set(profile) >= {"length", "formatting", "tone", "total_signals"}
        assert "tendency" in profile["length"]
