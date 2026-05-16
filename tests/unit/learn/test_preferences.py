"""Tests for user preference learning."""

from __future__ import annotations

import pytest

from llmstack.learn.feedback import Feedback, FeedbackType
from llmstack.learn.preferences import PreferenceLearner
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


class TestPreferenceLearner:
    def test_learn_conciseness(self, learner):
        """User consistently prefers shorter responses."""
        for _ in range(10):
            fb = Feedback(
                feedback_type=FeedbackType.CORRECTION,
                query="How do I print in Python?",
                response="To print in Python, you use the print() function. "
                         "This function takes arguments and outputs them to stdout. "
                         "It was introduced in Python 3 as a replacement for the print statement.",
                correction="Use `print('hello')`",
            )
            learner.learn_from_feedback(fb)

        assert learner.preferences.length.tendency == "concise"

    def test_learn_detail_preference(self, learner):
        """User consistently prefers longer responses."""
        for _ in range(10):
            fb = Feedback(
                feedback_type=FeedbackType.CORRECTION,
                query="What is async?",
                response="Async is asynchronous.",
                correction="Async (asynchronous) programming allows you to write concurrent code "
                           "that doesn't block. In Python, you use `async def` to define coroutines "
                           "and `await` to pause execution until a result is available. This is useful "
                           "for I/O-bound operations like network requests and file operations.",
            )
            learner.learn_from_feedback(fb)

        assert learner.preferences.length.tendency == "detailed"

    def test_learn_code_block_preference(self, learner):
        """User adds code blocks to corrections."""
        for _ in range(8):
            fb = Feedback(
                feedback_type=FeedbackType.CORRECTION,
                query="Show me a function",
                response="def hello(): print('hi')",
                correction="```python\ndef hello():\n    print('hi')\n```",
            )
            learner.learn_from_feedback(fb)

        assert learner.preferences.formatting.prefers_code_blocks > 0.6

    def test_learn_directness(self, learner):
        """User removes hedging language."""
        for _ in range(8):
            fb = Feedback(
                feedback_type=FeedbackType.CORRECTION,
                query="Is this approach correct?",
                response="I think this might perhaps be a reasonable approach, "
                         "although it seems like there could be alternatives.",
                correction="Yes, this approach is correct for your use case.",
            )
            learner.learn_from_feedback(fb)

        assert learner.preferences.tone.directness > 0.6

    def test_system_prompt_generation(self, learner):
        """After learning, generates appropriate system prompt additions."""
        # Learn conciseness + directness
        for _ in range(10):
            fb = Feedback(
                feedback_type=FeedbackType.CORRECTION,
                query="question",
                response="I think perhaps maybe the answer might be this long explanation "
                         "that goes on and on with unnecessary detail and hedging.",
                correction="The answer is X.",
            )
            learner.learn_from_feedback(fb)

        additions = learner.get_system_prompt_additions()
        assert "concise" in additions.lower() or "direct" in additions.lower()

    def test_persistence(self, store, tmp_path):
        """Preferences persist across instances."""
        path = tmp_path / "prefs.json"
        learner1 = PreferenceLearner(store=store, preferences_path=path)

        for _ in range(10):
            fb = Feedback(
                feedback_type=FeedbackType.CORRECTION,
                query="test",
                response="long " * 100,
                correction="short",
            )
            learner1.learn_from_feedback(fb)

        # Create new instance
        learner2 = PreferenceLearner(store=store, preferences_path=path)
        assert learner2.preferences.length.tendency == "concise"
