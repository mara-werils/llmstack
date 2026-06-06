"""Feedback collector — convenient API for collecting signals in commands.

Provides a simple interface that CLI commands can use to collect
feedback without knowing the details of the learning pipeline.
Handles the UX of prompting users and parsing responses.
"""

from __future__ import annotations

import logging
from typing import Any

from llmstack.learn.config import LearnConfig
from llmstack.learn.feedback import Feedback, FeedbackType
from llmstack.learn.patterns import PatternLearner
from llmstack.learn.preferences import PreferenceLearner
from llmstack.learn.store import FeedbackStore

logger = logging.getLogger(__name__)


class FeedbackCollector:
    """Simplified interface for collecting feedback in interactive sessions.

    Usage:
        collector = FeedbackCollector()
        collector.record_interaction(query, response, model)

        # After user provides feedback:
        collector.thumbs_up()
        # or
        collector.correct("better response here")
    """

    def __init__(
        self,
        store: FeedbackStore | None = None,
        config: LearnConfig | None = None,
    ):
        self.config = config or LearnConfig()
        self._store = store
        self._pref_learner: PreferenceLearner | None = None
        self._pattern_learner: PatternLearner | None = None
        self._current_query = ""
        self._current_response = ""
        self._current_model = ""
        self._current_command = ""
        self._interaction_count = 0

    @property
    def store(self) -> FeedbackStore:
        if self._store is None:
            from pathlib import Path

            self._store = FeedbackStore(db_path=Path(self.config.storage.db_path))
        return self._store

    @property
    def preference_learner(self) -> PreferenceLearner:
        if self._pref_learner is None:
            from pathlib import Path

            self._pref_learner = PreferenceLearner(
                store=self.store,
                preferences_path=Path(self.config.storage.preferences_path),
            )
        return self._pref_learner

    @property
    def pattern_learner(self) -> PatternLearner:
        if self._pattern_learner is None:
            from pathlib import Path

            self._pattern_learner = PatternLearner(
                store=self.store,
                patterns_path=Path(self.config.storage.prompts_dir) / "code_patterns.json",
            )
        return self._pattern_learner

    def record_interaction(
        self,
        query: str,
        response: str,
        model: str = "",
        command: str = "",
    ) -> None:
        """Record the current interaction for potential feedback."""
        self._current_query = query
        self._current_response = response
        self._current_model = model
        self._current_command = command
        self._interaction_count += 1

    def thumbs_up(self) -> Feedback:
        """Record positive feedback for the current interaction."""
        return self._submit(FeedbackType.THUMBS_UP)

    def thumbs_down(self) -> Feedback:
        """Record negative feedback for the current interaction."""
        return self._submit(FeedbackType.THUMBS_DOWN)

    def correct(self, correction: str) -> Feedback:
        """Record a correction for the current interaction."""
        return self._submit(FeedbackType.CORRECTION, correction=correction)

    def edit(self, edited_response: str) -> Feedback:
        """Record an edit to the current response."""
        return self._submit(FeedbackType.EDIT, correction=edited_response)

    def prefer(self, preferred: str, rejected: str) -> Feedback:
        """Record a preference between two responses."""
        return self._submit(
            FeedbackType.PREFERENCE,
            correction=preferred,
            preferred_over=rejected,
        )

    def on_regenerate(self) -> Feedback:
        """Record that the user regenerated (implicit negative)."""
        return self._submit(FeedbackType.REGENERATE)

    def on_copy(self) -> Feedback:
        """Record that the user copied the response (implicit positive)."""
        return self._submit(FeedbackType.COPY)

    @property
    def interaction_count(self) -> int:
        """Return the number of interactions recorded in this session."""
        return self._interaction_count

    @property
    def has_pending_interaction(self) -> bool:
        """Return True when an interaction has been recorded but no feedback given."""
        return bool(self._current_query)

    def should_prompt(self) -> bool:
        """Check if we should prompt for feedback based on interaction count."""
        if not self.config.enabled or not self.config.feedback.interactive_feedback:
            return False
        interval = self.config.feedback.prompt_interval
        return self._interaction_count > 0 and self._interaction_count % interval == 0

    def parse_feedback_input(self, user_input: str) -> Feedback | None:
        """Parse user's feedback input and return appropriate feedback.

        Supported inputs:
        - 'y' or '+' → thumbs_up
        - 'n' or '-' → thumbs_down
        - 'e:...' or 'edit:...' → edit with correction
        - 'c:...' or 'correct:...' → correction
        - 's' or 'skip' or '' → None (skip)
        """
        text = user_input.strip().lower()

        if not text or text in ("s", "skip"):
            return None

        if text in ("y", "yes", "+", "good", "ok"):
            return self.thumbs_up()

        if text in ("n", "no", "-", "bad"):
            return self.thumbs_down()

        if text.startswith(("e:", "edit:")):
            correction = user_input.split(":", 1)[1].strip()
            if correction:
                return self.edit(correction)

        if text.startswith(("c:", "correct:")):
            correction = user_input.split(":", 1)[1].strip()
            if correction:
                return self.correct(correction)

        return None

    def get_stats(self) -> dict[str, Any]:
        """Get feedback collection stats for the current session."""
        return {
            "interactions": self._interaction_count,
            "current_query": self._current_query[:50] if self._current_query else "",
            "model": self._current_model,
            "total_stored": self.store.get_stats().get("total_feedback", 0),
            "pending": self.store.get_unused_feedback_count(),
        }

    def close(self) -> None:
        """Clean up resources."""
        if self._store:
            self._store.close()

    def _submit(
        self,
        feedback_type: FeedbackType,
        correction: str = "",
        preferred_over: str = "",
    ) -> Feedback:
        """Submit feedback to the store and update learners."""
        feedback = Feedback(
            feedback_type=feedback_type,
            query=self._current_query,
            response=self._current_response,
            model=self._current_model,
            correction=correction,
            preferred_over=preferred_over,
            command=self._current_command,
        )

        self.store.add_feedback(feedback)

        # Update learners
        if feedback.has_correction:
            self.preference_learner.learn_from_feedback(feedback)
            self.pattern_learner.learn_from_feedback(feedback)

        return feedback
