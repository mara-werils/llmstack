"""Integration hooks — connects learning pipeline to ask/chat/agent commands.

Provides middleware-style hooks that intercept LLM interactions,
inject learned preferences, collect feedback, and update quality metrics.
"""

from __future__ import annotations

import logging
from typing import Any

from llmstack.learn.config import LearnConfig
from llmstack.learn.feedback import Feedback, FeedbackType
from llmstack.learn.preferences import PreferenceLearner
from llmstack.learn.regression import RegressionDetector
from llmstack.learn.store import FeedbackStore

logger = logging.getLogger(__name__)


class LearningHooks:
    """Hooks that integrate the learning pipeline into LLM interactions.

    Attach these hooks to the ask/chat/agent command flow to:
    1. Inject learned preferences into system prompts
    2. Collect quality scores on every response
    3. Process explicit feedback (thumbs, corrections)
    4. Track implicit signals (regenerate, copy, abandon)
    """

    def __init__(
        self,
        store: FeedbackStore,
        preference_learner: PreferenceLearner,
        regression_detector: RegressionDetector | None = None,
        config: LearnConfig | None = None,
    ):
        self.store = store
        self.preference_learner = preference_learner
        self.regression_detector = regression_detector
        self.config = config or LearnConfig()
        self._interaction_count = 0
        self._current_query: str = ""
        self._current_response: str = ""
        self._current_model: str = ""

    def pre_generate(
        self,
        messages: list[dict[str, str]],
        model: str = "",
        **kwargs: Any,
    ) -> list[dict[str, str]]:
        """Hook called before LLM generation.

        Injects learned preferences into the system prompt.
        Returns modified messages.
        """
        if not self.config.enabled:
            return messages

        if not self.config.preferences.inject_into_prompts:
            return messages

        additions = self.preference_learner.get_system_prompt_additions()
        if not additions:
            return messages

        # Inject into system message
        messages = list(messages)  # don't modify original
        if messages and messages[0].get("role") == "system":
            messages[0] = {
                "role": "system",
                "content": messages[0]["content"] + "\n\n" + additions,
            }
        else:
            messages.insert(0, {"role": "system", "content": additions})

        self._current_model = model
        return messages

    def post_generate(
        self,
        query: str,
        response: str,
        model: str = "",
        quality_score: float = 0.0,
        **kwargs: Any,
    ) -> None:
        """Hook called after LLM generation.

        Records quality metrics and tracks the interaction.
        """
        if not self.config.enabled:
            return

        self._current_query = query
        self._current_response = response
        self._current_model = model or self._current_model
        self._interaction_count += 1

        # Record quality for regression detection
        if self.regression_detector and quality_score > 0:

            active = self.regression_detector.version_mgr.get_active()
            if active:
                self.regression_detector.record_quality(
                    model_version=active.version,
                    metric="overall",
                    value=quality_score,
                )

    def on_feedback(
        self,
        feedback_type: FeedbackType,
        correction: str = "",
        rating: int = 0,
        command: str = "",
        **kwargs: Any,
    ) -> None:
        """Hook called when user provides explicit feedback.

        Stores the feedback and updates preference learner.
        """
        if not self.config.enabled:
            return

        feedback = Feedback(
            feedback_type=feedback_type,
            query=self._current_query,
            response=self._current_response,
            model=self._current_model,
            correction=correction,
            rating=rating,
            command=command,
            context=kwargs,
        )

        self.store.add_feedback(feedback)

        # Update preferences from corrections
        if feedback.has_correction:
            self.preference_learner.learn_from_feedback(feedback)

        logger.debug(
            "Recorded feedback: type=%s, model=%s",
            feedback_type.value,
            self._current_model,
        )

    def on_copy(self) -> None:
        """Hook called when user copies a response (implicit positive signal)."""
        if not self.config.enabled or not self.config.feedback.implicit_signals:
            return

        self.on_feedback(FeedbackType.COPY)

    def on_regenerate(self) -> None:
        """Hook called when user regenerates (implicit negative signal)."""
        if not self.config.enabled or not self.config.feedback.implicit_signals:
            return

        self.on_feedback(FeedbackType.REGENERATE)

    def on_abandon(self) -> None:
        """Hook called when user abandons conversation (implicit negative)."""
        if not self.config.enabled or not self.config.feedback.implicit_signals:
            return

        if self._current_response:  # only if there was an unanswered response
            self.on_feedback(FeedbackType.ABANDON)

    def should_prompt_feedback(self) -> bool:
        """Check if we should prompt the user for feedback."""
        if not self.config.enabled or not self.config.feedback.interactive_feedback:
            return False
        interval = self.config.feedback.prompt_interval
        return self._interaction_count > 0 and self._interaction_count % interval == 0

    def get_feedback_prompt(self) -> str:
        """Get the feedback prompt text to show the user."""
        return (
            "\n[Learning] Was this response helpful? "
            "(y/n/e=edit/s=skip): "
        )


def create_hooks(config: LearnConfig | None = None) -> LearningHooks:
    """Factory to create learning hooks with default configuration."""
    from pathlib import Path

    cfg = config or LearnConfig()

    store = FeedbackStore(db_path=Path(cfg.storage.db_path))
    preference_learner = PreferenceLearner(
        store=store,
        preferences_path=Path(cfg.storage.preferences_path),
    )

    from llmstack.learn.versions import ModelVersionManager

    version_mgr = ModelVersionManager(
        store=store,
        versions_dir=Path(cfg.storage.versions_dir),
    )

    regression_detector = RegressionDetector(
        store=store,
        version_mgr=version_mgr,
    ) if cfg.quality.enabled else None

    return LearningHooks(
        store=store,
        preference_learner=preference_learner,
        regression_detector=regression_detector,
        config=cfg,
    )
