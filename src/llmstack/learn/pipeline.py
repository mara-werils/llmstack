"""Learning pipeline orchestrator — ties all components into a unified system.

The Pipeline class is the single entry point for the entire adaptive learning
system. It initializes all components, wires them together, and provides
a clean API for the rest of llmstack to interact with.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from llmstack.learn.analytics import LearningAnalytics
from llmstack.learn.collector import FeedbackCollector
from llmstack.learn.config import LearnConfig
from llmstack.learn.dataset import DatasetGenerator
from llmstack.learn.evaluator import ModelEvaluator
from llmstack.learn.optimizer import PromptOptimizer
from llmstack.learn.patterns import PatternLearner
from llmstack.learn.preferences import PreferenceLearner
from llmstack.learn.regression import RegressionConfig, RegressionDetector
from llmstack.learn.scheduler import SchedulerConfig, TrainScheduler
from llmstack.learn.store import FeedbackStore
from llmstack.learn.versions import ModelVersionManager

logger = logging.getLogger(__name__)


class LearningPipeline:
    """Unified orchestrator for the adaptive learning system.

    Initializes and connects all learning components:
    - Feedback collection and storage
    - User preference learning
    - Code pattern recognition
    - Dataset generation
    - Training scheduling
    - Model versioning
    - Quality regression detection
    - Prompt optimization
    - Analytics and reporting

    Usage:
        pipeline = LearningPipeline()
        collector = pipeline.collector()

        # In chat/ask commands:
        collector.record_interaction(query, response, model)
        collector.thumbs_up()

        # Check if training should trigger:
        pipeline.check_training()

        # Get analytics:
        pipeline.get_status()
    """

    def __init__(self, config: LearnConfig | None = None):
        self.config = config or LearnConfig()
        self._store: FeedbackStore | None = None
        self._version_mgr: ModelVersionManager | None = None
        self._dataset_gen: DatasetGenerator | None = None
        self._scheduler: TrainScheduler | None = None
        self._regression: RegressionDetector | None = None
        self._pref_learner: PreferenceLearner | None = None
        self._pattern_learner: PatternLearner | None = None
        self._optimizer: PromptOptimizer | None = None
        self._evaluator: ModelEvaluator | None = None
        self._analytics: LearningAnalytics | None = None

    @property
    def store(self) -> FeedbackStore:
        if self._store is None:
            self._store = FeedbackStore(db_path=Path(self.config.storage.db_path))
        return self._store

    @property
    def version_mgr(self) -> ModelVersionManager:
        if self._version_mgr is None:
            self._version_mgr = ModelVersionManager(
                store=self.store,
                versions_dir=Path(self.config.storage.versions_dir),
            )
        return self._version_mgr

    @property
    def dataset_gen(self) -> DatasetGenerator:
        if self._dataset_gen is None:
            self._dataset_gen = DatasetGenerator(store=self.store)
        return self._dataset_gen

    @property
    def scheduler(self) -> TrainScheduler:
        if self._scheduler is None:
            sched_config = SchedulerConfig(
                min_feedback_threshold=self.config.training.min_feedback,
                min_interval_seconds=self.config.training.min_interval_hours * 3600,
                max_wait_seconds=self.config.training.max_wait_hours * 3600,
                strategy=self.config.training.dataset_strategy,
                base_model=self.config.training.base_model,
                max_examples=self.config.training.max_examples,
                auto_activate=self.config.training.auto_activate,
                min_quality_improvement=self.config.training.min_improvement,
            )
            self._scheduler = TrainScheduler(
                store=self.store,
                dataset_gen=self.dataset_gen,
                version_mgr=self.version_mgr,
                config=sched_config,
            )
        return self._scheduler

    @property
    def regression_detector(self) -> RegressionDetector:
        if self._regression is None:
            reg_config = RegressionConfig(
                min_samples=self.config.quality.min_samples,
                mild_threshold=self.config.quality.mild_threshold,
                moderate_threshold=self.config.quality.moderate_threshold,
                severe_threshold=self.config.quality.severe_threshold,
                auto_rollback=self.config.quality.auto_rollback,
                monitored_metrics=self.config.quality.metrics,
            )
            self._regression = RegressionDetector(
                store=self.store,
                version_mgr=self.version_mgr,
                config=reg_config,
            )
        return self._regression

    @property
    def preference_learner(self) -> PreferenceLearner:
        if self._pref_learner is None:
            self._pref_learner = PreferenceLearner(
                store=self.store,
                preferences_path=Path(self.config.storage.preferences_path),
            )
        return self._pref_learner

    @property
    def pattern_learner(self) -> PatternLearner:
        if self._pattern_learner is None:
            self._pattern_learner = PatternLearner(
                store=self.store,
                patterns_path=Path(self.config.storage.prompts_dir) / "code_patterns.json",
            )
        return self._pattern_learner

    @property
    def optimizer(self) -> PromptOptimizer:
        if self._optimizer is None:
            self._optimizer = PromptOptimizer(
                store=self.store,
                prompts_dir=Path(self.config.storage.prompts_dir),
            )
        return self._optimizer

    @property
    def evaluator(self) -> ModelEvaluator:
        if self._evaluator is None:
            self._evaluator = ModelEvaluator(store=self.store)
        return self._evaluator

    @property
    def analytics(self) -> LearningAnalytics:
        if self._analytics is None:
            self._analytics = LearningAnalytics(
                store=self.store,
                version_mgr=self.version_mgr,
            )
        return self._analytics

    def collector(self, command: str = "") -> FeedbackCollector:
        """Create a feedback collector for use in commands."""
        c = FeedbackCollector(store=self.store, config=self.config)
        c._current_command = command
        return c

    def check_training(self) -> dict[str, Any] | None:
        """Check if training should be triggered and run if needed.

        Returns training result dict if triggered, None otherwise.
        """
        if not self.config.enabled:
            return None

        trigger = self.scheduler.check()
        if trigger:
            return self.scheduler.trigger(trigger)
        return None

    def check_regression(self) -> list[dict[str, Any]]:
        """Run regression check and return any alerts."""
        if not self.config.enabled or not self.config.quality.enabled:
            return []

        alerts = self.regression_detector.check()
        return [a.to_dict() for a in alerts]

    def get_system_prompt_additions(self) -> str:
        """Get all learned context to inject into system prompts.

        Combines user preferences, code patterns, and prompt optimizations.
        """
        if not self.config.enabled:
            return ""

        parts: list[str] = []

        # User preferences
        if self.config.preferences.enabled and self.config.preferences.inject_into_prompts:
            pref_additions = self.preference_learner.get_system_prompt_additions()
            if pref_additions:
                parts.append(pref_additions)

        # Code style guide
        style_guide = self.pattern_learner.get_style_guide()
        if style_guide:
            parts.append(style_guide)

        return " ".join(parts)

    def get_status(self) -> dict[str, Any]:
        """Get complete pipeline status."""
        return self.analytics.get_summary()

    def close(self) -> None:
        """Clean up resources."""
        if self._store:
            self._store.close()
            self._store = None


# Singleton-ish factory
_pipeline_instance: LearningPipeline | None = None


def get_pipeline(config: LearnConfig | None = None) -> LearningPipeline:
    """Get or create the global learning pipeline instance."""
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = LearningPipeline(config=config)
    return _pipeline_instance
