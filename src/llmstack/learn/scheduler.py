"""Training scheduler — decides when to trigger incremental fine-tuning.

Monitors feedback accumulation and triggers training when conditions are met:
- Minimum feedback threshold reached
- Time since last training exceeded
- Quality regression detected
- Manual trigger via CLI
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from llmstack.learn.dataset import DatasetGenerator, DatasetStrategy, GeneratedDataset
from llmstack.learn.store import FeedbackStore
from llmstack.learn.versions import ModelVersionManager

logger = logging.getLogger(__name__)


class TriggerReason(str, Enum):
    """Why a training run was triggered."""

    THRESHOLD = "threshold"  # Enough feedback accumulated
    SCHEDULED = "scheduled"  # Periodic schedule
    REGRESSION = "regression"  # Quality dropped
    MANUAL = "manual"  # User requested


@dataclass
class SchedulerConfig:
    """Configuration for the training scheduler."""

    # Minimum feedback items before triggering training
    min_feedback_threshold: int = 25

    # Minimum time between training runs (seconds)
    min_interval_seconds: float = 3600  # 1 hour

    # Maximum time before forced training if feedback exists (seconds)
    max_wait_seconds: float = 86400  # 24 hours

    # Quality regression threshold (drop from baseline triggers retrain)
    regression_threshold: float = -0.05

    # Training configuration
    strategy: DatasetStrategy = DatasetStrategy.MIXED
    base_model: str = "unsloth/llama-3.2-1b-instruct-bnb-4bit"
    output_dir: str = str(Path.home() / ".llmstack" / "training")
    max_examples: int = 5000

    # Auto-activate new version only if quality improves
    auto_activate: bool = True
    min_quality_improvement: float = 0.01


@dataclass
class SchedulerState:
    """Current state of the training scheduler."""

    last_train_time: float = 0.0
    last_check_time: float = 0.0
    pending_feedback: int = 0
    current_quality: float = 0.0
    is_training: bool = False
    next_trigger: TriggerReason | None = None


class TrainScheduler:
    """Decides when to trigger incremental fine-tuning.

    Monitors feedback accumulation and quality metrics,
    triggering training when conditions are met. Integrates
    with the dataset generator and version manager.
    """

    def __init__(
        self,
        store: FeedbackStore,
        dataset_gen: DatasetGenerator,
        version_mgr: ModelVersionManager,
        config: SchedulerConfig | None = None,
    ):
        self.store = store
        self.dataset_gen = dataset_gen
        self.version_mgr = version_mgr
        self.config = config or SchedulerConfig()
        self._state = SchedulerState()
        self._train_callback: Callable[[GeneratedDataset], Any] | None = None

    def set_train_callback(self, callback: Callable[[GeneratedDataset], Any]) -> None:
        """Set the actual training function to call when triggered.

        The callback receives a GeneratedDataset and should return
        a dict with keys: success, final_loss, best_loss, adapter_path, etc.
        """
        self._train_callback = callback

    @property
    def state(self) -> SchedulerState:
        return self._state

    def check(self) -> TriggerReason | None:
        """Check if training should be triggered.

        Returns the trigger reason if training should start, None otherwise.
        """
        self._state.last_check_time = time.time()
        self._state.pending_feedback = self.store.get_unused_feedback_count()

        if self._state.is_training:
            return None

        if self._state.pending_feedback == 0:
            return None

        now = time.time()
        time_since_train = now - self._state.last_train_time

        # Don't train too frequently
        if time_since_train < self.config.min_interval_seconds:
            return None

        # Check threshold trigger
        if self._state.pending_feedback >= self.config.min_feedback_threshold:
            self._state.next_trigger = TriggerReason.THRESHOLD
            return TriggerReason.THRESHOLD

        # Check time-based trigger (max wait exceeded)
        if (
            self._state.pending_feedback > 0
            and self._state.last_train_time > 0
            and time_since_train >= self.config.max_wait_seconds
        ):
            self._state.next_trigger = TriggerReason.SCHEDULED
            return TriggerReason.SCHEDULED

        # Check regression trigger
        if self._should_trigger_regression():
            self._state.next_trigger = TriggerReason.REGRESSION
            return TriggerReason.REGRESSION

        return None

    def trigger(self, reason: TriggerReason = TriggerReason.MANUAL) -> dict[str, Any]:
        """Trigger a training run.

        Returns a summary dict with training results.
        """
        if self._state.is_training:
            return {"error": "Training already in progress"}

        if not self._train_callback:
            return {"error": "No training callback configured"}

        self._state.is_training = True
        logger.info("Training triggered: %s", reason.value)

        try:
            # Generate dataset
            dataset = self.dataset_gen.generate(
                strategy=self.config.strategy,
                max_examples=self.config.max_examples,
            )

            if dataset.total_examples == 0:
                self._state.is_training = False
                return {"error": "No training data generated from feedback"}

            logger.info(
                "Generated dataset: %d SFT + %d DPO examples from %d feedback items",
                len(dataset.sft_examples),
                len(dataset.dpo_examples),
                len(dataset.feedback_ids),
            )

            # Save dataset
            output_dir = Path(self.config.output_dir)
            dataset.save(output_dir / "datasets")

            # Run training
            result = self._train_callback(dataset)

            if not result or not result.get("success"):
                self._state.is_training = False
                error = result.get("error", "Unknown training error") if result else "No result"
                return {"error": error, "dataset_size": dataset.total_examples}

            # Mark feedback as used
            self.store.mark_feedback_used(dataset.feedback_ids)

            # Create new version
            quality = result.get("quality_score", 0.0)
            version = self.version_mgr.create_version(
                base_model=self.config.base_model,
                adapter_path=result.get("adapter_path", ""),
                train_run_id=result.get("train_run_id", 0),
                quality_score=quality,
                activate=self._should_activate(quality),
                metadata={
                    "trigger_reason": reason.value,
                    "dataset_size": dataset.total_examples,
                    "final_loss": result.get("final_loss", 0.0),
                },
            )

            # Record training run
            self.store.add_train_run(
                model_version=version.version,
                base_model=self.config.base_model,
                feedback_count=len(dataset.feedback_ids),
                dataset_size=dataset.total_examples,
                final_loss=result.get("final_loss", 0.0),
                best_loss=result.get("best_loss", 0.0),
                train_time_seconds=result.get("train_time_seconds", 0.0),
                metadata={"trigger_reason": reason.value},
            )

            self._state.last_train_time = time.time()
            self._state.is_training = False
            self._state.pending_feedback = 0

            return {
                "success": True,
                "version": version.version,
                "dataset_size": dataset.total_examples,
                "final_loss": result.get("final_loss", 0.0),
                "quality_score": quality,
                "activated": version.is_active,
                "trigger_reason": reason.value,
            }

        except Exception as exc:
            self._state.is_training = False
            logger.error("Training run failed: %s", exc)
            return {"error": str(exc)}

    def _should_trigger_regression(self) -> bool:
        """Check if quality has regressed enough to warrant retraining."""
        active = self.version_mgr.get_active()
        if not active:
            return False

        trend = self.store.get_quality_trend(active.version, "overall", limit=5)
        if len(trend) < 2:
            return False

        recent_avg = sum(t["value"] for t in trend[:3]) / min(3, len(trend))
        baseline = trend[-1]["value"]
        drop = recent_avg - baseline

        return drop <= self.config.regression_threshold

    def _should_activate(self, new_quality: float) -> bool:
        """Decide if the new version should be activated."""
        if not self.config.auto_activate:
            return False

        active = self.version_mgr.get_active()
        if not active:
            return True

        improvement = new_quality - active.quality_score
        return improvement >= self.config.min_quality_improvement
