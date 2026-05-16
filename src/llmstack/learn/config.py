"""Learning pipeline configuration schema.

Extends llmstack.yaml with learning-specific configuration for
feedback collection, training triggers, and quality monitoring.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llmstack.learn.dataset import DatasetStrategy


@dataclass
class LearnConfig:
    """Complete learning pipeline configuration."""

    # Enable/disable the learning pipeline
    enabled: bool = True

    # Feedback collection
    feedback: FeedbackConfig = field(default_factory=lambda: FeedbackConfig())

    # Training triggers
    training: TrainingTriggerConfig = field(
        default_factory=lambda: TrainingTriggerConfig()
    )

    # Quality monitoring
    quality: QualityMonitorConfig = field(
        default_factory=lambda: QualityMonitorConfig()
    )

    # Preferences
    preferences: PreferencesConfig = field(
        default_factory=lambda: PreferencesConfig()
    )

    # Storage
    storage: StorageConfig = field(default_factory=lambda: StorageConfig())

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "feedback": self.feedback.to_dict(),
            "training": self.training.to_dict(),
            "quality": self.quality.to_dict(),
            "preferences": self.preferences.to_dict(),
            "storage": self.storage.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LearnConfig:
        config = cls()
        config.enabled = data.get("enabled", True)
        if "feedback" in data:
            config.feedback = FeedbackConfig.from_dict(data["feedback"])
        if "training" in data:
            config.training = TrainingTriggerConfig.from_dict(data["training"])
        if "quality" in data:
            config.quality = QualityMonitorConfig.from_dict(data["quality"])
        if "preferences" in data:
            config.preferences = PreferencesConfig.from_dict(data["preferences"])
        if "storage" in data:
            config.storage = StorageConfig.from_dict(data["storage"])
        return config


@dataclass
class FeedbackConfig:
    """Configuration for feedback collection."""

    # Collect implicit signals (copy, regenerate, abandon)
    implicit_signals: bool = True

    # Auto-prompt for feedback after N interactions
    prompt_interval: int = 5

    # Enable feedback in interactive modes (chat, ask -i)
    interactive_feedback: bool = True

    # Minimum response length to collect feedback on
    min_response_length: int = 20

    def to_dict(self) -> dict[str, Any]:
        return {
            "implicit_signals": self.implicit_signals,
            "prompt_interval": self.prompt_interval,
            "interactive_feedback": self.interactive_feedback,
            "min_response_length": self.min_response_length,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FeedbackConfig:
        return cls(
            implicit_signals=data.get("implicit_signals", True),
            prompt_interval=data.get("prompt_interval", 5),
            interactive_feedback=data.get("interactive_feedback", True),
            min_response_length=data.get("min_response_length", 20),
        )


@dataclass
class TrainingTriggerConfig:
    """Configuration for training triggers."""

    # Minimum feedback before training
    min_feedback: int = 25

    # Minimum hours between training runs
    min_interval_hours: float = 1.0

    # Maximum hours to wait if feedback exists
    max_wait_hours: float = 24.0

    # Dataset strategy
    strategy: str = "mixed"  # sft, dpo, positive, mixed

    # Base model for fine-tuning
    base_model: str = "unsloth/llama-3.2-1b-instruct-bnb-4bit"

    # Maximum training examples per run
    max_examples: int = 5000

    # Auto-activate only if quality improves
    auto_activate: bool = True

    # Minimum quality improvement to activate (0-1)
    min_improvement: float = 0.01

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_feedback": self.min_feedback,
            "min_interval_hours": self.min_interval_hours,
            "max_wait_hours": self.max_wait_hours,
            "strategy": self.strategy,
            "base_model": self.base_model,
            "max_examples": self.max_examples,
            "auto_activate": self.auto_activate,
            "min_improvement": self.min_improvement,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainingTriggerConfig:
        return cls(
            min_feedback=data.get("min_feedback", 25),
            min_interval_hours=data.get("min_interval_hours", 1.0),
            max_wait_hours=data.get("max_wait_hours", 24.0),
            strategy=data.get("strategy", "mixed"),
            base_model=data.get("base_model", "unsloth/llama-3.2-1b-instruct-bnb-4bit"),
            max_examples=data.get("max_examples", 5000),
            auto_activate=data.get("auto_activate", True),
            min_improvement=data.get("min_improvement", 0.01),
        )

    @property
    def dataset_strategy(self) -> DatasetStrategy:
        return DatasetStrategy(self.strategy)


@dataclass
class QualityMonitorConfig:
    """Configuration for quality monitoring."""

    # Enable quality tracking
    enabled: bool = True

    # Auto-rollback on severe regression
    auto_rollback: bool = True

    # Minimum samples for regression detection
    min_samples: int = 10

    # Regression thresholds
    mild_threshold: float = 0.03
    moderate_threshold: float = 0.08
    severe_threshold: float = 0.15

    # Metrics to monitor
    metrics: list[str] = field(
        default_factory=lambda: ["overall", "coherence", "relevance"]
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "auto_rollback": self.auto_rollback,
            "min_samples": self.min_samples,
            "mild_threshold": self.mild_threshold,
            "moderate_threshold": self.moderate_threshold,
            "severe_threshold": self.severe_threshold,
            "metrics": self.metrics,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QualityMonitorConfig:
        return cls(
            enabled=data.get("enabled", True),
            auto_rollback=data.get("auto_rollback", True),
            min_samples=data.get("min_samples", 10),
            mild_threshold=data.get("mild_threshold", 0.03),
            moderate_threshold=data.get("moderate_threshold", 0.08),
            severe_threshold=data.get("severe_threshold", 0.15),
            metrics=data.get("metrics", ["overall", "coherence", "relevance"]),
        )


@dataclass
class PreferencesConfig:
    """Configuration for preference learning."""

    # Enable preference learning
    enabled: bool = True

    # Inject learned preferences into system prompts
    inject_into_prompts: bool = True

    # Minimum signals before applying preferences
    min_signals: int = 5

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "inject_into_prompts": self.inject_into_prompts,
            "min_signals": self.min_signals,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PreferencesConfig:
        return cls(
            enabled=data.get("enabled", True),
            inject_into_prompts=data.get("inject_into_prompts", True),
            min_signals=data.get("min_signals", 5),
        )


@dataclass
class StorageConfig:
    """Configuration for learning data storage."""

    # Database path (default: ~/.llmstack/learning.db)
    db_path: str = str(Path.home() / ".llmstack" / "learning.db")

    # Model versions directory
    versions_dir: str = str(Path.home() / ".llmstack" / "model_versions")

    # Training output directory
    training_dir: str = str(Path.home() / ".llmstack" / "training")

    # Preferences file
    preferences_path: str = str(Path.home() / ".llmstack" / "preferences.json")

    # Prompts directory
    prompts_dir: str = str(Path.home() / ".llmstack" / "prompts")

    def to_dict(self) -> dict[str, Any]:
        return {
            "db_path": self.db_path,
            "versions_dir": self.versions_dir,
            "training_dir": self.training_dir,
            "preferences_path": self.preferences_path,
            "prompts_dir": self.prompts_dir,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StorageConfig:
        return cls(
            db_path=data.get("db_path", str(Path.home() / ".llmstack" / "learning.db")),
            versions_dir=data.get("versions_dir", str(Path.home() / ".llmstack" / "model_versions")),
            training_dir=data.get("training_dir", str(Path.home() / ".llmstack" / "training")),
            preferences_path=data.get("preferences_path", str(Path.home() / ".llmstack" / "preferences.json")),
            prompts_dir=data.get("prompts_dir", str(Path.home() / ".llmstack" / "prompts")),
        )
