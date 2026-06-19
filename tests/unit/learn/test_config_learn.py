"""Tests for the learning pipeline configuration schema."""

from __future__ import annotations

from pathlib import Path

import pytest

from llmstack.learn.config import (
    FeedbackConfig,
    LearnConfig,
    PreferencesConfig,
    QualityMonitorConfig,
    StorageConfig,
    TrainingTriggerConfig,
)
from llmstack.learn.dataset import DatasetStrategy


class TestFeedbackConfig:
    def test_defaults(self):
        cfg = FeedbackConfig()
        assert cfg.implicit_signals is True
        assert cfg.prompt_interval == 5
        assert cfg.interactive_feedback is True
        assert cfg.min_response_length == 20

    def test_to_dict(self):
        cfg = FeedbackConfig(
            implicit_signals=False,
            prompt_interval=10,
            interactive_feedback=False,
            min_response_length=50,
        )
        assert cfg.to_dict() == {
            "implicit_signals": False,
            "prompt_interval": 10,
            "interactive_feedback": False,
            "min_response_length": 50,
        }

    def test_from_dict_full(self):
        cfg = FeedbackConfig.from_dict(
            {
                "implicit_signals": False,
                "prompt_interval": 3,
                "interactive_feedback": False,
                "min_response_length": 7,
            }
        )
        assert cfg.implicit_signals is False
        assert cfg.prompt_interval == 3
        assert cfg.interactive_feedback is False
        assert cfg.min_response_length == 7

    def test_from_dict_empty_uses_defaults(self):
        cfg = FeedbackConfig.from_dict({})
        assert cfg == FeedbackConfig()

    def test_round_trip(self):
        cfg = FeedbackConfig(prompt_interval=42)
        assert FeedbackConfig.from_dict(cfg.to_dict()) == cfg


class TestTrainingTriggerConfig:
    def test_defaults(self):
        cfg = TrainingTriggerConfig()
        assert cfg.min_feedback == 25
        assert cfg.min_interval_hours == 1.0
        assert cfg.max_wait_hours == 24.0
        assert cfg.strategy == "mixed"
        assert cfg.base_model == "unsloth/llama-3.2-1b-instruct-bnb-4bit"
        assert cfg.max_examples == 5000
        assert cfg.auto_activate is True
        assert cfg.min_improvement == 0.01

    def test_to_dict(self):
        cfg = TrainingTriggerConfig()
        d = cfg.to_dict()
        assert d == {
            "min_feedback": 25,
            "min_interval_hours": 1.0,
            "max_wait_hours": 24.0,
            "strategy": "mixed",
            "base_model": "unsloth/llama-3.2-1b-instruct-bnb-4bit",
            "max_examples": 5000,
            "auto_activate": True,
            "min_improvement": 0.01,
        }

    def test_from_dict_full(self):
        cfg = TrainingTriggerConfig.from_dict(
            {
                "min_feedback": 100,
                "min_interval_hours": 2.5,
                "max_wait_hours": 48.0,
                "strategy": "dpo",
                "base_model": "custom/model",
                "max_examples": 1000,
                "auto_activate": False,
                "min_improvement": 0.5,
            }
        )
        assert cfg.min_feedback == 100
        assert cfg.min_interval_hours == 2.5
        assert cfg.max_wait_hours == 48.0
        assert cfg.strategy == "dpo"
        assert cfg.base_model == "custom/model"
        assert cfg.max_examples == 1000
        assert cfg.auto_activate is False
        assert cfg.min_improvement == 0.5

    def test_from_dict_empty_uses_defaults(self):
        assert TrainingTriggerConfig.from_dict({}) == TrainingTriggerConfig()

    def test_round_trip(self):
        cfg = TrainingTriggerConfig(strategy="sft", max_examples=9)
        assert TrainingTriggerConfig.from_dict(cfg.to_dict()) == cfg

    def test_dataset_strategy_property(self):
        cfg = TrainingTriggerConfig(strategy="sft")
        assert cfg.dataset_strategy == DatasetStrategy.SFT
        assert isinstance(cfg.dataset_strategy, DatasetStrategy)

    @pytest.mark.parametrize(
        ("strategy", "expected"),
        [
            ("sft", DatasetStrategy.SFT),
            ("dpo", DatasetStrategy.DPO),
            ("positive", DatasetStrategy.POSITIVE),
            ("mixed", DatasetStrategy.MIXED),
        ],
    )
    def test_dataset_strategy_all_values(self, strategy, expected):
        assert TrainingTriggerConfig(strategy=strategy).dataset_strategy == expected

    def test_dataset_strategy_invalid_raises(self):
        with pytest.raises(ValueError):
            _ = TrainingTriggerConfig(strategy="bogus").dataset_strategy


class TestQualityMonitorConfig:
    def test_defaults(self):
        cfg = QualityMonitorConfig()
        assert cfg.enabled is True
        assert cfg.auto_rollback is True
        assert cfg.min_samples == 10
        assert cfg.mild_threshold == 0.03
        assert cfg.moderate_threshold == 0.08
        assert cfg.severe_threshold == 0.15
        assert cfg.metrics == ["overall", "coherence", "relevance"]

    def test_metrics_default_is_independent(self):
        a = QualityMonitorConfig()
        b = QualityMonitorConfig()
        a.metrics.append("extra")
        assert b.metrics == ["overall", "coherence", "relevance"]

    def test_to_dict(self):
        cfg = QualityMonitorConfig()
        d = cfg.to_dict()
        assert d == {
            "enabled": True,
            "auto_rollback": True,
            "min_samples": 10,
            "mild_threshold": 0.03,
            "moderate_threshold": 0.08,
            "severe_threshold": 0.15,
            "metrics": ["overall", "coherence", "relevance"],
        }

    def test_from_dict_full(self):
        cfg = QualityMonitorConfig.from_dict(
            {
                "enabled": False,
                "auto_rollback": False,
                "min_samples": 99,
                "mild_threshold": 0.1,
                "moderate_threshold": 0.2,
                "severe_threshold": 0.3,
                "metrics": ["accuracy"],
            }
        )
        assert cfg.enabled is False
        assert cfg.auto_rollback is False
        assert cfg.min_samples == 99
        assert cfg.mild_threshold == 0.1
        assert cfg.moderate_threshold == 0.2
        assert cfg.severe_threshold == 0.3
        assert cfg.metrics == ["accuracy"]

    def test_from_dict_empty_uses_defaults(self):
        assert QualityMonitorConfig.from_dict({}) == QualityMonitorConfig()

    def test_round_trip(self):
        cfg = QualityMonitorConfig(min_samples=7, metrics=["a", "b"])
        assert QualityMonitorConfig.from_dict(cfg.to_dict()) == cfg


class TestPreferencesConfig:
    def test_defaults(self):
        cfg = PreferencesConfig()
        assert cfg.enabled is True
        assert cfg.inject_into_prompts is True
        assert cfg.min_signals == 5

    def test_to_dict(self):
        cfg = PreferencesConfig(enabled=False, inject_into_prompts=False, min_signals=1)
        assert cfg.to_dict() == {
            "enabled": False,
            "inject_into_prompts": False,
            "min_signals": 1,
        }

    def test_from_dict_full(self):
        cfg = PreferencesConfig.from_dict(
            {"enabled": False, "inject_into_prompts": False, "min_signals": 12}
        )
        assert cfg.enabled is False
        assert cfg.inject_into_prompts is False
        assert cfg.min_signals == 12

    def test_from_dict_empty_uses_defaults(self):
        assert PreferencesConfig.from_dict({}) == PreferencesConfig()

    def test_round_trip(self):
        cfg = PreferencesConfig(min_signals=3)
        assert PreferencesConfig.from_dict(cfg.to_dict()) == cfg


class TestStorageConfig:
    def test_defaults_under_home(self):
        # Field defaults are bound at import time, so they reflect the real home.
        cfg = StorageConfig()
        base = str(Path.home() / ".llmstack")
        assert cfg.db_path == str(Path.home() / ".llmstack" / "learning.db")
        assert cfg.versions_dir == str(Path.home() / ".llmstack" / "model_versions")
        assert cfg.training_dir == str(Path.home() / ".llmstack" / "training")
        assert cfg.preferences_path == str(Path.home() / ".llmstack" / "preferences.json")
        assert cfg.prompts_dir == str(Path.home() / ".llmstack" / "prompts")
        assert all(p.startswith(base) for p in cfg.to_dict().values())

    def test_to_dict(self):
        cfg = StorageConfig(
            db_path="/a/db",
            versions_dir="/a/v",
            training_dir="/a/t",
            preferences_path="/a/p.json",
            prompts_dir="/a/pr",
        )
        assert cfg.to_dict() == {
            "db_path": "/a/db",
            "versions_dir": "/a/v",
            "training_dir": "/a/t",
            "preferences_path": "/a/p.json",
            "prompts_dir": "/a/pr",
        }

    def test_from_dict_full(self):
        cfg = StorageConfig.from_dict(
            {
                "db_path": "/x/db",
                "versions_dir": "/x/v",
                "training_dir": "/x/t",
                "preferences_path": "/x/p.json",
                "prompts_dir": "/x/pr",
            }
        )
        assert cfg.db_path == "/x/db"
        assert cfg.versions_dir == "/x/v"
        assert cfg.training_dir == "/x/t"
        assert cfg.preferences_path == "/x/p.json"
        assert cfg.prompts_dir == "/x/pr"

    def test_from_dict_empty_uses_home_defaults(self, monkeypatch, tmp_path):
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
        cfg = StorageConfig.from_dict({})
        assert cfg.db_path == str(tmp_path / ".llmstack" / "learning.db")
        assert cfg.versions_dir == str(tmp_path / ".llmstack" / "model_versions")
        assert cfg.training_dir == str(tmp_path / ".llmstack" / "training")
        assert cfg.preferences_path == str(tmp_path / ".llmstack" / "preferences.json")
        assert cfg.prompts_dir == str(tmp_path / ".llmstack" / "prompts")

    def test_from_dict_partial(self, monkeypatch, tmp_path):
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
        cfg = StorageConfig.from_dict({"db_path": "/custom/db"})
        assert cfg.db_path == "/custom/db"
        # The rest fall back to home-based defaults.
        assert cfg.prompts_dir == str(tmp_path / ".llmstack" / "prompts")

    def test_round_trip(self):
        cfg = StorageConfig(db_path="/r/db")
        assert StorageConfig.from_dict(cfg.to_dict()) == cfg


class TestLearnConfig:
    def test_defaults(self):
        cfg = LearnConfig()
        assert cfg.enabled is True
        assert isinstance(cfg.feedback, FeedbackConfig)
        assert isinstance(cfg.training, TrainingTriggerConfig)
        assert isinstance(cfg.quality, QualityMonitorConfig)
        assert isinstance(cfg.preferences, PreferencesConfig)
        assert isinstance(cfg.storage, StorageConfig)

    def test_nested_defaults_are_independent(self):
        a = LearnConfig()
        b = LearnConfig()
        assert a.feedback is not b.feedback
        assert a.storage is not b.storage

    def test_to_dict_structure(self):
        cfg = LearnConfig()
        d = cfg.to_dict()
        assert set(d.keys()) == {
            "enabled",
            "feedback",
            "training",
            "quality",
            "preferences",
            "storage",
        }
        assert d["enabled"] is True
        assert d["feedback"] == cfg.feedback.to_dict()
        assert d["training"] == cfg.training.to_dict()
        assert d["quality"] == cfg.quality.to_dict()
        assert d["preferences"] == cfg.preferences.to_dict()
        assert d["storage"] == cfg.storage.to_dict()

    def test_from_dict_empty_uses_defaults(self):
        cfg = LearnConfig.from_dict({})
        assert cfg.enabled is True
        assert cfg.feedback == FeedbackConfig()
        assert cfg.training == TrainingTriggerConfig()
        assert cfg.quality == QualityMonitorConfig()
        assert cfg.preferences == PreferencesConfig()

    def test_from_dict_disabled(self):
        cfg = LearnConfig.from_dict({"enabled": False})
        assert cfg.enabled is False

    def test_from_dict_full_nested(self):
        data = {
            "enabled": False,
            "feedback": {"prompt_interval": 99},
            "training": {"strategy": "dpo"},
            "quality": {"min_samples": 1},
            "preferences": {"min_signals": 2},
            "storage": {"db_path": "/nested/db"},
        }
        cfg = LearnConfig.from_dict(data)
        assert cfg.enabled is False
        assert cfg.feedback.prompt_interval == 99
        assert cfg.training.strategy == "dpo"
        assert cfg.quality.min_samples == 1
        assert cfg.preferences.min_signals == 2
        assert cfg.storage.db_path == "/nested/db"

    def test_round_trip(self, monkeypatch, tmp_path):
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
        original = LearnConfig.from_dict(
            {
                "enabled": False,
                "feedback": {"prompt_interval": 11},
                "training": {"strategy": "sft", "max_examples": 3},
                "quality": {"metrics": ["x"]},
                "preferences": {"enabled": False},
                "storage": {"db_path": "/rt/db"},
            }
        )
        rebuilt = LearnConfig.from_dict(original.to_dict())
        assert rebuilt.to_dict() == original.to_dict()
        assert rebuilt.enabled is False
        assert rebuilt.feedback.prompt_interval == 11
        assert rebuilt.training.dataset_strategy == DatasetStrategy.SFT

    def test_from_dict_ignores_partial_sections(self):
        # Only "feedback" provided; other sections remain default instances.
        cfg = LearnConfig.from_dict({"feedback": {"min_response_length": 1}})
        assert cfg.feedback.min_response_length == 1
        assert cfg.training == TrainingTriggerConfig()
        assert cfg.quality == QualityMonitorConfig()
