"""Tests for the training scheduler decision logic."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from llmstack.learn.dataset import (
    DPOExample,
    GeneratedDataset,
    TrainingExample,
)
from llmstack.learn.scheduler import (
    SchedulerConfig,
    SchedulerState,
    TrainScheduler,
    TriggerReason,
)


def make_dataset(sft=2, dpo=1, feedback_ids=None):
    """Build a GeneratedDataset with the requested number of examples."""
    ds = GeneratedDataset()
    ds.sft_examples = [
        TrainingExample(messages=[{"role": "user", "content": f"q{i}"}])
        for i in range(sft)
    ]
    ds.dpo_examples = [
        DPOExample(prompt=f"p{i}", chosen=f"c{i}", rejected=f"r{i}")
        for i in range(dpo)
    ]
    ds.feedback_ids = feedback_ids if feedback_ids is not None else ["f1", "f2"]
    return ds


@pytest.fixture
def store():
    s = MagicMock()
    s.get_unused_feedback_count.return_value = 0
    return s


@pytest.fixture
def dataset_gen():
    g = MagicMock()
    g.generate.return_value = make_dataset()
    return g


@pytest.fixture
def version_mgr():
    m = MagicMock()
    m.get_active.return_value = None
    version = MagicMock()
    version.version = "1.0.0"
    version.is_active = True
    m.create_version.return_value = version
    return m


@pytest.fixture
def scheduler(store, dataset_gen, version_mgr):
    return TrainScheduler(store, dataset_gen, version_mgr)


# ---------------------------------------------------------------------------
# Construction / properties
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_default_config_created(self, store, dataset_gen, version_mgr):
        sched = TrainScheduler(store, dataset_gen, version_mgr)
        assert isinstance(sched.config, SchedulerConfig)
        assert isinstance(sched.state, SchedulerState)

    def test_custom_config_used(self, store, dataset_gen, version_mgr):
        cfg = SchedulerConfig(min_feedback_threshold=3)
        sched = TrainScheduler(store, dataset_gen, version_mgr, config=cfg)
        assert sched.config.min_feedback_threshold == 3

    def test_state_property(self, scheduler):
        assert scheduler.state is scheduler._state

    def test_is_training_property(self, scheduler):
        assert scheduler.is_training is False
        scheduler._state.is_training = True
        assert scheduler.is_training is True

    def test_has_train_callback_property(self, scheduler):
        assert scheduler.has_train_callback is False
        scheduler.set_train_callback(lambda ds: {"success": True})
        assert scheduler.has_train_callback is True

    def test_set_train_callback(self, scheduler):
        cb = MagicMock()
        scheduler.set_train_callback(cb)
        assert scheduler._train_callback is cb


# ---------------------------------------------------------------------------
# check()
# ---------------------------------------------------------------------------


class TestCheck:
    def test_check_updates_state(self, scheduler, store):
        store.get_unused_feedback_count.return_value = 7
        scheduler.check()
        assert scheduler._state.last_check_time > 0
        assert scheduler._state.pending_feedback == 7

    def test_check_returns_none_when_training(self, scheduler, store):
        store.get_unused_feedback_count.return_value = 100
        scheduler._state.is_training = True
        assert scheduler.check() is None

    def test_check_returns_none_when_no_feedback(self, scheduler, store):
        store.get_unused_feedback_count.return_value = 0
        assert scheduler.check() is None

    def test_check_returns_none_when_interval_not_elapsed(self, scheduler, store):
        store.get_unused_feedback_count.return_value = 100
        # last_train_time is "now-ish" so interval has not elapsed
        import time

        scheduler._state.last_train_time = time.time()
        assert scheduler.check() is None

    def test_check_threshold_trigger(self, scheduler, store):
        store.get_unused_feedback_count.return_value = 50
        # last_train_time = 0 -> interval definitely elapsed
        reason = scheduler.check()
        assert reason is TriggerReason.THRESHOLD
        assert scheduler._state.next_trigger is TriggerReason.THRESHOLD

    def test_check_scheduled_trigger_on_max_wait(self, scheduler, store):
        import time

        cfg = SchedulerConfig(
            min_feedback_threshold=1000,  # never hit threshold
            min_interval_seconds=10,
            max_wait_seconds=100,
        )
        scheduler.config = cfg
        store.get_unused_feedback_count.return_value = 5
        # last train was long ago but > 0 so SCHEDULED branch is reachable
        scheduler._state.last_train_time = time.time() - 1000
        reason = scheduler.check()
        assert reason is TriggerReason.SCHEDULED
        assert scheduler._state.next_trigger is TriggerReason.SCHEDULED

    def test_check_regression_trigger(self, scheduler, store, version_mgr):
        cfg = SchedulerConfig(
            min_feedback_threshold=1000,  # never hit threshold
            min_interval_seconds=0,
            max_wait_seconds=10**9,  # never hit scheduled
        )
        scheduler.config = cfg
        store.get_unused_feedback_count.return_value = 5
        # active version present + declining trend => regression
        active = MagicMock()
        active.version = "1.0.0"
        version_mgr.get_active.return_value = active
        store.get_quality_trend.return_value = [
            {"value": 0.5},
            {"value": 0.5},
            {"value": 0.5},
            {"value": 0.9},  # baseline (last) much higher than recent
        ]
        reason = scheduler.check()
        assert reason is TriggerReason.REGRESSION
        assert scheduler._state.next_trigger is TriggerReason.REGRESSION

    def test_check_returns_none_when_no_trigger(self, scheduler, store, version_mgr):
        cfg = SchedulerConfig(
            min_feedback_threshold=1000,
            min_interval_seconds=0,
            max_wait_seconds=10**9,
        )
        scheduler.config = cfg
        store.get_unused_feedback_count.return_value = 5
        version_mgr.get_active.return_value = None  # no regression
        assert scheduler.check() is None


# ---------------------------------------------------------------------------
# trigger()
# ---------------------------------------------------------------------------


class TestTrigger:
    def test_trigger_error_when_already_training(self, scheduler):
        scheduler._state.is_training = True
        result = scheduler.trigger()
        assert result == {"error": "Training already in progress"}

    def test_trigger_error_without_callback(self, scheduler):
        result = scheduler.trigger()
        assert result == {"error": "No training callback configured"}
        # state must not be stuck in training
        assert scheduler._state.is_training is False

    def test_trigger_error_when_no_data_generated(self, scheduler, dataset_gen):
        scheduler.set_train_callback(lambda ds: {"success": True})
        dataset_gen.generate.return_value = make_dataset(sft=0, dpo=0, feedback_ids=[])
        result = scheduler.trigger()
        assert result == {"error": "No training data generated from feedback"}
        assert scheduler._state.is_training is False

    def test_trigger_error_when_callback_returns_none(self, scheduler, tmp_path):
        scheduler.config.output_dir = str(tmp_path)
        scheduler.set_train_callback(lambda ds: None)
        result = scheduler.trigger()
        assert result["error"] == "No result"
        assert "dataset_size" in result
        assert scheduler._state.is_training is False

    def test_trigger_error_when_callback_unsuccessful(self, scheduler, tmp_path):
        scheduler.config.output_dir = str(tmp_path)
        scheduler.set_train_callback(
            lambda ds: {"success": False, "error": "boom"}
        )
        result = scheduler.trigger()
        assert result["error"] == "boom"
        assert result["dataset_size"] > 0
        assert scheduler._state.is_training is False

    def test_trigger_unsuccessful_default_error(self, scheduler, tmp_path):
        scheduler.config.output_dir = str(tmp_path)
        scheduler.set_train_callback(lambda ds: {"success": False})
        result = scheduler.trigger()
        assert result["error"] == "Unknown training error"

    def test_trigger_success_full_flow(
        self, scheduler, store, version_mgr, tmp_path
    ):
        scheduler.config.output_dir = str(tmp_path)
        version = MagicMock()
        version.version = "2.0.0"
        version.is_active = True
        version_mgr.create_version.return_value = version

        ds = make_dataset(sft=3, dpo=2, feedback_ids=["a", "b", "c"])
        scheduler.dataset_gen.generate.return_value = ds

        scheduler.set_train_callback(
            lambda dataset: {
                "success": True,
                "quality_score": 0.8,
                "adapter_path": "/tmp/adapter",
                "train_run_id": 42,
                "final_loss": 0.1,
                "best_loss": 0.05,
                "train_time_seconds": 12.0,
            }
        )

        result = scheduler.trigger(TriggerReason.THRESHOLD)

        assert result["success"] is True
        assert result["version"] == "2.0.0"
        assert result["dataset_size"] == 5
        assert result["quality_score"] == 0.8
        assert result["activated"] is True
        assert result["trigger_reason"] == "threshold"

        # Side effects
        store.mark_feedback_used.assert_called_once_with(["a", "b", "c"])
        store.add_train_run.assert_called_once()
        version_mgr.create_version.assert_called_once()

        # State cleaned up
        assert scheduler._state.is_training is False
        assert scheduler._state.pending_feedback == 0
        assert scheduler._state.last_train_time > 0

        # Dataset was saved to the configured output dir
        assert (tmp_path / "datasets").exists()

    def test_trigger_default_reason_is_manual(
        self, scheduler, version_mgr, tmp_path
    ):
        scheduler.config.output_dir = str(tmp_path)
        scheduler.set_train_callback(
            lambda ds: {"success": True, "quality_score": 0.5}
        )
        result = scheduler.trigger()
        assert result["trigger_reason"] == "manual"

    def test_trigger_handles_exception(self, scheduler, tmp_path):
        scheduler.config.output_dir = str(tmp_path)

        def boom(ds):
            raise RuntimeError("kaboom")

        scheduler.set_train_callback(boom)
        result = scheduler.trigger()
        assert result["error"] == "kaboom"
        assert scheduler._state.is_training is False

    def test_trigger_exception_during_generate(self, scheduler, dataset_gen):
        scheduler.set_train_callback(lambda ds: {"success": True})
        dataset_gen.generate.side_effect = ValueError("gen failed")
        result = scheduler.trigger()
        assert result["error"] == "gen failed"
        assert scheduler._state.is_training is False


# ---------------------------------------------------------------------------
# _should_trigger_regression()
# ---------------------------------------------------------------------------


class TestRegression:
    def test_no_active_version(self, scheduler, version_mgr):
        version_mgr.get_active.return_value = None
        assert scheduler._should_trigger_regression() is False

    def test_insufficient_trend_data(self, scheduler, store, version_mgr):
        active = MagicMock()
        active.version = "1.0.0"
        version_mgr.get_active.return_value = active
        store.get_quality_trend.return_value = [{"value": 0.5}]  # < 2
        assert scheduler._should_trigger_regression() is False

    def test_no_regression_when_stable(self, scheduler, store, version_mgr):
        active = MagicMock()
        active.version = "1.0.0"
        version_mgr.get_active.return_value = active
        store.get_quality_trend.return_value = [
            {"value": 0.8},
            {"value": 0.8},
            {"value": 0.8},
            {"value": 0.8},
        ]
        assert scheduler._should_trigger_regression() is False

    def test_regression_detected(self, scheduler, store, version_mgr):
        active = MagicMock()
        active.version = "1.0.0"
        version_mgr.get_active.return_value = active
        store.get_quality_trend.return_value = [
            {"value": 0.4},
            {"value": 0.4},
            {"value": 0.4},
            {"value": 0.9},  # baseline higher -> drop below threshold
        ]
        assert scheduler._should_trigger_regression() is True


# ---------------------------------------------------------------------------
# _should_activate()
# ---------------------------------------------------------------------------


class TestShouldActivate:
    def test_auto_activate_disabled(self, scheduler):
        scheduler.config.auto_activate = False
        assert scheduler._should_activate(0.99) is False

    def test_activate_when_no_active_version(self, scheduler, version_mgr):
        version_mgr.get_active.return_value = None
        assert scheduler._should_activate(0.1) is True

    def test_activate_when_improvement_sufficient(self, scheduler, version_mgr):
        active = MagicMock()
        active.quality_score = 0.5
        version_mgr.get_active.return_value = active
        scheduler.config.min_quality_improvement = 0.01
        assert scheduler._should_activate(0.6) is True

    def test_no_activate_when_improvement_insufficient(
        self, scheduler, version_mgr
    ):
        active = MagicMock()
        active.quality_score = 0.5
        version_mgr.get_active.return_value = active
        scheduler.config.min_quality_improvement = 0.1
        assert scheduler._should_activate(0.55) is False
