"""Tests for learning rate scheduler."""

from __future__ import annotations

import pytest

from llmstack.learn.lr_scheduler import (
    LearningRateScheduler,
    LRSchedulerConfig,
    ScheduleType,
)


@pytest.fixture
def default_scheduler():
    return LearningRateScheduler()


class TestLearningRateScheduler:
    def test_constant_schedule(self):
        config = LRSchedulerConfig(schedule=ScheduleType.CONSTANT, initial_lr=1e-3)
        sched = LearningRateScheduler(config)
        assert sched.get_lr(0) == 1e-3
        assert sched.get_lr(500) == 1e-3
        assert sched.get_lr(999) == 1e-3

    def test_linear_warmup(self):
        config = LRSchedulerConfig(
            schedule=ScheduleType.LINEAR_WARMUP,
            min_lr=0.0,
            max_lr=1e-3,
            warmup_steps=100,
            total_steps=1000,
        )
        sched = LearningRateScheduler(config)
        assert sched.get_lr(0) == 0.0
        assert abs(sched.get_lr(50) - 5e-4) < 1e-6
        assert abs(sched.get_lr(100) - 1e-3) < 1e-6

    def test_cosine_decay(self):
        config = LRSchedulerConfig(
            schedule=ScheduleType.COSINE,
            min_lr=0.0,
            max_lr=1e-3,
            total_steps=1000,
        )
        sched = LearningRateScheduler(config)
        # Start high, end low
        start = sched.get_lr(0)
        end = sched.get_lr(999)
        assert start > end

    def test_cosine_with_warmup(self):
        config = LRSchedulerConfig(
            schedule=ScheduleType.COSINE_WITH_WARMUP,
            min_lr=0.0,
            max_lr=1e-3,
            warmup_steps=100,
            total_steps=1000,
        )
        sched = LearningRateScheduler(config)
        # Warmup phase
        assert sched.get_lr(0) == 0.0
        # Peak at warmup end
        peak = sched.get_lr(100)
        assert abs(peak - 1e-3) < 1e-6
        # Decay after warmup
        late = sched.get_lr(900)
        assert late < peak

    def test_step_decay(self):
        config = LRSchedulerConfig(
            schedule=ScheduleType.STEP_DECAY,
            initial_lr=1e-3,
            min_lr=1e-6,
            decay_factor=0.5,
            decay_every=100,
        )
        sched = LearningRateScheduler(config)
        assert sched.get_lr(0) == 1e-3
        assert abs(sched.get_lr(100) - 5e-4) < 1e-6
        assert abs(sched.get_lr(200) - 2.5e-4) < 1e-6

    def test_linear_decay(self):
        config = LRSchedulerConfig(
            schedule=ScheduleType.LINEAR_DECAY,
            min_lr=0.0,
            max_lr=1e-3,
            warmup_steps=0,
            total_steps=1000,
        )
        sched = LearningRateScheduler(config)
        start = sched.get_lr(0)
        end = sched.get_lr(999)
        assert start > end

    def test_warmup_steps_fraction(self):
        config = LRSchedulerConfig(warmup_steps=0.1, total_steps=1000)
        sched = LearningRateScheduler(config)
        assert sched.warmup_steps_abs == 100

    def test_warmup_steps_absolute(self):
        config = LRSchedulerConfig(warmup_steps=50.0, total_steps=1000)
        sched = LearningRateScheduler(config)
        assert sched.warmup_steps_abs == 50

    def test_get_schedule(self):
        sched = LearningRateScheduler()
        schedule = sched.get_schedule(num_points=10)
        assert len(schedule) > 0
        assert all("step" in p and "lr" in p for p in schedule)

    def test_get_summary(self, default_scheduler):
        summary = default_scheduler.get_summary()
        assert "schedule" in summary
        assert "initial_lr" in summary
        assert "warmup_steps" in summary
        assert "start_lr" in summary
        assert "peak_lr" in summary

    def test_min_lr_floor(self):
        config = LRSchedulerConfig(
            schedule=ScheduleType.STEP_DECAY,
            initial_lr=1e-3,
            min_lr=1e-4,
            decay_factor=0.1,
            decay_every=10,
            total_steps=1000,
        )
        sched = LearningRateScheduler(config)
        # After many decays, should not go below min_lr
        assert sched.get_lr(999) >= 1e-4

    def test_has_warmup_property(self):
        warmup_sched = LearningRateScheduler(LRSchedulerConfig(schedule=ScheduleType.LINEAR_WARMUP))
        no_warmup_sched = LearningRateScheduler(LRSchedulerConfig(schedule=ScheduleType.CONSTANT))
        assert warmup_sched.has_warmup is True
        assert no_warmup_sched.has_warmup is False

    def test_final_lr_property(self):
        config = LRSchedulerConfig(schedule=ScheduleType.CONSTANT, initial_lr=1e-3, total_steps=100)
        sched = LearningRateScheduler(config)
        assert sched.final_lr == sched.get_lr(99)

    def test_unknown_schedule_falls_back_to_initial_lr(self):
        config = LRSchedulerConfig(schedule=ScheduleType.CONSTANT, initial_lr=7e-4)
        sched = LearningRateScheduler(config)
        sched.config.schedule = "not-a-real-schedule"
        assert sched.get_lr(0) == 7e-4

    def test_cosine_with_zero_total_steps(self):
        config = LRSchedulerConfig(schedule=ScheduleType.COSINE, total_steps=0, initial_lr=5e-4)
        sched = LearningRateScheduler(config)
        assert sched.get_lr(0) == 5e-4

    def test_cosine_with_warmup_no_remaining_steps(self):
        config = LRSchedulerConfig(
            schedule=ScheduleType.COSINE_WITH_WARMUP,
            warmup_steps=100,
            total_steps=100,
            max_lr=9e-4,
        )
        sched = LearningRateScheduler(config)
        assert sched.get_lr(100) == 9e-4

    def test_linear_decay_zero_total_steps(self):
        config = LRSchedulerConfig(
            schedule=ScheduleType.LINEAR_DECAY, total_steps=0, initial_lr=3e-4
        )
        sched = LearningRateScheduler(config)
        assert sched.get_lr(0) == 3e-4

    def test_linear_decay_during_warmup(self):
        config = LRSchedulerConfig(
            schedule=ScheduleType.LINEAR_DECAY,
            warmup_steps=100,
            total_steps=1000,
            min_lr=0.0,
            max_lr=1e-3,
        )
        sched = LearningRateScheduler(config)
        assert abs(sched.get_lr(50) - 5e-4) < 1e-6

    def test_linear_decay_floors_at_min_lr_past_total(self):
        config = LRSchedulerConfig(
            schedule=ScheduleType.LINEAR_DECAY,
            warmup_steps=0,
            total_steps=1000,
            min_lr=1e-5,
            max_lr=1e-3,
        )
        sched = LearningRateScheduler(config)
        # At and beyond the final step the LR must settle at min_lr, never below.
        assert sched.get_lr(1000) == pytest.approx(1e-5)
        assert sched.get_lr(5000) == pytest.approx(1e-5)
        assert sched.get_lr(5000) >= 0.0

    def test_linear_decay_no_remaining_steps(self):
        config = LRSchedulerConfig(
            schedule=ScheduleType.LINEAR_DECAY,
            warmup_steps=100,
            total_steps=100,
            max_lr=8e-4,
        )
        sched = LearningRateScheduler(config)
        assert sched.get_lr(100) == 8e-4
