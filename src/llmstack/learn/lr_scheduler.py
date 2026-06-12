"""Learning rate scheduler with warmup and decay.

Provides configurable learning rate schedules for fine-tuning,
including linear warmup, cosine decay, and step decay strategies.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class ScheduleType(str, Enum):
    """Available learning rate schedule types."""

    CONSTANT = "constant"
    LINEAR_WARMUP = "linear_warmup"
    COSINE = "cosine"
    COSINE_WITH_WARMUP = "cosine_with_warmup"
    STEP_DECAY = "step_decay"
    LINEAR_DECAY = "linear_decay"


@dataclass
class LRSchedulerConfig:
    """Configuration for the learning rate scheduler."""

    # Initial learning rate
    initial_lr: float = 2e-4

    # Minimum learning rate (floor)
    min_lr: float = 1e-6

    # Maximum learning rate (for warmup target)
    max_lr: float = 5e-4

    # Schedule type
    schedule: ScheduleType = ScheduleType.COSINE_WITH_WARMUP

    # Warmup steps (fraction of total if < 1.0, absolute if >= 1.0)
    warmup_steps: float = 0.1

    # Step decay factor (for step_decay schedule)
    decay_factor: float = 0.5

    # Step decay interval (for step_decay schedule)
    decay_every: int = 100

    # Total training steps (must be set before use)
    total_steps: int = 1000


class LearningRateScheduler:
    """Computes learning rates at each training step.

    Supports multiple schedule strategies commonly used in LLM fine-tuning.
    """

    def __init__(self, config: LRSchedulerConfig | None = None):
        self.config = config or LRSchedulerConfig()

    @property
    def has_warmup(self) -> bool:
        """Return True if the schedule includes a warmup phase."""
        return self.config.schedule in (
            ScheduleType.LINEAR_WARMUP,
            ScheduleType.COSINE_WITH_WARMUP,
        )

    @property
    def final_lr(self) -> float:
        """Return the learning rate at the last training step."""
        return self.get_lr(self.config.total_steps - 1)

    @property
    def warmup_steps_abs(self) -> int:
        """Get warmup steps as absolute count."""
        ws = self.config.warmup_steps
        if ws < 1.0:
            return int(ws * self.config.total_steps)
        return int(ws)

    def get_lr(self, step: int) -> float:
        """Get learning rate for the given step."""
        schedule = self.config.schedule

        if schedule == ScheduleType.CONSTANT:
            return self._constant(step)
        elif schedule == ScheduleType.LINEAR_WARMUP:
            return self._linear_warmup(step)
        elif schedule == ScheduleType.COSINE:
            return self._cosine(step)
        elif schedule == ScheduleType.COSINE_WITH_WARMUP:
            return self._cosine_with_warmup(step)
        elif schedule == ScheduleType.STEP_DECAY:
            return self._step_decay(step)
        elif schedule == ScheduleType.LINEAR_DECAY:
            return self._linear_decay(step)
        else:
            return self.config.initial_lr

    def get_schedule(self, num_points: int = 50) -> list[dict[str, float]]:
        """Get the full learning rate schedule for visualization."""
        total = self.config.total_steps
        step_size = max(1, total // num_points)
        points = []
        for step in range(0, total, step_size):
            points.append(
                {
                    "step": step,
                    "lr": round(self.get_lr(step), 8),
                    "progress": round(step / total, 4),
                }
            )
        return points

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of the schedule configuration."""
        return {
            "schedule": self.config.schedule.value,
            "initial_lr": self.config.initial_lr,
            "max_lr": self.config.max_lr,
            "min_lr": self.config.min_lr,
            "total_steps": self.config.total_steps,
            "warmup_steps": self.warmup_steps_abs,
            "start_lr": round(self.get_lr(0), 8),
            "peak_lr": round(self.get_lr(self.warmup_steps_abs), 8),
            "end_lr": round(self.get_lr(self.config.total_steps - 1), 8),
        }

    def _constant(self, step: int) -> float:
        return self.config.initial_lr

    def _linear_warmup(self, step: int) -> float:
        warmup = self.warmup_steps_abs
        if warmup == 0 or step >= warmup:
            return self.config.max_lr
        return self.config.min_lr + (self.config.max_lr - self.config.min_lr) * (step / warmup)

    def _cosine(self, step: int) -> float:
        total = self.config.total_steps
        if total <= 0:
            return self.config.initial_lr
        progress = min(step / total, 1.0)
        cosine_value = 0.5 * (1 + math.cos(math.pi * progress))
        return self.config.min_lr + (self.config.max_lr - self.config.min_lr) * cosine_value

    def _cosine_with_warmup(self, step: int) -> float:
        warmup = self.warmup_steps_abs
        if step < warmup:
            return self._linear_warmup(step)
        total = self.config.total_steps
        remaining = total - warmup
        if remaining <= 0:
            return self.config.max_lr
        progress = min((step - warmup) / remaining, 1.0)
        cosine_value = 0.5 * (1 + math.cos(math.pi * progress))
        return self.config.min_lr + (self.config.max_lr - self.config.min_lr) * cosine_value

    def _step_decay(self, step: int) -> float:
        num_decays = step // self.config.decay_every
        lr = self.config.initial_lr * (self.config.decay_factor**num_decays)
        return max(self.config.min_lr, lr)

    def _linear_decay(self, step: int) -> float:
        total = self.config.total_steps
        if total <= 0:
            return self.config.initial_lr
        warmup = self.warmup_steps_abs
        if step < warmup:
            return self._linear_warmup(step)
        remaining = total - warmup
        if remaining <= 0:
            return self.config.max_lr
        progress = (step - warmup) / remaining
        return self.config.max_lr - (self.config.max_lr - self.config.min_lr) * progress
