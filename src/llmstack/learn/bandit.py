"""Multi-armed bandit for model selection optimization.

Uses Thompson Sampling to learn which model performs best for different
query types. Over time, routes queries to the model most likely to
produce a high-quality response, balancing exploration and exploitation.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ArmStats:
    """Statistics for a single arm (model)."""

    name: str
    successes: int = 0
    failures: int = 0
    total_reward: float = 0.0
    pulls: int = 0

    @property
    def mean_reward(self) -> float:
        return self.total_reward / self.pulls if self.pulls > 0 else 0.0

    @property
    def success_rate(self) -> float:
        total = self.successes + self.failures
        return self.successes / total if total > 0 else 0.5

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "successes": self.successes,
            "failures": self.failures,
            "total_reward": round(self.total_reward, 4),
            "pulls": self.pulls,
            "mean_reward": round(self.mean_reward, 4),
            "success_rate": round(self.success_rate, 4),
        }


@dataclass
class BanditConfig:
    """Configuration for the multi-armed bandit."""

    # Exploration parameter for UCB1
    exploration_weight: float = 2.0

    # Minimum pulls before making decisions
    min_pulls_per_arm: int = 5

    # Strategy: "thompson", "ucb1", "epsilon_greedy"
    strategy: str = "thompson"

    # Epsilon for epsilon-greedy
    epsilon: float = 0.1

    # Decay factor for older observations (1.0 = no decay)
    decay_factor: float = 0.99


class ModelBandit:
    """Multi-armed bandit for selecting the best model per query type.

    Each model is an "arm". Feedback signals (thumbs up/down, corrections)
    provide reward signals. Over time, the bandit learns to route queries
    to the best-performing model while still exploring alternatives.
    """

    def __init__(
        self,
        models: list[str],
        config: BanditConfig | None = None,
    ):
        self.config = config or BanditConfig()
        self.arms: dict[str, ArmStats] = {model: ArmStats(name=model) for model in models}
        self._category_arms: dict[str, dict[str, ArmStats]] = {}

    @property
    def best_arm(self) -> str | None:
        """Return the model name with the highest mean reward, or None if no pulls."""
        if not self.arms:
            return None
        best = max(self.arms.values(), key=lambda a: a.mean_reward)
        return best.name if best.pulls > 0 else None

    @property
    def total_pulls(self) -> int:
        """Return total pulls across all arms."""
        return sum(a.pulls for a in self.arms.values())

    def select(self, category: str = "general") -> str:
        """Select a model for the given query category.

        Returns the model name to use for the next request.
        """
        arms = self._get_arms(category)

        # Ensure minimum exploration
        for name, arm in arms.items():
            if arm.pulls < self.config.min_pulls_per_arm:
                return name

        if self.config.strategy == "thompson":
            return self._thompson_select(arms)
        elif self.config.strategy == "ucb1":
            return self._ucb1_select(arms)
        elif self.config.strategy == "epsilon_greedy":
            return self._epsilon_greedy_select(arms)
        else:
            return self._thompson_select(arms)

    def update(
        self,
        model: str,
        reward: float,
        category: str = "general",
    ) -> None:
        """Update the arm statistics after observing a reward.

        Args:
            model: The model that was used.
            reward: Reward signal (0.0-1.0, where 1.0 is best).
            category: Query category for per-category tracking.
        """
        # Update global arm
        if model in self.arms:
            arm = self.arms[model]
            arm.pulls += 1
            arm.total_reward += reward
            if reward >= 0.5:
                arm.successes += 1
            else:
                arm.failures += 1

        # Update category-specific arm
        arms = self._get_arms(category)
        if model in arms:
            arm = arms[model]
            arm.pulls += 1
            arm.total_reward += reward
            if reward >= 0.5:
                arm.successes += 1
            else:
                arm.failures += 1

    def get_stats(self, category: str | None = None) -> dict[str, Any]:
        """Get current bandit statistics."""
        if category and category in self._category_arms:
            arms = self._category_arms[category]
        else:
            arms = self.arms

        total_pulls = sum(a.pulls for a in arms.values())
        best = max(arms.values(), key=lambda a: a.mean_reward) if arms else None

        return {
            "total_pulls": total_pulls,
            "strategy": self.config.strategy,
            "best_model": best.name if best else None,
            "arms": {name: arm.to_dict() for name, arm in arms.items()},
            "categories": list(self._category_arms.keys()),
        }

    def _get_arms(self, category: str) -> dict[str, ArmStats]:
        """Get or create per-category arms."""
        if category not in self._category_arms:
            self._category_arms[category] = {name: ArmStats(name=name) for name in self.arms}
        return self._category_arms[category]

    def _thompson_select(self, arms: dict[str, ArmStats]) -> str:
        """Thompson Sampling: sample from Beta distribution for each arm."""
        best_sample = -1.0
        best_arm = ""
        for name, arm in arms.items():
            # Beta(successes + 1, failures + 1)
            alpha = arm.successes + 1
            beta = arm.failures + 1
            sample = random.betavariate(alpha, beta)
            if sample > best_sample:
                best_sample = sample
                best_arm = name
        return best_arm

    def _ucb1_select(self, arms: dict[str, ArmStats]) -> str:
        """UCB1: select arm with highest upper confidence bound."""
        import math

        total = sum(a.pulls for a in arms.values())
        if total == 0:
            return next(iter(arms))

        best_ucb = -1.0
        best_arm = ""
        for name, arm in arms.items():
            if arm.pulls == 0:
                return name
            mean = arm.mean_reward
            exploration = self.config.exploration_weight * math.sqrt(math.log(total) / arm.pulls)
            ucb = mean + exploration
            if ucb > best_ucb:
                best_ucb = ucb
                best_arm = name
        return best_arm

    def _epsilon_greedy_select(self, arms: dict[str, ArmStats]) -> str:
        """Epsilon-greedy: exploit best arm with probability 1-epsilon."""
        if random.random() < self.config.epsilon:
            return random.choice(list(arms.keys()))
        # Exploit: pick best
        best = max(arms.values(), key=lambda a: a.mean_reward)
        return best.name
