"""Curriculum learning — progressive difficulty training strategy.

Organizes training examples by difficulty, starting with simple patterns
and gradually introducing harder ones. This mimics how humans learn and
produces more stable training convergence than random ordering.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from llmstack.learn.feedback import Feedback

logger = logging.getLogger(__name__)


class DifficultyLevel(str, Enum):
    """Training example difficulty levels."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXPERT = "expert"


@dataclass
class CurriculumStage:
    """A stage in the curriculum with difficulty bounds."""

    level: DifficultyLevel
    min_score: float
    max_score: float
    examples: list[Feedback] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.examples)


@dataclass
class CurriculumConfig:
    """Configuration for curriculum learning."""

    # Number of stages
    num_stages: int = 4

    # Difficulty thresholds (auto-computed if empty)
    thresholds: list[float] = field(default_factory=list)

    # Minimum examples per stage before advancing
    min_examples_per_stage: int = 5

    # Pacing: how many stages to unlock per epoch
    stages_per_epoch: int = 1

    # Whether to re-include easier examples in later stages
    include_prior_stages: bool = True

    # Weight decay for prior stage examples (1.0 = full weight)
    prior_stage_weight: float = 0.5


class CurriculumScheduler:
    """Schedules training examples in curriculum order.

    Scores each example by difficulty and organizes them into progressive
    stages. During training, starts with easy examples and gradually
    introduces harder ones for more stable convergence.
    """

    def __init__(self, config: CurriculumConfig | None = None):
        self.config = config or CurriculumConfig()
        self._stages: list[CurriculumStage] = []

    @property
    def stage_count(self) -> int:
        """Return the number of curriculum stages."""
        return len(self._stages)

    @property
    def total_examples(self) -> int:
        """Return the total number of examples across all stages."""
        return sum(stage.count for stage in self._stages)

    def score_difficulty(self, feedback: Feedback) -> float:
        """Score the difficulty of a training example (0.0 = easy, 1.0 = hard).

        Uses multiple heuristics:
        - Query length and complexity
        - Response length
        - Whether a correction was needed
        - Code vs natural language
        - Number of concepts involved
        """
        scores: list[float] = []

        # Query complexity (longer queries tend to be harder)
        query_len = len(feedback.query)
        if query_len < 30:
            scores.append(0.1)
        elif query_len < 100:
            scores.append(0.3)
        elif query_len < 300:
            scores.append(0.6)
        else:
            scores.append(0.9)

        # Response length (longer responses indicate harder tasks)
        resp_len = len(feedback.response)
        if resp_len < 100:
            scores.append(0.1)
        elif resp_len < 500:
            scores.append(0.3)
        elif resp_len < 1500:
            scores.append(0.6)
        else:
            scores.append(0.8)

        # Correction complexity
        if feedback.correction:
            corr_len = len(feedback.correction)
            ratio = corr_len / max(resp_len, 1)
            # Large corrections relative to response = harder
            scores.append(min(1.0, ratio))
        else:
            scores.append(0.2)

        # Code content (code tasks are generally harder)
        code_indicators = ["```", "def ", "class ", "import ", "function ", "const "]
        has_code = any(ind in feedback.response for ind in code_indicators)
        scores.append(0.7 if has_code else 0.3)

        # Multi-concept queries
        concept_keywords = [
            "and",
            "also",
            "additionally",
            "plus",
            "with",
            "including",
            "both",
            "multiple",
            "several",
        ]
        concept_count = sum(1 for kw in concept_keywords if kw in feedback.query.lower())
        scores.append(min(1.0, concept_count * 0.2))

        return sum(scores) / len(scores) if scores else 0.5

    def organize(self, examples: list[Feedback]) -> list[CurriculumStage]:
        """Organize examples into curriculum stages by difficulty."""
        if not examples:
            return []

        # Score all examples
        scored = [(ex, self.score_difficulty(ex)) for ex in examples]
        scored.sort(key=lambda x: x[1])

        # Determine thresholds
        if self.config.thresholds and len(self.config.thresholds) >= self.config.num_stages - 1:
            thresholds = self.config.thresholds[: self.config.num_stages - 1]
        else:
            # Auto-compute evenly spaced thresholds
            thresholds = [
                (i + 1) / self.config.num_stages for i in range(self.config.num_stages - 1)
            ]

        # Create stages
        levels = list(DifficultyLevel)
        stages: list[CurriculumStage] = []
        bounds = [0.0] + thresholds + [1.01]

        for i in range(min(self.config.num_stages, len(levels))):
            stage = CurriculumStage(
                level=levels[i],
                min_score=bounds[i],
                max_score=bounds[i + 1],
            )
            stage.examples = [ex for ex, score in scored if bounds[i] <= score < bounds[i + 1]]
            stages.append(stage)

        self._stages = stages
        return stages

    def get_training_order(
        self,
        examples: list[Feedback],
        current_epoch: int = 0,
    ) -> list[Feedback]:
        """Get examples in curriculum order for the given epoch.

        Returns examples from unlocked stages, with optional inclusion
        of prior stage examples at reduced weight.
        """
        stages = self.organize(examples)
        if not stages:
            return []

        # Determine how many stages are unlocked
        unlocked = min(
            len(stages),
            1 + current_epoch * self.config.stages_per_epoch,
        )

        ordered: list[Feedback] = []
        for i in range(unlocked):
            stage = stages[i]
            if i < unlocked - 1 and self.config.include_prior_stages:
                # Prior stages: subsample based on weight
                keep = max(1, int(len(stage.examples) * self.config.prior_stage_weight))
                ordered.extend(stage.examples[:keep])
            else:
                ordered.extend(stage.examples)

        return ordered

    def get_stage_summary(self, examples: list[Feedback]) -> dict[str, Any]:
        """Get a summary of how examples distribute across stages."""
        stages = self.organize(examples)
        return {
            "total_examples": len(examples),
            "num_stages": len(stages),
            "stages": [
                {
                    "level": s.level.value,
                    "count": s.count,
                    "score_range": [round(s.min_score, 2), round(s.max_score, 2)],
                }
                for s in stages
            ],
        }
