"""Cross-validation evaluator for model quality assessment.

Splits feedback data into folds and evaluates model performance
across them to get a more reliable quality estimate than a single
train/test split.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from llmstack.learn.feedback import Feedback

logger = logging.getLogger(__name__)


@dataclass
class FoldResult:
    """Result from evaluating a single fold."""

    fold_index: int
    train_size: int
    test_size: int
    accuracy: float = 0.0
    avg_quality: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "fold": self.fold_index,
            "train_size": self.train_size,
            "test_size": self.test_size,
            "accuracy": round(self.accuracy, 4),
            "avg_quality": round(self.avg_quality, 4),
        }


@dataclass
class CrossValidationResult:
    """Aggregate result from cross-validation."""

    num_folds: int
    fold_results: list[FoldResult]
    mean_accuracy: float = 0.0
    std_accuracy: float = 0.0
    mean_quality: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_folds": self.num_folds,
            "mean_accuracy": round(self.mean_accuracy, 4),
            "std_accuracy": round(self.std_accuracy, 4),
            "mean_quality": round(self.mean_quality, 4),
            "folds": [f.to_dict() for f in self.fold_results],
        }


class CrossValidator:
    """Performs k-fold cross-validation on feedback data.

    Splits data into k folds and evaluates quality metrics across
    each fold to provide a robust quality estimate.
    """

    def __init__(self, k: int = 5):
        self.k = max(2, k)

    def create_folds(self, data: list[Feedback]) -> list[list[Feedback]]:
        """Split data into k roughly equal folds."""
        if not data:
            return []

        folds: list[list[Feedback]] = [[] for _ in range(self.k)]
        for i, item in enumerate(data):
            folds[i % self.k].append(item)
        return folds

    def evaluate(
        self,
        data: list[Feedback],
        quality_fn: Any | None = None,
    ) -> CrossValidationResult:
        """Run cross-validation and return results.

        Args:
            data: Feedback entries to evaluate.
            quality_fn: Optional function(train, test) -> (accuracy, quality).
                        If None, uses a heuristic based on feedback types.
        """
        folds = self.create_folds(data)
        if not folds:
            return CrossValidationResult(
                num_folds=0,
                fold_results=[],
            )

        fold_results: list[FoldResult] = []

        for i in range(len(folds)):
            test_fold = folds[i]
            train_folds = [f for j, f in enumerate(folds) if j != i]
            train_data = [item for fold in train_folds for item in fold]

            if quality_fn:
                accuracy, quality = quality_fn(train_data, test_fold)
            else:
                accuracy, quality = self._heuristic_evaluate(train_data, test_fold)

            fold_results.append(
                FoldResult(
                    fold_index=i,
                    train_size=len(train_data),
                    test_size=len(test_fold),
                    accuracy=accuracy,
                    avg_quality=quality,
                )
            )

        accuracies = [f.accuracy for f in fold_results]
        qualities = [f.avg_quality for f in fold_results]
        mean_acc = sum(accuracies) / len(accuracies) if accuracies else 0.0
        mean_qual = sum(qualities) / len(qualities) if qualities else 0.0

        # Standard deviation
        if len(accuracies) > 1:
            variance = sum((a - mean_acc) ** 2 for a in accuracies) / len(accuracies)
            std_acc = variance**0.5
        else:
            std_acc = 0.0

        return CrossValidationResult(
            num_folds=len(folds),
            fold_results=fold_results,
            mean_accuracy=mean_acc,
            std_accuracy=std_acc,
            mean_quality=mean_qual,
        )

    def _heuristic_evaluate(
        self,
        train: list[Feedback],
        test: list[Feedback],
    ) -> tuple[float, float]:
        """Heuristic evaluation based on feedback type distribution.

        Computes how well the training set distribution matches the test set.
        """
        if not train or not test:
            return 0.0, 0.0

        # Count positive feedback ratio in train vs test
        train_positive = sum(
            1 for fb in train if fb.feedback_type.value in ("thumbs_up", "accepted")
        )
        test_positive = sum(1 for fb in test if fb.feedback_type.value in ("thumbs_up", "accepted"))

        train_ratio = train_positive / len(train) if train else 0.0
        test_ratio = test_positive / len(test) if test else 0.0

        # Accuracy: how close train ratio predicts test ratio
        accuracy = 1.0 - abs(train_ratio - test_ratio)

        # Quality: overall positive rate
        quality = test_ratio

        return accuracy, quality
