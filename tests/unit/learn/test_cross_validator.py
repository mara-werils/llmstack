"""Tests for cross-validation evaluator."""

from __future__ import annotations

import pytest

from llmstack.learn.cross_validator import CrossValidator, CrossValidationResult
from llmstack.learn.feedback import Feedback, FeedbackType


def _fb(ftype: FeedbackType = FeedbackType.THUMBS_UP) -> Feedback:
    return Feedback(
        feedback_type=ftype,
        query="test query",
        response="test response",
    )


@pytest.fixture
def cv():
    return CrossValidator(k=3)


class TestCrossValidator:
    def test_create_folds(self, cv):
        data = [_fb() for _ in range(9)]
        folds = cv.create_folds(data)
        assert len(folds) == 3
        assert all(len(f) == 3 for f in folds)

    def test_create_folds_uneven(self, cv):
        data = [_fb() for _ in range(10)]
        folds = cv.create_folds(data)
        assert len(folds) == 3
        total = sum(len(f) for f in folds)
        assert total == 10

    def test_create_folds_empty(self, cv):
        assert cv.create_folds([]) == []

    def test_evaluate_returns_result(self, cv):
        data = [_fb() for _ in range(15)]
        result = cv.evaluate(data)
        assert isinstance(result, CrossValidationResult)
        assert result.num_folds == 3
        assert len(result.fold_results) == 3

    def test_evaluate_empty_data(self, cv):
        result = cv.evaluate([])
        assert result.num_folds == 0

    def test_fold_sizes(self, cv):
        data = [_fb() for _ in range(12)]
        result = cv.evaluate(data)
        for fold in result.fold_results:
            assert fold.train_size == 8
            assert fold.test_size == 4

    def test_mean_accuracy_range(self, cv):
        data = [_fb(FeedbackType.THUMBS_UP) for _ in range(15)]
        result = cv.evaluate(data)
        assert 0.0 <= result.mean_accuracy <= 1.0

    def test_std_accuracy(self, cv):
        data = [_fb(FeedbackType.THUMBS_UP) for _ in range(15)]
        result = cv.evaluate(data)
        assert result.std_accuracy >= 0.0

    def test_custom_quality_fn(self, cv):
        data = [_fb() for _ in range(9)]
        result = cv.evaluate(
            data,
            quality_fn=lambda train, test: (0.8, 0.9),
        )
        assert abs(result.mean_accuracy - 0.8) < 0.01
        assert abs(result.mean_quality - 0.9) < 0.01

    def test_serialization(self, cv):
        data = [_fb() for _ in range(9)]
        result = cv.evaluate(data)
        d = result.to_dict()
        assert "num_folds" in d
        assert "mean_accuracy" in d
        assert "folds" in d
        assert len(d["folds"]) == 3

    def test_min_k(self):
        cv = CrossValidator(k=1)  # Should be bumped to 2
        assert cv.k == 2
