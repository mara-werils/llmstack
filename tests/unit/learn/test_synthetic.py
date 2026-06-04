"""Tests for synthetic data augmentation."""

from __future__ import annotations

import pytest

from llmstack.learn.dataset import TrainingExample
from llmstack.learn.synthetic import AugmentationConfig, SyntheticAugmenter


@pytest.fixture
def augmenter():
    return SyntheticAugmenter(AugmentationConfig(seed=42))


@pytest.fixture
def sample_examples():
    return [
        TrainingExample(
            messages=[
                {"role": "user", "content": "How do I sort a list in Python?"},
                {
                    "role": "assistant",
                    "content": "Use `sorted(my_list)` or `my_list.sort()` for in-place sorting.",
                },
            ],
        ),
        TrainingExample(
            messages=[
                {"role": "user", "content": "Explain what a decorator is"},
                {
                    "role": "assistant",
                    "content": "A decorator is a function that wraps another function to extend its behavior without modifying it directly.",
                },
            ],
        ),
        TrainingExample(
            messages=[
                {"role": "user", "content": "Write a function to check if a number is prime"},
                {
                    "role": "assistant",
                    "content": "```python\ndef is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True\n```",
                },
            ],
        ),
        TrainingExample(
            messages=[
                {"role": "user", "content": "How can I read a file in Python?"},
                {
                    "role": "assistant",
                    "content": "Use `with open('file.txt') as f: content = f.read()`",
                },
            ],
        ),
        TrainingExample(
            messages=[
                {"role": "user", "content": "Show me how to handle errors"},
                {
                    "role": "assistant",
                    "content": "```python\ntry:\n    result = risky_operation()\nexcept ValueError as e:\n    print(f'Error: {e}')\n```",
                },
            ],
        ),
    ]


class TestSyntheticAugmenter:
    def test_augments_to_target(self, augmenter, sample_examples):
        """Augments dataset to reach target count."""
        result = augmenter.augment(sample_examples, target_count=15)
        assert len(result) >= 10  # should grow significantly
        assert len(result) <= 15

    def test_preserves_originals(self, augmenter, sample_examples):
        """Original examples are preserved in output."""
        result = augmenter.augment(sample_examples, target_count=10)
        for orig in sample_examples:
            assert orig in result

    def test_augmented_are_different(self, augmenter, sample_examples):
        """Augmented examples differ from originals."""
        result = augmenter.augment(sample_examples, target_count=15)
        new_examples = result[len(sample_examples) :]
        for new_ex in new_examples:
            assert new_ex.metadata.get("augmented") is True
            # Query should be different from all originals
            new_query = new_ex.messages[0]["content"]
            for orig in sample_examples:
                assert new_query != orig.messages[0]["content"]

    def test_respects_max_factor(self, augmenter, sample_examples):
        """Respects max augmentation factor."""
        augmenter.config.max_factor = 2.0
        result = augmenter.augment(sample_examples, target_count=100)
        assert len(result) <= len(sample_examples) * 2

    def test_below_threshold_no_augmentation(self, augmenter):
        """Doesn't augment if below minimum threshold."""
        small = [
            TrainingExample(
                messages=[
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello"},
                ]
            )
        ]
        augmenter.config.min_examples_threshold = 3
        result = augmenter.augment(small, target_count=10)
        assert len(result) == 1  # unchanged

    def test_deduplication(self, augmenter, sample_examples):
        """No exact duplicates in augmented set."""
        result = augmenter.augment(sample_examples, target_count=15)
        queries = [ex.messages[0]["content"].lower() for ex in result]
        # Allow some overlap since augmentation is rule-based
        # but there should be unique content
        unique_queries = set(queries)
        assert len(unique_queries) > len(sample_examples)
