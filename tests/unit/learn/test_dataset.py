"""Tests for dataset generation from feedback."""

from __future__ import annotations

import pytest

from llmstack.learn.dataset import DatasetGenerator, DatasetStrategy
from llmstack.learn.feedback import Feedback, FeedbackType
from llmstack.learn.store import FeedbackStore


@pytest.fixture
def store(tmp_path):
    s = FeedbackStore(db_path=tmp_path / "test.db")
    yield s
    s.close()


@pytest.fixture
def store_with_feedback(store):
    """Store with various feedback types."""
    # Corrections
    for i in range(10):
        store.add_feedback(
            Feedback(
                feedback_type=FeedbackType.CORRECTION,
                query=f"How do I do task {i}?",
                response=f"You can do it like this: bad_answer_{i}",
                correction=f"The correct way is: good_answer_{i} with more detail",
                model="llama3.2",
            )
        )

    # Thumbs up
    for i in range(5):
        store.add_feedback(
            Feedback(
                feedback_type=FeedbackType.THUMBS_UP,
                query=f"Explain concept {i}",
                response=f"This is a great explanation of concept {i} with details",
                model="llama3.2",
            )
        )

    # Thumbs down
    for i in range(3):
        store.add_feedback(
            Feedback(
                feedback_type=FeedbackType.THUMBS_DOWN,
                query=f"Bad question {i}",
                response=f"Bad response {i}",
                model="llama3.2",
            )
        )

    return store


class TestDatasetGenerator:
    def test_generate_sft(self, store_with_feedback):
        gen = DatasetGenerator(store=store_with_feedback)
        dataset = gen.generate(strategy=DatasetStrategy.SFT)

        assert dataset.total_examples > 0
        assert len(dataset.sft_examples) > 0

        # Check structure
        for ex in dataset.sft_examples:
            assert len(ex.messages) == 2
            assert ex.messages[0]["role"] == "user"
            assert ex.messages[1]["role"] == "assistant"

    def test_generate_dpo(self, store_with_feedback):
        gen = DatasetGenerator(store=store_with_feedback)
        dataset = gen.generate(strategy=DatasetStrategy.DPO)

        assert len(dataset.dpo_examples) > 0
        for ex in dataset.dpo_examples:
            assert ex.prompt
            assert ex.chosen
            assert ex.rejected
            assert ex.chosen != ex.rejected

    def test_generate_mixed(self, store_with_feedback):
        gen = DatasetGenerator(store=store_with_feedback)
        dataset = gen.generate(strategy=DatasetStrategy.MIXED)

        assert len(dataset.sft_examples) > 0
        assert len(dataset.dpo_examples) > 0

    def test_generate_positive_only(self, store_with_feedback):
        gen = DatasetGenerator(store=store_with_feedback)
        dataset = gen.generate(strategy=DatasetStrategy.POSITIVE)

        # Only thumbs_up responses should be included
        for ex in dataset.sft_examples:
            assert ex.metadata.get("source") == "positive"

    def test_empty_store(self, store):
        gen = DatasetGenerator(store=store)
        dataset = gen.generate()
        assert dataset.total_examples == 0

    def test_quality_filter(self, store):
        # Add feedback with too-short content
        store.add_feedback(
            Feedback(
                feedback_type=FeedbackType.CORRECTION,
                query="hi",  # too short
                response="ok",
                correction="yes",  # too short
            )
        )
        gen = DatasetGenerator(store=store, min_query_length=5, min_response_length=20)
        dataset = gen.generate(strategy=DatasetStrategy.SFT)
        assert dataset.total_examples == 0

    def test_deduplication(self, store):
        # Add duplicate feedback
        for _ in range(3):
            store.add_feedback(
                Feedback(
                    feedback_type=FeedbackType.CORRECTION,
                    query="What is the meaning of life?",
                    response="42 but wrong context here",
                    correction="The meaning of life varies by philosophy and person",
                )
            )

        gen = DatasetGenerator(store=store, dedup=True)
        dataset = gen.generate(strategy=DatasetStrategy.SFT)
        # After dedup, should only have 1
        assert len(dataset.sft_examples) == 1

    def test_save_dataset(self, store_with_feedback, tmp_path):
        gen = DatasetGenerator(store=store_with_feedback)
        dataset = gen.generate(strategy=DatasetStrategy.MIXED)

        path = dataset.save(tmp_path / "output")
        assert path.exists()
        assert path.stat().st_size > 0

        # Check metadata file exists
        meta_path = tmp_path / "output" / f"dataset_{dataset.id}_meta.json"
        assert meta_path.exists()
