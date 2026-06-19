"""Tests for feedback collection and storage."""

from __future__ import annotations

import time

import pytest

from llmstack.learn.feedback import Feedback, FeedbackType
from llmstack.learn.store import FeedbackStore


@pytest.fixture
def tmp_db(tmp_path):
    """Create a temporary database."""
    return tmp_path / "test_learning.db"


@pytest.fixture
def store(tmp_db):
    """Create a test feedback store."""
    s = FeedbackStore(db_path=tmp_db)
    yield s
    s.close()


class TestFeedback:
    def test_create_feedback(self):
        fb = Feedback(
            feedback_type=FeedbackType.THUMBS_UP,
            query="What is Python?",
            response="Python is a programming language.",
            model="llama3.2",
        )
        assert fb.feedback_type == FeedbackType.THUMBS_UP
        assert fb.is_positive
        assert not fb.is_negative
        assert not fb.has_correction

    def test_correction_feedback(self):
        fb = Feedback(
            feedback_type=FeedbackType.CORRECTION,
            query="Write a hello world",
            response="print('hello')",
            correction="print('Hello, World!')",
        )
        assert fb.has_correction
        assert not fb.is_positive
        assert not fb.is_negative

    def test_to_dict_roundtrip(self):
        fb = Feedback(
            feedback_type=FeedbackType.EDIT,
            query="test query",
            response="test response",
            correction="better response",
            tags=["code", "python"],
        )
        data = fb.to_dict()
        restored = Feedback.from_dict(data)
        assert restored.feedback_type == fb.feedback_type
        assert restored.query == fb.query
        assert restored.correction == fb.correction
        assert restored.tags == fb.tags

    def test_negative_types(self):
        for ft in (FeedbackType.THUMBS_DOWN, FeedbackType.REGENERATE, FeedbackType.ABANDON):
            fb = Feedback(feedback_type=ft)
            assert fb.is_negative
            assert not fb.is_positive

    def test_implicit_and_explicit(self):
        for ft in (FeedbackType.COPY, FeedbackType.REGENERATE, FeedbackType.ABANDON):
            fb = Feedback(feedback_type=ft)
            assert fb.is_implicit is True
            assert fb.is_explicit is False

        for ft in (FeedbackType.THUMBS_UP, FeedbackType.CORRECTION, FeedbackType.PREFERENCE):
            fb = Feedback(feedback_type=ft)
            assert fb.is_implicit is False
            assert fb.is_explicit is True


class TestFeedbackStore:
    def test_add_and_retrieve(self, store):
        fb = Feedback(
            feedback_type=FeedbackType.THUMBS_UP,
            query="test",
            response="response",
            model="test-model",
        )
        store.add_feedback(fb)
        results = store.get_feedback(limit=10)
        assert len(results) == 1
        assert results[0].query == "test"
        assert results[0].feedback_type == FeedbackType.THUMBS_UP

    def test_filter_by_type(self, store):
        store.add_feedback(Feedback(feedback_type=FeedbackType.THUMBS_UP, query="a", response="b"))
        store.add_feedback(
            Feedback(feedback_type=FeedbackType.THUMBS_DOWN, query="c", response="d")
        )
        store.add_feedback(
            Feedback(feedback_type=FeedbackType.CORRECTION, query="e", response="f", correction="g")
        )

        ups = store.get_feedback(feedback_type=FeedbackType.THUMBS_UP)
        assert len(ups) == 1
        assert ups[0].query == "a"

        corrections = store.get_feedback(feedback_type=FeedbackType.CORRECTION)
        assert len(corrections) == 1

    def test_unused_feedback_count(self, store):
        for i in range(5):
            store.add_feedback(
                Feedback(
                    feedback_type=FeedbackType.THUMBS_UP,
                    query=f"q{i}",
                    response=f"r{i}",
                )
            )

        assert store.get_unused_feedback_count() == 5

        # Mark some as used
        feedback = store.get_feedback(limit=3)
        store.mark_feedback_used([fb.id for fb in feedback[:2]])
        assert store.get_unused_feedback_count() == 3

    def test_filter_by_model(self, store):
        store.add_feedback(
            Feedback(feedback_type=FeedbackType.THUMBS_UP, query="a", response="b", model="llama")
        )
        store.add_feedback(
            Feedback(feedback_type=FeedbackType.THUMBS_UP, query="c", response="d", model="gpt4")
        )

        results = store.get_feedback(model="llama")
        assert len(results) == 1
        assert results[0].query == "a"

    def test_stats(self, store):
        store.add_feedback(Feedback(feedback_type=FeedbackType.THUMBS_UP, query="a", response="b"))
        store.add_feedback(
            Feedback(feedback_type=FeedbackType.THUMBS_DOWN, query="c", response="d")
        )
        store.add_feedback(
            Feedback(feedback_type=FeedbackType.CORRECTION, query="e", response="f", correction="g")
        )

        stats = store.get_stats()
        assert stats["total_feedback"] == 3
        assert stats["unused_feedback"] == 3
        assert stats["feedback_by_type"]["thumbs_up"] == 1
        assert stats["feedback_by_type"]["thumbs_down"] == 1

    def test_model_versions(self, store):
        store.add_model_version(
            version="1",
            base_model="llama3.2",
            quality_score=0.7,
            is_active=True,
        )
        store.add_model_version(
            version="2",
            base_model="llama3.2",
            quality_score=0.8,
            is_active=True,
        )

        active = store.get_active_version()
        assert active is not None
        assert active["version"] == "2"
        assert active["quality_score"] == 0.8

    def test_quality_snapshots(self, store):
        store.add_quality_snapshot("1", "overall", 0.75, sample_size=10)
        store.add_quality_snapshot("1", "overall", 0.80, sample_size=15)

        trend = store.get_quality_trend("1", "overall")
        assert len(trend) == 2
        assert trend[0]["value"] == 0.80  # most recent first

    def test_is_connected_property(self, store):
        # __init__ eagerly connects via _ensure_schema, so this is already True.
        assert store.is_connected is True

    def test_db_size_bytes(self, store):
        assert store.db_size_bytes > 0

    def test_db_size_bytes_missing_file(self, tmp_path):
        s = FeedbackStore(db_path=tmp_path / "never_created.db")
        # No write has happened yet, so the file may not exist on disk.
        assert s.db_size_bytes >= 0
        s.close()

    def test_mark_feedback_used_empty_list_is_noop(self, store):
        store.add_feedback(Feedback(feedback_type=FeedbackType.THUMBS_UP, query="a", response="b"))
        store.mark_feedback_used([])  # should not raise
        assert store.get_unused_feedback_count() == 1

    def test_get_feedback_since_filter(self, store):
        store.add_feedback(Feedback(feedback_type=FeedbackType.THUMBS_UP, query="a", response="b"))
        results = store.get_feedback(since=0)
        assert len(results) == 1
        results = store.get_feedback(since=time.time() + 1000)
        assert len(results) == 0

    def test_add_train_run(self, store):
        run_id = store.add_train_run(
            model_version="2",
            base_model="llama3.2",
            feedback_count=10,
            dataset_size=8,
            final_loss=0.5,
        )
        assert run_id > 0
