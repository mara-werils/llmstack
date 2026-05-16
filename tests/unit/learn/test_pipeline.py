"""Tests for the learning pipeline orchestrator."""

from __future__ import annotations

import pytest

from llmstack.learn.config import LearnConfig, StorageConfig
from llmstack.learn.feedback import FeedbackType
from llmstack.learn.pipeline import LearningPipeline


@pytest.fixture
def pipeline(tmp_path):
    """Create a pipeline with temp storage."""
    config = LearnConfig(
        storage=StorageConfig(
            db_path=str(tmp_path / "test.db"),
            versions_dir=str(tmp_path / "versions"),
            training_dir=str(tmp_path / "training"),
            preferences_path=str(tmp_path / "prefs.json"),
            prompts_dir=str(tmp_path / "prompts"),
        ),
    )
    p = LearningPipeline(config=config)
    yield p
    p.close()


class TestLearningPipeline:
    def test_initialization(self, pipeline):
        """Pipeline initializes lazily without errors."""
        assert pipeline.config.enabled
        assert pipeline.store is not None

    def test_collector_creation(self, pipeline):
        """Creates a working feedback collector."""
        collector = pipeline.collector(command="test")
        collector.record_interaction("hello", "hi there", model="test", command="test")
        fb = collector.thumbs_up()
        assert fb.feedback_type == FeedbackType.THUMBS_UP
        assert fb.command == "test"

    def test_feedback_flow(self, pipeline):
        """End-to-end feedback collection and query."""
        collector = pipeline.collector()

        # Record interactions with feedback
        for i in range(5):
            collector.record_interaction(
                f"How do I do task {i}?",
                f"Here is the answer for task {i}",
                model="llama3.2",
            )
            collector.thumbs_up()

        # Check stats
        stats = pipeline.store.get_stats()
        assert stats["total_feedback"] == 5
        assert stats["feedback_by_type"]["thumbs_up"] == 5

    def test_correction_updates_preferences(self, pipeline):
        """Corrections update the preference learner."""
        collector = pipeline.collector()

        for _ in range(8):
            collector.record_interaction(
                "How do I print?",
                "Well, I think you might perhaps want to try using "
                "the print function which possibly outputs text.",
                model="llama3.2",
            )
            collector.correct("Use `print('hello')`")

        # Check that preferences were learned
        profile = pipeline.preference_learner.get_profile()
        assert profile["length"]["tendency"] == "concise"

    def test_training_check_no_data(self, pipeline):
        """Training check returns None with no feedback."""
        result = pipeline.check_training()
        assert result is None

    def test_status_empty(self, pipeline):
        """Status works on empty pipeline."""
        status = pipeline.get_status()
        assert status["status"] == "inactive"
        assert "metrics" in status
        assert "recommendations" in status

    def test_system_prompt_additions(self, pipeline):
        """System prompt additions generated from preferences."""
        collector = pipeline.collector()

        # Train conciseness preference
        for _ in range(10):
            collector.record_interaction("question", "very " * 100, model="test")
            collector.correct("short answer")

        additions = pipeline.get_system_prompt_additions()
        # After enough signals, should contain style guidance
        assert isinstance(additions, str)

    def test_regression_check_no_versions(self, pipeline):
        """Regression check returns empty with no model versions."""
        alerts = pipeline.check_regression()
        assert alerts == []

    def test_full_lifecycle(self, pipeline):
        """Test the full feedback → dataset → version lifecycle."""
        collector = pipeline.collector()

        # Collect enough feedback
        for i in range(30):
            collector.record_interaction(
                f"Write a function to calculate {['sum', 'product', 'average'][i % 3]} of a list",
                f"def calc(lst): return bad_answer_{i}",
                model="llama3.2",
            )
            collector.correct(
                f"def calc(lst): return good_answer_{i} # with proper implementation"
            )

        # Verify dataset can be generated
        dataset = pipeline.dataset_gen.generate()
        assert dataset.total_examples > 0
        assert len(dataset.sft_examples) > 0

        # Verify evaluator can build eval set
        eval_set = pipeline.evaluator.build_eval_set()
        assert len(eval_set) > 0
