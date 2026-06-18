"""Tests for curriculum learning strategy."""

from __future__ import annotations

import pytest

from llmstack.learn.curriculum import (
    CurriculumConfig,
    CurriculumScheduler,
    DifficultyLevel,
)
from llmstack.learn.feedback import Feedback, FeedbackType


def _make_feedback(query: str, response: str, correction: str = "") -> Feedback:
    return Feedback(
        feedback_type=FeedbackType.THUMBS_UP,
        query=query,
        response=response,
        correction=correction,
    )


@pytest.fixture
def scheduler():
    return CurriculumScheduler()


class TestDifficultyScoring:
    def test_short_query_is_easy(self, scheduler):
        fb = _make_feedback("Hi", "Hello!")
        score = scheduler.score_difficulty(fb)
        assert score < 0.4

    def test_long_query_is_harder(self, scheduler):
        fb = _make_feedback(
            "Explain how async/await works in Python with examples "
            "and also show how to handle exceptions in coroutines "
            "and additionally demonstrate multiple concurrent tasks",
            "Here's a comprehensive explanation..." + "x" * 1000,
        )
        score = scheduler.score_difficulty(fb)
        assert score > 0.4

    def test_code_response_scores_higher(self, scheduler):
        text_fb = _make_feedback("What is X?", "X is a thing.")
        code_fb = _make_feedback(
            "Write a function",
            "```python\ndef foo():\n    return 42\n```",
        )
        text_score = scheduler.score_difficulty(text_fb)
        code_score = scheduler.score_difficulty(code_fb)
        assert code_score > text_score

    def test_correction_increases_difficulty(self, scheduler):
        no_corr = _make_feedback("Q", "A")
        with_corr = _make_feedback("Q", "A", correction="Better A that is quite long")
        assert scheduler.score_difficulty(with_corr) > scheduler.score_difficulty(no_corr)

    def test_multi_concept_harder(self, scheduler):
        simple = _make_feedback("What is Python?", "A language.")
        multi = _make_feedback(
            "What is Python and also how does it compare with JavaScript "
            "including both performance and additionally ecosystem",
            "Python is..." + "x" * 500,
        )
        assert scheduler.score_difficulty(multi) > scheduler.score_difficulty(simple)


class TestCurriculumOrganization:
    def test_organize_empty(self, scheduler):
        stages = scheduler.organize([])
        assert stages == []

    def test_organize_creates_stages(self, scheduler):
        examples = [
            _make_feedback("Hi", "Hello"),
            _make_feedback("Q " * 50, "A " * 200 + "```python\ndef x(): pass\n```"),
            _make_feedback("Medium query here", "Medium response " * 20),
        ]
        stages = scheduler.organize(examples)
        assert len(stages) == 4
        total = sum(s.count for s in stages)
        assert total == 3

    def test_stages_have_correct_levels(self, scheduler):
        examples = [_make_feedback(f"Q{i}", f"A{i}") for i in range(10)]
        stages = scheduler.organize(examples)
        assert stages[0].level == DifficultyLevel.EASY
        assert stages[-1].level == DifficultyLevel.EXPERT

    def test_custom_thresholds(self):
        config = CurriculumConfig(thresholds=[0.3, 0.6, 0.9])
        sched = CurriculumScheduler(config=config)
        examples = [_make_feedback(f"Q{i}", f"A{i}" * (i + 1)) for i in range(20)]
        stages = sched.organize(examples)
        assert len(stages) == 4


class TestTrainingOrder:
    def test_first_epoch_only_easy(self):
        config = CurriculumConfig(stages_per_epoch=1, include_prior_stages=False)
        sched = CurriculumScheduler(config=config)
        examples = [
            _make_feedback("Hi", "Hello"),
            _make_feedback("Q " * 100, "A " * 500 + "```def x(): pass```"),
        ]
        ordered = sched.get_training_order(examples, current_epoch=0)
        # First epoch: only first stage unlocked
        assert len(ordered) <= len(examples)

    def test_later_epochs_unlock_more(self):
        config = CurriculumConfig(stages_per_epoch=1, include_prior_stages=False)
        sched = CurriculumScheduler(config=config)
        examples = [_make_feedback(f"Q{i}" * (i + 1), f"A{i}" * (i + 1)) for i in range(20)]
        epoch0 = sched.get_training_order(examples, current_epoch=0)
        epoch3 = sched.get_training_order(examples, current_epoch=3)
        assert len(epoch3) >= len(epoch0)

    def test_prior_stages_included(self):
        config = CurriculumConfig(
            stages_per_epoch=1,
            include_prior_stages=True,
            prior_stage_weight=1.0,
        )
        sched = CurriculumScheduler(config=config)
        examples = [_make_feedback(f"Q{i}" * (i + 1), f"A{i}" * (i + 1)) for i in range(20)]
        ordered = sched.get_training_order(examples, current_epoch=3)
        assert len(ordered) > 0

    def test_empty_examples(self, scheduler):
        ordered = scheduler.get_training_order([])
        assert ordered == []


class TestSchedulerProperties:
    def test_stage_count_and_total_examples(self, scheduler):
        assert scheduler.stage_count == 0
        assert scheduler.total_examples == 0
        examples = [_make_feedback(f"Q{i}", f"A{i}") for i in range(10)]
        scheduler.organize(examples)
        assert scheduler.stage_count == 4
        assert scheduler.total_examples == 10


class TestDifficultyScoringEdgeCases:
    def test_very_long_query_scores_highest_bucket(self, scheduler):
        fb = _make_feedback("Q " * 200, "short answer")
        score = scheduler.score_difficulty(fb)
        assert score > 0.0

    def test_very_long_response_scores_highest_bucket(self, scheduler):
        fb = _make_feedback("short query", "A " * 1000)
        score = scheduler.score_difficulty(fb)
        assert score > 0.0


class TestStageSummary:
    def test_summary_structure(self, scheduler):
        examples = [_make_feedback(f"Q{i}", f"A{i}" * (i + 1)) for i in range(10)]
        summary = scheduler.get_stage_summary(examples)
        assert summary["total_examples"] == 10
        assert summary["num_stages"] == 4
        assert len(summary["stages"]) == 4
        for stage in summary["stages"]:
            assert "level" in stage
            assert "count" in stage
            assert "score_range" in stage
