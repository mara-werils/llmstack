"""Tests for prompt optimization (llmstack.learn.optimizer)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from llmstack.learn.feedback import Feedback, FeedbackType
from llmstack.learn.optimizer import (
    OptimizationResult,
    PromptOptimizer,
    PromptVariant,
)


@pytest.fixture
def prompts_dir(tmp_path):
    """Isolated prompts directory."""
    d = tmp_path / "prompts"
    d.mkdir()
    return d


@pytest.fixture
def store():
    """A mock FeedbackStore — only get_feedback is exercised by the optimizer."""
    s = MagicMock()
    s.get_feedback.return_value = []
    return s


@pytest.fixture
def optimizer(store, prompts_dir):
    return PromptOptimizer(store=store, prompts_dir=prompts_dir)


def _correction(response: str, correction: str) -> Feedback:
    return Feedback(
        feedback_type=FeedbackType.CORRECTION,
        query="q",
        response=response,
        correction=correction,
    )


class TestPromptVariant:
    def test_satisfaction_rate_no_data(self):
        v = PromptVariant(id="a", name="n", template="t")
        assert v.satisfaction_rate == 0.5

    def test_satisfaction_rate_with_data(self):
        v = PromptVariant(
            id="a", name="n", template="t", positive_feedback=3, negative_feedback=1
        )
        assert v.satisfaction_rate == 0.75

    def test_correction_rate_no_uses(self):
        v = PromptVariant(id="a", name="n", template="t")
        assert v.correction_rate == 0.0

    def test_correction_rate_with_uses(self):
        v = PromptVariant(id="a", name="n", template="t", total_uses=4, corrections=1)
        assert v.correction_rate == 0.25

    def test_to_dict(self):
        v = PromptVariant(
            id="abc",
            name="greeter",
            template="hello",
            positive_feedback=1,
            negative_feedback=1,
            total_uses=2,
            corrections=1,
            avg_quality=0.123456,
        )
        d = v.to_dict()
        assert d["id"] == "abc"
        assert d["name"] == "greeter"
        assert d["template"] == "hello"
        assert d["avg_quality"] == 0.1235  # rounded to 4 places
        assert d["satisfaction_rate"] == 0.5
        assert d["correction_rate"] == 0.5
        assert d["is_active"] is True
        assert d["metadata"] == {}


class TestOptimizationResult:
    def test_not_optimized(self):
        original = PromptVariant(id="a", name="n", template="t")
        result = OptimizationResult(original=original)
        assert result.was_optimized is False
        assert result.improvement_count == 0

    def test_optimized(self):
        original = PromptVariant(id="a", name="n", template="t")
        optimized = PromptVariant(id="b", name="n", template="t2")
        result = OptimizationResult(
            original=original,
            optimized=optimized,
            improvements=["x", "y"],
        )
        assert result.was_optimized is True
        assert result.improvement_count == 2


class TestPromptOptimizerInit:
    def test_init_creates_dir_and_empty(self, store, tmp_path):
        target = tmp_path / "nested" / "prompts"
        assert not target.exists()
        opt = PromptOptimizer(store=store, prompts_dir=target)
        assert target.exists()
        assert opt.variant_count == 0
        assert opt.active_variants == []

    def test_init_default_dir(self, store, monkeypatch, tmp_path):
        # Patch the module-level default so we never touch the real home dir.
        import llmstack.learn.optimizer as mod

        default = tmp_path / "default_prompts"
        monkeypatch.setattr(mod, "PROMPTS_DIR", default)
        opt = PromptOptimizer(store=store)
        assert opt.prompts_dir == default
        assert default.exists()

    def test_load_variants_from_disk(self, store, prompts_dir):
        v = PromptVariant(id="seed01", name="seeded", template="hi")
        (prompts_dir / f"{v.id}.json").write_text(json.dumps(v.to_dict(), indent=2))
        opt = PromptOptimizer(store=store, prompts_dir=prompts_dir)
        assert opt.variant_count == 1
        loaded = opt.get_variants("seeded")[0]
        assert loaded.id == "seed01"
        assert loaded.template == "hi"

    def test_load_variants_skips_bad_json(self, store, prompts_dir):
        (prompts_dir / "broken.json").write_text("{not valid json")
        opt = PromptOptimizer(store=store, prompts_dir=prompts_dir)
        assert opt.variant_count == 0

    def test_load_variants_skips_missing_keys(self, store, prompts_dir):
        # Valid JSON but missing required keys (id/name/template) -> KeyError branch.
        (prompts_dir / "partial.json").write_text(json.dumps({"foo": "bar"}))
        opt = PromptOptimizer(store=store, prompts_dir=prompts_dir)
        assert opt.variant_count == 0


class TestRegisterAndPersist:
    def test_register_prompt(self, optimizer, prompts_dir):
        v = optimizer.register_prompt("coder", "write code", metadata={"k": "v"})
        assert v.name == "coder"
        assert v.template == "write code"
        assert v.metadata == {"k": "v"}
        assert len(v.id) == 10  # md5 hexdigest truncated to 10
        assert optimizer.variant_count == 1
        # File was written to disk.
        assert (prompts_dir / f"{v.id}.json").exists()

    def test_register_prompt_default_metadata(self, optimizer):
        v = optimizer.register_prompt("plain", "tmpl")
        assert v.metadata == {}

    def test_register_is_deterministic(self, optimizer):
        v1 = optimizer.register_prompt("same", "tmpl")
        v2 = optimizer.register_prompt("same", "tmpl")
        assert v1.id == v2.id

    def test_register_persists_and_reloads(self, store, prompts_dir):
        opt = PromptOptimizer(store=store, prompts_dir=prompts_dir)
        v = opt.register_prompt("persist", "tmpl")
        reloaded = PromptOptimizer(store=store, prompts_dir=prompts_dir)
        assert reloaded.variant_count == 1
        assert reloaded.get_variants("persist")[0].id == v.id


class TestActiveVariants:
    def test_active_variants_filters_inactive(self, optimizer):
        a = optimizer.register_prompt("a", "ta")
        optimizer.register_prompt("b", "tb")
        a.is_active = False
        active = optimizer.active_variants
        assert len(active) == 1
        assert active[0].name == "b"


class TestRecordUse:
    def test_record_use_unknown_id_is_noop(self, optimizer):
        # Should silently return without raising.
        optimizer.record_use("does-not-exist")

    def test_record_use_increments_total(self, optimizer):
        v = optimizer.register_prompt("n", "t")
        optimizer.record_use(v.id)
        assert v.total_uses == 1

    def test_record_use_positive_feedback(self, optimizer):
        v = optimizer.register_prompt("n", "t")
        fb = Feedback(feedback_type=FeedbackType.THUMBS_UP)
        optimizer.record_use(v.id, feedback=fb)
        assert v.positive_feedback == 1
        assert v.negative_feedback == 0

    def test_record_use_negative_feedback(self, optimizer):
        v = optimizer.register_prompt("n", "t")
        fb = Feedback(feedback_type=FeedbackType.THUMBS_DOWN)
        optimizer.record_use(v.id, feedback=fb)
        assert v.negative_feedback == 1
        assert v.positive_feedback == 0

    def test_record_use_correction_feedback(self, optimizer):
        v = optimizer.register_prompt("n", "t")
        fb = Feedback(
            feedback_type=FeedbackType.CORRECTION,
            response="bad",
            correction="good",
        )
        optimizer.record_use(v.id, feedback=fb)
        assert v.corrections == 1

    def test_record_use_quality_running_average(self, optimizer):
        v = optimizer.register_prompt("n", "t")
        optimizer.record_use(v.id, quality_score=0.8)
        assert v.avg_quality == pytest.approx(0.8)
        optimizer.record_use(v.id, quality_score=0.6)
        # running avg of 0.8 then 0.6 over 2 uses = 0.7
        assert v.avg_quality == pytest.approx(0.7)

    def test_record_use_zero_quality_ignored(self, optimizer):
        v = optimizer.register_prompt("n", "t")
        optimizer.record_use(v.id, quality_score=0.0)
        assert v.avg_quality == 0.0

    def test_record_use_persists(self, optimizer, prompts_dir):
        v = optimizer.register_prompt("n", "t")
        optimizer.record_use(v.id, quality_score=0.9)
        data = json.loads((prompts_dir / f"{v.id}.json").read_text())
        assert data["total_uses"] == 1
        assert data["avg_quality"] == 0.9


class TestAnalyzePatterns:
    def test_no_variants_returns_empty(self, optimizer):
        assert optimizer.analyze_patterns("missing") == []

    def test_no_corrections_returns_empty(self, optimizer, store):
        optimizer.register_prompt("n", "t")
        store.get_feedback.return_value = []
        assert optimizer.analyze_patterns("n") == []

    def test_verbose_pattern(self, optimizer, store):
        optimizer.register_prompt("n", "t")
        # Corrections much shorter than response -> shortening pattern.
        store.get_feedback.return_value = [
            _correction(response="x" * 100, correction="y" * 10) for _ in range(5)
        ]
        patterns = optimizer.analyze_patterns("n")
        assert any("too verbose" in p for p in patterns)

    def test_lacks_detail_pattern(self, optimizer, store):
        optimizer.register_prompt("n", "t")
        # Corrections much longer than response -> lengthening pattern.
        store.get_feedback.return_value = [
            _correction(response="x" * 10, correction="y" * 100) for _ in range(5)
        ]
        patterns = optimizer.analyze_patterns("n")
        assert any("lacks detail" in p for p in patterns)

    def test_code_formatting_pattern(self, optimizer, store):
        optimizer.register_prompt("n", "t")
        # >3 corrections that add code fences absent from the response.
        store.get_feedback.return_value = [
            _correction(response="plain text", correction="```python\nx=1\n```")
            for _ in range(4)
        ]
        patterns = optimizer.analyze_patterns("n")
        assert any("format code" in p for p in patterns)

    def test_hedging_pattern(self, optimizer, store):
        optimizer.register_prompt("n", "t")
        # >3 corrections that remove hedging language from the response.
        store.get_feedback.return_value = [
            _correction(response="I think it works", correction="It works")
            for _ in range(4)
        ]
        patterns = optimizer.analyze_patterns("n")
        assert any("hedging" in p for p in patterns)

    def test_structure_pattern(self, optimizer, store):
        optimizer.register_prompt("n", "t")
        # >3 corrections that add many list items vs response.
        store.get_feedback.return_value = [
            _correction(response="flat", correction="\n- a\n- b\n- c")
            for _ in range(4)
        ]
        patterns = optimizer.analyze_patterns("n")
        assert any("structure" in p for p in patterns)

    def test_structure_pattern_headers(self, optimizer, store):
        optimizer.register_prompt("n", "t")
        # Headers added beats the list-item branch.
        store.get_feedback.return_value = [
            _correction(response="flat", correction="\n# Heading\ncontent")
            for _ in range(4)
        ]
        patterns = optimizer.analyze_patterns("n")
        assert any("structure" in p for p in patterns)

    def test_no_pattern_below_thresholds(self, optimizer, store):
        optimizer.register_prompt("n", "t")
        # A few corrections that don't trip any threshold.
        store.get_feedback.return_value = [
            _correction(response="abc", correction="abcd"),
        ]
        assert optimizer.analyze_patterns("n") == []

    def test_get_feedback_called_with_correction_type(self, optimizer, store):
        optimizer.register_prompt("n", "t")
        optimizer.analyze_patterns("n")
        _, kwargs = store.get_feedback.call_args
        assert kwargs["feedback_type"] == FeedbackType.CORRECTION
        assert kwargs["limit"] == 100


class TestSuggestImprovements:
    def test_suggestions_for_verbose(self, optimizer, store):
        optimizer.register_prompt("n", "t")
        store.get_feedback.return_value = [
            _correction(response="x" * 100, correction="y" * 10) for _ in range(5)
        ]
        suggestions = optimizer.suggest_improvements("n")
        assert any("concise" in s.lower() for s in suggestions)

    def test_suggestions_for_detail(self, optimizer, store):
        optimizer.register_prompt("n", "t")
        store.get_feedback.return_value = [
            _correction(response="x" * 10, correction="y" * 100) for _ in range(5)
        ]
        suggestions = optimizer.suggest_improvements("n")
        assert any("detailed" in s.lower() for s in suggestions)

    def test_suggestions_for_code(self, optimizer, store):
        optimizer.register_prompt("n", "t")
        store.get_feedback.return_value = [
            _correction(response="plain", correction="```py\nx\n```") for _ in range(4)
        ]
        suggestions = optimizer.suggest_improvements("n")
        assert any("```" in s for s in suggestions)

    def test_suggestions_for_hedging(self, optimizer, store):
        optimizer.register_prompt("n", "t")
        store.get_feedback.return_value = [
            _correction(response="maybe it works", correction="it works")
            for _ in range(4)
        ]
        suggestions = optimizer.suggest_improvements("n")
        assert any("direct" in s.lower() for s in suggestions)

    def test_suggestions_for_structure(self, optimizer, store):
        optimizer.register_prompt("n", "t")
        store.get_feedback.return_value = [
            _correction(response="flat", correction="\n- a\n- b\n- c")
            for _ in range(4)
        ]
        suggestions = optimizer.suggest_improvements("n")
        assert any("bullet" in s.lower() for s in suggestions)

    def test_no_patterns_no_suggestions(self, optimizer, store):
        optimizer.register_prompt("n", "t")
        store.get_feedback.return_value = []
        assert optimizer.suggest_improvements("n") == []


class TestGetBestVariant:
    def test_none_when_no_variants(self, optimizer):
        assert optimizer.get_best_variant("missing") is None

    def test_none_when_below_min_uses(self, optimizer):
        v = optimizer.register_prompt("n", "t")
        # Fewer than 5 uses -> excluded.
        for _ in range(4):
            optimizer.record_use(v.id)
        assert optimizer.get_best_variant("n") is None

    def test_picks_highest_satisfaction(self, optimizer):
        low = optimizer.register_prompt("dup", "low template")
        high = optimizer.register_prompt("dup", "high template")
        low.total_uses = 5
        low.positive_feedback = 1
        low.negative_feedback = 4
        high.total_uses = 5
        high.positive_feedback = 4
        high.negative_feedback = 1
        best = optimizer.get_best_variant("dup")
        assert best is high


class TestGetVariants:
    def test_all_variants(self, optimizer):
        optimizer.register_prompt("a", "ta")
        optimizer.register_prompt("b", "tb")
        assert len(optimizer.get_variants()) == 2

    def test_filter_by_name(self, optimizer):
        optimizer.register_prompt("a", "ta")
        optimizer.register_prompt("b", "tb")
        result = optimizer.get_variants("a")
        assert len(result) == 1
        assert result[0].name == "a"

    def test_sorted_by_satisfaction_desc(self, optimizer):
        a = optimizer.register_prompt("a", "ta")
        b = optimizer.register_prompt("b", "tb")
        a.positive_feedback = 1
        a.negative_feedback = 9  # 0.1
        b.positive_feedback = 9
        b.negative_feedback = 1  # 0.9
        result = optimizer.get_variants()
        assert result[0].name == "b"
        assert result[1].name == "a"

    def test_empty(self, optimizer):
        assert optimizer.get_variants() == []
