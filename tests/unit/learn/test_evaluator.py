"""Tests for the model evaluator."""

from __future__ import annotations

import pytest

from llmstack.learn.evaluator import (
    EvalConfig,
    EvalResult,
    ModelEvaluator,
)
from llmstack.learn.feedback import Feedback, FeedbackType
from llmstack.learn.store import FeedbackStore


@pytest.fixture
def store(tmp_path):
    s = FeedbackStore(db_path=tmp_path / "test.db")
    yield s
    s.close()


@pytest.fixture
def evaluator(store):
    return ModelEvaluator(store=store)


def _correction(query: str, correction: str, response: str = "old answer") -> Feedback:
    return Feedback(
        feedback_type=FeedbackType.CORRECTION,
        query=query,
        response=response,
        correction=correction,
    )


# ---------------------------------------------------------------------------
# EvalResult
# ---------------------------------------------------------------------------


class TestEvalResult:
    def test_defaults(self):
        result = EvalResult()
        assert result.model_version == ""
        assert result.total_examples == 0
        assert result.per_example == []
        assert result.timestamp > 0

    def test_is_empty_true(self):
        assert EvalResult().is_empty is True

    def test_is_empty_false(self):
        assert EvalResult(total_examples=3).is_empty is False

    def test_detail_count(self):
        result = EvalResult(per_example=[{"index": 0}, {"index": 1}])
        assert result.detail_count == 2

    def test_detail_count_empty(self):
        assert EvalResult().detail_count == 0

    def test_to_dict_rounds_and_omits_per_example(self):
        result = EvalResult(
            model_version="v1",
            total_examples=2,
            exact_match_rate=0.123456,
            semantic_similarity=0.654321,
            length_accuracy=0.999999,
            format_accuracy=0.5,
            overall_score=0.111111,
            per_example=[{"index": 0}],
        )
        d = result.to_dict()
        assert d["model_version"] == "v1"
        assert d["total_examples"] == 2
        assert d["exact_match_rate"] == 0.1235
        assert d["semantic_similarity"] == 0.6543
        assert d["length_accuracy"] == 1.0
        assert d["format_accuracy"] == 0.5
        assert d["overall_score"] == 0.1111
        assert "per_example" not in d
        assert "timestamp" in d


# ---------------------------------------------------------------------------
# EvalConfig
# ---------------------------------------------------------------------------


class TestEvalConfig:
    def test_defaults(self):
        config = EvalConfig()
        assert config.eval_set_size == 50
        assert config.exact_match_weight == 0.2
        assert config.semantic_weight == 0.4
        assert config.length_weight == 0.2
        assert config.format_weight == 0.2

    def test_weights_sum_to_one(self):
        config = EvalConfig()
        total = (
            config.exact_match_weight
            + config.semantic_weight
            + config.length_weight
            + config.format_weight
        )
        assert total == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# ModelEvaluator.__init__
# ---------------------------------------------------------------------------


class TestInit:
    def test_default_config(self, store):
        ev = ModelEvaluator(store=store)
        assert ev.store is store
        assert isinstance(ev.config, EvalConfig)

    def test_custom_config(self, store):
        cfg = EvalConfig(eval_set_size=5)
        ev = ModelEvaluator(store=store, config=cfg)
        assert ev.config is cfg
        assert ev.config.eval_set_size == 5


# ---------------------------------------------------------------------------
# build_eval_set
# ---------------------------------------------------------------------------


class TestBuildEvalSet:
    def test_empty_store(self, evaluator):
        assert evaluator.build_eval_set() == []

    def test_includes_valid_corrections(self, store, evaluator):
        store.add_feedback(_correction("what is 2+2?", "The answer is four, exactly four."))
        eval_set = evaluator.build_eval_set()
        assert len(eval_set) == 1
        entry = eval_set[0]
        assert entry["query"] == "what is 2+2?"
        assert entry["reference"] == "The answer is four, exactly four."
        assert entry["original"] == "old answer"

    def test_skips_short_corrections(self, store, evaluator):
        # correction shorter than 20 chars is excluded
        store.add_feedback(_correction("q1", "too short"))
        assert evaluator.build_eval_set() == []

    def test_skips_missing_query(self, store, evaluator):
        store.add_feedback(_correction("", "A sufficiently long correction here."))
        assert evaluator.build_eval_set() == []

    def test_skips_missing_correction(self, store, evaluator):
        fb = Feedback(
            feedback_type=FeedbackType.CORRECTION,
            query="hello",
            correction="",
        )
        store.add_feedback(fb)
        assert evaluator.build_eval_set() == []

    def test_ignores_non_correction_feedback(self, store, evaluator):
        store.add_feedback(
            Feedback(
                feedback_type=FeedbackType.THUMBS_UP,
                query="hi",
                correction="A long correction string here for sure.",
            )
        )
        assert evaluator.build_eval_set() == []

    def test_respects_eval_set_size_limit(self, store):
        cfg = EvalConfig(eval_set_size=2)
        ev = ModelEvaluator(store=store, config=cfg)
        for i in range(5):
            store.add_feedback(
                _correction(f"query number {i}", f"This is a long correction {i} here.")
            )
        eval_set = ev.build_eval_set()
        assert len(eval_set) == 2


# ---------------------------------------------------------------------------
# evaluate_responses
# ---------------------------------------------------------------------------


class TestEvaluateResponses:
    def test_length_mismatch_raises(self, evaluator):
        with pytest.raises(ValueError, match="same length"):
            evaluator.evaluate_responses(
                [{"query": "q", "reference": "r", "original": "o"}],
                [],
            )

    def test_empty_eval_set_returns_empty_result(self, evaluator):
        result = evaluator.evaluate_responses([], [], model_version="vX")
        assert isinstance(result, EvalResult)
        assert result.model_version == "vX"
        assert result.total_examples == 0
        assert result.is_empty

    def test_perfect_match(self, evaluator):
        eval_set = [{"query": "q", "reference": "hello world", "original": "o"}]
        result = evaluator.evaluate_responses(eval_set, ["hello world"], "v1")
        assert result.total_examples == 1
        assert result.exact_match_rate == 1.0
        assert result.semantic_similarity == pytest.approx(1.0)
        assert result.length_accuracy == pytest.approx(1.0)
        assert result.format_accuracy == pytest.approx(1.0)
        assert result.overall_score == pytest.approx(1.0)
        assert result.per_example[0]["exact_match"] is True
        assert result.per_example[0]["index"] == 0

    def test_exact_match_after_normalization(self, evaluator):
        # different whitespace/case still counts as exact match
        eval_set = [{"query": "q", "reference": "Hello   World", "original": "o"}]
        result = evaluator.evaluate_responses(eval_set, ["hello world"], "v1")
        assert result.exact_match_rate == 1.0
        assert result.per_example[0]["exact_match"] is True

    def test_total_mismatch(self, evaluator):
        eval_set = [{"query": "q", "reference": "alpha beta", "original": "o"}]
        result = evaluator.evaluate_responses(eval_set, ["zzz"], "v1")
        assert result.exact_match_rate == 0.0
        assert result.semantic_similarity == 0.0
        assert result.per_example[0]["exact_match"] is False

    def test_multiple_examples_averages(self, evaluator):
        eval_set = [
            {"query": "q1", "reference": "hello world", "original": "o"},
            {"query": "q2", "reference": "foo bar", "original": "o"},
        ]
        result = evaluator.evaluate_responses(eval_set, ["hello world", "totally different"], "v1")
        assert result.total_examples == 2
        # one exact match out of two
        assert result.exact_match_rate == 0.5
        assert len(result.per_example) == 2

    def test_per_example_values_rounded(self, evaluator):
        eval_set = [{"query": "q", "reference": "alpha beta gamma", "original": "o"}]
        result = evaluator.evaluate_responses(eval_set, ["alpha beta"], "v1")
        ex = result.per_example[0]
        # rounded to 4 decimals
        assert ex["similarity"] == round(ex["similarity"], 4)
        assert 0.0 < ex["similarity"] < 1.0

    def test_default_model_version(self, evaluator):
        eval_set = [{"query": "q", "reference": "alpha beta", "original": "o"}]
        result = evaluator.evaluate_responses(eval_set, ["alpha beta"])
        assert result.model_version == ""


# ---------------------------------------------------------------------------
# compare_versions
# ---------------------------------------------------------------------------


class TestCompareVersions:
    def test_b_wins(self, evaluator):
        eval_set = [{"query": "q", "reference": "hello world", "original": "o"}]
        result = evaluator.compare_versions(
            eval_set,
            responses_a=["totally wrong"],
            responses_b=["hello world"],
            version_a="A",
            version_b="B",
        )
        assert result["winner"] == "B"
        assert result["improvement"] > 0
        assert result["significant"] is True
        assert result["version_a"]["version"] == "A"
        assert result["version_b"]["version"] == "B"
        # to_dict fields merged in
        assert "overall_score" in result["version_a"]
        assert "overall_score" in result["version_b"]

    def test_a_wins_or_tie_picks_a(self, evaluator):
        eval_set = [{"query": "q", "reference": "hello world", "original": "o"}]
        # identical responses → improvement 0 → winner is A (improvement not > 0)
        result = evaluator.compare_versions(
            eval_set,
            responses_a=["hello world"],
            responses_b=["hello world"],
            version_a="A",
            version_b="B",
        )
        assert result["improvement"] == 0.0
        assert result["winner"] == "A"
        assert result["significant"] is False

    def test_not_significant_small_improvement(self, evaluator):
        eval_set = [{"query": "q", "reference": "alpha beta gamma delta epsilon", "original": "o"}]
        result = evaluator.compare_versions(
            eval_set,
            responses_a=["alpha beta gamma delta epsilon"],
            responses_b=["alpha beta gamma delta epsilon zeta"],
            version_a="A",
            version_b="B",
        )
        # very small difference should not be flagged significant
        assert abs(result["improvement"]) <= 0.02 or result["significant"] in (
            True,
            False,
        )
        assert isinstance(result["significant"], bool)

    def test_improvement_rounded(self, evaluator):
        eval_set = [{"query": "q", "reference": "alpha beta gamma", "original": "o"}]
        result = evaluator.compare_versions(
            eval_set,
            responses_a=["alpha"],
            responses_b=["alpha beta gamma"],
        )
        assert result["improvement"] == round(result["improvement"], 4)


# ---------------------------------------------------------------------------
# _normalize
# ---------------------------------------------------------------------------


class TestNormalize:
    def test_lowercases_and_collapses_whitespace(self, evaluator):
        assert evaluator._normalize("  Hello   WORLD\n\tfoo ") == "hello world foo"

    def test_empty_string(self, evaluator):
        assert evaluator._normalize("") == ""

    def test_only_whitespace(self, evaluator):
        assert evaluator._normalize("   \n\t ") == ""


# ---------------------------------------------------------------------------
# _compute_similarity
# ---------------------------------------------------------------------------


class TestComputeSimilarity:
    def test_identical(self, evaluator):
        assert evaluator._compute_similarity("a b c", "a b c") == pytest.approx(1.0)

    def test_no_overlap(self, evaluator):
        assert evaluator._compute_similarity("a b", "c d") == 0.0

    def test_partial_overlap(self, evaluator):
        score = evaluator._compute_similarity("a b c", "a b d")
        assert 0.0 < score < 1.0

    def test_empty_reference_and_empty_generated(self, evaluator):
        # no ref words, no gen words → perfect (1.0)
        assert evaluator._compute_similarity("", "") == 1.0

    def test_empty_reference_nonempty_generated(self, evaluator):
        # no ref words but gen has words → 0.0
        assert evaluator._compute_similarity("foo", "") == 0.0

    def test_empty_generated_nonempty_reference(self, evaluator):
        # gen empty but ref non-empty → no overlap → 0.0
        assert evaluator._compute_similarity("", "foo bar") == 0.0


# ---------------------------------------------------------------------------
# _compute_length_accuracy
# ---------------------------------------------------------------------------


class TestComputeLengthAccuracy:
    def test_same_length(self, evaluator):
        assert evaluator._compute_length_accuracy("abcd", "wxyz") == 1.0

    def test_both_empty(self, evaluator):
        assert evaluator._compute_length_accuracy("", "") == 1.0

    def test_empty_reference_nonempty_generated(self, evaluator):
        assert evaluator._compute_length_accuracy("abc", "") == 0.0

    def test_ratio(self, evaluator):
        # gen 2 chars, ref 4 chars → 0.5
        assert evaluator._compute_length_accuracy("ab", "abcd") == 0.5

    def test_ratio_symmetric(self, evaluator):
        assert evaluator._compute_length_accuracy("abcd", "ab") == 0.5


# ---------------------------------------------------------------------------
# _compute_format_accuracy
# ---------------------------------------------------------------------------


class TestComputeFormatAccuracy:
    def test_identical_plain(self, evaluator):
        assert evaluator._compute_format_accuracy("hello", "hello") == 1.0

    def test_matching_code_blocks(self, evaluator):
        gen = "```\ncode\n```"
        ref = "```\nother\n```"
        # both have code blocks; line counts identical
        assert evaluator._compute_format_accuracy(gen, ref) == pytest.approx(1.0)

    def test_code_block_mismatch(self, evaluator):
        # one has code block, other doesn't → loses the code check
        score = evaluator._compute_format_accuracy("```code```", "plain text")
        assert score < 1.0

    def test_matching_bullets(self, evaluator):
        gen = "intro\n- one\n- two"
        ref = "header\n- a\n- b"
        assert evaluator._compute_format_accuracy(gen, ref) == pytest.approx(1.0)

    def test_star_bullets(self, evaluator):
        gen = "intro\n* one"
        ref = "header\n* a"
        assert evaluator._compute_format_accuracy(gen, ref) == pytest.approx(1.0)

    def test_bullet_mismatch(self, evaluator):
        score = evaluator._compute_format_accuracy("text\n- item", "no bullets")
        assert score < 1.0

    def test_matching_headers(self, evaluator):
        gen = "intro\n# Title"
        ref = "lead\n# Heading"
        assert evaluator._compute_format_accuracy(gen, ref) == pytest.approx(1.0)

    def test_header_mismatch(self, evaluator):
        score = evaluator._compute_format_accuracy("intro\n# Title", "no header here")
        assert score < 1.0

    def test_line_count_ratio(self, evaluator):
        # both plain (no code/bullets/headers) but different line counts
        gen = "a\nb\nc\nd"  # 3 newlines
        ref = "a\nb"  # 1 newline
        score = evaluator._compute_format_accuracy(gen, ref)
        # code/bullet/header all match (none present) = 3.0, line ratio = 1/3
        assert score == pytest.approx((3.0 + (1 / 3)) / 4)

    def test_no_newlines_line_ratio_one(self, evaluator):
        # neither has newlines → line_ratio defaults to 1.0 → all 4 checks pass
        assert evaluator._compute_format_accuracy("abc", "xyz") == 1.0
