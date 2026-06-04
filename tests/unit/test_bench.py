"""Tests for the bench command — suite definitions, scoring, and display."""

from __future__ import annotations

import json

import pytest

from llmstack.cli.commands.bench import (
    MAX_SCORE_BLOCKS,
    SUITE_NAMES,
    SUITES,
    ModelResult,
    PromptResult,
    SuiteResult,
    _display_comparison,
    _display_single_model,
    _results_to_dict,
    _score_bar,
)


# ---------------------------------------------------------------------------
# Suite definitions
# ---------------------------------------------------------------------------


class TestSuiteDefinitions:
    def test_all_expected_suites_exist(self):
        expected = {"simple", "reasoning", "coding", "long_context", "creative"}
        assert expected == set(SUITE_NAMES)

    def test_simple_suite_has_prompts(self):
        assert len(SUITES["simple"]) == 3

    def test_reasoning_suite_has_prompts(self):
        assert len(SUITES["reasoning"]) == 2

    def test_coding_suite_has_prompts(self):
        assert len(SUITES["coding"]) == 2

    def test_long_context_suite_has_prompts(self):
        assert len(SUITES["long_context"]) == 1
        assert len(SUITES["long_context"][0]) > 200  # actually long

    def test_creative_suite_has_prompts(self):
        assert len(SUITES["creative"]) == 2

    def test_all_prompts_are_strings(self):
        for suite_name, prompts in SUITES.items():
            for prompt in prompts:
                assert isinstance(prompt, str), f"Non-string prompt in {suite_name}"
                assert len(prompt) > 0, f"Empty prompt in {suite_name}"


# ---------------------------------------------------------------------------
# Score bar
# ---------------------------------------------------------------------------


class TestScoreBar:
    def test_full_bar(self):
        bar = _score_bar(100.0, 100.0)
        assert bar == "█" * MAX_SCORE_BLOCKS

    def test_empty_bar(self):
        bar = _score_bar(0.0, 100.0)
        assert bar == "░" * MAX_SCORE_BLOCKS

    def test_half_bar(self):
        bar = _score_bar(50.0, 100.0)
        filled = bar.count("█")
        empty = bar.count("░")
        assert filled + empty == MAX_SCORE_BLOCKS
        assert filled == 4

    def test_zero_max(self):
        bar = _score_bar(50.0, 0.0)
        assert bar == "░" * MAX_SCORE_BLOCKS

    def test_over_max_clamps(self):
        bar = _score_bar(200.0, 100.0)
        assert bar == "█" * MAX_SCORE_BLOCKS


# ---------------------------------------------------------------------------
# PromptResult
# ---------------------------------------------------------------------------


class TestPromptResult:
    def test_basic_creation(self):
        pr = PromptResult(
            suite="simple",
            prompt="Hello!",
            ttft_ms=45.0,
            total_time_s=0.4,
            input_tokens=2,
            output_tokens=30,
            tokens_per_second=75.0,
        )
        assert pr.error is None
        assert pr.tokens_per_second == 75.0

    def test_error_result(self):
        pr = PromptResult(
            suite="simple",
            prompt="Hello!",
            ttft_ms=0,
            total_time_s=0.1,
            input_tokens=0,
            output_tokens=0,
            tokens_per_second=0,
            error="Connection refused",
        )
        assert pr.error == "Connection refused"


# ---------------------------------------------------------------------------
# SuiteResult aggregation
# ---------------------------------------------------------------------------


def _make_prompt_result(
    ttft_ms: float, tps: float, total_s: float, error: str | None = None
) -> PromptResult:
    return PromptResult(
        suite="test",
        prompt="test prompt",
        ttft_ms=ttft_ms,
        total_time_s=total_s,
        input_tokens=10,
        output_tokens=int(tps * total_s),
        tokens_per_second=tps,
        error=error,
    )


class TestSuiteResult:
    def test_avg_ttft(self):
        sr = SuiteResult(
            name="simple",
            prompts=[
                _make_prompt_result(40.0, 80.0, 0.4),
                _make_prompt_result(60.0, 70.0, 0.5),
            ],
        )
        assert sr.avg_ttft_ms == pytest.approx(50.0)

    def test_avg_tokens_per_second(self):
        sr = SuiteResult(
            name="simple",
            prompts=[
                _make_prompt_result(40.0, 80.0, 0.4),
                _make_prompt_result(60.0, 60.0, 0.5),
            ],
        )
        assert sr.avg_tokens_per_second == pytest.approx(70.0)

    def test_errors_excluded_from_averages(self):
        sr = SuiteResult(
            name="simple",
            prompts=[
                _make_prompt_result(40.0, 80.0, 0.4),
                _make_prompt_result(0, 0, 0.1, error="fail"),
            ],
        )
        assert sr.avg_ttft_ms == pytest.approx(40.0)
        assert sr.avg_tokens_per_second == pytest.approx(80.0)
        assert sr.errors == 1

    def test_total_time(self):
        sr = SuiteResult(
            name="simple",
            prompts=[
                _make_prompt_result(40.0, 80.0, 1.0),
                _make_prompt_result(60.0, 70.0, 2.0),
            ],
        )
        assert sr.total_time == pytest.approx(3.0)

    def test_empty_suite(self):
        sr = SuiteResult(name="empty")
        assert sr.avg_ttft_ms == 0.0
        assert sr.avg_tokens_per_second == 0.0
        assert sr.errors == 0


# ---------------------------------------------------------------------------
# ModelResult aggregation
# ---------------------------------------------------------------------------


class TestModelResult:
    def test_aggregation(self):
        sr1 = SuiteResult(
            name="simple",
            prompts=[
                _make_prompt_result(40.0, 80.0, 0.4),
            ],
        )
        sr2 = SuiteResult(
            name="reasoning",
            prompts=[
                _make_prompt_result(100.0, 60.0, 2.0),
            ],
        )
        mr = ModelResult(model="llama3.2", suites=[sr1, sr2])
        assert mr.avg_ttft_ms == pytest.approx(70.0)
        assert mr.avg_tokens_per_second == pytest.approx(70.0)
        assert mr.total_prompts == 2
        assert mr.total_errors == 0

    def test_total_time(self):
        sr = SuiteResult(
            name="simple",
            prompts=[
                _make_prompt_result(40.0, 80.0, 1.0),
                _make_prompt_result(60.0, 70.0, 2.0),
            ],
        )
        mr = ModelResult(model="test", suites=[sr])
        assert mr.total_time == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# JSON export
# ---------------------------------------------------------------------------


class TestResultsToDict:
    def test_structure(self):
        sr = SuiteResult(
            name="simple",
            prompts=[
                _make_prompt_result(45.0, 82.3, 0.4),
            ],
        )
        mr = ModelResult(model="llama3.2", suites=[sr])
        data = _results_to_dict([mr])

        assert len(data) == 1
        assert data[0]["model"] == "llama3.2"
        assert "avg_ttft_ms" in data[0]
        assert "avg_tokens_per_second" in data[0]
        assert len(data[0]["suites"]) == 1
        assert data[0]["suites"][0]["name"] == "simple"
        assert len(data[0]["suites"][0]["prompts"]) == 1

    def test_serializable(self):
        sr = SuiteResult(
            name="simple",
            prompts=[
                _make_prompt_result(45.0, 82.3, 0.4),
            ],
        )
        mr = ModelResult(model="llama3.2", suites=[sr])
        data = _results_to_dict([mr])
        # Should not raise
        json_str = json.dumps(data)
        assert "llama3.2" in json_str

    def test_multiple_models(self):
        sr1 = SuiteResult(name="simple", prompts=[_make_prompt_result(40.0, 80.0, 0.4)])
        sr2 = SuiteResult(name="simple", prompts=[_make_prompt_result(80.0, 40.0, 1.0)])
        data = _results_to_dict(
            [
                ModelResult(model="fast-model", suites=[sr1]),
                ModelResult(model="slow-model", suites=[sr2]),
            ]
        )
        assert len(data) == 2
        assert data[0]["model"] == "fast-model"
        assert data[1]["model"] == "slow-model"


# ---------------------------------------------------------------------------
# Display functions (smoke tests — verify they don't crash)
# ---------------------------------------------------------------------------


class TestDisplay:
    def _make_model_result(self, model: str = "llama3.2") -> ModelResult:
        suites = []
        for name in ["simple", "reasoning"]:
            sr = SuiteResult(
                name=name,
                prompts=[
                    _make_prompt_result(50.0, 75.0, 1.0),
                    _make_prompt_result(60.0, 65.0, 1.5),
                ],
            )
            suites.append(sr)
        return ModelResult(model=model, suites=suites)

    def test_display_single_model_no_crash(self):
        mr = self._make_model_result()
        # Should not raise
        _display_single_model(mr)

    def test_display_comparison_no_crash(self):
        results = [
            self._make_model_result("llama3.2"),
            self._make_model_result("mistral:7b"),
        ]
        # Should not raise
        _display_comparison(results)

    def test_display_single_model_with_errors(self):
        sr = SuiteResult(
            name="simple",
            prompts=[
                _make_prompt_result(0, 0, 0.1, error="Connection refused"),
            ],
        )
        mr = ModelResult(model="broken", suites=[sr])
        # Should not raise
        _display_single_model(mr)
