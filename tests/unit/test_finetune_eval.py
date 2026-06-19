"""Tests for llmstack.finetune.eval."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from llmstack.finetune.data import ChatExample
from llmstack.finetune.eval import (
    EvalResult,
    _combined_score,
    _generate,
    _length_ratio,
    _word_overlap,
    evaluate_model,
)


def test_word_overlap_empty_inputs():
    assert _word_overlap("", "anything") == 0.0
    assert _word_overlap("anything", "") == 0.0


def test_word_overlap_whitespace_only_reference():
    assert _word_overlap("   ", "hello world") == 0.0


def test_word_overlap_partial_match():
    score = _word_overlap("the quick brown fox", "the slow brown dog")
    assert 0.0 < score < 1.0


def test_word_overlap_identical():
    assert _word_overlap("hello world", "hello world") == 1.0


def test_length_ratio_both_empty():
    assert _length_ratio("", "") == 0.0


def test_length_ratio_empty_reference_nonempty_response():
    assert _length_ratio("", "something") == 1.0


def test_length_ratio_too_short():
    ratio = _length_ratio("a" * 100, "short")
    assert ratio == len("short") / 100


def test_length_ratio_too_long():
    ratio = _length_ratio("a" * 10, "b" * 100)
    assert ratio == 10 / 100


def test_length_ratio_normal():
    assert _length_ratio("a" * 10, "b" * 8) == 0.8


def test_combined_score_blends_overlap_and_length():
    score = _combined_score("hello world", "hello world")
    assert score == pytest.approx(0.7 * 1.0 + 0.3 * 1.0)


def test_eval_result_to_dict_rounds():
    result = EvalResult(
        base_model="base",
        tuned_model="tuned",
        num_examples=3,
        base_avg_score=0.12345,
        tuned_avg_score=0.6789,
        improvement_pct=12.345,
        base_avg_latency_ms=100.05,
        tuned_avg_latency_ms=80.04,
    )
    d = result.to_dict()
    assert d["base_avg_score"] == 0.1235
    assert d["tuned_avg_score"] == 0.6789
    assert d["improvement_pct"] == 12.3
    assert d["base_avg_latency_ms"] == round(100.05, 1)
    assert d["tuned_avg_latency_ms"] == 80.0


class _FakeResponse:
    def __init__(self, payload, status_error=None):
        self._payload = payload
        self._status_error = status_error

    def raise_for_status(self):
        if self._status_error:
            raise self._status_error

    def json(self):
        return self._payload


class _FakeAsyncClient:
    response = _FakeResponse({"message": {"content": "hi there"}})

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def post(self, url, json=None):
        return self.response


@pytest.mark.asyncio
async def test_generate_success():
    with patch("llmstack.finetune.eval.httpx.AsyncClient", _FakeAsyncClient):
        text, latency_ms = await _generate(
            "http://ollama", "model-a", [{"role": "user", "content": "hi"}]
        )
    assert text == "hi there"
    assert latency_ms >= 0


@pytest.mark.asyncio
async def test_generate_handles_exception():
    class RaisingClient(_FakeAsyncClient):
        async def post(self, url, json=None):
            raise ConnectionError("down")

    with patch("llmstack.finetune.eval.httpx.AsyncClient", RaisingClient):
        text, latency_ms = await _generate(
            "http://ollama", "model-a", [{"role": "user", "content": "hi"}]
        )
    assert text == ""
    assert latency_ms >= 0


@pytest.mark.asyncio
async def test_evaluate_model_no_examples():
    result = await evaluate_model([], "base", "tuned")
    assert result.error == "No eval examples provided"


@pytest.mark.asyncio
async def test_evaluate_model_skips_examples_without_user_message():
    examples = [ChatExample(messages=[{"role": "assistant", "content": "no user turn"}])]
    result = await evaluate_model(examples, "base", "tuned")
    assert result.num_examples == 0


@pytest.mark.asyncio
async def test_evaluate_model_full_flow_with_missing_assistant_message():
    examples = [
        ChatExample(
            messages=[
                {"role": "user", "content": "what is the capital of france"},
                {"role": "assistant", "content": "Paris is the capital of France"},
            ]
        ),
        ChatExample(messages=[{"role": "user", "content": "no expected answer here"}]),
    ]

    async def fake_generate(ollama_url, model, messages):
        if model == "base":
            return "Paris", 50.0
        return "Paris is the capital of France", 30.0

    with patch("llmstack.finetune.eval._generate", side_effect=fake_generate):
        result = await evaluate_model(examples, "base", "tuned", max_examples=5)

    assert result.error is None
    assert result.num_examples == 2
    assert result.tuned_avg_score >= result.base_avg_score
    assert result.base_avg_latency_ms == 50.0
    assert result.tuned_avg_latency_ms == 30.0
    assert len(result.scores) == 2


@pytest.mark.asyncio
async def test_evaluate_model_respects_max_examples():
    examples = [
        ChatExample(
            messages=[
                {"role": "user", "content": f"q{i}"},
                {"role": "assistant", "content": f"a{i}"},
            ]
        )
        for i in range(5)
    ]

    with patch("llmstack.finetune.eval._generate", new=AsyncMock(return_value=("resp", 10.0))):
        result = await evaluate_model(examples, "base", "tuned", max_examples=2)

    assert result.num_examples == 2
