"""Evaluation — compare base vs fine-tuned model quality.

Runs a set of eval prompts through both models and computes quality metrics.
Uses Ollama for inference (no GPU required for eval).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import httpx

from llmstack.finetune.data import ChatExample

logger = logging.getLogger(__name__)


@dataclass
class EvalScore:
    """Score for a single eval example."""

    prompt: str
    expected: str
    base_response: str = ""
    tuned_response: str = ""
    base_similarity: float = 0.0
    tuned_similarity: float = 0.0
    improvement: float = 0.0


@dataclass
class EvalResult:
    """Aggregated evaluation results."""

    base_model: str = ""
    tuned_model: str = ""
    num_examples: int = 0
    base_avg_score: float = 0.0
    tuned_avg_score: float = 0.0
    improvement_pct: float = 0.0
    base_avg_latency_ms: float = 0.0
    tuned_avg_latency_ms: float = 0.0
    scores: list[EvalScore] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "base_model": self.base_model,
            "tuned_model": self.tuned_model,
            "num_examples": self.num_examples,
            "base_avg_score": round(self.base_avg_score, 4),
            "tuned_avg_score": round(self.tuned_avg_score, 4),
            "improvement_pct": round(self.improvement_pct, 1),
            "base_avg_latency_ms": round(self.base_avg_latency_ms, 1),
            "tuned_avg_latency_ms": round(self.tuned_avg_latency_ms, 1),
        }


def _word_overlap(reference: str, response: str) -> float:
    """Simple word-overlap similarity (Jaccard)."""
    if not reference or not response:
        return 0.0
    ref_words = set(reference.lower().split())
    resp_words = set(response.lower().split())
    if not ref_words:
        return 0.0
    intersection = ref_words & resp_words
    union = ref_words | resp_words
    return len(intersection) / len(union) if union else 0.0


def _length_ratio(reference: str, response: str) -> float:
    """Score based on response length relative to reference."""
    if not reference:
        return 1.0 if response else 0.0
    ratio = len(response) / max(len(reference), 1)
    # Penalize both too short and too long
    if ratio < 0.3:
        return ratio
    if ratio > 3.0:
        return 1.0 / ratio
    return min(1.0, ratio)


def _combined_score(reference: str, response: str) -> float:
    """Combined quality score: word overlap + length ratio."""
    overlap = _word_overlap(reference, response)
    length = _length_ratio(reference, response)
    return 0.7 * overlap + 0.3 * length


async def _generate(
    ollama_url: str, model: str, messages: list[dict],
) -> tuple[str, float]:
    """Generate a response from Ollama. Returns (text, latency_ms)."""
    t0 = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                f"{ollama_url}/api/chat",
                json={"model": model, "messages": messages, "stream": False},
            )
            resp.raise_for_status()
            data = resp.json()
            text = data.get("message", {}).get("content", "")
            elapsed = (time.monotonic() - t0) * 1000
            return text, elapsed
    except Exception as exc:
        elapsed = (time.monotonic() - t0) * 1000
        logger.warning("Eval generation failed for %s: %s", model, exc)
        return "", elapsed


async def evaluate_model(
    eval_data: list[ChatExample],
    base_model: str,
    tuned_model: str,
    ollama_url: str = "http://localhost:11434",
    max_examples: int = 20,
) -> EvalResult:
    """Run evaluation comparing base model vs fine-tuned model.

    For each eval example, generates responses from both models and
    compares against the expected output using word-overlap similarity.
    """
    examples = eval_data[:max_examples]
    if not examples:
        return EvalResult(error="No eval examples provided")

    scores: list[EvalScore] = []
    base_latencies: list[float] = []
    tuned_latencies: list[float] = []

    for ex in examples:
        # Extract user prompt and expected response
        user_msgs = [m for m in ex.messages if m["role"] == "user"]
        asst_msgs = [m for m in ex.messages if m["role"] == "assistant"]

        if not user_msgs:
            continue

        prompt = user_msgs[-1]["content"]
        expected = asst_msgs[-1]["content"] if asst_msgs else ""

        # Build messages for inference (without the assistant response)
        infer_msgs = [m for m in ex.messages if m["role"] != "assistant"]

        # Generate from base model
        base_resp, base_lat = await _generate(ollama_url, base_model, infer_msgs)
        base_latencies.append(base_lat)

        # Generate from tuned model
        tuned_resp, tuned_lat = await _generate(ollama_url, tuned_model, infer_msgs)
        tuned_latencies.append(tuned_lat)

        # Score
        base_score = _combined_score(expected, base_resp)
        tuned_score = _combined_score(expected, tuned_resp)
        improvement = tuned_score - base_score

        scores.append(EvalScore(
            prompt=prompt[:200],
            expected=expected[:200],
            base_response=base_resp[:200],
            tuned_response=tuned_resp[:200],
            base_similarity=round(base_score, 4),
            tuned_similarity=round(tuned_score, 4),
            improvement=round(improvement, 4),
        ))

    # Aggregate
    base_avg = sum(s.base_similarity for s in scores) / max(len(scores), 1)
    tuned_avg = sum(s.tuned_similarity for s in scores) / max(len(scores), 1)
    improvement_pct = ((tuned_avg - base_avg) / max(base_avg, 0.001)) * 100

    return EvalResult(
        base_model=base_model,
        tuned_model=tuned_model,
        num_examples=len(scores),
        base_avg_score=base_avg,
        tuned_avg_score=tuned_avg,
        improvement_pct=improvement_pct,
        base_avg_latency_ms=sum(base_latencies) / max(len(base_latencies), 1),
        tuned_avg_latency_ms=sum(tuned_latencies) / max(len(tuned_latencies), 1),
        scores=scores,
    )
