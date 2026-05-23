"""Model evaluator — measures quality improvement after fine-tuning.

Runs standardized evaluation on a held-out set of corrections to measure
whether a fine-tuned version actually improves over the base model.
Uses both automated metrics and reference-based comparison.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from llmstack.learn.feedback import FeedbackType
from llmstack.learn.store import FeedbackStore

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """Result of evaluating a model version."""

    model_version: str = ""
    timestamp: float = field(default_factory=time.time)
    total_examples: int = 0
    exact_match_rate: float = 0.0
    semantic_similarity: float = 0.0
    length_accuracy: float = 0.0
    format_accuracy: float = 0.0
    overall_score: float = 0.0
    per_example: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_version": self.model_version,
            "timestamp": self.timestamp,
            "total_examples": self.total_examples,
            "exact_match_rate": round(self.exact_match_rate, 4),
            "semantic_similarity": round(self.semantic_similarity, 4),
            "length_accuracy": round(self.length_accuracy, 4),
            "format_accuracy": round(self.format_accuracy, 4),
            "overall_score": round(self.overall_score, 4),
        }


@dataclass
class EvalConfig:
    """Configuration for model evaluation."""

    # Number of examples to hold out for evaluation
    eval_set_size: int = 50

    # Weights for overall score
    exact_match_weight: float = 0.2
    semantic_weight: float = 0.4
    length_weight: float = 0.2
    format_weight: float = 0.2


class ModelEvaluator:
    """Evaluates fine-tuned models against held-out corrections.

    Builds an eval set from corrections where we know the "correct" answer,
    then measures how well a model version reproduces those corrections.
    """

    def __init__(self, store: FeedbackStore, config: EvalConfig | None = None):
        self.store = store
        self.config = config or EvalConfig()

    def build_eval_set(self) -> list[dict[str, str]]:
        """Build evaluation set from corrections with known-good answers."""
        corrections = self.store.get_feedback(
            feedback_type=FeedbackType.CORRECTION,
            limit=self.config.eval_set_size * 2,
        )

        eval_set: list[dict[str, str]] = []
        for fb in corrections:
            if fb.query and fb.correction and len(fb.correction) >= 20:
                eval_set.append({
                    "query": fb.query,
                    "reference": fb.correction,
                    "original": fb.response,
                })
                if len(eval_set) >= self.config.eval_set_size:
                    break

        return eval_set

    def evaluate_responses(
        self,
        eval_set: list[dict[str, str]],
        generated_responses: list[str],
        model_version: str = "",
    ) -> EvalResult:
        """Evaluate generated responses against reference answers.

        Args:
            eval_set: List of {query, reference, original} dicts
            generated_responses: Model's actual outputs for each query
            model_version: Version identifier for tracking
        """
        if len(eval_set) != len(generated_responses):
            raise ValueError(
                f"Eval set ({len(eval_set)}) and responses ({len(generated_responses)}) "
                "must be the same length"
            )

        if not eval_set:
            return EvalResult(model_version=model_version)

        exact_matches = 0
        sim_scores: list[float] = []
        length_scores: list[float] = []
        format_scores: list[float] = []
        per_example: list[dict[str, Any]] = []

        for i, (ex, response) in enumerate(zip(eval_set, generated_responses)):
            reference = ex["reference"]

            # Exact match (after normalization)
            is_exact = self._normalize(response) == self._normalize(reference)
            if is_exact:
                exact_matches += 1

            # Semantic similarity (word overlap approximation)
            sim = self._compute_similarity(response, reference)
            sim_scores.append(sim)

            # Length accuracy
            len_score = self._compute_length_accuracy(response, reference)
            length_scores.append(len_score)

            # Format accuracy
            fmt_score = self._compute_format_accuracy(response, reference)
            format_scores.append(fmt_score)

            per_example.append({
                "index": i,
                "exact_match": is_exact,
                "similarity": round(sim, 4),
                "length_accuracy": round(len_score, 4),
                "format_accuracy": round(fmt_score, 4),
            })

        n = len(eval_set)
        result = EvalResult(
            model_version=model_version,
            total_examples=n,
            exact_match_rate=exact_matches / n,
            semantic_similarity=sum(sim_scores) / n,
            length_accuracy=sum(length_scores) / n,
            format_accuracy=sum(format_scores) / n,
            per_example=per_example,
        )

        # Weighted overall
        result.overall_score = (
            self.config.exact_match_weight * result.exact_match_rate
            + self.config.semantic_weight * result.semantic_similarity
            + self.config.length_weight * result.length_accuracy
            + self.config.format_weight * result.format_accuracy
        )

        return result

    def compare_versions(
        self,
        eval_set: list[dict[str, str]],
        responses_a: list[str],
        responses_b: list[str],
        version_a: str = "",
        version_b: str = "",
    ) -> dict[str, Any]:
        """Compare two model versions on the same eval set."""
        result_a = self.evaluate_responses(eval_set, responses_a, version_a)
        result_b = self.evaluate_responses(eval_set, responses_b, version_b)

        improvement = result_b.overall_score - result_a.overall_score

        return {
            "version_a": {"version": version_a, **result_a.to_dict()},
            "version_b": {"version": version_b, **result_b.to_dict()},
            "improvement": round(improvement, 4),
            "winner": version_b if improvement > 0 else version_a,
            "significant": abs(improvement) > 0.02,
        }

    def _normalize(self, text: str) -> str:
        """Normalize text for comparison."""
        return " ".join(text.lower().split())

    def _compute_similarity(self, generated: str, reference: str) -> float:
        """Compute word-overlap similarity (F1-like)."""
        gen_words = set(self._normalize(generated).split())
        ref_words = set(self._normalize(reference).split())

        if not ref_words:
            return 1.0 if not gen_words else 0.0

        overlap = len(gen_words & ref_words)
        precision = overlap / len(gen_words) if gen_words else 0.0
        recall = overlap / len(ref_words)

        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    def _compute_length_accuracy(self, generated: str, reference: str) -> float:
        """Score how close the lengths are (1.0 = same length)."""
        gen_len = len(generated)
        ref_len = len(reference)
        if ref_len == 0:
            return 1.0 if gen_len == 0 else 0.0
        ratio = min(gen_len, ref_len) / max(gen_len, ref_len)
        return ratio

    def _compute_format_accuracy(self, generated: str, reference: str) -> float:
        """Score format similarity (code blocks, lists, headers)."""
        score = 0.0
        checks = 0

        # Code blocks
        gen_code = "```" in generated
        ref_code = "```" in reference
        if gen_code == ref_code:
            score += 1.0
        checks += 1

        # Bullet lists
        gen_bullets = "\n- " in generated or "\n* " in generated
        ref_bullets = "\n- " in reference or "\n* " in reference
        if gen_bullets == ref_bullets:
            score += 1.0
        checks += 1

        # Headers
        gen_headers = "\n#" in generated
        ref_headers = "\n#" in reference
        if gen_headers == ref_headers:
            score += 1.0
        checks += 1

        # Line count similarity
        gen_lines = generated.count("\n")
        ref_lines = reference.count("\n")
        if max(gen_lines, ref_lines) > 0:
            line_ratio = min(gen_lines, ref_lines) / max(gen_lines, ref_lines)
        else:
            line_ratio = 1.0
        score += line_ratio
        checks += 1

        return score / checks if checks > 0 else 0.0
