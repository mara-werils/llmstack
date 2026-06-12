"""Prompt optimization — learns optimal prompts from feedback patterns.

Analyzes which system prompts, templates, and phrasing patterns produce
the best user satisfaction. Automatically evolves prompts over time
based on collected feedback signals.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llmstack.learn.feedback import Feedback, FeedbackType
from llmstack.learn.store import FeedbackStore

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path.home() / ".llmstack" / "prompts"


@dataclass
class PromptVariant:
    """A versioned prompt template with performance metrics."""

    id: str
    name: str
    template: str
    version: int = 1
    created_at: float = field(default_factory=time.time)
    total_uses: int = 0
    positive_feedback: int = 0
    negative_feedback: int = 0
    corrections: int = 0
    avg_quality: float = 0.0
    is_active: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def satisfaction_rate(self) -> float:
        total = self.positive_feedback + self.negative_feedback
        if total == 0:
            return 0.5  # no data = neutral
        return self.positive_feedback / total

    @property
    def correction_rate(self) -> float:
        if self.total_uses == 0:
            return 0.0
        return self.corrections / self.total_uses

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "template": self.template,
            "version": self.version,
            "created_at": self.created_at,
            "total_uses": self.total_uses,
            "positive_feedback": self.positive_feedback,
            "negative_feedback": self.negative_feedback,
            "corrections": self.corrections,
            "avg_quality": round(self.avg_quality, 4),
            "satisfaction_rate": round(self.satisfaction_rate, 4),
            "correction_rate": round(self.correction_rate, 4),
            "is_active": self.is_active,
            "metadata": self.metadata,
        }


@dataclass
class OptimizationResult:
    """Result of a prompt optimization cycle."""

    original: PromptVariant
    optimized: PromptVariant | None = None
    improvements: list[str] = field(default_factory=list)
    patterns_found: list[str] = field(default_factory=list)
    confidence: float = 0.0

    @property
    def was_optimized(self) -> bool:
        """Return True if an optimized variant was produced."""
        return self.optimized is not None

    @property
    def improvement_count(self) -> int:
        """Return the number of suggested improvements."""
        return len(self.improvements)


class PromptOptimizer:
    """Optimizes prompts based on feedback patterns.

    Tracks which prompt variants perform best, identifies patterns
    in corrections, and suggests or applies improvements.
    """

    def __init__(self, store: FeedbackStore, prompts_dir: Path | None = None):
        self.store = store
        self.prompts_dir = prompts_dir or PROMPTS_DIR
        self.prompts_dir.mkdir(parents=True, exist_ok=True)
        self._variants: dict[str, PromptVariant] = {}
        self._load_variants()

    def register_prompt(
        self,
        name: str,
        template: str,
        metadata: dict | None = None,
    ) -> PromptVariant:
        """Register a new prompt template for tracking."""
        prompt_id = hashlib.md5(f"{name}:{template}".encode()).hexdigest()[:10]

        variant = PromptVariant(
            id=prompt_id,
            name=name,
            template=template,
            metadata=metadata or {},
        )
        self._variants[prompt_id] = variant
        self._save_variant(variant)
        return variant

    def record_use(
        self,
        prompt_id: str,
        feedback: Feedback | None = None,
        quality_score: float = 0.0,
    ) -> None:
        """Record a usage of a prompt with optional feedback."""
        variant = self._variants.get(prompt_id)
        if not variant:
            return

        variant.total_uses += 1

        if feedback:
            if feedback.is_positive:
                variant.positive_feedback += 1
            elif feedback.is_negative:
                variant.negative_feedback += 1
            if feedback.has_correction:
                variant.corrections += 1

        if quality_score > 0:
            # Running average
            n = variant.total_uses
            variant.avg_quality = (variant.avg_quality * (n - 1) + quality_score) / n

        self._save_variant(variant)

    def analyze_patterns(self, name: str) -> list[str]:
        """Analyze feedback patterns for a named prompt.

        Returns identified patterns like:
        - "Users frequently correct code formatting"
        - "Responses are too verbose for simple questions"
        """
        variants = [v for v in self._variants.values() if v.name == name]
        if not variants:
            return []

        # Get corrections for this prompt's command context
        corrections = self.store.get_feedback(
            feedback_type=FeedbackType.CORRECTION,
            limit=100,
        )

        patterns: list[str] = []

        if not corrections:
            return patterns

        # Pattern: frequent shortening (users want conciseness)
        shortened = sum(
            1 for f in corrections if f.correction and len(f.correction) < len(f.response) * 0.7
        )
        if shortened > len(corrections) * 0.3:
            patterns.append("Users frequently shorten responses — model is too verbose")

        # Pattern: frequent lengthening (users want detail)
        lengthened = sum(
            1 for f in corrections if f.correction and len(f.correction) > len(f.response) * 1.5
        )
        if lengthened > len(corrections) * 0.3:
            patterns.append("Users frequently expand responses — model lacks detail")

        # Pattern: code formatting corrections
        code_corrections = sum(
            1
            for f in corrections
            if f.correction and "```" in f.correction and "```" not in f.response
        )
        if code_corrections > 3:
            patterns.append("Users add code blocks — model should format code properly")

        # Pattern: removing hedging language
        hedging = sum(
            1
            for f in corrections
            if f.response
            and any(
                h in f.response.lower()
                for h in ["i think", "perhaps", "maybe", "it seems", "might be"]
            )
            and f.correction
            and not any(
                h in f.correction.lower()
                for h in ["i think", "perhaps", "maybe", "it seems", "might be"]
            )
        )
        if hedging > 3:
            patterns.append("Users remove hedging language — model should be more direct")

        # Pattern: adding structure (lists, headers)
        structure_added = sum(
            1
            for f in corrections
            if f.correction
            and (
                f.correction.count("\n- ") > f.response.count("\n- ") + 2
                or f.correction.count("\n#") > f.response.count("\n#")
            )
        )
        if structure_added > 3:
            patterns.append("Users add structure (lists/headers) — model should organize better")

        return patterns

    def suggest_improvements(self, name: str) -> list[str]:
        """Suggest prompt improvements based on patterns."""
        patterns = self.analyze_patterns(name)
        suggestions: list[str] = []

        for pattern in patterns:
            if "too verbose" in pattern:
                suggestions.append("Add instruction: 'Be concise. Avoid unnecessary explanation.'")
            elif "lacks detail" in pattern:
                suggestions.append(
                    "Add instruction: 'Provide detailed explanations with examples.'"
                )
            elif "format code" in pattern:
                suggestions.append("Add instruction: 'Always wrap code in ```language blocks.'")
            elif "hedging" in pattern:
                suggestions.append(
                    "Add instruction: 'Be direct and confident. Avoid hedging words.'"
                )
            elif "organize" in pattern:
                suggestions.append(
                    "Add instruction: 'Use bullet points and headers for complex answers.'"
                )

        return suggestions

    def get_best_variant(self, name: str) -> PromptVariant | None:
        """Get the best-performing variant of a named prompt."""
        variants = [v for v in self._variants.values() if v.name == name and v.total_uses >= 5]
        if not variants:
            return None
        return max(variants, key=lambda v: v.satisfaction_rate)

    def get_variants(self, name: str | None = None) -> list[PromptVariant]:
        """List all variants, optionally filtered by name."""
        variants = list(self._variants.values())
        if name:
            variants = [v for v in variants if v.name == name]
        return sorted(variants, key=lambda v: v.satisfaction_rate, reverse=True)

    def _load_variants(self) -> None:
        """Load saved prompt variants from disk."""
        for path in self.prompts_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text())
                variant = PromptVariant(
                    id=data["id"],
                    name=data["name"],
                    template=data["template"],
                    version=data.get("version", 1),
                    created_at=data.get("created_at", 0),
                    total_uses=data.get("total_uses", 0),
                    positive_feedback=data.get("positive_feedback", 0),
                    negative_feedback=data.get("negative_feedback", 0),
                    corrections=data.get("corrections", 0),
                    avg_quality=data.get("avg_quality", 0.0),
                    is_active=data.get("is_active", True),
                    metadata=data.get("metadata", {}),
                )
                self._variants[variant.id] = variant
            except (json.JSONDecodeError, KeyError) as exc:
                logger.debug("Failed to load prompt variant %s: %s", path, exc)

    def _save_variant(self, variant: PromptVariant) -> None:
        """Save a prompt variant to disk."""
        path = self.prompts_dir / f"{variant.id}.json"
        path.write_text(json.dumps(variant.to_dict(), indent=2))
