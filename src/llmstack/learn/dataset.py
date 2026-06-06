"""Dataset generation — automatically creates training data from feedback.

Transforms user corrections, preferences, and edits into chat-format
training examples suitable for fine-tuning. Supports multiple strategies:
- Direct correction pairs (bad response → good response)
- DPO pairs (preferred vs rejected)
- Filtered positive examples (high-rated responses)
- Synthetic augmentation (paraphrasing corrections)
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from llmstack.learn.feedback import Feedback, FeedbackType
from llmstack.learn.store import FeedbackStore

logger = logging.getLogger(__name__)


class DatasetStrategy(str, Enum):
    """Strategy for generating training data from feedback."""

    SFT = "sft"  # Supervised fine-tuning on corrections
    DPO = "dpo"  # Direct preference optimization pairs
    POSITIVE = "positive"  # Only high-quality approved responses
    MIXED = "mixed"  # Combine all strategies


@dataclass
class TrainingExample:
    """A single training example in chat format."""

    messages: list[dict[str, str]]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"messages": self.messages, "metadata": self.metadata}


@dataclass
class DPOExample:
    """A preference pair for DPO training."""

    prompt: str
    chosen: str
    rejected: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "chosen": self.chosen,
            "rejected": self.rejected,
            "metadata": self.metadata,
        }


@dataclass
class GeneratedDataset:
    """A generated training dataset with metadata."""

    id: str = ""
    timestamp: float = field(default_factory=time.time)
    strategy: DatasetStrategy = DatasetStrategy.SFT
    sft_examples: list[TrainingExample] = field(default_factory=list)
    dpo_examples: list[DPOExample] = field(default_factory=list)
    feedback_ids: list[str] = field(default_factory=list)
    stats: dict[str, int] = field(default_factory=dict)

    @property
    def sft_count(self) -> int:
        """Return the number of SFT training examples."""
        return len(self.sft_examples)

    @property
    def dpo_count(self) -> int:
        """Return the number of DPO preference pairs."""
        return len(self.dpo_examples)

    @property
    def is_empty(self) -> bool:
        """Return True when no training examples have been generated."""
        return not self.sft_examples and not self.dpo_examples

    @property
    def total_examples(self) -> int:
        return len(self.sft_examples) + len(self.dpo_examples)

    def save(self, output_dir: Path) -> Path:
        """Save dataset to disk in JSONL format."""
        output_dir.mkdir(parents=True, exist_ok=True)
        dataset_path = output_dir / f"dataset_{self.id}.jsonl"

        with open(dataset_path, "w") as f:
            for ex in self.sft_examples:
                f.write(json.dumps(ex.to_dict()) + "\n")
            for ex in self.dpo_examples:
                f.write(json.dumps(ex.to_dict()) + "\n")

        # Save metadata
        meta_path = output_dir / f"dataset_{self.id}_meta.json"
        meta_path.write_text(
            json.dumps(
                {
                    "id": self.id,
                    "timestamp": self.timestamp,
                    "strategy": self.strategy.value,
                    "sft_count": len(self.sft_examples),
                    "dpo_count": len(self.dpo_examples),
                    "total": self.total_examples,
                    "feedback_count": len(self.feedback_ids),
                    "stats": self.stats,
                },
                indent=2,
            )
        )

        return dataset_path


class DatasetGenerator:
    """Generates training datasets from collected feedback.

    Supports multiple generation strategies and applies quality
    filters to ensure only high-signal data enters training.
    """

    def __init__(
        self,
        store: FeedbackStore,
        min_query_length: int = 5,
        min_response_length: int = 20,
        dedup: bool = True,
    ):
        self.store = store
        self.min_query_length = min_query_length
        self.min_response_length = min_response_length
        self.dedup = dedup

    def generate(
        self,
        strategy: DatasetStrategy = DatasetStrategy.MIXED,
        model: str | None = None,
        since: float | None = None,
        max_examples: int = 5000,
    ) -> GeneratedDataset:
        """Generate a training dataset from unused feedback."""
        feedback = self.store.get_feedback(
            model=model,
            since=since,
            unused_only=True,
            limit=max_examples * 2,
        )

        if not feedback:
            return GeneratedDataset(
                id=self._make_id([]),
                strategy=strategy,
                stats={"source_feedback": 0, "filtered": 0},
            )

        dataset = GeneratedDataset(
            id=self._make_id(feedback),
            strategy=strategy,
            feedback_ids=[f.id for f in feedback],
        )

        if strategy in (DatasetStrategy.SFT, DatasetStrategy.MIXED):
            dataset.sft_examples = self._generate_sft(feedback, max_examples)

        if strategy in (DatasetStrategy.DPO, DatasetStrategy.MIXED):
            dataset.dpo_examples = self._generate_dpo(feedback, max_examples)

        if strategy == DatasetStrategy.POSITIVE:
            dataset.sft_examples = self._generate_positive(feedback, max_examples)

        # Dedup
        if self.dedup:
            dataset.sft_examples = self._dedup_sft(dataset.sft_examples)
            dataset.dpo_examples = self._dedup_dpo(dataset.dpo_examples)

        dataset.stats = {
            "source_feedback": len(feedback),
            "sft_generated": len(dataset.sft_examples),
            "dpo_generated": len(dataset.dpo_examples),
            "total": dataset.total_examples,
        }

        return dataset

    def _generate_sft(self, feedback: list[Feedback], max_examples: int) -> list[TrainingExample]:
        """Generate SFT examples from corrections and edits."""
        examples: list[TrainingExample] = []

        for fb in feedback:
            if len(examples) >= max_examples:
                break

            if not self._passes_quality_filter(fb):
                continue

            # Corrections: use the user's correction as the target
            if fb.feedback_type == FeedbackType.CORRECTION and fb.correction:
                examples.append(
                    TrainingExample(
                        messages=[
                            {"role": "user", "content": fb.query},
                            {"role": "assistant", "content": fb.correction},
                        ],
                        metadata={"source": "correction", "feedback_id": fb.id},
                    )
                )

            # Edits: apply the diff to get the corrected response
            elif fb.feedback_type == FeedbackType.EDIT and fb.correction:
                examples.append(
                    TrainingExample(
                        messages=[
                            {"role": "user", "content": fb.query},
                            {"role": "assistant", "content": fb.correction},
                        ],
                        metadata={"source": "edit", "feedback_id": fb.id},
                    )
                )

            # Thumbs up: the original response is good training data
            elif fb.feedback_type == FeedbackType.THUMBS_UP:
                examples.append(
                    TrainingExample(
                        messages=[
                            {"role": "user", "content": fb.query},
                            {"role": "assistant", "content": fb.response},
                        ],
                        metadata={"source": "approved", "feedback_id": fb.id},
                    )
                )

        return examples

    def _generate_dpo(self, feedback: list[Feedback], max_examples: int) -> list[DPOExample]:
        """Generate DPO preference pairs."""
        examples: list[DPOExample] = []

        for fb in feedback:
            if len(examples) >= max_examples:
                break

            if not self._passes_quality_filter(fb):
                continue

            # Corrections give us a clear preference pair
            if fb.feedback_type == FeedbackType.CORRECTION and fb.correction:
                examples.append(
                    DPOExample(
                        prompt=fb.query,
                        chosen=fb.correction,
                        rejected=fb.response,
                        metadata={"source": "correction", "feedback_id": fb.id},
                    )
                )

            # Explicit preferences (A/B testing)
            elif fb.feedback_type == FeedbackType.PREFERENCE:
                if fb.correction and fb.preferred_over:
                    examples.append(
                        DPOExample(
                            prompt=fb.query,
                            chosen=fb.correction,
                            rejected=fb.preferred_over,
                            metadata={"source": "preference", "feedback_id": fb.id},
                        )
                    )

            # Edits imply the edited version is preferred
            elif fb.feedback_type == FeedbackType.EDIT and fb.correction:
                examples.append(
                    DPOExample(
                        prompt=fb.query,
                        chosen=fb.correction,
                        rejected=fb.response,
                        metadata={"source": "edit", "feedback_id": fb.id},
                    )
                )

        return examples

    def _generate_positive(
        self, feedback: list[Feedback], max_examples: int
    ) -> list[TrainingExample]:
        """Generate examples from only positively-rated responses."""
        examples: list[TrainingExample] = []

        for fb in feedback:
            if len(examples) >= max_examples:
                break

            if not self._passes_quality_filter(fb):
                continue

            if fb.is_positive:
                examples.append(
                    TrainingExample(
                        messages=[
                            {"role": "user", "content": fb.query},
                            {"role": "assistant", "content": fb.response},
                        ],
                        metadata={"source": "positive", "feedback_id": fb.id},
                    )
                )

        return examples

    def _passes_quality_filter(self, fb: Feedback) -> bool:
        """Apply quality filters to a feedback item."""
        if len(fb.query) < self.min_query_length:
            return False
        if fb.feedback_type in (FeedbackType.CORRECTION, FeedbackType.EDIT):
            target = fb.correction or fb.response
        else:
            target = fb.response
        if len(target) < self.min_response_length:
            return False
        return True

    def _dedup_sft(self, examples: list[TrainingExample]) -> list[TrainingExample]:
        """Remove duplicate SFT examples based on content hash."""
        seen: set[str] = set()
        unique: list[TrainingExample] = []
        for ex in examples:
            content = json.dumps(ex.messages, sort_keys=True)
            h = hashlib.md5(content.encode()).hexdigest()
            if h not in seen:
                seen.add(h)
                unique.append(ex)
        return unique

    def _dedup_dpo(self, examples: list[DPOExample]) -> list[DPOExample]:
        """Remove duplicate DPO examples."""
        seen: set[str] = set()
        unique: list[DPOExample] = []
        for ex in examples:
            content = f"{ex.prompt}|{ex.chosen}|{ex.rejected}"
            h = hashlib.md5(content.encode()).hexdigest()
            if h not in seen:
                seen.add(h)
                unique.append(ex)
        return unique

    def _make_id(self, feedback: list[Feedback]) -> str:
        """Generate a deterministic dataset ID."""
        ids = sorted(f.id for f in feedback)
        content = "|".join(ids) if ids else str(time.time())
        return hashlib.sha256(content.encode()).hexdigest()[:12]
