"""Synthetic data augmentation — expands training data from limited feedback.

When feedback is scarce, augments the dataset with synthetic variations:
- Paraphrases of corrections (same intent, different wording)
- Domain transfer (similar patterns in adjacent contexts)
- Difficulty scaling (simpler/harder versions of same task)
"""

from __future__ import annotations

import hashlib
import logging
import random
import re
from dataclasses import dataclass

from llmstack.learn.dataset import TrainingExample

logger = logging.getLogger(__name__)


@dataclass
class AugmentationConfig:
    """Configuration for synthetic data augmentation."""

    # Maximum augmentation factor (e.g., 3x = triple the dataset)
    max_factor: float = 3.0

    # Minimum examples before augmentation kicks in
    min_examples_threshold: int = 5

    # Augmentation strategies to use
    paraphrase: bool = True
    word_swap: bool = True
    format_variation: bool = True
    instruction_variation: bool = True

    # Random seed for reproducibility
    seed: int = 42


class SyntheticAugmenter:
    """Augments training data with synthetic variations.

    Uses rule-based transformations (no external model required) to
    create variations of existing training examples, expanding sparse
    feedback into a more robust training set.
    """

    def __init__(self, config: AugmentationConfig | None = None):
        self.config = config or AugmentationConfig()
        self._rng = random.Random(self.config.seed)

    @property
    def enabled_strategies(self) -> list[str]:
        """Return the names of enabled augmentation strategies."""
        names: list[str] = []
        if self.config.paraphrase:
            names.append("paraphrase")
        if self.config.word_swap:
            names.append("word_swap")
        if self.config.format_variation:
            names.append("format_variation")
        if self.config.instruction_variation:
            names.append("instruction_variation")
        return names

    @property
    def strategy_count(self) -> int:
        """Return the number of enabled augmentation strategies."""
        return len(self.enabled_strategies)

    def augment(
        self,
        examples: list[TrainingExample],
        target_count: int | None = None,
    ) -> list[TrainingExample]:
        """Augment examples to reach target count.

        Returns original + augmented examples.
        """
        if len(examples) < self.config.min_examples_threshold:
            return examples

        if target_count is None:
            target_count = int(len(examples) * self.config.max_factor)

        target_count = min(target_count, int(len(examples) * self.config.max_factor))

        augmented: list[TrainingExample] = list(examples)
        attempts = 0
        max_attempts = target_count * 3

        while len(augmented) < target_count and attempts < max_attempts:
            source = self._rng.choice(examples)
            variant = self._create_variant(source)
            if variant and not self._is_duplicate(variant, augmented):
                augmented.append(variant)
            attempts += 1

        logger.info(
            "Augmented %d → %d examples (%d new)",
            len(examples),
            len(augmented),
            len(augmented) - len(examples),
        )
        return augmented

    def _create_variant(self, example: TrainingExample) -> TrainingExample | None:
        """Create a single variant of an example."""
        strategies = []
        if self.config.paraphrase:
            strategies.append(self._paraphrase)
        if self.config.word_swap:
            strategies.append(self._word_swap)
        if self.config.format_variation:
            strategies.append(self._format_variation)
        if self.config.instruction_variation:
            strategies.append(self._instruction_variation)

        if not strategies:
            return None

        strategy = self._rng.choice(strategies)
        return strategy(example)

    def _paraphrase(self, example: TrainingExample) -> TrainingExample | None:
        """Create a paraphrase by restructuring the query."""
        if len(example.messages) < 2:
            return None

        query = example.messages[0]["content"]
        response = example.messages[1]["content"]

        # Simple paraphrase patterns
        transformations = [
            (r"^how (do I|to|can I) ", lambda m: "What's the way to "),
            (r"^what is ", lambda m: "Can you explain "),
            (r"^can you ", lambda m: "Please "),
            (r"^explain ", lambda m: "Help me understand "),
            (r"^write ", lambda m: "Create "),
            (r"^show me ", lambda m: "Give an example of "),
            (r"^why does ", lambda m: "What's the reason "),
            (r"^is there ", lambda m: "Does there exist "),
        ]

        new_query = query
        self._rng.shuffle(transformations)
        for pattern, replacement in transformations:
            match = re.match(pattern, query, re.IGNORECASE)
            if match:
                new_query = re.sub(pattern, replacement(match), query, count=1, flags=re.IGNORECASE)
                break
        else:
            # No pattern matched — add a prefix/suffix variation
            prefixes = ["I need help: ", "Quick question — ", ""]
            suffixes = ["", " (concise please)", " — thanks"]
            new_query = self._rng.choice(prefixes) + query + self._rng.choice(suffixes)

        if new_query == query:
            return None

        return TrainingExample(
            messages=[
                {"role": "user", "content": new_query},
                {"role": "assistant", "content": response},
            ],
            metadata={**example.metadata, "augmented": True, "strategy": "paraphrase"},
        )

    def _word_swap(self, example: TrainingExample) -> TrainingExample | None:
        """Swap synonyms in the query."""
        if len(example.messages) < 2:
            return None

        query = example.messages[0]["content"]
        response = example.messages[1]["content"]

        synonyms = {
            "function": ["method", "procedure", "routine"],
            "variable": ["parameter", "value", "field"],
            "error": ["issue", "bug", "problem"],
            "fix": ["resolve", "solve", "address"],
            "create": ["make", "build", "generate"],
            "check": ["verify", "validate", "test"],
            "fast": ["quick", "efficient", "performant"],
            "simple": ["basic", "straightforward", "easy"],
            "file": ["module", "script", "source"],
            "list": ["array", "collection", "sequence"],
        }

        new_query = query
        swapped = False
        for word, alternatives in synonyms.items():
            if word in new_query.lower():
                replacement = self._rng.choice(alternatives)
                new_query = re.sub(
                    rf"\b{word}\b", replacement, new_query, count=1, flags=re.IGNORECASE
                )
                swapped = True
                break

        if not swapped:
            return None

        return TrainingExample(
            messages=[
                {"role": "user", "content": new_query},
                {"role": "assistant", "content": response},
            ],
            metadata={**example.metadata, "augmented": True, "strategy": "word_swap"},
        )

    def _format_variation(self, example: TrainingExample) -> TrainingExample | None:
        """Vary the formatting of the response."""
        if len(example.messages) < 2:
            return None

        query = example.messages[0]["content"]
        response = example.messages[1]["content"]

        # Only apply to responses with some structure
        if len(response) < 50:
            return None

        new_response = response
        choice = self._rng.randint(0, 2)

        if choice == 0 and "```" not in response and "def " in response:
            # Wrap code in code block
            new_response = "```python\n" + response + "\n```"
        elif choice == 1 and "\n- " not in response and ". " in response:
            # Convert sentences to bullet points
            sentences = re.split(r"\.\s+", response)
            if len(sentences) >= 3:
                new_response = "\n".join(f"- {s.strip()}" for s in sentences if s.strip())
        elif choice == 2:
            # Add brief explanation prefix
            prefixes = [
                "Here's the solution:\n\n",
                "You can do this:\n\n",
                "",
            ]
            new_response = self._rng.choice(prefixes) + response

        if new_response == response:
            return None

        return TrainingExample(
            messages=[
                {"role": "user", "content": query},
                {"role": "assistant", "content": new_response},
            ],
            metadata={**example.metadata, "augmented": True, "strategy": "format_variation"},
        )

    def _instruction_variation(self, example: TrainingExample) -> TrainingExample | None:
        """Add instruction prefix to make the example more directive."""
        if len(example.messages) < 2:
            return None

        query = example.messages[0]["content"]
        response = example.messages[1]["content"]

        # Add context to query
        contexts = [
            f"In Python, {query.lower()}" if not query.lower().startswith("in ") else None,
            f"For a project I'm working on: {query}",
            f"I'm trying to {query.lower()}" if not query.lower().startswith("i") else None,
        ]

        valid_contexts = [c for c in contexts if c is not None]
        if not valid_contexts:  # pragma: no cover - the middle entry above is never None
            return None

        new_query = self._rng.choice(valid_contexts)

        return TrainingExample(
            messages=[
                {"role": "user", "content": new_query},
                {"role": "assistant", "content": response},
            ],
            metadata={**example.metadata, "augmented": True, "strategy": "instruction_variation"},
        )

    def _is_duplicate(self, candidate: TrainingExample, existing: list[TrainingExample]) -> bool:
        """Check if candidate is too similar to existing examples."""
        candidate_hash = hashlib.md5(candidate.messages[0]["content"].lower().encode()).hexdigest()

        for ex in existing[-50:]:  # only check recent to avoid O(n^2)
            ex_hash = hashlib.md5(ex.messages[0]["content"].lower().encode()).hexdigest()
            if candidate_hash == ex_hash:
                return True
        return False
