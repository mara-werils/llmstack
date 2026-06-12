"""Code pattern learning — learns coding style and conventions from feedback.

Extracts patterns from code-related corrections to build a style guide
that the model can follow. Learns:
- Naming conventions (camelCase, snake_case, etc.)
- Comment style preferences
- Import ordering
- Error handling patterns
- Testing patterns
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llmstack.learn.feedback import Feedback, FeedbackType
from llmstack.learn.store import FeedbackStore

logger = logging.getLogger(__name__)

PATTERNS_PATH = Path.home() / ".llmstack" / "code_patterns.json"


@dataclass
class CodePattern:
    """A learned code pattern/convention."""

    name: str
    description: str
    examples: list[str] = field(default_factory=list)
    counter_examples: list[str] = field(default_factory=list)
    confidence: float = 0.0
    occurrences: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "examples": self.examples[:5],
            "counter_examples": self.counter_examples[:3],
            "confidence": round(self.confidence, 3),
            "occurrences": self.occurrences,
        }


@dataclass
class CodeStyleProfile:
    """Complete code style profile learned from corrections."""

    naming: dict[str, float] = field(default_factory=dict)
    patterns: list[CodePattern] = field(default_factory=list)
    language_preferences: dict[str, Any] = field(default_factory=dict)
    last_updated: float = 0.0
    total_code_corrections: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "naming": self.naming,
            "patterns": [p.to_dict() for p in self.patterns],
            "language_preferences": self.language_preferences,
            "last_updated": self.last_updated,
            "total_code_corrections": self.total_code_corrections,
        }

    def to_style_guide(self) -> str:
        """Convert learned patterns to a style guide string."""
        lines: list[str] = []

        # Naming convention
        if self.naming:
            dominant = max(self.naming, key=self.naming.get)
            if self.naming[dominant] > 0.6:
                lines.append(f"Use {dominant} naming convention.")

        # High-confidence patterns
        for pattern in self.patterns:
            if pattern.confidence > 0.7:
                lines.append(pattern.description)

        return " ".join(lines)


class PatternLearner:
    """Learns code patterns and conventions from corrections.

    Analyzes code-related feedback to identify consistent patterns
    in how the user prefers code to be written.
    """

    def __init__(self, store: FeedbackStore, patterns_path: Path | None = None):
        self.store = store
        self.patterns_path = patterns_path or PATTERNS_PATH
        self.profile = self._load()

    @property
    def pattern_count(self) -> int:
        """Return the number of learned code patterns."""
        return len(self.profile.patterns)

    @property
    def high_confidence_patterns(self) -> list[CodePattern]:
        """Return patterns with confidence above 0.7."""
        return [p for p in self.profile.patterns if p.confidence > 0.7]

    def learn_from_correction(self, original: str, correction: str) -> None:
        """Learn patterns from a code correction pair."""
        if not self._is_code_content(original) and not self._is_code_content(correction):
            return

        self.profile.total_code_corrections += 1
        self._learn_naming(original, correction)
        self._learn_formatting(original, correction)
        self._learn_error_handling(original, correction)
        self._learn_imports(original, correction)
        self.profile.last_updated = time.time()
        self._save()

    def learn_from_feedback(self, feedback: Feedback) -> None:
        """Learn from a feedback event containing code."""
        if feedback.feedback_type in (FeedbackType.CORRECTION, FeedbackType.EDIT):
            if feedback.correction:
                self.learn_from_correction(feedback.response, feedback.correction)

    def rebuild_from_history(self, limit: int = 500) -> None:
        """Rebuild patterns from all historical code corrections."""
        self.profile = CodeStyleProfile()

        corrections = self.store.get_feedback(feedback_type=FeedbackType.CORRECTION, limit=limit)
        edits = self.store.get_feedback(feedback_type=FeedbackType.EDIT, limit=limit)

        for fb in corrections + edits:
            if fb.correction:
                self.learn_from_correction(fb.response, fb.correction)

        self._save()
        logger.info(
            "Rebuilt code patterns from %d corrections",
            self.profile.total_code_corrections,
        )

    def get_style_guide(self) -> str:
        """Get the current style guide as text."""
        return self.profile.to_style_guide()

    def get_profile(self) -> dict[str, Any]:
        """Get the complete code style profile."""
        return self.profile.to_dict()

    def _learn_naming(self, original: str, correction: str) -> None:
        """Learn naming convention preferences."""
        self._extract_identifiers(original)
        corr_identifiers = self._extract_identifiers(correction)

        if not corr_identifiers:
            return

        # Detect naming style of correction
        styles = Counter()
        for ident in corr_identifiers:
            style = self._classify_naming(ident)
            if style:
                styles[style] += 1

        if styles:
            total = sum(styles.values())
            for style, count in styles.items():
                current = self.profile.naming.get(style, 0.5)
                weight = count / total
                # Exponential moving average
                self.profile.naming[style] = 0.85 * current + 0.15 * weight

    def _learn_formatting(self, original: str, correction: str) -> None:
        """Learn formatting preferences."""
        # Trailing newline preference
        if correction.endswith("\n") and not original.endswith("\n"):
            self._update_pattern(
                "trailing_newline",
                "Add trailing newline to code files.",
                correction[-20:],
            )

        # Blank lines between functions
        orig_blank = len(re.findall(r"\n\n\n", original))
        corr_blank = len(re.findall(r"\n\n\n", correction))
        if corr_blank > orig_blank:
            self._update_pattern(
                "blank_lines",
                "Use blank lines to separate code blocks.",
                "",
            )

        # Line length
        orig_long = sum(1 for line in original.split("\n") if len(line) > 100)
        corr_long = sum(1 for line in correction.split("\n") if len(line) > 100)
        if orig_long > corr_long and orig_long > 2:
            self._update_pattern(
                "line_length",
                "Keep lines under 100 characters.",
                "",
            )

    def _learn_error_handling(self, original: str, correction: str) -> None:
        """Learn error handling preferences."""
        # Try/except patterns
        orig_try = original.count("try:")
        corr_try = correction.count("try:")
        if corr_try > orig_try:
            self._update_pattern(
                "error_handling",
                "Wrap risky operations in try/except blocks.",
                "",
            )

        # Specific vs bare except
        if "except:" in original and "except " in correction:
            self._update_pattern(
                "specific_exceptions",
                "Use specific exception types, not bare except.",
                "",
            )

    def _learn_imports(self, original: str, correction: str) -> None:
        """Learn import style preferences."""
        orig_imports = [
            line
            for line in original.split("\n")
            if line.startswith("import ") or line.startswith("from ")
        ]
        corr_imports = [
            line
            for line in correction.split("\n")
            if line.startswith("import ") or line.startswith("from ")
        ]

        if not corr_imports:
            return

        # Sorted imports preference
        if corr_imports == sorted(corr_imports) and orig_imports != sorted(orig_imports):
            self._update_pattern(
                "sorted_imports",
                "Keep imports sorted alphabetically.",
                "",
            )

        # from vs direct import
        corr_from = sum(1 for i in corr_imports if i.startswith("from "))
        if len(corr_imports) > 2 and corr_from > len(corr_imports) * 0.7:
            self._update_pattern(
                "from_imports",
                "Prefer 'from x import y' over 'import x'.",
                "",
            )

    def _update_pattern(self, name: str, description: str, example: str) -> None:
        """Update or create a pattern."""
        for pattern in self.profile.patterns:
            if pattern.name == name:
                pattern.occurrences += 1
                pattern.confidence = min(1.0, pattern.occurrences / 10)
                if example and example not in pattern.examples:
                    pattern.examples.append(example)
                    pattern.examples = pattern.examples[-5:]
                return

        self.profile.patterns.append(
            CodePattern(
                name=name,
                description=description,
                examples=[example] if example else [],
                occurrences=1,
                confidence=0.1,
            )
        )

    def _is_code_content(self, text: str) -> bool:
        """Check if text likely contains code."""
        code_indicators = [
            "def ",
            "class ",
            "import ",
            "function ",
            "const ",
            "let ",
            "var ",
            "return ",
            "if (",
            "for ",
            "```",
            "    ",
            "\t",
        ]
        return any(indicator in text for indicator in code_indicators)

    def _extract_identifiers(self, text: str) -> list[str]:
        """Extract likely identifiers from code."""
        # Match function/variable names
        patterns = [
            r"def\s+(\w+)",
            r"class\s+(\w+)",
            r"(\w+)\s*=",
            r"function\s+(\w+)",
            r"(?:const|let|var)\s+(\w+)",
        ]
        identifiers: list[str] = []
        for pattern in patterns:
            identifiers.extend(re.findall(pattern, text))
        return [i for i in identifiers if len(i) > 2 and not i.isupper()]

    def _classify_naming(self, identifier: str) -> str | None:
        """Classify an identifier's naming convention."""
        if "_" in identifier and identifier.islower():
            return "snake_case"
        if identifier[0].islower() and any(c.isupper() for c in identifier[1:]):
            return "camelCase"
        if identifier[0].isupper() and any(c.isupper() for c in identifier[1:]):
            return "PascalCase"
        if identifier.isupper() and "_" in identifier:
            return "UPPER_SNAKE"
        return None

    def _load(self) -> CodeStyleProfile:
        """Load saved patterns."""
        if not self.patterns_path.exists():
            return CodeStyleProfile()
        try:
            data = json.loads(self.patterns_path.read_text())
            profile = CodeStyleProfile()
            profile.naming = data.get("naming", {})
            profile.language_preferences = data.get("language_preferences", {})
            profile.last_updated = data.get("last_updated", 0)
            profile.total_code_corrections = data.get("total_code_corrections", 0)
            for p in data.get("patterns", []):
                profile.patterns.append(
                    CodePattern(
                        name=p["name"],
                        description=p["description"],
                        examples=p.get("examples", []),
                        counter_examples=p.get("counter_examples", []),
                        confidence=p.get("confidence", 0),
                        occurrences=p.get("occurrences", 0),
                    )
                )
            return profile
        except (json.JSONDecodeError, KeyError):
            return CodeStyleProfile()

    def _save(self) -> None:
        """Save patterns to disk."""
        self.patterns_path.parent.mkdir(parents=True, exist_ok=True)
        self.patterns_path.write_text(json.dumps(self.profile.to_dict(), indent=2))
