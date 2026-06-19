"""User preference learning — captures and applies personal style preferences.

Learns from corrections to build a preference profile:
- Response length preference (concise vs detailed)
- Formatting style (markdown, plain, code-heavy)
- Tone (formal, casual, technical)
- Domain expertise level (beginner, intermediate, expert)
- Language preferences
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llmstack.learn.feedback import Feedback, FeedbackType
from llmstack.learn.store import FeedbackStore

logger = logging.getLogger(__name__)

PREFERENCES_PATH = Path.home() / ".llmstack" / "preferences.json"


@dataclass
class LengthPreference:
    """Learned preference for response length."""

    avg_preferred_length: float = 500.0
    avg_rejected_length: float = 500.0
    samples: int = 0
    tendency: str = "neutral"  # concise, detailed, neutral

    def update(self, preferred_len: float, rejected_len: float) -> None:
        n = self.samples + 1
        self.avg_preferred_length = (self.avg_preferred_length * self.samples + preferred_len) / n
        self.avg_rejected_length = (self.avg_rejected_length * self.samples + rejected_len) / n
        self.samples = n
        self._update_tendency()

    @property
    def has_signal(self) -> bool:
        """Return True if enough samples exist to infer a tendency."""
        return self.samples >= 3

    @property
    def length_ratio(self) -> float:
        """Return the ratio of preferred to rejected response length."""
        if self.avg_rejected_length == 0:
            return 1.0
        return self.avg_preferred_length / self.avg_rejected_length

    def _update_tendency(self) -> None:
        if self.samples < 3:
            self.tendency = "neutral"
        elif self.avg_preferred_length < self.avg_rejected_length * 0.7:
            self.tendency = "concise"
        elif self.avg_preferred_length > self.avg_rejected_length * 1.3:
            self.tendency = "detailed"
        else:
            self.tendency = "neutral"


@dataclass
class FormatPreference:
    """Learned preference for response formatting."""

    prefers_code_blocks: float = 0.5  # 0-1 (higher = prefers)
    prefers_bullet_lists: float = 0.5
    prefers_headers: float = 0.5
    prefers_markdown: float = 0.5
    samples: int = 0

    def update(self, correction: str, original: str) -> None:
        self.samples += 1
        decay = 0.9  # exponential moving average

        # Code blocks
        corr_code = correction.count("```")
        orig_code = original.count("```")
        if corr_code > orig_code:
            self.prefers_code_blocks = decay * self.prefers_code_blocks + (1 - decay)
        elif corr_code < orig_code:
            self.prefers_code_blocks = decay * self.prefers_code_blocks

        # Bullet lists
        corr_bullets = correction.count("\n- ") + correction.count("\n* ")
        orig_bullets = original.count("\n- ") + original.count("\n* ")
        if corr_bullets > orig_bullets:
            self.prefers_bullet_lists = decay * self.prefers_bullet_lists + (1 - decay)
        elif corr_bullets < orig_bullets:
            self.prefers_bullet_lists = decay * self.prefers_bullet_lists

        # Headers
        corr_headers = correction.count("\n#")
        orig_headers = original.count("\n#")
        if corr_headers > orig_headers:
            self.prefers_headers = decay * self.prefers_headers + (1 - decay)
        elif corr_headers < orig_headers:
            self.prefers_headers = decay * self.prefers_headers

        # Markdown overall
        md_chars = ["**", "__", "`", "[", "]("]
        corr_md = sum(correction.count(c) for c in md_chars)
        orig_md = sum(original.count(c) for c in md_chars)
        if corr_md > orig_md:
            self.prefers_markdown = decay * self.prefers_markdown + (1 - decay)
        elif corr_md < orig_md:
            self.prefers_markdown = decay * self.prefers_markdown


@dataclass
class TonePreference:
    """Learned preference for response tone."""

    formality: float = 0.5  # 0=casual, 1=formal
    directness: float = 0.5  # 0=hedging, 1=direct
    technicality: float = 0.5  # 0=simple, 1=technical
    samples: int = 0

    def update(self, correction: str, original: str) -> None:
        self.samples += 1
        decay = 0.9

        # Directness: removal of hedging words
        hedges = ["i think", "perhaps", "maybe", "possibly", "might", "seems"]
        orig_hedges = sum(1 for h in hedges if h in original.lower())
        corr_hedges = sum(1 for h in hedges if h in correction.lower())
        if corr_hedges < orig_hedges:
            self.directness = decay * self.directness + (1 - decay)
        elif corr_hedges > orig_hedges:
            self.directness = decay * self.directness

        # Formality: contractions, slang
        contractions = ["don't", "won't", "can't", "it's", "that's", "I'm"]
        orig_casual = sum(1 for c in contractions if c in original)
        corr_casual = sum(1 for c in contractions if c in correction)
        if corr_casual < orig_casual:
            self.formality = decay * self.formality + (1 - decay)
        elif corr_casual > orig_casual:
            self.formality = decay * self.formality


@dataclass
class UserPreferences:
    """Complete user preference profile learned from feedback."""

    length: LengthPreference = field(default_factory=LengthPreference)
    formatting: FormatPreference = field(default_factory=FormatPreference)
    tone: TonePreference = field(default_factory=TonePreference)
    last_updated: float = 0.0
    total_signals: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "length": {
                "avg_preferred": round(self.length.avg_preferred_length, 0),
                "avg_rejected": round(self.length.avg_rejected_length, 0),
                "tendency": self.length.tendency,
                "samples": self.length.samples,
            },
            "formatting": {
                "code_blocks": round(self.formatting.prefers_code_blocks, 3),
                "bullet_lists": round(self.formatting.prefers_bullet_lists, 3),
                "headers": round(self.formatting.prefers_headers, 3),
                "markdown": round(self.formatting.prefers_markdown, 3),
                "samples": self.formatting.samples,
            },
            "tone": {
                "formality": round(self.tone.formality, 3),
                "directness": round(self.tone.directness, 3),
                "technicality": round(self.tone.technicality, 3),
                "samples": self.tone.samples,
            },
            "last_updated": self.last_updated,
            "total_signals": self.total_signals,
        }

    def to_system_prompt_additions(self) -> str:
        """Generate system prompt additions based on learned preferences."""
        additions: list[str] = []

        # Length preference
        if self.length.samples >= 5:
            if self.length.tendency == "concise":
                additions.append("Keep responses concise and to the point.")
            elif self.length.tendency == "detailed":
                additions.append("Provide detailed, thorough explanations.")

        # Formatting preferences
        if self.formatting.samples >= 5:
            if self.formatting.prefers_code_blocks > 0.7:
                additions.append("Always use code blocks for code snippets.")
            if self.formatting.prefers_bullet_lists > 0.7:
                additions.append("Use bullet points for lists and steps.")
            if self.formatting.prefers_headers > 0.7:
                additions.append("Use headers to organize longer responses.")

        # Tone preferences
        if self.tone.samples >= 5:
            if self.tone.directness > 0.7:
                additions.append("Be direct and confident. Avoid hedging words.")
            if self.tone.formality > 0.7:
                additions.append("Maintain a professional, formal tone.")
            elif self.tone.formality < 0.3:
                additions.append("Use a casual, conversational tone.")

        return " ".join(additions)


class PreferenceLearner:
    """Learns user preferences from feedback signals over time.

    Builds a preference profile by analyzing corrections, edits,
    and ratings. The profile is used to inject style guidance
    into system prompts.
    """

    def __init__(
        self,
        store: FeedbackStore,
        preferences_path: Path | None = None,
    ):
        self.store = store
        self.preferences_path = preferences_path or PREFERENCES_PATH
        self.preferences = self._load()

    def learn_from_feedback(self, feedback: Feedback) -> None:
        """Update preferences from a single feedback event."""
        if feedback.feedback_type == FeedbackType.CORRECTION and feedback.correction:
            self._learn_from_correction(feedback.response, feedback.correction)
        elif feedback.feedback_type == FeedbackType.EDIT and feedback.correction:
            self._learn_from_correction(feedback.response, feedback.correction)
        elif feedback.feedback_type == FeedbackType.PREFERENCE:
            if feedback.correction and feedback.preferred_over:
                self._learn_from_correction(feedback.preferred_over, feedback.correction)

        self.preferences.total_signals += 1
        self.preferences.last_updated = time.time()
        self._save()

    def rebuild_from_history(self, limit: int = 500) -> None:
        """Rebuild preferences from all historical feedback."""
        self.preferences = UserPreferences()

        corrections = self.store.get_feedback(feedback_type=FeedbackType.CORRECTION, limit=limit)
        edits = self.store.get_feedback(feedback_type=FeedbackType.EDIT, limit=limit)

        for fb in corrections + edits:
            if fb.correction:
                self._learn_from_correction(fb.response, fb.correction)
                self.preferences.total_signals += 1

        self.preferences.last_updated = time.time()
        self._save()
        logger.info("Rebuilt preferences from %d signals", self.preferences.total_signals)

    def get_system_prompt_additions(self) -> str:
        """Get learned preferences as system prompt text."""
        return self.preferences.to_system_prompt_additions()

    def get_profile(self) -> dict[str, Any]:
        """Get the current preference profile."""
        return self.preferences.to_dict()

    def _learn_from_correction(self, original: str, correction: str) -> None:
        """Extract preferences from an original-correction pair."""
        self.preferences.length.update(len(correction), len(original))
        self.preferences.formatting.update(correction, original)
        self.preferences.tone.update(correction, original)

    def _load(self) -> UserPreferences:
        """Load preferences from disk."""
        if not self.preferences_path.exists():
            return UserPreferences()
        try:
            data = json.loads(self.preferences_path.read_text())
            prefs = UserPreferences()
            if "length" in data:
                prefs.length.avg_preferred_length = data["length"].get("avg_preferred", 500)
                prefs.length.avg_rejected_length = data["length"].get("avg_rejected", 500)
                prefs.length.tendency = data["length"].get("tendency", "neutral")
                prefs.length.samples = data["length"].get("samples", 0)
            if "formatting" in data:
                prefs.formatting.prefers_code_blocks = data["formatting"].get("code_blocks", 0.5)
                prefs.formatting.prefers_bullet_lists = data["formatting"].get("bullet_lists", 0.5)
                prefs.formatting.prefers_headers = data["formatting"].get("headers", 0.5)
                prefs.formatting.prefers_markdown = data["formatting"].get("markdown", 0.5)
                prefs.formatting.samples = data["formatting"].get("samples", 0)
            if "tone" in data:
                prefs.tone.formality = data["tone"].get("formality", 0.5)
                prefs.tone.directness = data["tone"].get("directness", 0.5)
                prefs.tone.technicality = data["tone"].get("technicality", 0.5)
                prefs.tone.samples = data["tone"].get("samples", 0)
            prefs.last_updated = data.get("last_updated", 0)
            prefs.total_signals = data.get("total_signals", 0)
            return prefs
        except (json.JSONDecodeError, KeyError, TypeError, ValueError, AttributeError) as exc:
            # A corrupt/partially-written profile (bad JSON, or valid JSON of
            # the wrong shape) shouldn't crash callers or vanish silently —
            # log and start fresh.
            logger.warning("Discarding unreadable preferences profile: %s", exc)
            return UserPreferences()

    def _save(self) -> None:
        """Save preferences to disk."""
        self.preferences_path.parent.mkdir(parents=True, exist_ok=True)
        self.preferences_path.write_text(json.dumps(self.preferences.to_dict(), indent=2))
