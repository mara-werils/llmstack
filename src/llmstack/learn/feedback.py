"""Feedback collection — captures user corrections, ratings, and edits.

Supports multiple feedback signals:
- Thumbs up/down (binary quality signal)
- Corrections (user provides better response)
- Edits (user modifies the AI's output)
- Preferences (user picks A over B)
- Implicit signals (copy, re-ask, abandon)
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class FeedbackType(str, Enum):
    """Types of feedback signal."""

    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    CORRECTION = "correction"
    EDIT = "edit"
    PREFERENCE = "preference"
    REGENERATE = "regenerate"
    COPY = "copy"
    ABANDON = "abandon"


@dataclass
class Feedback:
    """A single feedback event from the user."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: float = field(default_factory=time.time)
    feedback_type: FeedbackType = FeedbackType.THUMBS_UP

    # The interaction that was judged
    query: str = ""
    response: str = ""
    model: str = ""
    provider: str = ""

    # Feedback content
    correction: str = ""  # user's corrected/preferred response
    edit_diff: str = ""  # what the user changed
    preferred_over: str = ""  # in A/B: the response that lost
    rating: int = 0  # 1-5 star rating (optional)
    tags: list[str] = field(default_factory=list)

    # Context
    command: str = ""  # which llmstack command (ask, chat, agent, etc.)
    context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "feedback_type": self.feedback_type.value,
            "query": self.query,
            "response": self.response,
            "model": self.model,
            "provider": self.provider,
            "correction": self.correction,
            "edit_diff": self.edit_diff,
            "preferred_over": self.preferred_over,
            "rating": self.rating,
            "tags": self.tags,
            "command": self.command,
            "context": self.context,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Feedback:
        data = data.copy()
        data["feedback_type"] = FeedbackType(data.get("feedback_type", "thumbs_up"))
        data.setdefault("tags", [])
        data.setdefault("context", {})
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @property
    def is_positive(self) -> bool:
        return self.feedback_type in (
            FeedbackType.THUMBS_UP,
            FeedbackType.COPY,
        )

    @property
    def is_negative(self) -> bool:
        return self.feedback_type in (
            FeedbackType.THUMBS_DOWN,
            FeedbackType.REGENERATE,
            FeedbackType.ABANDON,
        )

    @property
    def is_implicit(self) -> bool:
        """Return True for implicit feedback signals (copy, regenerate, abandon)."""
        return self.feedback_type in (
            FeedbackType.COPY,
            FeedbackType.REGENERATE,
            FeedbackType.ABANDON,
        )

    @property
    def is_explicit(self) -> bool:
        """Return True for explicit feedback (thumbs, correction, edit, preference)."""
        return not self.is_implicit

    @property
    def has_correction(self) -> bool:
        return self.feedback_type in (
            FeedbackType.CORRECTION,
            FeedbackType.EDIT,
            FeedbackType.PREFERENCE,
        ) and bool(self.correction or self.edit_diff)
