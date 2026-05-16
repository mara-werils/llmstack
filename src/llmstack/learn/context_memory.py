"""Context memory — learns which context produces the best responses.

Tracks which types of context (file content, git history, documentation)
lead to better responses for different query types. Optimizes context
selection for RAG and ask commands over time.
"""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llmstack.learn.feedback import Feedback, FeedbackType
from llmstack.learn.store import FeedbackStore

logger = logging.getLogger(__name__)

CONTEXT_MEMORY_PATH = Path.home() / ".llmstack" / "context_memory.json"


@dataclass
class ContextSignal:
    """A signal about context quality for a query type."""

    context_type: str  # file, git, docs, chunk, search
    query_pattern: str  # broad category of query
    positive_count: int = 0
    negative_count: int = 0
    total_uses: int = 0

    @property
    def effectiveness(self) -> float:
        """Effectiveness score 0-1."""
        if self.total_uses == 0:
            return 0.5
        return self.positive_count / self.total_uses


@dataclass
class ContextProfile:
    """Learned context selection profile."""

    signals: dict[str, ContextSignal] = field(default_factory=dict)
    query_patterns: dict[str, list[str]] = field(default_factory=dict)
    last_updated: float = 0.0

    def get_best_contexts(self, query_pattern: str, top_k: int = 3) -> list[str]:
        """Get the most effective context types for a query pattern."""
        relevant = [
            s for s in self.signals.values()
            if s.query_pattern == query_pattern and s.total_uses >= 3
        ]
        relevant.sort(key=lambda s: s.effectiveness, reverse=True)
        return [s.context_type for s in relevant[:top_k]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "signals": {
                k: {
                    "context_type": v.context_type,
                    "query_pattern": v.query_pattern,
                    "positive_count": v.positive_count,
                    "negative_count": v.negative_count,
                    "total_uses": v.total_uses,
                    "effectiveness": round(v.effectiveness, 3),
                }
                for k, v in self.signals.items()
            },
            "query_patterns": self.query_patterns,
            "last_updated": self.last_updated,
        }


class ContextMemory:
    """Learns which context produces the best results for different queries.

    Tracks the relationship between:
    - Query type (code question, bug fix, explanation, generation)
    - Context provided (file chunks, git history, docs, search results)
    - Response quality (from feedback)

    Over time, learns to recommend optimal context selection strategies.
    """

    def __init__(self, store: FeedbackStore, memory_path: Path | None = None):
        self.store = store
        self.memory_path = memory_path or CONTEXT_MEMORY_PATH
        self.profile = self._load()

    def record_context_use(
        self,
        query: str,
        context_types: list[str],
        feedback_type: FeedbackType | None = None,
    ) -> None:
        """Record which context types were used and whether it helped."""
        query_pattern = self._classify_query(query)

        for ctx_type in context_types:
            key = f"{query_pattern}:{ctx_type}"
            if key not in self.profile.signals:
                self.profile.signals[key] = ContextSignal(
                    context_type=ctx_type,
                    query_pattern=query_pattern,
                )

            signal = self.profile.signals[key]
            signal.total_uses += 1

            if feedback_type:
                if feedback_type in (FeedbackType.THUMBS_UP, FeedbackType.COPY):
                    signal.positive_count += 1
                elif feedback_type in (FeedbackType.THUMBS_DOWN, FeedbackType.REGENERATE):
                    signal.negative_count += 1

        # Track query pattern
        if query_pattern not in self.profile.query_patterns:
            self.profile.query_patterns[query_pattern] = []
        self.profile.query_patterns[query_pattern] = list(set(
            self.profile.query_patterns[query_pattern] + context_types
        ))[:10]

        self.profile.last_updated = time.time()
        self._save()

    def recommend_context(self, query: str) -> list[str]:
        """Recommend context types for a query based on learned effectiveness."""
        pattern = self._classify_query(query)
        return self.profile.get_best_contexts(pattern)

    def get_effectiveness_report(self) -> dict[str, Any]:
        """Get effectiveness of each context type per query pattern."""
        report: dict[str, dict[str, float]] = defaultdict(dict)
        for signal in self.profile.signals.values():
            if signal.total_uses >= 3:
                report[signal.query_pattern][signal.context_type] = signal.effectiveness
        return dict(report)

    def _classify_query(self, query: str) -> str:
        """Classify query into broad categories."""
        q = query.lower()

        if any(w in q for w in ["how to", "how do i", "how can i"]):
            return "howto"
        if any(w in q for w in ["why", "explain", "what is", "what does"]):
            return "explanation"
        if any(w in q for w in ["fix", "bug", "error", "broken", "not working"]):
            return "debugging"
        if any(w in q for w in ["write", "create", "generate", "implement", "add"]):
            return "generation"
        if any(w in q for w in ["review", "improve", "optimize", "refactor"]):
            return "review"
        if any(w in q for w in ["test", "coverage", "spec"]):
            return "testing"

        return "general"

    def _load(self) -> ContextProfile:
        """Load from disk."""
        if not self.memory_path.exists():
            return ContextProfile()
        try:
            data = json.loads(self.memory_path.read_text())
            profile = ContextProfile()
            profile.last_updated = data.get("last_updated", 0)
            profile.query_patterns = data.get("query_patterns", {})
            for key, sig_data in data.get("signals", {}).items():
                profile.signals[key] = ContextSignal(
                    context_type=sig_data["context_type"],
                    query_pattern=sig_data["query_pattern"],
                    positive_count=sig_data.get("positive_count", 0),
                    negative_count=sig_data.get("negative_count", 0),
                    total_uses=sig_data.get("total_uses", 0),
                )
            return profile
        except (json.JSONDecodeError, KeyError):
            return ContextProfile()

    def _save(self) -> None:
        """Save to disk."""
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)
        self.memory_path.write_text(json.dumps(self.profile.to_dict(), indent=2))
