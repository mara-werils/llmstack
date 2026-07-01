"""Quality scoring — heuristic-based quality metrics for LLM responses.

All scoring is local, fast, and requires no external API calls.
Designed for real-time scoring of every response in the gateway.

Metrics:
- coherence: structural quality (length, sentence count, formatting)
- relevance: response addresses the query (keyword overlap)
- refusal: detects refusal/inability patterns
- toxicity: flags potentially harmful content
- repetition: detects repetitive/looping output
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class QualityScore:
    """Quality scores for a single response."""

    coherence: float = 0.0  # 0-1: structural quality
    relevance: float = 0.0  # 0-1: addresses the query
    refusal: float = 0.0  # 0-1: how likely it's a refusal (lower is better)
    toxicity: float = 0.0  # 0-1: harmful content flag (lower is better)
    repetition: float = 0.0  # 0-1: repetition level (lower is better)
    overall: float = 0.0  # 0-1: weighted aggregate

    @property
    def is_refusal(self) -> bool:
        """Return True if the response is likely a refusal."""
        return self.refusal >= 0.5

    @property
    def is_toxic(self) -> bool:
        """Return True if the response was flagged as potentially harmful."""
        return self.toxicity >= 0.5

    def to_dict(self) -> dict[str, float]:
        return {
            "coherence": round(self.coherence, 4),
            "relevance": round(self.relevance, 4),
            "refusal": round(self.refusal, 4),
            "toxicity": round(self.toxicity, 4),
            "repetition": round(self.repetition, 4),
            "overall": round(self.overall, 4),
        }


# Refusal patterns
_REFUSAL_PATTERNS = [
    r"(?i)i (can't|cannot|am unable to|won't|will not|don't|do not) (help|assist|provide|generate|create|do)",
    r"(?i)as an ai",
    r"(?i)i'm (sorry|afraid)",
    r"(?i)it('s| is) (not appropriate|against my|beyond my)",
    r"(?i)i (must )?decline",
    r"(?i)this (is|goes) (beyond|against)",
]

_REFUSAL_RES = [re.compile(p) for p in _REFUSAL_PATTERNS]

# Toxicity indicators (lightweight heuristic, not a full classifier)
_TOXIC_PATTERNS = [
    r"(?i)\b(kill|murder|harm|attack|destroy|hate|racist|sexist)\b.*\b(how to|instructions|steps|guide)\b",
    r"(?i)\b(bomb|weapon|explosive|poison)\b.*\b(make|build|create|instructions)\b",
]

_TOXIC_RES = [re.compile(p) for p in _TOXIC_PATTERNS]


class QualityScorer:
    """Score LLM responses on multiple quality dimensions.

    All scoring is heuristic-based — fast enough to run on every request
    with negligible latency (<1ms).
    """

    def score(self, query: str, response: str) -> QualityScore:
        """Score a response given its query."""
        if not response:
            return QualityScore(overall=0.0)

        coherence = self._score_coherence(response)
        relevance = self._score_relevance(query, response)
        refusal = self._score_refusal(response)
        toxicity = self._score_toxicity(response)
        repetition = self._score_repetition(response)

        # Weighted overall: coherence and relevance are positive,
        # refusal/toxicity/repetition are negative signals
        overall = (
            0.35 * coherence
            + 0.35 * relevance
            + 0.10 * (1.0 - refusal)
            + 0.10 * (1.0 - toxicity)
            + 0.10 * (1.0 - repetition)
        )

        return QualityScore(
            coherence=coherence,
            relevance=relevance,
            refusal=refusal,
            toxicity=toxicity,
            repetition=repetition,
            overall=max(0.0, min(1.0, overall)),
        )

    def _score_coherence(self, response: str) -> float:
        """Score structural quality: length, sentences, formatting."""
        score = 0.0

        # Length (too short or too long is bad)
        length = len(response)
        if length < 10:
            score += 0.1
        elif length < 50:
            score += 0.3
        elif length < 200:
            score += 0.6
        elif length < 2000:
            score += 0.9
        else:
            score += 0.7  # very long responses slightly penalized

        # Sentence structure
        sentences = re.split(r"[.!?]+", response)
        sentences = [s.strip() for s in sentences if s.strip()]
        if len(sentences) >= 2:
            score += 0.1

        # Has some structure (paragraphs, lists, code)
        if "\n" in response:
            score += 0.05
        if re.search(r"^\s*[-*\d]+[.)]\s", response, re.MULTILINE):
            score += 0.05
        if "```" in response:
            score += 0.05

        return min(1.0, score)

    def _score_relevance(self, query: str, response: str) -> float:
        """Score how well the response addresses the query."""
        if not query:
            return 0.5

        query_words = set(re.findall(r"\b\w{3,}\b", query.lower()))
        response_words = set(re.findall(r"\b\w{3,}\b", response.lower()))

        if not query_words:
            return 0.5

        # Word overlap (Jaccard-like but asymmetric)
        overlap = len(query_words & response_words)
        coverage = overlap / len(query_words)

        # Bonus for longer, substantive responses
        length_bonus = min(0.2, len(response) / 5000)

        return min(1.0, coverage * 0.8 + length_bonus)

    def _score_refusal(self, response: str) -> float:
        """Detect refusal patterns. Returns 0-1 (0 = no refusal)."""
        hits = sum(1 for r in _REFUSAL_RES if r.search(response))
        if hits == 0:
            return 0.0
        if hits == 1:
            return 0.5
        return min(1.0, 0.5 + hits * 0.15)

    def _score_toxicity(self, response: str) -> float:
        """Lightweight toxicity detection. Returns 0-1 (0 = safe)."""
        hits = sum(1 for r in _TOXIC_RES if r.search(response))
        return min(1.0, hits * 0.5)

    def _score_repetition(self, response: str) -> float:
        """Detect repetitive/looping output."""
        if len(response) < 100:
            return 0.0

        # Check for repeated sentences
        sentences = re.split(r"[.!?\n]+", response)
        sentences = [s.strip().lower() for s in sentences if len(s.strip()) > 20]

        if not sentences:
            return 0.0

        unique = set(sentences)
        dup_ratio = 1.0 - (len(unique) / len(sentences))

        # Check for repeated n-grams (3-grams)
        words = response.lower().split()
        if len(words) >= 10:
            trigrams = [" ".join(words[i : i + 3]) for i in range(len(words) - 2)]
            unique_trigrams = set(trigrams)
            trigram_dup = 1.0 - (len(unique_trigrams) / max(len(trigrams), 1))
            # Normal text has ~10-20% trigram duplication
            trigram_score = max(0.0, (trigram_dup - 0.2) / 0.8)
        else:
            trigram_score = 0.0

        return min(1.0, max(dup_ratio, trigram_score))
