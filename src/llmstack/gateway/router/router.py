"""Smart Model Router — picks the optimal model for each query.

Analyses query complexity and selects the smallest model that can handle
the task well, saving compute without sacrificing quality.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from llmstack.gateway.router.classifier import QueryClassifier, QueryProfile

logger = logging.getLogger(__name__)


@dataclass
class ModelTier:
    """Describes a model and its capabilities."""

    name: str  # e.g. "llama3.2:1b", "gpt-4o", "claude-sonnet-4-20250514"
    tier: str  # "simple" | "medium" | "complex"
    max_context: int = 8192
    speed_score: float = 1.0  # relative speed (higher = faster)
    quality_score: float = 1.0  # relative quality (higher = better)
    provider: str = "local"  # provider name: "local", "openai", "anthropic", etc.
    cost_per_m_input: float = 0.0  # $ per 1M input tokens
    cost_per_m_output: float = 0.0  # $ per 1M output tokens


@dataclass
class RoutingDecision:
    """The result of a routing decision."""

    model: str  # Selected model name
    profile: QueryProfile  # Classification that led to this decision
    alternatives: list[str] = field(default_factory=list)
    estimated_speedup: float = 1.0  # vs always using the largest model
    provider: str = "local"  # which provider serves this model
    estimated_cost_per_1k: float = 0.0  # estimated cost per 1K tokens (input)


# Tier ordering for comparisons
_TIER_ORDER = {"simple": 0, "medium": 1, "complex": 2}


class ModelRouter:
    """Routes queries to the optimal model based on complexity analysis.

    Strategies
    ----------
    - ``cost``     : minimise compute — pick the smallest adequate model
    - ``quality``  : pick the best-quality model for the tier or above
    - ``balanced`` : weighted cost / quality tradeoff
    - ``latency``  : minimise response time
    """

    STRATEGIES = ("cost", "quality", "balanced", "latency")

    def __init__(self, models: list[ModelTier], strategy: str = "cost"):
        if strategy not in self.STRATEGIES:
            raise ValueError(f"Unknown strategy '{strategy}', choose from {self.STRATEGIES}")
        if not models:
            raise ValueError("At least one ModelTier must be provided")

        self.models = sorted(models, key=lambda m: _TIER_ORDER.get(m.tier, 1))
        self.strategy = strategy
        self._classifier = QueryClassifier()
        self._override: str | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def model_count(self) -> int:
        """Return the number of configured models."""
        return len(self.models)

    @property
    def available_strategies(self) -> tuple[str, ...]:
        """Return the tuple of supported routing strategies."""
        return self.STRATEGIES

    def route(self, messages: list[dict]) -> RoutingDecision:
        """Analyse messages and pick the best model."""
        profile = self._classifier.classify(messages)

        # Explicit override takes priority
        if self._override:
            return RoutingDecision(
                model=self._override,
                profile=profile,
                alternatives=[m.name for m in self.models if m.name != self._override],
                estimated_speedup=1.0,
            )

        # Select model according to strategy
        selected = self._select(profile)
        largest = self._largest_model()

        alternatives = [m.name for m in self.models if m.name != selected.name]

        speedup = 1.0
        if largest and selected.name != largest.name and selected.speed_score > 0:
            speedup = selected.speed_score / max(largest.speed_score, 0.01)

        decision = RoutingDecision(
            model=selected.name,
            profile=profile,
            alternatives=alternatives,
            estimated_speedup=round(speedup, 2),
            provider=selected.provider,
            estimated_cost_per_1k=round(selected.cost_per_m_input / 1000, 6),
        )

        logger.info(
            "Routed query: tier=%s score=%.3f model=%s provider=%s strategy=%s",
            profile.tier,
            profile.score,
            selected.name,
            selected.provider,
            self.strategy,
        )
        return decision

    def classify_only(self, messages: list[dict]) -> QueryProfile:
        """Classify without routing (useful for debugging)."""
        return self._classifier.classify(messages)

    def override(self, model: str | None) -> None:
        """Set or clear an explicit model override."""
        self._override = model

    # ------------------------------------------------------------------
    # Strategy implementations
    # ------------------------------------------------------------------

    def _select(self, profile: QueryProfile) -> ModelTier:
        """Dispatch to the appropriate strategy."""
        method = {
            "cost": self._select_cost,
            "quality": self._select_quality,
            "balanced": self._select_balanced,
            "latency": self._select_latency,
        }[self.strategy]
        return method(profile)

    def _select_cost(self, profile: QueryProfile) -> ModelTier:
        """Pick the cheapest adequate model.

        When providers have real dollar costs, sort by cost first.
        Falls back to tier ordering for local (free) models.
        """
        tier_val = _TIER_ORDER.get(profile.tier, 1)
        candidates = [m for m in self.models if _TIER_ORDER.get(m.tier, 1) >= tier_val]
        if not candidates:
            return self._largest_model()
        # Sort by: real cost first (cheaper is better), then tier, then speed
        return min(
            candidates,
            key=lambda m: (
                m.cost_per_m_input + m.cost_per_m_output,
                _TIER_ORDER.get(m.tier, 1),
                -m.speed_score,
            ),
        )

    def _select_quality(self, profile: QueryProfile) -> ModelTier:
        """Pick the highest-quality model for the tier or above."""
        tier_val = _TIER_ORDER.get(profile.tier, 1)
        candidates = [m for m in self.models if _TIER_ORDER.get(m.tier, 1) >= tier_val]
        if not candidates:
            return self._largest_model()
        return max(candidates, key=lambda m: m.quality_score)

    def _select_balanced(self, profile: QueryProfile) -> ModelTier:
        """Weighted score: 0.5 * quality + 0.3 * speed + 0.2 * tier_match."""
        tier_val = _TIER_ORDER.get(profile.tier, 1)

        def score(m: ModelTier) -> float:
            tier_match = 1.0 - abs(_TIER_ORDER.get(m.tier, 1) - tier_val) / 2.0
            return 0.5 * m.quality_score + 0.3 * m.speed_score + 0.2 * tier_match

        return max(self.models, key=score)

    def _select_latency(self, profile: QueryProfile) -> ModelTier:
        """Pick the fastest model whose tier >= query tier."""
        tier_val = _TIER_ORDER.get(profile.tier, 1)
        candidates = [m for m in self.models if _TIER_ORDER.get(m.tier, 1) >= tier_val]
        if not candidates:
            return max(self.models, key=lambda m: m.speed_score)
        return max(candidates, key=lambda m: m.speed_score)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _largest_model(self) -> ModelTier:
        """Return the highest-tier, highest-quality model."""
        return max(self.models, key=lambda m: (_TIER_ORDER.get(m.tier, 1), m.quality_score))
