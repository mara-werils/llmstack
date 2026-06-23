"""Turn local LLM usage into a concrete "dollars saved" figure.

This module is the engine behind llmstack's "actually saves you money" claim. It
takes real usage — input/output tokens served locally — and values it against a
dated cloud baseline from :mod:`llmstack.core.pricing`. Because a local request
costs effectively nothing, the equivalent cloud cost *is* the saving.

The math is deliberately conservative (the default baseline is a cheap mainstream
cloud model) so the figure is one we can defend rather than inflate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from llmstack.core.pricing import (
    SubscriptionPlan,
    TokenPrice,
    baseline_subscription,
    baseline_token_price,
)


@dataclass(frozen=True)
class SavingsEstimate:
    """What one (or a batch of) local request(s) saved versus the cloud baseline."""

    input_tokens: int
    output_tokens: int
    baseline_model: str
    cloud_cost_usd: float
    local_cost_usd: float
    saved_usd: float

    def as_dict(self) -> dict[str, float | int | str]:
        return asdict(self)


class SavingsCalculator:
    """Value local usage against a dated cloud price baseline."""

    def __init__(self, baseline_model: str | None = None) -> None:
        self._baseline: TokenPrice = baseline_token_price(baseline_model)

    @property
    def baseline(self) -> TokenPrice:
        return self._baseline

    def estimate(
        self,
        input_tokens: int,
        output_tokens: int,
        local_cost_usd: float = 0.0,
    ) -> SavingsEstimate:
        """Estimate the saving from serving ``input``/``output`` tokens locally.

        ``local_cost_usd`` defaults to 0 (local inference is free); pass a value
        to account for, e.g., a metered local endpoint. The saving never goes
        negative — if the local path somehow costs more, the saving is clamped
        to zero rather than reported as a loss.
        """
        if input_tokens < 0 or output_tokens < 0:
            raise ValueError("token counts must be non-negative")
        cloud_cost = self._baseline.cost_usd(input_tokens, output_tokens)
        saved = max(0.0, cloud_cost - local_cost_usd)
        return SavingsEstimate(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            baseline_model=self._baseline.model,
            cloud_cost_usd=cloud_cost,
            local_cost_usd=local_cost_usd,
            saved_usd=saved,
        )

    def subscription_months_covered(self, saved_usd: float, plan_key: str | None = None) -> float:
        """How many months of a paid subscription ``saved_usd`` would cover."""
        plan: SubscriptionPlan = baseline_subscription(plan_key)
        return saved_usd / plan.effective_monthly_usd
