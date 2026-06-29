"""Turn local LLM usage into a concrete "dollars saved" figure.

This module is the engine behind llmstack's "actually saves you money" claim. It
takes real usage — input/output tokens served locally — and values it against a
dated cloud baseline from :mod:`llmstack.core.pricing`. Because a local request
costs effectively nothing, the equivalent cloud cost *is* the saving.

The math is deliberately conservative (the default baseline is a cheap mainstream
cloud model) so the figure is one we can defend rather than inflate.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

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
        if local_cost_usd < 0:
            raise ValueError("local_cost_usd must be non-negative")
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


DEFAULT_LEDGER_PATH = Path.home() / ".llmstack" / "savings.json"


@dataclass
class SavingsState:
    """Cumulative savings totals, persisted across runs."""

    total_requests: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_saved_usd: float = 0.0
    first_recorded_at: float | None = None
    last_recorded_at: float | None = None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class SavingsLedger:
    """A small, file-backed running total of dollars saved by running locally.

    The ledger is deliberately I/O-light and deterministic: timestamps are passed
    in by the caller (``record(..., timestamp=...)``) rather than read from the
    clock, so tests are reproducible and the gateway controls the time source.
    """

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or DEFAULT_LEDGER_PATH
        self.state = self._load()

    def _load(self) -> SavingsState:
        if not self.path.exists():
            return SavingsState()
        try:
            raw = json.loads(self.path.read_text())
        except (json.JSONDecodeError, OSError):
            return SavingsState()
        known = SavingsState().as_dict().keys()
        return SavingsState(**{k: v for k, v in raw.items() if k in known})

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.state.as_dict(), indent=2))

    def record(
        self,
        estimate: SavingsEstimate,
        *,
        timestamp: float,
        persist: bool = True,
    ) -> SavingsState:
        """Fold a :class:`SavingsEstimate` into the running totals and persist."""
        s = self.state
        s.total_requests += 1
        s.total_input_tokens += estimate.input_tokens
        s.total_output_tokens += estimate.output_tokens
        s.total_saved_usd += estimate.saved_usd
        if s.first_recorded_at is None:
            s.first_recorded_at = timestamp
        s.last_recorded_at = timestamp
        if persist:
            self.save()
        return s

    def summary(self, plan_key: str | None = None) -> dict[str, object]:
        """A display-ready snapshot, including subscription-months equivalence."""
        s = self.state
        plan = baseline_subscription(plan_key)
        months = s.total_saved_usd / plan.effective_monthly_usd
        return {
            **s.as_dict(),
            "subscription": {
                "key": plan.key,
                "name": plan.name,
                "monthly_usd": plan.effective_monthly_usd,
                "months_covered": months,
            },
        }

    def reset(self, *, persist: bool = True) -> None:
        """Clear all totals."""
        self.state = SavingsState()
        if persist:
            self.save()


# A process-wide ledger, created lazily so importing this module is side-effect free.
_ledger: SavingsLedger | None = None


def get_ledger() -> SavingsLedger:
    """Return the process-wide :class:`SavingsLedger`, creating it on first use."""
    global _ledger
    if _ledger is None:
        _ledger = SavingsLedger()
    return _ledger


def set_ledger(ledger: SavingsLedger) -> None:
    """Override the process-wide ledger (used by the gateway and in tests)."""
    global _ledger
    _ledger = ledger
