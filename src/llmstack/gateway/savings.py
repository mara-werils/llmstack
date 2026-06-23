"""Gateway-side accrual of the money saved by serving requests locally.

Every request the gateway answers from a local backend (cost ``0``) would have
cost something on a metered cloud API. This tracker values that avoided cost with
:class:`llmstack.core.savings.SavingsCalculator` and folds it into a persistent
:class:`llmstack.core.savings.SavingsLedger`, so the gateway can report a running
"llmstack has saved you $X" total over its lifetime.

Cloud-routed requests (those that actually cost money) are intentionally *not*
counted as savings — you paid for them.
"""

from __future__ import annotations

import time

from llmstack.core.savings import (
    SavingsCalculator,
    SavingsEstimate,
    SavingsLedger,
    get_ledger,
)


class SavingsTracker:
    """Accrue savings for locally-served requests into a ledger."""

    def __init__(
        self,
        calculator: SavingsCalculator | None = None,
        ledger: SavingsLedger | None = None,
    ) -> None:
        self.calculator = calculator or SavingsCalculator()
        self.ledger = ledger or get_ledger()

    def record(
        self,
        input_tokens: int,
        output_tokens: int,
        *,
        cost_usd: float = 0.0,
        timestamp: float | None = None,
    ) -> SavingsEstimate | None:
        """Record a request. Returns the estimate, or ``None`` if it was paid.

        A request with ``cost_usd > 0`` was served by a metered cloud provider and
        produced no saving, so it is skipped. Otherwise the cloud-equivalent of the
        tokens served is booked as a saving.
        """
        if cost_usd > 0:
            return None
        if input_tokens <= 0 and output_tokens <= 0:
            return None
        estimate = self.calculator.estimate(input_tokens, output_tokens)
        ts = timestamp if timestamp is not None else time.time()
        self.ledger.record(estimate, timestamp=ts)
        return estimate

    def summary(self, plan_key: str | None = None) -> dict[str, object]:
        """Return the ledger summary with subscription-equivalence."""
        return self.ledger.summary(plan_key)

    def reset(self) -> None:
        self.ledger.reset()


_tracker: SavingsTracker | None = None


def get_savings_tracker() -> SavingsTracker:
    """Return the process-wide savings tracker, creating it on first use."""
    global _tracker
    if _tracker is None:
        _tracker = SavingsTracker()
    return _tracker


def init_savings_tracker(tracker: SavingsTracker) -> None:
    """Override the process-wide savings tracker (gateway startup / tests)."""
    global _tracker
    _tracker = tracker
