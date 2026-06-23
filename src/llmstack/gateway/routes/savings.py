"""Savings reporting API routes.

Exposes the running "money saved by running locally" total, plus the dated,
sourced pricing the figure is computed from — so the number is transparent and
auditable rather than a marketing assertion.
"""

from __future__ import annotations

from fastapi import APIRouter

from llmstack.core import pricing
from llmstack.gateway.savings import SavingsTracker, get_savings_tracker

router = APIRouter(tags=["Savings"])

_tracker: SavingsTracker | None = None


def get_tracker() -> SavingsTracker:
    return _tracker if _tracker is not None else get_savings_tracker()


def init_savings_route(tracker: SavingsTracker) -> None:
    global _tracker
    _tracker = tracker


@router.get("/savings/summary")
async def savings_summary(plan: str | None = None):
    """Cumulative savings vs the cloud baseline, with subscription equivalence."""
    return get_tracker().summary(plan)


@router.get("/savings/pricing")
async def savings_pricing():
    """The dated, sourced pricing catalog the savings figure is derived from."""
    return {
        "as_of": pricing.PRICING_AS_OF,
        "baseline_model": pricing.DEFAULT_API_BASELINE,
        "api_pricing": [
            {
                "model": p.model,
                "vendor": p.vendor,
                "input_per_million": p.input_per_million,
                "output_per_million": p.output_per_million,
                "source": p.source,
                "as_of": p.as_of,
            }
            for p in pricing.API_PRICING.values()
        ],
        "subscriptions": [
            {
                "key": s.key,
                "name": s.name,
                "vendor": s.vendor,
                "monthly_usd": s.monthly_usd,
                "annual_usd": s.annual_usd,
                "effective_monthly_usd": s.effective_monthly_usd,
                "source": s.source,
                "as_of": s.as_of,
            }
            for s in pricing.SUBSCRIPTIONS.values()
        ],
    }


@router.post("/savings/reset")
async def savings_reset():
    """Reset the savings ledger to zero."""
    get_tracker().reset()
    return {"reset": True}
