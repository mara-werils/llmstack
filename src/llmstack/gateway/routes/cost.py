"""Cost tracking and budget management API routes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from llmstack.gateway.cost_tracker import CostTracker, Budget, BudgetPeriod

router = APIRouter(tags=["Cost"])

_tracker: CostTracker | None = None


def get_tracker() -> CostTracker:
    global _tracker
    if _tracker is None:
        _tracker = CostTracker()
    return _tracker


def init_cost_tracker(tracker: CostTracker) -> None:
    global _tracker
    _tracker = tracker


class AddBudgetRequest(BaseModel):
    name: str
    limit_usd: float
    period: str = "monthly"
    model: str | None = None
    provider: str | None = None
    alert_at_percent: float = 80.0


class SetPricingRequest(BaseModel):
    model: str
    input_per_million: float
    output_per_million: float


@router.get("/cost/summary")
async def cost_summary():
    """Get comprehensive cost summary with breakdowns, plus local savings."""
    summary = get_tracker().get_summary()
    # Fold in the running "saved by running locally" total so a single cost view
    # shows both what cloud calls cost and what local calls avoided.
    try:
        from llmstack.gateway.savings import get_savings_tracker

        summary["savings"] = get_savings_tracker().summary()
    except Exception:  # pragma: no cover - never let savings break the cost view
        pass
    return summary


@router.get("/cost/budgets")
async def list_budgets():
    """List all configured budgets with current spend."""
    return {"budgets": get_tracker().get_budgets()}


@router.post("/cost/budgets", status_code=201)
async def add_budget(req: AddBudgetRequest):
    """Add or update a budget limit."""
    try:
        period = BudgetPeriod(req.period)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid period: {req.period}. Use: daily, weekly, monthly, total",
        )
    budget = Budget(
        name=req.name,
        limit_usd=req.limit_usd,
        period=period,
        model=req.model,
        provider=req.provider,
        alert_at_percent=req.alert_at_percent,
    )
    get_tracker().add_budget(budget)
    return {"created": True, "name": req.name}


@router.delete("/cost/budgets/{name}")
async def remove_budget(name: str):
    """Remove a budget."""
    if not get_tracker().remove_budget(name):
        raise HTTPException(status_code=404, detail="Budget not found")
    return {"deleted": True}


@router.get("/cost/alerts")
async def list_alerts(limit: int = 50):
    """Get recent budget alerts."""
    alerts = get_tracker().get_alerts(limit=limit)
    return {"alerts": [a.to_dict() for a in alerts]}


@router.post("/cost/pricing")
async def set_pricing(req: SetPricingRequest):
    """Set custom pricing for a model."""
    get_tracker().set_pricing(req.model, req.input_per_million, req.output_per_million)
    return {"updated": True, "model": req.model}
