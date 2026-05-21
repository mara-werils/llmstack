"""Model leaderboard API routes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from llmstack.gateway.leaderboard import Leaderboard

router = APIRouter(tags=["Leaderboard"])

_leaderboard: Leaderboard | None = None


def get_leaderboard() -> Leaderboard:
    global _leaderboard
    if _leaderboard is None:
        _leaderboard = Leaderboard()
    return _leaderboard


def init_leaderboard(lb: Leaderboard) -> None:
    global _leaderboard
    _leaderboard = lb


@router.get("/leaderboard")
async def get_rankings(sort_by: str = "quality", min_requests: int = 5):
    """Get model rankings sorted by quality, latency, cost, speed, or error_rate."""
    lb = get_leaderboard()
    return {"rankings": lb.get_rankings(sort_by=sort_by, min_requests=min_requests)}


@router.get("/leaderboard/summary")
async def leaderboard_summary():
    """Get leaderboard summary with top models."""
    return get_leaderboard().get_summary()


@router.get("/leaderboard/models/{model}")
async def get_model_metrics(model: str):
    """Get detailed metrics for a specific model."""
    lb = get_leaderboard()
    result = lb.get_model(model)
    if result is None:
        raise HTTPException(status_code=404, detail="Model not found in leaderboard")
    return result


@router.get("/leaderboard/compare")
async def compare_models(models: str = ""):
    """Compare models side by side. Pass comma-separated model names."""
    if not models:
        raise HTTPException(status_code=400, detail="Provide model names as comma-separated list")
    model_list = [m.strip() for m in models.split(",") if m.strip()]
    lb = get_leaderboard()
    return {"comparison": lb.compare(model_list)}
