"""Gateway routes for the adaptive learning pipeline.

Provides REST API endpoints for:
- Submitting feedback (thumbs, corrections, preferences)
- Querying learning status and metrics
- Managing model versions
- Viewing learned preferences
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/learn", tags=["learning"])


# --- Request/Response Models ---


class FeedbackRequest(BaseModel):
    """Submit feedback for a response."""

    feedback_type: str = Field(..., description="Type: thumbs_up, thumbs_down, correction, edit, preference")
    query: str = Field("", description="The original query")
    response: str = Field("", description="The AI response being judged")
    correction: str = Field("", description="User's corrected/preferred response")
    model: str = Field("", description="Model that generated the response")
    rating: int = Field(0, description="1-5 star rating (optional)")
    tags: list[str] = Field(default_factory=list)
    command: str = Field("", description="Source command (chat, ask, agent)")


class FeedbackResponse(BaseModel):
    """Response after submitting feedback."""

    id: str
    status: str = "recorded"
    pending_count: int = 0


class StatusResponse(BaseModel):
    """Learning pipeline status."""

    status: str
    metrics: dict[str, Any]
    recommendations: list[str]


class VersionsResponse(BaseModel):
    """Model versions list."""

    versions: list[dict[str, Any]]
    active_version: str | None = None


class PreferencesResponse(BaseModel):
    """Learned user preferences."""

    preferences: dict[str, Any]
    system_prompt_additions: str = ""


class TrainTriggerResponse(BaseModel):
    """Training trigger result."""

    success: bool
    message: str = ""
    details: dict[str, Any] = Field(default_factory=dict)


# --- Endpoints ---


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(request: FeedbackRequest) -> FeedbackResponse:
    """Submit feedback for a response to improve future outputs."""
    from llmstack.learn.feedback import Feedback, FeedbackType
    from llmstack.learn.store import FeedbackStore

    try:
        fb_type = FeedbackType(request.feedback_type)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid feedback_type. Valid types: {[t.value for t in FeedbackType]}",
        )

    store = FeedbackStore()
    feedback = Feedback(
        feedback_type=fb_type,
        query=request.query,
        response=request.response,
        correction=request.correction,
        model=request.model,
        rating=request.rating,
        tags=request.tags,
        command=request.command,
    )
    store.add_feedback(feedback)

    # Update preference learner
    if feedback.has_correction:
        from llmstack.learn.preferences import PreferenceLearner

        learner = PreferenceLearner(store=store)
        learner.learn_from_feedback(feedback)

    pending = store.get_unused_feedback_count()
    store.close()

    return FeedbackResponse(
        id=feedback.id,
        status="recorded",
        pending_count=pending,
    )


@router.get("/status", response_model=StatusResponse)
async def get_status() -> StatusResponse:
    """Get learning pipeline status and metrics."""
    from llmstack.learn.analytics import LearningAnalytics
    from llmstack.learn.store import FeedbackStore
    from llmstack.learn.versions import ModelVersionManager

    store = FeedbackStore()
    version_mgr = ModelVersionManager(store=store)
    analytics = LearningAnalytics(store=store, version_mgr=version_mgr)
    summary = analytics.get_summary()
    store.close()

    return StatusResponse(
        status=summary["status"],
        metrics=summary["metrics"],
        recommendations=summary.get("recommendations", []),
    )


@router.get("/versions", response_model=VersionsResponse)
async def get_versions() -> VersionsResponse:
    """List model versions."""
    from llmstack.learn.store import FeedbackStore
    from llmstack.learn.versions import ModelVersionManager

    store = FeedbackStore()
    version_mgr = ModelVersionManager(store=store)
    versions = version_mgr.list_versions()
    active = version_mgr.get_active()
    store.close()

    return VersionsResponse(
        versions=[
            {
                "version": v.version,
                "base_model": v.base_model,
                "quality_score": v.quality_score,
                "is_active": v.is_active,
                "timestamp": v.timestamp,
            }
            for v in versions
        ],
        active_version=active.version if active else None,
    )


@router.get("/preferences", response_model=PreferencesResponse)
async def get_preferences() -> PreferencesResponse:
    """Get learned user preferences."""
    from llmstack.learn.preferences import PreferenceLearner
    from llmstack.learn.store import FeedbackStore

    store = FeedbackStore()
    learner = PreferenceLearner(store=store)
    profile = learner.get_profile()
    additions = learner.get_system_prompt_additions()
    store.close()

    return PreferencesResponse(
        preferences=profile,
        system_prompt_additions=additions,
    )


@router.post("/train", response_model=TrainTriggerResponse)
async def trigger_training() -> TrainTriggerResponse:
    """Manually trigger a training run."""
    from llmstack.learn.dataset import DatasetGenerator
    from llmstack.learn.scheduler import TrainScheduler, TriggerReason
    from llmstack.learn.store import FeedbackStore
    from llmstack.learn.versions import ModelVersionManager

    store = FeedbackStore()

    if store.get_unused_feedback_count() == 0:
        store.close()
        return TrainTriggerResponse(
            success=False,
            message="No unused feedback available for training",
        )

    dataset_gen = DatasetGenerator(store=store)
    version_mgr = ModelVersionManager(store=store)
    scheduler = TrainScheduler(
        store=store,
        dataset_gen=dataset_gen,
        version_mgr=version_mgr,
    )

    # Generate dataset only (actual training requires GPU setup)
    from llmstack.learn.dataset import DatasetStrategy
    from pathlib import Path

    dataset = dataset_gen.generate(strategy=DatasetStrategy.MIXED)
    if dataset.total_examples == 0:
        store.close()
        return TrainTriggerResponse(
            success=False,
            message="No training examples could be generated from feedback",
        )

    output_dir = Path.home() / ".llmstack" / "training" / "datasets"
    path = dataset.save(output_dir)
    store.mark_feedback_used(dataset.feedback_ids)
    store.close()

    return TrainTriggerResponse(
        success=True,
        message=f"Dataset generated with {dataset.total_examples} examples",
        details={
            "sft_examples": len(dataset.sft_examples),
            "dpo_examples": len(dataset.dpo_examples),
            "dataset_path": str(path),
        },
    )


@router.post("/rollback")
async def rollback_version() -> dict[str, Any]:
    """Rollback to previous model version."""
    from llmstack.learn.store import FeedbackStore
    from llmstack.learn.versions import ModelVersionManager

    store = FeedbackStore()
    version_mgr = ModelVersionManager(store=store)
    result = version_mgr.rollback()
    store.close()

    if result:
        return {"success": True, "version": result.version, "quality": result.quality_score}
    return {"success": False, "error": "No previous version available"}
