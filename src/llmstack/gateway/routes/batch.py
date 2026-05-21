"""Batch processing API routes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from llmstack.gateway.batch import BatchProcessor

router = APIRouter(tags=["Batch"])

_processor: BatchProcessor | None = None


def get_processor() -> BatchProcessor:
    global _processor
    if _processor is None:
        _processor = BatchProcessor()
    return _processor


class CreateBatchRequest(BaseModel):
    requests: list[dict] = Field(..., description="List of chat completion payloads")
    concurrency: int = Field(5, description="Max concurrent requests")
    metadata: dict = Field(default_factory=dict)


@router.post("/batch/jobs", status_code=201)
async def create_batch_job(req: CreateBatchRequest):
    """Create a new batch processing job."""
    processor = get_processor()
    try:
        job = processor.create_job(
            requests=req.requests,
            concurrency=req.concurrency,
            metadata=req.metadata,
        )
        return job.to_dict()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/batch/jobs")
async def list_batch_jobs(limit: int = 50):
    """List recent batch jobs."""
    processor = get_processor()
    jobs = processor.list_jobs(limit=limit)
    return {"jobs": [j.to_dict() for j in jobs]}


@router.get("/batch/jobs/{job_id}")
async def get_batch_job(job_id: str):
    """Get batch job status and results."""
    processor = get_processor()
    job = processor.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job.summary()


@router.post("/batch/jobs/{job_id}/cancel")
async def cancel_batch_job(job_id: str):
    """Cancel a batch job."""
    processor = get_processor()
    if not processor.cancel_job(job_id):
        raise HTTPException(status_code=400, detail="Job cannot be cancelled")
    return {"cancelled": True}
