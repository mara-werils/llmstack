"""Tests for batch processing system."""


import pytest

from llmstack.gateway.batch import (
    BatchProcessor, BatchJob, BatchRequest, BatchResult, BatchStatus,
)


@pytest.fixture
def processor():
    return BatchProcessor()


class TestBatchProcessor:
    def test_create_job(self, processor):
        job = processor.create_job(
            requests=[
                {"messages": [{"role": "user", "content": "Hello"}], "model": "llama3.2"},
                {"messages": [{"role": "user", "content": "Hi"}], "model": "llama3.2"},
            ],
        )
        assert job.status == BatchStatus.PENDING
        assert len(job.requests) == 2
        assert len(job.results) == 2

    def test_create_job_exceeds_max(self, processor):
        requests = [{"messages": []} for _ in range(101)]
        with pytest.raises(ValueError, match="exceeds max"):
            processor.create_job(requests=requests)

    def test_get_job(self, processor):
        job = processor.create_job(requests=[{"messages": []}])
        fetched = processor.get_job(job.id)
        assert fetched is not None
        assert fetched.id == job.id

    def test_get_nonexistent(self, processor):
        assert processor.get_job("nope") is None

    def test_list_jobs(self, processor):
        processor.create_job(requests=[{"messages": []}])
        processor.create_job(requests=[{"messages": []}])
        jobs = processor.list_jobs()
        assert len(jobs) == 2

    def test_cancel_pending_job(self, processor):
        job = processor.create_job(requests=[{"messages": []}])
        assert processor.cancel_job(job.id) is True
        assert job.status == BatchStatus.CANCELLED

    def test_cancel_completed_job_fails(self, processor):
        job = processor.create_job(requests=[{"messages": []}])
        job.status = BatchStatus.COMPLETED
        assert processor.cancel_job(job.id) is False

    @pytest.mark.asyncio
    async def test_execute_job(self, processor):
        job = processor.create_job(
            requests=[
                {"messages": [{"role": "user", "content": "Hi"}], "model": "test"},
                {"messages": [{"role": "user", "content": "Hey"}], "model": "test"},
            ],
            concurrency=2,
        )

        async def mock_handler(payload):
            return {
                "choices": [{"message": {"content": "Hello!"}}],
                "usage": {"total_tokens": 10},
            }

        result = await processor.execute_job(job.id, mock_handler)
        assert result is not None
        assert result.status == BatchStatus.COMPLETED
        assert all(r.status == "completed" for r in result.results)

    @pytest.mark.asyncio
    async def test_execute_with_failures(self, processor):
        job = processor.create_job(
            requests=[{"messages": [], "model": "test"}],
        )

        async def failing_handler(payload):
            raise RuntimeError("Connection failed")

        result = await processor.execute_job(job.id, failing_handler)
        assert result.status == BatchStatus.FAILED
        assert result.results[0].status == "failed"
        assert "Connection failed" in result.results[0].error


class TestBatchJob:
    def test_progress(self):
        job = BatchJob(
            requests=[BatchRequest(), BatchRequest()],
            results=[
                BatchResult(request_id="a", status="completed"),
                BatchResult(request_id="b", status="pending"),
            ],
        )
        assert job.progress == 0.5

    def test_to_dict(self):
        job = BatchJob()
        d = job.to_dict()
        assert "id" in d
        assert d["status"] == "pending"

    def test_summary(self):
        job = BatchJob(
            requests=[BatchRequest()],
            results=[BatchResult(request_id="a", status="completed", latency_ms=100)],
        )
        s = job.summary()
        assert s["succeeded"] == 1
        assert s["avg_latency_ms"] == 100.0
