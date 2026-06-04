"""Tests for batch processor."""

from llmstack.sdk.batch import BatchItem, BatchProcessor, BatchSummary, BatchResult


def test_batch_item_creation():
    item = BatchItem(id=1, prompt="Hello", model="llama3.2")
    assert item.id == 1
    assert item.prompt == "Hello"
    assert item.model == "llama3.2"
    assert item.max_tokens == 1024
    assert item.temperature == 0.7


def test_batch_item_defaults():
    item = BatchItem(id="test", prompt="Hi")
    assert item.system_prompt == ""
    assert item.model is None
    assert item.max_tokens == 1024


def test_batch_processor_creation():
    processor = BatchProcessor(
        base_url="http://localhost:8000",
        concurrency=5,
        model="llama3.2",
    )
    assert processor.base_url == "http://localhost:8000"
    assert processor.concurrency == 5
    assert processor.default_model == "llama3.2"


def test_batch_processor_progress_callback():
    processor = BatchProcessor()
    called = []

    def callback(completed, total, result):
        called.append((completed, total))

    result = processor.on_progress(callback)
    assert result is processor  # Should return self for chaining
    assert processor._progress_callback is not None


def test_batch_summary():
    results = [
        BatchResult(id=1, prompt="a", response="b", tokens_used=10, duration=1.0, success=True),
        BatchResult(
            id=2,
            prompt="c",
            response="",
            tokens_used=0,
            duration=0.5,
            success=False,
            error="timeout",
        ),
    ]

    summary = BatchSummary(
        total=2,
        completed=1,
        failed=1,
        total_tokens=10,
        total_duration=1.5,
        avg_duration=0.75,
        results=results,
    )

    assert summary.total == 2
    assert summary.completed == 1
    assert summary.failed == 1
    assert summary.total_tokens == 10
