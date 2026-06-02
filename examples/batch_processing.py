"""Example: Batch processing multiple prompts concurrently.

Usage:
    python examples/batch_processing.py

Requires: llmstack gateway running at http://localhost:8000
"""

import asyncio
from llmstack.sdk.batch import BatchProcessor, BatchItem


async def main():
    # Create batch processor
    processor = BatchProcessor(
        base_url="http://localhost:8000",
        concurrency=3,
        model="llama3.2",
    )

    # Define prompts
    items = [
        BatchItem(id=1, prompt="What is Python?", max_tokens=100),
        BatchItem(id=2, prompt="Explain REST APIs in one sentence", max_tokens=100),
        BatchItem(id=3, prompt="What is Docker?", max_tokens=100),
        BatchItem(id=4, prompt="Explain microservices", max_tokens=100),
        BatchItem(id=5, prompt="What is Kubernetes?", max_tokens=100),
    ]

    # Progress callback
    def on_progress(completed, total, result):
        status = "OK" if result.success else "FAIL"
        print(f"  [{completed}/{total}] {status} - {result.prompt[:40]}... ({result.duration:.1f}s)")

    processor.on_progress(on_progress)

    print(f"Processing {len(items)} prompts with concurrency=3...\n")

    # Run batch
    summary = await processor.run(items)

    print(f"\n--- Batch Summary ---")
    print(f"Total: {summary.total}")
    print(f"Completed: {summary.completed}")
    print(f"Failed: {summary.failed}")
    print(f"Total tokens: {summary.total_tokens}")
    print(f"Total duration: {summary.total_duration:.1f}s")
    print(f"Avg duration: {summary.avg_duration:.1f}s")

    print(f"\n--- Results ---")
    for r in summary.results:
        print(f"\n[{r.id}] {r.prompt}")
        if r.success:
            print(f"  Response: {r.response[:200]}")
        else:
            print(f"  Error: {r.error}")


if __name__ == "__main__":
    asyncio.run(main())
