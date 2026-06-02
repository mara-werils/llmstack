"""Batch processing — run multiple prompts concurrently with progress tracking."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass


@dataclass
class BatchItem:
    """A single item in a batch."""
    id: str | int
    prompt: str
    system_prompt: str = ""
    model: str | None = None
    max_tokens: int = 1024
    temperature: float = 0.7


@dataclass
class BatchResult:
    """Result of a single batch item."""
    id: str | int
    prompt: str
    response: str
    tokens_used: int
    duration: float
    success: bool
    error: str | None = None


@dataclass
class BatchSummary:
    """Summary of a batch run."""
    total: int
    completed: int
    failed: int
    total_tokens: int
    total_duration: float
    avg_duration: float
    results: list[BatchResult]


class BatchProcessor:
    """Process multiple prompts concurrently."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: str | None = None,
        concurrency: int = 5,
        model: str = "llama3.2",
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.concurrency = concurrency
        self.default_model = model
        self._progress_callback = None

    def on_progress(self, callback) -> "BatchProcessor":
        """Set progress callback: callback(completed, total, item_result)."""
        self._progress_callback = callback
        return self

    async def run(self, items: list[BatchItem]) -> BatchSummary:
        """Run batch processing."""
        import httpx

        semaphore = asyncio.Semaphore(self.concurrency)
        results: list[BatchResult] = []
        start_time = time.time()
        completed = 0

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        async def process_item(item: BatchItem) -> BatchResult:
            nonlocal completed
            async with semaphore:
                item_start = time.time()
                try:
                    timeout = httpx.Timeout(120, connect=10, read=120, write=30)
                    async with httpx.AsyncClient(timeout=timeout) as client:
                        payload = {
                            "model": item.model or self.default_model,
                            "messages": [],
                            "max_tokens": item.max_tokens,
                            "temperature": item.temperature,
                        }
                        if item.system_prompt:
                            payload["messages"].append({"role": "system", "content": item.system_prompt})
                        payload["messages"].append({"role": "user", "content": item.prompt})

                        resp = await client.post(
                            f"{self.base_url}/v1/chat/completions",
                            json=payload, headers=headers,
                        )

                        if resp.status_code == 200:
                            data = resp.json()
                            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                            tokens = data.get("usage", {}).get("total_tokens", 0)
                            result = BatchResult(
                                id=item.id, prompt=item.prompt,
                                response=content, tokens_used=tokens,
                                duration=time.time() - item_start,
                                success=True,
                            )
                        else:
                            result = BatchResult(
                                id=item.id, prompt=item.prompt,
                                response="", tokens_used=0,
                                duration=time.time() - item_start,
                                success=False, error=f"HTTP {resp.status_code}",
                            )
                except Exception as e:
                    result = BatchResult(
                        id=item.id, prompt=item.prompt,
                        response="", tokens_used=0,
                        duration=time.time() - item_start,
                        success=False, error=str(e),
                    )

                completed += 1
                if self._progress_callback:
                    self._progress_callback(completed, len(items), result)
                return result

        tasks = [process_item(item) for item in items]
        results = await asyncio.gather(*tasks)

        total_duration = time.time() - start_time
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        total_tokens = sum(r.tokens_used for r in results)

        return BatchSummary(
            total=len(items),
            completed=len(successful),
            failed=len(failed),
            total_tokens=total_tokens,
            total_duration=total_duration,
            avg_duration=total_duration / max(1, len(items)),
            results=results,
        )

    def run_sync(self, items: list[BatchItem]) -> BatchSummary:
        """Synchronous wrapper for batch processing."""
        return asyncio.run(self.run(items))
