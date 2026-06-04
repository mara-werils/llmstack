"""Model warm-up — pre-load models on gateway startup for instant first requests.

Sends lightweight probe requests to each configured model during startup
to ensure they are loaded in memory and ready to serve.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

WARMUP_PROMPT = [{"role": "user", "content": "Hi"}]
WARMUP_MAX_TOKENS = 1


@dataclass
class WarmupResult:
    """Result of warming up a single model."""

    model: str
    provider: str = ""
    success: bool = False
    latency_ms: float = 0.0
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "provider": self.provider,
            "success": self.success,
            "latency_ms": round(self.latency_ms, 1),
            "error": self.error,
        }


@dataclass
class WarmupReport:
    """Report of a full warm-up cycle."""

    results: list[WarmupResult] = field(default_factory=list)
    total_time_ms: float = 0.0
    started_at: float = 0.0

    def __post_init__(self):
        if not self.started_at:
            self.started_at = time.time()

    @property
    def success_count(self) -> int:
        return sum(1 for r in self.results if r.success)

    @property
    def failure_count(self) -> int:
        return sum(1 for r in self.results if not r.success)

    def to_dict(self) -> dict:
        return {
            "total_models": len(self.results),
            "succeeded": self.success_count,
            "failed": self.failure_count,
            "total_time_ms": round(self.total_time_ms, 1),
            "results": [r.to_dict() for r in self.results],
        }


async def warmup_model(
    model: str,
    handler,
    provider: str = "",
    timeout: float = 30.0,
) -> WarmupResult:
    """Warm up a single model by sending a probe request.

    Args:
        model: Model name to warm up
        handler: async function(payload) -> response
        provider: Provider name (for logging)
        timeout: Max time to wait for warmup
    """
    t0 = time.monotonic()
    try:
        payload = {
            "model": model,
            "messages": WARMUP_PROMPT,
            "max_tokens": WARMUP_MAX_TOKENS,
            "temperature": 0,
            "stream": False,
        }
        await asyncio.wait_for(handler(payload), timeout=timeout)
        elapsed = (time.monotonic() - t0) * 1000
        logger.info("Model '%s' warmed up in %.1fms", model, elapsed)
        return WarmupResult(
            model=model,
            provider=provider,
            success=True,
            latency_ms=elapsed,
        )
    except asyncio.TimeoutError:
        elapsed = (time.monotonic() - t0) * 1000
        logger.warning("Model '%s' warmup timed out after %.1fms", model, elapsed)
        return WarmupResult(
            model=model,
            provider=provider,
            success=False,
            latency_ms=elapsed,
            error=f"Timeout after {timeout}s",
        )
    except Exception as exc:
        elapsed = (time.monotonic() - t0) * 1000
        logger.warning("Model '%s' warmup failed: %s", model, exc)
        return WarmupResult(
            model=model,
            provider=provider,
            success=False,
            latency_ms=elapsed,
            error=str(exc),
        )


async def warmup_all(
    models: list[dict],
    handler,
    concurrency: int = 3,
) -> WarmupReport:
    """Warm up all configured models.

    Args:
        models: List of {"name": str, "provider": str} dicts
        handler: async function(payload) -> response
        concurrency: Max models to warm up simultaneously
    """
    report = WarmupReport()
    t0 = time.monotonic()

    semaphore = asyncio.Semaphore(concurrency)

    async def _warmup_one(model_cfg: dict) -> WarmupResult:
        async with semaphore:
            return await warmup_model(
                model=model_cfg["name"],
                handler=handler,
                provider=model_cfg.get("provider", ""),
            )

    tasks = [_warmup_one(m) for m in models]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for r in results:
        if isinstance(r, WarmupResult):
            report.results.append(r)
        elif isinstance(r, Exception):
            report.results.append(
                WarmupResult(
                    model="unknown",
                    success=False,
                    error=str(r),
                )
            )

    report.total_time_ms = (time.monotonic() - t0) * 1000

    logger.info(
        "Warmup complete: %d/%d models ready in %.1fms",
        report.success_count,
        len(models),
        report.total_time_ms,
    )

    return report
