"""Retry logic for SDK HTTP requests."""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass

import httpx


@dataclass
class RetryConfig:
    """Configuration for request retry behavior."""

    max_retries: int = 3
    retry_on_status: tuple[int, ...] = (429, 500, 502, 503, 504)
    initial_delay: float = 0.5
    max_delay: float = 30.0
    backoff_factor: float = 2.0
    jitter: bool = True


def _calculate_delay(
    attempt: int,
    config: RetryConfig,
    retry_after: float | None = None,
) -> float:
    """Calculate delay before next retry with exponential backoff."""
    if retry_after is not None:
        return min(retry_after, config.max_delay)

    delay = config.initial_delay * (config.backoff_factor**attempt)
    delay = min(delay, config.max_delay)

    if config.jitter:
        delay = delay * (0.5 + random.random() * 0.5)  # noqa: S311

    return delay


def _get_retry_after(response: httpx.Response) -> float | None:
    """Extract Retry-After header value."""
    retry_after = response.headers.get("Retry-After")
    if retry_after is None:
        return None
    try:
        return float(retry_after)
    except (ValueError, TypeError):
        return None


def sync_retry(
    func,  # noqa: ANN001
    config: RetryConfig,
    *args,  # noqa: ANN002
    **kwargs,  # noqa: ANN003
) -> httpx.Response:
    """Execute a sync HTTP request with retry logic."""
    last_exc: Exception | None = None

    for attempt in range(config.max_retries + 1):
        try:
            response = func(*args, **kwargs)
            if response.status_code not in config.retry_on_status:
                return response
            if attempt == config.max_retries:
                return response

            delay = _calculate_delay(attempt, config, _get_retry_after(response))
            time.sleep(delay)

        except (
            httpx.ConnectError,
            httpx.ReadTimeout,
            httpx.ConnectTimeout,
        ) as exc:
            last_exc = exc
            if attempt == config.max_retries:
                raise
            delay = _calculate_delay(attempt, config)
            time.sleep(delay)

    raise last_exc or RuntimeError("Retry exhausted")


async def async_retry(
    func,  # noqa: ANN001
    config: RetryConfig,
    *args,  # noqa: ANN002
    **kwargs,  # noqa: ANN003
) -> httpx.Response:
    """Execute an async HTTP request with retry logic."""
    last_exc: Exception | None = None

    for attempt in range(config.max_retries + 1):
        try:
            response = await func(*args, **kwargs)
            if response.status_code not in config.retry_on_status:
                return response
            if attempt == config.max_retries:
                return response

            delay = _calculate_delay(attempt, config, _get_retry_after(response))
            await asyncio.sleep(delay)

        except (
            httpx.ConnectError,
            httpx.ReadTimeout,
            httpx.ConnectTimeout,
        ) as exc:
            last_exc = exc
            if attempt == config.max_retries:
                raise
            delay = _calculate_delay(attempt, config)
            await asyncio.sleep(delay)

    raise last_exc or RuntimeError("Retry exhausted")
