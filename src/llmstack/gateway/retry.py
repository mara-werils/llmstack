"""Smart retry with provider fallback — automatic retry with exponential backoff
and provider failover for resilient LLM request handling.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    max_retries: int = 3
    initial_delay_ms: float = 500
    max_delay_ms: float = 10000
    exponential_base: float = 2.0
    jitter: bool = True
    retry_on_status: set[int] = field(default_factory=lambda: {429, 500, 502, 503, 504})
    retry_on_exceptions: tuple = field(
        default_factory=lambda: (
            ConnectionError,
            TimeoutError,
            asyncio.TimeoutError,
        )
    )


@dataclass
class FallbackChain:
    """An ordered list of providers to try for failover."""

    providers: list[str] = field(default_factory=list)
    current_index: int = 0

    def next_provider(self) -> str | None:
        """Get the next provider in the chain."""
        if self.current_index >= len(self.providers):
            return None
        provider = self.providers[self.current_index]
        self.current_index += 1
        return provider

    def reset(self) -> None:
        self.current_index = 0

    @property
    def exhausted(self) -> bool:
        return self.current_index >= len(self.providers)


@dataclass
class RetryAttempt:
    """Record of a single retry attempt."""

    attempt: int
    provider: str
    status_code: int = 0
    error: str = ""
    latency_ms: float = 0.0
    success: bool = False
    timestamp: float = 0.0

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.time()

    def to_dict(self) -> dict:
        return {
            "attempt": self.attempt,
            "provider": self.provider,
            "status_code": self.status_code,
            "error": self.error,
            "latency_ms": round(self.latency_ms, 1),
            "success": self.success,
        }


@dataclass
class RetryResult:
    """Result of a retry sequence."""

    success: bool = False
    response: Any = None
    attempts: list[RetryAttempt] = field(default_factory=list)
    total_latency_ms: float = 0.0
    final_provider: str = ""

    @property
    def total_attempts(self) -> int:
        return len(self.attempts)

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "total_attempts": self.total_attempts,
            "total_latency_ms": round(self.total_latency_ms, 1),
            "final_provider": self.final_provider,
            "attempts": [a.to_dict() for a in self.attempts],
        }


def _compute_delay(attempt: int, config: RetryConfig) -> float:
    """Compute delay with exponential backoff and optional jitter."""
    delay = config.initial_delay_ms * (config.exponential_base**attempt)
    delay = min(delay, config.max_delay_ms)
    if config.jitter:
        import random

        delay *= random.uniform(0.5, 1.5)
    return delay


async def retry_with_fallback(
    handler: Callable,
    payload: dict,
    config: RetryConfig | None = None,
    fallback_providers: list[str] | None = None,
) -> RetryResult:
    """Execute a request with retry and provider fallback.

    Args:
        handler: async function(payload, provider=None) -> response
        payload: the request payload
        config: retry configuration
        fallback_providers: ordered list of providers to try on failure

    Returns:
        RetryResult with success/failure and all attempt records
    """
    if config is None:
        config = RetryConfig()

    chain = FallbackChain(providers=fallback_providers or [])
    result = RetryResult()
    t0 = time.monotonic()
    current_provider = fallback_providers[0] if fallback_providers else ""

    for attempt in range(config.max_retries + 1):
        attempt_t0 = time.monotonic()

        try:
            response = await handler(payload, provider=current_provider)

            # Check for error status codes in response
            status = 200
            if isinstance(response, dict):
                error = response.get("error", {})
                if isinstance(error, dict) and error.get("type") == "server_error":
                    status = error.get("status_code", 500)

            attempt_record = RetryAttempt(
                attempt=attempt + 1,
                provider=current_provider,
                status_code=status,
                latency_ms=(time.monotonic() - attempt_t0) * 1000,
                success=status < 400,
            )
            result.attempts.append(attempt_record)

            if status < 400:
                result.success = True
                result.response = response
                result.final_provider = current_provider
                break

            # Retryable status code
            if status not in config.retry_on_status:
                break

        except config.retry_on_exceptions as exc:
            attempt_record = RetryAttempt(
                attempt=attempt + 1,
                provider=current_provider,
                error=str(exc),
                latency_ms=(time.monotonic() - attempt_t0) * 1000,
            )
            result.attempts.append(attempt_record)
            logger.warning(
                "Retry attempt %d failed: %s (provider=%s)",
                attempt + 1,
                exc,
                current_provider,
            )
        except Exception as exc:
            # Non-retryable exception
            attempt_record = RetryAttempt(
                attempt=attempt + 1,
                provider=current_provider,
                error=str(exc),
                latency_ms=(time.monotonic() - attempt_t0) * 1000,
            )
            result.attempts.append(attempt_record)
            break

        # Try next provider in fallback chain
        next_provider = chain.next_provider()
        if next_provider and next_provider != current_provider:
            current_provider = next_provider
            logger.info("Falling back to provider: %s", current_provider)

        # Wait before retry (skip wait for provider switch)
        if attempt < config.max_retries:
            delay_ms = _compute_delay(attempt, config)
            await asyncio.sleep(delay_ms / 1000)

    result.total_latency_ms = (time.monotonic() - t0) * 1000
    return result
