"""Proxy layer — forwards requests to inference backends via providers.

Integrates semantic caching (Redis), circuit breaker (resilience),
and the provider registry for multi-provider routing.
"""

from __future__ import annotations

import os
from typing import AsyncIterator

import httpx

from llmstack.gateway.cache import get_cache
from llmstack.gateway.circuit_breaker import get_inference_breaker
from llmstack.gateway.middleware.metrics import record_tokens

_raw_inference = os.getenv("LLMSTACK_INFERENCE_URL", "http://llmstack-ollama:11434/v1")
INFERENCE_URL = _raw_inference.rstrip("/") if _raw_inference.rstrip("/").endswith("/v1") else _raw_inference.rstrip("/") + "/v1"
EMBEDDINGS_URL = os.getenv("LLMSTACK_EMBEDDINGS_URL", "")

# Timeout for inference (can be long for large models)
REQUEST_TIMEOUT = int(os.getenv("LLMSTACK_REQUEST_TIMEOUT", "120"))

# ---------------------------------------------------------------------------
# Persistent connection pool — reused across all requests.
# ---------------------------------------------------------------------------
_pool: httpx.AsyncClient | None = None


def _get_pool() -> httpx.AsyncClient:
    """Return the module-level ``httpx.AsyncClient``, creating it lazily."""
    global _pool
    if _pool is None or _pool.is_closed:
        _pool = httpx.AsyncClient(
            timeout=httpx.Timeout(REQUEST_TIMEOUT, connect=10),
            limits=httpx.Limits(
                max_connections=100,
                max_keepalive_connections=20,
            ),
            http2=True,
        )
    return _pool


async def close_pool() -> None:
    """Gracefully close the connection pool (called during shutdown)."""
    global _pool
    if _pool is not None:
        await _pool.aclose()
        _pool = None


async def proxy_chat_completion(
    payload: dict,
    stream: bool = False,
    provider_name: str | None = None,
) -> dict | AsyncIterator[bytes]:
    """Forward a chat completion request with caching, circuit breaker, and provider routing.

    If ``provider_name`` is set and a provider registry is available,
    the request is dispatched through the provider. Otherwise, falls back
    to direct HTTP forwarding to the local inference backend.
    """
    # Try provider registry first
    if provider_name and provider_name != "local":
        return await _proxy_via_provider(payload, stream, provider_name)

    # Legacy path: direct proxy to local Ollama/vLLM
    breaker = get_inference_breaker()
    model = payload.get("model", "")
    messages = payload.get("messages", [])
    temperature = payload.get("temperature", 1.0)

    # Check circuit breaker first
    breaker.check()  # raises CircuitBreakerError if open

    # Try cache (only for non-streaming, low-temperature requests)
    if not stream:
        cache = await get_cache()
        cached = await cache.get(model, messages, temperature)
        if cached is not None:
            return cached

    url = f"{INFERENCE_URL}/chat/completions"
    timeout = httpx.Timeout(REQUEST_TIMEOUT, connect=10)

    if stream:
        return _stream_response(url, payload, timeout)

    try:
        client = _get_pool()
        resp = await client.post(url, json=payload)
        resp.raise_for_status()
        result = resp.json()

        breaker.record_success()

        # Extract and record token usage
        usage = result.get("usage", {})
        record_tokens(
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
        )

        # Cache the response
        cache = await get_cache()
        await cache.put(model, messages, result, temperature)

        return result

    except httpx.HTTPStatusError as exc:
        if exc.response.status_code >= 500:
            breaker.record_failure()
        raise
    except (httpx.ConnectError, httpx.ReadTimeout, httpx.ConnectTimeout):
        breaker.record_failure()
        raise


async def _proxy_via_provider(
    payload: dict,
    stream: bool,
    provider_name: str,
) -> dict | AsyncIterator[bytes]:
    """Route request through the provider registry."""
    from llmstack.gateway.providers.registry import get_registry

    registry = get_registry()
    if registry is None:
        raise RuntimeError("Provider registry not initialized")

    if stream:
        return registry.stream_with_fallback(payload)

    # Non-streaming: use caching
    model = payload.get("model", "")
    messages = payload.get("messages", [])
    temperature = payload.get("temperature", 1.0)

    cache = await get_cache()
    cached = await cache.get(model, messages, temperature)
    if cached is not None:
        return cached

    result = await registry.chat_with_fallback(payload)

    # Record token usage
    usage = result.get("usage", {})
    record_tokens(
        input_tokens=usage.get("prompt_tokens", 0),
        output_tokens=usage.get("completion_tokens", 0),
    )

    # Cache
    await cache.put(model, messages, result, temperature)

    return result


async def _stream_response(
    url: str,
    payload: dict,
    timeout: httpx.Timeout,  # noqa: ARG001 — kept for signature compat
) -> AsyncIterator[bytes]:
    """Stream SSE chunks from the inference backend with circuit breaker tracking."""
    breaker = get_inference_breaker()
    try:
        client = _get_pool()
        async with client.stream("POST", url, json=payload) as resp:
            resp.raise_for_status()
            breaker.record_success()
            async for chunk in resp.aiter_bytes():
                yield chunk
    except (httpx.ConnectError, httpx.ReadTimeout, httpx.ConnectTimeout):
        breaker.record_failure()
        raise


async def proxy_embeddings(payload: dict) -> dict:
    """Forward an embeddings request to the embedding backend."""
    url = EMBEDDINGS_URL or INFERENCE_URL
    if not url.endswith("/embeddings"):
        url = f"{url}/embeddings"

    client = _get_pool()
    resp = await client.post(url, json=payload)
    resp.raise_for_status()
    return resp.json()


async def proxy_models() -> dict:
    """List available models from the inference backend."""
    url = f"{INFERENCE_URL}/models"
    client = _get_pool()
    resp = await client.get(url)
    resp.raise_for_status()
    return resp.json()
