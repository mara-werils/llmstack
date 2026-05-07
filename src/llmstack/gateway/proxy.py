"""Proxy layer — forwards requests to inference and embedding backends.

Integrates semantic caching (Redis) and circuit breaker (resilience).
"""

from __future__ import annotations

import os
from typing import AsyncIterator

import httpx

from llmstack.gateway.cache import get_cache
from llmstack.gateway.circuit_breaker import get_inference_breaker
from llmstack.gateway.middleware.metrics import record_tokens

INFERENCE_URL = os.getenv("LLMSTACK_INFERENCE_URL", "http://llmstack-ollama:11434/v1")
EMBEDDINGS_URL = os.getenv("LLMSTACK_EMBEDDINGS_URL", "")

# Timeout for inference (can be long for large models)
REQUEST_TIMEOUT = int(os.getenv("LLMSTACK_REQUEST_TIMEOUT", "120"))


async def proxy_chat_completion(payload: dict, stream: bool = False) -> dict | AsyncIterator[bytes]:
    """Forward a chat completion request with caching and circuit breaker."""
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
        async with httpx.AsyncClient(timeout=timeout) as client:
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


async def _stream_response(url: str, payload: dict, timeout: httpx.Timeout) -> AsyncIterator[bytes]:
    """Stream SSE chunks from the inference backend with circuit breaker tracking."""
    breaker = get_inference_breaker()
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
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

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(url, json=payload)
        resp.raise_for_status()
        return resp.json()


async def proxy_models() -> dict:
    """List available models from the inference backend."""
    url = f"{INFERENCE_URL}/models"
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        return resp.json()
