"""Proxy layer — forwards requests to inference and embedding backends."""

from __future__ import annotations

import os
from typing import AsyncIterator

import httpx

INFERENCE_URL = os.getenv("LLMSTACK_INFERENCE_URL", "http://llmstack-ollama:11434/v1")
EMBEDDINGS_URL = os.getenv("LLMSTACK_EMBEDDINGS_URL", "")

# Timeout for inference (can be long for large models)
REQUEST_TIMEOUT = int(os.getenv("LLMSTACK_REQUEST_TIMEOUT", "120"))


async def proxy_chat_completion(payload: dict, stream: bool = False) -> dict | AsyncIterator[bytes]:
    """Forward a chat completion request to the inference backend."""
    url = f"{INFERENCE_URL}/chat/completions"
    timeout = httpx.Timeout(REQUEST_TIMEOUT, connect=10)

    if stream:
        return _stream_response(url, payload, timeout)
    else:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            return resp.json()


async def _stream_response(url: str, payload: dict, timeout: httpx.Timeout) -> AsyncIterator[bytes]:
    """Stream SSE chunks from the inference backend."""
    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream("POST", url, json=payload) as resp:
            resp.raise_for_status()
            async for chunk in resp.aiter_bytes():
                yield chunk


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
