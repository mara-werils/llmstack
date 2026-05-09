"""Local provider — wraps Ollama/vLLM backends via the existing proxy layer."""

from __future__ import annotations

import os
import time
from typing import AsyncIterator

import httpx

from llmstack.gateway.providers.base import (
    Provider,
    ProviderError,
    ProviderModel,
    ProviderResponse,
)

_raw = os.getenv("LLMSTACK_INFERENCE_URL", "http://llmstack-ollama:11434/v1")
_DEFAULT_URL = _raw.rstrip("/") if _raw.rstrip("/").endswith("/v1") else _raw.rstrip("/") + "/v1"
_TIMEOUT = int(os.getenv("LLMSTACK_REQUEST_TIMEOUT", "120"))


class LocalProvider(Provider):
    """Provider that forwards to a local Ollama or vLLM backend."""

    name = "local"

    def __init__(self, base_url: str = "", **kwargs):
        super().__init__(base_url=base_url or _DEFAULT_URL, **kwargs)

    async def chat(self, payload: dict) -> ProviderResponse:
        url = f"{self.base_url}/chat/completions"
        timeout = httpx.Timeout(_TIMEOUT, connect=10)
        t0 = time.monotonic()

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
                result = resp.json()
        except httpx.HTTPStatusError as exc:
            raise ProviderError(
                f"Local backend error: {exc.response.status_code}",
                status_code=exc.response.status_code,
                retryable=exc.response.status_code >= 500,
            ) from exc
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.ConnectTimeout) as exc:
            raise ProviderError(f"Local backend unreachable: {exc}", retryable=True) from exc

        elapsed = (time.monotonic() - t0) * 1000
        usage = result.get("usage", {})

        return ProviderResponse(
            content=result.get("choices", [{}])[0].get("message", {}).get("content", ""),
            model=result.get("model", payload.get("model", "")),
            provider="local",
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            latency_ms=elapsed,
            cost_usd=0.0,
            raw=result,
        )

    async def chat_stream(self, payload: dict) -> AsyncIterator[bytes]:
        url = f"{self.base_url}/chat/completions"
        payload = {**payload, "stream": True}
        timeout = httpx.Timeout(_TIMEOUT, connect=10)

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream("POST", url, json=payload) as resp:
                    resp.raise_for_status()
                    async for chunk in resp.aiter_bytes():
                        yield chunk
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.ConnectTimeout) as exc:
            raise ProviderError(f"Local stream error: {exc}", retryable=True) from exc

    async def list_models(self) -> list[ProviderModel]:
        url = f"{self.base_url}/models"
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                data = resp.json()
        except Exception:
            return self._models

        models = []
        for m in data.get("data", []):
            models.append(ProviderModel(
                id=m.get("id", ""),
                provider="local",
                context_length=m.get("context_length", 8192),
                cost_per_m_input=0.0,
                cost_per_m_output=0.0,
            ))
        self._models = models
        return models
