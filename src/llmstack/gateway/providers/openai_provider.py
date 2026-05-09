"""OpenAI provider — passthrough since the gateway already speaks OpenAI format."""

from __future__ import annotations

import time
from typing import AsyncIterator

import httpx

from llmstack.gateway.providers.base import (
    Provider,
    ProviderError,
    ProviderModel,
    ProviderResponse,
)

# Pricing per 1M tokens (USD) — May 2025
_PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "o3": (2.00, 8.00),
    "o3-mini": (1.10, 4.40),
    "o4-mini": (1.10, 4.40),
    "gpt-4-turbo": (10.00, 30.00),
    "gpt-3.5-turbo": (0.50, 1.50),
}

_DEFAULT_MODELS = [
    ProviderModel(id=mid, provider="openai", context_length=128_000,
                  cost_per_m_input=p[0], cost_per_m_output=p[1])
    for mid, p in _PRICING.items()
]


class OpenAIProvider(Provider):
    """OpenAI API provider — direct passthrough (API is already OpenAI-format)."""

    name = "openai"

    def __init__(self, api_key: str = "", base_url: str = "", **kwargs):
        super().__init__(
            api_key=api_key,
            base_url=base_url or "https://api.openai.com/v1",
            **kwargs,
        )
        self._models = list(_DEFAULT_MODELS)

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def chat(self, payload: dict) -> ProviderResponse:
        url = f"{self.base_url}/chat/completions"
        payload = {k: v for k, v in payload.items() if not k.startswith("x_")}
        t0 = time.monotonic()

        try:
            async with httpx.AsyncClient(timeout=120) as client:
                resp = await client.post(url, json=payload, headers=self._headers())
                resp.raise_for_status()
                result = resp.json()
        except httpx.HTTPStatusError as exc:
            raise ProviderError(
                f"OpenAI API error: {exc.response.status_code} {exc.response.text[:200]}",
                status_code=exc.response.status_code,
                retryable=exc.response.status_code in (429, 500, 502, 503),
            ) from exc
        except (httpx.ConnectError, httpx.ReadTimeout) as exc:
            raise ProviderError(f"OpenAI unreachable: {exc}", retryable=True) from exc

        elapsed = (time.monotonic() - t0) * 1000
        usage = result.get("usage", {})
        model = result.get("model", payload.get("model", ""))
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)

        return ProviderResponse(
            content=result.get("choices", [{}])[0].get("message", {}).get("content", ""),
            model=model,
            provider="openai",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=elapsed,
            cost_usd=self.calculate_cost(model, input_tokens, output_tokens),
            raw=result,
        )

    async def chat_stream(self, payload: dict) -> AsyncIterator[bytes]:
        url = f"{self.base_url}/chat/completions"
        payload = {k: v for k, v in payload.items() if not k.startswith("x_")}
        payload["stream"] = True

        try:
            async with httpx.AsyncClient(timeout=120) as client:
                async with client.stream("POST", url, json=payload, headers=self._headers()) as resp:
                    resp.raise_for_status()
                    async for chunk in resp.aiter_bytes():
                        yield chunk
        except httpx.HTTPStatusError as exc:
            raise ProviderError(
                f"OpenAI stream error: {exc.response.status_code}",
                retryable=exc.response.status_code in (429, 500, 502, 503),
            ) from exc
        except (httpx.ConnectError, httpx.ReadTimeout) as exc:
            raise ProviderError(f"OpenAI stream unreachable: {exc}", retryable=True) from exc

    async def list_models(self) -> list[ProviderModel]:
        return self._models
