"""Anthropic provider — translates OpenAI format to/from Anthropic Messages API."""

from __future__ import annotations

import json
import time
from typing import AsyncIterator

import httpx

from llmstack.gateway.providers.base import (
    Provider,
    ProviderError,
    ProviderModel,
    ProviderResponse,
)

_PRICING: dict[str, tuple[float, float]] = {
    "claude-opus-4-20250514": (15.00, 75.00),
    "claude-sonnet-4-20250514": (3.00, 15.00),
    "claude-haiku-4-20250514": (0.80, 4.00),
    "claude-3-5-sonnet-20241022": (3.00, 15.00),
    "claude-3-5-haiku-20241022": (0.80, 4.00),
}

_DEFAULT_MODELS = [
    ProviderModel(
        id=mid,
        provider="anthropic",
        context_length=200_000,
        cost_per_m_input=p[0],
        cost_per_m_output=p[1],
    )
    for mid, p in _PRICING.items()
]


def _openai_to_anthropic(payload: dict) -> dict:
    """Convert OpenAI chat completion request to Anthropic Messages API format."""
    messages = payload.get("messages", [])

    # Extract system message
    system_parts = []
    api_messages = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            system_parts.append(content)
        else:
            # Anthropic only accepts "user" and "assistant" roles
            api_role = "assistant" if role == "assistant" else "user"
            api_messages.append({"role": api_role, "content": content})

    # Ensure conversation starts with a user message
    if api_messages and api_messages[0]["role"] != "user":
        api_messages.insert(0, {"role": "user", "content": "Hello"})

    # Merge consecutive same-role messages
    merged = []
    for msg in api_messages:
        if merged and merged[-1]["role"] == msg["role"]:
            merged[-1]["content"] += "\n" + msg["content"]
        else:
            merged.append(msg)

    body: dict = {
        "model": payload.get("model", "claude-sonnet-4-20250514"),
        "messages": merged,
        "max_tokens": payload.get("max_tokens") or 4096,
    }

    if system_parts:
        body["system"] = "\n\n".join(system_parts)

    if payload.get("temperature") is not None:
        body["temperature"] = payload["temperature"]
    if payload.get("top_p") is not None:
        body["top_p"] = payload["top_p"]
    if payload.get("stop"):
        body["stop_sequences"] = (
            payload["stop"] if isinstance(payload["stop"], list) else [payload["stop"]]
        )

    return body


def _anthropic_to_openai(result: dict, model: str, latency_ms: float) -> dict:
    """Convert Anthropic Messages API response to OpenAI format."""
    content_blocks = result.get("content", [])
    text = "".join(b.get("text", "") for b in content_blocks if b.get("type") == "text")

    usage = result.get("usage", {})
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)

    return {
        "id": result.get("id", f"chatcmpl-{int(time.time())}"),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": _map_stop_reason(result.get("stop_reason", "end_turn")),
            }
        ],
        "usage": {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    }


def _map_stop_reason(reason: str) -> str:
    return {"end_turn": "stop", "max_tokens": "length", "stop_sequence": "stop"}.get(reason, "stop")


class AnthropicProvider(Provider):
    """Anthropic Claude provider — translates between OpenAI and Messages API."""

    name = "anthropic"

    def __init__(self, api_key: str = "", base_url: str = "", **kwargs):
        super().__init__(
            api_key=api_key,
            base_url=base_url or "https://api.anthropic.com",
            **kwargs,
        )
        self._models = list(_DEFAULT_MODELS)

    def _headers(self) -> dict[str, str]:
        return {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

    async def chat(self, payload: dict) -> ProviderResponse:
        url = f"{self.base_url}/v1/messages"
        body = _openai_to_anthropic(payload)
        t0 = time.monotonic()

        try:
            async with httpx.AsyncClient(timeout=120) as client:
                resp = await client.post(url, json=body, headers=self._headers())
                resp.raise_for_status()
                result = resp.json()
        except httpx.HTTPStatusError as exc:
            raise ProviderError(
                f"Anthropic API error: {exc.response.status_code} {exc.response.text[:200]}",
                status_code=exc.response.status_code,
                retryable=exc.response.status_code in (429, 500, 502, 503, 529),
            ) from exc
        except (httpx.ConnectError, httpx.ReadTimeout) as exc:
            raise ProviderError(f"Anthropic unreachable: {exc}", retryable=True) from exc

        elapsed = (time.monotonic() - t0) * 1000
        model = result.get("model", payload.get("model", ""))
        usage = result.get("usage", {})
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)

        content_blocks = result.get("content", [])
        text = "".join(b.get("text", "") for b in content_blocks if b.get("type") == "text")

        openai_response = _anthropic_to_openai(result, model, elapsed)

        return ProviderResponse(
            content=text,
            model=model,
            provider="anthropic",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=elapsed,
            cost_usd=self.calculate_cost(model, input_tokens, output_tokens),
            raw=openai_response,
        )

    async def chat_stream(self, payload: dict) -> AsyncIterator[bytes]:
        url = f"{self.base_url}/v1/messages"
        body = _openai_to_anthropic(payload)
        body["stream"] = True

        try:
            async with httpx.AsyncClient(timeout=120) as client:
                async with client.stream("POST", url, json=body, headers=self._headers()) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        data = line[6:]
                        if data == "[DONE]":
                            yield b"data: [DONE]\n\n"
                            return

                        try:
                            event = json.loads(data)
                        except json.JSONDecodeError:
                            continue

                        event_type = event.get("type", "")

                        if event_type == "content_block_delta":
                            delta_text = event.get("delta", {}).get("text", "")
                            if delta_text:
                                openai_chunk = {
                                    "id": f"chatcmpl-{int(time.time())}",
                                    "object": "chat.completion.chunk",
                                    "created": int(time.time()),
                                    "model": payload.get("model", ""),
                                    "choices": [
                                        {
                                            "index": 0,
                                            "delta": {"content": delta_text},
                                            "finish_reason": None,
                                        }
                                    ],
                                }
                                yield f"data: {json.dumps(openai_chunk)}\n\n".encode()

                        elif event_type == "message_stop":
                            stop_chunk = {
                                "id": f"chatcmpl-{int(time.time())}",
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": payload.get("model", ""),
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {},
                                        "finish_reason": "stop",
                                    }
                                ],
                            }
                            yield f"data: {json.dumps(stop_chunk)}\n\n".encode()
                            yield b"data: [DONE]\n\n"

        except httpx.HTTPStatusError as exc:
            raise ProviderError(
                f"Anthropic stream error: {exc.response.status_code}",
                retryable=exc.response.status_code in (429, 500, 502, 503, 529),
            ) from exc
        except (httpx.ConnectError, httpx.ReadTimeout) as exc:
            raise ProviderError(f"Anthropic stream error: {exc}", retryable=True) from exc

    async def list_models(self) -> list[ProviderModel]:
        return self._models
