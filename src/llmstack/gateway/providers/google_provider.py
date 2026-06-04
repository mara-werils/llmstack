"""Google Gemini provider — translates OpenAI format to/from Gemini API."""

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
    "gemini-2.5-pro": (1.25, 10.00),
    "gemini-2.5-flash": (0.15, 0.60),
    "gemini-2.0-flash": (0.10, 0.40),
    "gemini-1.5-pro": (1.25, 5.00),
    "gemini-1.5-flash": (0.075, 0.30),
}

_DEFAULT_MODELS = [
    ProviderModel(
        id=mid,
        provider="google",
        context_length=1_000_000,
        cost_per_m_input=p[0],
        cost_per_m_output=p[1],
    )
    for mid, p in _PRICING.items()
]


def _openai_to_gemini(payload: dict, api_key: str) -> tuple[str, dict]:
    """Convert OpenAI chat request to Gemini generateContent format."""
    messages = payload.get("messages", [])
    model = payload.get("model", "gemini-2.5-flash")

    system_instruction = None
    contents = []

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role == "system":
            system_instruction = {"parts": [{"text": content}]}
        else:
            gemini_role = "model" if role == "assistant" else "user"
            contents.append({"role": gemini_role, "parts": [{"text": content}]})

    body: dict = {"contents": contents}

    if system_instruction:
        body["systemInstruction"] = system_instruction

    generation_config = {}
    if payload.get("temperature") is not None:
        generation_config["temperature"] = payload["temperature"]
    if payload.get("max_tokens"):
        generation_config["maxOutputTokens"] = payload["max_tokens"]
    if payload.get("top_p") is not None:
        generation_config["topP"] = payload["top_p"]
    if payload.get("stop"):
        stops = payload["stop"] if isinstance(payload["stop"], list) else [payload["stop"]]
        generation_config["stopSequences"] = stops

    if generation_config:
        body["generationConfig"] = generation_config

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}"
        f":generateContent?key={api_key}"
    )
    return url, body


class GoogleProvider(Provider):
    """Google Gemini provider."""

    name = "google"

    def __init__(self, api_key: str = "", **kwargs):
        super().__init__(api_key=api_key, **kwargs)
        self._models = list(_DEFAULT_MODELS)

    async def chat(self, payload: dict) -> ProviderResponse:
        url, body = _openai_to_gemini(payload, self.api_key)
        t0 = time.monotonic()

        try:
            async with httpx.AsyncClient(timeout=120) as client:
                resp = await client.post(url, json=body)
                resp.raise_for_status()
                result = resp.json()
        except httpx.HTTPStatusError as exc:
            raise ProviderError(
                f"Google API error: {exc.response.status_code} {exc.response.text[:200]}",
                status_code=exc.response.status_code,
                retryable=exc.response.status_code in (429, 500, 502, 503),
            ) from exc
        except (httpx.ConnectError, httpx.ReadTimeout) as exc:
            raise ProviderError(f"Google unreachable: {exc}", retryable=True) from exc

        elapsed = (time.monotonic() - t0) * 1000
        model = payload.get("model", "gemini-2.5-flash")

        # Extract text from Gemini response
        candidates = result.get("candidates", [])
        text = ""
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            text = "".join(p.get("text", "") for p in parts)

        usage_meta = result.get("usageMetadata", {})
        input_tokens = usage_meta.get("promptTokenCount", 0)
        output_tokens = usage_meta.get("candidatesTokenCount", 0)

        openai_response = {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            },
        }

        return ProviderResponse(
            content=text,
            model=model,
            provider="google",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=elapsed,
            cost_usd=self.calculate_cost(model, input_tokens, output_tokens),
            raw=openai_response,
        )

    async def chat_stream(self, payload: dict) -> AsyncIterator[bytes]:
        model = payload.get("model", "gemini-2.5-flash")
        messages = payload.get("messages", [])

        system_instruction = None
        contents = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                system_instruction = {"parts": [{"text": content}]}
            else:
                gemini_role = "model" if role == "assistant" else "user"
                contents.append({"role": gemini_role, "parts": [{"text": content}]})

        body: dict = {"contents": contents}
        if system_instruction:
            body["systemInstruction"] = system_instruction

        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}"
            f":streamGenerateContent?alt=sse&key={self.api_key}"
        )

        try:
            async with httpx.AsyncClient(timeout=120) as client:
                async with client.stream("POST", url, json=body) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        data = line[6:]
                        try:
                            event = json.loads(data)
                        except json.JSONDecodeError:
                            continue

                        candidates = event.get("candidates", [])
                        if not candidates:
                            continue
                        parts = candidates[0].get("content", {}).get("parts", [])
                        text = "".join(p.get("text", "") for p in parts)

                        if text:
                            chunk = {
                                "id": f"chatcmpl-{int(time.time())}",
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": model,
                                "choices": [
                                    {"index": 0, "delta": {"content": text}, "finish_reason": None}
                                ],
                            }
                            yield f"data: {json.dumps(chunk)}\n\n".encode()

            # Final chunk
            stop = {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(stop)}\n\n".encode()
            yield b"data: [DONE]\n\n"

        except httpx.HTTPStatusError as exc:
            raise ProviderError(
                f"Google stream error: {exc.response.status_code}",
                retryable=exc.response.status_code in (429, 500, 502, 503),
            ) from exc

    async def list_models(self) -> list[ProviderModel]:
        return self._models
