"""Base provider interface — all LLM providers implement this ABC."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncIterator


class ProviderError(Exception):
    """Raised when a provider request fails."""

    def __init__(self, message: str, status_code: int = 502, retryable: bool = True):
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable


@dataclass
class ProviderModel:
    """A model available through a provider."""

    id: str                         # e.g. "gpt-4o", "claude-sonnet-4-20250514"
    provider: str                   # e.g. "openai", "anthropic"
    display_name: str | None = None
    context_length: int = 8192
    cost_per_m_input: float = 0.0   # $ per 1M input tokens
    cost_per_m_output: float = 0.0  # $ per 1M output tokens


@dataclass
class ProviderResponse:
    """Normalized response from any provider."""

    content: str
    model: str
    provider: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    raw: dict = field(default_factory=dict)  # full OpenAI-format response

    def to_openai_dict(self) -> dict:
        """Return an OpenAI-compatible response dict."""
        return self.raw if self.raw else {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": self.content},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": self.input_tokens,
                "completion_tokens": self.output_tokens,
                "total_tokens": self.input_tokens + self.output_tokens,
            },
            "x_llmstack": {
                "provider": self.provider,
                "cost_usd": self.cost_usd,
                "latency_ms": round(self.latency_ms, 1),
            },
        }


class Provider(ABC):
    """Abstract base class for LLM providers.

    Each provider translates OpenAI-format requests to its native API
    and returns normalized ProviderResponse objects.
    """

    name: str = "base"

    def __init__(self, api_key: str = "", base_url: str = "", **kwargs):
        self.api_key = api_key
        self.base_url = base_url
        self._models: list[ProviderModel] = []

    @abstractmethod
    async def chat(self, payload: dict) -> ProviderResponse:
        """Send a chat completion request and return a normalized response."""

    @abstractmethod
    async def chat_stream(self, payload: dict) -> AsyncIterator[bytes]:
        """Stream a chat completion, yielding SSE-formatted bytes."""

    @abstractmethod
    async def list_models(self) -> list[ProviderModel]:
        """Return models available from this provider."""

    def get_model_cost(self, model_id: str) -> tuple[float, float]:
        """Return (cost_per_m_input, cost_per_m_output) for a model."""
        for m in self._models:
            if m.id == model_id:
                return m.cost_per_m_input, m.cost_per_m_output
        return 0.0, 0.0

    def calculate_cost(self, model_id: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost in USD for a request."""
        ci, co = self.get_model_cost(model_id)
        return (input_tokens * ci + output_tokens * co) / 1_000_000

    async def health_check(self) -> bool:
        """Check if the provider is reachable."""
        try:
            await self.list_models()
            return True
        except Exception:
            return False
