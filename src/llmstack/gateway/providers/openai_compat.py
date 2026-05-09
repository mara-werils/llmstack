"""OpenAI-compatible providers — Groq, Together, Mistral, and any other
provider that implements the OpenAI chat completions API format.

Each provider subclass only needs to set name, base_url, and default models.
"""

from __future__ import annotations

from llmstack.gateway.providers.base import ProviderModel
from llmstack.gateway.providers.openai_provider import OpenAIProvider


class GroqProvider(OpenAIProvider):
    """Groq — ultra-fast inference via OpenAI-compatible API."""

    name = "groq"

    def __init__(self, api_key: str = "", **kwargs):
        super().__init__(api_key=api_key, base_url="https://api.groq.com/openai/v1", **kwargs)
        self._models = [
            ProviderModel(id="llama-3.3-70b-versatile", provider="groq",
                          context_length=128_000, cost_per_m_input=0.59, cost_per_m_output=0.79),
            ProviderModel(id="llama-3.1-8b-instant", provider="groq",
                          context_length=128_000, cost_per_m_input=0.05, cost_per_m_output=0.08),
            ProviderModel(id="gemma2-9b-it", provider="groq",
                          context_length=8192, cost_per_m_input=0.20, cost_per_m_output=0.20),
            ProviderModel(id="mixtral-8x7b-32768", provider="groq",
                          context_length=32_768, cost_per_m_input=0.24, cost_per_m_output=0.24),
            ProviderModel(id="llama-3.2-1b-preview", provider="groq",
                          context_length=128_000, cost_per_m_input=0.04, cost_per_m_output=0.04),
            ProviderModel(id="llama-3.2-3b-preview", provider="groq",
                          context_length=128_000, cost_per_m_input=0.06, cost_per_m_output=0.06),
        ]

    async def chat(self, payload: dict):
        resp = await super().chat(payload)
        resp.provider = "groq"
        return resp


class TogetherProvider(OpenAIProvider):
    """Together AI — wide model selection via OpenAI-compatible API."""

    name = "together"

    def __init__(self, api_key: str = "", **kwargs):
        super().__init__(api_key=api_key, base_url="https://api.together.xyz/v1", **kwargs)
        self._models = [
            ProviderModel(id="meta-llama/Llama-3.3-70B-Instruct-Turbo", provider="together",
                          context_length=128_000, cost_per_m_input=0.88, cost_per_m_output=0.88),
            ProviderModel(id="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo", provider="together",
                          context_length=128_000, cost_per_m_input=0.18, cost_per_m_output=0.18),
            ProviderModel(id="meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo", provider="together",
                          context_length=128_000, cost_per_m_input=3.50, cost_per_m_output=3.50),
            ProviderModel(id="Qwen/Qwen2.5-72B-Instruct-Turbo", provider="together",
                          context_length=128_000, cost_per_m_input=1.20, cost_per_m_output=1.20),
            ProviderModel(id="deepseek-ai/DeepSeek-R1", provider="together",
                          context_length=64_000, cost_per_m_input=3.00, cost_per_m_output=7.00),
            ProviderModel(id="deepseek-ai/DeepSeek-V3", provider="together",
                          context_length=64_000, cost_per_m_input=0.50, cost_per_m_output=0.90),
        ]

    async def chat(self, payload: dict):
        resp = await super().chat(payload)
        resp.provider = "together"
        return resp


class MistralProvider(OpenAIProvider):
    """Mistral AI — via OpenAI-compatible API."""

    name = "mistral"

    def __init__(self, api_key: str = "", **kwargs):
        super().__init__(api_key=api_key, base_url="https://api.mistral.ai/v1", **kwargs)
        self._models = [
            ProviderModel(id="mistral-large-latest", provider="mistral",
                          context_length=128_000, cost_per_m_input=2.00, cost_per_m_output=6.00),
            ProviderModel(id="mistral-medium-latest", provider="mistral",
                          context_length=128_000, cost_per_m_input=2.70, cost_per_m_output=8.10),
            ProviderModel(id="mistral-small-latest", provider="mistral",
                          context_length=128_000, cost_per_m_input=0.20, cost_per_m_output=0.60),
            ProviderModel(id="codestral-latest", provider="mistral",
                          context_length=256_000, cost_per_m_input=0.30, cost_per_m_output=0.90),
            ProviderModel(id="pixtral-large-latest", provider="mistral",
                          context_length=128_000, cost_per_m_input=2.00, cost_per_m_output=6.00),
        ]

    async def chat(self, payload: dict):
        resp = await super().chat(payload)
        resp.provider = "mistral"
        return resp
