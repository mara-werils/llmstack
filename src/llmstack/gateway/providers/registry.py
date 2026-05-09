"""Provider registry — manages provider instances and model-to-provider mapping."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from llmstack.gateway.providers.base import Provider, ProviderError, ProviderModel

logger = logging.getLogger(__name__)

_registry: ProviderRegistry | None = None


@dataclass
class FallbackChain:
    """Ordered list of (provider, model) pairs to try for a request."""

    steps: list[tuple[str, str]]  # [(provider_name, model_id), ...]

    def __len__(self) -> int:
        return len(self.steps)


class ProviderRegistry:
    """Central registry for all configured LLM providers.

    Responsibilities:
    - Stores provider instances
    - Maps model IDs to their provider
    - Builds fallback chains for resilience
    - Aggregates models from all providers
    """

    def __init__(self):
        self._providers: dict[str, Provider] = {}
        self._model_map: dict[str, str] = {}  # model_id -> provider_name
        self._fallbacks: dict[str, list[str]] = {}  # provider -> [fallback_providers]
        self._all_models: list[ProviderModel] = []

    def register(self, provider: Provider) -> None:
        """Register a provider instance."""
        self._providers[provider.name] = provider
        logger.info("Registered provider: %s", provider.name)

    def get_provider(self, name: str) -> Provider | None:
        """Get a provider by name."""
        return self._providers.get(name)

    def get_provider_for_model(self, model_id: str) -> Provider | None:
        """Look up which provider serves a given model."""
        provider_name = self._model_map.get(model_id)
        if provider_name:
            return self._providers.get(provider_name)
        return None

    def register_model(self, model_id: str, provider_name: str) -> None:
        """Map a model ID to its provider."""
        self._model_map[model_id] = provider_name

    def set_fallbacks(self, provider: str, fallbacks: list[str]) -> None:
        """Configure fallback providers for a given provider."""
        self._fallbacks[provider] = fallbacks

    def get_fallback_chain(self, model_id: str) -> FallbackChain:
        """Build a fallback chain for a model.

        First tries the primary provider, then walks through configured
        fallbacks, mapping to equivalent-tier models.
        """
        primary_provider = self._model_map.get(model_id)
        if not primary_provider:
            return FallbackChain(steps=[])

        steps = [(primary_provider, model_id)]

        fallback_providers = self._fallbacks.get(primary_provider, [])
        for fb_name in fallback_providers:
            fb_provider = self._providers.get(fb_name)
            if fb_provider is None:
                continue
            # Find a model from the fallback provider
            fb_models = [m for m in self._all_models if m.provider == fb_name]
            if fb_models:
                # Pick the first available model (could be smarter with tier matching)
                steps.append((fb_name, fb_models[0].id))

        return FallbackChain(steps=steps)

    async def refresh_models(self) -> list[ProviderModel]:
        """Fetch model lists from all providers and rebuild the model map."""
        self._all_models = []
        for name, provider in self._providers.items():
            try:
                models = await provider.list_models()
                for m in models:
                    m.provider = name
                    self._model_map[m.id] = name
                self._all_models.extend(models)
            except Exception:
                logger.warning("Failed to list models from provider %s", name)
        return self._all_models

    def all_models(self) -> list[ProviderModel]:
        """Return all known models across providers."""
        return list(self._all_models)

    def all_providers(self) -> dict[str, Provider]:
        """Return all registered providers."""
        return dict(self._providers)

    async def chat_with_fallback(self, payload: dict) -> dict:
        """Execute a chat request with automatic failover.

        Tries the primary provider first, then walks the fallback chain
        on retryable errors.
        """
        model_id = payload.get("model", "")
        chain = self.get_fallback_chain(model_id)

        if not chain.steps:
            # No provider found — try to route by model prefix
            provider = self._guess_provider(model_id)
            if provider:
                chain = FallbackChain(steps=[(provider.name, model_id)])
            else:
                raise ProviderError(f"No provider found for model '{model_id}'", retryable=False)

        last_error: Exception | None = None
        for provider_name, fallback_model in chain.steps:
            provider = self._providers.get(provider_name)
            if provider is None:
                continue

            try:
                request = {**payload, "model": fallback_model}
                response = await provider.chat(request)
                result = response.to_openai_dict()
                result["x_llmstack"] = {
                    "provider": provider_name,
                    "model": fallback_model,
                    "cost_usd": response.cost_usd,
                    "latency_ms": round(response.latency_ms, 1),
                    "fallback": fallback_model != model_id,
                }
                return result
            except ProviderError as exc:
                last_error = exc
                if not exc.retryable:
                    raise
                logger.warning(
                    "Provider %s failed for model %s, trying fallback: %s",
                    provider_name, fallback_model, exc,
                )
                continue
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Provider %s error: %s, trying fallback", provider_name, exc,
                )
                continue

        raise ProviderError(
            f"All providers failed for model '{model_id}': {last_error}",
            retryable=False,
        )

    async def stream_with_fallback(self, payload: dict):
        """Stream a chat request with automatic failover."""
        model_id = payload.get("model", "")
        chain = self.get_fallback_chain(model_id)

        if not chain.steps:
            provider = self._guess_provider(model_id)
            if provider:
                chain = FallbackChain(steps=[(provider.name, model_id)])
            else:
                raise ProviderError(f"No provider found for model '{model_id}'", retryable=False)

        last_error: Exception | None = None
        for provider_name, fallback_model in chain.steps:
            provider = self._providers.get(provider_name)
            if provider is None:
                continue

            try:
                request = {**payload, "model": fallback_model}
                async for chunk in provider.chat_stream(request):
                    yield chunk
                return
            except ProviderError as exc:
                last_error = exc
                if not exc.retryable:
                    raise
                logger.warning(
                    "Stream: provider %s failed, trying fallback: %s",
                    provider_name, exc,
                )
                continue
            except Exception as exc:
                last_error = exc
                continue

        raise ProviderError(
            f"All providers failed streaming for model '{model_id}': {last_error}",
            retryable=False,
        )

    def _guess_provider(self, model_id: str) -> Provider | None:
        """Guess provider from model ID prefix (e.g. gpt- -> openai)."""
        prefixes = {
            "gpt-": "openai", "o1": "openai", "o3": "openai", "o4": "openai",
            "claude-": "anthropic",
            "gemini-": "google",
            "mistral-": "mistral", "codestral": "mistral", "pixtral": "mistral",
            "llama": "groq", "mixtral": "groq",
        }
        for prefix, provider_name in prefixes.items():
            if model_id.startswith(prefix) or model_id.startswith(prefix):
                return self._providers.get(provider_name)
        return None


def init_registry(registry: ProviderRegistry) -> None:
    global _registry
    _registry = registry


def get_registry() -> ProviderRegistry | None:
    return _registry
