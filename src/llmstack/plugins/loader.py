"""Plugin loader — re-exports the registry for convenience."""

from llmstack.services.registry import ServiceRegistry

__all__ = ["ServiceRegistry"]
