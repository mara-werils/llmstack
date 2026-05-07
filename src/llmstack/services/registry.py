"""Service registry — discovers built-in and plugin services."""

from __future__ import annotations

from importlib.metadata import entry_points
from typing import Type

from llmstack.services.base import ServiceBase
from llmstack.services.inference.ollama import OllamaService
from llmstack.services.inference.vllm import VllmService
from llmstack.services.vectordb.qdrant import QdrantService
from llmstack.services.cache.redis import RedisService
from llmstack.services.embeddings.tei import TEIService


class ServiceRegistry:
    """Discovers all built-in + plugin services."""

    def __init__(self):
        self._services: dict[str, Type[ServiceBase]] = {}
        self._load_builtins()
        self._load_plugins()

    def _load_builtins(self) -> None:
        for cls in [OllamaService, VllmService, QdrantService, RedisService, TEIService]:
            self._services[cls.name] = cls

    def _load_plugins(self) -> None:
        try:
            eps = entry_points(group="llmstack.services")
        except TypeError:
            # Python 3.11 compat
            eps = entry_points().get("llmstack.services", [])

        for ep in eps:
            try:
                cls = ep.load()
                if hasattr(cls, "name"):
                    self._services[cls.name] = cls
            except Exception:
                pass

    def get(self, name: str) -> Type[ServiceBase]:
        if name not in self._services:
            available = ", ".join(sorted(self._services.keys()))
            raise KeyError(f"Unknown service '{name}'. Available: {available}")
        return self._services[name]

    def list_by_category(self, category: str) -> list[Type[ServiceBase]]:
        return [s for s in self._services.values() if s.category == category]

    def all_names(self) -> list[str]:
        return sorted(self._services.keys())
