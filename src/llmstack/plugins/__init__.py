"""Plugin system — discover and load third-party extensions."""

from __future__ import annotations

__all__ = ["PluginSpec", "PluginLoader"]

from llmstack.plugins.registry import PluginSpec, PluginLoader
