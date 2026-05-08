"""LLMStack — Stop running 70B for 'Hello'. Smart model routing for local LLMs."""

__version__ = "0.5.0"

from llmstack.sdk import Client, AsyncClient

__all__ = ["__version__", "Client", "AsyncClient"]
