"""Router preset — multi-model setup with smart routing.

Configures three model tiers (small / medium / large) and enables the
smart model router to automatically select the optimal model per query.
"""

from llmstack.config.schema import StackConfig, ModelsConfig, ModelSpec, EmbeddingSpec


def router_preset() -> dict:
    """Multi-model setup with smart routing.

    Returns a plain dict (not a StackConfig) because the router
    configuration extends the schema with fields that only the
    gateway interprets at runtime.
    """
    return {
        "version": "1",
        "models": {
            "chat": [
                {"name": "llama3.2:1b", "tier": "simple", "backend": "ollama"},
                {"name": "llama3.2", "tier": "medium", "backend": "ollama"},
                {"name": "llama3.1:70b", "tier": "complex", "backend": "auto"},
            ],
            "embeddings": {"name": "bge-m3"},
        },
        "router": {
            "enabled": True,
            "strategy": "cost",
        },
        "services": {
            "vectors": {"provider": "qdrant", "port": 6333},
            "cache": {"provider": "redis", "port": 6379, "max_memory": "256mb"},
        },
        "gateway": {
            "port": 8000,
            "auth": "api_key",
            "rate_limit": "100/min",
            "cors": ["*"],
            "request_timeout": 120,
        },
        "observe": {
            "metrics": True,
            "dashboard_port": 8080,
            "retention": "7d",
        },
        "docker": {
            "network": "llmstack_net",
            "gpu": "auto",
            "data_dir": "~/.llmstack/data",
        },
    }


ROUTER_PRESET = router_preset()
