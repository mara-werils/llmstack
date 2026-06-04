"""Agent preset — large model + long context for agentic workflows."""

from llmstack.config.schema import (
    StackConfig,
    ModelsConfig,
    ModelSpec,
    EmbeddingSpec,
    GatewayConfig,
)

AGENT_PRESET = StackConfig(
    models=ModelsConfig(
        chat=ModelSpec(name="llama3.1:70b", backend="auto", context_length=16384),
        embeddings=EmbeddingSpec(name="bge-m3"),
    ),
    gateway=GatewayConfig(rate_limit="30/min", request_timeout=300),
)
