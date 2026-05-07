"""Chat preset — minimal setup for conversational AI."""

from llmstack.config.schema import (
    StackConfig, ModelsConfig, ModelSpec, EmbeddingSpec,
    ObserveConfig,
)

CHAT_PRESET = StackConfig(
    models=ModelsConfig(
        chat=ModelSpec(name="llama3.2", backend="auto"),
        embeddings=EmbeddingSpec(name="bge-m3"),
    ),
    observe=ObserveConfig(metrics=False),
)
