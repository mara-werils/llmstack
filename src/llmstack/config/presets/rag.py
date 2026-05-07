"""RAG preset — chat + vector search + document ingestion."""

from llmstack.config.schema import StackConfig, ModelsConfig, ModelSpec, EmbeddingSpec

RAG_PRESET = StackConfig(
    models=ModelsConfig(
        chat=ModelSpec(name="llama3.2", backend="auto", context_length=8192),
        embeddings=EmbeddingSpec(name="bge-m3"),
    ),
)
