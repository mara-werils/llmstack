"""
LlamaIndex integration with llmstack.

Demonstrates connecting LlamaIndex to llmstack's OpenAI-compatible API
for document indexing, retrieval, and query answering.

Install:
    pip install llama-index llama-index-llms-openai-like llama-index-embeddings-openai \
                llama-index-vector-stores-qdrant qdrant-client

Usage:
    1. Start llmstack with RAG preset:  llmstack init --preset rag && llmstack up
    2. Run this script:                  python llamaindex_rag.py
"""

from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai_like import OpenAILike
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

# ── Configuration ─────────────────────────────────────────────────────
LLMSTACK_URL = "http://localhost:8000/v1"
LLMSTACK_API_KEY = "llmstack"
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "llamaindex_demo"


def configure_llama_index():
    """Point LlamaIndex's global settings at llmstack."""
    Settings.llm = OpenAILike(
        api_base=LLMSTACK_URL,
        api_key=LLMSTACK_API_KEY,
        model="llama3.2",
        temperature=0.1,
        max_tokens=1024,
        is_chat_model=True,
        # Required for non-OpenAI endpoints
        is_function_calling_model=False,
    )
    Settings.embed_model = OpenAIEmbedding(
        api_base=LLMSTACK_URL,
        api_key=LLMSTACK_API_KEY,
        model_name="bge-m3",
    )
    Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=64)


# ── 1. In-memory index from text nodes ───────────────────────────────
def inmemory_index_example():
    """Build an in-memory vector index from hand-crafted text nodes."""
    nodes = [
        TextNode(
            text=(
                "LLMStack boots a full LLM stack with a single command. "
                "It manages Ollama or vLLM for inference, Qdrant for vector search, "
                "Redis for caching and rate limiting, and a FastAPI gateway."
            ),
            metadata={"source": "docs", "topic": "overview"},
        ),
        TextNode(
            text=(
                "The gateway includes a circuit breaker that prevents cascading failures. "
                "After 5 consecutive errors the circuit opens and requests fail fast with 503. "
                "It uses exponential backoff before attempting recovery."
            ),
            metadata={"source": "docs", "topic": "resilience"},
        ),
        TextNode(
            text=(
                "Hardware detection runs at init time. NVIDIA GPUs with 16GB+ VRAM get vLLM "
                "for maximum throughput via PagedAttention. Apple Silicon uses Ollama with "
                "Metal acceleration. CPU-only systems use GGUF quantized models via Ollama."
            ),
            metadata={"source": "docs", "topic": "hardware"},
        ),
        TextNode(
            text=(
                "llmstack supports three presets: 'chat' for minimal inference + cache, "
                "'rag' adds Qdrant and embeddings, and 'agent' configures a 70B model "
                "with 16K context and longer timeouts for complex agent workflows."
            ),
            metadata={"source": "docs", "topic": "presets"},
        ),
        TextNode(
            text=(
                "Observability is built in. When metrics are enabled, llmstack starts "
                "Prometheus and Grafana with dashboards showing request rate, latency "
                "percentiles, token throughput, cache hit rate, and circuit breaker state."
            ),
            metadata={"source": "docs", "topic": "observability"},
        ),
    ]

    index = VectorStoreIndex(nodes=nodes)

    # Query the index
    query_engine = index.as_query_engine(similarity_top_k=3)

    questions = [
        "What inference backends does llmstack support?",
        "How does llmstack handle failures?",
        "What monitoring does llmstack provide?",
    ]

    print("[In-memory index]")
    for q in questions:
        response = query_engine.query(q)
        print(f"Q: {q}")
        print(f"A: {response}\n")


# ── 2. Qdrant-backed persistent index ────────────────────────────────
def qdrant_index_example():
    """Build a persistent vector index using Qdrant (managed by llmstack).

    This stores embeddings in the same Qdrant instance that llmstack runs,
    so your data persists across restarts.
    """
    qdrant_client = QdrantClient(url=QDRANT_URL)

    vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=COLLECTION_NAME,
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Sample documents as text nodes
    nodes = [
        TextNode(
            text=(
                "Python is a high-level programming language known for its readability "
                "and versatility. It is widely used in web development, data science, "
                "machine learning, and automation."
            ),
            metadata={"source": "python-guide.txt"},
        ),
        TextNode(
            text=(
                "FastAPI is a modern Python web framework for building APIs. It is "
                "built on top of Starlette and Pydantic, offering automatic validation, "
                "serialization, and interactive documentation."
            ),
            metadata={"source": "fastapi-guide.txt"},
        ),
        TextNode(
            text=(
                "Docker containers package applications with their dependencies into "
                "standardized units. They ensure consistent environments across development, "
                "testing, and production. llmstack manages Docker containers automatically."
            ),
            metadata={"source": "docker-guide.txt"},
        ),
        TextNode(
            text=(
                "Vector databases like Qdrant store high-dimensional embeddings and "
                "support fast approximate nearest neighbor search. They are essential "
                "for semantic search and retrieval-augmented generation (RAG)."
            ),
            metadata={"source": "vectordb-guide.txt"},
        ),
    ]

    # Build the index — this embeds all nodes via llmstack and stores in Qdrant
    index = VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
    )

    # Query
    query_engine = index.as_query_engine(similarity_top_k=2)

    print("[Qdrant-backed index]")
    questions = [
        "What is FastAPI built on?",
        "What are vector databases used for?",
        "How does llmstack use Docker?",
    ]
    for q in questions:
        response = query_engine.query(q)
        print(f"Q: {q}")
        print(f"A: {response}")
        # Print source nodes
        for node in response.source_nodes:
            print(f"   Source: {node.metadata.get('source', 'unknown')} (score: {node.score:.3f})")
        print()

    # Cleanup
    qdrant_client.delete_collection(COLLECTION_NAME)


# ── 3. Chat engine with conversation memory ──────────────────────────
def chat_engine_example():
    """Use LlamaIndex's chat engine for multi-turn conversation with context."""
    nodes = [
        TextNode(
            text=(
                "The llmstack rate limiter uses a token-bucket algorithm backed by Redis. "
                "It supports per-API-key limiting with IP fallback. Atomic Lua scripts "
                "ensure race-free counting. Standard X-RateLimit headers are included "
                "in every response."
            ),
            metadata={"source": "docs"},
        ),
        TextNode(
            text=(
                "Semantic caching in llmstack hashes the model name and messages with "
                "SHA-256. Cache hits are returned in under 1ms. Only deterministic requests "
                "(temperature <= 0.1) are cached, with a default TTL of 1 hour."
            ),
            metadata={"source": "docs"},
        ),
    ]

    index = VectorStoreIndex(nodes=nodes)
    chat_engine = index.as_chat_engine(
        chat_mode="condense_plus_context",
        similarity_top_k=2,
    )

    print("[Chat engine with memory]")
    # Turn 1
    response1 = chat_engine.chat("How does rate limiting work in llmstack?")
    print("User:      How does rate limiting work in llmstack?")
    print(f"Assistant: {response1}\n")

    # Turn 2 — follow-up referencing previous context
    response2 = chat_engine.chat("What about caching? How is that different?")
    print("User:      What about caching? How is that different?")
    print(f"Assistant: {response2}\n")


# ── 4. Loading documents from a directory ─────────────────────────────
def directory_reader_example():
    """Load and index documents from a local directory.

    Create a 'sample_docs/' directory with some .txt files to test this.
    """
    import os

    sample_dir = os.path.join(os.path.dirname(__file__), "sample_docs")

    if not os.path.exists(sample_dir):
        # Create sample documents for the demo
        os.makedirs(sample_dir, exist_ok=True)
        files = {
            "intro.txt": (
                "LLMStack is an open-source tool for running a complete LLM stack locally. "
                "It handles inference, embeddings, vector storage, caching, and API routing."
            ),
            "install.txt": (
                "To install llmstack, run: pip install llmstack-cli. "
                "Then initialize with: llmstack init --preset rag. "
                "Finally start everything with: llmstack up."
            ),
            "api.txt": (
                "The llmstack API is OpenAI-compatible. Use /v1/chat/completions for chat, "
                "/v1/embeddings for embeddings, /v1/rag/ingest to add documents, "
                "and /v1/rag/query for retrieval-augmented generation."
            ),
        }
        for filename, content in files.items():
            with open(os.path.join(sample_dir, filename), "w") as f:
                f.write(content)
        print("Created sample_docs/ with demo files.\n")

    # Load all documents from the directory
    reader = SimpleDirectoryReader(sample_dir)
    documents = reader.load_data()

    # Build index
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine(similarity_top_k=2)

    print("[Directory reader index]")
    response = query_engine.query("How do I install and start llmstack?")
    print("Q: How do I install and start llmstack?")
    print(f"A: {response}\n")


# ── Run all examples ─────────────────────────────────────────────────
if __name__ == "__main__":
    configure_llama_index()
    inmemory_index_example()
    qdrant_index_example()
    chat_engine_example()
    directory_reader_example()
