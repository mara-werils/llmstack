"""
llmstack SDK quickstart — all features in one file.

Since llmstack exposes an OpenAI-compatible API plus custom RAG endpoints,
this example shows how to use both via the ``openai`` and ``httpx`` libraries
as a lightweight "SDK" for every llmstack feature.

Install:
    pip install openai httpx

Usage:
    1. Start llmstack:  llmstack init --preset rag && llmstack up
    2. Run this script:  python sdk_quickstart.py
"""

from __future__ import annotations

import json

import httpx
from openai import OpenAI

# ── Configuration ─────────────────────────────────────────────────────
BASE_URL = "http://localhost:8000"
API_KEY = "llmstack"

# OpenAI client for standard endpoints
client = OpenAI(base_url=f"{BASE_URL}/v1", api_key=API_KEY)

# httpx client for llmstack-specific endpoints (RAG, health)
http = httpx.Client(
    base_url=BASE_URL,
    headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
    timeout=60.0,
)


# ── 1. Health check ──────────────────────────────────────────────────
def check_health():
    """Verify llmstack and all its services are running."""
    resp = http.get("/healthz")
    health = resp.json()

    print("[Health check]")
    print(f"  Status: {health['status']}")
    for service, ok in health.get("services", {}).items():
        status = "UP" if ok else "DOWN"
        print(f"  {service}: {status}")

    if "circuit_breaker" in health:
        cb = health["circuit_breaker"]
        print(f"  Circuit breaker: {cb.get('state', 'unknown')}")

    if "cache" in health:
        cache = health["cache"]
        print(f"  Cache hits: {cache.get('hits', 0)}, misses: {cache.get('misses', 0)}")

    print()
    return health["status"] == "ok"


# ── 2. List models ───────────────────────────────────────────────────
def list_models():
    """List all models available on the inference backend."""
    models = client.models.list()
    print("[Models]")
    for m in models.data:
        print(f"  {m.id}")
    print()
    return [m.id for m in models.data]


# ── 3. Chat completion ───────────────────────────────────────────────
def chat(message: str, model: str = "llama3.2", **kwargs) -> str:
    """Send a chat message and return the response text."""
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": message}],
        **kwargs,
    )
    return response.choices[0].message.content


# ── 4. Streaming chat ────────────────────────────────────────────────
def chat_stream(message: str, model: str = "llama3.2", **kwargs):
    """Stream a chat response, yielding tokens as they arrive."""
    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": message}],
        stream=True,
        **kwargs,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            yield delta.content


# ── 5. Embeddings ────────────────────────────────────────────────────
def embed(texts: list[str], model: str = "bge-m3") -> list[list[float]]:
    """Generate embeddings for a list of texts."""
    response = client.embeddings.create(model=model, input=texts)
    return [item.embedding for item in response.data]


# ── 6. RAG: Ingest a document ────────────────────────────────────────
def rag_ingest(text: str, source: str, metadata: dict | None = None) -> dict:
    """Ingest a document into llmstack's RAG pipeline (chunk + embed + store)."""
    resp = http.post("/v1/rag/ingest", json={
        "text": text,
        "source": source,
        "metadata": metadata or {},
    })
    resp.raise_for_status()
    return resp.json()


# ── 7. RAG: Query ────────────────────────────────────────────────────
def rag_query(question: str, top_k: int = 5, model: str = "llama3.2") -> dict:
    """Query the RAG pipeline and return the answer with sources."""
    resp = http.post("/v1/rag/query", json={
        "question": question,
        "top_k": top_k,
        "model": model,
    })
    resp.raise_for_status()
    return resp.json()


# ── 8. RAG: Streaming query ──────────────────────────────────────────
def rag_query_stream(question: str, top_k: int = 5, model: str = "llama3.2"):
    """Stream a RAG query response via SSE, yielding tokens and sources."""
    with httpx.Client(base_url=BASE_URL, headers={
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }, timeout=60.0) as stream_client:
        with stream_client.stream("POST", "/v1/rag/query", json={
            "question": question,
            "top_k": top_k,
            "model": model,
            "stream": True,
        }) as resp:
            for line in resp.iter_lines():
                if line.startswith("data: "):
                    data = json.loads(line[6:])
                    if data.get("done"):
                        yield {"done": True, "sources": data.get("sources", [])}
                    else:
                        yield {"token": data.get("token", "")}


# ── 9. RAG: Collection status ────────────────────────────────────────
def rag_status() -> dict:
    """Get statistics about the RAG vector collection."""
    resp = http.get("/v1/rag/status")
    resp.raise_for_status()
    return resp.json()


# ── 10. RAG: Delete documents ────────────────────────────────────────
def rag_delete(source: str) -> dict:
    """Delete all chunks from a specific source."""
    resp = http.delete(f"/v1/rag/documents/{source}")
    resp.raise_for_status()
    return resp.json()


# ── Demo: run all features ───────────────────────────────────────────
def main():
    print("=" * 60)
    print("llmstack SDK Quickstart")
    print("=" * 60)
    print()

    # Health
    if not check_health():
        print("WARNING: llmstack is not fully healthy, some examples may fail.\n")

    # Models
    list_models()

    # Chat
    print("[Chat]")
    answer = chat("What is the Pythagorean theorem?", temperature=0.2, max_tokens=200)
    print(f"  {answer}\n")

    # Streaming chat
    print("[Streaming chat]")
    print("  ", end="")
    for token in chat_stream("Name 5 prime numbers.", temperature=0.1, max_tokens=100):
        print(token, end="", flush=True)
    print("\n")

    # Embeddings
    print("[Embeddings]")
    vectors = embed(["Hello world", "Goodbye world"])
    for i, v in enumerate(vectors):
        print(f"  Text {i}: dim={len(v)}, norm_sample={v[:3]}")
    print()

    # RAG ingest
    print("[RAG ingest]")
    docs = [
        ("Python was created by Guido van Rossum and first released in 1991. "
         "It emphasizes code readability and supports multiple paradigms.",
         "python-history.txt"),
        ("FastAPI is a modern Python web framework created by Sebastian Ramirez. "
         "It uses Python type hints for automatic validation and documentation.",
         "fastapi-intro.txt"),
        ("Docker containers package applications with all dependencies. "
         "They provide consistent environments from development to production.",
         "docker-basics.txt"),
    ]
    for text, source in docs:
        result = rag_ingest(text, source)
        print(f"  Ingested '{source}': {result['chunks_stored']} chunks")
    print()

    # RAG status
    print("[RAG status]")
    status = rag_status()
    print(f"  {status}\n")

    # RAG query
    print("[RAG query]")
    result = rag_query("Who created Python and when?")
    print(f"  Answer: {result['answer']}")
    print(f"  Sources: {result.get('sources', [])}\n")

    # RAG streaming query
    print("[RAG streaming query]")
    print("  ", end="")
    for chunk in rag_query_stream("What is FastAPI?"):
        if chunk.get("done"):
            print(f"\n  Sources: {chunk['sources']}")
        else:
            print(chunk["token"], end="", flush=True)
    print()

    # Cleanup
    print("\n[Cleanup]")
    for _, source in docs:
        rag_delete(source)
        print(f"  Deleted '{source}'")
    print()

    print("Done.")


if __name__ == "__main__":
    main()
