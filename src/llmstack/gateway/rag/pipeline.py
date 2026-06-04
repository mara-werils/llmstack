"""RAG pipeline — retrieval-augmented generation over ingested documents.

Implements the full retrieve → rerank → augment → generate flow.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import AsyncIterator

import httpx

from llmstack.gateway.rag.store import SearchResult, get_store

# System prompt template for RAG
_RAG_SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided context.

Rules:
- Answer ONLY based on the context provided below. If the context doesn't contain enough information, say so.
- Cite the source when possible using [source: <filename>] notation.
- Be concise and precise.
- If asked about something not in the context, say "I don't have enough information in the provided documents to answer this question."

Context:
{context}"""


@dataclass
class RAGResponse:
    answer: str
    sources: list[dict]
    model: str
    usage: dict
    latency: dict = field(default_factory=dict)


@dataclass
class RAGStreamChunk:
    token: str
    done: bool = False
    sources: list[dict] | None = None


class RAGPipeline:
    """Retrieve → Augment → Generate pipeline."""

    def __init__(self, inference_url: str):
        self._inference_url = inference_url

    async def query(
        self,
        question: str,
        model: str = "llama3.2",
        top_k: int = 5,
        score_threshold: float = 0.3,
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> RAGResponse:
        """Run the full RAG pipeline: retrieve → augment → generate."""
        timings: dict[str, float] = {}

        # 1. Retrieve relevant chunks
        t0 = time.monotonic()
        store = get_store()
        results = await store.search(question, top_k=top_k, score_threshold=score_threshold)
        timings["retrieval_ms"] = round((time.monotonic() - t0) * 1000, 1)

        if not results:
            return RAGResponse(
                answer="No relevant documents found. Please ingest documents first using POST /v1/rag/ingest.",
                sources=[],
                model=model,
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                latency=timings,
            )

        # 2. Build augmented prompt
        context = self._build_context(results)
        system_prompt = _RAG_SYSTEM_PROMPT.format(context=context)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]

        # 3. Generate
        t0 = time.monotonic()
        url = f"{self._inference_url}/chat/completions"
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                url,
                json={
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "stream": False,
                },
            )
            resp.raise_for_status()
            completion = resp.json()
        timings["generation_ms"] = round((time.monotonic() - t0) * 1000, 1)

        answer = completion["choices"][0]["message"]["content"]
        usage = completion.get("usage", {})

        sources = [
            {
                "text": r.text[:200],
                "source": r.metadata.get("source", ""),
                "score": round(r.score, 4),
            }
            for r in results
        ]

        return RAGResponse(
            answer=answer,
            sources=sources,
            model=model,
            usage=usage,
            latency=timings,
        )

    async def query_stream(
        self,
        question: str,
        model: str = "llama3.2",
        top_k: int = 5,
        score_threshold: float = 0.3,
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> AsyncIterator[RAGStreamChunk]:
        """Streaming RAG: retrieve first, then stream the generation."""
        store = get_store()
        results = await store.search(question, top_k=top_k, score_threshold=score_threshold)

        if not results:
            yield RAGStreamChunk(
                token="No relevant documents found.",
                done=True,
                sources=[],
            )
            return

        context = self._build_context(results)
        system_prompt = _RAG_SYSTEM_PROMPT.format(context=context)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]

        sources = [
            {
                "text": r.text[:200],
                "source": r.metadata.get("source", ""),
                "score": round(r.score, 4),
            }
            for r in results
        ]

        url = f"{self._inference_url}/chat/completions"
        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream(
                "POST",
                url,
                json={
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "stream": True,
                },
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data.strip() == "[DONE]":
                        yield RAGStreamChunk(token="", done=True, sources=sources)
                        return

                    import json

                    try:
                        chunk = json.loads(data)
                        delta = chunk["choices"][0].get("delta", {})
                        token = delta.get("content", "")
                        if token:
                            yield RAGStreamChunk(token=token)
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue

        yield RAGStreamChunk(token="", done=True, sources=sources)

    @staticmethod
    def _build_context(results: list[SearchResult]) -> str:
        """Format search results into a context block for the LLM."""
        parts = []
        for i, r in enumerate(results, 1):
            source = r.metadata.get("source", "unknown")
            parts.append(f"[{i}] (source: {source}, relevance: {r.score:.2f})\n{r.text}")
        return "\n\n---\n\n".join(parts)
