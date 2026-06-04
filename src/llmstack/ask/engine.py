"""Core ask engine — parse files, embed, search, and generate answers."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator

import httpx

from llmstack.ask.embeddings import LocalEmbeddings
from llmstack.ask.parsers import TextChunk, collect_files, parse_file

logger = logging.getLogger(__name__)


@dataclass
class SourceRef:
    """A reference to a source chunk used in an answer."""

    file: str
    lines: str  # e.g. "42-67"
    relevance: float
    preview: str  # first 100 chars of the chunk


@dataclass
class AskResult:
    """Full result from an ask query."""

    answer: str
    sources: list[SourceRef] = field(default_factory=list)
    chunks_searched: int = 0
    total_chunks: int = 0


_PROMPT_TEMPLATE = """\
Answer the question based ONLY on the following context. If the context doesn't \
contain enough information, say so.
Cite your sources using [filename:lines] format.

Context:
{context}

Question: {question}"""


def _build_context(results: list[tuple[TextChunk, float]]) -> str:
    """Build context string from search results."""
    parts: list[str] = []
    for chunk, _score in results:
        header = f"{chunk.source}:{chunk.start_line}-{chunk.end_line}"
        parts.append(f"---\n{header}\n{chunk.content}\n---")
    return "\n".join(parts)


def _build_sources(results: list[tuple[TextChunk, float]]) -> list[SourceRef]:
    """Build source references from search results."""
    sources: list[SourceRef] = []
    for chunk, score in results:
        sources.append(
            SourceRef(
                file=chunk.source,
                lines=f"{chunk.start_line}-{chunk.end_line}",
                relevance=round(score, 4),
                preview=chunk.content[:100].replace("\n", " "),
            )
        )
    return sources


class AskEngine:
    """Parse files, embed, search, and generate answers."""

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        model: str = "llama3.2",
        embed_model: str = "nomic-embed-text",
    ) -> None:
        self.ollama_url = ollama_url.rstrip("/")
        self.model = model
        self.embeddings = LocalEmbeddings(ollama_url=ollama_url, model=embed_model)
        self._chunks: list[TextChunk] = []
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(300, connect=10))

    @property
    def total_chunks(self) -> int:
        """Return the total number of indexed chunks."""
        return len(self._chunks)

    async def load(
        self,
        paths: list[Path],
        show_progress: bool = True,
        progress_callback: object | None = None,
    ) -> None:
        """Parse and index all files.

        Args:
            paths: List of file or directory paths to parse.
            show_progress: Whether to show progress (used by CLI layer).
            progress_callback: Optional callable(stage, current, total) for
                progress reporting from the CLI.
        """
        # Collect all files
        all_files: list[Path] = []
        for p in paths:
            all_files.extend(collect_files(p))

        # Parse files into chunks
        all_chunks: list[TextChunk] = []
        parse_cb = None
        if progress_callback and callable(progress_callback):
            parse_cb = progress_callback

        for i, fpath in enumerate(all_files):
            try:
                chunks = parse_file(fpath)
                all_chunks.extend(chunks)
            except Exception as exc:
                logger.debug("Skipping unparseable file %s: %s", fpath, exc)
            if parse_cb:
                parse_cb("parse", i + 1, len(all_files))

        self._chunks = all_chunks

        if not all_chunks:
            return

        # Index chunks with embeddings
        await self.embeddings.index(all_chunks)

    async def ask(self, question: str, top_k: int = 5) -> AsyncIterator[str]:
        """Ask a question and stream the answer token by token.

        Yields string tokens as they arrive from the LLM.
        """
        if not self._chunks:
            yield "No files were loaded. Please provide files or directories to search."
            return

        # Search for relevant chunks
        results = await self.embeddings.search(question, top_k=top_k)
        if not results:
            yield "No relevant context found for your question."
            return

        # Build prompt
        context = _build_context(results)
        prompt = _PROMPT_TEMPLATE.format(context=context, question=question)

        # Stream response from Ollama
        async with self._client.stream(
            "POST",
            f"{self.ollama_url}/api/chat",
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": True,
            },
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    token = data.get("message", {}).get("content", "")
                    if token:
                        yield token
                    if data.get("done", False):
                        break
                except json.JSONDecodeError:
                    continue

    async def ask_full(self, question: str, top_k: int = 5) -> AskResult:
        """Ask and return full result with sources.

        This collects the entire streamed response and returns a structured
        AskResult with source citations.
        """
        if not self._chunks:
            return AskResult(
                answer="No files were loaded. Please provide files or directories to search.",
                sources=[],
                chunks_searched=0,
                total_chunks=0,
            )

        # Search for relevant chunks
        results = await self.embeddings.search(question, top_k=top_k)
        if not results:
            return AskResult(
                answer="No relevant context found for your question.",
                sources=[],
                chunks_searched=0,
                total_chunks=len(self._chunks),
            )

        # Build prompt
        context = _build_context(results)
        prompt = _PROMPT_TEMPLATE.format(context=context, question=question)

        # Get full response from Ollama
        resp = await self._client.post(
            f"{self.ollama_url}/api/chat",
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        answer = data.get("message", {}).get("content", "")

        return AskResult(
            answer=answer,
            sources=_build_sources(results),
            chunks_searched=len(results),
            total_chunks=len(self._chunks),
        )

    async def close(self) -> None:
        """Clean up resources."""
        await self.embeddings.close()
        await self._client.aclose()
