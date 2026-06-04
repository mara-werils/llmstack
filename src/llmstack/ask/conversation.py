"""Interactive conversation mode — multi-turn chat with project context.

Maintains conversation history across turns, re-uses the indexed project
context, and supports commands like /clear, /sources, /files.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import AsyncIterator

import httpx

from llmstack.ask.parsers import TextChunk


_SYSTEM_PROMPT = """\
You are a knowledgeable assistant helping a developer understand their codebase.
You have access to the project's source code through retrieved context.

Rules:
1. Answer based on the provided context. If the context doesn't contain enough information, say so.
2. Cite sources using [filename:lines] format.
3. When referring to code, include relevant snippets.
4. Remember previous messages in this conversation — build on prior answers.
5. Be concise but thorough.
"""


@dataclass
class ConversationTurn:
    """A single turn in the conversation."""

    role: str  # "user" or "assistant"
    content: str
    sources: list[str] = field(default_factory=list)  # source references for this turn


class ConversationEngine:
    """Multi-turn conversation engine with persistent project context.

    Maintains conversation history and enriches each query with relevant
    context from the project index.
    """

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        model: str = "llama3.2",
        system_prompt: str = "",
        git_context: str = "",
    ):
        self.ollama_url = ollama_url.rstrip("/")
        self.model = model
        self._system_prompt = system_prompt or _SYSTEM_PROMPT
        self._git_context = git_context
        self._history: list[ConversationTurn] = []
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(300, connect=10))

    @property
    def history(self) -> list[ConversationTurn]:
        return list(self._history)

    @property
    def turn_count(self) -> int:
        return len([t for t in self._history if t.role == "user"])

    def clear(self) -> None:
        """Clear conversation history."""
        self._history = []

    async def ask(
        self,
        question: str,
        context_chunks: list[tuple[TextChunk, float]],
    ) -> AsyncIterator[str]:
        """Ask a question with context, streaming the response.

        Args:
            question: User's question
            context_chunks: Relevant (chunk, score) pairs from search
        """
        # Build context from search results
        context_parts: list[str] = []
        sources: list[str] = []
        for chunk, score in context_chunks:
            header = f"{chunk.source}:{chunk.start_line}-{chunk.end_line}"
            context_parts.append(f"---\n{header}\n{chunk.content}\n---")
            sources.append(header)

        context_text = "\n".join(context_parts)

        # Build the augmented user message
        user_message = f"Context from codebase:\n{context_text}\n\nQuestion: {question}"

        # Build messages array
        messages = self._build_messages(user_message)

        # Record user turn
        self._history.append(ConversationTurn(role="user", content=question))

        # Stream response
        assistant_text = ""
        async for token in self._stream_chat(messages):
            assistant_text += token
            yield token

        # Record assistant turn
        self._history.append(
            ConversationTurn(
                role="assistant",
                content=assistant_text,
                sources=sources,
            )
        )

    def _build_messages(self, current_user_msg: str) -> list[dict[str, str]]:
        """Build the full message array including history."""
        messages: list[dict[str, str]] = []

        # System prompt with optional git context
        system = self._system_prompt
        if self._git_context:
            system += f"\n\nProject git info:\n{self._git_context}"

        messages.append({"role": "system", "content": system})

        # Conversation history (keep last 10 turns to fit context window)
        recent = self._history[-(10 * 2) :]  # 10 user+assistant pairs
        for turn in recent:
            messages.append({"role": turn.role, "content": turn.content})

        # Current message
        messages.append({"role": "user", "content": current_user_msg})

        return messages

    async def _stream_chat(self, messages: list[dict]) -> AsyncIterator[str]:
        """Stream tokens from Ollama."""
        async with self._client.stream(
            "POST",
            f"{self.ollama_url}/api/chat",
            json={
                "model": self.model,
                "messages": messages,
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

    async def close(self) -> None:
        """Clean up resources."""
        await self._client.aclose()
