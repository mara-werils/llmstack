"""Response types for the LLMStack Python SDK."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------

@dataclass
class ChatMessage:
    """A single message in a chat conversation."""

    role: str
    content: str


@dataclass
class ChatChoice:
    """One completion choice returned by the chat endpoint."""

    index: int
    message: ChatMessage
    finish_reason: str | None = None


@dataclass
class Usage:
    """Token usage statistics."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class ChatResponse:
    """Response from POST /v1/chat/completions (non-streaming)."""

    id: str
    object: str
    model: str
    choices: list[ChatChoice] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    created: int = 0
    cached: bool = False
    cache_age: int = 0

    @classmethod
    def from_dict(cls, data: dict[str, Any], headers: dict[str, str] | None = None) -> ChatResponse:
        headers = headers or {}
        choices = [
            ChatChoice(
                index=c.get("index", i),
                message=ChatMessage(
                    role=c.get("message", {}).get("role", "assistant"),
                    content=c.get("message", {}).get("content", ""),
                ),
                finish_reason=c.get("finish_reason"),
            )
            for i, c in enumerate(data.get("choices", []))
        ]
        usage_raw = data.get("usage", {})
        usage = Usage(
            prompt_tokens=usage_raw.get("prompt_tokens", 0),
            completion_tokens=usage_raw.get("completion_tokens", 0),
            total_tokens=usage_raw.get("total_tokens", 0),
        )
        return cls(
            id=data.get("id", ""),
            object=data.get("object", "chat.completion"),
            model=data.get("model", ""),
            choices=choices,
            usage=usage,
            created=data.get("created", 0),
            cached=headers.get("x-cache", "").upper() == "HIT",
            cache_age=int(headers.get("x-cache-age", 0)),
        )


@dataclass
class ChatStreamDelta:
    """A single token/chunk from a streaming chat response."""

    id: str = ""
    model: str = ""
    delta_role: str | None = None
    delta_content: str | None = None
    finish_reason: str | None = None
    done: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChatStreamDelta:
        choices = data.get("choices", [])
        delta = choices[0].get("delta", {}) if choices else {}
        finish = choices[0].get("finish_reason") if choices else None
        return cls(
            id=data.get("id", ""),
            model=data.get("model", ""),
            delta_role=delta.get("role"),
            delta_content=delta.get("content"),
            finish_reason=finish,
            done=finish is not None,
        )


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

@dataclass
class Embedding:
    """A single embedding vector."""

    index: int
    embedding: list[float]
    object: str = "embedding"


@dataclass
class EmbeddingsResponse:
    """Response from POST /v1/embeddings."""

    object: str
    model: str
    data: list[Embedding] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EmbeddingsResponse:
        embeddings = [
            Embedding(
                index=e.get("index", i),
                embedding=e.get("embedding", []),
                object=e.get("object", "embedding"),
            )
            for i, e in enumerate(data.get("data", []))
        ]
        usage_raw = data.get("usage", {})
        return cls(
            object=data.get("object", "list"),
            model=data.get("model", ""),
            data=embeddings,
            usage=Usage(
                prompt_tokens=usage_raw.get("prompt_tokens", 0),
                completion_tokens=usage_raw.get("completion_tokens", 0),
                total_tokens=usage_raw.get("total_tokens", 0),
            ),
        )


# ---------------------------------------------------------------------------
# RAG
# ---------------------------------------------------------------------------

@dataclass
class IngestResponse:
    """Response from POST /v1/rag/ingest."""

    status: str
    chunks_stored: int
    source: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IngestResponse:
        return cls(
            status=data.get("status", ""),
            chunks_stored=data.get("chunks_stored", 0),
            source=data.get("source", ""),
        )


@dataclass
class RAGResponse:
    """Response from POST /v1/rag/query (non-streaming)."""

    answer: str
    sources: list[str] = field(default_factory=list)
    model: str = ""
    usage: dict[str, Any] = field(default_factory=dict)
    latency: float = 0.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RAGResponse:
        return cls(
            answer=data.get("answer", ""),
            sources=data.get("sources", []),
            model=data.get("model", ""),
            usage=data.get("usage", {}),
            latency=data.get("latency", 0.0),
        )


@dataclass
class RAGStreamDelta:
    """A single chunk from a streaming RAG response."""

    token: str | None = None
    done: bool = False
    sources: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

@dataclass
class Model:
    """A single model entry."""

    id: str
    object: str = "model"
    owned_by: str = ""
    created: int = 0


@dataclass
class ModelsResponse:
    """Response from GET /v1/models."""

    object: str = "list"
    data: list[Model] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelsResponse:
        models = [
            Model(
                id=m.get("id", ""),
                object=m.get("object", "model"),
                owned_by=m.get("owned_by", ""),
                created=m.get("created", 0),
            )
            for m in data.get("data", [])
        ]
        return cls(object=data.get("object", "list"), data=models)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@dataclass
class HealthResponse:
    """Response from GET /healthz."""

    status: str
    services: dict[str, bool] = field(default_factory=dict)
    circuit_breaker: dict[str, Any] = field(default_factory=dict)
    cache: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HealthResponse:
        return cls(
            status=data.get("status", "unknown"),
            services=data.get("services", {}),
            circuit_breaker=data.get("circuit_breaker", {}),
            cache=data.get("cache", {}),
        )
