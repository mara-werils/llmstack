"""Pydantic v2 request validation models for the LLMStack Gateway API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str = Field(max_length=100_000)


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage] = Field(min_length=1, max_length=200)
    stream: bool = False
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=1, le=100_000)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    n: int = Field(default=1, ge=1, le=10)
    user: str | None = None


class EmbeddingRequest(BaseModel):
    input: str | list[str]
    model: str


class RAGQueryRequest(BaseModel):
    query: str = Field(max_length=10_000)
    collection: str = "default"
    top_k: int = Field(default=5, ge=1, le=100)
