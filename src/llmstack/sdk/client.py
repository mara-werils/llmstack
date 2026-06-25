"""Synchronous and asynchronous HTTP clients for the LLMStack gateway API."""

from __future__ import annotations

import json
from typing import Any, Generator, AsyncGenerator

import httpx

from llmstack.sdk.retry import RetryConfig, async_retry, sync_retry
from llmstack.sdk.types import (
    ChatResponse,
    ChatStreamDelta,
    EmbeddingsResponse,
    HealthResponse,
    IngestResponse,
    ModelsResponse,
    RAGResponse,
    RAGStreamDelta,
)

__all__ = ["Client", "AsyncClient", "LLMStackError", "RetryConfig"]

_DEFAULT_BASE_URL = "http://localhost:8000"
_DEFAULT_TIMEOUT = 120.0


class LLMStackError(Exception):
    """Raised when the LLMStack API returns an error response."""

    def __init__(self, status_code: int, detail: Any = None) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"HTTP {status_code}: {detail}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_headers(api_key: str | None) -> dict[str, str]:
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _raise_for_error(response: httpx.Response) -> None:
    if response.status_code >= 400:
        try:
            detail = response.json()
        except Exception:
            detail = response.text
        raise LLMStackError(response.status_code, detail)


def _parse_sse_line(line: str) -> dict[str, Any] | None:
    """Parse a single SSE ``data:`` line into a dict, or *None* for blanks/comments."""
    line = line.strip()
    if not line or line.startswith(":"):
        return None
    if line.startswith("data:"):
        payload = line[len("data:") :].strip()
        if payload == "[DONE]":
            return None
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            return None
    return None


# ===================================================================
# Synchronous Client
# ===================================================================


class Client:
    """Synchronous Python client for the LLMStack gateway.

    Usage::

        from llmstack import Client

        with Client() as llm:
            resp = llm.chat(messages=[{"role": "user", "content": "Hello!"}])
            print(resp.choices[0].message.content)
    """

    def __init__(
        self,
        base_url: str = _DEFAULT_BASE_URL,
        api_key: str | None = None,
        timeout: float = _DEFAULT_TIMEOUT,
        retry: RetryConfig | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._retry = retry or RetryConfig()
        self._client = httpx.Client(
            base_url=self.base_url,
            headers=_build_headers(api_key),
            timeout=timeout,
        )

    def __repr__(self) -> str:
        return f"Client(base_url={self.base_url!r}, api_key={'***' if self.api_key else None!r})"

    # -- context manager --------------------------------------------------

    def __enter__(self) -> Client:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        """Close the underlying HTTP connection pool."""
        self._client.close()

    # -- internal request helpers -----------------------------------------

    def _post(self, url: str, **kwargs: Any) -> httpx.Response:
        return sync_retry(self._client.post, self._retry, url, **kwargs)

    def _get(self, url: str, **kwargs: Any) -> httpx.Response:
        return sync_retry(self._client.get, self._retry, url, **kwargs)

    # -- chat -------------------------------------------------------------

    def chat(
        self,
        messages: list[dict[str, str]],
        model: str = "llama3.2",
        stream: bool = False,
        **kwargs: Any,
    ) -> ChatResponse | Generator[ChatStreamDelta, None, None]:
        """Send a chat completion request.

        Args:
            messages: List of message dicts, each with ``role`` and ``content``.
            model: Model identifier.
            stream: If *True*, return a generator that yields ``ChatStreamDelta`` objects.
            **kwargs: Additional fields forwarded to the API (e.g. ``temperature``).

        Returns:
            ``ChatResponse`` when *stream=False*, or a generator of
            ``ChatStreamDelta`` when *stream=True*.
        """
        payload: dict[str, Any] = {"model": model, "messages": messages, "stream": stream, **kwargs}

        if stream:
            return self._chat_stream(payload)

        resp = self._post("/v1/chat/completions", json=payload)
        _raise_for_error(resp)
        return ChatResponse.from_dict(resp.json(), headers=dict(resp.headers))

    def _chat_stream(self, payload: dict[str, Any]) -> Generator[ChatStreamDelta, None, None]:
        with self._client.stream("POST", "/v1/chat/completions", json=payload) as resp:
            _raise_for_error(resp)
            for line in resp.iter_lines():
                data = _parse_sse_line(line)
                if data is not None:
                    yield ChatStreamDelta.from_dict(data)

    # -- embeddings -------------------------------------------------------

    def embed(
        self,
        input: str | list[str],
        model: str = "bge-m3",
    ) -> EmbeddingsResponse:
        """Generate embeddings for the given input.

        Args:
            input: A string or list of strings to embed.
            model: Embedding model identifier.

        Returns:
            ``EmbeddingsResponse`` containing embedding vectors.
        """
        payload: dict[str, Any] = {"input": input, "model": model}
        resp = self._post("/v1/embeddings", json=payload)
        _raise_for_error(resp)
        return EmbeddingsResponse.from_dict(resp.json())

    # -- RAG ingest -------------------------------------------------------

    def rag_ingest(
        self,
        text: str,
        source: str,
        chunk_size: int = 512,
        metadata: dict[str, Any] | None = None,
    ) -> IngestResponse:
        """Ingest a document into the RAG store.

        Args:
            text: Raw text content.
            source: Filename or URL used for citation.
            chunk_size: Target chunk size in tokens.
            metadata: Optional extra metadata to store alongside chunks.

        Returns:
            ``IngestResponse`` with ingestion status.
        """
        payload: dict[str, Any] = {
            "text": text,
            "source": source,
            "chunk_size": chunk_size,
        }
        if metadata:
            payload["metadata"] = metadata
        resp = self._post("/v1/rag/ingest", json=payload)
        _raise_for_error(resp)
        return IngestResponse.from_dict(resp.json())

    # -- RAG query --------------------------------------------------------

    def rag_query(
        self,
        question: str,
        top_k: int = 5,
        stream: bool = False,
        **kwargs: Any,
    ) -> RAGResponse | Generator[RAGStreamDelta, None, None]:
        """Query the RAG pipeline.

        Args:
            question: The question to answer.
            top_k: Number of context chunks to retrieve.
            stream: If *True*, return a generator that yields ``RAGStreamDelta`` objects.
            **kwargs: Additional fields forwarded to the API (e.g. ``model``, ``temperature``).

        Returns:
            ``RAGResponse`` when *stream=False*, or a generator of ``RAGStreamDelta``
            when *stream=True*.
        """
        payload: dict[str, Any] = {
            "question": question,
            "top_k": top_k,
            "stream": stream,
            **kwargs,
        }

        if stream:
            return self._rag_query_stream(payload)

        resp = self._post("/v1/rag/query", json=payload)
        _raise_for_error(resp)
        return RAGResponse.from_dict(resp.json())

    def _rag_query_stream(self, payload: dict[str, Any]) -> Generator[RAGStreamDelta, None, None]:
        with self._client.stream("POST", "/v1/rag/query", json=payload) as resp:
            _raise_for_error(resp)
            for line in resp.iter_lines():
                data = _parse_sse_line(line)
                if data is None:
                    continue
                yield RAGStreamDelta(
                    token=data.get("token"),
                    done=data.get("done", False),
                    sources=data.get("sources", []),
                )

    # -- models -----------------------------------------------------------

    def models(self) -> ModelsResponse:
        """List models available on the inference backend.

        Returns:
            ``ModelsResponse`` containing the list of models.
        """
        resp = self._get("/v1/models")
        _raise_for_error(resp)
        return ModelsResponse.from_dict(resp.json())

    # -- health -----------------------------------------------------------

    def health(self) -> HealthResponse:
        """Check gateway health and service status.

        Returns:
            ``HealthResponse`` with per-service status flags.
        """
        resp = self._get("/healthz")
        _raise_for_error(resp)
        return HealthResponse.from_dict(resp.json())

    # -- savings -----------------------------------------------------------

    def savings(self, plan: str | None = None) -> dict[str, Any]:
        """Return cumulative savings from serving requests locally.

        Args:
            plan: Subscription to compare against (e.g. ``"copilot-pro"``,
                ``"cursor-pro"``); defaults to the gateway's baseline.

        Returns:
            A dict with the running totals (``total_saved_usd``,
            ``total_requests``, …) and a ``subscription`` block giving how many
            months of the chosen plan the savings would cover.
        """
        params = {"plan": plan} if plan else None
        resp = self._get("/v1/savings/summary", params=params)
        _raise_for_error(resp)
        return resp.json()

    def onboarding(self, ollama_url: str | None = None) -> dict[str, Any]:
        """Report first-run readiness for zero-key local inference.

        Args:
            ollama_url: Probe a specific Ollama URL instead of the gateway default.

        Returns:
            A dict with ``ready``, the recommended chat/embed models, detected
            hardware, which models are present, and concrete next-step ``hints``.
        """
        params = {"ollama_url": ollama_url} if ollama_url else None
        resp = self._get("/v1/onboarding", params=params)
        _raise_for_error(resp)
        return resp.json()

    def ready(self, ollama_url: str | None = None) -> bool:
        """True when the machine is ready for zero-key local inference."""
        return bool(self.onboarding(ollama_url).get("ready"))

    # -- convenience methods -----------------------------------------------

    def ask(self, question: str, model: str = "llama3.2", **kwargs: Any) -> str:
        """One-liner: send a question and get the response text.

        Args:
            question: The user's question.
            model: Model identifier.
            **kwargs: Additional chat completion parameters.

        Returns:
            The assistant's reply as a plain string.
        """
        resp = self.chat(
            messages=[{"role": "user", "content": question}],
            model=model,
            stream=False,
            **kwargs,
        )
        return resp.choices[0].message.content if resp.choices else ""

    def complete(
        self,
        prompt: str,
        model: str = "llama3.2",
        system: str = "",
        **kwargs: Any,
    ) -> str:
        """Send a prompt with optional system message and get text back.

        Args:
            prompt: The user prompt.
            model: Model identifier.
            system: Optional system prompt.
            **kwargs: Additional chat completion parameters.

        Returns:
            The assistant's reply as a plain string.
        """
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = self.chat(messages=messages, model=model, stream=False, **kwargs)
        return resp.choices[0].message.content if resp.choices else ""


# ===================================================================
# Asynchronous Client
# ===================================================================


class AsyncClient:
    """Asynchronous Python client for the LLMStack gateway.

    Usage::

        import asyncio
        from llmstack import AsyncClient

        async def main():
            async with AsyncClient() as llm:
                resp = await llm.chat(messages=[{"role": "user", "content": "Hello!"}])
                print(resp.choices[0].message.content)

        asyncio.run(main())
    """

    def __init__(
        self,
        base_url: str = _DEFAULT_BASE_URL,
        api_key: str | None = None,
        timeout: float = _DEFAULT_TIMEOUT,
        retry: RetryConfig | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._retry = retry or RetryConfig()
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=_build_headers(api_key),
            timeout=timeout,
        )

    def __repr__(self) -> str:
        return (
            f"AsyncClient(base_url={self.base_url!r}, api_key={'***' if self.api_key else None!r})"
        )

    # -- context manager --------------------------------------------------

    async def __aenter__(self) -> AsyncClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    async def close(self) -> None:
        """Close the underlying HTTP connection pool."""
        await self._client.aclose()

    # -- internal request helpers -----------------------------------------

    async def _post(self, url: str, **kwargs: Any) -> httpx.Response:
        return await async_retry(self._client.post, self._retry, url, **kwargs)

    async def _get(self, url: str, **kwargs: Any) -> httpx.Response:
        return await async_retry(self._client.get, self._retry, url, **kwargs)

    # -- chat -------------------------------------------------------------

    async def chat(
        self,
        messages: list[dict[str, str]],
        model: str = "llama3.2",
        stream: bool = False,
        **kwargs: Any,
    ) -> ChatResponse | AsyncGenerator[ChatStreamDelta, None]:
        """Send a chat completion request.

        Args:
            messages: List of message dicts, each with ``role`` and ``content``.
            model: Model identifier.
            stream: If *True*, return an async generator yielding ``ChatStreamDelta`` objects.
            **kwargs: Additional fields forwarded to the API.

        Returns:
            ``ChatResponse`` when *stream=False*, or an async generator of
            ``ChatStreamDelta`` when *stream=True*.
        """
        payload: dict[str, Any] = {"model": model, "messages": messages, "stream": stream, **kwargs}

        if stream:
            return self._chat_stream(payload)

        resp = await self._post("/v1/chat/completions", json=payload)
        _raise_for_error(resp)
        return ChatResponse.from_dict(resp.json(), headers=dict(resp.headers))

    async def _chat_stream(self, payload: dict[str, Any]) -> AsyncGenerator[ChatStreamDelta, None]:
        async with self._client.stream("POST", "/v1/chat/completions", json=payload) as resp:
            _raise_for_error(resp)
            async for line in resp.aiter_lines():
                data = _parse_sse_line(line)
                if data is not None:
                    yield ChatStreamDelta.from_dict(data)

    # -- embeddings -------------------------------------------------------

    async def embed(
        self,
        input: str | list[str],
        model: str = "bge-m3",
    ) -> EmbeddingsResponse:
        """Generate embeddings for the given input.

        Args:
            input: A string or list of strings to embed.
            model: Embedding model identifier.

        Returns:
            ``EmbeddingsResponse`` containing embedding vectors.
        """
        payload: dict[str, Any] = {"input": input, "model": model}
        resp = await self._post("/v1/embeddings", json=payload)
        _raise_for_error(resp)
        return EmbeddingsResponse.from_dict(resp.json())

    # -- RAG ingest -------------------------------------------------------

    async def rag_ingest(
        self,
        text: str,
        source: str,
        chunk_size: int = 512,
        metadata: dict[str, Any] | None = None,
    ) -> IngestResponse:
        """Ingest a document into the RAG store.

        Args:
            text: Raw text content.
            source: Filename or URL used for citation.
            chunk_size: Target chunk size in tokens.
            metadata: Optional extra metadata to store alongside chunks.

        Returns:
            ``IngestResponse`` with ingestion status.
        """
        payload: dict[str, Any] = {
            "text": text,
            "source": source,
            "chunk_size": chunk_size,
        }
        if metadata:
            payload["metadata"] = metadata
        resp = await self._post("/v1/rag/ingest", json=payload)
        _raise_for_error(resp)
        return IngestResponse.from_dict(resp.json())

    # -- RAG query --------------------------------------------------------

    async def rag_query(
        self,
        question: str,
        top_k: int = 5,
        stream: bool = False,
        **kwargs: Any,
    ) -> RAGResponse | AsyncGenerator[RAGStreamDelta, None]:
        """Query the RAG pipeline.

        Args:
            question: The question to answer.
            top_k: Number of context chunks to retrieve.
            stream: If *True*, return an async generator yielding ``RAGStreamDelta`` objects.
            **kwargs: Additional fields forwarded to the API.

        Returns:
            ``RAGResponse`` when *stream=False*, or an async generator of
            ``RAGStreamDelta`` when *stream=True*.
        """
        payload: dict[str, Any] = {
            "question": question,
            "top_k": top_k,
            "stream": stream,
            **kwargs,
        }

        if stream:
            return self._rag_query_stream(payload)

        resp = await self._post("/v1/rag/query", json=payload)
        _raise_for_error(resp)
        return RAGResponse.from_dict(resp.json())

    async def _rag_query_stream(
        self, payload: dict[str, Any]
    ) -> AsyncGenerator[RAGStreamDelta, None]:
        async with self._client.stream("POST", "/v1/rag/query", json=payload) as resp:
            _raise_for_error(resp)
            async for line in resp.aiter_lines():
                data = _parse_sse_line(line)
                if data is None:
                    continue
                yield RAGStreamDelta(
                    token=data.get("token"),
                    done=data.get("done", False),
                    sources=data.get("sources", []),
                )

    # -- models -----------------------------------------------------------

    async def models(self) -> ModelsResponse:
        """List models available on the inference backend.

        Returns:
            ``ModelsResponse`` containing the list of models.
        """
        resp = await self._get("/v1/models")
        _raise_for_error(resp)
        return ModelsResponse.from_dict(resp.json())

    # -- health -----------------------------------------------------------

    async def health(self) -> HealthResponse:
        """Check gateway health and service status.

        Returns:
            ``HealthResponse`` with per-service status flags.
        """
        resp = await self._get("/healthz")
        _raise_for_error(resp)
        return HealthResponse.from_dict(resp.json())

    # -- savings -----------------------------------------------------------

    async def savings(self, plan: str | None = None) -> dict[str, Any]:
        """Return cumulative savings from serving requests locally.

        Args:
            plan: Subscription to compare against (e.g. ``"copilot-pro"``);
                defaults to the gateway's baseline.

        Returns:
            A dict with the running totals and a ``subscription`` block.
        """
        params = {"plan": plan} if plan else None
        resp = await self._get("/v1/savings/summary", params=params)
        _raise_for_error(resp)
        return resp.json()

    async def onboarding(self, ollama_url: str | None = None) -> dict[str, Any]:
        """Report first-run readiness for zero-key local inference.

        Args:
            ollama_url: Probe a specific Ollama URL instead of the gateway default.

        Returns:
            A dict with ``ready``, recommended models, hardware, and ``hints``.
        """
        params = {"ollama_url": ollama_url} if ollama_url else None
        resp = await self._get("/v1/onboarding", params=params)
        _raise_for_error(resp)
        return resp.json()

    async def ready(self, ollama_url: str | None = None) -> bool:
        """True when the machine is ready for zero-key local inference."""
        return bool((await self.onboarding(ollama_url)).get("ready"))

    # -- convenience methods -----------------------------------------------

    async def ask(self, question: str, model: str = "llama3.2", **kwargs: Any) -> str:
        """One-liner: send a question and get the response text.

        Args:
            question: The user's question.
            model: Model identifier.
            **kwargs: Additional chat completion parameters.

        Returns:
            The assistant's reply as a plain string.
        """
        resp = await self.chat(
            messages=[{"role": "user", "content": question}],
            model=model,
            stream=False,
            **kwargs,
        )
        return resp.choices[0].message.content if resp.choices else ""

    async def complete(
        self,
        prompt: str,
        model: str = "llama3.2",
        system: str = "",
        **kwargs: Any,
    ) -> str:
        """Send a prompt with optional system message and get text back.

        Args:
            prompt: The user prompt.
            model: Model identifier.
            system: Optional system prompt.
            **kwargs: Additional chat completion parameters.

        Returns:
            The assistant's reply as a plain string.
        """
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = await self.chat(messages=messages, model=model, stream=False, **kwargs)
        return resp.choices[0].message.content if resp.choices else ""
