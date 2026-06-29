"""Comprehensive tests for SDK client — sync and async clients, helpers."""

from __future__ import annotations

from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from llmstack.sdk.client import (
    AsyncClient,
    Client,
    LLMStackError,
    _build_headers,
    _parse_sse_line,
    _raise_for_error,
)
from llmstack.sdk.types import (
    ChatResponse,
    EmbeddingsResponse,
    HealthResponse,
    IngestResponse,
    ModelsResponse,
    RAGResponse,
)


# ---------------------------------------------------------------------------
# _build_headers
# ---------------------------------------------------------------------------


class TestBuildHeaders:
    def test_without_api_key(self) -> None:
        headers = _build_headers(None)
        assert headers["Content-Type"] == "application/json"
        assert "Authorization" not in headers

    def test_with_api_key(self) -> None:
        headers = _build_headers("sk-12345")
        assert headers["Authorization"] == "Bearer sk-12345"


# ---------------------------------------------------------------------------
# _parse_sse_line
# ---------------------------------------------------------------------------


class TestParseSSELine:
    def test_valid_data(self) -> None:
        result = _parse_sse_line('data: {"key": "val"}')
        assert result == {"key": "val"}

    def test_done_marker(self) -> None:
        assert _parse_sse_line("data: [DONE]") is None

    def test_empty(self) -> None:
        assert _parse_sse_line("") is None

    def test_whitespace(self) -> None:
        assert _parse_sse_line("   ") is None

    def test_comment(self) -> None:
        assert _parse_sse_line(": keep-alive") is None

    def test_invalid_json(self) -> None:
        assert _parse_sse_line("data: {bad}") is None

    def test_non_data_line(self) -> None:
        assert _parse_sse_line("event: update") is None

    def test_data_with_extra_spaces(self) -> None:
        result = _parse_sse_line('data:   {"k": 1}  ')
        assert result == {"k": 1}


# ---------------------------------------------------------------------------
# _raise_for_error
# ---------------------------------------------------------------------------


class TestRaiseForError:
    def test_success_no_raise(self) -> None:
        resp = MagicMock()
        resp.status_code = 200
        _raise_for_error(resp)  # should not raise

    def test_400_raises(self) -> None:
        resp = MagicMock()
        resp.status_code = 400
        resp.json.return_value = {"error": "bad request"}
        with pytest.raises(LLMStackError) as exc_info:
            _raise_for_error(resp)
        assert exc_info.value.status_code == 400

    def test_500_with_non_json_body(self) -> None:
        resp = MagicMock()
        resp.status_code = 500
        resp.json.side_effect = Exception("not json")
        resp.text = "Internal Server Error"
        with pytest.raises(LLMStackError) as exc_info:
            _raise_for_error(resp)
        assert exc_info.value.detail == "Internal Server Error"


# ---------------------------------------------------------------------------
# LLMStackError
# ---------------------------------------------------------------------------


class TestLLMStackError:
    def test_attributes(self) -> None:
        err = LLMStackError(404, "not found")
        assert err.status_code == 404
        assert err.detail == "not found"
        assert "404" in str(err)

    def test_dict_detail(self) -> None:
        err = LLMStackError(422, {"error": "validation"})
        assert err.detail == {"error": "validation"}


# ---------------------------------------------------------------------------
# Client (sync)
# ---------------------------------------------------------------------------


class _FakeStreamResp:
    def __init__(self, lines, status_code=200):
        self.status_code = status_code
        self._lines = lines
        self.headers = {}
        self.text = ""

    def iter_lines(self):
        return iter(self._lines)

    async def aiter_lines(self):
        for line in self._lines:
            yield line

    def json(self):
        return {}


class _FakeSyncStreamCtx:
    def __init__(self, resp):
        self._resp = resp

    def __enter__(self):
        return self._resp

    def __exit__(self, *args):
        return False


class _FakeAsyncStreamCtx:
    def __init__(self, resp):
        self._resp = resp

    async def __aenter__(self):
        return self._resp

    async def __aexit__(self, *args):
        return False


class TestSyncClient:
    def test_init_strips_url(self) -> None:
        c = Client(base_url="http://localhost:8000/")
        assert c.base_url == "http://localhost:8000"
        c.close()

    def test_repr_hides_api_key(self) -> None:
        c = Client(api_key="sk-secret")
        assert "***" in repr(c)
        assert "sk-secret" not in repr(c)
        c.close()

    def test_repr_no_api_key(self) -> None:
        c = Client()
        assert "None" in repr(c)
        c.close()

    def test_chat_streaming(self) -> None:
        lines = [
            'data: {"id":"1","choices":[{"index":0,"delta":{"content":"Hi"},'
            '"finish_reason":null}]}',
            "data: [DONE]",
        ]
        fake_resp = _FakeStreamResp(lines)
        with Client() as c:
            with patch.object(c._client, "stream", return_value=_FakeSyncStreamCtx(fake_resp)):
                deltas = list(c.chat(messages=[{"role": "user", "content": "hi"}], stream=True))
        assert len(deltas) == 1
        assert deltas[0].delta_content == "Hi"

    def test_rag_query_streaming(self) -> None:
        lines = [
            'data: {"token": "Hel", "done": false}',
            'data: {"token": "lo", "done": true, "sources": ["a.py"]}',
            "data: [DONE]",
        ]
        fake_resp = _FakeStreamResp(lines)
        with Client() as c:
            with patch.object(c._client, "stream", return_value=_FakeSyncStreamCtx(fake_resp)):
                deltas = list(c.rag_query(question="what?", stream=True))
        assert len(deltas) == 2
        assert deltas[1].sources == ["a.py"]

    def test_context_manager(self) -> None:
        with Client() as c:
            assert c is not None

    def test_chat_non_streaming(self) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "id": "test-id",
            "object": "chat.completion",
            "model": "test",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hi!"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        }
        mock_resp.headers = {}

        with Client() as c:
            with patch.object(c._client, "post", return_value=mock_resp):
                resp = c.chat(messages=[{"role": "user", "content": "hello"}])
                assert isinstance(resp, ChatResponse)
                assert resp.choices[0].message.content == "Hi!"

    def test_embed(self) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "object": "list",
            "model": "bge-m3",
            "data": [{"index": 0, "embedding": [0.1, 0.2], "object": "embedding"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 0, "total_tokens": 3},
        }

        with Client() as c:
            with patch.object(c._client, "post", return_value=mock_resp):
                resp = c.embed("hello")
                assert isinstance(resp, EmbeddingsResponse)
                assert len(resp.data) == 1
                assert resp.data[0].embedding == [0.1, 0.2]

    def test_models(self) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "object": "list",
            "data": [{"id": "llama3.2", "object": "model", "owned_by": "meta"}],
        }

        with Client() as c:
            with patch.object(c._client, "get", return_value=mock_resp):
                resp = c.models()
                assert isinstance(resp, ModelsResponse)
                assert resp.data[0].id == "llama3.2"

    def test_health(self) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"status": "healthy", "services": {"ollama": True}}

        with Client() as c:
            with patch.object(c._client, "get", return_value=mock_resp):
                resp = c.health()
                assert isinstance(resp, HealthResponse)
                assert resp.status == "healthy"

    def test_savings(self) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "total_saved_usd": 1.23,
            "total_requests": 10,
            "subscription": {"key": "cursor-pro", "months_covered": 0.06},
        }

        with Client() as c:
            with patch.object(c._client, "get", return_value=mock_resp) as mock_get:
                summary = c.savings(plan="cursor-pro")
                assert summary["total_saved_usd"] == 1.23
                assert summary["subscription"]["key"] == "cursor-pro"
                # plan is passed through as a query param
                assert mock_get.call_args.kwargs["params"] == {"plan": "cursor-pro"}

    def test_savings_default_plan_sends_no_params(self) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"total_saved_usd": 0.0, "total_requests": 0}

        with Client() as c:
            with patch.object(c._client, "get", return_value=mock_resp) as mock_get:
                c.savings()
                assert mock_get.call_args.kwargs["params"] is None

    def test_rag_status(self) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"documents": 3, "chunks": 42}

        with Client() as c:
            with patch.object(c._client, "get", return_value=mock_resp) as mock_get:
                status = c.rag_status()
                assert status == {"documents": 3, "chunks": 42}
                assert mock_get.call_args.args[0] == "/v1/rag/status"

    def test_onboarding(self) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "ready": False,
            "recommended": {"chat_model": {"name": "llama3.2"}},
            "hints": ["ollama pull llama3.2"],
        }

        with Client() as c:
            with patch.object(c._client, "get", return_value=mock_resp) as mock_get:
                report = c.onboarding(ollama_url="http://host:11434")
                assert report["ready"] is False
                assert report["hints"] == ["ollama pull llama3.2"]
                assert mock_get.call_args.kwargs["params"] == {"ollama_url": "http://host:11434"}

    def test_onboarding_default_sends_no_params(self) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"ready": True}

        with Client() as c:
            with patch.object(c._client, "get", return_value=mock_resp) as mock_get:
                c.onboarding()
                assert mock_get.call_args.kwargs["params"] is None

    def test_ready_returns_bool(self) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"ready": True}

        with Client() as c:
            with patch.object(c._client, "get", return_value=mock_resp):
                assert c.ready() is True

    def test_ready_false_when_not_ready(self) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"ready": False}

        with Client() as c:
            with patch.object(c._client, "get", return_value=mock_resp):
                assert c.ready() is False

    def test_rag_ingest(self) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"status": "ok", "chunks_stored": 5, "source": "doc.txt"}

        with Client() as c:
            with patch.object(c._client, "post", return_value=mock_resp):
                resp = c.rag_ingest(text="content", source="doc.txt")
                assert isinstance(resp, IngestResponse)
                assert resp.chunks_stored == 5

    def test_rag_ingest_with_metadata(self) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"status": "ok", "chunks_stored": 3, "source": "x.py"}

        with Client() as c:
            with patch.object(c._client, "post", return_value=mock_resp) as mock_post:
                c.rag_ingest(text="x", source="x.py", metadata={"lang": "python"})
                payload = mock_post.call_args[1]["json"]
                assert payload["metadata"] == {"lang": "python"}

    def test_rag_query(self) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "answer": "The answer",
            "sources": ["a.py:1-5"],
            "model": "llama3.2",
        }

        with Client() as c:
            with patch.object(c._client, "post", return_value=mock_resp):
                resp = c.rag_query(question="what?")
                assert isinstance(resp, RAGResponse)
                assert resp.answer == "The answer"

    def test_ask_convenience(self) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "id": "c1",
            "object": "chat.completion",
            "model": "test",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "42"},
                    "finish_reason": "stop",
                }
            ],
        }
        mock_resp.headers = {}

        with Client() as c:
            with patch.object(c._client, "post", return_value=mock_resp):
                result = c.ask("What is 6*7?")
                assert result == "42"

    def test_ask_no_choices(self) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "id": "c1",
            "object": "chat.completion",
            "model": "test",
            "choices": [],
        }
        mock_resp.headers = {}

        with Client() as c:
            with patch.object(c._client, "post", return_value=mock_resp):
                result = c.ask("anything")
                assert result == ""

    def test_complete_with_system(self) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "id": "c1",
            "object": "chat.completion",
            "model": "test",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "result"},
                    "finish_reason": "stop",
                }
            ],
        }
        mock_resp.headers = {}

        with Client() as c:
            with patch.object(c._client, "post", return_value=mock_resp) as mock_post:
                result = c.complete("do stuff", system="Be helpful")
                assert result == "result"
                payload = mock_post.call_args[1]["json"]
                assert payload["messages"][0]["role"] == "system"

    def test_complete_without_system(self) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "id": "c1",
            "object": "chat.completion",
            "model": "test",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
        }
        mock_resp.headers = {}

        with Client() as c:
            with patch.object(c._client, "post", return_value=mock_resp) as mock_post:
                c.complete("do stuff")
                payload = mock_post.call_args[1]["json"]
                assert payload["messages"][0]["role"] == "user"


# ---------------------------------------------------------------------------
# AsyncClient
# ---------------------------------------------------------------------------


class TestAsyncClient:
    def test_init(self) -> None:
        c = AsyncClient(base_url="http://test:8000/", api_key="key-123")
        assert c.base_url == "http://test:8000"
        assert c.api_key == "key-123"

    @pytest.mark.asyncio
    async def test_context_manager(self) -> None:
        async with AsyncClient() as c:
            assert c is not None

    @pytest.mark.asyncio
    async def test_chat_non_streaming(self) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "id": "test-id",
            "object": "chat.completion",
            "model": "test",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello!"},
                    "finish_reason": "stop",
                }
            ],
        }
        mock_resp.headers = {}

        async with AsyncClient() as c:
            with patch.object(c._client, "post", new_callable=AsyncMock, return_value=mock_resp):
                resp = await c.chat(messages=[{"role": "user", "content": "hi"}])
                assert isinstance(resp, ChatResponse)
                assert resp.choices[0].message.content == "Hello!"

    @pytest.mark.asyncio
    async def test_embed(self) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "object": "list",
            "model": "bge",
            "data": [{"index": 0, "embedding": [0.5], "object": "embedding"}],
        }

        async with AsyncClient() as c:
            with patch.object(c._client, "post", new_callable=AsyncMock, return_value=mock_resp):
                resp = await c.embed("test")
                assert isinstance(resp, EmbeddingsResponse)

    @pytest.mark.asyncio
    async def test_rag_ingest(self) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"status": "ok", "chunks_stored": 2, "source": "f.py"}

        async with AsyncClient() as c:
            with patch.object(c._client, "post", new_callable=AsyncMock, return_value=mock_resp):
                resp = await c.rag_ingest(text="code", source="f.py")
                assert resp.chunks_stored == 2

    @pytest.mark.asyncio
    async def test_rag_ingest_with_metadata(self) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"status": "ok", "chunks_stored": 1, "source": "f.py"}

        async with AsyncClient() as c:
            with patch.object(
                c._client, "post", new_callable=AsyncMock, return_value=mock_resp
            ) as mock_post:
                await c.rag_ingest(text="code", source="f.py", metadata={"lang": "python"})
                payload = mock_post.call_args[1]["json"]
                assert payload["metadata"] == {"lang": "python"}

    @pytest.mark.asyncio
    async def test_rag_query(self) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"answer": "yes", "sources": []}

        async with AsyncClient() as c:
            with patch.object(c._client, "post", new_callable=AsyncMock, return_value=mock_resp):
                resp = await c.rag_query(question="is it?")
                assert resp.answer == "yes"

    @pytest.mark.asyncio
    async def test_models(self) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"object": "list", "data": []}

        async with AsyncClient() as c:
            with patch.object(c._client, "get", new_callable=AsyncMock, return_value=mock_resp):
                resp = await c.models()
                assert isinstance(resp, ModelsResponse)

    @pytest.mark.asyncio
    async def test_health(self) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"status": "ok", "services": {}}

        async with AsyncClient() as c:
            with patch.object(c._client, "get", new_callable=AsyncMock, return_value=mock_resp):
                resp = await c.health()
                assert resp.status == "ok"

    @pytest.mark.asyncio
    async def test_savings(self) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"total_saved_usd": 2.5, "total_requests": 4}

        async with AsyncClient() as c:
            with patch.object(c._client, "get", new_callable=AsyncMock, return_value=mock_resp):
                summary = await c.savings()
                assert summary["total_saved_usd"] == 2.5

    @pytest.mark.asyncio
    async def test_rag_status(self) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"documents": 1, "chunks": 7}

        async with AsyncClient() as c:
            with patch.object(c._client, "get", new_callable=AsyncMock, return_value=mock_resp):
                status = await c.rag_status()
                assert status == {"documents": 1, "chunks": 7}

    @pytest.mark.asyncio
    async def test_onboarding(self) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"ready": True, "hints": []}

        async with AsyncClient() as c:
            with patch.object(c._client, "get", new_callable=AsyncMock, return_value=mock_resp):
                report = await c.onboarding()
                assert report["ready"] is True

    @pytest.mark.asyncio
    async def test_ready(self) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"ready": False}

        async with AsyncClient() as c:
            with patch.object(c._client, "get", new_callable=AsyncMock, return_value=mock_resp):
                assert await c.ready() is False

    @pytest.mark.asyncio
    async def test_ask_convenience(self) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "id": "c1",
            "object": "chat.completion",
            "model": "test",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "answer"},
                    "finish_reason": "stop",
                }
            ],
        }
        mock_resp.headers = {}

        async with AsyncClient() as c:
            with patch.object(c._client, "post", new_callable=AsyncMock, return_value=mock_resp):
                result = await c.ask("question?")
                assert result == "answer"

    @pytest.mark.asyncio
    async def test_complete_with_system(self) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "id": "c1",
            "object": "chat.completion",
            "model": "test",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "done"},
                    "finish_reason": "stop",
                }
            ],
        }
        mock_resp.headers = {}

        async with AsyncClient() as c:
            with patch.object(
                c._client, "post", new_callable=AsyncMock, return_value=mock_resp
            ) as mock_post:
                result = await c.complete("prompt", system="sys")
                assert result == "done"
                payload = mock_post.call_args[1]["json"]
                assert payload["messages"][0]["role"] == "system"

    @pytest.mark.asyncio
    async def test_chat_error_raises(self) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.json.return_value = {"error": "server error"}

        async with AsyncClient() as c:
            with patch.object(c._client, "post", new_callable=AsyncMock, return_value=mock_resp):
                with pytest.raises(LLMStackError):
                    await c.chat(messages=[{"role": "user", "content": "hi"}])

    def test_repr_hides_api_key(self) -> None:
        c = AsyncClient(api_key="sk-secret")
        assert "***" in repr(c)
        assert "sk-secret" not in repr(c)

    @pytest.mark.asyncio
    async def test_chat_streaming(self) -> None:
        lines = [
            'data: {"id":"1","choices":[{"index":0,"delta":{"content":"Hi"},'
            '"finish_reason":null}]}',
            "data: [DONE]",
        ]
        fake_resp = _FakeStreamResp(lines)
        async with AsyncClient() as c:
            with patch.object(c._client, "stream", return_value=_FakeAsyncStreamCtx(fake_resp)):
                result = await c.chat(messages=[{"role": "user", "content": "hi"}], stream=True)
                deltas = [d async for d in result]
        assert len(deltas) == 1
        assert deltas[0].delta_content == "Hi"

    @pytest.mark.asyncio
    async def test_rag_query_streaming(self) -> None:
        lines = [
            'data: {"token": "Hel", "done": false}',
            'data: {"token": "lo", "done": true, "sources": ["a.py"]}',
            "data: [DONE]",
        ]
        fake_resp = _FakeStreamResp(lines)
        async with AsyncClient() as c:
            with patch.object(c._client, "stream", return_value=_FakeAsyncStreamCtx(fake_resp)):
                result = await c.rag_query(question="what?", stream=True)
                deltas = [d async for d in result]
        assert len(deltas) == 2
        assert deltas[1].sources == ["a.py"]


class TestLearnNamespace:
    def test_learn_returns_learn_client_bound_to_base_url(self) -> None:
        from llmstack.sdk.learn_client import LearnClient

        with Client(base_url="http://gw:9000", api_key="sk-abc") as c:
            assert isinstance(c.learn, LearnClient)
            assert c.learn._base_url == "http://gw:9000"
            # Auth header is propagated to the learn namespace.
            assert c.learn._headers.get("Authorization") == "Bearer sk-abc"

    def test_learn_is_cached(self) -> None:
        with Client() as c:
            assert c.learn is c.learn
