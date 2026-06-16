"""Tests for llmstack.mcp.server — JSON-RPC over stdio MCP server."""

from __future__ import annotations

import io
import json
import sys

import httpx
import pytest

from llmstack.agent.tools import ToolRegistry, ToolResult
from llmstack.mcp import server as mcp_server
from llmstack.mcp.server import (
    MCP_PROTOCOL_VERSION,
    SERVER_INFO,
    MCPServer,
    _read_line,
    _write_error,
    _write_result,
)


# ---------------------------------------------------------------------------
# Fakes / helpers
# ---------------------------------------------------------------------------


class _FakeTool:
    """Minimal Tool stand-in with schema() + execute()."""

    def __init__(self, name="echo", result=None, exc=None):
        self.name = name
        self._result = result if result is not None else ToolResult(output="ok")
        self._exc = exc
        self.calls: list[dict] = []

    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": f"{self.name} description",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        }

    async def execute(self, **kwargs) -> ToolResult:
        self.calls.append(kwargs)
        if self._exc:
            raise self._exc
        return self._result


def _registry(*tools) -> ToolRegistry:
    reg = ToolRegistry()
    for t in tools:
        reg.register(t)
    return reg


class _Resp:
    def __init__(self, payload=None, *, raise_exc=None):
        self._payload = payload or {}
        self._raise_exc = raise_exc

    def raise_for_status(self):
        if self._raise_exc:
            raise self._raise_exc

    def json(self):
        return self._payload


class _FakeClient:
    def __init__(self, *, post=None, post_exc=None, capture=None):
        self._post = post
        self._post_exc = post_exc
        self._capture = capture

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def post(self, url, json=None):
        if self._capture is not None:
            self._capture["url"] = url
            self._capture["json"] = json
        if self._post_exc:
            raise self._post_exc
        return self._post


def _patch_httpx(monkeypatch, **kw):
    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: _FakeClient(**kw))


def _capture_stdout(monkeypatch):
    buf = io.StringIO()
    monkeypatch.setattr(sys, "stdout", buf)
    return buf


# ---------------------------------------------------------------------------
# Stdio helpers
# ---------------------------------------------------------------------------


class TestWriteHelpers:
    def test_write_result(self, monkeypatch):
        buf = _capture_stdout(monkeypatch)
        _write_result(7, {"ok": True})
        payload = json.loads(buf.getvalue())
        assert payload == {"jsonrpc": "2.0", "id": 7, "result": {"ok": True}}

    def test_write_error(self, monkeypatch):
        buf = _capture_stdout(monkeypatch)
        _write_error(3, -32601, "Method not found")
        payload = json.loads(buf.getvalue())
        assert payload["error"] == {"code": -32601, "message": "Method not found"}
        assert payload["id"] == 3


class TestReadLine:
    async def test_reads_and_strips(self, monkeypatch):
        monkeypatch.setattr(sys, "stdin", io.StringIO("hello world\n"))
        line = await _read_line()
        assert line == "hello world"

    async def test_empty_returns_none(self, monkeypatch):
        monkeypatch.setattr(sys, "stdin", io.StringIO(""))
        assert await _read_line() is None

    async def test_eof_returns_none(self, monkeypatch):
        class _Stdin:
            def readline(self):
                raise EOFError

        monkeypatch.setattr(sys, "stdin", _Stdin())
        assert await _read_line() is None


# ---------------------------------------------------------------------------
# Simple handlers
# ---------------------------------------------------------------------------


class TestSimpleHandlers:
    async def test_initialize(self):
        result = await MCPServer(tools=_registry())._handle_initialize({})
        assert result["protocolVersion"] == MCP_PROTOCOL_VERSION
        assert result["serverInfo"] == SERVER_INFO
        assert "tools" in result["capabilities"]

    async def test_initialized_and_ping_empty(self):
        srv = MCPServer(tools=_registry())
        assert await srv._handle_initialized({}) == {}
        assert await srv._handle_ping({}) == {}

    async def test_resources_and_prompts_empty(self):
        srv = MCPServer(tools=_registry())
        assert await srv._handle_resources_list({}) == {"resources": []}
        assert await srv._handle_prompts_list({}) == {"prompts": []}

    def test_stop_sets_flag(self):
        srv = MCPServer(tools=_registry())
        srv._running = True
        srv.stop()
        assert srv._running is False


# ---------------------------------------------------------------------------
# tools/list
# ---------------------------------------------------------------------------


class TestToolsList:
    async def test_lists_agent_and_llm_tools(self):
        srv = MCPServer(tools=_registry(_FakeTool("echo")), model="m9")
        result = await srv._handle_tools_list({})
        names = [t["name"] for t in result["tools"]]
        assert "echo" in names
        assert "llmstack_chat" in names
        assert "llmstack_ask" in names

    async def test_agent_tool_schema_mapped(self):
        srv = MCPServer(tools=_registry(_FakeTool("echo")))
        result = await srv._handle_tools_list({})
        echo = next(t for t in result["tools"] if t["name"] == "echo")
        assert echo["description"] == "echo description"
        assert echo["inputSchema"] == {"type": "object", "properties": {}, "required": []}

    async def test_chat_tool_mentions_model_default(self):
        srv = MCPServer(tools=_registry(), model="customllm")
        result = await srv._handle_tools_list({})
        chat = next(t for t in result["tools"] if t["name"] == "llmstack_chat")
        model_desc = chat["inputSchema"]["properties"]["model"]["description"]
        assert "customllm" in model_desc
        assert chat["inputSchema"]["required"] == ["message"]


# ---------------------------------------------------------------------------
# tools/call dispatch
# ---------------------------------------------------------------------------


class TestToolsCall:
    async def test_dispatches_agent_tool(self):
        tool = _FakeTool("echo", result=ToolResult(output="hi there"))
        srv = MCPServer(tools=_registry(tool))
        result = await srv._handle_tools_call({"name": "echo", "arguments": {"x": 1}})
        assert result["content"][0]["text"] == "hi there"
        assert result["isError"] is False
        assert tool.calls == [{"x": 1}]

    async def test_agent_tool_failure_sets_iserror(self):
        tool = _FakeTool("echo", result=ToolResult(output="", success=False, error="boom"))
        srv = MCPServer(tools=_registry(tool))
        result = await srv._handle_tools_call({"name": "echo", "arguments": {}})
        assert result["isError"] is True
        assert "boom" in result["content"][0]["text"]

    async def test_unknown_tool(self):
        srv = MCPServer(tools=_registry())
        result = await srv._handle_tools_call({"name": "nope", "arguments": {}})
        assert result["isError"] is True
        assert "Unknown tool: nope" in result["content"][0]["text"]

    async def test_dispatches_llmstack_chat(self, monkeypatch):
        _patch_httpx(monkeypatch, post=_Resp({"message": {"content": "chat-out"}}))
        srv = MCPServer(tools=_registry())
        result = await srv._handle_tools_call(
            {"name": "llmstack_chat", "arguments": {"message": "hey"}}
        )
        assert result["content"][0]["text"] == "chat-out"

    async def test_dispatches_llmstack_ask(self, monkeypatch):
        engine = _install_fake_ask_engine(monkeypatch, answer="ans", sources=[])
        srv = MCPServer(tools=_registry())
        result = await srv._handle_tools_call(
            {"name": "llmstack_ask", "arguments": {"question": "q?", "paths": ["."]}}
        )
        assert result["content"][0]["text"] == "ans"
        assert engine.loaded is True


# ---------------------------------------------------------------------------
# llmstack_chat
# ---------------------------------------------------------------------------


class TestToolChat:
    async def test_chat_success_builds_request(self, monkeypatch):
        capture: dict = {}
        _patch_httpx(
            monkeypatch, post=_Resp({"message": {"content": "answer"}}), capture=capture
        )
        srv = MCPServer(tools=_registry(), ollama_url="http://host:1234", model="def-model")
        result = await srv._tool_chat({"message": "hello", "system": "be nice"})
        assert result["content"][0]["text"] == "answer"
        assert capture["url"] == "http://host:1234/api/chat"
        body = capture["json"]
        assert body["model"] == "def-model"
        assert body["stream"] is False
        assert body["messages"][0] == {"role": "system", "content": "be nice"}
        assert body["messages"][1] == {"role": "user", "content": "hello"}

    async def test_chat_no_system_prompt(self, monkeypatch):
        capture: dict = {}
        _patch_httpx(monkeypatch, post=_Resp({"message": {"content": "x"}}), capture=capture)
        srv = MCPServer(tools=_registry())
        await srv._tool_chat({"message": "hi"})
        assert len(capture["json"]["messages"]) == 1
        assert capture["json"]["messages"][0]["role"] == "user"

    async def test_chat_model_override(self, monkeypatch):
        capture: dict = {}
        _patch_httpx(monkeypatch, post=_Resp({"message": {"content": "x"}}), capture=capture)
        srv = MCPServer(tools=_registry(), model="default")
        await srv._tool_chat({"message": "hi", "model": "override-model"})
        assert capture["json"]["model"] == "override-model"

    async def test_chat_missing_content_defaults_empty(self, monkeypatch):
        _patch_httpx(monkeypatch, post=_Resp({}))
        srv = MCPServer(tools=_registry())
        result = await srv._tool_chat({"message": "hi"})
        assert result["content"][0]["text"] == ""

    async def test_chat_http_error_returns_iserror(self, monkeypatch):
        err = httpx.HTTPStatusError("e", request=None, response=httpx.Response(500))
        _patch_httpx(monkeypatch, post=_Resp(raise_exc=err))
        srv = MCPServer(tools=_registry())
        result = await srv._tool_chat({"message": "hi"})
        assert result["isError"] is True
        assert "LLM error" in result["content"][0]["text"]

    async def test_chat_connect_error_returns_iserror(self, monkeypatch):
        _patch_httpx(monkeypatch, post_exc=httpx.ConnectError("down"))
        srv = MCPServer(tools=_registry())
        result = await srv._tool_chat({"message": "hi"})
        assert result["isError"] is True
        assert "LLM error" in result["content"][0]["text"]


# ---------------------------------------------------------------------------
# llmstack_ask
# ---------------------------------------------------------------------------


class _FakeSource:
    def __init__(self, file="a.py", lines="1-2", relevance=0.9):
        self.file = file
        self.lines = lines
        self.relevance = relevance


class _FakeAskResult:
    def __init__(self, answer="ans", sources=None):
        self.answer = answer
        self.sources = sources or []


def _install_fake_ask_engine(monkeypatch, *, answer="ans", sources=None, load_exc=None):
    """Install a fake AskEngine into llmstack.ask.engine so the lazy import picks it up."""
    import llmstack.ask.engine as ask_engine_mod

    state = type("S", (), {})()
    state.loaded = False
    state.closed = False
    state.init_kwargs = None

    class _FakeAskEngine:
        def __init__(self, **kwargs):
            state.init_kwargs = kwargs

        async def load(self, paths, show_progress=False):
            if load_exc:
                raise load_exc
            state.loaded = True
            state.paths = paths

        async def ask_full(self, question):
            state.question = question
            return _FakeAskResult(answer=answer, sources=sources)

        async def close(self):
            state.closed = True

    monkeypatch.setattr(ask_engine_mod, "AskEngine", _FakeAskEngine)
    return state


class TestToolAsk:
    async def test_ask_success_no_sources(self, monkeypatch):
        state = _install_fake_ask_engine(monkeypatch, answer="the answer", sources=[])
        srv = MCPServer(tools=_registry(), ollama_url="http://o:1", model="askmodel")
        result = await srv._tool_ask({"question": "why?", "paths": ["x", "y"]})
        assert result["content"][0]["text"] == "the answer"
        assert state.closed is True
        assert state.init_kwargs["ollama_url"] == "http://o:1"
        assert state.init_kwargs["model"] == "askmodel"
        # paths converted to Path objects
        assert [str(p) for p in state.paths] == ["x", "y"]
        assert state.question == "why?"

    async def test_ask_with_sources_appends_citations(self, monkeypatch):
        sources = [_FakeSource("f.py", "10-12", 0.77)]
        _install_fake_ask_engine(monkeypatch, answer="body", sources=sources)
        srv = MCPServer(tools=_registry())
        result = await srv._tool_ask({"question": "q", "paths": ["."]})
        text = result["content"][0]["text"]
        assert text.startswith("body")
        assert "Sources:" in text
        assert "f.py:10-12" in text
        assert "0.77" in text

    async def test_ask_default_paths(self, monkeypatch):
        state = _install_fake_ask_engine(monkeypatch, answer="a", sources=[])
        srv = MCPServer(tools=_registry())
        await srv._tool_ask({"question": "q"})
        assert [str(p) for p in state.paths] == ["."]

    async def test_ask_model_override(self, monkeypatch):
        state = _install_fake_ask_engine(monkeypatch, answer="a", sources=[])
        srv = MCPServer(tools=_registry(), model="default")
        await srv._tool_ask({"question": "q", "paths": ["."], "model": "mymodel"})
        assert state.init_kwargs["model"] == "mymodel"

    async def test_ask_error_returns_iserror(self, monkeypatch):
        _install_fake_ask_engine(monkeypatch, load_exc=RuntimeError("load failed"))
        srv = MCPServer(tools=_registry())
        result = await srv._tool_ask({"question": "q", "paths": ["."]})
        assert result["isError"] is True
        assert "Ask error" in result["content"][0]["text"]
        assert "load failed" in result["content"][0]["text"]


# ---------------------------------------------------------------------------
# _handle_message dispatch
# ---------------------------------------------------------------------------


class TestHandleMessage:
    async def test_dispatches_and_writes_result(self, monkeypatch):
        buf = _capture_stdout(monkeypatch)
        srv = MCPServer(tools=_registry())
        await srv._handle_message({"id": 1, "method": "ping", "params": {}})
        payload = json.loads(buf.getvalue())
        assert payload["id"] == 1
        assert payload["result"] == {}

    async def test_notification_no_id_no_output(self, monkeypatch):
        buf = _capture_stdout(monkeypatch)
        srv = MCPServer(tools=_registry())
        await srv._handle_message({"method": "initialized", "params": {}})
        assert buf.getvalue() == ""

    async def test_unknown_method_with_id_writes_error(self, monkeypatch):
        buf = _capture_stdout(monkeypatch)
        srv = MCPServer(tools=_registry())
        await srv._handle_message({"id": 5, "method": "bogus", "params": {}})
        payload = json.loads(buf.getvalue())
        assert payload["error"]["code"] == -32601
        assert "bogus" in payload["error"]["message"]

    async def test_unknown_method_no_id_silent(self, monkeypatch):
        buf = _capture_stdout(monkeypatch)
        srv = MCPServer(tools=_registry())
        await srv._handle_message({"method": "bogus", "params": {}})
        assert buf.getvalue() == ""

    async def test_handler_exception_with_id_writes_internal_error(self, monkeypatch):
        buf = _capture_stdout(monkeypatch)
        srv = MCPServer(tools=_registry())

        async def _boom(params):
            raise ValueError("kaboom")

        monkeypatch.setattr(srv, "_handle_ping", _boom)
        await srv._handle_message({"id": 9, "method": "ping", "params": {}})
        payload = json.loads(buf.getvalue())
        assert payload["error"]["code"] == -32603
        assert "kaboom" in payload["error"]["message"]

    async def test_handler_exception_no_id_silent(self, monkeypatch):
        buf = _capture_stdout(monkeypatch)
        srv = MCPServer(tools=_registry())

        async def _boom(params):
            raise ValueError("kaboom")

        monkeypatch.setattr(srv, "_handle_initialized", _boom)
        await srv._handle_message({"method": "initialized", "params": {}})
        assert buf.getvalue() == ""

    async def test_defaults_for_missing_fields(self, monkeypatch):
        # No method/id/params keys -> method "" not found, id None -> silent
        buf = _capture_stdout(monkeypatch)
        srv = MCPServer(tools=_registry())
        await srv._handle_message({})
        assert buf.getvalue() == ""


# ---------------------------------------------------------------------------
# run() main loop
# ---------------------------------------------------------------------------


class TestRunLoop:
    async def test_processes_messages_then_eof(self, monkeypatch):
        buf = _capture_stdout(monkeypatch)
        srv = MCPServer(tools=_registry())

        lines = [json.dumps({"id": 1, "method": "ping", "params": {}}), None]

        async def _fake_read_line():
            return lines.pop(0)

        monkeypatch.setattr(mcp_server, "_read_line", _fake_read_line)
        await srv.run()
        # ping result was written, loop ended on None
        out_lines = [line for line in buf.getvalue().splitlines() if line]
        assert len(out_lines) == 1
        assert json.loads(out_lines[0])["id"] == 1
        assert srv._running is True  # loop exits via break, not stop()

    async def test_parse_error_on_bad_json(self, monkeypatch):
        buf = _capture_stdout(monkeypatch)
        srv = MCPServer(tools=_registry())

        lines = ["not-json", None]

        async def _fake_read_line():
            return lines.pop(0)

        monkeypatch.setattr(mcp_server, "_read_line", _fake_read_line)
        await srv.run()
        payload = json.loads(buf.getvalue().splitlines()[0])
        assert payload["error"]["code"] == -32700
        assert payload["error"]["message"] == "Parse error"
        assert payload["id"] is None

    async def test_loop_breaks_on_read_exception(self, monkeypatch):
        _capture_stdout(monkeypatch)
        srv = MCPServer(tools=_registry())

        async def _fake_read_line():
            raise RuntimeError("read blew up")

        monkeypatch.setattr(mcp_server, "_read_line", _fake_read_line)
        # Should swallow the exception and exit cleanly
        await srv.run()
        assert srv._running is True

    async def test_stop_ends_loop(self, monkeypatch):
        _capture_stdout(monkeypatch)
        srv = MCPServer(tools=_registry())

        async def _fake_read_line():
            srv.stop()
            return json.dumps({"method": "initialized", "params": {}})

        monkeypatch.setattr(mcp_server, "_read_line", _fake_read_line)
        await srv.run()
        assert srv._running is False


# ---------------------------------------------------------------------------
# Defaults / construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_default_registry_has_tools(self):
        srv = MCPServer()
        assert srv.tools.names()  # non-empty default registry
        assert srv.ollama_url == "http://localhost:11434"
        assert srv.model == "llama3.2"

    @pytest.mark.parametrize("method", ["read_file", "write_file", "grep", "shell"])
    def test_default_registry_known_tools(self, method):
        srv = MCPServer()
        assert srv.tools.get(method) is not None
