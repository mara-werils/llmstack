"""Comprehensive tests for the AgentLoop — ReAct agent with tool calling."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from llmstack.agent.loop import AgentConfig, AgentEvent, AgentLoop
from llmstack.agent.tools import Tool, ToolParam, ToolRegistry, ToolResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class MockTool(Tool):
    name = "mock_tool"
    description = "A mock tool for testing"
    parameters = [ToolParam(name="arg", description="An argument")]

    async def execute(self, **kwargs) -> ToolResult:
        return ToolResult(output=f"mock result: {kwargs.get('arg', '')}")


class FailingTool(Tool):
    name = "failing_tool"
    description = "Always fails"
    parameters = []

    async def execute(self, **kwargs) -> ToolResult:
        raise RuntimeError("tool broke")


@pytest.fixture
def registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(MockTool())
    return reg


@pytest.fixture
def config() -> AgentConfig:
    return AgentConfig(
        model="test",
        api_url="http://localhost:11434",
        max_steps=5,
        timeout=10,
    )


@pytest.fixture
def agent(config: AgentConfig, registry: ToolRegistry) -> AgentLoop:
    return AgentLoop(config, registry)


# ---------------------------------------------------------------------------
# AgentConfig
# ---------------------------------------------------------------------------


class TestAgentConfig:
    def test_defaults(self) -> None:
        c = AgentConfig()
        assert c.model == "llama3.2"
        assert c.max_steps == 25
        assert c.temperature == 0.1

    def test_custom_values(self) -> None:
        c = AgentConfig(model="gpt-4", max_steps=10, temperature=0.5)
        assert c.model == "gpt-4"
        assert c.max_steps == 10
        assert c.temperature == 0.5


# ---------------------------------------------------------------------------
# AgentEvent
# ---------------------------------------------------------------------------


class TestAgentEvent:
    def test_to_dict_minimal(self) -> None:
        e = AgentEvent(type="thinking", step=1)
        d = e.to_dict()
        assert d["type"] == "thinking"
        assert d["step"] == 1
        assert "content" not in d

    def test_to_dict_full(self) -> None:
        e = AgentEvent(
            type="tool_call",
            content="",
            tool_name="grep",
            tool_args={"pattern": "foo"},
            step=2,
            elapsed_ms=123.456,
        )
        d = e.to_dict()
        assert d["tool_name"] == "grep"
        assert d["tool_args"] == {"pattern": "foo"}
        assert d["elapsed_ms"] == 123.5

    def test_to_dict_excludes_empty_fields(self) -> None:
        e = AgentEvent(type="done", step=3)
        d = e.to_dict()
        assert "content" not in d
        assert "tool_name" not in d
        assert "tool_args" not in d
        assert "elapsed_ms" not in d


# ---------------------------------------------------------------------------
# AgentLoop.run — final answer (no tool calls)
# ---------------------------------------------------------------------------


class TestAgentFinalAnswer:
    @pytest.mark.asyncio
    async def test_direct_answer(self, agent: AgentLoop) -> None:
        """LLM responds with text only — no tool calls."""
        mock_response = {
            "message": {"role": "assistant", "content": "The answer is 42."},
        }
        with patch.object(agent, "_call_llm", new_callable=AsyncMock, return_value=mock_response):
            events = []
            async for event in agent.run("What is 6*7?"):
                events.append(event)

        types = [e.type for e in events]
        assert "message" in types
        assert "done" in types
        assert events[-1].type == "done"
        msg_event = next(e for e in events if e.type == "message")
        assert msg_event.content == "The answer is 42."

    @pytest.mark.asyncio
    async def test_empty_content_still_yields_done(self, agent: AgentLoop) -> None:
        mock_response = {
            "message": {"role": "assistant", "content": ""},
        }
        with patch.object(agent, "_call_llm", new_callable=AsyncMock, return_value=mock_response):
            events = []
            async for event in agent.run("empty?"):
                events.append(event)
        types = [e.type for e in events]
        assert "done" in types
        # No message event if content is empty
        assert "message" not in types


# ---------------------------------------------------------------------------
# AgentLoop.run — tool calls
# ---------------------------------------------------------------------------


class TestAgentToolCalls:
    @pytest.mark.asyncio
    async def test_tool_call_then_answer(self, agent: AgentLoop) -> None:
        """LLM calls a tool, then gives final answer."""
        tool_response = {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {
                            "name": "mock_tool",
                            "arguments": json.dumps({"arg": "hello"}),
                        },
                    }
                ],
            },
        }
        final_response = {
            "message": {"role": "assistant", "content": "Done."},
        }

        call_count = 0

        async def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return tool_response
            return final_response

        with patch.object(agent, "_call_llm", side_effect=side_effect):
            events = []
            async for event in agent.run("use the tool"):
                events.append(event)

        types = [e.type for e in events]
        assert "tool_call" in types
        assert "tool_result" in types
        assert "message" in types
        assert "done" in types

        tc_event = next(e for e in events if e.type == "tool_call")
        assert tc_event.tool_name == "mock_tool"

        tr_event = next(e for e in events if e.type == "tool_result")
        assert "mock result: hello" in tr_event.content

    @pytest.mark.asyncio
    async def test_unknown_tool_returns_error(self, agent: AgentLoop) -> None:
        tool_response = {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {"name": "nonexistent_tool", "arguments": "{}"},
                    }
                ],
            },
        }
        final_response = {"message": {"role": "assistant", "content": "ok"}}

        responses = iter([tool_response, final_response])

        with patch.object(agent, "_call_llm", new_callable=AsyncMock, side_effect=lambda *a, **k: next(responses)):
            events = []
            async for event in agent.run("call unknown"):
                events.append(event)

        tr_event = next(e for e in events if e.type == "tool_result")
        assert "Unknown tool" in tr_event.content

    @pytest.mark.asyncio
    async def test_tool_execution_exception(self, agent: AgentLoop) -> None:
        agent.tools.register(FailingTool())
        tool_response = {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "call_1", "function": {"name": "failing_tool", "arguments": "{}"}},
                ],
            },
        }
        final_response = {"message": {"role": "assistant", "content": "done"}}
        responses = iter([tool_response, final_response])

        with patch.object(agent, "_call_llm", new_callable=AsyncMock, side_effect=lambda *a, **k: next(responses)):
            events = []
            async for event in agent.run("use failing"):
                events.append(event)

        tr_event = next(e for e in events if e.type == "tool_result")
        assert "tool broke" in tr_event.content

    @pytest.mark.asyncio
    async def test_tool_args_as_dict(self, agent: AgentLoop) -> None:
        """Arguments already parsed as dict (not JSON string)."""
        tool_response = {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "c1", "function": {"name": "mock_tool", "arguments": {"arg": "val"}}},
                ],
            },
        }
        final_response = {"message": {"role": "assistant", "content": "done"}}
        responses = iter([tool_response, final_response])

        with patch.object(agent, "_call_llm", new_callable=AsyncMock, side_effect=lambda *a, **k: next(responses)):
            events = []
            async for event in agent.run("go"):
                events.append(event)

        tc_event = next(e for e in events if e.type == "tool_call")
        assert tc_event.tool_args == {"arg": "val"}

    @pytest.mark.asyncio
    async def test_tool_args_invalid_json(self, agent: AgentLoop) -> None:
        """Invalid JSON arguments should default to empty dict."""
        tool_response = {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "c1", "function": {"name": "mock_tool", "arguments": "{invalid}"}},
                ],
            },
        }
        final_response = {"message": {"role": "assistant", "content": "done"}}
        responses = iter([tool_response, final_response])

        with patch.object(agent, "_call_llm", new_callable=AsyncMock, side_effect=lambda *a, **k: next(responses)):
            events = []
            async for event in agent.run("go"):
                events.append(event)

        tc_event = next(e for e in events if e.type == "tool_call")
        assert tc_event.tool_args == {}


# ---------------------------------------------------------------------------
# AgentLoop.run — error and max steps
# ---------------------------------------------------------------------------


class TestAgentErrors:
    @pytest.mark.asyncio
    async def test_llm_error_yields_error_event(self, agent: AgentLoop) -> None:
        with patch.object(agent, "_call_llm", new_callable=AsyncMock, side_effect=RuntimeError("LLM down")):
            events = []
            async for event in agent.run("task"):
                events.append(event)

        assert len(events) == 1
        assert events[0].type == "error"
        assert "LLM down" in events[0].content

    @pytest.mark.asyncio
    async def test_max_steps_reached(self, agent: AgentLoop) -> None:
        """Agent keeps calling tools until max_steps is reached."""
        tool_response = {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "c", "function": {"name": "mock_tool", "arguments": "{}"}},
                ],
            },
        }

        with patch.object(agent, "_call_llm", new_callable=AsyncMock, return_value=tool_response):
            events = []
            async for event in agent.run("loop forever"):
                events.append(event)

        assert events[-1].type == "error"
        assert "maximum steps" in events[-1].content.lower()
        assert agent.steps_taken == agent.config.max_steps


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


class TestAgentProperties:
    @pytest.mark.asyncio
    async def test_messages_property(self, agent: AgentLoop) -> None:
        mock_response = {"message": {"role": "assistant", "content": "hi"}}
        with patch.object(agent, "_call_llm", new_callable=AsyncMock, return_value=mock_response):
            async for _ in agent.run("hello"):
                pass
        msgs = agent.messages
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert msgs[1]["content"] == "hello"

    @pytest.mark.asyncio
    async def test_steps_taken(self, agent: AgentLoop) -> None:
        mock_response = {"message": {"role": "assistant", "content": "done"}}
        with patch.object(agent, "_call_llm", new_callable=AsyncMock, return_value=mock_response):
            async for _ in agent.run("task"):
                pass
        assert agent.steps_taken == 1


# ---------------------------------------------------------------------------
# _call_llm routing
# ---------------------------------------------------------------------------


class TestCallLLMRouting:
    @pytest.mark.asyncio
    async def test_ollama_detected_by_port(self) -> None:
        config = AgentConfig(api_url="http://localhost:11434")
        agent = AgentLoop(config, ToolRegistry())
        with (
            patch.object(agent, "_call_ollama", new_callable=AsyncMock, return_value={}) as mock_ollama,
            patch.object(agent, "_call_openai_compat", new_callable=AsyncMock) as mock_openai,
        ):
            await agent._call_llm([], True)
            mock_ollama.assert_awaited_once()
            mock_openai.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_openai_compat_for_other_urls(self) -> None:
        config = AgentConfig(api_url="http://api.example.com")
        agent = AgentLoop(config, ToolRegistry())
        with (
            patch.object(agent, "_call_ollama", new_callable=AsyncMock) as mock_ollama,
            patch.object(agent, "_call_openai_compat", new_callable=AsyncMock, return_value={}) as mock_openai,
        ):
            await agent._call_llm([], False)
            mock_openai.assert_awaited_once()
            mock_ollama.assert_not_awaited()
