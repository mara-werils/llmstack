"""Agent loop — ReAct-style agent that plans, executes tools, and iterates.

The agent works with any OpenAI-compatible API (local Ollama, gateway, or cloud).
It uses the function-calling / tool-use protocol to decide which tools to invoke.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import AsyncIterator

import httpx

from llmstack.agent.tools import ToolRegistry, ToolResult

logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """Configuration for an agent run."""

    model: str = "llama3.2"
    api_url: str = "http://localhost:11434"
    api_key: str = ""
    max_steps: int = 25
    max_tokens: int = 4096
    temperature: float = 0.1
    system_prompt: str = ""
    timeout: int = 300


@dataclass
class AgentEvent:
    """An event in the agent's execution trace."""

    type: str  # "thinking", "tool_call", "tool_result", "message", "error", "done"
    content: str = ""
    tool_name: str = ""
    tool_args: dict = field(default_factory=dict)
    step: int = 0
    elapsed_ms: float = 0.0

    def to_dict(self) -> dict:
        d: dict = {"type": self.type, "step": self.step}
        if self.content:
            d["content"] = self.content
        if self.tool_name:
            d["tool_name"] = self.tool_name
        if self.tool_args:
            d["tool_args"] = self.tool_args
        if self.elapsed_ms:
            d["elapsed_ms"] = round(self.elapsed_ms, 1)
        return d


_DEFAULT_SYSTEM = """\
You are a capable AI agent. You have access to tools to help complete tasks.

Rules:
1. Break complex tasks into steps and use tools to accomplish each step.
2. After each tool result, reflect on what you learned and decide your next action.
3. When you have enough information to answer, provide a clear final response.
4. If a tool fails, try an alternative approach.
5. Never fabricate file contents or command outputs — always use tools to verify.
"""


class AgentLoop:
    """ReAct agent loop with tool calling.

    Flow:
    1. Send messages + tool schemas to LLM
    2. If LLM returns tool_calls → execute tools → add results → goto 1
    3. If LLM returns a text message → done (final answer)
    4. Repeat up to max_steps
    """

    def __init__(self, config: AgentConfig, tools: ToolRegistry):
        self.config = config
        self.tools = tools
        self._messages: list[dict] = []
        self._step = 0

    async def run(self, task: str) -> AsyncIterator[AgentEvent]:
        """Execute the agent task, yielding events as they occur."""
        system = self.config.system_prompt or _DEFAULT_SYSTEM
        self._messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": task},
        ]
        self._step = 0

        tool_schemas = self.tools.schemas()
        is_ollama = "/api/" in self.config.api_url or ":11434" in self.config.api_url

        while self._step < self.config.max_steps:
            self._step += 1
            t0 = time.monotonic()

            try:
                response = await self._call_llm(tool_schemas, is_ollama)
            except Exception as exc:
                yield AgentEvent(
                    type="error",
                    content=str(exc),
                    step=self._step,
                    elapsed_ms=(time.monotonic() - t0) * 1000,
                )
                return

            elapsed = (time.monotonic() - t0) * 1000
            message = response.get("message", {})

            # Check if LLM wants to call tools
            tool_calls = message.get("tool_calls", [])

            if tool_calls:
                # Add assistant message with tool calls to history
                self._messages.append(message)

                for tc in tool_calls:
                    func = tc.get("function", {})
                    tool_name = func.get("name", "")
                    raw_args = func.get("arguments", "{}")

                    if isinstance(raw_args, str):
                        try:
                            tool_args = json.loads(raw_args)
                        except json.JSONDecodeError:
                            tool_args = {}
                    else:
                        tool_args = raw_args

                    yield AgentEvent(
                        type="tool_call",
                        tool_name=tool_name,
                        tool_args=tool_args,
                        step=self._step,
                        elapsed_ms=elapsed,
                    )

                    # Execute tool
                    tool = self.tools.get(tool_name)
                    if tool is None:
                        result = ToolResult(
                            output="",
                            success=False,
                            error=f"Unknown tool: {tool_name}",
                        )
                    else:
                        try:
                            result = await tool.execute(**tool_args)
                        except Exception as exc:
                            result = ToolResult(output="", success=False, error=str(exc))

                    yield AgentEvent(
                        type="tool_result",
                        content=result.to_message()[:2000],
                        tool_name=tool_name,
                        step=self._step,
                    )

                    # Add tool result to messages
                    self._messages.append(
                        {
                            "role": "tool",
                            "content": result.to_message(),
                            "tool_call_id": tc.get("id", tool_name),
                        }
                    )

            else:
                # No tool calls — this is the final answer
                content = message.get("content", "")
                if content:
                    yield AgentEvent(
                        type="message",
                        content=content,
                        step=self._step,
                        elapsed_ms=elapsed,
                    )
                yield AgentEvent(type="done", step=self._step, elapsed_ms=elapsed)
                return

        # Max steps reached
        yield AgentEvent(
            type="error",
            content=f"Agent reached maximum steps ({self.config.max_steps}) without completing.",
            step=self._step,
        )

    async def _call_llm(self, tool_schemas: list[dict], is_ollama: bool) -> dict:
        """Call the LLM with current messages and tool schemas."""
        if is_ollama:
            return await self._call_ollama(tool_schemas)
        return await self._call_openai_compat(tool_schemas)

    async def _call_ollama(self, tool_schemas: list[dict]) -> dict:
        """Call Ollama's native /api/chat endpoint with tool support."""
        url = f"{self.config.api_url}/api/chat"

        body: dict = {
            "model": self.config.model,
            "messages": self._messages,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
            },
        }

        if tool_schemas:
            body["tools"] = tool_schemas

        headers = {}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        async with httpx.AsyncClient(timeout=self.config.timeout) as client:
            resp = await client.post(url, json=body, headers=headers)
            resp.raise_for_status()
            return resp.json()

    async def _call_openai_compat(self, tool_schemas: list[dict]) -> dict:
        """Call an OpenAI-compatible API with tool support."""
        url = f"{self.config.api_url}/v1/chat/completions"

        body: dict = {
            "model": self.config.model,
            "messages": self._messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }

        if tool_schemas:
            body["tools"] = tool_schemas
            body["tool_choice"] = "auto"

        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        async with httpx.AsyncClient(timeout=self.config.timeout) as client:
            resp = await client.post(url, json=body, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        # Normalize OpenAI response to Ollama-like format
        choice = data.get("choices", [{}])[0]
        return {"message": choice.get("message", {})}

    @property
    def messages(self) -> list[dict]:
        """Return the full message history."""
        return list(self._messages)

    @property
    def steps_taken(self) -> int:
        return self._step
