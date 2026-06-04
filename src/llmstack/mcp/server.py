"""MCP Server — JSON-RPC over stdio implementing Model Context Protocol.

This server exposes llmstack's tools and LLM inference capabilities to
MCP clients like Claude Code, Cursor, VS Code Copilot, etc.

Protocol: JSON-RPC 2.0 over stdin/stdout, one message per line.
Spec: https://modelcontextprotocol.io/specification
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field
from typing import Any

from llmstack.agent.tools import ToolRegistry, create_default_registry

logger = logging.getLogger(__name__)

MCP_PROTOCOL_VERSION = "2024-11-05"

SERVER_INFO = {
    "name": "llmstack",
    "version": "0.6.0",
}


@dataclass
class MCPServer:
    """MCP server that communicates via stdin/stdout JSON-RPC.

    Exposes:
    - Tools: all agent tools (read_file, write_file, grep, shell, etc.)
    - Custom tools: llmstack_chat (LLM inference), llmstack_ask (file Q&A)
    """

    tools: ToolRegistry = field(default_factory=lambda: create_default_registry())
    ollama_url: str = "http://localhost:11434"
    model: str = "llama3.2"
    _running: bool = False

    async def run(self) -> None:
        """Main loop — read JSON-RPC messages from stdin, write responses to stdout."""
        self._running = True
        logger.info("MCP server starting on stdio")

        while self._running:
            try:
                line = await _read_line()
                if line is None:
                    break

                try:
                    message = json.loads(line)
                except json.JSONDecodeError:
                    _write_error(None, -32700, "Parse error")
                    continue

                await self._handle_message(message)

            except Exception as exc:
                logger.error("MCP server error: %s", exc)
                break

        logger.info("MCP server stopped")

    async def _handle_message(self, message: dict) -> None:
        """Dispatch a JSON-RPC message to the appropriate handler."""
        method = message.get("method", "")
        msg_id = message.get("id")
        params = message.get("params", {})

        handler = {
            "initialize": self._handle_initialize,
            "initialized": self._handle_initialized,
            "tools/list": self._handle_tools_list,
            "tools/call": self._handle_tools_call,
            "resources/list": self._handle_resources_list,
            "prompts/list": self._handle_prompts_list,
            "ping": self._handle_ping,
        }.get(method)

        if handler is None:
            if msg_id is not None:
                _write_error(msg_id, -32601, f"Method not found: {method}")
            return

        try:
            result = await handler(params)
            if msg_id is not None:
                _write_result(msg_id, result)
        except Exception as exc:
            if msg_id is not None:
                _write_error(msg_id, -32603, str(exc))

    async def _handle_initialize(self, params: dict) -> dict:
        """Handle initialize request — negotiate capabilities."""
        return {
            "protocolVersion": MCP_PROTOCOL_VERSION,
            "capabilities": {
                "tools": {"listChanged": False},
                "resources": {"subscribe": False, "listChanged": False},
                "prompts": {"listChanged": False},
            },
            "serverInfo": SERVER_INFO,
        }

    async def _handle_initialized(self, params: dict) -> dict:
        """Handle initialized notification."""
        return {}

    async def _handle_ping(self, params: dict) -> dict:
        return {}

    async def _handle_tools_list(self, params: dict) -> dict:
        """List all available tools in MCP format."""
        tools = []

        # Add agent tools
        for tool in self.tools.all_tools():
            schema = tool.schema()["function"]
            tools.append(
                {
                    "name": schema["name"],
                    "description": schema["description"],
                    "inputSchema": schema["parameters"],
                }
            )

        # Add LLM-specific tools
        tools.append(
            {
                "name": "llmstack_chat",
                "description": "Send a message to the local LLM and get a response. "
                "Use this for reasoning, code generation, or any LLM task.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "The message to send to the LLM",
                        },
                        "system": {"type": "string", "description": "Optional system prompt"},
                        "model": {
                            "type": "string",
                            "description": f"Model name (default: {self.model})",
                        },
                    },
                    "required": ["message"],
                },
            }
        )

        tools.append(
            {
                "name": "llmstack_ask",
                "description": "Ask a question about local files using RAG. "
                "Parses files, finds relevant context, and generates an answer with citations.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "Question to ask about the files",
                        },
                        "paths": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "File or directory paths to search",
                        },
                        "model": {
                            "type": "string",
                            "description": f"Model name (default: {self.model})",
                        },
                    },
                    "required": ["question", "paths"],
                },
            }
        )

        return {"tools": tools}

    async def _handle_tools_call(self, params: dict) -> dict:
        """Execute a tool and return the result."""
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})

        # Handle LLM-specific tools
        if tool_name == "llmstack_chat":
            return await self._tool_chat(arguments)
        if tool_name == "llmstack_ask":
            return await self._tool_ask(arguments)

        # Handle agent tools
        tool = self.tools.get(tool_name)
        if tool is None:
            return {
                "content": [{"type": "text", "text": f"Unknown tool: {tool_name}"}],
                "isError": True,
            }

        result = await tool.execute(**arguments)
        return {
            "content": [{"type": "text", "text": result.to_message()}],
            "isError": not result.success,
        }

    async def _tool_chat(self, args: dict) -> dict:
        """Handle llmstack_chat tool — send a message to the LLM."""
        import httpx

        message = args.get("message", "")
        system = args.get("system", "")
        model = args.get("model", self.model)

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": message})

        try:
            async with httpx.AsyncClient(timeout=300) as client:
                resp = await client.post(
                    f"{self.ollama_url}/api/chat",
                    json={"model": model, "messages": messages, "stream": False},
                )
                resp.raise_for_status()
                data = resp.json()
                content = data.get("message", {}).get("content", "")
                return {"content": [{"type": "text", "text": content}]}
        except Exception as exc:
            return {
                "content": [{"type": "text", "text": f"LLM error: {exc}"}],
                "isError": True,
            }

    async def _tool_ask(self, args: dict) -> dict:
        """Handle llmstack_ask tool — RAG over local files."""
        from pathlib import Path
        from llmstack.ask.engine import AskEngine

        question = args.get("question", "")
        paths = [Path(p) for p in args.get("paths", ["."])]
        model = args.get("model", self.model)

        try:
            engine = AskEngine(
                ollama_url=self.ollama_url,
                model=model,
                embed_model="nomic-embed-text",
            )
            await engine.load(paths, show_progress=False)
            result = await engine.ask_full(question)
            await engine.close()

            text = result.answer
            if result.sources:
                text += "\n\nSources:\n"
                for src in result.sources:
                    text += f"  - {src.file}:{src.lines} (relevance: {src.relevance})\n"

            return {"content": [{"type": "text", "text": text}]}
        except Exception as exc:
            return {
                "content": [{"type": "text", "text": f"Ask error: {exc}"}],
                "isError": True,
            }

    async def _handle_resources_list(self, params: dict) -> dict:
        return {"resources": []}

    async def _handle_prompts_list(self, params: dict) -> dict:
        return {"prompts": []}

    def stop(self) -> None:
        self._running = False


# ---------------------------------------------------------------------------
# Stdio I/O helpers
# ---------------------------------------------------------------------------


async def _read_line() -> str | None:
    """Read a single line from stdin (async-compatible via thread)."""
    import asyncio

    loop = asyncio.get_event_loop()
    try:
        line = await loop.run_in_executor(None, sys.stdin.readline)
        if not line:
            return None
        return line.strip()
    except (EOFError, KeyboardInterrupt):
        return None


def _write_result(msg_id: Any, result: dict) -> None:
    """Write a JSON-RPC success response to stdout."""
    response = {"jsonrpc": "2.0", "id": msg_id, "result": result}
    sys.stdout.write(json.dumps(response) + "\n")
    sys.stdout.flush()


def _write_error(msg_id: Any, code: int, message: str) -> None:
    """Write a JSON-RPC error response to stdout."""
    response = {
        "jsonrpc": "2.0",
        "id": msg_id,
        "error": {"code": code, "message": message},
    }
    sys.stdout.write(json.dumps(response) + "\n")
    sys.stdout.flush()
