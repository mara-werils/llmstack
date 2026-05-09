"""CLI command: llmstack mcp — run the MCP server for AI client integration."""

from __future__ import annotations

import asyncio
import sys



def mcp_serve(
    model: str | None = None,
    ollama_url: str = "http://localhost:11434",
    working_dir: str = ".",
) -> None:
    """Start the MCP server on stdin/stdout."""
    # Log to stderr so stdout stays clean for JSON-RPC
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )

    # Print startup info to stderr
    print(f"llmstack MCP server starting (model={model or 'llama3.2'})", file=sys.stderr)

    asyncio.run(_mcp_async(
        model=model,
        ollama_url=ollama_url,
        working_dir=working_dir,
    ))


async def _mcp_async(
    model: str | None,
    ollama_url: str,
    working_dir: str,
) -> None:
    from pathlib import Path

    from llmstack.agent.tools import create_default_registry
    from llmstack.mcp.server import MCPServer

    cwd = str(Path(working_dir).resolve())
    registry = create_default_registry(cwd)

    server = MCPServer(
        tools=registry,
        ollama_url=ollama_url,
        model=model or "llama3.2",
    )

    await server.run()
