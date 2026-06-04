"""llmstack serve — start the gateway API server directly (no Docker)."""

from __future__ import annotations

import sys

from llmstack.cli.console import console, banner


def serve(
    host: str = "0.0.0.0",
    port: int = 8000,
    reload: bool = False,
    workers: int = 1,
) -> None:
    """Start the LLMStack gateway API server directly without Docker."""
    try:
        import uvicorn
    except ImportError:
        console.print(
            "[error]uvicorn is required. Install with: pip install llmstack-cli[gateway][/]"
        )
        sys.exit(1)

    banner("LLMStack Gateway", f"Starting on {host}:{port}")
    console.print(f"  [muted]Workers: {workers} | Reload: {reload}[/]")
    console.print(f"  [muted]Docs: http://localhost:{port}/docs[/]")
    console.print(f"  [muted]Web UI: http://localhost:{port}/[/]")
    console.print()

    uvicorn.run(
        "llmstack.gateway.main:app",
        host=host,
        port=port,
        reload=reload,
        workers=workers,
        log_level="info",
    )
