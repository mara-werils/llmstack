"""llmstack models — list available models from all sources."""

from __future__ import annotations

import httpx
from rich.table import Table

from llmstack.cli.console import console, banner


def models(
    ollama_url: str = "http://localhost:11434",
    gateway_url: str | None = None,
) -> None:
    """List all available models from Ollama and/or running gateway."""
    banner("Available Models")

    found_any = False

    # Try Ollama directly
    try:
        resp = httpx.get(f"{ollama_url}/api/tags", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            ollama_models = data.get("models", [])
            if ollama_models:
                found_any = True
                table = Table(title="Ollama Models", show_header=True)
                table.add_column("Name", style="model")
                table.add_column("Size", justify="right")
                table.add_column("Quantization")
                table.add_column("Modified")

                for m in ollama_models:
                    name = m.get("name", "unknown")
                    size_gb = m.get("size", 0) / (1024**3)
                    size_str = (
                        f"{size_gb:.1f} GB"
                        if size_gb >= 1
                        else f"{m.get('size', 0) / (1024**2):.0f} MB"
                    )
                    quant = m.get("details", {}).get("quantization_level", "-")
                    modified = m.get("modified_at", "-")
                    if isinstance(modified, str) and "T" in modified:
                        modified = modified.split("T")[0]

                    table.add_row(name, size_str, quant, modified)

                console.print(table)
    except (httpx.ConnectError, httpx.TimeoutException):
        console.print("[muted]  Ollama not reachable at {ollama_url}[/]")

    # Try gateway if available
    gw_url = gateway_url
    if not gw_url:
        try:
            from llmstack.config.loader import load_config

            config = load_config()
            gw_url = f"http://localhost:{config.gateway.port}"
        except (FileNotFoundError, SystemExit):
            pass

    if gw_url:
        try:
            resp = httpx.get(f"{gw_url}/v1/models", timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                gw_models = data.get("data", [])
                if gw_models:
                    found_any = True
                    table = Table(title="Gateway Models", show_header=True)
                    table.add_column("ID", style="model")
                    table.add_column("Provider")
                    table.add_column("Context", justify="right")

                    for m in gw_models:
                        mid = m.get("id", "unknown")
                        provider = m.get("owned_by", "local")
                        ctx = m.get("context_length", "-")
                        ctx_str = f"{ctx:,}" if isinstance(ctx, int) else str(ctx)

                        table.add_row(mid, provider, ctx_str)

                    console.print(table)
        except (httpx.ConnectError, httpx.TimeoutException):
            pass

    if not found_any:
        console.print("[warning]No models found. Make sure Ollama is running.[/]")
        console.print("[muted]  Install Ollama: https://ollama.com/download[/]")
        console.print("[muted]  Pull a model: ollama pull llama3.2[/]")
