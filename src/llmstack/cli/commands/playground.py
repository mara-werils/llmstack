"""llmstack playground — open the gateway Web UI in a browser."""

from __future__ import annotations

import sys
import webbrowser

import httpx

from llmstack.cli.console import console


def playground(gateway_url: str | None = None) -> None:
    """Open the LLMStack Web UI in the default browser."""
    url = gateway_url
    if not url:
        try:
            from llmstack.config.loader import load_config
            config = load_config()
            url = f"http://localhost:{config.gateway.port}"
        except (FileNotFoundError, SystemExit):
            url = "http://localhost:8000"

    # Check if gateway is reachable
    try:
        resp = httpx.get(f"{url}/healthz", timeout=5)
        if resp.status_code == 200:
            console.print(f"[success]\u2713[/] Gateway is running at [path]{url}[/]")
        else:
            console.print(f"[warning]Gateway returned status {resp.status_code}[/]")
    except (httpx.ConnectError, httpx.TimeoutException):
        console.print("[error]Gateway is not reachable. Run 'llmstack up' first.[/]")
        sys.exit(1)

    console.print(f"[info]Opening [path]{url}[/] in browser...[/]")
    webbrowser.open(url)
