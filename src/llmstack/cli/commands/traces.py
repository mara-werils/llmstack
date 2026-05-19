"""llmstack traces — view recent request traces from the gateway."""

from __future__ import annotations

import sys

import httpx
from rich.table import Table

from llmstack.cli.console import console, banner, failure


def traces(
    gateway_url: str | None = None,
    limit: int = 20,
    model_filter: str | None = None,
) -> None:
    """Show recent request traces from the gateway observability system."""
    url = gateway_url
    if not url:
        try:
            from llmstack.config.loader import load_config
            config = load_config()
            url = f"http://localhost:{config.gateway.port}"
        except (FileNotFoundError, SystemExit):
            url = "http://localhost:8000"

    try:
        resp = httpx.get(f"{url}/v1/observe/traces", params={"limit": limit}, timeout=10)
        if resp.status_code != 200:
            failure(f"Gateway returned {resp.status_code}")
            sys.exit(1)
        data = resp.json()
    except (httpx.ConnectError, httpx.TimeoutException):
        failure("Cannot connect to gateway. Run 'llmstack up' first.")
        sys.exit(1)

    traces_list = data.get("traces", [])
    if not traces_list:
        banner("Request Traces")
        console.print("\n[muted]No traces recorded yet. Send some requests first.[/]\n")
        return

    if model_filter:
        traces_list = [t for t in traces_list if model_filter.lower() in t.get("model", "").lower()]

    banner("Request Traces", f"Showing {len(traces_list)} most recent")

    table = Table(show_header=True, show_edge=False)
    table.add_column("Time", style="muted", width=8)
    table.add_column("Model", style="model")
    table.add_column("Provider")
    table.add_column("Tier")
    table.add_column("Tokens", justify="right")
    table.add_column("Latency", justify="right")
    table.add_column("Cost", justify="right", style="cost")
    table.add_column("Quality", justify="right")
    table.add_column("Cache")

    for t in traces_list:
        ts = t.get("timestamp", "")
        if "T" in str(ts):
            ts = str(ts).split("T")[1][:8]

        model = t.get("model", "-")
        provider = t.get("provider", "local")
        tier = t.get("routed_tier", "-")

        inp = t.get("input_tokens", 0)
        out = t.get("output_tokens", 0)
        tokens = f"{inp}+{out}"

        latency_ms = t.get("latency_ms", 0)
        latency_str = f"{latency_ms:.0f}ms" if latency_ms < 1000 else f"{latency_ms/1000:.1f}s"

        cost_usd = t.get("cost_usd", 0.0)
        cost_str = f"${cost_usd:.4f}" if cost_usd > 0 else "-"

        quality = t.get("quality", {})
        overall = quality.get("overall", quality.get("relevance", 0))
        q_str = f"{overall:.2f}" if overall else "-"

        cached = "HIT" if t.get("cached") else "-"
        cache_style = "green" if cached == "HIT" else "muted"

        tier_style = {"simple": "green", "medium": "yellow", "complex": "magenta"}.get(tier, "")

        table.add_row(
            ts, model, provider,
            f"[{tier_style}]{tier}[/]" if tier_style else tier,
            tokens, latency_str, cost_str, q_str,
            f"[{cache_style}]{cached}[/]",
        )

    console.print(table)
    console.print()
