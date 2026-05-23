"""llmstack cost — view cost and usage summary from the gateway."""

from __future__ import annotations

import sys

import httpx
from rich.table import Table

from llmstack.cli.console import console, banner, failure


def cost(gateway_url: str | None = None) -> None:
    """Show cost and usage summary from the running gateway."""
    url = gateway_url
    if not url:
        try:
            from llmstack.config.loader import load_config
            config = load_config()
            url = f"http://localhost:{config.gateway.port}"
        except (FileNotFoundError, SystemExit):
            url = "http://localhost:8000"

    try:
        resp = httpx.get(f"{url}/healthz", timeout=5)
        data = resp.json()
    except (httpx.ConnectError, httpx.TimeoutException):
        failure("Cannot connect to gateway. Run 'llmstack up' first.")
        sys.exit(1)

    banner("Cost & Usage Summary")

    router_data = data.get("router", {})
    if router_data:
        total_cost = router_data.get("total_cost_usd", 0.0)
        total_requests = router_data.get("total_requests", 0)
        savings_pct = router_data.get("estimated_savings_pct", 0.0)
        tiers = router_data.get("tier_distribution", {})
        cost_by_provider = router_data.get("cost_by_provider_usd", {})

        console.print("\n[accent]Overview[/]")
        console.print(f"  Total requests    [bold]{total_requests:,}[/]")
        console.print(f"  Total cost        [cost]${total_cost:.4f}[/]")
        console.print(f"  Savings           [speed]{savings_pct:.1f}%[/] vs always using largest model")

        if tiers:
            console.print("\n[accent]Tier Distribution[/]")
            table = Table(show_header=True, show_edge=False, pad_edge=False)
            table.add_column("Tier", style="bold")
            table.add_column("Requests", justify="right")
            table.add_column("Percentage", justify="right")

            total = sum(tiers.values()) or 1
            for tier_name in ["simple", "medium", "complex"]:
                count = tiers.get(tier_name, 0)
                pct = count / total * 100
                style = {"simple": "green", "medium": "yellow", "complex": "magenta"}.get(tier_name, "")
                table.add_row(f"[{style}]{tier_name}[/]", str(count), f"{pct:.1f}%")

            console.print(table)

        if cost_by_provider:
            console.print("\n[accent]Cost by Provider[/]")
            table = Table(show_header=True, show_edge=False, pad_edge=False)
            table.add_column("Provider", style="bold")
            table.add_column("Cost", justify="right", style="cost")

            for provider, prov_cost in sorted(cost_by_provider.items(), key=lambda x: -x[1]):
                table.add_row(provider, f"${prov_cost:.4f}")

            console.print(table)
    else:
        console.print("\n[muted]No routing data available yet. Send some requests first.[/]")

    # Cache stats
    cache_data = data.get("cache", {})
    if cache_data:
        hits = cache_data.get("hits", 0)
        misses = cache_data.get("misses", 0)
        total = hits + misses
        hit_rate = hits / total * 100 if total > 0 else 0

        console.print("\n[accent]Cache Performance[/]")
        console.print(f"  Hit rate      [bold]{hit_rate:.1f}%[/]")
        console.print(f"  Hits          [success]{hits}[/]")
        console.print(f"  Misses        {misses}")

    console.print()
