"""llmstack savings — show how much running locally has saved you.

Reads the local savings ledger directly (no gateway or network required), values
it against a dated cloud baseline, and prints a shareable summary. This is the
concrete, auditable form of the "actually saves you money" claim.
"""

from __future__ import annotations

import json as _json

from rich.table import Table

from llmstack.cli.console import banner, console, info, success
from llmstack.core import pricing
from llmstack.core.savings import get_ledger


def _show_pricing() -> None:
    """Print the dated, sourced pricing the savings figure is based on."""
    banner("Savings pricing catalog", f"as of {pricing.PRICING_AS_OF}")

    api = Table(title="Metered API baselines (USD per 1M tokens)", show_edge=False)
    api.add_column("Model", style="model")
    api.add_column("Vendor")
    api.add_column("Input", justify="right", style="cost")
    api.add_column("Output", justify="right", style="cost")
    for p in pricing.API_PRICING.values():
        marker = "  (baseline)" if p.model == pricing.DEFAULT_API_BASELINE else ""
        api.add_row(
            f"{p.model}{marker}",
            p.vendor,
            f"${p.input_per_million:g}",
            f"${p.output_per_million:g}",
        )
    console.print(api)

    subs = Table(title="Subscription alternatives", show_edge=False)
    subs.add_column("Plan", style="model")
    subs.add_column("Vendor")
    subs.add_column("Monthly", justify="right", style="cost")
    for s in pricing.SUBSCRIPTIONS.values():
        subs.add_row(s.name, s.vendor, f"${s.effective_monthly_usd:.2f}")
    console.print(subs)
    console.print(
        "\n  [muted]All figures are public list prices captured for comparison; "
        "see each entry's source in the API at /v1/savings/pricing.[/]"
    )


def savings(
    plan: str | None = None,
    as_json: bool = False,
    reset: bool = False,
    show_pricing: bool = False,
) -> None:
    """Show cumulative local-inference savings versus a paid alternative."""
    if show_pricing:
        _show_pricing()
        return

    ledger = get_ledger()

    if reset:
        ledger.reset()
        success("Savings ledger reset to zero.")
        return

    plan_key = plan or pricing.DEFAULT_SUBSCRIPTION_BASELINE
    try:
        summary = ledger.summary(plan_key)
    except KeyError:
        from llmstack.cli.console import failure

        failure(f"Unknown plan '{plan_key}'. Try one of: {', '.join(pricing.SUBSCRIPTIONS)}")
        return

    if as_json:
        console.print_json(_json.dumps(summary))
        return

    sub = summary["subscription"]
    saved = float(summary["total_saved_usd"])
    requests = int(summary["total_requests"])
    in_tok = int(summary["total_input_tokens"])
    out_tok = int(summary["total_output_tokens"])

    banner("llmstack savings", f"vs {sub['name']} · baseline {pricing.DEFAULT_API_BASELINE}")

    if requests == 0:
        info("No local requests recorded yet — run the gateway and start chatting.")
        info("Every locally-served request books the cloud price you didn't pay.")
        return

    console.print()
    console.print(f"  Saved so far      [cost]${saved:,.4f}[/]")
    console.print(f"  Local requests    [bold]{requests:,}[/]")
    console.print(
        f"  Tokens served     [bold]{in_tok + out_tok:,}[/] ({in_tok:,} in / {out_tok:,} out)"
    )
    months = float(sub["months_covered"])
    console.print(
        f"  That's            [speed]{months:.1f}[/] month(s) of "
        f"[model]{sub['name']}[/] (${sub['monthly_usd']:.0f}/mo)"
    )
    console.print()
    console.print(
        f"  [muted]Valued against {pricing.DEFAULT_API_BASELINE} list pricing "
        f"as of {pricing.PRICING_AS_OF}. See 'llmstack savings --json' for detail.[/]"
    )
