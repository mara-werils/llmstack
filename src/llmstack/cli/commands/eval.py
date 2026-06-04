"""CLI command: llmstack eval — evaluate model quality against a test suite."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from llmstack.cli.console import console


def eval_cmd(
    data: str | None = None,
    model: str = "llama3.2",
    ollama_url: str = "http://localhost:11434",
    gateway_url: str | None = None,
    max_examples: int = 20,
    output: str | None = None,
) -> None:
    """Evaluate a model's quality using a test dataset or the live gateway."""
    if gateway_url:
        _eval_gateway(gateway_url)
    elif data:
        asyncio.run(
            _eval_dataset(
                data_path=data,
                model=model,
                ollama_url=ollama_url,
                max_examples=max_examples,
                output=output,
            )
        )
    else:
        console.print("[error]Provide --data or --gateway-url[/]")
        sys.exit(1)


def _eval_gateway(gateway_url: str) -> None:
    """Show quality summary from a running gateway."""
    import httpx
    from rich.panel import Panel
    from rich.table import Table

    url = gateway_url.rstrip("/")

    try:
        resp = httpx.get(f"{url}/v1/observe/quality", timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        console.print(f"[error]Cannot connect to gateway: {exc}[/]")
        sys.exit(1)

    # Global quality
    global_q = data.get("global", {})
    if global_q:
        table = Table(title="Quality Summary", border_style="cyan")
        table.add_column("Metric", style="bold")
        table.add_column("Mean")
        table.add_column("Recent (20)")
        table.add_column("Trend")
        table.add_column("Samples")

        for metric, info in global_q.items():
            trend = info.get("trend", 0)
            trend_str = f"[green]{trend:+.4f}[/]" if trend >= 0 else f"[red]{trend:+.4f}[/]"
            table.add_row(
                metric,
                f"{info['mean']:.4f}",
                f"{info['recent']:.4f}",
                trend_str,
                str(info["count"]),
            )
        console.print(table)

    # Per-model quality
    by_model = data.get("by_model", {})
    if by_model:
        console.print()
        for model_name, metrics in by_model.items():
            table = Table(title=f"Model: {model_name}", border_style="blue", show_header=True)
            table.add_column("Metric", style="bold")
            table.add_column("Mean")
            table.add_column("Recent")
            table.add_column("Count")
            for m, info in metrics.items():
                table.add_row(m, f"{info['mean']:.4f}", f"{info['recent']:.4f}", str(info["count"]))
            console.print(table)

    # Alerts
    alerts = data.get("alerts", [])
    if alerts:
        console.print()
        console.print(
            Panel(
                "\n".join(
                    f"[{'red' if a['severity'] == 'critical' else 'yellow'}]{a['message']}[/]"
                    for a in alerts
                ),
                title="Active Alerts",
                border_style="red",
            )
        )


async def _eval_dataset(
    data_path: str,
    model: str,
    ollama_url: str,
    max_examples: int,
    output: str | None,
) -> None:
    """Evaluate model quality against a dataset."""
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn
    from rich.table import Table

    from llmstack.observe.scoring import QualityScorer

    import httpx

    # Check Ollama
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            await client.get(ollama_url)
    except Exception:
        console.print(f"[error]Cannot connect to Ollama at {ollama_url}[/]")
        sys.exit(1)

    # Load data
    path = Path(data_path)
    if not path.exists():
        console.print(f"[error]File not found: {data_path}[/]")
        sys.exit(1)

    from llmstack.finetune.data import load_raw_data, _detect_columns, _row_to_chat

    rows, fmt = load_raw_data(path)
    if not rows:
        console.print("[error]No data found[/]")
        sys.exit(1)

    input_col, output_col = _detect_columns(rows[0])

    examples = []
    for row in rows[:max_examples]:
        ex = _row_to_chat(row, input_col, output_col, "")
        if ex:
            examples.append(ex)

    scorer = QualityScorer()
    results = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]Evaluating..."),
        BarColumn(),
        MofNCompleteColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("eval", total=len(examples))

        for ex in examples:
            query = ""
            expected = ""
            for m in ex.messages:
                if m["role"] == "user":
                    query = m["content"]
                elif m["role"] == "assistant":
                    expected = m["content"]

            # Generate response
            infer_msgs = [m for m in ex.messages if m["role"] != "assistant"]
            try:
                async with httpx.AsyncClient(timeout=120) as client:
                    resp = await client.post(
                        f"{ollama_url}/api/chat",
                        json={"model": model, "messages": infer_msgs, "stream": False},
                    )
                    resp.raise_for_status()
                    response_text = resp.json().get("message", {}).get("content", "")
            except Exception:
                response_text = ""

            score = scorer.score(query, response_text)
            results.append(
                {
                    "query": query[:100],
                    "expected": expected[:100],
                    "response": response_text[:100],
                    "scores": score.to_dict(),
                }
            )
            progress.advance(task)

    # Show results
    table = Table(title=f"Eval Results — {model}", border_style="green")
    table.add_column("Metric", style="bold")
    table.add_column("Mean")
    table.add_column("Min")
    table.add_column("Max")

    metrics = ["overall", "coherence", "relevance", "refusal", "repetition"]
    for m in metrics:
        vals = [r["scores"][m] for r in results]
        table.add_row(
            m,
            f"{sum(vals) / len(vals):.4f}",
            f"{min(vals):.4f}",
            f"{max(vals):.4f}",
        )
    console.print(table)

    if output:
        Path(output).write_text(json.dumps(results, indent=2))
        console.print(f"\n[success]Results saved to {output}[/]")
