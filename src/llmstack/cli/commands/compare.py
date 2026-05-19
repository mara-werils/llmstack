"""llmstack compare — side-by-side model comparison in the terminal."""

from __future__ import annotations

import json
import time

import httpx
from rich.columns import Columns
from rich.panel import Panel

from llmstack.cli.console import console, banner


def compare(
    prompt: str,
    models: list[str],
    ollama_url: str = "http://localhost:11434",
) -> None:
    """Run the same prompt through multiple models and compare outputs."""
    if len(models) < 2:
        console.print("[error]Need at least 2 models to compare. Use --models m1,m2[/]")
        return

    banner("Model Comparison", f"{len(models)} models, same prompt")
    console.print(f"\n[accent]Prompt:[/] {prompt}\n")

    results = []

    for model_name in models:
        console.print(f"[muted]Running {model_name}...[/]")
        t0 = time.monotonic()

        try:
            resp = httpx.post(
                f"{ollama_url}/api/generate",
                json={"model": model_name, "prompt": prompt, "stream": False},
                timeout=httpx.Timeout(120, connect=10),
            )
            elapsed = time.monotonic() - t0

            if resp.status_code != 200:
                results.append({
                    "model": model_name,
                    "response": f"[error]HTTP {resp.status_code}[/]",
                    "time": elapsed,
                    "tokens": 0,
                })
                continue

            data = resp.json()
            response_text = data.get("response", "")
            eval_count = data.get("eval_count", 0)
            tokens_per_sec = eval_count / elapsed if elapsed > 0 else 0

            results.append({
                "model": model_name,
                "response": response_text,
                "time": elapsed,
                "tokens": eval_count,
                "tps": tokens_per_sec,
            })

        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            results.append({
                "model": model_name,
                "response": f"[error]{exc}[/]",
                "time": 0,
                "tokens": 0,
            })

    # Display side by side
    panels = []
    for r in results:
        tps = r.get("tps", 0)
        subtitle = f"{r['time']:.1f}s | {r['tokens']} tok"
        if tps:
            subtitle += f" | {tps:.1f} tok/s"

        panels.append(
            Panel(
                r["response"][:1000] + ("..." if len(r.get("response", "")) > 1000 else ""),
                title=f"[bold]{r['model']}[/]",
                subtitle=f"[muted]{subtitle}[/]",
                border_style="blue",
                width=60,
            )
        )

    console.print()
    console.print(Columns(panels, equal=True, expand=True))
    console.print()
