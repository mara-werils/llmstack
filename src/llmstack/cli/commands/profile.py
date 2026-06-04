"""llmstack profile — quick model performance profiling."""

from __future__ import annotations

import time

import httpx
from rich.table import Table

from llmstack.cli.console import console, banner


_TEST_PROMPTS = [
    "Hello, how are you?",
    "Explain quicksort in 3 sentences.",
    "Write a Python function that checks if a string is a palindrome.",
    "What are the pros and cons of microservices architecture?",
]


def profile(
    model: str = "llama3.2",
    ollama_url: str = "http://localhost:11434",
    runs: int = 4,
) -> None:
    """Profile model performance with test prompts."""
    banner("Model Profile", f"Testing {model} with {runs} prompts")

    results = []
    total_tokens = 0
    total_time = 0.0

    for i, prompt in enumerate(_TEST_PROMPTS[:runs]):
        console.print(f"  [muted]Run {i + 1}/{runs}: {prompt[:50]}...[/]")

        t0 = time.monotonic()
        try:
            resp = httpx.post(
                f"{ollama_url}/api/generate",
                json={"model": model, "prompt": prompt, "stream": False},
                timeout=httpx.Timeout(120, connect=10),
            )
            elapsed = time.monotonic() - t0

            if resp.status_code != 200:
                results.append(
                    {"prompt": prompt[:40], "time": elapsed, "tokens": 0, "tps": 0, "error": True}
                )
                continue

            data = resp.json()
            eval_count = data.get("eval_count", 0)
            tps = eval_count / elapsed if elapsed > 0 else 0
            total_tokens += eval_count
            total_time += elapsed

            results.append(
                {
                    "prompt": prompt[:40],
                    "time": elapsed,
                    "tokens": eval_count,
                    "tps": tps,
                    "error": False,
                }
            )

        except Exception:
            results.append({"prompt": prompt[:40], "time": 0, "tokens": 0, "tps": 0, "error": True})

    # Results table
    console.print()
    table = Table(title=f"Profile: {model}", show_header=True)
    table.add_column("Prompt", style="muted")
    table.add_column("Time", justify="right")
    table.add_column("Tokens", justify="right")
    table.add_column("Tok/s", justify="right", style="speed")

    for r in results:
        if r["error"]:
            table.add_row(r["prompt"], "[red]error[/]", "-", "-")
        else:
            table.add_row(
                r["prompt"],
                f"{r['time']:.2f}s",
                str(r["tokens"]),
                f"{r['tps']:.1f}",
            )

    console.print(table)

    # Summary
    if total_time > 0:
        avg_tps = total_tokens / total_time
        console.print(
            f"\n  [accent]Average:[/] [speed]{avg_tps:.1f} tok/s[/] | {total_tokens} total tokens in {total_time:.1f}s"
        )
    console.print()
