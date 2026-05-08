"""llmstack bench — benchmark local LLM models and display comparative results."""

from __future__ import annotations

import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.text import Text

from llmstack.cli.console import console
from llmstack.config.loader import load_config

# ---------------------------------------------------------------------------
# Benchmark suites — predefined prompts grouped by category
# ---------------------------------------------------------------------------

SUITES: dict[str, list[str]] = {
    "simple": [
        "Hello!",
        "What is 2+2?",
        "Translate 'hello' to Spanish",
    ],
    "reasoning": [
        "Explain why the sky is blue in 2-3 sentences.",
        "What are the pros and cons of microservices architecture?",
    ],
    "coding": [
        "Write a Python function to check if a string is a palindrome.",
        "Explain this code:\n\ndef fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a",
    ],
    "long_context": [
        (
            "You are a senior software architect reviewing a system design document. "
            "The system is a real-time event processing pipeline that ingests clickstream "
            "data from a fleet of 500 web servers, each producing roughly 10,000 events "
            "per second. Events are serialized as JSON and published to Apache Kafka with "
            "a three-day retention window. A Flink streaming job consumes these events, "
            "enriches them with user-profile data stored in Redis, performs sessionization "
            "using a 30-minute inactivity gap, and writes the sessionized records to an "
            "Apache Iceberg table on S3. A nightly Spark batch job compacts the Iceberg "
            "table, computes aggregate metrics (page views, bounce rate, conversion "
            "funnel drop-offs), and materializes the results into a PostgreSQL analytics "
            "database. A Grafana dashboard queries PostgreSQL for near-real-time "
            "operational metrics, while a Metabase instance serves self-service analytics "
            "to the product team. The system must guarantee at-least-once delivery, "
            "tolerate single-node failures in every tier, and maintain end-to-end latency "
            "under 5 seconds from event ingestion to dashboard visibility during normal "
            "operation. Storage costs must stay below $2,000 per month.\n\n"
            "Please analyze this architecture and identify:\n"
            "1. Potential bottlenecks and single points of failure\n"
            "2. Suggestions for improving fault tolerance\n"
            "3. Cost optimization opportunities\n"
            "4. Monitoring and alerting recommendations\n"
            "5. A brief comparison of at-least-once vs exactly-once semantics for this use case"
        ),
    ],
    "creative": [
        "Write a haiku about programming.",
        "Tell me a short joke about databases.",
    ],
}

SUITE_NAMES = list(SUITES.keys())


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class PromptResult:
    """Metrics captured for a single prompt."""

    suite: str
    prompt: str
    ttft_ms: float  # time to first token in milliseconds
    total_time_s: float  # total generation wall-clock time in seconds
    input_tokens: int
    output_tokens: int
    tokens_per_second: float
    error: str | None = None


@dataclass
class SuiteResult:
    """Aggregated metrics for one benchmark suite."""

    name: str
    prompts: list[PromptResult] = field(default_factory=list)

    @property
    def avg_ttft_ms(self) -> float:
        vals = [p.ttft_ms for p in self.prompts if p.error is None]
        return statistics.mean(vals) if vals else 0.0

    @property
    def avg_tokens_per_second(self) -> float:
        vals = [p.tokens_per_second for p in self.prompts if p.error is None]
        return statistics.mean(vals) if vals else 0.0

    @property
    def avg_total_time(self) -> float:
        vals = [p.total_time_s for p in self.prompts if p.error is None]
        return statistics.mean(vals) if vals else 0.0

    @property
    def total_time(self) -> float:
        return sum(p.total_time_s for p in self.prompts)

    @property
    def errors(self) -> int:
        return sum(1 for p in self.prompts if p.error is not None)


@dataclass
class ModelResult:
    """All results for a single model."""

    model: str
    suites: list[SuiteResult] = field(default_factory=list)

    @property
    def avg_ttft_ms(self) -> float:
        vals = [s.avg_ttft_ms for s in self.suites if s.avg_ttft_ms > 0]
        return statistics.mean(vals) if vals else 0.0

    @property
    def avg_tokens_per_second(self) -> float:
        vals = [s.avg_tokens_per_second for s in self.suites if s.avg_tokens_per_second > 0]
        return statistics.mean(vals) if vals else 0.0

    @property
    def total_time(self) -> float:
        return sum(s.total_time for s in self.suites)

    @property
    def total_prompts(self) -> int:
        return sum(len(s.prompts) for s in self.suites)

    @property
    def total_errors(self) -> int:
        return sum(s.errors for s in self.suites)


# ---------------------------------------------------------------------------
# Score bar helper
# ---------------------------------------------------------------------------

MAX_SCORE_BLOCKS = 8


def _score_bar(tokens_per_second: float, max_tps: float) -> str:
    """Return a unicode bar showing relative performance (0-8 blocks)."""
    if max_tps <= 0:
        return "░" * MAX_SCORE_BLOCKS
    ratio = min(tokens_per_second / max_tps, 1.0)
    filled = round(ratio * MAX_SCORE_BLOCKS)
    return "█" * filled + "░" * (MAX_SCORE_BLOCKS - filled)


# ---------------------------------------------------------------------------
# API interaction — run a single prompt with streaming
# ---------------------------------------------------------------------------


def _run_prompt(
    base_url: str,
    headers: dict[str, str],
    model: str,
    prompt: str,
    timeout: float = 300.0,
) -> PromptResult:
    """Send a single prompt via streaming and collect timing metrics."""
    suite_label = ""  # filled later by caller
    t_start = time.perf_counter()
    ttft: float | None = None
    output_tokens = 0
    input_tokens = 0

    try:
        with httpx.stream(
            "POST",
            f"{base_url}/v1/chat/completions",
            headers=headers,
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": True,
            },
            timeout=httpx.Timeout(timeout, connect=10),
        ) as resp:
            if resp.status_code != 200:
                return PromptResult(
                    suite=suite_label,
                    prompt=prompt,
                    ttft_ms=0,
                    total_time_s=time.perf_counter() - t_start,
                    input_tokens=0,
                    output_tokens=0,
                    tokens_per_second=0,
                    error=f"HTTP {resp.status_code}",
                )

            for line in resp.iter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)

                    # Try to capture token counts from usage field
                    usage = chunk.get("usage")
                    if usage:
                        input_tokens = usage.get("prompt_tokens", input_tokens)
                        output_tokens = usage.get("completion_tokens", output_tokens)

                    delta = chunk["choices"][0].get("delta", {})
                    token = delta.get("content", "")
                    if token:
                        if ttft is None:
                            ttft = (time.perf_counter() - t_start) * 1000  # ms
                        output_tokens += 1  # approximate: count content chunks
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue

    except httpx.ConnectError:
        return PromptResult(
            suite=suite_label,
            prompt=prompt,
            ttft_ms=0,
            total_time_s=time.perf_counter() - t_start,
            input_tokens=0,
            output_tokens=0,
            tokens_per_second=0,
            error="Connection refused",
        )
    except httpx.HTTPError as exc:
        return PromptResult(
            suite=suite_label,
            prompt=prompt,
            ttft_ms=0,
            total_time_s=time.perf_counter() - t_start,
            input_tokens=0,
            output_tokens=0,
            tokens_per_second=0,
            error=str(exc),
        )

    total_time = time.perf_counter() - t_start
    if ttft is None:
        ttft = total_time * 1000

    # Estimate input tokens (~4 chars per token) if not provided by API
    if input_tokens == 0:
        input_tokens = max(1, len(prompt) // 4)

    tps = output_tokens / total_time if total_time > 0 else 0.0

    return PromptResult(
        suite=suite_label,
        prompt=prompt,
        ttft_ms=ttft,
        total_time_s=total_time,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        tokens_per_second=tps,
    )


# ---------------------------------------------------------------------------
# Run benchmarks for a single model
# ---------------------------------------------------------------------------


def _run_model_benchmark(
    base_url: str,
    headers: dict[str, str],
    model: str,
    suite_names: list[str],
    progress: Progress,
) -> ModelResult:
    """Run all selected suites against a model and return aggregated results."""
    result = ModelResult(model=model)

    total_prompts = sum(len(SUITES[s]) for s in suite_names)
    task = progress.add_task(f"[cyan]{model}[/]", total=total_prompts)

    for suite_name in suite_names:
        suite_result = SuiteResult(name=suite_name)
        for prompt_text in SUITES[suite_name]:
            pr = _run_prompt(base_url, headers, model, prompt_text)
            pr.suite = suite_name
            suite_result.prompts.append(pr)
            progress.advance(task)
        result.suites.append(suite_result)

    return result


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def _display_single_model(model_result: ModelResult) -> None:
    """Render results for a single model using Rich tables."""
    total_prompts = model_result.total_prompts
    suite_list = ", ".join(s.name for s in model_result.suites)

    header = Text.assemble(
        ("llmstack bench", "bold magenta"),
        " — ", ("Model Benchmark", "bold"),
    )
    console.print(Panel(header, expand=True, border_style="bright_magenta"))
    console.print()
    console.print(f"  Model: [bold cyan]{model_result.model}[/]")
    console.print(f"  Suites: [dim]{suite_list}[/] ({total_prompts} prompts)")
    console.print()

    # Find max tps for score bars
    max_tps = max(
        (s.avg_tokens_per_second for s in model_result.suites),
        default=1.0,
    )

    table = Table(
        show_header=True,
        header_style="bold",
        border_style="bright_magenta",
        title_style="bold",
        pad_edge=True,
        expand=True,
    )
    table.add_column("Suite", style="bold white", min_width=14)
    table.add_column("TTFT", justify="right", min_width=8)
    table.add_column("Tokens/sec", justify="right", min_width=12)
    table.add_column("Avg Time", justify="right", min_width=10)
    table.add_column("Score", min_width=10)

    for sr in model_result.suites:
        ttft_str = f"{sr.avg_ttft_ms:.0f}ms" if sr.avg_ttft_ms > 0 else "—"
        tps_str = f"{sr.avg_tokens_per_second:.1f} t/s" if sr.avg_tokens_per_second > 0 else "—"
        avg_time_str = f"{sr.avg_total_time:.1f}s" if sr.avg_total_time > 0 else "—"
        bar = _score_bar(sr.avg_tokens_per_second, max_tps)

        error_suffix = f" [red]({sr.errors} err)[/]" if sr.errors else ""
        table.add_row(
            sr.name,
            ttft_str,
            tps_str,
            avg_time_str,
            f"[green]{bar}[/]{error_suffix}",
        )

    console.print(table)
    console.print()

    # Summary
    console.print(
        Panel(
            f"  Avg TTFT:       [bold]{model_result.avg_ttft_ms:.0f}ms[/]\n"
            f"  Avg Tokens/sec: [bold]{model_result.avg_tokens_per_second:.2f} t/s[/]\n"
            f"  Total time:     [bold]{model_result.total_time:.1f}s[/]"
            + (f"\n  Errors:         [red]{model_result.total_errors}[/]" if model_result.total_errors else ""),
            title="[bold]Summary[/]",
            border_style="dim",
            expand=True,
        )
    )


def _display_comparison(results: list[ModelResult]) -> None:
    """Render a comparison table when multiple models are benchmarked."""
    header = Text.assemble(
        ("llmstack bench", "bold magenta"),
        " — ", ("Multi-Model Comparison", "bold"),
    )
    console.print(Panel(header, expand=True, border_style="bright_magenta"))
    console.print()

    max_tps = max((r.avg_tokens_per_second for r in results), default=1.0)

    table = Table(
        show_header=True,
        header_style="bold",
        border_style="bright_magenta",
        pad_edge=True,
        expand=True,
    )
    table.add_column("Model", style="bold cyan", min_width=18)
    table.add_column("TTFT", justify="right", min_width=8)
    table.add_column("Tokens/sec", justify="right", min_width=12)
    table.add_column("Total Time", justify="right", min_width=10)
    table.add_column("Overall", min_width=12)

    for mr in sorted(results, key=lambda r: r.avg_tokens_per_second, reverse=True):
        ttft_str = f"{mr.avg_ttft_ms:.0f}ms"
        tps_str = f"{mr.avg_tokens_per_second:.1f} t/s"
        time_str = f"{mr.total_time:.1f}s"
        bar = _score_bar(mr.avg_tokens_per_second, max_tps)
        table.add_row(mr.model, ttft_str, tps_str, time_str, f"[green]{bar}[/]")

    console.print(table)
    console.print()

    # Router savings estimate
    if len(results) >= 2:
        sorted_by_speed = sorted(results, key=lambda r: r.avg_tokens_per_second, reverse=True)
        fastest = sorted_by_speed[0]
        slowest = sorted_by_speed[-1]
        if slowest.avg_tokens_per_second > 0:
            speedup = fastest.avg_tokens_per_second / slowest.avg_tokens_per_second
        else:
            speedup = 0.0

        console.print(
            Panel(
                f"  If 60% of queries are simple → route to [bold cyan]{fastest.model}[/]\n"
                f"  Estimated [bold]{speedup:.1f}x[/] average speedup with smart routing",
                title="[bold]Router Savings Estimate[/]",
                border_style="dim",
                expand=True,
            )
        )

    # Per-model detail tables
    console.print()
    for mr in results:
        _display_single_model(mr)
        console.print()


# ---------------------------------------------------------------------------
# JSON export
# ---------------------------------------------------------------------------


def _results_to_dict(results: list[ModelResult]) -> list[dict[str, Any]]:
    """Convert results to a JSON-serializable structure."""
    output: list[dict[str, Any]] = []
    for mr in results:
        model_data: dict[str, Any] = {
            "model": mr.model,
            "avg_ttft_ms": round(mr.avg_ttft_ms, 2),
            "avg_tokens_per_second": round(mr.avg_tokens_per_second, 2),
            "total_time_s": round(mr.total_time, 2),
            "total_prompts": mr.total_prompts,
            "total_errors": mr.total_errors,
            "suites": [],
        }
        for sr in mr.suites:
            suite_data: dict[str, Any] = {
                "name": sr.name,
                "avg_ttft_ms": round(sr.avg_ttft_ms, 2),
                "avg_tokens_per_second": round(sr.avg_tokens_per_second, 2),
                "avg_total_time_s": round(sr.avg_total_time, 2),
                "total_time_s": round(sr.total_time, 2),
                "errors": sr.errors,
                "prompts": [],
            }
            for pr in sr.prompts:
                suite_data["prompts"].append({
                    "prompt": pr.prompt[:120],
                    "ttft_ms": round(pr.ttft_ms, 2),
                    "total_time_s": round(pr.total_time_s, 2),
                    "input_tokens": pr.input_tokens,
                    "output_tokens": pr.output_tokens,
                    "tokens_per_second": round(pr.tokens_per_second, 2),
                    "error": pr.error,
                })
            model_data["suites"].append(suite_data)
        output.append(model_data)
    return output


# ---------------------------------------------------------------------------
# Discover available models from the gateway
# ---------------------------------------------------------------------------


def _list_models(base_url: str, headers: dict[str, str]) -> list[str]:
    """Query the gateway /v1/models endpoint and return available model IDs."""
    try:
        resp = httpx.get(f"{base_url}/v1/models", headers=headers, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            models = data.get("data", [])
            return [m["id"] for m in models if "id" in m]
    except (httpx.HTTPError, KeyError, ValueError):
        pass
    return []


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def bench(
    model: str | None = None,
    suite: str = "all",
    output: str | None = None,
) -> None:
    """Run benchmarks against local LLM models and display results."""
    config = load_config()
    base_url = f"http://localhost:{config.gateway.port}"

    # Auth headers
    api_key = config.gateway.api_keys[0] if config.gateway.api_keys else ""
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    # Verify gateway is reachable
    try:
        resp = httpx.get(f"{base_url}/healthz", headers=headers, timeout=5)
        if resp.status_code != 200:
            console.print("[error]Gateway is not healthy. Run 'llmstack up' first.[/]")
            sys.exit(1)
    except httpx.ConnectError:
        console.print("[error]Cannot connect to gateway at {base_url}. Run 'llmstack up' first.[/]")
        sys.exit(1)
    except httpx.HTTPError as exc:
        console.print(f"[error]Gateway error: {exc}[/]")
        sys.exit(1)

    # Resolve models
    if model:
        models = [m.strip() for m in model.split(",")]
    else:
        # Try to discover models from gateway, fall back to config
        discovered = _list_models(base_url, headers)
        if discovered:
            models = discovered
        else:
            models = [config.models.chat.name]

    # Resolve suites
    if suite == "all":
        selected_suites = SUITE_NAMES
    else:
        selected_suites = [s.strip() for s in suite.split(",")]
        for s in selected_suites:
            if s not in SUITES:
                console.print(f"[error]Unknown suite '{s}'. Available: {', '.join(SUITE_NAMES)}[/]")
                sys.exit(1)

    total_prompts = sum(len(SUITES[s]) for s in selected_suites) * len(models)
    console.print()
    console.print(
        Panel(
            Text.assemble(
                ("llmstack bench", "bold magenta"),
                "\n",
                (f"{len(models)} model(s) × {len(selected_suites)} suite(s) × {total_prompts} prompt(s)", "dim"),
            ),
            border_style="bright_magenta",
            expand=True,
        )
    )
    console.print()

    # Run benchmarks with progress bar
    all_results: list[ModelResult] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=30),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    ) as progress:
        for m in models:
            mr = _run_model_benchmark(base_url, headers, m, selected_suites, progress)
            all_results.append(mr)

    console.print()

    # Display results
    if len(all_results) == 1:
        _display_single_model(all_results[0])
    else:
        _display_comparison(all_results)

    # Export JSON
    if output:
        export_data = _results_to_dict(all_results)
        out_path = Path(output)
        out_path.write_text(json.dumps(export_data, indent=2))
        console.print(f"\n[success]Results exported to {out_path.resolve()}[/]")
