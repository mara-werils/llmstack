"""llmstack benchmark — run the reproducible benchmark suite and print a report.

Drives the harness in :mod:`llmstack.benchmark` against a local Ollama model (or a
deterministic ``--mock`` generator for CI/demos), measures latency and throughput,
values the run against a dated cloud baseline, proves zero external egress, and
writes a shareable JSON + Markdown report.
"""

from __future__ import annotations

import time
from pathlib import Path

from llmstack.benchmark import Generation, get_suite, run_benchmark
from llmstack.benchmark.runner import GenerateFn
from llmstack.benchmark.spec import SUITES
from llmstack.cli.console import banner, console, failure, info, success


def _mock_generator() -> GenerateFn:
    """A deterministic generator so the suite runs without a model present."""

    def generate(prompt: str) -> Generation:
        # Token counts derive from prompt length so the report is stable and
        # plausible without ever touching the network.
        in_tokens = max(1, len(prompt) // 4)
        out_tokens = max(1, len(prompt) // 8 + 16)
        return Generation(text="(mock response)", input_tokens=in_tokens, output_tokens=out_tokens)

    return generate


def _ollama_generator(model: str, ollama_url: str) -> GenerateFn:
    """A generator backed by a local Ollama ``/api/generate`` call."""
    import httpx

    def generate(prompt: str) -> Generation:
        resp = httpx.post(
            f"{ollama_url}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        return Generation(
            text=data.get("response", ""),
            input_tokens=int(data.get("prompt_eval_count", 0)),
            output_tokens=int(data.get("eval_count", 0)),
        )

    return generate


def benchmark(
    model: str = "llama3.2",
    suite_name: str = "default",
    baseline: str | None = None,
    ollama_url: str = "http://localhost:11434",
    output: str | None = None,
    proof: bool = True,
    warmup: int = 1,
    mock: bool = False,
) -> None:
    """Run the benchmark suite and print (and optionally save) a report."""
    try:
        suite = get_suite(suite_name)
    except KeyError:
        failure(f"Unknown suite '{suite_name}'. Available: {', '.join(SUITES)}")
        return

    banner(
        "llmstack benchmark",
        f"suite {suite.name} v{suite.version} · {len(suite)} tasks · model {model}",
    )

    if mock:
        info("Running in --mock mode (deterministic, no model required).")
        generate = _mock_generator()
    else:
        generate = _ollama_generator(model, ollama_url)
        info(f"Driving local Ollama at {ollama_url} (warmup={warmup}).")

    try:
        with console.status("Running benchmark..."):
            report = run_benchmark(
                suite,
                generate,
                model=model,
                baseline=baseline,
                with_egress_proof=proof,
                warmup=warmup,
                generated_at=time.time(),
            )
    except Exception as exc:  # noqa: BLE001 - surface a friendly message, not a traceback
        failure(f"Benchmark failed: {exc}")
        info("Is the model pulled and Ollama running? Try 'llmstack benchmark --mock' first.")
        return

    if report.latency is None:
        failure("Every task failed — no measurements were produced.")
        info("Is the model pulled and Ollama running? Try 'llmstack benchmark --mock' first.")
        return

    console.print()
    console.print(report.to_markdown())

    if report.egress_proof is not None and report.egress_proof.is_local_only:
        success("Zero external connections during the run — provably local.")

    if output:
        out = Path(output)
        out.write_text(report.to_markdown())
        json_path = out.with_suffix(".json")
        json_path.write_text(report.to_json())
        success(f"Wrote {out} and {json_path}")
