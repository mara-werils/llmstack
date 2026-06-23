"""Reproducible benchmark suite for llmstack.

A provider-agnostic harness that measures local LLM latency/throughput, values
the run against dated cloud pricing, and proves zero external egress — producing
a deterministic, shareable report. See :mod:`llmstack.benchmark.spec` for the
task suite and :func:`run_benchmark` for the orchestrator.
"""

from __future__ import annotations

import time
from collections.abc import Callable

from llmstack.benchmark.baselines import CloudBaseline, get_baseline
from llmstack.benchmark.compare import Comparison, compare
from llmstack.benchmark.environment import Environment, capture_environment
from llmstack.benchmark.privacy import EgressProof, run_with_egress_proof
from llmstack.benchmark.report import BenchmarkReport, build_report
from llmstack.benchmark.runner import (
    Generation,
    GenerateFn,
    RunResult,
    TaskResult,
    run_suite,
)
from llmstack.benchmark.spec import (
    DEFAULT_SUITE,
    BenchmarkSuite,
    BenchmarkTask,
    get_suite,
)
from llmstack.core import pricing

__all__ = [
    "BenchmarkReport",
    "BenchmarkSuite",
    "BenchmarkTask",
    "CloudBaseline",
    "Comparison",
    "DEFAULT_SUITE",
    "EgressProof",
    "Environment",
    "GenerateFn",
    "Generation",
    "RunResult",
    "TaskResult",
    "capture_environment",
    "compare",
    "get_baseline",
    "get_suite",
    "run_benchmark",
    "run_suite",
]


def run_benchmark(
    suite: BenchmarkSuite,
    generate: GenerateFn,
    *,
    model: str,
    baseline: str | None = None,
    local_cost_usd: float = 0.0,
    with_egress_proof: bool = True,
    clock: Callable[[], float] = time.monotonic,
    warmup: int = 0,
    environment: Environment | None = None,
    generated_at: float | None = None,
) -> BenchmarkReport:
    """Run a full benchmark and return a complete, renderable report.

    Wires together the pieces: capture the environment, run the suite (optionally
    under the egress monitor to prove zero external connections), value the run
    against a cloud ``baseline``, and assemble the report. ``generate`` is any
    callable mapping a prompt to a :class:`Generation`; nothing here touches the
    network on its own, so the harness is deterministic when given a fake.
    """
    env = environment or capture_environment()

    proof: EgressProof | None = None
    if with_egress_proof:
        run, proof = run_with_egress_proof(
            lambda: run_suite(suite, generate, model=model, clock=clock, warmup=warmup)
        )
    else:
        run = run_suite(suite, generate, model=model, clock=clock, warmup=warmup)

    comparison: Comparison = compare(
        run, baseline=baseline, egress_proof=proof, local_cost_usd=local_cost_usd
    )
    return build_report(
        run,
        env,
        comparison,
        egress_proof=proof,
        pricing_as_of=pricing.PRICING_AS_OF,
        generated_at=generated_at,
    )
