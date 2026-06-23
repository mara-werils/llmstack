"""Drive a benchmark suite against any text generator and time each task.

The runner is provider-agnostic: callers pass a ``generate`` callable that maps a
prompt to a :class:`Generation`. That callable might wrap Ollama, the llmstack
gateway, or a fake in tests — the runner never touches the network itself, and
its clock is injectable, so runs are fully deterministic under test.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

from llmstack.benchmark.spec import BenchmarkSuite


@dataclass(frozen=True)
class Generation:
    """What a generator returned for a single prompt."""

    text: str
    input_tokens: int
    output_tokens: int
    ttft_s: float = 0.0  # time-to-first-token; 0 means "not reported"


GenerateFn = Callable[[str], Generation]


@dataclass(frozen=True)
class TaskResult:
    """The measured outcome of one benchmark task."""

    task_id: str
    category: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    ttft_s: float
    ok: bool
    error: str | None = None


@dataclass(frozen=True)
class RunResult:
    """All task results for one suite run against one model."""

    suite_name: str
    suite_version: str
    model: str
    results: tuple[TaskResult, ...] = field(default_factory=tuple)

    @property
    def ok_results(self) -> list[TaskResult]:
        return [r for r in self.results if r.ok]

    @property
    def failed(self) -> list[TaskResult]:
        return [r for r in self.results if not r.ok]

    @property
    def total_input_tokens(self) -> int:
        return sum(r.input_tokens for r in self.ok_results)

    @property
    def total_output_tokens(self) -> int:
        return sum(r.output_tokens for r in self.ok_results)


def run_suite(
    suite: BenchmarkSuite,
    generate: GenerateFn,
    *,
    model: str,
    clock: Callable[[], float] = time.monotonic,
    warmup: int = 0,
) -> RunResult:
    """Run every task in ``suite`` through ``generate``, timing each call.

    ``warmup`` runs the first task's prompt a few times (results discarded) to
    let a cold model settle before measurement. A task whose generation raises is
    recorded as a failure with zero tokens rather than aborting the whole run.
    """
    if warmup and len(suite):
        first = next(iter(suite))
        for _ in range(warmup):
            try:
                generate(first.prompt)
            except Exception:
                break

    results: list[TaskResult] = []
    for task in suite:
        start = clock()
        try:
            gen = generate(task.prompt)
            latency = clock() - start
            results.append(
                TaskResult(
                    task_id=task.id,
                    category=task.category,
                    input_tokens=gen.input_tokens,
                    output_tokens=gen.output_tokens,
                    latency_s=latency,
                    ttft_s=gen.ttft_s,
                    ok=True,
                )
            )
        except Exception as exc:  # noqa: BLE001 - a failed task must not abort the run
            latency = clock() - start
            results.append(
                TaskResult(
                    task_id=task.id,
                    category=task.category,
                    input_tokens=0,
                    output_tokens=0,
                    latency_s=latency,
                    ttft_s=0.0,
                    ok=False,
                    error=str(exc),
                )
            )

    return RunResult(
        suite_name=suite.name,
        suite_version=suite.version,
        model=model,
        results=tuple(results),
    )
