"""Tests for the benchmark runner (llmstack.benchmark.runner)."""

from __future__ import annotations

import pytest

from llmstack.benchmark.runner import Generation, RunResult, run_suite
from llmstack.benchmark.spec import BenchmarkSuite, BenchmarkTask


class FakeClock:
    """Deterministic clock returning a preset sequence of timestamps."""

    def __init__(self, times):
        self._times = list(times)
        self._i = 0

    def __call__(self) -> float:
        value = self._times[self._i]
        self._i += 1
        return value


def _suite(*tasks) -> BenchmarkSuite:
    return BenchmarkSuite("t", "1", tuple(tasks))


def _gen(text="ok", in_t=5, out_t=3, ttft=0.0):
    def generate(prompt: str) -> Generation:
        return Generation(text=text, input_tokens=in_t, output_tokens=out_t, ttft_s=ttft)

    return generate


def test_runs_all_tasks_and_times_them() -> None:
    suite = _suite(
        BenchmarkTask("a", "latency", "p1"),
        BenchmarkTask("b", "coding", "p2"),
    )
    # start,end per task: task a 0->0.5 (0.5s), task b 1.0->3.0 (2.0s)
    clock = FakeClock([0.0, 0.5, 1.0, 3.0])
    run = run_suite(suite, _gen(), model="llama3.2", clock=clock)
    assert isinstance(run, RunResult)
    assert run.model == "llama3.2"
    assert len(run.results) == 2
    assert run.results[0].latency_s == pytest.approx(0.5)
    assert run.results[1].latency_s == pytest.approx(2.0)
    assert all(r.ok for r in run.results)


def test_token_totals_from_ok_results() -> None:
    suite = _suite(BenchmarkTask("a", "latency", "p"))
    run = run_suite(suite, _gen(in_t=10, out_t=20), model="m", clock=FakeClock([0.0, 1.0]))
    assert run.total_input_tokens == 10
    assert run.total_output_tokens == 20


def test_failed_task_is_recorded_not_raised() -> None:
    def boom(prompt):
        raise RuntimeError("backend down")

    suite = _suite(BenchmarkTask("a", "latency", "p"))
    run = run_suite(suite, boom, model="m", clock=FakeClock([0.0, 0.25]))
    assert len(run.results) == 1
    r = run.results[0]
    assert r.ok is False
    assert r.error == "backend down"
    assert r.input_tokens == 0 and r.output_tokens == 0
    assert r.latency_s == pytest.approx(0.25)
    assert run.failed == [r]
    assert run.ok_results == []


def test_mixed_success_and_failure() -> None:
    calls = {"n": 0}

    def flaky(prompt):
        calls["n"] += 1
        if calls["n"] == 2:
            raise ValueError("second fails")
        return Generation("ok", 1, 1)

    suite = _suite(
        BenchmarkTask("a", "latency", "p"),
        BenchmarkTask("b", "latency", "p"),
        BenchmarkTask("c", "latency", "p"),
    )
    run = run_suite(suite, flaky, model="m", clock=FakeClock([0, 1, 1, 2, 2, 3]))
    assert [r.ok for r in run.results] == [True, False, True]
    assert run.total_output_tokens == 2  # only the two successes


def test_warmup_invokes_generator_without_recording() -> None:
    seen = []

    def generate(prompt):
        seen.append(prompt)
        return Generation("ok", 1, 1)

    suite = _suite(BenchmarkTask("a", "latency", "warm-me"))
    # warmup=2 -> 2 untimed calls, then 1 timed call = 3 invocations, 1 result
    run = run_suite(suite, generate, model="m", clock=FakeClock([0.0, 1.0]), warmup=2)
    assert len(seen) == 3
    assert len(run.results) == 1


def test_warmup_stops_on_error() -> None:
    def boom(prompt):
        raise RuntimeError("cold")

    suite = _suite(BenchmarkTask("a", "latency", "p"))
    # warmup loop breaks on first error; timed run then also fails but is recorded
    run = run_suite(suite, boom, model="m", clock=FakeClock([0.0, 0.1]), warmup=3)
    assert run.results[0].ok is False


def test_ttft_passthrough() -> None:
    suite = _suite(BenchmarkTask("a", "latency", "p"))
    run = run_suite(suite, _gen(ttft=0.05), model="m", clock=FakeClock([0.0, 1.0]))
    assert run.results[0].ttft_s == pytest.approx(0.05)


def test_empty_suite_yields_no_results() -> None:
    run = run_suite(_suite(), _gen(), model="m", clock=FakeClock([]))
    assert run.results == ()
    assert run.ok_results == []
