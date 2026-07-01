"""Tests for benchmark report assembly and rendering (llmstack.benchmark.report)."""

from __future__ import annotations

import json

from llmstack.benchmark.compare import compare
from llmstack.benchmark.environment import capture_environment
from llmstack.benchmark.privacy import EgressProof
from llmstack.benchmark.report import (
    BenchmarkReport,
    build_report,
    methodology_hash,
)
from llmstack.benchmark.runner import RunResult, TaskResult
from llmstack.core.hardware import HardwareProfile


def _env():
    hw = HardwareProfile(
        gpu_vendor="apple",
        gpu_name="Apple M3",
        gpu_vram_mb=16384,
        cpu_cores=8,
        ram_mb=16384,
        os="darwin",
        docker_runtime="default",
    )
    return capture_environment(hardware=hw)


def _run(ok=True):
    results = (
        TaskResult("a", "latency", 100, 50, 0.5, 0.05, ok=ok, error=None if ok else "boom"),
        TaskResult("b", "coding", 200, 100, 1.5, 0.0, ok=True),
    )
    return RunResult("default", "1", "llama3.2", results)


def _report(run=None, proof=None):
    run = run or _run()
    cmp = compare(run, baseline="gpt-4o", egress_proof=proof)
    return build_report(run, _env(), cmp, egress_proof=proof, pricing_as_of="2026-06")


# --------------------------------------------------------------------------- #
# methodology hash
# --------------------------------------------------------------------------- #
def test_methodology_hash_is_deterministic() -> None:
    a = methodology_hash("default", "1", ["a", "b"], "gpt-4o", "2026-06")
    b = methodology_hash("default", "1", ["a", "b"], "gpt-4o", "2026-06")
    assert a == b
    assert len(a) == 64


def test_methodology_hash_changes_with_definition() -> None:
    base = methodology_hash("default", "1", ["a", "b"], "gpt-4o", "2026-06")
    assert base != methodology_hash("default", "2", ["a", "b"], "gpt-4o", "2026-06")
    assert base != methodology_hash("default", "1", ["a", "c"], "gpt-4o", "2026-06")
    assert base != methodology_hash("default", "1", ["a", "b"], "gpt-4o-mini", "2026-06")


def test_hash_excludes_latency_two_runs_same_hash() -> None:
    # Same definition, different measured latencies -> identical methodology hash.
    fast = RunResult("default", "1", "m", (TaskResult("a", "latency", 1, 1, 0.1, 0.0, ok=True),))
    slow = RunResult("default", "1", "m", (TaskResult("a", "latency", 1, 1, 9.9, 0.0, ok=True),))
    r1 = build_report(fast, _env(), compare(fast), pricing_as_of="2026-06")
    r2 = build_report(slow, _env(), compare(slow), pricing_as_of="2026-06")
    assert r1.methodology_hash == r2.methodology_hash


# --------------------------------------------------------------------------- #
# build_report
# --------------------------------------------------------------------------- #
def test_build_report_computes_latency_and_throughput() -> None:
    report = _report()
    assert isinstance(report, BenchmarkReport)
    assert report.latency is not None
    assert report.latency.count == 2  # both tasks ok
    assert report.throughput is not None
    assert report.throughput.total_output_tokens == 150


def test_build_report_with_all_failures_has_no_metrics() -> None:
    run = RunResult(
        "default", "1", "m", (TaskResult("a", "latency", 0, 0, 0.1, 0.0, ok=False, error="x"),)
    )
    report = build_report(run, _env(), compare(run), pricing_as_of="2026-06")
    assert report.latency is None
    assert report.throughput is None
    # to_dict()/to_json() must handle None latency/throughput, not crash.
    d = report.to_dict()
    assert d["latency_ms"] is None
    assert d["throughput"] is None
    assert json.loads(report.to_json())["latency_ms"] is None


def test_ttft_only_counts_reported_values() -> None:
    report = _report()
    # Only task "a" reports a ttft (0.05s -> 50ms); "b" reports 0 (ignored).
    assert report.throughput.mean_ttft_ms == 50.0


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #
def test_to_json_is_valid_and_sorted() -> None:
    report = _report(proof=EgressProof(True, 2, ()))
    payload = json.loads(report.to_json())
    assert payload["suite"] == "default"
    assert payload["methodology_hash"] == report.methodology_hash
    assert payload["egress_proof"]["is_local_only"] is True
    assert payload["comparison"]["saved_usd"] >= 0


def test_to_markdown_contains_key_sections() -> None:
    md = _report(proof=EgressProof(True, 0, ())).to_markdown()
    assert "# llmstack benchmark" in md
    assert "## Latency" in md
    assert "## Cost vs cloud" in md
    assert "## Privacy" in md
    assert "zero external connections: yes" in md


def test_to_markdown_handles_no_successful_tasks() -> None:
    run = RunResult(
        "default", "1", "m", (TaskResult("a", "latency", 0, 0, 0.1, 0.0, ok=False, error="x"),)
    )
    md = build_report(run, _env(), compare(run), pricing_as_of="2026-06").to_markdown()
    assert "No successful tasks to measure" in md


def test_generated_at_excluded_from_hash_but_present_in_dict() -> None:
    run = _run()
    cmp = compare(run)
    r1 = build_report(run, _env(), cmp, pricing_as_of="2026-06", generated_at=111.0)
    r2 = build_report(run, _env(), cmp, pricing_as_of="2026-06", generated_at=999.0)
    assert r1.methodology_hash == r2.methodology_hash
    assert r1.to_dict()["generated_at"] == 111.0
