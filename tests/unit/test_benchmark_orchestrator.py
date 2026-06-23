"""Tests for the benchmark orchestrator (llmstack.benchmark.run_benchmark)."""

from __future__ import annotations


from llmstack.benchmark import BenchmarkReport, Generation, run_benchmark
from llmstack.benchmark.environment import capture_environment
from llmstack.benchmark.spec import BenchmarkSuite, BenchmarkTask
from llmstack.core.hardware import HardwareProfile


def _env():
    hw = HardwareProfile("none", None, 0, 4, 8192, "linux", "default")
    return capture_environment(hardware=hw)


def _suite():
    return BenchmarkSuite(
        "default",
        "1",
        (
            BenchmarkTask("a", "latency", "p1"),
            BenchmarkTask("b", "coding", "p2"),
        ),
    )


def _generate(prompt: str) -> Generation:
    return Generation(text="ok", input_tokens=100, output_tokens=50, ttft_s=0.02)


def test_end_to_end_report_is_complete() -> None:
    report = run_benchmark(
        _suite(),
        _generate,
        model="llama3.2",
        baseline="gpt-4o",
        environment=_env(),
        generated_at=123.0,
    )
    assert isinstance(report, BenchmarkReport)
    assert report.model == "llama3.2"
    assert report.latency is not None
    assert report.comparison.saved_usd > 0
    assert report.egress_proof is not None
    # No network in the fake generator -> provably local-only.
    assert report.egress_proof.is_local_only is True
    assert report.generated_at == 123.0


def test_without_egress_proof() -> None:
    report = run_benchmark(
        _suite(), _generate, model="m", with_egress_proof=False, environment=_env()
    )
    assert report.egress_proof is None
    # Without a proof, no privacy claim is made about the local run.
    assert report.comparison.local_sends_offdevice is False


def test_default_environment_is_captured_when_omitted(monkeypatch) -> None:
    import llmstack.benchmark as bench

    monkeypatch.setattr(bench, "capture_environment", lambda: _env())
    report = run_benchmark(_suite(), _generate, model="m")
    assert report.environment.os == "linux"


def test_report_renders_markdown_and_json() -> None:
    report = run_benchmark(_suite(), _generate, model="m", environment=_env())
    assert "## Cost vs cloud" in report.to_markdown()
    assert report.to_json().startswith("{")


def test_failures_do_not_break_orchestration() -> None:
    def flaky(prompt):
        raise RuntimeError("backend down")

    report = run_benchmark(_suite(), flaky, model="m", environment=_env())
    assert report.latency is None  # no successful tasks
    assert report.comparison.cloud_cost_usd == 0.0
    assert report.egress_proof.is_local_only is True


def test_pricing_as_of_flows_into_hash() -> None:
    from llmstack.core import pricing

    report = run_benchmark(_suite(), _generate, model="m", environment=_env())
    # The methodology hash must match recomputation with the live pricing month.
    from llmstack.benchmark.report import methodology_hash

    expected = methodology_hash(
        "default", "1", ["a", "b"], report.comparison.baseline_key, pricing.PRICING_AS_OF
    )
    assert report.methodology_hash == expected
