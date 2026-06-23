"""Assemble and render a benchmark report (JSON + Markdown).

The report carries a ``methodology_hash``: a deterministic fingerprint of *what
was measured* — suite name/version, the exact task ids, the cloud baseline, and
the pricing snapshot — but **not** the measured latencies (which vary by machine).
Two people who run the same suite version against the same baseline get the same
methodology hash, so they can confirm they benchmarked the identical definition.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from llmstack.benchmark.compare import Comparison
from llmstack.benchmark.environment import Environment
from llmstack.benchmark.metrics import (
    LatencyStats,
    ThroughputStats,
    latency_stats,
    throughput_stats,
)
from llmstack.benchmark.privacy import EgressProof
from llmstack.benchmark.runner import RunResult, TaskResult


def methodology_hash(
    suite_name: str,
    suite_version: str,
    task_ids: list[str],
    baseline_key: str,
    pricing_as_of: str,
) -> str:
    """Deterministic fingerprint of the benchmark *definition* (not its results)."""
    payload = json.dumps(
        {
            "suite": suite_name,
            "version": suite_version,
            "tasks": list(task_ids),
            "baseline": baseline_key,
            "pricing_as_of": pricing_as_of,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class BenchmarkReport:
    """A complete, renderable benchmark result."""

    suite_name: str
    suite_version: str
    model: str
    environment: Environment
    comparison: Comparison
    methodology_hash: str
    latency: LatencyStats | None = None
    throughput: ThroughputStats | None = None
    egress_proof: EgressProof | None = None
    results: tuple[TaskResult, ...] = ()
    generated_at: float | None = None  # caller-stamped; excluded from the hash

    def to_dict(self) -> dict[str, object]:
        return {
            "suite": self.suite_name,
            "suite_version": self.suite_version,
            "model": self.model,
            "methodology_hash": self.methodology_hash,
            "generated_at": self.generated_at,
            "environment": self.environment.as_dict(),
            "latency_ms": _stats_dict(self.latency),
            "throughput": _throughput_dict(self.throughput),
            "comparison": self.comparison.as_dict(),
            "egress_proof": self.egress_proof.as_dict() if self.egress_proof else None,
            "results": [
                {
                    "task_id": r.task_id,
                    "category": r.category,
                    "ok": r.ok,
                    "latency_ms": round(r.latency_s * 1000, 2),
                    "input_tokens": r.input_tokens,
                    "output_tokens": r.output_tokens,
                    "error": r.error,
                }
                for r in self.results
            ],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    def to_markdown(self) -> str:
        c = self.comparison
        lines = [
            f"# llmstack benchmark — {self.suite_name} v{self.suite_version}",
            "",
            f"- Model: `{self.model}`",
            f"- Methodology hash: `{self.methodology_hash[:16]}`",
            f"- Environment: {self.environment.summary()}",
            "",
            "## Latency",
            "",
        ]
        if self.latency is not None:
            lines += [
                "| mean | p50 | p95 | p99 | min | max |",
                "| ---: | ---: | ---: | ---: | ---: | ---: |",
                "| {:.0f} | {:.0f} | {:.0f} | {:.0f} | {:.0f} | {:.0f} |".format(
                    self.latency.mean_ms,
                    self.latency.p50_ms,
                    self.latency.p95_ms,
                    self.latency.p99_ms,
                    self.latency.min_ms,
                    self.latency.max_ms,
                ),
                "",
                "_All values in milliseconds, measured on the machine above._",
            ]
        else:
            lines.append("_No successful tasks to measure._")
        if self.throughput is not None:
            lines += [
                "",
                "## Throughput",
                "",
                f"- {self.throughput.tokens_per_second:.1f} output tokens/sec",
                f"- {self.throughput.total_output_tokens} output tokens "
                f"in {self.throughput.total_time_s:.2f}s",
            ]
        lines += [
            "",
            "## Cost vs cloud",
            "",
            f"- Baseline: {c.baseline_name}",
            f"- Cloud would have charged: **${c.cloud_cost_usd:.6f}**",
            f"- You paid locally: ${c.local_cost_usd:.6f}",
            f"- Saved: **${c.saved_usd:.6f}**",
            "",
            "## Privacy",
            "",
            f"- Cloud baseline sends prompts off-device: "
            f"{'yes' if c.cloud_sends_offdevice else 'no'}",
        ]
        if self.egress_proof is not None:
            local_only = "yes" if self.egress_proof.is_local_only else "no"
            lines.append(f"- Local run made zero external connections: {local_only}")
        return "\n".join(lines) + "\n"


def _stats_dict(stats: LatencyStats | None) -> dict[str, object] | None:
    if stats is None:
        return None
    return {
        "count": stats.count,
        "mean": round(stats.mean_ms, 2),
        "p50": round(stats.p50_ms, 2),
        "p95": round(stats.p95_ms, 2),
        "p99": round(stats.p99_ms, 2),
        "min": round(stats.min_ms, 2),
        "max": round(stats.max_ms, 2),
    }


def _throughput_dict(stats: ThroughputStats | None) -> dict[str, object] | None:
    if stats is None:
        return None
    return {
        "total_output_tokens": stats.total_output_tokens,
        "total_time_s": round(stats.total_time_s, 4),
        "tokens_per_second": round(stats.tokens_per_second, 2),
        "mean_ttft_ms": round(stats.mean_ttft_ms, 2),
    }


def build_report(
    run: RunResult,
    environment: Environment,
    comparison: Comparison,
    *,
    egress_proof: EgressProof | None = None,
    pricing_as_of: str,
    generated_at: float | None = None,
) -> BenchmarkReport:
    """Assemble a :class:`BenchmarkReport` from a run, environment, and comparison."""
    ok = run.ok_results
    latency = throughput = None
    if ok:
        latency = latency_stats([r.latency_s * 1000 for r in ok])
        ttfts = [r.ttft_s * 1000 for r in ok if r.ttft_s > 0]
        throughput = throughput_stats(
            [r.output_tokens for r in ok],
            [r.latency_s for r in ok],
            ttfts,
        )
    digest = methodology_hash(
        run.suite_name,
        run.suite_version,
        [r.task_id for r in run.results],
        comparison.baseline_key,
        pricing_as_of,
    )
    return BenchmarkReport(
        suite_name=run.suite_name,
        suite_version=run.suite_version,
        model=run.model,
        environment=environment,
        comparison=comparison,
        methodology_hash=digest,
        latency=latency,
        throughput=throughput,
        egress_proof=egress_proof,
        results=run.results,
        generated_at=generated_at,
    )
