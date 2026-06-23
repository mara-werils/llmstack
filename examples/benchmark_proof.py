"""Run the reproducible benchmark suite and prove it made zero external egress.

This is the benchmark counterpart to ``airgapped_proof.py``. It runs the llmstack
benchmark harness against a deterministic in-process generator (so it needs no
model and no network) under the runtime egress monitor, then asserts the run made
no external connection and prints the shareable report.

    python examples/benchmark_proof.py

Exit code is non-zero if any external connection was observed during the run.
"""

from __future__ import annotations

import sys

from llmstack.benchmark import Generation, get_suite, run_benchmark


def _deterministic_generator():
    def generate(prompt: str) -> Generation:
        return Generation(
            text="(reproducible mock answer)",
            input_tokens=max(1, len(prompt) // 4),
            output_tokens=max(1, len(prompt) // 8 + 16),
        )

    return generate


class _FixedClock:
    """A clock that advances a fixed 50 ms per call, so latencies are stable."""

    def __init__(self) -> None:
        self._t = 0.0

    def __call__(self) -> float:
        value = self._t
        self._t += 0.05
        return value


def main() -> int:
    suite = get_suite("default")
    report = run_benchmark(
        suite,
        _deterministic_generator(),
        model="mock-local",
        with_egress_proof=True,
        warmup=0,
        clock=_FixedClock(),
        generated_at=0.0,  # fixed so the printed report is reproducible
    )

    print(report.to_markdown())
    print(f"Methodology hash: {report.methodology_hash}")

    proof = report.egress_proof
    if proof is None:
        print("\n[FAIL] No egress proof was produced.")
        return 1
    print(f"\nObserved {proof.total_connections} connection(s) during the run.")
    if not proof.is_local_only:
        for target in proof.external_connections:
            print(f"  [EXTERNAL] {target}")
        print("\n[FAIL] External egress detected during the benchmark.")
        return 1

    print("[OK] Benchmark ran with zero external connections — provably local.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
