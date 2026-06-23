"""Value local LLM usage in dollars — the engine behind `llmstack savings`.

This shows, end to end, how llmstack turns "it runs locally" into a concrete
"dollars saved" figure:

1. :class:`SavingsCalculator` values a request's tokens against a dated cloud
   baseline (the avoided cloud cost *is* the saving).
2. :class:`SavingsLedger` accumulates those savings across requests and persists
   them, so the total survives restarts.

It uses an isolated temp ledger so running it never touches your real
``~/.llmstack/savings.json``.

    python examples/savings_demo.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from llmstack.core.pricing import baseline_subscription
from llmstack.core.savings import SavingsCalculator, SavingsLedger


def main() -> int:
    calc = SavingsCalculator()  # default conservative baseline: gpt-4o-mini
    print(f"Baseline: {calc.baseline.model} ({calc.baseline.vendor})")

    # A handful of representative local requests (input_tokens, output_tokens).
    requests = [(1200, 400), (800, 600), (3000, 1500), (450, 120)]

    with tempfile.TemporaryDirectory() as tmp:
        ledger = SavingsLedger(path=Path(tmp) / "savings.json")
        for i, (in_tok, out_tok) in enumerate(requests, start=1):
            est = calc.estimate(in_tok, out_tok)
            ledger.record(est, timestamp=float(i))
            print(f"  request {i}: {in_tok:>5} in / {out_tok:>4} out -> saved ${est.saved_usd:.6f}")

        summary = ledger.summary()
        sub = summary["subscription"]
        print(
            f"\nTotal saved over {summary['total_requests']} requests: "
            f"${summary['total_saved_usd']:.6f}"
        )
        print(f"That covers {sub['months_covered']:.3f} month(s) of {sub['name']}.")

        # The same math drives `llmstack savings` and GET /v1/savings/summary.
        cursor = baseline_subscription("cursor-pro")
        months_cursor = float(summary["total_saved_usd"]) / cursor.effective_monthly_usd
        print(f"(or {months_cursor:.3f} month(s) of {cursor.name})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
