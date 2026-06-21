"""Prove an LLMStack workload is air-gapped — no data leaves the machine.

This example uses two layers of LLMStack's privacy tooling:

1. ``audit_privacy`` — a static check of your configuration for external egress.
2. ``monitor_egress`` — a runtime monitor that records every outbound socket
   connection a block of code makes and flags any that leave the local network.

Run it in CI to *fail the build* if anything ever phones home.

    python examples/airgapped_proof.py

Exit code is non-zero if the configuration is not private or if any external
connection was observed.
"""

from __future__ import annotations

import sys

from llmstack.config.schema import StackConfig
from llmstack.core.egress import assert_local_only, monitor_egress
from llmstack.core.privacy import audit_privacy


def main() -> int:
    config = StackConfig()

    # --- Layer 1: static configuration audit --------------------------------
    report = audit_privacy(config)
    print(f"Static audit verdict: {report.verdict}")
    for finding in report.critical:
        print(f"  ✖ {finding.category}: {finding.detail}")

    # --- Layer 2: runtime egress monitor ------------------------------------
    # Wrap whatever local workload you want to prove is offline. Here we build
    # the config and run the audit; swap in your own `ask`/`chat` calls against
    # a local gateway and the monitor will record exactly what they contact.
    with monitor_egress() as mon:
        audit_privacy(StackConfig())

    print(f"Observed {len(mon.connections)} connection(s):")
    for conn in mon.connections:
        scope = "local" if conn.is_local else "EXTERNAL"
        print(f"  [{scope}] {conn}")

    try:
        assert_local_only(mon)
    except Exception as exc:  # ExternalEgressError
        print(f"\n❌ {exc}")
        return 1

    if not report.is_private:
        print("\n❌ Configuration is not private (see critical findings above).")
        return 1

    print("\n✅ Air-gapped: configuration is private and no external egress observed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
