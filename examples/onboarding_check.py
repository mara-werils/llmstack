"""Check first-run readiness for zero-key local inference -- offline, no Docker.

This is the engine behind `llmstack quickstart`/`doctor` and the gateway's
`/v1/onboarding` route. It:

1. detects your hardware and recommends a chat + embedding model sized to it,
2. probes whether Ollama is reachable and which models are installed,
3. prints a readiness verdict and the concrete next steps to fix any gap.

It performs no external network calls (only a local Ollama probe), so it is safe
to run anywhere.

    python examples/onboarding_check.py
"""

from __future__ import annotations

from llmstack.core.hardware import detect_hardware
from llmstack.core.onboarding import (
    assess_readiness,
    probe_ollama,
    recommend_embed_model,
    recommend_model,
)


def main() -> int:
    hw = detect_hardware()
    print(f"Hardware: {hw.cpu_cores} CPU cores, {hw.ram_gb:.0f} GB RAM, GPU={hw.gpu_vendor}")

    chat = recommend_model(hw)
    embed = recommend_embed_model(hw)
    print(f"Recommended chat model:  {chat.name} -- {chat.reason}")
    print(f"Recommended embed model: {embed.name} -- {embed.reason}")

    status = probe_ollama()
    report = assess_readiness(hw, status)

    print(f"\nOllama running: {status.running}")
    print(f"Ready for zero-key local inference: {report.ready}")
    if not report.ready:
        print("Next steps:")
        for hint in report.hints:
            print(f"  - {hint}")

    # Exit non-zero when not ready so this doubles as a CI gate.
    return 0 if report.ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
