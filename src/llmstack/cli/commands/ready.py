"""llmstack ready -- fast first-run readiness check for scripts and CI.

Exits non-zero when the machine isn't ready for zero-key local inference, so it
can gate a CI job or a shell script (``llmstack ready && llmstack up``). Use
``--json`` for machine-readable output.
"""

from __future__ import annotations

from llmstack.cli.console import console, success, warn
from llmstack.core.hardware import detect_hardware
from llmstack.core.onboarding import DEFAULT_OLLAMA_URL, assess_readiness, probe_ollama


def ready(ollama_url: str = DEFAULT_OLLAMA_URL, as_json: bool = False) -> None:
    """Report readiness and exit non-zero when not ready."""
    report = assess_readiness(detect_hardware(), probe_ollama(ollama_url))

    if as_json:
        console.print_json(
            data={
                "ready": report.ready,
                "ollama_running": report.ollama_running,
                "chat_model": report.chat_model,
                "chat_model_ready": report.chat_model_ready,
                "embed_model": report.embed_model,
                "embed_model_ready": report.embed_model_ready,
                "hints": list(report.hints),
            }
        )
    elif report.ready:
        success(report.summary())
    else:
        warn("Not ready for zero-key local inference")
        for hint in report.hints:
            console.print(f"  [muted]{hint}[/]")

    if not report.ready:
        raise SystemExit(1)
