"""llmstack init — create llmstack.yaml with smart defaults or an interactive wizard."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import typer

from llmstack.cli.console import console
from llmstack.config.loader import save_config, CONFIG_FILENAME
from llmstack.config.presets import PRESETS
from llmstack.config.schema import StackConfig
from llmstack.core.hardware import detect_hardware

# Use cases offered by the wizard, mapped to a preset.
_USE_CASES: list[tuple[str, str]] = [
    ("chat", "Q&A and general chat with a local model"),
    ("rag", "Chat grounded in your own documents"),
    ("agent", "Autonomous, tool-using agent"),
]


def recommend_models(hw) -> list[tuple[str, str]]:
    """Return ``(model, note)`` choices for the detected hardware, best first.

    Ordered highest-quality-affordable to lightest, so the first entry is always
    the recommended default. A 1B model is always offered as a guaranteed fallback.
    """
    ram_gb = hw.ram_mb // 1024
    vram_mb = hw.gpu_vram_mb
    models: list[tuple[str, str]] = []

    if vram_mb >= 48_000 or ram_gb >= 64:
        models.append(("llama3.1:70b", "70B — highest quality (needs lots of memory)"))
    if vram_mb >= 16_000 or ram_gb >= 32:
        models.append(("deepseek-coder:33b", "33B — strong on code"))
    if vram_mb >= 8_000 or ram_gb >= 16:
        models.append(("llama3.2", "8B — best balance of quality and speed"))
    if ram_gb >= 8:
        models.append(("llama3.2:3b", "3B — fast, good for simple tasks"))
    models.append(("llama3.2:1b", "1B — fastest, great for smart routing"))

    # De-duplicate while preserving order (overlapping thresholds can't add dupes
    # today, but keep this robust to future tuning).
    seen: set[str] = set()
    return [m for m in models if not (m[0] in seen or seen.add(m[0]))]


def _run_wizard(hw) -> StackConfig:
    """Walk the user through use case, model, and privacy — return a ready config."""
    from rich.prompt import Confirm, IntPrompt

    console.print("\n[bold]Let's set up llmstack[/] [dim](press Enter to accept each default)[/]")

    # 1. Use case → preset
    console.print("\n[accent]What will you use it for?[/]")
    for i, (name, desc) in enumerate(_USE_CASES, start=1):
        console.print(f"  [bold]{i}[/]. {name:<6} [dim]{desc}[/]")
    uc_idx = IntPrompt.ask(
        "Use case",
        choices=[str(i) for i in range(1, len(_USE_CASES) + 1)],
        default=1,
        console=console,
    )
    preset_name = _USE_CASES[uc_idx - 1][0]
    config = PRESETS[preset_name].model_copy(deep=True)

    # 2. Model → tuned to detected hardware
    models = recommend_models(hw)
    console.print("\n[accent]Which model? [dim](tuned to your hardware)[/]")
    for i, (name, note) in enumerate(models, start=1):
        marker = "  [green](recommended)[/]" if i == 1 else ""
        console.print(f"  [bold]{i}[/]. {name:<20} [dim]{note}[/]{marker}")
    m_idx = IntPrompt.ask(
        "Model",
        choices=[str(i) for i in range(1, len(models) + 1)],
        default=1,
        console=console,
    )
    config.models.chat.name = models[m_idx - 1][0]

    # 3. Privacy
    console.print()
    if Confirm.ask(
        "Enable privacy guardrails (PII + prompt-injection detection)?",
        default=True,
        console=console,
    ):
        config.gateway.guardrails.enabled = True

    return config


def init(
    preset: Optional[str] = typer.Option(
        None,
        "--preset",
        "-p",
        help="Preset to use: chat, rag, agent",
    ),
    directory: Optional[Path] = typer.Option(
        None,
        "--dir",
        "-d",
        help="Directory to create llmstack.yaml in",
    ),
    yes: bool = False,
) -> None:
    """Initialize a new llmstack.yaml configuration file.

    With no ``--preset`` on an interactive terminal, runs a short setup wizard.
    Pass ``--preset`` (or ``--yes``) to skip the wizard and use defaults.
    """
    target = directory or Path.cwd()

    if (target / CONFIG_FILENAME).exists():
        console.print(
            f"[warning]{CONFIG_FILENAME} already exists. Use --dir to specify another location.[/]"
        )
        raise typer.Exit(1)

    # Detect hardware
    hw = detect_hardware()
    console.print("\n[info]Hardware detected:[/]")
    console.print(f"  CPU: {hw.cpu_cores} cores")
    console.print(f"  RAM: {hw.ram_mb // 1024} GB")
    if hw.gpu_vendor != "none":
        console.print(f"  GPU: {hw.gpu_name} ({hw.gpu_vram_mb // 1024} GB VRAM)")
    else:
        console.print("  GPU: none (will use CPU inference)")

    # Pick config: wizard (interactive, no preset) → preset → plain default.
    interactive = preset is None and not yes and sys.stdin.isatty()
    if interactive:
        config = _run_wizard(hw)
    elif preset and preset in PRESETS:
        config = PRESETS[preset].model_copy(deep=True)
        console.print(f"\n[info]Using preset:[/] {preset}")
    elif preset:
        console.print(
            f"[error]Unknown preset '{preset}'. Available: {', '.join(PRESETS.keys())}[/]"
        )
        raise typer.Exit(1)
    else:
        config = StackConfig()
        console.print("\n[info]Using default configuration[/]")

    # Auto-resolve backend hint
    if hw.gpu_vendor == "nvidia" and hw.gpu_vram_mb >= 16_000:
        config.models.chat.backend = "vllm"
        console.print("  Backend: [success]vLLM[/] (NVIDIA GPU detected)")
    else:
        config.models.chat.backend = "ollama"
        console.print("  Backend: [success]Ollama[/]")

    # Save
    path = save_config(config, target)
    console.print(f"\n[success]Created {path}[/]  (model: [cyan]{config.models.chat.name}[/])")
    console.print("Next: run [bold]llmstack up[/]")
