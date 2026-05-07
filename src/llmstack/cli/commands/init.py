"""llmstack init — create llmstack.yaml with smart defaults."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from llmstack.cli.console import console
from llmstack.config.loader import save_config, CONFIG_FILENAME
from llmstack.config.presets import PRESETS
from llmstack.config.schema import StackConfig
from llmstack.core.hardware import detect_hardware


def init(
    preset: Optional[str] = typer.Option(
        None, "--preset", "-p",
        help="Preset to use: chat, rag, agent",
    ),
    directory: Optional[Path] = typer.Option(
        None, "--dir", "-d",
        help="Directory to create llmstack.yaml in",
    ),
) -> None:
    """Initialize a new llmstack.yaml configuration file."""
    target = directory or Path.cwd()

    if (target / CONFIG_FILENAME).exists():
        console.print(f"[warning]{CONFIG_FILENAME} already exists. Use --dir to specify another location.[/]")
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

    # Pick config
    if preset and preset in PRESETS:
        config = PRESETS[preset].model_copy(deep=True)
        console.print(f"\n[info]Using preset:[/] {preset}")
    elif preset:
        console.print(f"[error]Unknown preset '{preset}'. Available: {', '.join(PRESETS.keys())}[/]")
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
    console.print(f"\n[success]Created {path}[/]")
    console.print("Next: edit the config if needed, then run [bold]llmstack up[/]")
