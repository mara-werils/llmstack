"""llmstack info — show detailed system and project information."""

from __future__ import annotations

import platform
import sys

from llmstack import __version__
from llmstack.cli.console import console, banner


def info() -> None:
    """Display detailed system, hardware, and project information."""
    banner("LLMStack Info", f"v{__version__}")

    console.print("\n[accent]System[/]")
    console.print(f"  Python      {sys.version.split()[0]}")
    console.print(f"  Platform    {platform.platform()}")
    console.print(f"  Machine     {platform.machine()}")

    try:
        from llmstack.core.hardware import detect_hardware

        hw = detect_hardware()
        console.print("\n[accent]Hardware[/]")
        console.print(f"  CPU cores   {hw.cpu_cores}")
        console.print(f"  RAM         {hw.ram_mb // 1024} GB")
        if hw.gpu_vendor != "none":
            console.print(f"  GPU         {hw.gpu_name}")
            console.print(f"  VRAM        {hw.gpu_vram_mb} MB")
            console.print(f"  Runtime     {hw.docker_runtime}")
        else:
            console.print("  GPU         [muted]not detected[/]")

        from llmstack.core.onboarding import recommend_embed_model, recommend_model

        rec = recommend_model(hw)
        rec_embed = recommend_embed_model(hw)
        console.print(f"  Recommended {rec.name} (chat) + {rec_embed.name} (embeddings)")
    except Exception:
        console.print("\n[accent]Hardware[/]")
        console.print("  [muted]detection unavailable[/]")

    # Config info
    console.print("\n[accent]Configuration[/]")
    try:
        from llmstack.config.loader import load_config

        config = load_config()
        console.print(f"  Chat model      {config.models.chat.name}")
        console.print(f"  Embed model     {config.models.embeddings.name}")
        console.print(f"  Backend         {config.models.chat.backend}")
        console.print(f"  Gateway port    {config.gateway.port}")
        console.print(f"  Auth            {config.gateway.auth}")
        console.print(f"  Rate limit      {config.gateway.rate_limit}")
        if config.providers.enabled:
            names = [p.name for p in config.providers.providers]
            console.print(f"  Providers       {', '.join(names)}")
            console.print(f"  Strategy        {config.providers.strategy}")
    except FileNotFoundError:
        console.print("  [muted]No llmstack.yaml found (run 'llmstack init')[/]")
    except SystemExit:
        console.print("  [error]llmstack.yaml has validation errors[/]")

    # Dependency versions
    console.print("\n[accent]Dependencies[/]")
    _show_dep("typer")
    _show_dep("rich")
    _show_dep("httpx")
    _show_dep("pydantic")
    _show_dep("fastapi")
    _show_dep("docker")
    _show_dep("numpy")

    console.print()


def _show_dep(name: str) -> None:
    """Show dependency version if installed."""
    try:
        from importlib.metadata import version

        ver = version(name)
        console.print(f"  {name:<14}{ver}")
    except Exception:
        console.print(f"  {name:<14}[muted]not installed[/]")
