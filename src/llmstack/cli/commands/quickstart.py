"""llmstack quickstart -- zero to a working local completion in one command.

No API key. No Docker. The only prerequisite is Ollama; Docker is reserved for
the full gateway stack (``llmstack up``). The command picks a model sized to the
machine, ensures it is pulled, and proves first value with a real completion.
"""

from __future__ import annotations

from pathlib import Path

from llmstack.cli.console import banner, console, success, warn
from llmstack.core.hardware import detect_hardware
from llmstack.core.onboarding import (
    DEFAULT_OLLAMA_URL,
    ollama_install_hint,
    probe_ollama,
    recommend_embed_model,
    recommend_model,
    verify_first_value,
)


def quickstart(
    model: str | None = None,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    skip_pull: bool = False,
    verify: bool = True,
    embed_model: str | None = None,
) -> None:
    """Get from zero to a working local completion -- no API key, no Docker."""
    banner("LLMStack Quickstart", "Zero to a working local completion -- no API key, no Docker")
    console.print()

    # Step 1/4 -- detect hardware and choose a model that fits it.
    console.print("[accent]Step 1/4[/] Detecting hardware...")
    hw = detect_hardware()
    gpu = f"{hw.gpu_name} ({hw.gpu_vram_gb:.0f} GB VRAM)" if hw.has_gpu else "none (CPU inference)"
    console.print(f"  [muted]CPU {hw.cpu_cores} cores - RAM {hw.ram_gb:.0f} GB - GPU {gpu}[/]")

    if model is None:
        rec = recommend_model(hw)
        model = rec.name
        success(f"Recommended model: {rec.label} -- {rec.reason}")
        console.print(f"  [muted]~{rec.approx_download_gb:.1f} GB download - override with --model[/]")
    else:
        success(f"Using model: {model}")

    # Step 2/4 -- Ollama is the only prerequisite for local inference.
    console.print(f"\n[accent]Step 2/4[/] Checking Ollama at {ollama_url}...")
    status = probe_ollama(ollama_url)
    if not status.running:
        warn("Ollama is not reachable -- it is the only thing LLMStack needs to run locally.")
        for line in ollama_install_hint():
            console.print(f"  [highlight]{line}[/]")
        console.print("  [muted]Then re-run: llmstack quickstart[/]")
        raise SystemExit(1)
    success("Ollama is running")

    # Step 3/4 -- ensure the chat + embedding models are present (ask/RAG need both).
    embed = embed_model or recommend_embed_model(hw).name
    console.print(f"\n[accent]Step 3/4[/] Ensuring models ('{model}' + '{embed}' for ask/RAG)...")
    if skip_pull:
        console.print("  [muted]Skipping pull (--skip-pull)[/]")
    else:
        from llmstack.cli.commands.pull import pull

        for needed in (model, embed):
            if status.has_model(needed):
                success(f"Model '{needed}' already pulled")
            else:
                warn(f"Model '{needed}' not found locally, pulling...")
                pull(model=needed, ollama_url=ollama_url)

    # Step 4/4 -- prove first value with a real, private, local completion.
    console.print("\n[accent]Step 4/4[/] Proving first value (one local completion)...")
    if verify and not skip_pull:
        result = verify_first_value(model, ollama_url)
        if result.ok:
            success("Local inference works -- here is your first private completion:")
            console.print(f"  [highlight]{result.reply}[/]")
        else:
            warn(f"Could not verify a completion: {result.error or 'empty response'}")
            console.print("  [muted]Ollama is up; the model may still be loading. Try: llmstack chat[/]")
    else:
        console.print("  [muted]Skipped[/]")

    # Make sure a config exists so the rest of the CLI is ready to go.
    if not (Path.cwd() / "llmstack.yaml").exists():
        from llmstack.cli.commands.init import init

        init(preset="chat", yes=True)

    console.print()
    banner("Ready", "You are running a private, local AI -- zero keys, zero egress")
    console.print()
    console.print("  [muted]Try it now:[/]")
    console.print("  [highlight]llmstack ask 'How does auth work?' ./src/[/]   chat with your codebase")
    console.print("  [highlight]llmstack chat[/]                               interactive chat")
    console.print("  [highlight]llmstack savings[/]                            what you save vs cloud")
    console.print()
    console.print("  [muted]In your editor (VS Code / Cursor / VSCodium):[/]")
    console.print("  [highlight]code --install-extension llmstack.llmstack-vscode[/]")
    console.print("  [muted]Open VSX: https://open-vsx.org/extension/llmstack/llmstack-vscode[/]")
    console.print()
    console.print("  [muted]Optional -- the full gateway stack (routing, RAG, dashboard) uses Docker:[/]")
    console.print("  [highlight]llmstack up[/]")
    console.print()
