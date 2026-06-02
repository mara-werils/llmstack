"""llmstack recommend — Recommend the best model for your hardware and task."""

from __future__ import annotations

import platform
import subprocess

from llmstack.cli.console import console


# Model recommendations based on hardware and task
MODEL_CATALOG = [
    # Small models (1-3B) — good for simple tasks, low-end hardware
    {"name": "llama3.2:1b", "size_gb": 1.3, "context": 131072, "strength": "fast, lightweight",
     "tasks": ["chat", "simple-qa", "commit-msg"], "min_ram_gb": 4, "speed": "fast"},
    {"name": "qwen2.5:1.5b", "size_gb": 1.5, "context": 131072, "strength": "multilingual, fast",
     "tasks": ["chat", "translation", "simple-qa"], "min_ram_gb": 4, "speed": "fast"},
    {"name": "phi3:mini", "size_gb": 2.3, "context": 131072, "strength": "reasoning, efficient",
     "tasks": ["chat", "reasoning", "code"], "min_ram_gb": 6, "speed": "fast"},

    # Medium models (7-8B) — great for most tasks
    {"name": "llama3.2", "size_gb": 4.7, "context": 131072, "strength": "best overall balance",
     "tasks": ["chat", "code", "reasoning", "review", "explain"], "min_ram_gb": 8, "speed": "medium"},
    {"name": "gemma2:9b", "size_gb": 5.4, "context": 8192, "strength": "code, instruction following",
     "tasks": ["code", "review", "test-gen"], "min_ram_gb": 10, "speed": "medium"},
    {"name": "deepseek-coder-v2:lite", "size_gb": 9.0, "context": 131072, "strength": "best for code",
     "tasks": ["code", "review", "fix", "translate", "test-gen"], "min_ram_gb": 12, "speed": "medium"},
    {"name": "mistral", "size_gb": 4.1, "context": 32768, "strength": "reasoning, multilingual",
     "tasks": ["chat", "reasoning", "explain", "review"], "min_ram_gb": 8, "speed": "medium"},

    # Large models (13-34B) — best quality
    {"name": "llama3.1:70b", "size_gb": 40, "context": 131072, "strength": "top quality, slow",
     "tasks": ["complex-reasoning", "architecture", "security", "review"], "min_ram_gb": 48, "speed": "slow"},
    {"name": "codellama:34b", "size_gb": 19, "context": 16384, "strength": "advanced code tasks",
     "tasks": ["code", "refactor", "security", "architecture"], "min_ram_gb": 24, "speed": "slow"},
    {"name": "mixtral", "size_gb": 26, "context": 32768, "strength": "MoE, good quality/speed",
     "tasks": ["chat", "reasoning", "code", "review"], "min_ram_gb": 32, "speed": "medium"},

    # Embedding models
    {"name": "nomic-embed-text", "size_gb": 0.3, "context": 8192, "strength": "embeddings",
     "tasks": ["embeddings", "search", "rag"], "min_ram_gb": 2, "speed": "fast"},
    {"name": "bge-m3", "size_gb": 1.2, "context": 8192, "strength": "multilingual embeddings",
     "tasks": ["embeddings", "search", "rag"], "min_ram_gb": 4, "speed": "fast"},
]

TASK_DESCRIPTIONS = {
    "chat": "General conversation and Q&A",
    "code": "Code generation and completion",
    "review": "Code review and bug detection",
    "explain": "Code explanation and documentation",
    "fix": "Bug fixing and code repair",
    "refactor": "Code refactoring suggestions",
    "test-gen": "Test case generation",
    "translate": "Code translation between languages",
    "security": "Security audit and vulnerability detection",
    "commit-msg": "Commit message generation",
    "reasoning": "Complex reasoning and analysis",
    "architecture": "Architecture design and diagrams",
    "simple-qa": "Simple questions and lookups",
    "translation": "Natural language translation",
    "embeddings": "Text embeddings for search/RAG",
    "search": "Semantic code search",
    "rag": "Retrieval-augmented generation",
}


def _get_system_info() -> dict:
    """Get system hardware info."""
    info = {
        "platform": platform.system(),
        "machine": platform.machine(),
        "ram_gb": 8,  # Default
        "gpu": None,
        "gpu_vram_gb": 0,
    }

    # Get RAM
    try:
        if platform.system() == "Darwin":
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                info["ram_gb"] = int(result.stdout.strip()) / (1024**3)
        elif platform.system() == "Linux":
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        info["ram_gb"] = int(line.split()[1]) / (1024**2)
                        break
    except Exception:
        pass

    # Check GPU
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            parts = result.stdout.strip().split(",")
            info["gpu"] = parts[0].strip()
            vram_str = parts[1].strip() if len(parts) > 1 else ""
            if "MiB" in vram_str:
                info["gpu_vram_gb"] = int(vram_str.replace("MiB", "").strip()) / 1024
    except Exception:
        pass

    # Check Apple Silicon
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        info["gpu"] = "Apple Silicon (Metal)"
        info["gpu_vram_gb"] = info["ram_gb"]  # Unified memory

    return info


def recommend(
    task: str | None = None,
    show_all: bool = False,
) -> None:
    """Recommend models based on hardware and task."""
    from rich.table import Table
    from rich.panel import Panel

    sys_info = _get_system_info()
    ram = sys_info["ram_gb"]
    gpu = sys_info["gpu"]

    console.print()
    console.print("[bold]llmstack recommend[/]")
    console.print()

    # System info panel
    gpu_str = f"{gpu} ({sys_info['gpu_vram_gb']:.0f}GB)" if gpu else "None detected"
    console.print(Panel(
        f"[bold]Platform:[/] {sys_info['platform']} {sys_info['machine']}\n"
        f"[bold]RAM:[/] {ram:.0f} GB\n"
        f"[bold]GPU:[/] {gpu_str}",
        title="System Info",
        border_style="cyan",
    ))

    # Filter models by hardware
    if show_all:
        compatible = MODEL_CATALOG
    else:
        compatible = [m for m in MODEL_CATALOG if m["min_ram_gb"] <= ram]

    if task:
        task_lower = task.lower()
        # Filter by task
        task_models = [m for m in compatible if task_lower in m["tasks"]]
        if not task_models:
            console.print(f"[warning]No models found for task '{task}' with your hardware.[/]")
            task_models = compatible
    else:
        task_models = compatible

    # Display recommendations
    table = Table(
        title="Recommended Models",
        show_header=True,
        header_style="bold cyan",
        border_style="dim",
    )
    table.add_column("Model", style="bold")
    table.add_column("Size", justify="right", width=8)
    table.add_column("Context", justify="right", width=8)
    table.add_column("Speed", width=8)
    table.add_column("Best For")
    table.add_column("Min RAM", justify="right", width=8)
    table.add_column("Fit")

    speed_colors = {"fast": "green", "medium": "yellow", "slow": "red"}

    for m in task_models:
        speed_color = speed_colors.get(m["speed"], "white")
        fits = m["min_ram_gb"] <= ram
        fit_str = "[green]YES[/]" if fits else "[red]NO[/]"
        context_str = f"{m['context'] // 1024}K"

        table.add_row(
            m["name"],
            f"{m['size_gb']}GB",
            context_str,
            f"[{speed_color}]{m['speed']}[/]",
            m["strength"],
            f"{m['min_ram_gb']}GB",
            fit_str,
        )

    console.print()
    console.print(table)

    # Top recommendation
    if task_models:
        best = [m for m in task_models if m["min_ram_gb"] <= ram]
        if best:
            # Pick the largest model that fits
            best_model = max(best, key=lambda m: m["size_gb"])
            console.print()
            console.print(Panel(
                f"[bold green]Recommended: {best_model['name']}[/]\n\n"
                f"{best_model['strength']}\n"
                f"Install: [cyan]ollama pull {best_model['name']}[/]",
                title="Top Pick",
                border_style="green",
            ))

    # Show available tasks
    if not task:
        console.print()
        console.print("[dim]Filter by task: llmstack recommend --task <task>[/]")
        console.print(f"[dim]Tasks: {', '.join(sorted(TASK_DESCRIPTIONS.keys()))}[/]")
