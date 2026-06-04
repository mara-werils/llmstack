"""Generate shell completions for llmstack CLI."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from llmstack.cli.console import console, success, info


def completion(shell: str = "", install: bool = False) -> None:
    """Generate or install shell completions for bash, zsh, or fish."""
    if not shell:
        # Auto-detect shell
        import os

        shell_path = os.environ.get("SHELL", "")
        if "zsh" in shell_path:
            shell = "zsh"
        elif "fish" in shell_path:
            shell = "fish"
        else:
            shell = "bash"

    info(f"Detected shell: {shell}")

    # Generate completion script
    try:
        import os as _os

        env = {**dict(_os.environ), "_LLMSTACK_COMPLETE": f"{shell}_source"}
        result = subprocess.run(
            [sys.executable, "-m", "llmstack.cli.app"],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        script = result.stdout
        if not script:
            console.print("[red]Failed to generate completion script[/]")
            return

        if install:
            _install_completion(shell, script)
        else:
            console.print(script)
            info("Run 'llmstack completion --install' to install automatically")
    except Exception as e:
        console.print(f"[red]Error: {e}[/]")


def _install_completion(shell: str, script: str) -> None:
    """Install completion script to the appropriate location."""
    if shell == "zsh":
        comp_dir = Path.home() / ".zsh" / "completions"
        comp_dir.mkdir(parents=True, exist_ok=True)
        comp_file = comp_dir / "_llmstack"
        comp_file.write_text(script)
        success(f"Installed zsh completion to {comp_file}")
        info(
            "Add to .zshrc: fpath=(~/.zsh/completions $fpath)"
            " && autoload -Uz compinit && compinit"
        )
    elif shell == "bash":
        comp_dir = Path.home() / ".local" / "share" / "bash-completion" / "completions"
        comp_dir.mkdir(parents=True, exist_ok=True)
        comp_file = comp_dir / "llmstack"
        comp_file.write_text(script)
        success(f"Installed bash completion to {comp_file}")
    elif shell == "fish":
        comp_dir = Path.home() / ".config" / "fish" / "completions"
        comp_dir.mkdir(parents=True, exist_ok=True)
        comp_file = comp_dir / "llmstack.fish"
        comp_file.write_text(script)
        success(f"Installed fish completion to {comp_file}")
    else:
        console.print(f"[red]Unsupported shell: {shell}[/]")
