"""llmstack config — validate and inspect configuration."""

from __future__ import annotations

import json

import yaml
from rich.syntax import Syntax

from llmstack.cli.console import console, success, failure


def config_validate() -> None:
    """Validate llmstack.yaml and print results."""
    try:
        from llmstack.config.loader import load_config

        config = load_config()
        success("llmstack.yaml is valid")

        # Show summary
        console.print(f"  [muted]Model:     {config.models.chat.name}[/]")
        console.print(f"  [muted]Backend:   {config.models.chat.backend}[/]")
        console.print(f"  [muted]Gateway:   :{config.gateway.port}[/]")
        console.print(f"  [muted]Auth:      {config.gateway.auth}[/]")

        if config.providers.enabled:
            n = len(config.providers.providers)
            console.print(f"  [muted]Providers: {n} configured[/]")

    except FileNotFoundError as exc:
        failure(str(exc))
        raise SystemExit(1) from exc
    except SystemExit as exc:
        failure(f"Validation failed: {exc}")
        raise SystemExit(1) from exc


def config_show(output_format: str = "yaml") -> None:
    """Display the current configuration."""
    try:
        from llmstack.config.loader import load_config

        config = load_config()
    except FileNotFoundError as exc:
        failure(str(exc))
        return
    except SystemExit as exc:
        failure(f"Config error: {exc}")
        return

    data = config.model_dump(mode="json", exclude_defaults=False)

    if output_format == "json":
        text = json.dumps(data, indent=2)
        syntax = Syntax(text, "json", theme="monokai", line_numbers=True)
    else:
        text = yaml.dump(data, default_flow_style=False, sort_keys=False, allow_unicode=True)
        syntax = Syntax(text, "yaml", theme="monokai", line_numbers=True)

    console.print(syntax)


def config_path() -> None:
    """Show the path to the active llmstack.yaml."""
    try:
        from llmstack.config.loader import find_config

        path = find_config()
        console.print(f"[path]{path}[/]")
    except FileNotFoundError as exc:
        failure(str(exc))
        raise SystemExit(1) from exc
