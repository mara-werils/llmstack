"""Find, read, and validate llmstack.yaml."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import ValidationError

from llmstack.config.schema import StackConfig

CONFIG_FILENAME = "llmstack.yaml"


def find_config(directory: Path | None = None) -> Path:
    """Locate llmstack.yaml in the given or current directory."""
    base = directory or Path.cwd()
    path = base / CONFIG_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"{CONFIG_FILENAME} not found in {base}. Run 'llmstack init' first."
        )
    return path


def load_config(directory: Path | None = None) -> StackConfig:
    """Load and validate llmstack.yaml, returning a StackConfig."""
    path = find_config(directory)
    raw = yaml.safe_load(path.read_text())
    if raw is None:
        raw = {}
    try:
        return StackConfig(**raw)
    except ValidationError as exc:
        raise SystemExit(f"Invalid {CONFIG_FILENAME}:\n{exc}") from exc


def save_config(config: StackConfig, directory: Path | None = None) -> Path:
    """Write a StackConfig to llmstack.yaml."""
    base = directory or Path.cwd()
    path = base / CONFIG_FILENAME
    data = config.model_dump(mode="json", exclude_defaults=False)
    path.write_text(yaml.dump(data, default_flow_style=False, sort_keys=False, allow_unicode=True))
    return path
