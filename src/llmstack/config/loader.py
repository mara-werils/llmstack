"""Find, read, and validate llmstack.yaml with local override support."""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import ValidationError

from llmstack.config.schema import StackConfig

CONFIG_FILENAME = "llmstack.yaml"
LOCAL_CONFIG_FILENAME = "llmstack.local.yaml"


def find_config(directory: Path | None = None) -> Path:
    """Locate llmstack.yaml in the given or current directory."""
    base = directory or Path.cwd()
    path = base / CONFIG_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"{CONFIG_FILENAME} not found in {base}. Run 'llmstack init' first."
        )
    return path


def config_exists(directory: Path | None = None) -> bool:
    """Return True if llmstack.yaml exists in the given or current directory."""
    base = directory or Path.cwd()
    return (base / CONFIG_FILENAME).exists()


def has_local_override(directory: Path | None = None) -> bool:
    """Return True if llmstack.local.yaml exists in the given or current directory."""
    base = directory or Path.cwd()
    return (base / LOCAL_CONFIG_FILENAME).exists()


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override dict into base dict."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _apply_env_overrides(raw: dict) -> dict:
    """Apply environment variable overrides to config.

    Convention: LLMSTACK_MODELS__CHAT__NAME=llama3.1 overrides
    models.chat.name in the config.
    """
    prefix = "LLMSTACK_CFG_"
    for key, value in os.environ.items():
        if not key.startswith(prefix):
            continue
        parts = key[len(prefix) :].lower().split("__")
        target = raw
        for part in parts[:-1]:
            if part not in target:
                target[part] = {}
            target = target[part]
        # Attempt type coercion
        final_key = parts[-1]
        if value.lower() in ("true", "false"):
            target[final_key] = value.lower() == "true"
        elif value.isdigit():
            target[final_key] = int(value)
        else:
            try:
                target[final_key] = float(value)
            except ValueError:
                target[final_key] = value
    return raw


def load_config(directory: Path | None = None) -> StackConfig:
    """Load and validate llmstack.yaml, with optional local override.

    Merge order: llmstack.yaml < llmstack.local.yaml < LLMSTACK_CFG_* env vars.
    """
    path = find_config(directory)
    raw = yaml.safe_load(path.read_text()) or {}

    # Merge local overrides (not committed to git)
    base = directory or Path.cwd()
    local_path = base / LOCAL_CONFIG_FILENAME
    if local_path.exists():
        local_raw = yaml.safe_load(local_path.read_text()) or {}
        raw = _deep_merge(raw, local_raw)

    # Apply env overrides
    raw = _apply_env_overrides(raw)

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
