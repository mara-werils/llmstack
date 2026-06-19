"""Generate and manage API keys for the LLMStack gateway."""

from __future__ import annotations

import secrets
import string

from llmstack.cli.console import console, success, info


def apikey_generate(prefix: str = "llmsk", length: int = 48) -> None:
    """Generate a cryptographically secure API key."""
    alphabet = string.ascii_letters + string.digits
    random_part = "".join(secrets.choice(alphabet) for _ in range(length))
    key = f"{prefix}_{random_part}"

    success("Generated API key:")
    console.print(f"\n  [bold]{key}[/]\n")
    info("Add to your environment:")
    console.print(f'  export LLMSTACK_API_KEYS="{key}"')
    console.print()
    info("Or add to .env file:")
    console.print(f"  LLMSTACK_API_KEYS={key}")
    console.print()


def apikey_validate(key: str) -> None:
    """Validate an API key format.

    Exits non-zero on an empty (2) or malformed (1) key so the check can be
    scripted in CI.
    """
    if not key:
        console.print("[red]No key provided[/]")
        raise SystemExit(2)

    # Both prefixes are produced by the tool: 'llmsk_' by `apikey generate`
    # and 'sk-llmstack-' by the key `llmstack up` auto-generates.
    valid_prefixes = ("llmsk_", "sk-llmstack-")
    matched = next((p for p in valid_prefixes if key.startswith(p)), None)
    if matched and len(key) >= 20:
        success(f"Key format is valid (prefix: {matched}, length: {len(key)})")
    else:
        console.print(
            "[yellow]Warning: Key does not follow a recommended format "
            "(llmsk_... or sk-llmstack-...)[/]"
        )
        info("Generate a new key with: llmstack apikey generate")
        raise SystemExit(1)
        raise SystemExit(1)
