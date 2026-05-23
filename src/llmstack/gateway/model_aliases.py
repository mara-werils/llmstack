"""Model alias mapping for user-friendly names.

Maps human-readable model aliases to actual model identifiers,
allowing users to reference models by short names like "fast"
or "smart" instead of full model IDs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Default aliases for common models
DEFAULT_ALIASES: dict[str, str] = {
    "fast": "llama3.2:1b",
    "balanced": "llama3.2:3b",
    "smart": "llama3.1:8b",
    "best": "llama3.1:70b",
    "code": "codellama:7b",
    "tiny": "tinyllama:1.1b",
}


@dataclass
class AliasConfig:
    """Configuration for model alias mapping."""

    # User-defined aliases (merged with defaults)
    custom_aliases: dict[str, str] = field(default_factory=dict)

    # Whether to include default aliases
    include_defaults: bool = True

    # Allow partial matching (e.g., "llama3" matches "llama3.2:3b")
    partial_match: bool = False


class ModelAliasResolver:
    """Resolves model aliases to actual model identifiers."""

    def __init__(self, config: AliasConfig | None = None):
        self.config = config or AliasConfig()
        self._aliases: dict[str, str] = {}
        self._rebuild()

    def _rebuild(self) -> None:
        """Rebuild the alias map."""
        self._aliases = {}
        if self.config.include_defaults:
            self._aliases.update(DEFAULT_ALIASES)
        self._aliases.update(self.config.custom_aliases)

    def resolve(self, model: str) -> str:
        """Resolve a model name or alias to the actual model identifier.

        If the model is an alias, returns the mapped model.
        If not found, returns the original model name unchanged.
        """
        # Direct alias match
        resolved = self._aliases.get(model.lower())
        if resolved:
            logger.debug("Resolved alias '%s' → '%s'", model, resolved)
            return resolved

        return model

    def add_alias(self, alias: str, model: str) -> None:
        """Add or update an alias mapping."""
        self._aliases[alias.lower()] = model
        self.config.custom_aliases[alias.lower()] = model

    def remove_alias(self, alias: str) -> bool:
        """Remove an alias. Returns True if found and removed."""
        key = alias.lower()
        if key in self._aliases:
            del self._aliases[key]
            self.config.custom_aliases.pop(key, None)
            return True
        return False

    def list_aliases(self) -> dict[str, str]:
        """List all current aliases."""
        return dict(self._aliases)

    def get_stats(self) -> dict[str, Any]:
        """Get alias system statistics."""
        return {
            "total_aliases": len(self._aliases),
            "default_count": len(DEFAULT_ALIASES) if self.config.include_defaults else 0,
            "custom_count": len(self.config.custom_aliases),
            "aliases": dict(self._aliases),
        }
