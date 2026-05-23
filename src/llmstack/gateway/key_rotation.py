"""API key rotation support with graceful migration.

Allows rotating API keys without downtime by supporting multiple
active keys per client, with configurable grace periods for old keys.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class KeyInfo:
    """Information about an API key."""

    key_hash: str
    created_at: float = field(default_factory=time.time)
    expires_at: float | None = None
    label: str = ""
    is_primary: bool = False

    @property
    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at

    @property
    def is_active(self) -> bool:
        return not self.is_expired

    def to_dict(self) -> dict[str, Any]:
        return {
            "key_hash": self.key_hash[:8] + "...",
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "label": self.label,
            "is_primary": self.is_primary,
            "is_active": self.is_active,
        }


@dataclass
class RotationConfig:
    """Configuration for key rotation."""

    # Grace period for old keys after rotation (seconds)
    grace_period: float = 86400  # 24 hours

    # Maximum active keys per client
    max_keys_per_client: int = 3


class KeyRotationManager:
    """Manages API key lifecycle with rotation support.

    Supports multiple concurrent keys per client to enable zero-downtime
    key rotation. Old keys remain valid during a grace period.
    """

    def __init__(self, config: RotationConfig | None = None):
        self.config = config or RotationConfig()
        self._keys: dict[str, list[KeyInfo]] = {}  # client_id -> keys

    @staticmethod
    def hash_key(api_key: str) -> str:
        """Hash an API key for storage."""
        return hashlib.sha256(api_key.encode()).hexdigest()

    def register_key(
        self,
        client_id: str,
        api_key: str,
        label: str = "",
        is_primary: bool = True,
    ) -> KeyInfo:
        """Register a new API key for a client."""
        key_hash = self.hash_key(api_key)
        key_info = KeyInfo(
            key_hash=key_hash,
            label=label,
            is_primary=is_primary,
        )

        if client_id not in self._keys:
            self._keys[client_id] = []

        # If primary, demote existing primaries
        if is_primary:
            for existing in self._keys[client_id]:
                existing.is_primary = False

        self._keys[client_id].append(key_info)

        # Enforce max keys
        self._enforce_limits(client_id)

        return key_info

    def validate_key(self, api_key: str) -> str | None:
        """Validate an API key and return the client_id if valid.

        Returns None if the key is invalid or expired.
        """
        key_hash = self.hash_key(api_key)
        for client_id, keys in self._keys.items():
            for ki in keys:
                if ki.key_hash == key_hash and ki.is_active:
                    return client_id
        return None

    def rotate_key(
        self,
        client_id: str,
        new_api_key: str,
        label: str = "",
    ) -> KeyInfo | None:
        """Rotate to a new key, keeping old keys active during grace period."""
        if client_id not in self._keys:
            return None

        # Set expiration on current primary
        for ki in self._keys[client_id]:
            if ki.is_primary:
                ki.is_primary = False
                ki.expires_at = time.time() + self.config.grace_period

        return self.register_key(client_id, new_api_key, label=label, is_primary=True)

    def revoke_key(self, client_id: str, api_key: str) -> bool:
        """Immediately revoke a key."""
        if client_id not in self._keys:
            return False

        key_hash = self.hash_key(api_key)
        before = len(self._keys[client_id])
        self._keys[client_id] = [
            ki for ki in self._keys[client_id]
            if ki.key_hash != key_hash
        ]
        return len(self._keys[client_id]) < before

    def get_client_keys(self, client_id: str) -> list[dict[str, Any]]:
        """List all keys for a client."""
        if client_id not in self._keys:
            return []
        return [ki.to_dict() for ki in self._keys[client_id] if ki.is_active]

    def cleanup_expired(self) -> int:
        """Remove all expired keys. Returns count of removed keys."""
        removed = 0
        for client_id in list(self._keys.keys()):
            before = len(self._keys[client_id])
            self._keys[client_id] = [
                ki for ki in self._keys[client_id] if ki.is_active
            ]
            removed += before - len(self._keys[client_id])
            if not self._keys[client_id]:
                del self._keys[client_id]
        return removed

    def _enforce_limits(self, client_id: str) -> None:
        """Enforce max keys per client by removing oldest non-primary keys."""
        keys = self._keys.get(client_id, [])
        while len(keys) > self.config.max_keys_per_client:
            # Remove oldest non-primary key
            for i, ki in enumerate(keys):
                if not ki.is_primary:
                    keys.pop(i)
                    break
            else:
                break
