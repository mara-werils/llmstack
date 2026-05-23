"""Tests for API key rotation."""

from __future__ import annotations

import time

import pytest

from llmstack.gateway.key_rotation import (
    KeyRotationManager,
    RotationConfig,
)


@pytest.fixture
def manager():
    return KeyRotationManager()


class TestKeyRotationManager:
    def test_register_and_validate(self, manager):
        manager.register_key("client1", "sk-test-123")
        assert manager.validate_key("sk-test-123") == "client1"

    def test_invalid_key_returns_none(self, manager):
        assert manager.validate_key("invalid") is None

    def test_register_sets_primary(self, manager):
        ki = manager.register_key("c1", "key1")
        assert ki.is_primary is True

    def test_rotate_demotes_old_key(self, manager):
        manager.register_key("c1", "old-key", label="old")
        manager.rotate_key("c1", "new-key", label="new")

        # Both keys should be valid
        assert manager.validate_key("old-key") == "c1"
        assert manager.validate_key("new-key") == "c1"

    def test_rotate_nonexistent_client(self, manager):
        result = manager.rotate_key("nope", "key")
        assert result is None

    def test_revoke_key(self, manager):
        manager.register_key("c1", "key-to-revoke")
        assert manager.revoke_key("c1", "key-to-revoke") is True
        assert manager.validate_key("key-to-revoke") is None

    def test_revoke_nonexistent(self, manager):
        assert manager.revoke_key("c1", "nope") is False

    def test_get_client_keys(self, manager):
        manager.register_key("c1", "k1", label="first")
        manager.register_key("c1", "k2", label="second")
        keys = manager.get_client_keys("c1")
        assert len(keys) == 2

    def test_expired_key_invalid(self):
        config = RotationConfig(grace_period=0)  # Instant expiry
        mgr = KeyRotationManager(config)
        mgr.register_key("c1", "old")
        mgr.rotate_key("c1", "new")
        time.sleep(0.01)
        # Old key should be expired
        assert mgr.validate_key("old") is None
        assert mgr.validate_key("new") == "c1"

    def test_cleanup_expired(self):
        config = RotationConfig(grace_period=0)
        mgr = KeyRotationManager(config)
        mgr.register_key("c1", "old")
        mgr.rotate_key("c1", "new")
        time.sleep(0.01)
        removed = mgr.cleanup_expired()
        assert removed >= 1

    def test_max_keys_enforced(self):
        config = RotationConfig(max_keys_per_client=2)
        mgr = KeyRotationManager(config)
        mgr.register_key("c1", "k1", is_primary=False)
        mgr.register_key("c1", "k2", is_primary=False)
        mgr.register_key("c1", "k3")
        keys = mgr.get_client_keys("c1")
        assert len(keys) <= 2

    def test_hash_key_deterministic(self):
        h1 = KeyRotationManager.hash_key("test")
        h2 = KeyRotationManager.hash_key("test")
        assert h1 == h2

    def test_get_keys_empty_client(self, manager):
        assert manager.get_client_keys("nobody") == []
