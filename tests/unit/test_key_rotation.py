"""Tests for API key rotation."""

from __future__ import annotations

import time

import pytest

from llmstack.gateway.key_rotation import (
    KeyInfo,
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

    def test_days_until_expiry_none_for_permanent_key(self, manager):
        ki = manager.register_key("c1", "key1")
        assert ki.days_until_expiry is None

    def test_days_until_expiry_for_expiring_key(self, manager):
        ki = KeyInfo(key_hash="h", expires_at=time.time() + 86400 * 2)
        assert 1.9 < ki.days_until_expiry < 2.1

    def test_days_until_expiry_clamped_to_zero_when_past(self, manager):
        ki = KeyInfo(key_hash="h", expires_at=time.time() - 1000)
        assert ki.days_until_expiry == 0.0

    def test_cleanup_removes_client_with_no_remaining_keys(self, manager):
        config = RotationConfig(grace_period=0)
        mgr = KeyRotationManager(config)
        mgr.register_key("lone-client", "only-key")
        time.sleep(0.01)
        # Force expiry by setting expires_at in the past directly.
        mgr._keys["lone-client"][0].expires_at = time.time() - 1
        mgr.cleanup_expired()
        assert "lone-client" not in mgr._keys

    def test_enforce_limits_skips_when_all_keys_primary(self, manager):
        config = RotationConfig(max_keys_per_client=1)
        mgr = KeyRotationManager(config)
        mgr._keys["c1"] = [
            KeyInfo(key_hash="a", is_primary=True),
            KeyInfo(key_hash="b", is_primary=True),
        ]
        mgr._enforce_limits("c1")
        # Nothing removable (all primary) -> both keys remain despite exceeding the limit.
        assert len(mgr._keys["c1"]) == 2
