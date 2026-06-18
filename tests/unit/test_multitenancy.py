"""Tests for multi-tenant namespace isolation."""

import pytest

from llmstack.gateway.multitenancy import TenantManager, Tenant


@pytest.fixture
def manager():
    return TenantManager()


class TestTenantManager:
    def test_create_tenant(self, manager):
        t = manager.create_tenant(name="Acme Corp", tier="pro")
        assert t.name == "Acme Corp"
        assert t.tier == "pro"
        assert t.active is True

    def test_create_with_api_keys(self, manager):
        t = manager.create_tenant(
            name="Test",
            api_keys=["sk-123", "sk-456"],
        )
        assert len(t.api_keys) == 2

    def test_resolve_tenant_by_key(self, manager):
        t = manager.create_tenant(name="Test", api_keys=["sk-test"])
        resolved = manager.resolve_tenant("sk-test")
        assert resolved is not None
        assert resolved.id == t.id

    def test_resolve_unknown_key(self, manager):
        assert manager.resolve_tenant("unknown") is None

    def test_add_api_key(self, manager):
        t = manager.create_tenant(name="Test")
        assert manager.add_api_key(t.id, "sk-new") is True
        resolved = manager.resolve_tenant("sk-new")
        assert resolved.id == t.id

    def test_add_duplicate_key_fails(self, manager):
        manager.create_tenant(name="T1", api_keys=["sk-dup"])
        t2 = manager.create_tenant(name="T2")
        assert manager.add_api_key(t2.id, "sk-dup") is False

    def test_remove_api_key(self, manager):
        manager.create_tenant(name="Test", api_keys=["sk-rem"])
        assert manager.remove_api_key("sk-rem") is True
        assert manager.resolve_tenant("sk-rem") is None

    def test_deactivate_tenant(self, manager):
        t = manager.create_tenant(name="Test")
        assert manager.deactivate_tenant(t.id) is True
        assert t.active is False

    def test_list_active_only(self, manager):
        manager.create_tenant(name="Active")
        t2 = manager.create_tenant(name="Inactive")
        manager.deactivate_tenant(t2.id)

        active = manager.list_tenants(active_only=True)
        assert len(active) == 1
        assert active[0].name == "Active"

    def test_namespace_key(self, manager):
        t = manager.create_tenant(name="NS", api_keys=["sk-ns"])
        key = manager.namespace_key("sk-ns", "conversations")
        assert key == f"tenant:{t.id}:conversations"

    def test_namespace_key_no_tenant(self, manager):
        key = manager.namespace_key("unknown-key", "conversations")
        assert key == "default:conversations"

    def test_stats(self, manager):
        manager.create_tenant(name="A", tier="pro")
        manager.create_tenant(name="B", tier="free")
        stats = manager.get_stats()
        assert stats["total_tenants"] == 2
        assert stats["active_tenants"] == 2
        assert stats["by_tier"]["pro"] == 1

    def test_to_dict(self):
        t = Tenant(name="Test", tier="enterprise")
        d = t.to_dict()
        assert d["name"] == "Test"
        assert "limits" in d

    def test_get_tenant_found(self, manager):
        t = manager.create_tenant(name="Test")
        assert manager.get_tenant(t.id) is t

    def test_get_tenant_not_found(self, manager):
        assert manager.get_tenant("nonexistent") is None

    def test_add_api_key_to_unknown_tenant_fails(self, manager):
        assert manager.add_api_key("nonexistent", "sk-x") is False

    def test_remove_unknown_api_key_fails(self, manager):
        assert manager.remove_api_key("never-existed") is False

    def test_deactivate_unknown_tenant_fails(self, manager):
        assert manager.deactivate_tenant("nonexistent") is False

    def test_tenant_count_and_active_count(self, manager):
        t1 = manager.create_tenant(name="A")
        manager.create_tenant(name="B")
        manager.deactivate_tenant(t1.id)
        assert manager.tenant_count == 2
        assert manager.active_count == 1
