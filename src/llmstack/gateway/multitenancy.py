"""Multi-tenant support — namespace isolation for API keys.

Each tenant gets isolated access to their own conversations,
templates, and usage data, preventing cross-tenant data leakage.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from threading import Lock
from typing import Any


@dataclass
class Tenant:
    """A tenant with isolated namespace."""

    id: str = ""
    name: str = ""
    api_keys: list[str] = field(default_factory=list)
    tier: str = "standard"
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = 0.0
    active: bool = True

    # Resource limits
    max_models: int = 0  # 0 = unlimited
    max_requests_per_day: int = 0
    max_tokens_per_day: int = 0
    max_cost_per_day: float = 0.0

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())[:12]
        if not self.created_at:
            self.created_at = time.time()

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "tier": self.tier,
            "api_keys": len(self.api_keys),
            "active": self.active,
            "created_at": self.created_at,
            "limits": {
                "max_models": self.max_models,
                "max_requests_per_day": self.max_requests_per_day,
                "max_tokens_per_day": self.max_tokens_per_day,
                "max_cost_per_day": self.max_cost_per_day,
            },
        }


class TenantManager:
    """Manages tenants and resolves API keys to tenant namespaces."""

    def __init__(self):
        self._lock = Lock()
        self._tenants: dict[str, Tenant] = {}
        self._key_map: dict[str, str] = {}  # api_key -> tenant_id

    def create_tenant(
        self,
        name: str,
        tier: str = "standard",
        api_keys: list[str] | None = None,
        **kwargs,
    ) -> Tenant:
        """Create a new tenant."""
        tenant = Tenant(name=name, tier=tier, api_keys=api_keys or [], **kwargs)
        with self._lock:
            self._tenants[tenant.id] = tenant
            for key in tenant.api_keys:
                self._key_map[key] = tenant.id
        return tenant

    def get_tenant(self, tenant_id: str) -> Tenant | None:
        """Get a tenant by ID."""
        with self._lock:
            return self._tenants.get(tenant_id)

    def resolve_tenant(self, api_key: str) -> Tenant | None:
        """Resolve an API key to its tenant."""
        with self._lock:
            tenant_id = self._key_map.get(api_key)
            if tenant_id:
                return self._tenants.get(tenant_id)
            return None

    def add_api_key(self, tenant_id: str, api_key: str) -> bool:
        """Add an API key to a tenant."""
        with self._lock:
            tenant = self._tenants.get(tenant_id)
            if tenant is None:
                return False
            if api_key in self._key_map:
                return False  # Key already assigned
            tenant.api_keys.append(api_key)
            self._key_map[api_key] = tenant_id
            return True

    def remove_api_key(self, api_key: str) -> bool:
        """Remove an API key."""
        with self._lock:
            tenant_id = self._key_map.pop(api_key, None)
            if tenant_id:
                tenant = self._tenants.get(tenant_id)
                if tenant:
                    tenant.api_keys = [k for k in tenant.api_keys if k != api_key]
                return True
            return False

    def deactivate_tenant(self, tenant_id: str) -> bool:
        """Deactivate a tenant (soft delete)."""
        with self._lock:
            tenant = self._tenants.get(tenant_id)
            if tenant is None:
                return False
            tenant.active = False
            return True

    @property
    def tenant_count(self) -> int:
        """Total number of tenants (including inactive)."""
        with self._lock:
            return len(self._tenants)

    @property
    def active_count(self) -> int:
        """Number of active tenants."""
        with self._lock:
            return sum(1 for t in self._tenants.values() if t.active)

    def list_tenants(self, active_only: bool = True) -> list[Tenant]:
        """List all tenants."""
        with self._lock:
            tenants = list(self._tenants.values())
            if active_only:
                tenants = [t for t in tenants if t.active]
            return sorted(tenants, key=lambda t: t.created_at, reverse=True)

    def namespace_key(self, api_key: str, resource: str) -> str:
        """Generate a namespaced key for tenant-isolated resources.

        Example: namespace_key("sk-123", "conversations") -> "tenant:abc123:conversations"
        """
        tenant = self.resolve_tenant(api_key)
        if tenant:
            return f"tenant:{tenant.id}:{resource}"
        return f"default:{resource}"

    def get_stats(self) -> dict:
        """Get multi-tenancy statistics."""
        with self._lock:
            active = sum(1 for t in self._tenants.values() if t.active)
            total_keys = len(self._key_map)
            tier_counts: dict[str, int] = {}
            for t in self._tenants.values():
                if t.active:
                    tier_counts[t.tier] = tier_counts.get(t.tier, 0) + 1
            return {
                "total_tenants": len(self._tenants),
                "active_tenants": active,
                "total_api_keys": total_keys,
                "by_tier": tier_counts,
            }
