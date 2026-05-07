"""Redis cache service."""

from __future__ import annotations

from typing import Any

from llmstack.config.schema import CacheConfig
from llmstack.services.base import ServiceBase


class RedisService(ServiceBase):
    name = "redis"
    category = "cache"

    def __init__(self, config: CacheConfig):
        self.config = config

    def container_spec(self) -> dict[str, Any]:
        return {
            "image": "redis:7-alpine",
            "name": "llmstack-redis",
            "ports": {"6379/tcp": self.config.port},
            "command": [
                "redis-server",
                "--maxmemory", self.config.max_memory,
                "--maxmemory-policy", "allkeys-lru",
            ],
            "environment": {},
        }

    def health_url(self) -> str:
        # Redis doesn't have HTTP health, we check via TCP
        return f"http://localhost:{self.config.port}"
