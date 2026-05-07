"""Qdrant vector database service."""

from __future__ import annotations

from typing import Any

from llmstack.config.schema import VectorDBConfig
from llmstack.services.base import ServiceBase


class QdrantService(ServiceBase):
    name = "qdrant"
    category = "vectordb"

    def __init__(self, config: VectorDBConfig):
        self.config = config

    def container_spec(self) -> dict[str, Any]:
        return {
            "image": "qdrant/qdrant:latest",
            "name": "llmstack-qdrant",
            "ports": {
                "6333/tcp": self.config.port,
                "6334/tcp": self.config.port + 1,
            },
            "volumes": {
                "llmstack_qdrant_data": {"bind": "/qdrant/storage", "mode": "rw"},
            },
            "environment": {},
        }

    def health_url(self) -> str:
        return f"http://localhost:{self.config.port}/healthz"
