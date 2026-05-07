"""Gateway service — runs the FastAPI proxy as a Docker container."""

from __future__ import annotations

from typing import Any

from llmstack.config.schema import GatewayConfig
from llmstack.services.base import ServiceBase


class GatewayService(ServiceBase):
    name = "gateway"
    category = "gateway"

    def __init__(
        self,
        config: GatewayConfig,
        inference_url: str,
        embeddings_url: str,
        qdrant_url: str = "",
        redis_url: str = "",
    ):
        self.config = config
        self.inference_url = inference_url
        self.embeddings_url = embeddings_url
        self.qdrant_url = qdrant_url
        self.redis_url = redis_url

    def container_spec(self) -> dict[str, Any]:
        return {
            "image": "ghcr.io/mara-werils/llmstack-gateway:latest",
            "name": "llmstack-gateway",
            "ports": {"8000/tcp": self.config.port},
            "environment": {
                "LLMSTACK_INFERENCE_URL": self.inference_url,
                "LLMSTACK_EMBEDDINGS_URL": self.embeddings_url,
                "LLMSTACK_QDRANT_URL": self.qdrant_url,
                "LLMSTACK_REDIS_URL": self.redis_url,
                "LLMSTACK_API_KEYS": ",".join(self.config.api_keys),
                "LLMSTACK_CORS_ORIGINS": ",".join(self.config.cors),
                "LLMSTACK_REQUEST_TIMEOUT": str(self.config.request_timeout),
                "LLMSTACK_RATE_LIMIT": self.config.rate_limit,
            },
        }

    def health_url(self) -> str:
        return f"http://localhost:{self.config.port}/healthz"
