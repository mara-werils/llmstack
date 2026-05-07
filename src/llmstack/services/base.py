"""Base class for all managed services."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ServiceState(str, Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    UNHEALTHY = "unhealthy"
    ERROR = "error"


@dataclass
class ServiceStatus:
    name: str
    state: ServiceState
    port: int | None = None
    container_id: str | None = None
    message: str = ""


class ServiceBase(ABC):
    """Every llmstack service (inference, vectordb, cache, etc.) implements this."""

    name: str
    category: str  # inference, vectordb, cache, embeddings, gateway, observe

    @abstractmethod
    def container_spec(self) -> dict[str, Any]:
        """Return kwargs for docker.containers.run().

        Must include at least: image, ports, environment.
        May include: volumes, device_requests, healthcheck, command.
        """

    @abstractmethod
    def health_url(self) -> str:
        """HTTP URL to GET for health checks (from the host)."""

    async def post_start(self) -> None:
        """Hook called after the container is healthy.

        Override for actions like pulling a model.
        """

    def openai_base_url(self) -> str | None:
        """If this service exposes an OpenAI-compatible API, return its internal Docker URL."""
        return None

    def internal_url(self) -> str:
        """Return the URL reachable from other containers on the Docker network."""
        spec = self.container_spec()
        ports = spec.get("ports", {})
        # Get the first container port
        if ports:
            container_port = list(ports.values())[0] if isinstance(ports, dict) else None
            if container_port:
                return f"http://{self.name}:{container_port}"
        return f"http://{self.name}"
