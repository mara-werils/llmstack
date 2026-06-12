"""Docker container lifecycle management."""

from __future__ import annotations

from typing import Iterator

import docker
from docker.errors import NotFound, APIError
from docker.models.containers import Container

from llmstack.services.base import ServiceBase


class DockerManager:
    """Wraps Docker SDK to manage llmstack containers."""

    LABEL_MANAGED = "llmstack.managed"
    LABEL_SERVICE = "llmstack.service"

    def __init__(self, network_name: str = "llmstack_net"):
        try:
            self.client = docker.from_env()
            self.client.ping()
        except docker.errors.DockerException as exc:
            raise SystemExit(
                "Cannot connect to Docker daemon. Is Docker running?\n"
                "Install: https://docs.docker.com/get-docker/"
            ) from exc
        self.network_name = network_name

    @property
    def managed_container_count(self) -> int:
        """Return the number of llmstack-managed containers."""
        return len(list(self._managed_containers()))

    def is_service_running(self, service_name: str) -> bool:
        """Return True if a managed container exists for the service."""
        return any(
            c.labels.get(self.LABEL_SERVICE) == service_name for c in self._managed_containers()
        )

    def ensure_network(self) -> None:
        """Create the bridge network if it doesn't exist."""
        try:
            self.client.networks.get(self.network_name)
        except NotFound:
            self.client.networks.create(self.network_name, driver="bridge")

    def build_image(self, path: str, dockerfile: str, tag: str) -> str:
        """Build a Docker image from a local Dockerfile. Returns the image tag."""
        self.client.images.build(path=path, dockerfile=dockerfile, tag=tag, rm=True)
        return tag

    def run_service(self, service: ServiceBase) -> Container:
        """Start a container for a service. Removes any existing container with the same name."""
        spec = service.container_spec()
        name = spec.pop("name", f"llmstack-{service.name}")

        # Remove existing container if present
        try:
            existing = self.client.containers.get(name)
            existing.stop(timeout=10)
            existing.remove(force=True)
        except NotFound:
            pass

        labels = {
            self.LABEL_MANAGED: "true",
            self.LABEL_SERVICE: service.name,
        }

        try:
            container = self.client.containers.run(
                detach=True,
                name=name,
                network=self.network_name,
                labels=labels,
                **spec,
            )
        except APIError as exc:
            msg = str(exc)
            if "port is already allocated" in msg or "address already in use" in msg:
                ports = spec.get("ports", {})
                port = next(iter(ports.values()), "unknown") if ports else "unknown"
                raise SystemExit(
                    f"Port {port} is already in use. "
                    f"Stop the conflicting service or change the port in llmstack.yaml."
                ) from exc
            raise
        return container

    def stop_service(self, service_name: str) -> None:
        """Stop and remove a container by service name."""
        for container in self._managed_containers():
            if container.labels.get(self.LABEL_SERVICE) == service_name:
                container.stop(timeout=10)
                container.remove(force=True)
                return

    def stop_all(self, remove_volumes: bool = False) -> list[str]:
        """Stop and remove all llmstack containers. Returns names of stopped containers."""
        stopped = []
        for container in self._managed_containers():
            name = container.name
            container.stop(timeout=10)
            container.remove(force=True)
            stopped.append(name)

        if remove_volumes:
            for vol in self.client.volumes.list():
                if vol.name.startswith("llmstack_"):
                    try:
                        vol.remove(force=True)
                    except APIError:
                        pass

        # Remove network
        try:
            net = self.client.networks.get(self.network_name)
            net.remove()
        except (NotFound, APIError):
            pass

        return stopped

    def get_container(self, service_name: str) -> Container | None:
        """Find a running container by service name."""
        for container in self._managed_containers():
            if container.labels.get(self.LABEL_SERVICE) == service_name:
                return container
        return None

    def stream_logs(self, service_name: str, follow: bool = True, tail: int = 50) -> Iterator[str]:
        """Yield decoded log lines from a service container."""
        container = self.get_container(service_name)
        if container is None:
            raise ValueError(f"No running container for service '{service_name}'")

        for chunk in container.logs(stream=True, follow=follow, tail=tail):
            yield chunk.decode("utf-8", errors="replace")

    def list_services(self) -> list[dict]:
        """Return info about all managed containers."""
        result = []
        for container in self._managed_containers():
            container.reload()
            result.append(
                {
                    "name": container.labels.get(self.LABEL_SERVICE, "unknown"),
                    "container_name": container.name,
                    "container_id": container.short_id,
                    "status": container.status,
                    "ports": container.ports,
                }
            )
        return result

    def _managed_containers(self) -> list[Container]:
        """List all containers with the llmstack.managed label."""
        return self.client.containers.list(
            all=True,
            filters={"label": f"{self.LABEL_MANAGED}=true"},
        )
