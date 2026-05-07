"""Stack orchestrator — manages the full lifecycle of an llmstack deployment."""

from __future__ import annotations

import asyncio
import secrets
from typing import AsyncIterator

from rich.console import Console

from llmstack.config.schema import StackConfig
from llmstack.core.hardware import HardwareProfile, detect_hardware
from llmstack.core.health import wait_healthy
from llmstack.core.resolver import resolve_inference_backend, resolve_embedding_backend
from llmstack.docker.manager import DockerManager
from llmstack.services.base import ServiceBase, ServiceStatus, ServiceState
from llmstack.services.inference.ollama import OllamaService
from llmstack.services.inference.vllm import VllmService
from llmstack.services.vectordb.qdrant import QdrantService
from llmstack.services.cache.redis import RedisService

console = Console()


class Stack:
    """Orchestrates boot and teardown of all services."""

    def __init__(self, config: StackConfig):
        self.config = config
        self.hw = detect_hardware()
        self.docker = DockerManager(network_name=config.docker.network)
        self._services: list[ServiceBase] = []

    def _build_services(self) -> list[ServiceBase]:
        """Instantiate services in boot order."""
        services: list[ServiceBase] = []

        # 1. Vector DB
        services.append(QdrantService(self.config.services.vectors))

        # 2. Cache
        services.append(RedisService(self.config.services.cache))

        # 3. Inference
        backend = resolve_inference_backend(self.config.models.chat, self.hw)
        if backend == "vllm":
            services.append(VllmService(self.config.models.chat, self.hw))
        else:
            services.append(OllamaService(self.config.models.chat, self.hw))

        return services

    async def up(self) -> None:
        """Boot all services in order with health checks."""
        self._services = self._build_services()
        self.docker.ensure_network()

        # Generate API key if needed
        if self.config.gateway.auth == "api_key" and not self.config.gateway.api_keys:
            key = f"sk-llmstack-{secrets.token_urlsafe(24)}"
            self.config.gateway.api_keys = [key]
            console.print(f"\n[bold green]Generated API key:[/] {key}\n")

        for svc in self._services:
            console.print(f"  [cyan]Starting {svc.name}...[/]", end="")
            self.docker.run_service(svc)

            # Health check (skip Redis — no HTTP health endpoint)
            if svc.category != "cache":
                healthy = await wait_healthy(svc.health_url(), timeout=120)
                if not healthy:
                    console.print(f" [red]FAILED[/]")
                    raise RuntimeError(f"Service {svc.name} failed to start")

            console.print(f" [green]ready[/]")

            # Post-start hook (e.g., pull model)
            await svc.post_start()

        # Print summary
        self._print_summary()

    def down(self, remove_volumes: bool = False) -> list[str]:
        """Stop all services in reverse order."""
        return self.docker.stop_all(remove_volumes=remove_volumes)

    def status(self) -> list[ServiceStatus]:
        """Get status of all managed services."""
        containers = self.docker.list_services()
        result = []
        for info in containers:
            state = ServiceState.RUNNING if info["status"] == "running" else ServiceState.STOPPED
            ports = info.get("ports", {})
            port = None
            if ports:
                first = list(ports.values())[0]
                if first and isinstance(first, list) and first:
                    port = first[0].get("HostPort")

            result.append(ServiceStatus(
                name=info["name"],
                state=state,
                port=int(port) if port else None,
                container_id=info["container_id"],
            ))
        return result

    def _print_summary(self) -> None:
        """Print a summary table of running services."""
        from rich.table import Table

        table = Table(title="LLMStack Services", show_header=True)
        table.add_column("Service", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("URL")

        for svc in self._services:
            url = svc.health_url().replace("/healthz", "").replace("/health", "")
            table.add_row(svc.name, "running", url)

        console.print()
        console.print(table)

        # Print usage hints
        inference_svc = next((s for s in self._services if s.category == "inference"), None)
        if inference_svc:
            base_url = inference_svc.health_url().rsplit("/", 1)[0]
            console.print(f"\n[bold]Try it:[/]")
            console.print(f'  curl {base_url}/v1/chat/completions \\')
            console.print(f'    -d \'{{"model":"{self.config.models.chat.name}","messages":[{{"role":"user","content":"Hello!"}}]}}\'')
        console.print()
