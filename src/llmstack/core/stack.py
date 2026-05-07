"""Stack orchestrator — manages the full lifecycle of an llmstack deployment."""

from __future__ import annotations

import secrets

from rich.console import Console
from rich.table import Table

from llmstack.config.schema import StackConfig
from llmstack.core.hardware import detect_hardware
from llmstack.core.health import wait_healthy
from llmstack.core.resolver import resolve_inference_backend, resolve_embedding_backend
from llmstack.docker.manager import DockerManager
from llmstack.services.base import ServiceBase, ServiceStatus, ServiceState
from llmstack.services.inference.ollama import OllamaService
from llmstack.services.inference.vllm import VllmService
from llmstack.services.embeddings.tei import TEIService
from llmstack.services.vectordb.qdrant import QdrantService
from llmstack.services.cache.redis import RedisService
from llmstack.services.gateway.service import GatewayService
from llmstack.services.observe.prometheus import PrometheusService, GrafanaService

console = Console()


class Stack:
    """Orchestrates boot and teardown of all services."""

    def __init__(self, config: StackConfig):
        self.config = config
        self.hw = detect_hardware()
        self.docker = DockerManager(network_name=config.docker.network)
        self._services: list[ServiceBase] = []

    def _build_services(self) -> list[ServiceBase]:
        """Instantiate services in boot order:
        vectordb -> cache -> inference -> embeddings
        """
        services: list[ServiceBase] = []

        # 1. Vector DB
        services.append(QdrantService(self.config.services.vectors))

        # 2. Cache
        services.append(RedisService(self.config.services.cache))

        # 3. Inference
        inference_backend = resolve_inference_backend(self.config.models.chat, self.hw)
        if inference_backend == "vllm":
            services.append(VllmService(self.config.models.chat, self.hw))
        else:
            services.append(OllamaService(self.config.models.chat, self.hw))

        # 4. Embeddings
        embed_backend = resolve_embedding_backend(self.config.models.embeddings, self.hw)
        if embed_backend == "tei":
            services.append(TEIService(self.config.models.embeddings, self.hw))
        # If embed_backend == "ollama", we reuse the Ollama container (no extra service)

        # 5. Gateway
        inference_url = self._resolve_inference_url(services, backend=inference_backend)
        embeddings_url = self._resolve_embeddings_url(services, embed_backend)
        qdrant_url = f"http://llmstack-qdrant:{self.config.services.vectors.port}"
        redis_url = f"redis://llmstack-redis:{self.config.services.cache.port}"

        services.append(GatewayService(
            config=self.config.gateway,
            inference_url=inference_url,
            embeddings_url=embeddings_url,
            qdrant_url=qdrant_url,
            redis_url=redis_url,
        ))

        # 6. Observability (optional)
        if self.config.observe.metrics:
            services.append(PrometheusService(self.config.observe))
            services.append(GrafanaService(self.config.observe))

        return services

    def _resolve_inference_url(self, services: list[ServiceBase], backend: str) -> str:
        for svc in services:
            if svc.category == "inference":
                return svc.openai_base_url() or ""
        return ""

    def _resolve_embeddings_url(self, services: list[ServiceBase], backend: str) -> str:
        for svc in services:
            if svc.category == "embeddings":
                return svc.openai_base_url() or ""
        # Fallback to inference (Ollama can do embeddings)
        for svc in services:
            if svc.category == "inference":
                return svc.openai_base_url() or ""
        return ""

    async def up(self) -> None:
        """Boot all services in order with health checks."""
        self._services = self._build_services()
        self.docker.ensure_network()

        # Build local images if needed
        for svc in self._services:
            if hasattr(svc, "build_info") and svc.build_info():
                info = svc.build_info()
                console.print(f"  [cyan]Building {svc.name} image...[/]", end="")
                self.docker.build_image(**info)
                console.print(" [green]done[/]")

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
                healthy = await wait_healthy(svc.health_url(), timeout=180)
                if not healthy:
                    console.print(" [red]FAILED[/]")
                    raise RuntimeError(f"Service {svc.name} failed to start")

            console.print(" [green]ready[/]")

            # Post-start hook (e.g., pull model)
            if svc.category == "inference":
                model_name = self.config.models.chat.name
                console.print(f"  [cyan]Pulling model {model_name}...[/]", end="")
            await svc.post_start()
            if svc.category == "inference":
                console.print(" [green]done[/]")

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

    def _get_inference_url(self) -> str:
        """Get the OpenAI base URL for the inference service."""
        for svc in self._services:
            if svc.category == "inference":
                return svc.openai_base_url() or ""
        return ""

    def _get_embeddings_url(self) -> str:
        """Get the embeddings URL."""
        for svc in self._services:
            if svc.category == "embeddings":
                return svc.openai_base_url() or ""
        # Fallback: use Ollama for embeddings
        for svc in self._services:
            if svc.category == "inference" and isinstance(svc, OllamaService):
                return svc.openai_base_url() or ""
        return ""

    def _print_summary(self) -> None:
        """Print a summary table of running services."""
        table = Table(title="LLMStack Services", show_header=True)
        table.add_column("Service", style="cyan")
        table.add_column("Category")
        table.add_column("Status", style="green")
        table.add_column("URL")

        for svc in self._services:
            url = svc.health_url()
            # Clean up URL for display
            for suffix in ["/healthz", "/health", "/api/tags"]:
                url = url.replace(suffix, "")
            table.add_row(svc.name, svc.category, "running", url)

        console.print()
        console.print(table)

        # Print usage hint
        inference_svc = next((s for s in self._services if s.category == "inference"), None)
        if inference_svc:
            base = inference_svc.health_url()
            for suffix in ["/healthz", "/health", "/api/tags"]:
                base = base.replace(suffix, "")
            model = self.config.models.chat.name
            console.print("\n[bold]Try it:[/]")
            console.print(
                f"  curl {base}/v1/chat/completions \\\n"
                f"    -H 'Content-Type: application/json' \\\n"
                f"    -d '{{\"model\":\"{model}\",\"messages\":[{{\"role\":\"user\",\"content\":\"Hello!\"}}]}}'"
            )
        console.print()
