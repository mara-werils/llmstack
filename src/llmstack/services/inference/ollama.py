"""Ollama inference service."""

from __future__ import annotations

from typing import Any

import httpx

from llmstack.config.schema import ModelSpec
from llmstack.core.hardware import HardwareProfile
from llmstack.services.base import ServiceBase


class OllamaService(ServiceBase):
    name = "ollama"
    category = "inference"

    def __init__(self, model: ModelSpec, hw: HardwareProfile):
        self.model = model
        self.hw = hw
        self.host_port = 11434

    def container_spec(self) -> dict[str, Any]:
        spec: dict[str, Any] = {
            "image": "ollama/ollama:latest",
            "name": "llmstack-ollama",
            "ports": {"11434/tcp": self.host_port},
            "volumes": {
                "llmstack_ollama_data": {"bind": "/root/.ollama", "mode": "rw"},
            },
            "environment": {},
        }

        # GPU passthrough for NVIDIA
        if self.hw.gpu_vendor == "nvidia":
            import docker
            spec["device_requests"] = [
                docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])
            ]

        return spec

    def health_url(self) -> str:
        return f"http://localhost:{self.host_port}"

    async def post_start(self) -> None:
        """Pull the model after Ollama is healthy."""
        model_name = self.model.name
        if self.model.quantization:
            model_name = f"{self.model.name}:{self.model.quantization}"

        async with httpx.AsyncClient(timeout=600) as client:
            resp = await client.post(
                f"http://localhost:{self.host_port}/api/pull",
                json={"name": model_name, "stream": False},
            )
            resp.raise_for_status()

    def openai_base_url(self) -> str:
        return f"http://llmstack-ollama:{self.host_port}/v1"
