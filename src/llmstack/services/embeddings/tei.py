"""HuggingFace Text Embeddings Inference (TEI) service."""

from __future__ import annotations

from typing import Any

from llmstack.config.schema import EmbeddingSpec
from llmstack.core.hardware import HardwareProfile
from llmstack.services.base import ServiceBase


class TEIService(ServiceBase):
    name = "tei"
    category = "embeddings"

    def __init__(self, spec: EmbeddingSpec, hw: HardwareProfile):
        self.spec = spec
        self.hw = hw
        self.host_port = 8002

    def container_spec(self) -> dict[str, Any]:
        cmd = ["--model-id", self.spec.name, "--port", "80"]

        spec: dict[str, Any] = {
            "image": "ghcr.io/huggingface/text-embeddings-inference:cpu-latest",
            "name": "llmstack-tei",
            "ports": {"80/tcp": self.host_port},
            "command": cmd,
            "volumes": {
                "llmstack_tei_cache": {"bind": "/data", "mode": "rw"},
            },
            "environment": {},
        }

        # Use GPU image if NVIDIA available
        if self.hw.gpu_vendor == "nvidia":
            import docker
            spec["image"] = "ghcr.io/huggingface/text-embeddings-inference:latest"
            spec["device_requests"] = [
                docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])
            ]

        return spec

    def health_url(self) -> str:
        return f"http://localhost:{self.host_port}/health"

    def openai_base_url(self) -> str:
        return "http://llmstack-tei:80/v1"
