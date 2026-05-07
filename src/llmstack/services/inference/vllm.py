"""vLLM inference service."""

from __future__ import annotations

from typing import Any

from llmstack.config.schema import ModelSpec
from llmstack.core.hardware import HardwareProfile
from llmstack.services.base import ServiceBase


class VllmService(ServiceBase):
    name = "vllm"
    category = "inference"

    def __init__(self, model: ModelSpec, hw: HardwareProfile):
        self.model = model
        self.hw = hw
        self.host_port = 8001

    def container_spec(self) -> dict[str, Any]:
        import docker

        cmd = [
            "--model", self.model.name,
            "--host", "0.0.0.0",
            "--port", "8000",
            "--max-model-len", str(self.model.context_length),
        ]

        if self.model.quantization:
            cmd.extend(["--quantization", self.model.quantization])

        spec: dict[str, Any] = {
            "image": "vllm/vllm-openai:latest",
            "name": "llmstack-vllm",
            "ports": {"8000/tcp": self.host_port},
            "command": cmd,
            "environment": {
                "HUGGING_FACE_HUB_TOKEN": "",
            },
            "volumes": {
                "llmstack_vllm_cache": {"bind": "/root/.cache/huggingface", "mode": "rw"},
            },
            "device_requests": [
                docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])
            ],
            "shm_size": "4g",
        }

        return spec

    def health_url(self) -> str:
        return f"http://localhost:{self.host_port}/health"

    def openai_base_url(self) -> str:
        return "http://llmstack-vllm:8000/v1"
