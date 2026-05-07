"""Pydantic v2 models for llmstack.yaml configuration."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ModelSpec(BaseModel):
    name: str = "llama3.2"
    backend: Literal["auto", "ollama", "vllm"] = "auto"
    quantization: str | None = None
    gpu_layers: int = -1
    context_length: int = 8192
    extra_args: dict = Field(default_factory=dict)


class EmbeddingSpec(BaseModel):
    name: str = "bge-m3"
    backend: Literal["auto", "tei"] = "auto"
    dimensions: int | None = None


class ModelsConfig(BaseModel):
    chat: ModelSpec = Field(default_factory=ModelSpec)
    embeddings: EmbeddingSpec = Field(default_factory=EmbeddingSpec)


class VectorDBConfig(BaseModel):
    provider: Literal["qdrant"] = "qdrant"
    port: int = 6333
    storage_path: str = "./data/vectors"


class CacheConfig(BaseModel):
    provider: Literal["redis"] = "redis"
    port: int = 6379
    max_memory: str = "256mb"


class ServicesConfig(BaseModel):
    vectors: VectorDBConfig = Field(default_factory=VectorDBConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)


class GatewayConfig(BaseModel):
    port: int = 8000
    auth: Literal["none", "api_key"] = "api_key"
    api_keys: list[str] = Field(default_factory=list)
    rate_limit: str = "100/min"
    cors: list[str] = Field(default_factory=lambda: ["*"])
    request_timeout: int = 120


class ObserveConfig(BaseModel):
    metrics: bool = True
    dashboard_port: int = 8080
    retention: str = "7d"


class DockerConfig(BaseModel):
    network: str = "llmstack_net"
    gpu: Literal["auto", "true", "false"] = "auto"
    data_dir: str = "~/.llmstack/data"


class StackConfig(BaseModel):
    """Root config — 1:1 mapping with llmstack.yaml."""

    version: str = "1"
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    services: ServicesConfig = Field(default_factory=ServicesConfig)
    gateway: GatewayConfig = Field(default_factory=GatewayConfig)
    observe: ObserveConfig = Field(default_factory=ObserveConfig)
    docker: DockerConfig = Field(default_factory=DockerConfig)
