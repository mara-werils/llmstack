"""Pydantic v2 models for llmstack.yaml configuration.

These models provide strict validation with helpful error messages so that
users get clear feedback when their ``llmstack.yaml`` contains typos or
invalid values.

Minimal example::

    version: "1"
    models:
      chat:
        name: llama3.2
    gateway:
      port: 8000
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class ModelSpec(BaseModel):
    """Specification for a chat/completion model.

    Example::

        chat:
          name: llama3.2
          context_length: 8192
          gpu_layers: -1        # -1 = all layers on GPU
    """

    name: str = "llama3.2"
    backend: Literal["auto", "ollama", "vllm"] = "auto"
    quantization: str | None = None
    gpu_layers: int = -1
    context_length: int = 8192
    extra_args: dict = Field(default_factory=dict)

    @field_validator("context_length")
    @classmethod
    def context_length_must_be_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(
                f"context_length must be a positive integer, got {v}. "
                "Common values: 2048, 4096, 8192, 16384, 32768, 131072."
            )
        return v

    @field_validator("gpu_layers")
    @classmethod
    def gpu_layers_range(cls, v: int) -> int:
        if v < -1:
            raise ValueError(
                f"gpu_layers must be -1 (all) or >= 0, got {v}. "
                "Use -1 to offload all layers to GPU, 0 for CPU only."
            )
        return v


class EmbeddingSpec(BaseModel):
    name: str = "bge-m3"
    backend: Literal["auto", "tei"] = "auto"
    dimensions: int | None = None


class ModelsConfig(BaseModel):
    chat: ModelSpec = Field(default_factory=ModelSpec)
    embeddings: EmbeddingSpec = Field(default_factory=EmbeddingSpec)


class VectorDBConfig(BaseModel):
    """Vector database configuration.

    Example::

        vectors:
          provider: qdrant
          port: 6333
    """

    provider: Literal["qdrant"] = "qdrant"
    port: int = 6333
    storage_path: str = "./data/vectors"

    @field_validator("port")
    @classmethod
    def port_in_range(cls, v: int) -> int:
        if not (1 <= v <= 65535):
            raise ValueError(f"port must be between 1 and 65535, got {v}.")
        return v


class CacheConfig(BaseModel):
    """Redis cache configuration.

    Example::

        cache:
          port: 6379
          max_memory: 256mb
    """

    provider: Literal["redis"] = "redis"
    port: int = 6379
    max_memory: str = "256mb"

    @field_validator("max_memory")
    @classmethod
    def max_memory_format(cls, v: str) -> str:
        if not re.match(r"^\d+\s*(mb|gb|kb)$", v.lower().strip()):
            raise ValueError(
                f"max_memory must be like '256mb' or '1gb', got '{v}'. Supported units: kb, mb, gb."
            )
        return v


class ServicesConfig(BaseModel):
    vectors: VectorDBConfig = Field(default_factory=VectorDBConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)


class GuardrailsConfig(BaseModel):
    """Content safety guardrails configuration."""

    enabled: bool = False
    pii_detection: bool = True
    prompt_injection_detection: bool = True
    custom_rules: list[dict] = Field(default_factory=list)


class WebhooksConfig(BaseModel):
    """Webhook notification configuration."""

    enabled: bool = False
    endpoints: list[dict] = Field(default_factory=list)


class CostConfig(BaseModel):
    """Cost tracking and budget configuration."""

    enabled: bool = True
    budgets: list[dict] = Field(default_factory=list)


class BatchConfig(BaseModel):
    """Batch processing configuration."""

    enabled: bool = True
    max_batch_size: int = 100
    default_concurrency: int = 5


class GatewayConfig(BaseModel):
    """API gateway configuration.

    Example::

        gateway:
          port: 8000
          auth: api_key
          rate_limit: "100/min"
    """

    port: int = 8000
    auth: Literal["none", "api_key"] = "api_key"
    api_keys: list[str] = Field(default_factory=list)
    rate_limit: str = "100/min"
    cors: list[str] = Field(default_factory=lambda: ["*"])
    request_timeout: int = 120
    guardrails: GuardrailsConfig = Field(default_factory=GuardrailsConfig)
    webhooks: WebhooksConfig = Field(default_factory=WebhooksConfig)
    cost: CostConfig = Field(default_factory=CostConfig)
    batch: BatchConfig = Field(default_factory=BatchConfig)
    warmup_models: list[str] = Field(default_factory=list)

    @field_validator("port")
    @classmethod
    def port_in_range(cls, v: int) -> int:
        if not (1 <= v <= 65535):
            raise ValueError(f"port must be between 1 and 65535, got {v}.")
        return v

    @field_validator("rate_limit")
    @classmethod
    def rate_limit_format(cls, v: str) -> str:
        if not re.match(r"^\d+/(sec|min|hour|day)$", v.strip()):
            raise ValueError(
                f"rate_limit must be like '100/min' or '1000/hour', got '{v}'. "
                "Format: <number>/<sec|min|hour|day>."
            )
        return v

    @field_validator("request_timeout")
    @classmethod
    def timeout_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"request_timeout must be a positive integer (seconds), got {v}.")
        return v


class ObserveConfig(BaseModel):
    """Observability and quality tracking configuration.

    Example::

        observe:
          metrics: true
          quality_tracking: true
          alert_threshold: 0.4
    """

    metrics: bool = True
    dashboard_port: int = 8080
    retention: str = "7d"
    quality_tracking: bool = True  # enable AI quality scoring
    alert_threshold: float = 0.4  # fire alert below this quality score
    drift_threshold: float = -0.1  # fire alert on quality drift
    trace_store_size: int = 5000  # max traces in memory

    @field_validator("alert_threshold")
    @classmethod
    def alert_threshold_range(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(
                f"alert_threshold must be between 0.0 and 1.0, got {v}. "
                "This is the minimum quality score before an alert fires."
            )
        return v

    @field_validator("trace_store_size")
    @classmethod
    def trace_store_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"trace_store_size must be a positive integer, got {v}.")
        return v

    @field_validator("retention")
    @classmethod
    def retention_format(cls, v: str) -> str:
        if not re.match(r"^\d+[dhm]$", v.strip()):
            raise ValueError(f"retention must be like '7d', '24h', or '30m', got '{v}'.")
        return v


class DockerConfig(BaseModel):
    network: str = "llmstack_net"
    gpu: Literal["auto", "true", "false"] = "auto"
    data_dir: str = "~/.llmstack/data"


# ---------------------------------------------------------------------------
# Provider configuration (Universal Gateway)
# ---------------------------------------------------------------------------


class ProviderModelConfig(BaseModel):
    """A model exposed through a provider, with optional tier and cost overrides."""

    name: str  # model ID, e.g. "gpt-4o"
    tier: Literal["simple", "medium", "complex"] = "medium"
    context_length: int = 128_000
    cost_per_m_input: float = 0.0  # $ per 1M input tokens
    cost_per_m_output: float = 0.0  # $ per 1M output tokens
    speed_score: float = 1.0
    quality_score: float = 1.0


class ProviderConfig(BaseModel):
    """Configuration for a single LLM provider."""

    name: str  # "openai", "anthropic", "google", etc.
    api_key: str = ""  # can also come from env var
    api_key_env: str = ""  # env var name, e.g. "OPENAI_API_KEY"
    base_url: str = ""  # override default API base URL
    enabled: bool = True
    models: list[ProviderModelConfig] = Field(default_factory=list)
    fallback: list[str] = Field(default_factory=list)  # fallback provider names


class ProvidersConfig(BaseModel):
    """Top-level providers configuration."""

    enabled: bool = False
    strategy: Literal["cost", "quality", "balanced", "latency"] = "cost"
    providers: list[ProviderConfig] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Agent configuration
# ---------------------------------------------------------------------------


class AgentToolConfig(BaseModel):
    """Configuration for an individual agent tool."""

    name: str  # tool name, e.g. "shell", "read_file"
    enabled: bool = True
    timeout: int = 60  # per-tool timeout in seconds


class AgentProfileConfig(BaseModel):
    """Configuration for a named agent profile."""

    name: str = "default"  # profile name
    model: str = "llama3.2"  # LLM model for the agent
    max_steps: int = 25  # max tool-use iterations
    max_tokens: int = 4096
    temperature: float = 0.1
    system_prompt: str = ""  # custom system prompt
    tools: list[str] = Field(
        default_factory=lambda: [
            "read_file",
            "write_file",
            "list_directory",
            "grep",
            "shell",
            "http_get",
        ]
    )


class AgentsConfig(BaseModel):
    """Top-level agents configuration."""

    profiles: list[AgentProfileConfig] = Field(default_factory=list)


class MCPConfig(BaseModel):
    """MCP server configuration."""

    enabled: bool = False
    model: str = "llama3.2"
    tools: list[str] = Field(
        default_factory=lambda: [
            "read_file",
            "write_file",
            "list_directory",
            "grep",
            "shell",
            "http_get",
            "llmstack_chat",
            "llmstack_ask",
        ]
    )


# ---------------------------------------------------------------------------
# Fine-tuning configuration
# ---------------------------------------------------------------------------


class FinetuneConfig(BaseModel):
    """Fine-tuning pipeline configuration."""

    base_model: str = "unsloth/llama-3.2-1b-instruct-bnb-4bit"
    method: Literal["qlora", "lora", "full"] = "qlora"
    output_dir: str = "./finetune-output"
    lora_r: int = 16
    lora_alpha: int = 32
    epochs: int = 0  # 0 = auto
    batch_size: int = 0  # 0 = auto
    learning_rate: float = 0.0  # 0.0 = auto
    max_seq_length: int = 2048
    eval_split: float = 0.1
    quantization: str = "q4_k_m"  # for GGUF export


class StackConfig(BaseModel):
    """Root config -- 1:1 mapping with ``llmstack.yaml``.

    Example::

        version: "1"
        models:
          chat:
            name: llama3.2
        gateway:
          port: 8000
          auth: api_key
        observe:
          metrics: true
    """

    version: str = "1"
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    services: ServicesConfig = Field(default_factory=ServicesConfig)
    gateway: GatewayConfig = Field(default_factory=GatewayConfig)
    observe: ObserveConfig = Field(default_factory=ObserveConfig)
    docker: DockerConfig = Field(default_factory=DockerConfig)
    providers: ProvidersConfig = Field(default_factory=ProvidersConfig)
    agents: AgentsConfig = Field(default_factory=AgentsConfig)
    mcp: MCPConfig = Field(default_factory=MCPConfig)
    finetune: FinetuneConfig = Field(default_factory=FinetuneConfig)

    @field_validator("version")
    @classmethod
    def supported_version(cls, v: str) -> str:
        supported = {"1"}
        if v not in supported:
            raise ValueError(
                f"Unsupported config version '{v}'. "
                f"Supported versions: {', '.join(sorted(supported))}."
            )
        return v
