# Architecture

This page explains how llmstack works internally -- from the CLI command to running containers.

## High-Level Overview

llmstack follows a layered architecture:

```
┌─────────────────────────────────────────────┐
│                   CLI Layer                  │
│  Typer commands: init, up, down, status, ... │
├─────────────────────────────────────────────┤
│                  Core Layer                  │
│  Stack orchestrator, hardware detection,     │
│  backend resolver                            │
├─────────────────────────────────────────────┤
│                Service Layer                 │
│  Ollama, vLLM, Qdrant, Redis, TEI,          │
│  Gateway, Prometheus, Grafana               │
├─────────────────────────────────────────────┤
│                Docker Layer                  │
│  Docker SDK for Python (container mgmt)      │
├─────────────────────────────────────────────┤
│               Gateway Layer                  │
│  FastAPI app (runs inside its own container) │
│  Routes, middleware, RAG, cache, breaker     │
└─────────────────────────────────────────────┘
```

## Project Structure

```
src/llmstack/
├── cli/          # Typer CLI commands
├── config/       # Pydantic config schema + presets
├── core/         # Stack orchestrator, hardware detection, resolver
├── services/     # Service implementations (Ollama, vLLM, Qdrant, Redis, etc.)
├── gateway/      # FastAPI gateway (OpenAI-compatible proxy)
├── docker/       # Docker SDK wrapper
└── plugins/      # Plugin interface + loader
```

## Boot Sequence

When you run `llmstack up`, the following happens:

### 1. Load Configuration

The `StackConfig` Pydantic model parses `llmstack.yaml`. Every field has a sensible default, so an empty file is valid.

### 2. Detect Hardware

The `detect_hardware()` function probes the host system:

- **NVIDIA GPU**: Runs `nvidia-smi` to get GPU name and VRAM
- **Apple Silicon**: Reads `machdep.cpu.brand_string` via `sysctl`, uses total RAM as unified memory
- **CPU only**: Falls back gracefully

The result is a `HardwareProfile` dataclass:

```python
@dataclass(frozen=True)
class HardwareProfile:
    gpu_vendor: Literal["nvidia", "amd", "apple", "none"]
    gpu_name: str | None
    gpu_vram_mb: int
    cpu_cores: int
    ram_mb: int
    os: Literal["linux", "darwin", "windows"]
    docker_runtime: Literal["nvidia", "default"]
```

### 3. Resolve Backends

The resolver module maps hardware capabilities to backend choices:

| Hardware | Inference Backend | Embedding Backend |
|---|---|---|
| NVIDIA GPU 16GB+ VRAM | vLLM | TEI (GPU) |
| NVIDIA GPU < 16GB | Ollama | TEI (CPU) |
| Apple Silicon | Ollama | TEI (CPU) or Ollama |
| CPU only | Ollama | Ollama |

### 4. Build Service List

The `Stack._build_services()` method creates service instances in boot order:

1. **Qdrant** (vector DB) -- no dependencies
2. **Redis** (cache) -- no dependencies
3. **Inference** (Ollama or vLLM) -- no dependencies
4. **Embeddings** (TEI or reuse Ollama) -- depends on inference for fallback
5. **Gateway** (FastAPI) -- depends on all above
6. **Prometheus** (metrics) -- depends on gateway
7. **Grafana** (dashboard) -- depends on Prometheus

### 5. Start Services with Health Checks

Each service is started via the Docker SDK. After creating the container, llmstack polls the service's health endpoint until it responds with a success status or the timeout (180 seconds) is reached.

### 6. Post-Start Hooks

Some services have post-start hooks. For example, the Ollama service runs `ollama pull <model>` after the container is healthy to ensure the model weights are downloaded.

### 7. Generate API Key

If `gateway.auth` is `api_key` and no keys are configured, a secure random key is generated (`sk-llmstack-<random>`) and written back to `llmstack.yaml`.

## Service Abstraction

Every service implements the `ServiceBase` interface:

```python
class ServiceBase:
    name: str          # e.g., "ollama"
    category: str      # e.g., "inference"

    def container_spec(self) -> dict:
        """Return Docker container configuration."""
        ...

    def health_url(self) -> str:
        """Return HTTP URL for health checking."""
        ...

    async def post_start(self) -> None:
        """Run after the container is healthy."""
        ...

    def openai_base_url(self) -> str | None:
        """Return OpenAI-compatible base URL, if applicable."""
        ...
```

This abstraction makes it straightforward to add new backends or replace existing ones.

## Docker Management

llmstack uses the Docker SDK for Python directly -- it does not shell out to `docker` or require `docker-compose`. The `DockerManager` class handles:

- Creating and removing Docker networks
- Building images from Dockerfiles
- Running containers with port mappings, volumes, and GPU passthrough
- Streaming logs
- Listing and stopping containers by the `llmstack` label

All containers are labeled with `llmstack=true` and named with the `llmstack-` prefix for easy identification.

## Gateway Architecture

The gateway is a FastAPI application that runs inside its own Docker container. It is not a simple proxy -- it implements a full middleware stack:

```
Request
  │
  ├── LoggingMiddleware      (structured JSON logs, X-Request-ID)
  ├── AuthMiddleware         (API key validation)
  ├── RateLimitMiddleware    (token bucket via Redis)
  ├── MetricsMiddleware      (Prometheus counters/histograms)
  │
  └── Route Handler
       ├── /v1/chat/completions  → Cache check → Circuit breaker → Inference
       ├── /v1/embeddings        → Inference
       ├── /v1/models            → Inference
       ├── /v1/rag/ingest        → Chunk → Embed → Qdrant
       ├── /v1/rag/query         → Embed → Qdrant → LLM
       ├── /healthz              → Aggregate health
       └── /metrics              → Prometheus exposition
```

See the [Gateway Guide](gateway.md) for detailed documentation of each feature.

## Configuration Flow

```
llmstack.yaml
    │
    ▼
StackConfig (Pydantic v2)
    │
    ├── ModelSpec          → Inference service config
    ├── EmbeddingSpec      → Embedding service config
    ├── VectorDBConfig     → Qdrant config
    ├── CacheConfig        → Redis config
    ├── GatewayConfig      → Gateway env vars
    ├── ObserveConfig      → Prometheus/Grafana config
    └── DockerConfig       → Network, GPU, data paths
```

The Pydantic models provide validation, default values, and type safety. Invalid configuration is caught early with clear error messages.
