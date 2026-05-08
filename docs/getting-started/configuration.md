# Configuration Reference

llmstack uses a single YAML file -- `llmstack.yaml` -- to define the entire stack. This page documents every field and its default value.

## Full Example

```yaml
version: "1"

models:
  chat:
    name: llama3.2
    backend: auto
    quantization: null
    gpu_layers: -1
    context_length: 8192
    extra_args: {}
  embeddings:
    name: bge-m3
    backend: auto
    dimensions: null

services:
  vectors:
    provider: qdrant
    port: 6333
    storage_path: ./data/vectors
  cache:
    provider: redis
    port: 6379
    max_memory: 256mb

gateway:
  port: 8000
  auth: api_key
  api_keys: []
  rate_limit: 100/min
  cors: ["*"]
  request_timeout: 120

observe:
  metrics: true
  dashboard_port: 8080
  retention: 7d

docker:
  network: llmstack_net
  gpu: auto
  data_dir: ~/.llmstack/data
```

## Section Reference

### `version`

| Field | Type | Default | Description |
|---|---|---|---|
| `version` | string | `"1"` | Config schema version. Currently only `"1"` is supported. |

### `models.chat`

Configuration for the chat/completion model.

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | string | `"llama3.2"` | Model name. For Ollama, this is the model tag. For vLLM, this is the HuggingFace model ID. |
| `backend` | `"auto"` \| `"ollama"` \| `"vllm"` | `"auto"` | Inference backend. `auto` selects based on hardware detection: vLLM for NVIDIA GPUs with 16GB+ VRAM, Ollama otherwise. |
| `quantization` | string \| null | `null` | Quantization method for vLLM (e.g., `"awq"`, `"gptq"`). Ignored by Ollama. |
| `gpu_layers` | integer | `-1` | Number of layers to offload to GPU. `-1` means all layers. |
| `context_length` | integer | `8192` | Maximum context window size in tokens. |
| `extra_args` | object | `{}` | Additional arguments passed to the inference backend. |

### `models.embeddings`

Configuration for the embedding model.

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | string | `"bge-m3"` | Embedding model name. |
| `backend` | `"auto"` \| `"tei"` | `"auto"` | Embedding backend. `auto` uses TEI (Text Embeddings Inference) when available, falls back to Ollama. |
| `dimensions` | integer \| null | `null` | Output embedding dimensions. `null` uses the model's default. |

### `services.vectors`

Vector database configuration.

| Field | Type | Default | Description |
|---|---|---|---|
| `provider` | `"qdrant"` | `"qdrant"` | Vector database provider. Currently only Qdrant is supported as a built-in. Additional providers can be added via [plugins](../guide/plugins.md). |
| `port` | integer | `6333` | Host port for the Qdrant HTTP API. The gRPC port is automatically set to `port + 1`. |
| `storage_path` | string | `"./data/vectors"` | Path for persistent vector storage. |

### `services.cache`

Cache and rate limiter configuration.

| Field | Type | Default | Description |
|---|---|---|---|
| `provider` | `"redis"` | `"redis"` | Cache provider. Currently only Redis is supported. |
| `port` | integer | `6379` | Host port for Redis. |
| `max_memory` | string | `"256mb"` | Maximum memory Redis will use. Uses LRU eviction when the limit is reached. |

### `gateway`

API gateway configuration.

| Field | Type | Default | Description |
|---|---|---|---|
| `port` | integer | `8000` | Host port for the gateway API. |
| `auth` | `"none"` \| `"api_key"` | `"api_key"` | Authentication mode. `api_key` requires a Bearer token in the `Authorization` header. |
| `api_keys` | list of strings | `[]` | List of valid API keys. If empty and `auth` is `api_key`, a key is auto-generated on first `llmstack up`. |
| `rate_limit` | string | `"100/min"` | Rate limit specification. Format: `<count>/<period>` where period is `sec`, `min`, or `hour`. Examples: `10/sec`, `100/min`, `3600/hour`. |
| `cors` | list of strings | `["*"]` | Allowed CORS origins. Use `["*"]` to allow all origins. |
| `request_timeout` | integer | `120` | Maximum request duration in seconds before the gateway times out. |

### `observe`

Observability configuration.

| Field | Type | Default | Description |
|---|---|---|---|
| `metrics` | boolean | `true` | Enable Prometheus + Grafana. Set to `false` to skip observability containers. |
| `dashboard_port` | integer | `8080` | Host port for the Grafana dashboard. |
| `retention` | string | `"7d"` | Prometheus data retention period. |

### `docker`

Docker-level configuration.

| Field | Type | Default | Description |
|---|---|---|---|
| `network` | string | `"llmstack_net"` | Docker network name for inter-service communication. |
| `gpu` | `"auto"` \| `"true"` \| `"false"` | `"auto"` | GPU passthrough. `auto` enables GPU if a compatible GPU is detected. |
| `data_dir` | string | `"~/.llmstack/data"` | Base directory for persistent data (model cache, vector storage). |

## Presets

Presets provide sensible defaults for common use cases. Use them with `llmstack init --preset <name>`.

### `chat`

Minimal setup for chatbot applications. Includes inference, cache, and gateway. No vector DB or embeddings.

### `rag`

Full RAG setup. Includes everything in `chat` plus Qdrant and TEI for document ingestion and semantic search.

### `agent`

Heavy-duty setup for agent workflows. Uses a 70B parameter model, 16K context length, and extended timeouts.

## Environment Variable Overrides

The gateway reads several environment variables at runtime. These are set automatically by `llmstack up` based on your config, but can be overridden when running the gateway standalone:

| Variable | Description |
|---|---|
| `LLMSTACK_INFERENCE_URL` | OpenAI-compatible inference URL |
| `LLMSTACK_EMBEDDINGS_URL` | Embeddings service URL |
| `LLMSTACK_QDRANT_URL` | Qdrant HTTP URL |
| `LLMSTACK_REDIS_URL` | Redis connection URL |
| `LLMSTACK_API_KEYS` | Comma-separated API keys |
| `LLMSTACK_CORS_ORIGINS` | Comma-separated CORS origins |
| `LLMSTACK_REQUEST_TIMEOUT` | Request timeout in seconds |
| `LLMSTACK_RATE_LIMIT` | Rate limit (e.g., `100/min`) |
