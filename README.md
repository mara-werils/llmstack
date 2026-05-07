<p align="center">
  <h1 align="center">llmstack</h1>
  <p align="center"><strong>One command. Full LLM stack. Zero config.</strong></p>
</p>

<p align="center">
  <a href="https://pypi.org/project/llmstack/"><img src="https://img.shields.io/pypi/v/llmstack?color=blue" alt="PyPI"></a>
  <a href="https://github.com/mara-werils/llmstack/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-green" alt="License"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.11+-blue" alt="Python"></a>
</p>

---

**llmstack** spins up a production-grade LLM stack locally with a single command. It auto-detects your hardware, picks the optimal inference backend, and wires everything together: inference, vector database, embeddings, caching, and an OpenAI-compatible API gateway.

```
pip install llmstack
llmstack init
llmstack up
```

That's it. You now have a full LLM API running locally.

## What you get

| Layer | Service | Default |
|-------|---------|---------|
| Inference | Ollama / vLLM (auto-detected) | llama3.2 |
| Vector DB | Qdrant | localhost:6333 |
| Cache | Redis | localhost:6379 |
| API Gateway | OpenAI-compatible | localhost:8000 |
| Dashboard | Grafana + Prometheus | localhost:8080 |

## How it works

1. `llmstack init` detects your hardware (GPU, RAM) and generates `llmstack.yaml`
2. `llmstack up` boots all services via Docker in the correct order with health checks
3. You get an OpenAI-compatible API that works with LangChain, LlamaIndex, Vercel AI SDK, or plain `curl`

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer YOUR_KEY" \
  -d '{"model":"llama3.2","messages":[{"role":"user","content":"Hello!"}]}'
```

## Auto hardware detection

| Your hardware | What llmstack picks |
|---|---|
| NVIDIA GPU (16GB+ VRAM) | vLLM (max throughput) |
| NVIDIA GPU (<16GB) | Ollama (optimized) |
| Apple Silicon | Ollama (Metal acceleration) |
| CPU only | Ollama (CPU mode) |

## Presets

```bash
llmstack init --preset chat    # minimal: inference + cache + gateway
llmstack init --preset rag     # + vector DB + embeddings for RAG apps
llmstack init --preset agent   # large model (70B) + long context
```

## Configuration

Everything is in one file: `llmstack.yaml`

```yaml
version: "1"

models:
  chat:
    name: llama3.2
    backend: auto           # auto | ollama | vllm
    context_length: 8192
  embeddings:
    name: bge-m3

services:
  vectors:
    provider: qdrant
    port: 6333
  cache:
    provider: redis

gateway:
  port: 8000
  auth: api_key
  rate_limit: 100/min

observe:
  metrics: true
  dashboard_port: 8080
```

## CLI commands

| Command | Description |
|---------|-------------|
| `llmstack init` | Create llmstack.yaml with smart defaults |
| `llmstack up` | Start all services |
| `llmstack down` | Stop all services |
| `llmstack status` | Show health of all services |
| `llmstack logs <service>` | Stream logs from a service |
| `llmstack doctor` | Diagnose common issues |

## Extending with plugins

```bash
pip install llmstack-plugin-chromadb
# Now you can use: vectors.provider: chromadb
```

Plugins use standard Python entry points. See [docs/plugins.md](docs/plugins.md) for writing your own.

## Why llmstack?

| | llmstack | Ollama | Harbor | AnythingLLM | LiteLLM |
|---|---|---|---|---|---|
| One-command setup | Yes | Partial | No | Partial | No |
| Auto hardware detection | Yes | No | No | No | No |
| Built-in vector DB | Yes | No | Config | Bundled | No |
| OpenAI-compatible API | Yes | Yes | Varies | No | Yes |
| Caching layer | Yes | No | No | No | No |
| Observability | Yes | No | Partial | No | Partial |
| Plugin ecosystem | Yes | No | No | No | No |

## Requirements

- Python 3.11+
- Docker

## License

Apache-2.0
