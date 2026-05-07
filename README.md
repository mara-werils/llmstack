<p align="center">
  <h1 align="center">llmstack</h1>
  <p align="center"><strong>One command. Full LLM stack. Zero config.</strong></p>
  <p align="center">Stop wiring Docker containers. Start building AI apps.</p>
</p>

<p align="center">
  <a href="https://pypi.org/project/llmstack-cli/"><img src="https://img.shields.io/pypi/v/llmstack-cli?color=blue" alt="PyPI"></a>
  <a href="https://github.com/mara-werils/llmstack/actions/workflows/ci.yml"><img src="https://github.com/mara-werils/llmstack/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://github.com/mara-werils/llmstack/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-green" alt="License"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.11+-blue" alt="Python"></a>
  <a href="https://github.com/mara-werils/llmstack/stargazers"><img src="https://img.shields.io/github/stars/mara-werils/llmstack?style=social" alt="Stars"></a>
</p>

---

<p align="center">
  <img src="demo.gif" alt="llmstack demo" width="800">
</p>

## Quick Start

```bash
pip install llmstack-cli
llmstack init --preset rag
llmstack up
```

That's it. You now have **7 services** running: inference, embeddings, vector DB, cache, API gateway, Prometheus, and Grafana.

```bash
# Test it immediately
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"llama3.2","messages":[{"role":"user","content":"Hello!"}]}'
```

Works with **any OpenAI-compatible client**: LangChain, LlamaIndex, Vercel AI SDK, openai-python.

## Who is this for?

- **AI app developers** who want local inference without Docker boilerplate
- **Teams** who need an OpenAI-compatible API backed by local models
- **Hobbyists** running LLMs locally who want vector search, caching, and monitoring out of the box
- **Anyone** tired of writing 200+ lines of docker-compose.yml every time

## What you get

```
                         llmstack up
                              |
                    +---------v----------+
                    |   Hardware Detect   |
                    |  NVIDIA / Apple / CPU|
                    +---------+----------+
                              |
              +-------+-------+-------+-------+
              |       |       |       |       |
         +----v--+ +--v---+ +v-----+ +v----+ +v--------+
         |Qdrant | |Redis | |Ollama| | TEI | | Gateway  |
         |Vector | |Cache | | or   | |Embed| | FastAPI  |
         |  DB   | |      | | vLLM | |     | | OpenAI   |
         +-------+ +------+ +------+ +-----+ |compatible|
              :6333   :6379   :11434   :8002  +----+-----+
                                                   |:8000
                                              +----v-----+
                                              |Prometheus |
                                              | + Grafana |
                                              +----------+
                                                   :8080
```

| Layer | Service | Default | Port |
|-------|---------|---------|------|
| Inference | Ollama / vLLM (auto) | llama3.2 | 11434 |
| Embeddings | TEI / Ollama (auto) | bge-m3 | 8002 |
| Vector DB | Qdrant | - | 6333 |
| Cache | Redis | 256MB LRU | 6379 |
| API Gateway | FastAPI (OpenAI-compatible) | auth + rate limit | 8000 |
| Dashboard | Grafana + Prometheus | pre-built panels | 8080 |

## How it works

```bash
llmstack init       # Detects hardware, generates llmstack.yaml
                    # Picks optimal backend: vLLM for NVIDIA 16GB+, Ollama otherwise

llmstack up         # Boots services in order with health checks:
                    # Qdrant -> Redis -> Inference -> Embeddings -> Gateway -> Metrics

llmstack status     # Shows health of all running services
llmstack logs ollama # Stream inference logs
llmstack down       # Stops everything
```

## Auto hardware detection

| Your hardware | Backend | Why |
|---|---|---|
| NVIDIA GPU 16GB+ VRAM | vLLM | Max throughput, PagedAttention |
| NVIDIA GPU <16GB | Ollama | Lower memory overhead |
| Apple Silicon (M1-M4) | Ollama | Metal acceleration |
| CPU only | Ollama | GGUF quantized models |

## Presets

```bash
llmstack init --preset chat    # Minimal: inference + cache + gateway
llmstack init --preset rag     # + Qdrant + embeddings for RAG apps
llmstack init --preset agent   # 70B model + 16K context + longer timeouts
```

## Configuration

One file: `llmstack.yaml`

```yaml
version: "1"

models:
  chat:
    name: llama3.2
    backend: auto              # auto | ollama | vllm
    context_length: 8192
  embeddings:
    name: bge-m3

services:
  vectors:
    provider: qdrant
    port: 6333
  cache:
    provider: redis
    max_memory: 256mb

gateway:
  port: 8000
  auth: api_key
  rate_limit: 100/min
  cors: ["*"]

observe:
  metrics: true
  dashboard_port: 8080
```

## Use the API

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="YOUR_KEY")
response = client.chat.completions.create(
    model="llama3.2",
    messages=[{"role": "user", "content": "Explain quantum computing"}]
)
```

## CLI

| Command | Description |
|---------|-------------|
| `llmstack init [--preset]` | Create config with smart defaults |
| `llmstack up [--attach]` | Start all services |
| `llmstack down [--volumes]` | Stop and clean up |
| `llmstack status` | Health check all services |
| `llmstack logs <service>` | Stream service logs |
| `llmstack doctor` | Diagnose system issues |

## Observability

When `observe.metrics: true`, llmstack boots Prometheus + Grafana with a pre-built dashboard:

- **Request rate** per endpoint
- **Latency** p50 / p99 histograms
- **Token throughput** (input + output)
- **Error rate** (4xx / 5xx)
- **Service health** (up/down)

Access at `http://localhost:8080` (login: admin / llmstack)

## Why not just Docker Compose?

Here's what llmstack replaces:

```yaml
# Without llmstack: ~200 lines of docker-compose.yml
# You have to configure each service, write health checks,
# set up networking, manage GPU passthrough, create Prometheus
# scrape configs, provision Grafana dashboards...

# With llmstack:
llmstack init && llmstack up
```

## Comparison

| | llmstack | Ollama | LocalAI | AnythingLLM | LiteLLM |
|---|---|---|---|---|---|
| One-command full stack | **Yes** | No (inference only) | No | Partial | No (proxy only) |
| Auto hardware detection | **Yes** | No | No | No | No |
| OpenAI-compatible API | **Yes** | Yes | Yes | No | Yes |
| Built-in vector DB | **Yes** | No | No | Bundled | No |
| Built-in embeddings | **Yes** | No | No | Bundled | No |
| Caching (Redis) | **Yes** | No | No | No | No |
| Auth + rate limiting | **Yes** | No | No | Yes | Yes |
| Observability dashboard | **Yes** | No | Partial | No | Partial |
| Plugin ecosystem | **Yes** | No | No | No | No |

## Plugins

Extend llmstack with new backends via pip:

```bash
pip install llmstack-cli-plugin-chromadb
# Now: vectors.provider: chromadb in llmstack.yaml
```

Create your own: implement `ServiceBase`, register via entry_points. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Tech stack

- **CLI**: [Typer](https://typer.tiangolo.com/) + [Rich](https://rich.readthedocs.io/)
- **Config**: [Pydantic v2](https://docs.pydantic.dev/)
- **Gateway**: [FastAPI](https://fastapi.tiangolo.com/)
- **Containers**: [Docker SDK for Python](https://docker-py.readthedocs.io/)
- **Metrics**: Prometheus + Grafana

## Requirements

- Python 3.11+
- Docker

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.

## License

Apache-2.0
