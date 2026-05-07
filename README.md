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
# Chat completion (OpenAI-compatible)
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"llama3.2","messages":[{"role":"user","content":"Hello!"}]}'

# Ingest a document for RAG
curl http://localhost:8000/v1/rag/ingest \
  -H "Authorization: Bearer YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{"text":"LLMStack is an open-source tool for...","source":"docs.txt"}'

# Query with RAG
curl http://localhost:8000/v1/rag/query \
  -H "Authorization: Bearer YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{"question":"What is LLMStack?"}'
```

Works with **any OpenAI-compatible client**: LangChain, LlamaIndex, Vercel AI SDK, openai-python.

## Who is this for?

- **AI app developers** who want local inference + RAG without Docker boilerplate
- **Teams** who need an OpenAI-compatible API backed by local models
- **Hobbyists** running LLMs locally who want vector search, caching, and monitoring out of the box
- **Anyone** tired of writing 200+ lines of docker-compose.yml every time

## Architecture

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
         +----v--+ +--v---+ +v-----+ +v----+ +v-----------+
         |Qdrant | |Redis | |Ollama| | TEI | |  Gateway    |
         |Vector | |Cache | | or   | |Embed| |  FastAPI    |
         |  DB   | |+ Rate| | vLLM | |     | |  + RAG      |
         |       | | Limit| |      | |     | |  + Cache    |
         +-------+ +------+ +------+ +-----+ |  + Breaker  |
              :6333   :6379   :11434   :8002  |  + Metrics  |
                                              +-----+------+
                                                    |:8000
                                              +-----v------+
                                              | Prometheus  |
                                              |  + Grafana  |
                                              +------------+
                                                    :8080
```

| Layer | Service | What it does | Port |
|-------|---------|-------------|------|
| Inference | Ollama / vLLM (auto) | LLM chat completions | 11434 |
| Embeddings | TEI / Ollama (auto) | Text embeddings for RAG | 8002 |
| Vector DB | Qdrant | Document storage + semantic search | 6333 |
| Cache | Redis | Response cache + rate limit state | 6379 |
| API Gateway | FastAPI | Routing, auth, caching, RAG, circuit breaker | 8000 |
| Dashboard | Grafana + Prometheus | Request rate, latency, tokens, errors | 8080 |

## Gateway Features

The gateway is not a simple proxy — it's a production-grade API layer:

### Semantic Response Cache (Redis)
```
Request → SHA-256(model + messages) → Redis lookup
  HIT  → Return cached response (< 1ms)
  MISS → Forward to inference → Cache result → Return
```
- Only caches deterministic requests (temperature <= 0.1)
- TTL-based expiration (default: 1 hour)
- `X-Cache: HIT/MISS` response headers
- Cache stats in `/healthz`

### Token Bucket Rate Limiter (Redis + Lua)
```
Request → Extract API key/IP → Redis EVALSHA (atomic Lua) → Allow/Reject
```
- Configurable: `100/min`, `10/sec`, `3600/hour`
- Per-API-key rate limiting with IP fallback
- Atomic Lua script prevents race conditions
- In-memory fallback if Redis is unavailable
- Standard `X-RateLimit-*` and `Retry-After` headers

### Circuit Breaker (Inference Resilience)
```
CLOSED ──[5 failures]──> OPEN ──[timeout]──> HALF_OPEN ──[success]──> CLOSED
                           |                      |
                           └──[reject fast]       └──[failure]──> OPEN (backoff x2)
```
- Prevents cascading failures when inference is down
- Exponential backoff on recovery timeout
- Fail-fast with `503 Service Unavailable`
- Metrics: state, failure count, rejections, time in state

### RAG Pipeline (Qdrant + Embeddings)
```
Ingest: Document → Chunk (512 words, 64 overlap) → Embed → Qdrant
Query:  Question → Embed → Qdrant search → Build context → LLM generate
```
- `POST /v1/rag/ingest` — chunk, embed, and store documents
- `POST /v1/rag/query` — semantic search + augmented generation
- Source citations in responses
- Streaming support via SSE
- Deterministic chunk IDs (deduplication)

### Structured Logging
```json
{"ts":"2026-05-07T14:23:01","level":"INFO","msg":"POST /v1/chat/completions 200 1234.5ms","request_id":"a1b2c3d4","method":"POST","path":"/v1/chat/completions","status":200,"duration_ms":1234.5,"client_ip":"10.0.0.1"}
```
- `X-Request-ID` correlation headers
- JSON structured output for log aggregation
- Configurable level and format

## How it works

```bash
llmstack init       # Detects hardware, generates llmstack.yaml
                    # Picks optimal backend: vLLM for NVIDIA 16GB+, Ollama otherwise

llmstack up         # Boots services in order with health checks:
                    # Qdrant -> Redis -> Inference -> Embeddings -> Gateway -> Metrics

llmstack status     # Shows health of all running services
llmstack chat       # Interactive terminal chat with streaming
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

## API Reference

### OpenAI-compatible endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Chat completion (streaming + non-streaming) |
| `/v1/embeddings` | POST | Text embeddings |
| `/v1/models` | GET | List available models |

### RAG endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/rag/ingest` | POST | Ingest a document (chunk + embed + store) |
| `/v1/rag/query` | POST | Query with retrieval-augmented generation |
| `/v1/rag/documents/{source}` | DELETE | Delete documents by source |
| `/v1/rag/status` | GET | Collection statistics |

### System endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/healthz` | GET | Health check with circuit breaker + cache stats |
| `/metrics` | GET | Prometheus metrics |

## Interactive Chat

```bash
llmstack chat
```

```
LLMStack Chat — model: llama3.2
Type 'exit' or Ctrl+C to quit. '/clear' to reset conversation.

You: What is quantum computing?
Assistant: Quantum computing uses quantum mechanical phenomena like
superposition and entanglement to process information...

You: /clear
Conversation cleared.
```

## Export to Docker Compose

```bash
llmstack export
# Exported 7 services to docker-compose.yml
# Run with: docker compose up -d
```

Share the generated file with your team — no llmstack dependency required.

## Use the API

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="YOUR_KEY")

# Chat completion
response = client.chat.completions.create(
    model="llama3.2",
    messages=[{"role": "user", "content": "Explain quantum computing"}]
)

# Embeddings
embeddings = client.embeddings.create(
    model="bge-m3",
    input=["Hello world"]
)
```

```python
import httpx

# RAG: Ingest documents
httpx.post("http://localhost:8000/v1/rag/ingest", json={
    "text": open("whitepaper.txt").read(),
    "source": "whitepaper.txt",
}, headers={"Authorization": "Bearer YOUR_KEY"})

# RAG: Query
response = httpx.post("http://localhost:8000/v1/rag/query", json={
    "question": "What are the key findings?",
    "top_k": 5,
}, headers={"Authorization": "Bearer YOUR_KEY"})

print(response.json()["answer"])
print(response.json()["sources"])
```

## CLI

| Command | Description |
|---------|-------------|
| `llmstack init [--preset]` | Create config with smart defaults |
| `llmstack up [--attach]` | Start all services |
| `llmstack down [--volumes]` | Stop and clean up |
| `llmstack status` | Health check all services |
| `llmstack chat [--model]` | Interactive terminal chat |
| `llmstack export [--output]` | Generate docker-compose.yml |
| `llmstack logs <service>` | Stream service logs |
| `llmstack doctor` | Diagnose system issues |

## Observability

When `observe.metrics: true`, llmstack boots Prometheus + Grafana with a pre-built dashboard:

- **Request rate** per endpoint
- **Latency** p50 / p99 histograms
- **Token throughput** (input + output)
- **Error rate** (4xx / 5xx)
- **Cache hit rate**
- **Circuit breaker state**
- **Rate limit rejections**

Access at `http://localhost:8080` (login: admin / llmstack)

## Comparison

| | llmstack | Ollama | LocalAI | AnythingLLM | LiteLLM |
|---|---|---|---|---|---|
| One-command full stack | **Yes** | No | No | Partial | No |
| Built-in RAG pipeline | **Yes** | No | No | Bundled | No |
| Response caching | **Yes** | No | No | No | No |
| Circuit breaker | **Yes** | No | No | No | No |
| Rate limiting (Redis) | **Yes** | No | No | Yes | Yes |
| Auto hardware detection | **Yes** | No | No | No | No |
| OpenAI-compatible API | **Yes** | Yes | Yes | No | Yes |
| Built-in vector DB | **Yes** | No | No | Bundled | No |
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
- **Gateway**: [FastAPI](https://fastapi.tiangolo.com/) + Redis + Qdrant
- **Containers**: [Docker SDK for Python](https://docker-py.readthedocs.io/)
- **Cache**: Redis with semantic hashing
- **Rate Limiting**: Token bucket with Redis Lua scripts
- **Resilience**: Circuit breaker with exponential backoff
- **Metrics**: Prometheus + Grafana

## Requirements

- Python 3.11+
- Docker

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.

## License

Apache-2.0
