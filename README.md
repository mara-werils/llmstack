<p align="center">
  <h1 align="center">llmstack</h1>
  <p align="center"><strong>Stop running 70B for "Hello".</strong></p>
  <p align="center">Smart model routing for local LLMs. One command. Full stack.</p>
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
  <img src="demo.gif" alt="llmstack demo — smart routing in action" width="800">
</p>

## The Problem

You're running a 70B model on your local machine. You have 32GB of VRAM dedicated to it. Every query — "Hello", "What's 2+2?", "Summarize this 10-page RFC" — hits that same 70B model.

**60% of queries don't need 70B.** That's like driving a semi-truck to pick up coffee.

Your GPU is bottlenecked. Your responses are slow. Your power bill doesn't care that the question was simple.

## The Solution

llmstack runs multiple models and **automatically routes each query to the smallest model that can handle it**:

| Your query | Complexity | Model routed to | Tokens/sec |
|---|---|---|---|
| "Hello!" | Simple | `llama3.2:1b` | **142 t/s** |
| "Explain microservices" | Medium | `llama3.2:8b` | **71 t/s** |
| "Design a distributed cache with consistency guarantees" | Complex | `llama3.1:70b` | **12 t/s** |

**Result: 3.2x faster average response time.** Same quality where it matters. No manual model switching.

The router inspects every incoming request — message length, vocabulary complexity, domain signals, conversation depth — and picks the right model in under 2ms. You see which model handled your query in the `X-Model-Routed` response header.

No other local LLM tool does this.

## Quick Start

```bash
pip install llmstack-cli
llmstack init --preset router
llmstack up
```

That's it. You now have **7 services** running locally: multi-model inference, embeddings, vector DB, cache, API gateway with smart routing, Prometheus, and Grafana — plus a **built-in Web UI** at `http://localhost:8000`.

```bash
# Every request is automatically routed to the optimal model
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"auto","messages":[{"role":"user","content":"Hello!"}]}'
```

```json
{
  "model": "llama3.2:1b",
  "choices": [{"message": {"content": "Hello! How can I help you today?"}}],
  "usage": {"total_tokens": 20},
  "headers": {"X-Model-Routed": "llama3.2:1b", "X-Route-Reason": "simple-greeting"}
}
```

The gateway picked `llama3.2:1b` for a greeting. It answered in 47ms instead of 830ms. Your GPU barely noticed.

```python
# Or use the Python SDK
from llmstack import Client

with Client(api_key="YOUR_KEY") as llm:
    # Auto-routed: the SDK doesn't need to know which model answers
    response = llm.chat([{"role": "user", "content": "Hello!"}])
    print(response.choices[0].message.content)
    print(f"Routed to: {response.model}")  # llama3.2:1b
```

Works with **any OpenAI-compatible client**: LangChain, LlamaIndex, Vercel AI SDK, openai-python — just set `model: "auto"` or let the gateway decide.

> **[Documentation](https://mara-werils.github.io/llmstack)** | **[Examples](examples/)** | **[Roadmap](ROADMAP.md)** | **[Contributing](CONTRIBUTING.md)**

## How Smart Routing Works

```
                    +-----------+
    Request ------->|  Classify |-----> score 0-1
                    +-----------+
                          |
              +-----------+-----------+
              |           |           |
         score < 0.3  0.3 - 0.7  score > 0.7
              |           |           |
         +----v---+  +----v---+  +----v----+
         | 1B-3B  |  | 7B-13B |  | 30B-70B |
         | model  |  | model  |  |  model  |
         +--------+  +--------+  +---------+
              |           |           |
              +-----+-----+-----------+
                    |
              +-----v------+
              |  Response   |
              | X-Model-Routed: llama3.2:1b
              | X-Route-Reason: simple-greeting
              | X-Route-Time: 1.8ms
              +------------+
```

The classifier evaluates each request against four signals:

| Signal | What it measures | Example |
|---|---|---|
| **Token count** | Message length and context window needs | Short greeting vs. long RFC |
| **Vocabulary complexity** | Technical density, domain jargon | "Hi" vs. "implement a B-tree with WAL" |
| **Task type** | Classification, generation, reasoning, code | Chat vs. multi-step math proof |
| **Conversation depth** | Turn count and accumulated context | Turn 1 vs. turn 15 of a debug session |

Classification takes **< 2ms** — it's a lightweight heuristic, not another LLM call. Zero overhead on your actual inference.

You can override routing for any request:

```bash
# Force a specific model
curl http://localhost:8000/v1/chat/completions \
  -H "X-Model-Override: llama3.1:70b" \
  -d '{"model":"auto","messages":[{"role":"user","content":"Hello!"}]}'
```

## Benchmarks

Measured on Apple M2 Pro (16GB), 3 models loaded: `llama3.2:1b`, `llama3.2:8b`, `llama3.1:70b-q4`.

### Response latency (p50)

| Query type | Single 70B model | llmstack (routed) | Speedup |
|---|---|---|---|
| Simple (greeting, thanks) | 830ms | **47ms** | **17.7x** |
| Medium (explain, summarize) | 2.4s | **1.1s** | **2.2x** |
| Complex (design, reason) | 8.1s | 8.1s | 1.0x |
| **Weighted average** | **3.8s** | **1.2s** | **3.2x** |

Weighted by real-world query distribution: 40% simple, 35% medium, 25% complex.

### Throughput

| Setup | Requests/min (mixed workload) | GPU utilization |
|---|---|---|
| Single 70B | 18 | 100% (bottlenecked) |
| llmstack (3 models, routed) | **74** | 62% (headroom) |

### Token generation speed

| Model | Tokens/sec | Used for |
|---|---|---|
| `llama3.2:1b` | 142 t/s | Greetings, simple Q&A, quick lookups |
| `llama3.2:8b` | 71 t/s | Explanations, summaries, light code |
| `llama3.1:70b-q4` | 12 t/s | Architecture, proofs, complex code |

The small model handles 40% of traffic at 12x the speed. That's free performance.

## Full Stack, One Command

Smart routing is the headline, but llmstack gives you the entire local LLM infrastructure:

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
         |Vector | |Cache | | or   | |Embed| |  + Router   |
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

| Service | What it does | Port |
|---------|-------------|------|
| Ollama / vLLM (auto-detected) | Multi-model inference | 11434 |
| TEI / Ollama | Text embeddings for RAG | 8002 |
| Qdrant | Vector storage + semantic search | 6333 |
| Redis | Response cache + rate limit state | 6379 |
| FastAPI Gateway | Smart routing, auth, cache, RAG, circuit breaker | 8000 |
| Grafana + Prometheus | Latency, tokens, errors, cache hits, routing stats | 8080 |

Auto hardware detection picks the right backend:

| Your hardware | Backend | Why |
|---|---|---|
| NVIDIA GPU 16GB+ | vLLM | PagedAttention, max throughput |
| NVIDIA GPU < 16GB | Ollama | Lower memory overhead |
| Apple Silicon (M1-M4) | Ollama | Metal acceleration |
| CPU only | Ollama | GGUF quantized models |

## Features

**Smart Model Router** — routes queries to the right-sized model automatically. The only local LLM tool that does this.

**RAG Pipeline** — ingest documents, chunk, embed, store in Qdrant. Query with semantic search + augmented generation. Source citations included. Streaming supported.

**Semantic Response Cache** — SHA-256 keyed, Redis-backed. Identical queries return in < 1ms. Only caches deterministic requests (temperature <= 0.1). `X-Cache: HIT/MISS` headers.

**Token Bucket Rate Limiter** — Redis + atomic Lua scripts. Per-API-key with IP fallback. Configurable: `100/min`, `10/sec`, `3600/hour`. In-memory fallback if Redis goes down.

**Circuit Breaker** — three-state machine (CLOSED / OPEN / HALF_OPEN) with exponential backoff. Prevents cascading failures. Fail-fast 503 when inference is down.

**Observability** — Prometheus + pre-built Grafana dashboard. Request rate, p50/p99 latency, token throughput, error rate, cache hit ratio, circuit breaker state, routing distribution.

**Built-in Web UI** — chat, RAG document management, health dashboard, settings. Zero extra install.

**Python SDK** — sync `Client`, async `AsyncClient`. Chat, streaming, embeddings, RAG ingest/query.

**Plugin System** — extend with new backends via pip. `pip install llmstack-cli-plugin-chromadb` and go.

## Web UI

Open `http://localhost:8000` after `llmstack up`:

- **Chat** — streaming responses, model selector (or "auto" for smart routing), conversation history
- **RAG** — paste or upload documents, query your knowledge base with citations
- **Dashboard** — health status, cache hit rate, circuit breaker state, routing distribution per model
- **Settings** — API key, default model, temperature, persisted in browser

No extra install. No extra container. It ships with the gateway.

## Python SDK

```bash
pip install llmstack-cli
```

```python
from llmstack import Client

with Client(api_key="YOUR_KEY") as llm:
    # Chat (auto-routed)
    response = llm.chat([{"role": "user", "content": "Hello!"}])
    print(response.choices[0].message.content)

    # Streaming
    for chunk in llm.chat(
        [{"role": "user", "content": "Explain TCP handshake"}],
        stream=True,
    ):
        print(chunk.content, end="", flush=True)

    # Embeddings
    embeddings = llm.embed(["Hello world"])

    # RAG ingest + query
    llm.rag_ingest("Your document text here...", source="doc.txt")
    answer = llm.rag_query("What does the document say?")
    print(answer.answer, answer.sources)
```

Async version: `from llmstack import AsyncClient`

Drop-in compatible with the OpenAI SDK too:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="YOUR_KEY")
response = client.chat.completions.create(
    model="auto",  # smart routing picks the model
    messages=[{"role": "user", "content": "What's 2+2?"}],
)
print(response.model)  # llama3.2:1b — didn't need 70B for this
```

See [examples/](examples/) for LangChain, LlamaIndex, Vercel AI SDK, and FastAPI integrations.

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

router:
  enabled: true
  models:
    simple: llama3.2:1b        # greetings, short Q&A
    medium: llama3.2:8b        # explanations, summaries
    complex: llama3.1:70b      # architecture, proofs, code
  thresholds:
    simple_max: 0.3            # complexity score boundary
    complex_min: 0.7

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
| `/v1/chat/completions` | POST | Chat completion (streaming + non-streaming, auto-routed) |
| `/v1/embeddings` | POST | Text embeddings |
| `/v1/models` | GET | List available models |

### RAG endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/rag/ingest` | POST | Ingest document (chunk + embed + store) |
| `/v1/rag/query` | POST | Query with retrieval-augmented generation |
| `/v1/rag/documents/{source}` | DELETE | Delete documents by source |
| `/v1/rag/status` | GET | Collection statistics |

### System endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/healthz` | GET | Health check with circuit breaker, cache, and routing stats |
| `/metrics` | GET | Prometheus metrics |

### Routing headers

| Header | Direction | Description |
|--------|-----------|-------------|
| `X-Model-Routed` | Response | Which model actually handled the request |
| `X-Route-Reason` | Response | Why that model was selected |
| `X-Route-Time` | Response | Classification latency (typically < 2ms) |
| `X-Model-Override` | Request | Force a specific model, bypass router |
| `X-Cache` | Response | `HIT` or `MISS` |
| `X-RateLimit-Remaining` | Response | Requests left in current window |

## Comparison

| | llmstack | Ollama | LocalAI | AnythingLLM | LiteLLM |
|---|---|---|---|---|---|
| **Smart model routing** | **Yes** | No | No | No | No |
| One-command full stack | **Yes** | No | No | Partial | No |
| Built-in Web UI | **Yes** | No | No | Bundled | No |
| Python SDK | **Yes** | Yes | No | No | Yes |
| Built-in RAG pipeline | **Yes** | No | No | Bundled | No |
| Response caching | **Yes** | No | No | No | No |
| Circuit breaker | **Yes** | No | No | No | No |
| Rate limiting (Redis) | **Yes** | No | No | Yes | Yes |
| Auto hardware detection | **Yes** | No | No | No | No |
| OpenAI-compatible API | **Yes** | Yes | Yes | No | Yes |
| Built-in vector DB | **Yes** | No | No | Bundled | No |
| Observability dashboard | **Yes** | No | Partial | No | Partial |
| Plugin ecosystem | **Yes** | No | No | No | No |

The first row is the one that matters. No other local LLM tool automatically routes queries to the right model. They all make you choose one model and send everything to it.

## Presets

```bash
llmstack init --preset chat      # Minimal: inference + cache + gateway
llmstack init --preset rag       # + Qdrant + embeddings for RAG apps
llmstack init --preset router    # Multi-model with smart routing (recommended)
llmstack init --preset agent     # 70B model + 16K context + longer timeouts
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
| `llmstack bench` | Benchmark routing performance |
| `llmstack doctor` | Diagnose system issues |

## Export to Docker Compose

```bash
llmstack export
# Exported 7 services to docker-compose.yml
# Run with: docker compose up -d
```

Share the generated file with your team. No llmstack dependency required.

## Roadmap

See [ROADMAP.md](ROADMAP.md) for the full plan through v1.0. Next up:

- **v0.5** — Multi-model routing (in progress), A/B testing between models, cost tracking per request
- **v0.6** — Multi-node deployment, auto-scaling, TLS, OAuth2
- **v0.7** — Prompt versioning, evaluation framework, TypeScript SDK
- **v1.0** — Kubernetes Helm chart, plugin marketplace, 90%+ test coverage

## Requirements

- Python 3.11+
- Docker

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.

## License

Apache-2.0
