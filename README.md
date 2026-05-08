<p align="center">
  <h1 align="center">llmstack</h1>
  <p align="center"><strong>Ask your files anything. Locally. Privately. One command.</strong></p>
  <p align="center">Chat with your code, PDFs, and logs using a local LLM. Plus: smart model routing for your full LLM stack.</p>
</p>

<p align="center">
  <a href="https://pypi.org/project/llmstack-cli/"><img src="https://img.shields.io/pypi/v/llmstack-cli?color=blue" alt="PyPI"></a>
  <a href="https://github.com/mara-werils/llmstack/actions/workflows/ci.yml"><img src="https://github.com/mara-werils/llmstack/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://github.com/mara-werils/llmstack/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-green" alt="License"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.11+-blue" alt="Python"></a>
  <a href="https://github.com/mara-werils/llmstack/stargazers"><img src="https://img.shields.io/github/stars/mara-werils/llmstack?style=social" alt="Stars"></a>
</p>

---

## Ask Your Files Anything

```bash
llmstack ask "How does authentication work?" ./src/
```

```
Searching ./src/ (47 files, 12,841 lines)
Embedding chunks... done (0.8s)

Based on the source code, authentication works as follows:

1. API key authentication is the primary method. Keys are validated
   in the FastAPI gateway middleware (gateway/auth.py:23-45).

2. Each request must include an Authorization: Bearer <key> header.
   The middleware extracts the key, checks it against the stored
   keys in llmstack.yaml, and rejects invalid keys with 401.

3. Rate limiting is tied to the API key — each key gets its own
   token bucket tracked in Redis.

Sources:
  - src/gateway/auth.py (lines 23-45)
  - src/gateway/middleware.py (lines 12-38)
  - src/config/schema.py (lines 89-102)
```

No API keys. No cloud. No Docker. Just you, your files, and a local LLM.

**Supports:** PDF, DOCX, Markdown, Python, JavaScript, TypeScript, Go, Rust, Java, JSON, YAML, CSV, HTML, logs, and 20+ file types.

```bash
# Ask about any file or directory
llmstack ask "Summarize the key findings" report.pdf
llmstack ask "What went wrong?" error.log
llmstack ask "Find security vulnerabilities" ./src/ --model llama3.1:70b

# Pipe from stdin
cat contract.pdf | llmstack ask "Are there any risks?"
```

## Quick Start

```bash
pip install llmstack-cli
llmstack ask "summarize this" report.pdf
```

That's it. If [Ollama](https://ollama.com) is running, it works. No Docker, no Redis, no config file, no server to start.

> **[Full Documentation](https://mara-werils.github.io/llmstack)** | **[Ask Guide](https://mara-werils.github.io/llmstack/guide/ask/)** | **[Examples](examples/)** | **[Roadmap](ROADMAP.md)** | **[Contributing](CONTRIBUTING.md)**

---

## Full Stack Mode: Smart Model Routing

Want more than file Q&A? llmstack also runs a full local LLM stack with **smart model routing** -- it automatically sends each query to the smallest model that can handle it.

**Stop running 70B for "Hello".**

```bash
llmstack init --preset router
llmstack up
```

You now have **7 services** running locally: multi-model inference, embeddings, vector DB, cache, API gateway with smart routing, Prometheus, and Grafana -- plus a **built-in Web UI** at `http://localhost:8000`.

Real results from a GCP `e2-standard-4` (CPU-only, 16GB RAM, no GPU):

| Your query | Tier | Model routed to | Latency |
|---|---|---|---|
| "Hello!" | Simple | `llama3.2:1b` | **1.6s** |
| "Thanks!" | Simple | `llama3.2:1b` | **2.9s** |
| "What is 2+2?" | Simple | `llama3.2:1b` | **5.9s** |
| "Write binary search in Python" | Medium | `llama3.2:3b` | **52.2s** |

**71% of requests routed to the small model.** The 1b model generates at 8.5 tokens/sec -- 1.8x faster than the 3b model at 4.7 tokens/sec. On GPU hardware, the gap is much wider.

The router inspects every incoming request -- message length, vocabulary complexity, domain signals, conversation depth -- and picks the right model in under 2ms. You see which model handled your query in the `X-Model-Router` and `X-Query-Tier` response headers.

No other local LLM tool does this.

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

Classification takes **< 2ms** -- it's a lightweight heuristic, not another LLM call. Zero overhead on your actual inference.

You can override routing for any request:

```bash
# Force a specific model
curl http://localhost:8000/v1/chat/completions \
  -H "X-Model-Override: llama3.1:70b" \
  -d '{"model":"auto","messages":[{"role":"user","content":"Hello!"}]}'
```

## API Usage

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

Works with **any OpenAI-compatible client**: LangChain, LlamaIndex, Vercel AI SDK, openai-python -- just set `model: "auto"` or let the gateway decide.

## Benchmarks

Measured on GCP `e2-standard-4` (4 vCPU, 16GB RAM, CPU-only -- no GPU). Two models: `llama3.2:1b` + `llama3.2:3b`.

### Routing distribution (real traffic)

| Metric | Value |
|---|---|
| Requests routed to 1b (simple) | **71.4%** |
| Requests routed to 3b (medium) | **28.6%** |
| Avg latency -- 1b (simple queries) | **2.9s** |
| Avg latency -- 3b (complex queries) | **52.2s** |
| Estimated compute savings | **71%** |

### Token generation speed (CPU-only)

| Model | Tokens/sec | Avg latency | Used for |
|---|---|---|---|
| `llama3.2:1b` | **8.5 t/s** | 2.9s | Greetings, simple Q&A, quick lookups |
| `llama3.2:3b` | **4.7 t/s** | 52.2s | Explanations, code, complex reasoning |

On GPU hardware (NVIDIA RTX 3090 / Apple M-series), expect 10-50x faster speeds -- the routing advantage grows proportionally.

### What this means

Without routing, every query hits the 3b model. With routing, 71% of queries use the 1b model at 1.8x the token speed. The 3b model is reserved for queries that actually need it.

Run your own benchmarks:

```bash
llmstack bench --model llama3.2:1b --model llama3.2:3b
```

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

**`llmstack ask` -- Chat With Your Files** -- ask questions about local files and directories using a local LLM. PDF, code, logs, docs -- one command, no setup. Streaming answers with source citations.

**Smart Model Router** -- routes queries to the right-sized model automatically. The only local LLM tool that does this.

**RAG Pipeline** -- ingest documents, chunk, embed, store in Qdrant. Query with semantic search + augmented generation. Source citations included. Streaming supported.

**Semantic Response Cache** -- SHA-256 keyed, Redis-backed. Identical queries return in < 1ms. Only caches deterministic requests (temperature <= 0.1). `X-Cache: HIT/MISS` headers.

**Token Bucket Rate Limiter** -- Redis + atomic Lua scripts. Per-API-key with IP fallback. Configurable: `100/min`, `10/sec`, `3600/hour`. In-memory fallback if Redis goes down.

**Circuit Breaker** -- three-state machine (CLOSED / OPEN / HALF_OPEN) with exponential backoff. Prevents cascading failures. Fail-fast 503 when inference is down.

**Observability** -- Prometheus + pre-built Grafana dashboard. Request rate, p50/p99 latency, token throughput, error rate, cache hit ratio, circuit breaker state, routing distribution.

**Built-in Web UI** -- chat, RAG document management, health dashboard, settings. Zero extra install.

**Python SDK** -- sync `Client`, async `AsyncClient`. Chat, streaming, embeddings, RAG ingest/query.

**Plugin System** -- extend with new backends via pip. `pip install llmstack-cli-plugin-chromadb` and go.

## Web UI

Open `http://localhost:8000` after `llmstack up`:

- **Chat** -- streaming responses, model selector (or "auto" for smart routing), conversation history
- **RAG** -- paste or upload documents, query your knowledge base with citations
- **Dashboard** -- health status, cache hit rate, circuit breaker state, routing distribution per model
- **Settings** -- API key, default model, temperature, persisted in browser

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

!!! note
    `llmstack ask` does not require a config file. It uses Ollama directly with sensible defaults. The config above is only needed for the full stack mode (`llmstack up`).

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
| **Chat with local files** | **Yes** | No | No | No | No |
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

The first two rows are the ones that matter. No other local LLM tool lets you chat with your files from the terminal, and no other tool automatically routes queries to the right model.

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
| `llmstack ask <question> [path]` | Ask questions about local files using a local LLM |
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

- **v0.6** -- Multi-node deployment, auto-scaling, TLS, OAuth2
- **v0.7** -- Prompt versioning, evaluation framework, TypeScript SDK
- **v1.0** -- Kubernetes Helm chart, plugin marketplace, 90%+ test coverage

## Requirements

- Python 3.11+
- **`llmstack ask`**: Just [Ollama](https://ollama.com) running locally. No Docker needed.
- **Full stack mode** (`llmstack up`): Docker

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.

## License

Apache-2.0
