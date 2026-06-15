<p align="center">
  <h1 align="center">llmstack</h1>
  <p align="center"><strong>Chat with any codebase. Locally. Privately. Free.</strong></p>
  <p align="center">The open-source alternative to Cursor and Copilot — runs entirely on your machine with Ollama.<br>Plus: smart routing, fine-tuning, AI agents, MCP server, and full observability.</p>
</p>

<p align="center">
  <a href="https://pypi.org/project/llmstack-cli/"><img src="https://img.shields.io/pypi/v/llmstack-cli?color=blue" alt="PyPI"></a>
  <a href="https://github.com/mara-werils/llmstack/actions/workflows/ci.yml"><img src="https://github.com/mara-werils/llmstack/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://github.com/mara-werils/llmstack/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-green" alt="License"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.11+-blue" alt="Python"></a>
  <a href="https://github.com/mara-werils/llmstack/stargazers"><img src="https://img.shields.io/github/stars/mara-werils/llmstack?style=social" alt="Stars"></a>
</p>

<p align="center">
  <a href="#ask-your-codebase-anything">Ask</a> &bull;
  <a href="#universal-gateway">Gateway</a> &bull;
  <a href="#smart-routing">Smart Routing</a> &bull;
  <a href="#ai-agents--mcp">Agents & MCP</a> &bull;
  <a href="#fine-tuning">Fine-tuning</a> &bull;
  <a href="#ai-observability">Observability</a>
</p>

---

<p align="center">
  <img src="assets/demo.gif" alt="llmstack ask demo" width="700">
</p>

---

```bash
brew install mara-werils/llmstack/llmstack   # or: pipx install llmstack-cli
llmstack ask -i ./src/                         # start chatting with your codebase
```

## Ask Your Codebase Anything

```bash
llmstack ask "How does authentication work?" ./src/
```

One command. No API keys. No cloud. No Docker. No $20/month subscription. Just Ollama + your files.

```
  llmstack ask  model=llama3.2  embeddings=nomic-embed-text

  Git: main (15 recent commits)
  Index cached: 847 chunks (0 files changed)
  Embeddings loaded from cache

  Answer:
  Authentication works through API key validation in the FastAPI gateway
  middleware. Each request must include an `Authorization: Bearer <key>`
  header. The middleware validates keys against the stored list in
  llmstack.yaml [src/gateway/middleware/auth.py:23-45]. Rate limiting
  is tied to the API key — each key gets its own token bucket tracked
  in Redis [src/gateway/middleware/rate_limit.py:12-38].

  ┌─────────────── Sources ───────────────┐
  │ File                  Lines   Score    │
  │ gateway/middleware/auth.py  23-45  0.0142  │
  │ gateway/middleware/rate_limit.py  12-38  0.0098  │
  │ config/schema.py      89-102  0.0076  │
  └───────────────────────────────────────┘
```

### Why this is better than Cursor/Copilot/Aider

| | llmstack ask | Cursor | Copilot | Aider | Khoj |
|---|:---:|:---:|:---:|:---:|:---:|
| **AST-aware code chunking** | **Yes** | Yes | - | Partial | No |
| **Hybrid search (BM25 + vector)** | **Yes** | ? | - | No | No |
| **Persistent incremental index** | **Yes** | Yes | - | No | Yes |
| **Git-aware context** | **Yes** | Yes | - | Yes | No |
| **Interactive conversation** | **Yes** | Yes | - | Yes | Yes |
| **20+ file types (PDF, DOCX, logs...)** | **Yes** | No | No | No | Yes |
| **100% local, 100% private** | **Yes** | No | No | No | Yes |
| **100% free, forever** | **Yes** | $20/mo | $10/mo | API costs | Free |
| **Zero config CLI** | **Yes** | IDE only | IDE only | Config needed | Server needed |

### Key features

**Persistent index** — first query indexes your project (~30s). Every query after that: **~0.1s**. Only re-embeds files that changed (SHA-256 hash diff).

**AST-aware chunking** — Python files split by functions and classes using the `ast` module. Large classes (>50 lines) split into individual methods. JS/TS/Go/Rust/Java use regex boundary detection. No more broken chunks mid-function.

**Hybrid search** — combines BM25 keyword matching (catches exact function names, error messages) with vector cosine similarity (catches meaning and intent). Merged via Reciprocal Rank Fusion. Better recall than either alone.

**Git-aware** — the LLM sees your current branch, recent commits, and changed files. Ask "what changed this week?" and get real answers.

**Interactive mode** — multi-turn conversation with your codebase. Context preserved across questions.

```bash
# Interactive conversation with your project
llmstack ask -i ./src/
# You: How does the cache work?
# Assistant: The cache uses Redis with SHA-256 keys...
# You: What happens when Redis goes down?
# Assistant: There's an in-memory fallback in rate_limit.py...

# Single question
llmstack ask "Find security vulnerabilities" ./src/ --model llama3.1:70b

# Ask about any file type
llmstack ask "Summarize the key findings" report.pdf
llmstack ask "What went wrong at 3am?" error.log
cat contract.pdf | llmstack ask "Are there any risks?"

# Skip cache for fresh re-index
llmstack ask "What's new?" ./src/ --no-cache

# Without git context
llmstack ask "Explain the architecture" ./src/ --no-git
```

**20+ file types:** Python, JavaScript, TypeScript, Go, Rust, Java, C/C++, Ruby, PHP, Swift, Kotlin, PDF, DOCX, Markdown, HTML, JSON, YAML, TOML, CSV, logs, and more.

**Private by default** — everything runs on your machine. `llmstack verify-private` audits your config and fails loudly if anything (a cloud provider, a webhook, a network-capable agent tool, wide-open CORS) could send code or prompts off the box:

```bash
llmstack verify-private          # human-readable report, non-zero exit if not private
llmstack verify-private --json   # machine-readable, for CI gates
```

**In your editor** — the [VS Code / OpenVSX extension](editors/vscode) brings Ask and Explain commands to VS Code, Cursor, and Windsurf, all routed through your local gateway.

---

## Quick Start

```bash
# Install (pick one)
brew install mara-werils/llmstack/llmstack   # macOS / Linux (Homebrew)
pipx install llmstack-cli                     # isolated, no venv to manage
uv tool install llmstack-cli                  # same, via uv
pip install llmstack-cli                       # plain pip

# Chat with your codebase (just needs Ollama)
llmstack ask -i ./src/

# Full LLM stack with smart routing
llmstack init            # interactive setup wizard
llmstack up
```

### Install options

| Method | Command | Best for |
|--------|---------|----------|
| **Homebrew** | `brew install mara-werils/llmstack/llmstack` | macOS / Linux, auto-updates |
| **pipx** | `pipx install llmstack-cli` | Isolated CLI, no venv juggling |
| **uv** | `uv tool install llmstack-cli` | uv users |
| **pip** | `pip install llmstack-cli` | Inside an existing environment |
| **Docker** | `docker run -p 8000:8000 ghcr.io/mara-werils/llmstack:latest` | Running the gateway as a server |

---

## Universal Gateway

Route every request through a single OpenAI-compatible endpoint. llmstack picks the best provider and model automatically.

**6 cloud providers + local inference:**

| Provider | Models | Pricing tracked |
|----------|--------|----------------|
| **OpenAI** | GPT-4o, GPT-4.1, o3, o4-mini, GPT-4.1-nano | Per-token |
| **Anthropic** | Claude Opus 4, Sonnet 4, Haiku 4 | Per-token |
| **Google** | Gemini 2.5 Pro/Flash, Gemini 2.0 Flash | Per-token |
| **Groq** | Llama 3.3 70B, Llama 3.1 8B, Mixtral | Per-token |
| **Together** | Llama 405B, DeepSeek R1/V3, Qwen 72B | Per-token |
| **Mistral** | Mistral Large/Small, Codestral, Pixtral | Per-token |
| **Local** | Ollama / vLLM (any GGUF model) | Free |

**Fallback chains:** if OpenAI returns 429/503, the request automatically retries on Anthropic, then falls back to local.

```yaml
# llmstack.yaml
providers:
  enabled: true
  strategy: cost          # cost | quality | balanced | latency
  providers:
    - name: openai
      api_key_env: OPENAI_API_KEY
      models:
        - name: gpt-4.1-nano
          tier: simple
          cost_per_m_input: 0.10
        - name: gpt-4o
          tier: medium
          cost_per_m_input: 2.50
      fallback: [anthropic, local]
    - name: anthropic
      api_key_env: ANTHROPIC_API_KEY
    - name: local
```

**Response headers tell you exactly what happened:**

```
X-Provider: openai
X-Model-Router: gpt-4.1-nano
X-Query-Tier: simple
X-Cost-USD: 0.000003
X-Cache: MISS
```

---

## Smart Routing

The classifier scores every request in **< 2ms** using 7 heuristic signals (no ML model needed), then picks the cheapest adequate model.

```
Request → Classify (7 signals, <2ms) → Route to optimal model + provider
```

| Signal | What it measures |
|--------|-----------------|
| Token count | Message length |
| Task markers | "hello" vs "implement distributed consensus" |
| Code detection | Code blocks, programming terms |
| Conversation depth | Turn count |
| System prompt | Complexity of instructions |
| Language mix | Multilingual content |
| Question patterns | Simple fact vs multi-constraint reasoning |

**Real results (CPU-only, no GPU):**

| Query | Tier | Model | Latency |
|-------|------|-------|---------|
| "Hello!" | Simple | llama3.2:1b | **1.6s** |
| "What is 2+2?" | Simple | llama3.2:1b | **5.9s** |
| "Write binary search in Python" | Medium | llama3.2:3b | **52.2s** |

**71% of requests routed to the small model. 71% compute savings.**

With cloud providers, cost savings are even bigger — simple queries go to `gpt-4.1-nano` ($0.10/M) instead of `gpt-4o` ($2.50/M).

---

## AI Agents & MCP

### Agents with tool use

```bash
llmstack agent "Find all TODO comments in this repo and summarize them"
```

The agent uses a **ReAct loop** — it plans, calls tools, observes results, and iterates until the task is done.

**6 built-in tools:** `read_file`, `write_file`, `list_directory`, `grep`, `shell`, `http_get`

```bash
# Use specific tools only
llmstack agent "Check if tests pass" --tools shell,read_file

# Use a larger model for complex tasks
llmstack agent "Refactor auth.py to use JWT tokens" --model llama3.1:70b
```

### MCP Server

Connect any MCP-compatible AI client (Claude Code, Cursor, VS Code) to your local LLM:

```bash
llmstack mcp --model llama3.2
```

```json
// .claude/claude_desktop_config.json
{
  "mcpServers": {
    "llmstack": {
      "command": "llmstack",
      "args": ["mcp", "--model", "llama3.2"]
    }
  }
}
```

**8 tools exposed via MCP:** all agent tools + `llmstack_chat` (LLM inference) + `llmstack_ask` (file RAG with citations).

---

## Fine-tuning

Fine-tune a model on your data in one command. No Jupyter. No boilerplate. No ML expertise.

```bash
llmstack finetune data.jsonl --base llama3.2:1b --export-ollama my-model
```

**What happens:**

1. **Auto data prep** — detects format (CSV/JSON/JSONL/TXT/Parquet), auto-maps columns (`instruction`/`output`, `prompt`/`completion`, `question`/`answer`, chat `messages`), splits train/eval
2. **Auto hyperparameters** — epochs, LoRA rank, batch size, learning rate all auto-selected based on dataset size and model
3. **Training** — LoRA/QLoRA via [unsloth](https://github.com/unslothai/unsloth) (2x faster) or HuggingFace PEFT
4. **Export** — GGUF conversion + `ollama create` → model ready to serve

```bash
# Override any hyperparameter
llmstack finetune data.csv --base llama3.2:1b --epochs 5 --lr 1e-4 --lora-r 32

# Export to GGUF with custom quantization
llmstack finetune data.jsonl --base llama3.2:1b --export-gguf --quant q5_k_m

# Full pipeline: train + export + register in Ollama
llmstack finetune emails.jsonl --base llama3.2:1b --export-ollama email-assistant
# → ollama run email-assistant
```

**Auto hyperparameter selection:**

| Dataset size | Epochs | LoRA rank | Learning rate |
|-------------|--------|-----------|--------------|
| < 100 | 5 | 8 | 1e-4 |
| 100–500 | 3 | 16 | 2e-4 |
| 500–5K | 2 | 16 | 2e-4 |
| 5K+ | 1 | 32+ | 2e-4 |

---

## AI Observability

Every response is scored in real-time. Quality drift triggers alerts. Compare models with A/B testing.

### Quality scoring (every response, < 1ms)

5 metrics scored on every non-streaming response:

| Metric | What it measures |
|--------|-----------------|
| **Coherence** | Structural quality (length, sentences, formatting) |
| **Relevance** | Does the response address the query? |
| **Refusal** | "I can't help with that" detection |
| **Toxicity** | Harmful content flags |
| **Repetition** | Looping / repetitive output |

### Drift detection & alerts

```
Quality drops below 0.4 → CRITICAL alert
Quality trending negative over 50 requests → WARNING alert
```

```bash
# Check live quality from the gateway
llmstack eval --gateway-url http://localhost:8000
```

```
┌──────────── Quality Summary ────────────┐
│ Metric     Mean    Recent  Trend  Count  │
│ overall    0.7821  0.7534  -0.02   1042  │
│ coherence  0.8912  0.8845  +0.01   1042  │
│ relevance  0.6834  0.6223  -0.06   1042  │  ← trending down
│ refusal    0.0124  0.0098  -0.00   1042  │
│ repetition 0.0231  0.0187  -0.00   1042  │
└─────────────────────────────────────────┘
```

### A/B testing

```bash
# Create a test via API
curl -X POST http://localhost:8000/v1/observe/ab-test \
  -d '{"name":"gpt4o-vs-sonnet","model_a":"gpt-4o","model_b":"claude-sonnet-4-20250514","traffic_split":0.5}'

# Check results
curl http://localhost:8000/v1/observe/ab-test/gpt4o-vs-sonnet
```

```json
{
  "winner": "claude-sonnet-4-20250514",
  "confidence": "high",
  "avg_quality_a": 0.7821,
  "avg_quality_b": 0.8234,
  "requests_a": 523,
  "requests_b": 519,
  "avg_cost_a_usd": 0.000034,
  "avg_cost_b_usd": 0.000089
}
```

### Request tracing

Every request is traced end-to-end:

```
GET /v1/observe/traces?model=gpt-4o&limit=10
```

Each trace captures: prompt, routing decision, provider, model, response, latency, tokens, cost, quality scores.

---

## More about `llmstack ask`

See the [top of this README](#ask-your-codebase-anything) for the full feature breakdown. Under the hood:

```
Files → AST chunker (functions/classes) → Embed (Ollama) → Persistent SQLite index
                                                                    ↓
Question → BM25 keyword search ──┐
                                 ├── Reciprocal Rank Fusion → Top-K context → LLM → Streamed answer
Question → Vector cosine search ─┘
                                                                    ↑
                                                          Git context (branch, commits, diff)
```

---

## Full Stack Architecture

```
llmstack up
    │
    ├── Qdrant (vector DB)          :6333
    ├── Redis (cache + rate limit)  :6379
    ├── Ollama / vLLM (inference)   :11434
    ├── TEI (embeddings)            :8002
    ├── Gateway                     :8000
    │   ├── Smart Router (<2ms classification)
    │   ├── Provider Registry (6 cloud + local)
    │   ├── Semantic Cache (Redis, <1ms hit)
    │   ├── Circuit Breaker (3-state, exponential backoff)
    │   ├── Rate Limiter (token bucket, Redis + Lua)
    │   ├── Quality Scorer (5 metrics, every response)
    │   ├── Trace Store (5K rolling window)
    │   ├── RAG Pipeline (ingest + query)
    │   └── Web UI (chat, RAG, dashboard)
    ├── Prometheus (metrics)
    └── Grafana (dashboard)         :8080
```

Auto hardware detection:

| Hardware | Backend | Why |
|----------|---------|-----|
| NVIDIA GPU 16GB+ | vLLM | PagedAttention, max throughput |
| NVIDIA GPU < 16GB | Ollama | Lower memory overhead |
| Apple Silicon | Ollama | Metal acceleration |
| CPU only | Ollama | GGUF quantized models |

---

## CLI Reference

| Command | Description |
|---------|-------------|
| `llmstack ask <question> [path]` | Ask questions about local files (persistent index, hybrid search) |
| `llmstack ask -i [path]` | Interactive conversation with your codebase |
| `llmstack init [--preset]` | Create config (presets: chat, rag, router, agent) |
| `llmstack up` | Start all services |
| `llmstack down` | Stop all services |
| `llmstack status` | Health check |
| `llmstack chat` | Interactive terminal chat |
| `llmstack agent <task>` | Run an AI agent with tools |
| `llmstack mcp` | Start MCP server for AI clients |
| `llmstack finetune <data>` | Fine-tune a model on your data |
| `llmstack eval` | Evaluate model quality |
| `llmstack bench` | Benchmark routing performance |
| `llmstack export` | Generate docker-compose.yml |
| `llmstack logs <service>` | Stream service logs |
| `llmstack doctor` | Diagnose system issues |
| `llmstack verify-private` | Audit config for any external data egress |

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `POST /v1/chat/completions` | Chat (auto-routed across providers) |
| `POST /v1/embeddings` | Text embeddings |
| `GET /v1/models` | List models from all providers (with pricing) |
| `POST /v1/rag/ingest` | Ingest documents for RAG |
| `POST /v1/rag/query` | RAG query with citations |
| `GET /v1/observe/traces` | Request traces with quality scores |
| `GET /v1/observe/quality` | Quality summary with drift detection |
| `GET /v1/observe/alerts` | Active quality alerts |
| `POST /v1/observe/ab-test` | Create A/B test |
| `GET /v1/observe/ab-test/{name}` | A/B test results |
| `GET /healthz` | System health |
| `GET /metrics` | Prometheus metrics |

## Comparison

### Codebase Q&A

| | llmstack ask | Cursor | Aider | Khoj | Simon's llm |
|---|:---:|:---:|:---:|:---:|:---:|
| AST code chunking | **Yes** | Yes | Partial | No | No |
| Hybrid search (BM25 + vector) | **Yes** | ? | No | No | No |
| Persistent incremental index | **Yes** | Yes | No | Yes | Manual |
| Git-aware context | **Yes** | Yes | Yes | No | No |
| Interactive conversation | **Yes** | Yes | Yes | Yes | No |
| 20+ file types | **Yes** | No | No | Yes | No |
| 100% local + free | **Yes** | No | No | Yes | Yes |
| Zero config CLI | **Yes** | No | No | No | Yes |

### LLM Platform

| | llmstack | Ollama | LiteLLM | LocalAI | LangSmith |
|---|:---:|:---:|:---:|:---:|:---:|
| Multi-provider gateway | **Yes** | - | Yes | - | - |
| Smart cost-aware routing | **Yes** | - | - | - | - |
| Fallback chains | **Yes** | - | Yes | - | - |
| AI quality scoring | **Yes** | - | - | - | Yes |
| Drift detection + alerts | **Yes** | - | - | - | Yes |
| A/B testing | **Yes** | - | - | - | Yes |
| One-command fine-tuning | **Yes** | - | - | - | - |
| AI agents with tools | **Yes** | - | - | - | - |
| MCP server | **Yes** | - | - | - | - |
| Local inference | **Yes** | Yes | - | Yes | - |
| Self-hosted / free | **Yes** | Yes | Partial | Yes | Paid |

## Configuration

```yaml
# llmstack.yaml
version: "1"

models:
  chat:
    name: llama3.2
    backend: auto
  embeddings:
    name: bge-m3

providers:
  enabled: true
  strategy: cost
  providers:
    - name: openai
      api_key_env: OPENAI_API_KEY
      fallback: [anthropic, local]
    - name: anthropic
      api_key_env: ANTHROPIC_API_KEY
    - name: local

observe:
  quality_tracking: true
  alert_threshold: 0.4
  drift_threshold: -0.1

gateway:
  port: 8000
  auth: api_key
  rate_limit: 100/min
```

## Requirements

- Python 3.11+
- **`llmstack ask`**: [Ollama](https://ollama.com) running locally. No Docker needed.
- **Full stack** (`llmstack up`): Docker
- **Fine-tuning**: `pip install llmstack-cli[finetune]` (adds PyTorch, PEFT, TRL)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup. PRs welcome.

## License

Apache-2.0
