# Roadmap

> Where llmstack is headed. Checked items are shipped. Unchecked items are planned.
> Want to influence priorities? [Open a feature request](https://github.com/mara-werils/llmstack/issues/new?template=feature_request.yml) or upvote an existing one.

## v0.1 — Foundation (Shipped)

- [x] CLI with `init`, `up`, `down`, `status`, `logs`, `doctor`
- [x] Auto hardware detection (NVIDIA, Apple Silicon, CPU)
- [x] Smart backend resolver (Ollama or vLLM)
- [x] OpenAI-compatible API gateway with auth and streaming
- [x] Prometheus + Grafana observability dashboard
- [x] Plugin system via Python entry_points
- [x] Presets: `chat`, `rag`, `agent`

## v0.2 — Developer Experience (Shipped)

- [x] Interactive terminal chat (`llmstack chat`)
- [x] Docker Compose export (`llmstack export`)
- [x] GitHub issue templates, PR template, security policy

## v0.3 — Production Gateway (Shipped)

- [x] RAG pipeline with Qdrant (ingest, query, delete, status)
- [x] Semantic response cache (Redis + SHA-256 hashing)
- [x] Token bucket rate limiter (Redis + Lua, atomic)
- [x] Circuit breaker with exponential backoff
- [x] Structured JSON logging with request correlation IDs
- [x] 95 unit tests covering all gateway features

## v0.4 — Web UI & SDK (Shipped)

- [x] Built-in web UI (chat, RAG, dashboard)
- [x] Python SDK (`from llmstack import Client`)
- [x] Conversation history persistence (SQLite-backed)
- [ ] File upload for RAG (PDF, Markdown, HTML)
- [ ] Model download progress UI

## v0.5 — Ask, Multi-Model & Routing (Shipped)

- [x] **`llmstack ask`** — ask questions about local files using a local LLM
  - Supports PDF, DOCX, Markdown, 20+ code and text file types
  - In-memory RAG: parse, chunk, embed, search, generate — no Docker needed
  - Streaming answers with source citations
  - Stdin piping support
  - Configurable: `--model`, `--embed-model`, `--top-k`, `--chunk-size`
- [x] Run multiple models simultaneously
- [x] Smart request routing (fast model for simple queries, large for complex)
- [x] Model performance benchmarking (`llmstack bench`)
- [x] Cost tracking per model/request with budgets and alerts
- [ ] A/B testing between models

## v0.6 — Production Hardening (Shipped)

- [x] Guardrails: PII detection, prompt injection blocking, content filtering
- [x] Smart retry with provider fallback and exponential backoff
- [x] Request priority queue with tier-based scheduling
- [x] Tiered rate limiting per API key (enterprise/pro/standard/free)
- [x] Multi-tenant namespace isolation
- [x] Usage quotas per API key (requests, tokens, cost)
- [x] Backup and restore (`llmstack backup`, `llmstack restore`)
- [ ] Multi-node deployment (distributed inference)
- [ ] Auto-scaling based on queue depth
- [ ] TLS/HTTPS support
- [ ] OAuth2 / OIDC authentication

## v0.7 — Developer Platform (Shipped)

- [x] Prompt template management and versioning (5 built-in templates)
- [x] Webhook notifications (10 event types, HMAC signing)
- [x] Batch processing API (parallel requests with concurrency control)
- [x] Model performance leaderboard (quality/speed/cost rankings)
- [x] Structured output validation (JSON schema)
- [x] Streaming analytics (TTFT, inter-token latency, throughput)
- [x] Prompt prefix caching for shared system prompts
- [x] Model warm-up on startup for instant first requests
- [x] Request replay system for debugging and testing
- [ ] TypeScript/JavaScript SDK
- [ ] REST API for stack management

## v1.0 — Stable Release

- [ ] Comprehensive test coverage (>90%)
- [ ] Performance benchmarks vs alternatives
- [ ] Kubernetes Helm chart
- [ ] Official Docker images on GHCR
- [ ] Plugin marketplace

---

This roadmap is a living document. Priorities shift based on community feedback and real-world usage. If something here matters to you, let us know by opening an issue or starting a discussion.
