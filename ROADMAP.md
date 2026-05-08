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

## v0.4 — Web UI & SDK (Next)

- [x] Built-in web UI (chat, RAG, dashboard)
- [x] Python SDK (`from llmstack import Client`)
- [ ] File upload for RAG (PDF, Markdown, HTML)
- [ ] Conversation history persistence
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
- [ ] A/B testing between models
- [ ] Cost tracking per model/request

## v0.6 — Production Hardening

- [ ] Multi-node deployment (distributed inference)
- [ ] Auto-scaling based on queue depth
- [ ] TLS/HTTPS support
- [ ] OAuth2 / OIDC authentication
- [ ] Backup and restore

## v0.7 — Developer Platform

- [ ] Prompt management and versioning
- [ ] Built-in evaluation framework
- [ ] Webhook notifications
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
