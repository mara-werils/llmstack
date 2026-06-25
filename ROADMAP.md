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
- [x] TypeScript/JavaScript SDK (`@llmstack/client`)
- [ ] REST API for stack management

## v0.8 — Adaptive Learning Pipeline (Shipped)

- [x] Curriculum learning strategy (progressive difficulty training)
- [x] Multi-armed bandit for model selection (Thompson, UCB1, epsilon-greedy)
- [x] Feedback deduplication and normalization
- [x] Data quality scoring for training examples
- [x] Learning rate scheduler (constant, linear warmup, cosine, step decay)
- [x] Cross-validation evaluator for model quality
- [x] Request correlation ID middleware (X-Request-ID)
- [x] Model alias mapping (user-friendly short names)
- [x] API key rotation with graceful migration
- [x] Provider health checker with status tracking
- [x] Request deduplication for idempotent calls
- [x] Latency percentile tracking (p50, p95, p99)
- [x] Error rate monitoring with automatic alerting
- [x] System resource monitor (CPU, memory, disk)

## v0.9 — Distribution & Privacy (In progress)

> Strategy: win on the two proven levers of adoption — being **inside the IDE**
> and a **frictionless, provably-private first run**. See `PRODUCT_STRATEGY.md`.

- [x] Frictionless onboarding: `quickstart` proves first value (hardware-sized model,
      a real local completion, zero key, no Docker), `init` wizard, `up` pre-flight checks
- [x] Distribution: Homebrew tap, PyPI, GHCR (auto-published on release)
- [x] **`llmstack verify-private`** — audit config for any external data egress
- [x] `verify-private --live` — also probes the running gateway, catching env-var
      overrides that diverge from llmstack.yaml at runtime
- [x] VS Code / OpenVSX extension (Ask, Explain, gateway health, opt-in inline completion) — `editors/vscode`
- [x] **Editor agentic UX**: chat sidebar, AI edit with native-diff review +
      one-step checkpoint revert, apply/insert code blocks, model picker,
      in-editor 👍/👎 feedback wired to the learning pipeline, getting-started walkthrough
- [x] **Documentation site** published to GitHub Pages (mkdocs-material)
- [x] **Reproducible no-egress proof**: runtime egress monitor + privacy guide
- [x] One-line installer (`curl … | sh`) preferring isolated uv/pipx installs
- [x] Gateway test coverage backfill (replay, health, providers, proxy, cache, …)
- [x] **Provable savings**: `llmstack savings` + `/v1/savings` value local usage
      against a dated, sourced cloud-pricing catalog (`core/pricing`, `core/savings`)
- [x] **Reproducible benchmarks** (`llmstack benchmark`): cost + latency + a runtime
      zero-egress proof in one report, each carrying a methodology hash; CI-gated
- [x] **First-run readiness everywhere**: hardware-sized model recommendation
      (`core/onboarding`) shared by `llmstack quickstart`/`ready`/`doctor`/`info`,
      the gateway `GET /v1/onboarding` route, `client.onboarding()`/`ready()` in
      both SDKs, an `llmstack_onboarding` MCP tool, and the VS Code first-run prompt
- [ ] Publish extension to OpenVSX + VS Marketplace (CI ready; needs `OVSX_TOKEN` + a `vscode-v*` tag)
- [ ] JetBrains plugin

## v1.0 — Stable Release

- [x] Comprehensive test coverage (>90%) — currently 95%+ (`fail_under = 95`)
- [x] Performance benchmarks vs alternatives — reproducible `llmstack benchmark`
      harness (cost + latency + privacy), methodology-hashed and CI-gated
- [ ] Kubernetes Helm chart
- [ ] Official Docker images on GHCR
- [ ] Plugin marketplace

---

This roadmap is a living document. Priorities shift based on community feedback and real-world usage. If something here matters to you, let us know by opening an issue or starting a discussion.
