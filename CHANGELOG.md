# Changelog

## [0.4.0] - 2026-05-08

### Added
- **Web UI** — built-in chat interface served at `http://localhost:8000`
  - Chat panel with streaming, model selector, conversation history
  - RAG panel for document ingestion and knowledge base queries
  - Dashboard with health status, cache stats, circuit breaker state
  - Settings panel with localStorage persistence
  - Dark theme, zero external dependencies, fully self-contained
- **Python SDK** — `from llmstack import Client`
  - Sync `Client` and async `AsyncClient` with full API coverage
  - Chat, embeddings, RAG ingest/query, models, health
  - Streaming support via generators
  - Context manager support
- **Integration examples** — ready-to-run code for popular frameworks
  - LangChain (chat, chains, RAG)
  - LlamaIndex (indexing, chat engine)
  - OpenAI Python SDK (drop-in replacement)
  - Vercel AI SDK (TypeScript/Next.js)
  - FastAPI app template
  - SDK quickstart
- **Documentation site** — MkDocs Material with full docs
  - Getting started, architecture, API reference, CLI reference
  - Gateway features guide, observability guide, plugin guide
  - Dark/light theme toggle, search, code highlighting
- **ROADMAP.md** — public roadmap through v1.0
- **Improved GitHub templates** — YAML-based issue forms with dropdowns
- **GitHub Sponsors** — FUNDING.yml
- **CI docs job** — `mkdocs build --strict` in CI pipeline

## [0.3.0] - 2026-05-07

### Added
- **RAG Pipeline** — real retrieval-augmented generation with Qdrant
  - `POST /v1/rag/ingest` — chunk, embed, and store documents in Qdrant
  - `POST /v1/rag/query` — semantic search + LLM generation with source citations
  - `DELETE /v1/rag/documents/{source}` — delete documents by source
  - `GET /v1/rag/status` — collection statistics
  - Streaming support for RAG queries via SSE
- **Semantic Response Cache** — Redis-backed LLM response caching
  - SHA-256 hash of (model + messages) for deterministic cache keys
  - Only caches low-temperature requests (≤0.1) for correctness
  - TTL-based expiration with configurable duration
  - Cache hit/miss metrics exposed in `/healthz`
  - `X-Cache: HIT/MISS` response headers
- **Token Bucket Rate Limiter** — Redis-backed with Lua script for atomicity
  - Configurable via `rate_limit` in llmstack.yaml (e.g., `100/min`, `10/sec`)
  - Per-API-key rate limiting with IP fallback
  - In-memory fallback when Redis is unavailable
  - Standard `X-RateLimit-*` and `Retry-After` headers
  - Atomic Lua script prevents race conditions in distributed setup
- **Circuit Breaker** — resilience pattern for inference backend
  - Three-state machine: CLOSED → OPEN → HALF_OPEN → CLOSED
  - Exponential backoff on recovery timeout (capped)
  - Fail-fast with `503 Service Unavailable` when circuit is open
  - Metrics exposed in `/healthz` (state, failure count, rejections)
- **Structured Logging** — JSON request logs with correlation IDs
  - `X-Request-ID` header propagation
  - Per-request structured JSON with method, path, status, duration, client IP
  - Configurable log level and format (JSON / text)
- Token usage extraction from inference responses into Prometheus metrics
- Redis health check in `/healthz` endpoint
- 50 new unit tests (95 total) covering cache, circuit breaker, rate limiter, RAG

## [0.2.0] - 2026-05-07

### Added
- `llmstack chat` — interactive terminal chat with streaming responses
- `llmstack export` — generate standalone docker-compose.yml from llmstack.yaml
- GitHub issue templates, PR template, security policy

### Fixed
- Gateway Docker image now builds locally (no longer requires ghcr.io)
- Prometheus and Grafana configs are written to disk before container start
- Generated API keys persist to llmstack.yaml across restarts
- Clear error messages for port conflicts

## [0.1.0] - 2026-05-07

### Added
- CLI with `init`, `up`, `down`, `status`, `logs`, `doctor` commands
- Auto hardware detection (NVIDIA, Apple Silicon, CPU)
- Smart backend resolver (auto-picks Ollama or vLLM)
- Services: Ollama, vLLM, Qdrant, Redis, TEI (Text Embeddings Inference)
- API Gateway: OpenAI-compatible proxy with auth, rate limiting, SSE streaming
- Prometheus + Grafana observability with pre-provisioned dashboard
- Plugin system via Python entry_points
- Presets: `chat`, `rag`, `agent`
- Pydantic v2 config schema (`llmstack.yaml`)
- Docker SDK orchestration (no docker-compose dependency)
- CI/CD: GitHub Actions for lint/test and PyPI release
