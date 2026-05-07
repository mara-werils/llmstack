# Changelog

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
