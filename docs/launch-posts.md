# Launch Posts — llmstack v0.1.0

## Show HN: llmstack — One command to spin up a full local LLM stack

**URL:** https://github.com/mara-werils/llmstack

**Text:**

I built llmstack because every time I wanted to prototype an AI app locally, I spent hours wiring Docker containers — Ollama, vector DB, embeddings, Redis cache, API proxy. The same boilerplate, over and over.

llmstack fixes this. One command gives you a full production-grade LLM stack:

```
pip install llmstack-cli
llmstack init --preset rag
llmstack up
```

That's it. You now have:
- Inference (Ollama or vLLM, auto-detected based on your GPU)
- Vector DB (Qdrant)
- Embeddings (TEI with bge-m3)
- Redis cache
- OpenAI-compatible API gateway with auth and rate limiting
- Prometheus + Grafana dashboard

It auto-detects your hardware — NVIDIA GPU with 16GB+ VRAM gets vLLM for max throughput, Apple Silicon gets Ollama with Metal acceleration, CPU gets quantized models.

The API gateway is fully OpenAI-compatible, so it works with LangChain, LlamaIndex, Vercel AI SDK, or any OpenAI client library out of the box.

Tech: Python, Typer + Rich CLI, FastAPI gateway, Docker SDK (no docker-compose), Pydantic v2 config, plugin system via entry_points.

~3000 LOC, 42 tests, Apache 2.0. Would love feedback.

---

## Reddit r/LocalLLaMA Post

**Title:** I built a CLI that spins up a full local LLM stack with one command — inference, vector DB, embeddings, cache, API gateway, and monitoring

**Body:**

Hey r/LocalLLaMA!

I got tired of manually wiring Docker containers every time I wanted a local LLM setup for prototyping. So I built **llmstack** — a CLI that gives you a complete stack in three commands:

```bash
pip install llmstack-cli
llmstack init --preset rag
llmstack up
```

**What you get:**

| Layer | Service | Default |
|-------|---------|---------|
| Inference | Ollama or vLLM (auto) | llama3.2 |
| Embeddings | TEI | bge-m3 |
| Vector DB | Qdrant | — |
| Cache | Redis | 256MB LRU |
| API Gateway | FastAPI | OpenAI-compatible |
| Dashboard | Grafana + Prometheus | Pre-built panels |

**Key features:**
- **Auto hardware detection** — picks vLLM for NVIDIA 16GB+ (PagedAttention), Ollama for everything else
- **OpenAI-compatible API** — works with LangChain, LlamaIndex, any OpenAI client
- **Auth + rate limiting** out of the box
- **SSE streaming** support
- **Plugin system** — extend with new backends via pip
- **Presets** — `chat` (minimal), `rag` (+ vectors + embeddings), `agent` (70B + long context)

Built with Docker SDK directly (no docker-compose dependency), Pydantic v2 for config, Typer + Rich for the CLI.

The whole idea is: stop configuring infrastructure, start building your AI app.

GitHub: https://github.com/mara-werils/llmstack
PyPI: `pip install llmstack-cli`

Open source (Apache 2.0). Would love to hear what you think — what features would make this more useful for your workflow?
