# llmstack

**One command. Full LLM stack. Zero config.**

Stop wiring Docker containers. Start building AI apps.

---

## What is llmstack?

llmstack is a CLI tool that spins up a complete local LLM infrastructure with a single command. It detects your hardware, picks the optimal inference backend, and boots seven production-grade services: inference, embeddings, vector DB, cache, API gateway, Prometheus, and Grafana.

```bash
pip install llmstack-cli
llmstack init --preset rag
llmstack up
```

That gives you a fully functional stack with an OpenAI-compatible API at `http://localhost:8000`.

## Key Features

- **Zero configuration** -- hardware detection auto-selects vLLM or Ollama based on your GPU
- **OpenAI-compatible API** -- works with LangChain, LlamaIndex, Vercel AI SDK, openai-python
- **Built-in RAG pipeline** -- ingest documents, query with retrieval-augmented generation
- **Semantic response cache** -- Redis-backed caching with SHA-256 key hashing
- **Token bucket rate limiter** -- Redis + Lua atomicity, per-key or per-IP
- **Circuit breaker** -- fail-fast when inference is down, exponential backoff recovery
- **Observability** -- Prometheus + Grafana with a pre-built dashboard
- **Plugin ecosystem** -- extend with new backends via pip

## Who is this for?

- **AI app developers** who want local inference + RAG without Docker boilerplate
- **Teams** who need an OpenAI-compatible API backed by local models
- **Hobbyists** running LLMs locally who want vector search, caching, and monitoring out of the box
- **Anyone** tired of writing 200+ lines of docker-compose.yml every time

## Quick Example

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="YOUR_KEY")

response = client.chat.completions.create(
    model="llama3.2",
    messages=[{"role": "user", "content": "Explain quantum computing"}]
)
print(response.choices[0].message.content)
```

## Architecture Overview

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

## Comparison

| Feature | llmstack | Ollama | LocalAI | LiteLLM |
|---|---|---|---|---|
| One-command full stack | Yes | No | No | No |
| Built-in RAG pipeline | Yes | No | No | No |
| Response caching | Yes | No | No | No |
| Circuit breaker | Yes | No | No | No |
| Rate limiting (Redis) | Yes | No | No | Yes |
| Auto hardware detection | Yes | No | No | No |
| OpenAI-compatible API | Yes | Yes | Yes | Yes |
| Built-in vector DB | Yes | No | No | No |
| Observability dashboard | Yes | No | Partial | Partial |
| Plugin ecosystem | Yes | No | No | No |

## Requirements

- Python 3.11+
- Docker

## License

Apache-2.0
