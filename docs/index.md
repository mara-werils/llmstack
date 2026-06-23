# llmstack

**Ask your files anything. Locally. Privately. One command.**

Chat with your code, PDFs, and logs using a local LLM -- or run a full local LLM stack with smart model routing.

---

## Ask Your Files Anything

```bash
pip install llmstack-cli
llmstack ask "How does authentication work?" ./src/
```

That's it. If [Ollama](https://ollama.com) is running, it works. No Docker, no config, no API keys.

`llmstack ask` parses your files, finds the most relevant sections, and generates a streaming answer with source citations -- all locally, all private.

```bash
llmstack ask "Summarize the key findings" report.pdf
llmstack ask "What went wrong?" error.log
llmstack ask "Find security vulnerabilities" ./src/ --model llama3.1:70b
cat contract.pdf | llmstack ask "Are there any risks?"
```

**Supports:** PDF, DOCX, Markdown, Python, JavaScript, TypeScript, Go, Rust, Java, JSON, YAML, CSV, HTML, logs, and 20+ file types.

[Read the full `ask` guide](guide/ask.md){ .md-button .md-button--primary }

---

## Full Stack Mode: Smart Model Routing

Want more than file Q&A? llmstack also runs a complete local LLM infrastructure with a single command -- and **automatically routes each query to the smallest model that can handle it**.

```bash
llmstack init --preset router
llmstack up
```

You now have 7 production-grade services: multi-model inference, embeddings, vector DB, cache, API gateway with smart routing, Prometheus, and Grafana -- plus a built-in Web UI at `http://localhost:8000`.

**71% of requests get routed to the small model.** Your GPU handles the easy stuff fast and saves capacity for queries that actually need it.

---

## In Your Editor

Bring it into VS Code, Cursor, Windsurf, or VSCodium. The extension talks to your
local gateway, so code never leaves your machine.

- **Chat sidebar** with model picker and editor context.
- **Edit with AI** — review changes as a native diff before they touch your file,
  and revert in one step.
- **Inline completions** and 👍/👎 feedback that trains your local model.

[Editor extension guide](guide/editor.md){ .md-button .md-button--primary }

## Verifiable Privacy

"Runs locally" is easy to claim. llmstack lets you **prove** it — both statically
(`llmstack verify-private`) and at runtime (an egress monitor you can gate CI on).

[Privacy & no-egress proof](guide/privacy.md){ .md-button }

## Prove the Value

The savings and benchmark numbers are yours, generated locally — not our marketing.

- **`llmstack savings`** turns "saves you money" into a running total, valued
  against dated, sourced cloud pricing: *how many months of Copilot/Cursor your
  local usage has already paid for.*
- **`llmstack benchmark`** runs a reproducible suite that reports cost, latency,
  and a zero-egress proof in one shareable report — each carrying a methodology
  hash so anyone can confirm they ran the identical benchmark.

[Savings](guide/savings.md){ .md-button } [Reproducible benchmarks](guide/benchmarks.md){ .md-button }

---

## Key Features

- **`llmstack ask`** -- chat with your files from the terminal. PDF, code, logs, docs. One command.
- **Editor extension** -- chat, AI edits with diff review, and inline completion in VS Code & forks.
- **Verifiable privacy** -- static audit + runtime egress monitor prove no data leaves the machine.
- **Provable savings** -- `llmstack savings` tallies the cloud bill you didn't pay, vs dated pricing.
- **Reproducible benchmarks** -- `llmstack benchmark` reports cost, latency, and a zero-egress proof.
- **Smart Model Router** -- routes queries to the right-sized model automatically
- **Zero configuration** -- hardware detection auto-selects vLLM or Ollama based on your GPU
- **OpenAI-compatible API** -- works with LangChain, LlamaIndex, Vercel AI SDK, openai-python
- **Built-in RAG pipeline** -- ingest documents, query with retrieval-augmented generation
- **Semantic response cache** -- Redis-backed caching with SHA-256 key hashing
- **Token bucket rate limiter** -- Redis + Lua atomicity, per-key or per-IP
- **Circuit breaker** -- fail-fast when inference is down, exponential backoff recovery
- **Observability** -- Prometheus + Grafana with a pre-built dashboard
- **Built-in Web UI** -- chat, RAG, dashboard, settings
- **Plugin ecosystem** -- extend with new backends via pip

## Who Is This For?

- **Anyone** who wants to ask questions about local files without uploading them to the cloud
- **Developers** who want to understand codebases, debug logs, or analyze documents from the terminal
- **AI app developers** who want local inference + RAG without Docker boilerplate
- **Teams** who need an OpenAI-compatible API backed by local models
- **Hobbyists** running LLMs locally who want vector search, caching, and monitoring out of the box

## Quick Example

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="YOUR_KEY")

response = client.chat.completions.create(
    model="auto",  # smart routing picks the right model
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

## Comparison

| Feature | llmstack | Ollama | LocalAI | LiteLLM |
|---|---|---|---|---|
| Chat with local files | **Yes** | No | No | No |
| Smart model routing | **Yes** | No | No | No |
| One-command full stack | **Yes** | No | No | No |
| Built-in RAG pipeline | **Yes** | No | No | No |
| Response caching | **Yes** | No | No | No |
| Circuit breaker | **Yes** | No | No | No |
| Rate limiting (Redis) | **Yes** | No | No | Yes |
| Auto hardware detection | **Yes** | No | No | No |
| OpenAI-compatible API | **Yes** | Yes | Yes | Yes |
| Built-in vector DB | **Yes** | No | No | No |
| Observability dashboard | **Yes** | No | Partial | Partial |
| Plugin ecosystem | **Yes** | No | No | No |

See how llmstack compares to other AI coding tools (Cline, Continue, Tabby, Aider)
in the [full comparison](comparison.md).

## Requirements

- Python 3.11+
- **`llmstack ask`**: Just [Ollama](https://ollama.com). No Docker needed.
- **Full stack mode** (`llmstack up`): Docker

## License

Apache-2.0
