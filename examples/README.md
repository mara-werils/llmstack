# llmstack Integration Examples

Working examples showing how to use llmstack with popular AI frameworks and SDKs.

## Prerequisites

Start llmstack before running any example:

```bash
pip install llmstack-cli
llmstack init --preset rag
llmstack up
```

The gateway will be available at `http://localhost:8000`.

## Examples

| File | Framework | What it shows |
|------|-----------|---------------|
| [`openai_sdk.py`](openai_sdk.py) | OpenAI Python SDK | Chat, streaming, multi-turn, embeddings, JSON mode — drop-in replacement |
| [`langchain_chat.py`](langchain_chat.py) | LangChain | ChatOpenAI, LCEL chains, streaming, batch, full RAG with Qdrant |
| [`llamaindex_rag.py`](llamaindex_rag.py) | LlamaIndex | Document indexing, Qdrant vector store, chat engine with memory |
| [`fastapi_app.py`](fastapi_app.py) | FastAPI | Production app with RAG Q&A, summarization, translation endpoints |
| [`sdk_quickstart.py`](sdk_quickstart.py) | openai + httpx | All llmstack features: health, chat, stream, embed, RAG ingest/query |
| [`airgapped_proof.py`](airgapped_proof.py) | llmstack core | Prove a workload is air-gapped: static privacy audit + runtime egress monitor (CI-gateable) |
| [`onboarding_check.py`](onboarding_check.py) | llmstack core | Check first-run readiness: hardware-sized model recommendation + Ollama probe (exits non-zero when not ready) |
| [`vercel_ai/`](vercel_ai/) | Vercel AI SDK (TS) | Next.js streaming route, embeddings, multi-turn conversation |

## Quick start per example

### OpenAI SDK

```bash
pip install openai
python openai_sdk.py
```

The simplest integration. Change `base_url` and you are done — all your existing OpenAI code works.

### LangChain

```bash
pip install langchain langchain-openai langchain-qdrant qdrant-client
python langchain_chat.py
```

Shows LCEL chains, prompt templates, batch processing, and a full RAG pipeline using LangChain's Qdrant integration with llmstack embeddings.

### LlamaIndex

```bash
pip install llama-index llama-index-llms-openai-like llama-index-embeddings-openai \
            llama-index-vector-stores-qdrant qdrant-client
python llamaindex_rag.py
```

Covers in-memory indexes, Qdrant-backed persistent indexes, chat engines with conversation memory, and directory-based document loading.

### FastAPI app

```bash
pip install fastapi uvicorn httpx openai
uvicorn fastapi_app:app --port 3000 --reload
# Open http://localhost:3000/docs
```

A complete API with document ingestion, RAG querying, text summarization, and translation — all backed by llmstack.

### SDK quickstart

```bash
pip install openai httpx
python sdk_quickstart.py
```

Exercises every llmstack endpoint in one script: health check, model listing, chat, streaming, embeddings, RAG ingest, RAG query (normal + streaming), and cleanup.

### Vercel AI SDK (TypeScript)

```bash
cd vercel_ai
npm install
npx tsx index.ts
```

Or copy `index.ts` into a Next.js project as `app/api/chat/route.ts` for a streaming chat UI.

## Configuration

All examples default to:

| Setting | Value |
|---------|-------|
| Gateway URL | `http://localhost:8000` |
| API key | `llmstack` |
| Chat model | `llama3.2` |
| Embedding model | `bge-m3` |
| Qdrant URL | `http://localhost:6333` |

Change these at the top of each file to match your `llmstack.yaml` configuration.

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Connection refused on port 8000 | Run `llmstack up` and wait for health checks |
| 503 Service Unavailable | Inference backend is loading — check `llmstack status` |
| Model not found | Run `llmstack status` to see available models |
| Qdrant connection error | Make sure you used `--preset rag` (not `--preset chat`) |
| Rate limited (429) | Default is 100 req/min — wait or adjust `gateway.rate_limit` in `llmstack.yaml` |
