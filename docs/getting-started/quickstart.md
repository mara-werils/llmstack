# Quickstart

Get a full LLM stack running in under two minutes.

## Step 1: Initialize

```bash
llmstack init --preset rag
```

This does three things:

1. **Detects your hardware** -- GPU vendor, VRAM, CPU cores, RAM
2. **Picks the optimal backend** -- vLLM for NVIDIA GPUs with 16GB+ VRAM, Ollama otherwise
3. **Generates `llmstack.yaml`** -- a single configuration file for the entire stack

Example output:

```
Hardware detected:
  CPU: 10 cores
  RAM: 32 GB
  GPU: Apple M2 Pro (32 GB VRAM)

Using preset: rag
  Backend: Ollama

Created llmstack.yaml
Next: edit the config if needed, then run llmstack up
```

### Available Presets

| Preset | Services | Use Case |
|---|---|---|
| `chat` | Inference + Cache + Gateway | Minimal chatbot setup |
| `rag` | + Qdrant + Embeddings | Document Q&A, search |
| `agent` | 70B model + 16K context + longer timeouts | Complex agent workflows |

## Step 2: Start the Stack

```bash
llmstack up
```

llmstack boots services in dependency order with health checks:

```
Starting LLMStack...

  Starting qdrant... ready
  Starting redis... ready
  Starting ollama... ready
  Pulling model llama3.2... done
  Starting tei... ready
  Starting gateway... ready
  Starting prometheus... ready
  Starting grafana... ready

LLMStack Services
┌────────────┬────────────┬─────────┬───────────────────────────┐
│ Service    │ Category   │ Status  │ URL                       │
├────────────┼────────────┼─────────┼───────────────────────────┤
│ qdrant     │ vectordb   │ running │ http://localhost:6333     │
│ redis      │ cache      │ running │ http://localhost:6379     │
│ ollama     │ inference  │ running │ http://localhost:11434    │
│ tei        │ embeddings │ running │ http://localhost:8002     │
│ gateway    │ gateway    │ running │ http://localhost:8000     │
│ prometheus │ observe    │ running │ http://localhost:9090     │
│ grafana    │ observe    │ running │ http://localhost:8080     │
└────────────┴────────────┴─────────┴───────────────────────────┘
```

An API key is generated automatically and saved to `llmstack.yaml`.

## Step 3: Send a Request

### Chat Completion

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"llama3.2","messages":[{"role":"user","content":"Hello!"}]}'
```

### Ingest a Document for RAG

```bash
curl http://localhost:8000/v1/rag/ingest \
  -H "Authorization: Bearer YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{"text":"LLMStack is an open-source tool that...","source":"docs.txt"}'
```

### Query with RAG

```bash
curl http://localhost:8000/v1/rag/query \
  -H "Authorization: Bearer YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{"question":"What is LLMStack?"}'
```

## Step 4: Try the Interactive Chat

```bash
llmstack chat
```

```
LLMStack Chat -- model: llama3.2
Type 'exit' or Ctrl+C to quit. '/clear' to reset conversation.

You: What is quantum computing?
Assistant: Quantum computing uses quantum mechanical phenomena like
superposition and entanglement to process information...

You: /clear
Conversation cleared.
```

## Step 5: Check the Dashboard

Open [http://localhost:8080](http://localhost:8080) in your browser (login: `admin` / `llmstack`).

The Grafana dashboard shows:

- Request rate per endpoint
- Latency p50/p99 histograms
- Token throughput
- Error rate
- Cache hit rate
- Circuit breaker state

## Shutting Down

```bash
llmstack down
```

To also remove data volumes (model cache, vector data, etc.):

```bash
llmstack down --volumes
```

## Next Steps

- [Configuration Reference](configuration.md) -- customize your `llmstack.yaml`
- [Gateway Guide](../guide/gateway.md) -- learn about caching, rate limiting, and circuit breaker
- [API Reference](../api/openai-compatible.md) -- full endpoint documentation
- [CLI Reference](../cli.md) -- all available commands
