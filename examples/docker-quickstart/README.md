# Docker Quickstart

Start the full LLMStack stack with one command:

```bash
docker compose up -d
```

## Services

| Service | Port | Description |
|---------|------|-------------|
| Gateway | 8000 | API gateway |
| Ollama  | 11434| LLM inference |
| Redis   | 6379 | Response cache |
| Qdrant  | 6333 | Vector database |

## Quick test

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "llama3.2", "messages": [{"role": "user", "content": "Hello!"}]}'
```

## GPU support

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
```

## Monitoring

```bash
docker compose --profile monitoring up -d
```

Grafana dashboard: http://localhost:3000 (admin/llmstack)
