# Gateway

The llmstack gateway is a production-grade API layer built with FastAPI. It is not a simple reverse proxy -- it implements caching, rate limiting, circuit breaking, RAG, structured logging, and authentication.

## Overview

The gateway runs as a Docker container and exposes an OpenAI-compatible API on port 8000 (configurable). All requests flow through a middleware stack before reaching the route handlers.

```
Request
  │
  ├── LoggingMiddleware      (structured JSON logs, X-Request-ID)
  ├── AuthMiddleware         (API key validation)
  ├── RateLimitMiddleware    (token bucket via Redis)
  ├── MetricsMiddleware      (Prometheus counters/histograms)
  │
  └── Route Handler
```

## Semantic Response Cache

The cache stores LLM responses in Redis, keyed by a SHA-256 hash of the model name and message history. This means identical requests return instantly from cache.

### How It Works

```
Request → SHA-256(model + messages) → Redis lookup
  HIT  → Return cached response (< 1ms)
  MISS → Forward to inference → Cache result → Return
```

### Caching Rules

- **Only deterministic requests are cached**: requests with `temperature <= 0.1`. Higher temperatures produce different outputs each time, so caching would return stale results.
- **TTL-based expiration**: cached responses expire after a configurable duration (default: 1 hour).
- **Cache headers**: every response includes an `X-Cache: HIT` or `X-Cache: MISS` header so you can verify caching behavior.
- **Cache stats**: the `/healthz` endpoint reports cache hit/miss counts.

### Monitoring Cache Performance

Check the `/healthz` endpoint:

```bash
curl http://localhost:8000/healthz
```

The response includes cache statistics showing hit and miss counts.

## Token Bucket Rate Limiter

The rate limiter prevents any single client from overwhelming the API. It uses a token bucket algorithm implemented as an atomic Redis Lua script.

### How It Works

```
Request → Extract API key / IP → Redis EVALSHA (atomic Lua) → Allow / Reject
```

1. Each client (identified by API key, or IP address as fallback) has a token bucket in Redis
2. The bucket fills at the configured rate (e.g., 100 tokens per minute)
3. Each request consumes one token
4. When the bucket is empty, requests are rejected with `429 Too Many Requests`

### Configuration

Set the rate limit in `llmstack.yaml`:

```yaml
gateway:
  rate_limit: 100/min    # 100 requests per minute per client
```

Supported formats:

| Format | Meaning |
|---|---|
| `10/sec` | 10 requests per second |
| `100/min` | 100 requests per minute |
| `3600/hour` | 3600 requests per hour |

### Response Headers

Every response includes rate limit headers:

| Header | Description |
|---|---|
| `X-RateLimit-Limit` | Maximum requests in the current window |
| `X-RateLimit-Remaining` | Requests remaining in the current window |
| `X-RateLimit-Reset` | Seconds until the bucket refills |
| `Retry-After` | Seconds to wait (only on 429 responses) |

### Fallback Behavior

If Redis is unavailable, the rate limiter falls back to an in-memory token bucket. This ensures the API stays functional even if Redis goes down, though rate limits are not shared across gateway instances in this mode.

## Circuit Breaker

The circuit breaker prevents cascading failures when the inference backend is unhealthy. Instead of sending requests to a failing backend (which would pile up timeouts), the circuit breaker fails fast with a `503 Service Unavailable`.

### State Machine

```
CLOSED ──[5 failures]──> OPEN ──[timeout]──> HALF_OPEN ──[success]──> CLOSED
                           │                      │
                           └──[reject fast]       └──[failure]──> OPEN (backoff x2)
```

**CLOSED** (normal operation): All requests are forwarded to the inference backend. Failures are counted.

**OPEN** (backend is down): All requests are immediately rejected with `503`. After a timeout period, the circuit transitions to HALF_OPEN.

**HALF_OPEN** (testing recovery): A single request is allowed through. If it succeeds, the circuit moves back to CLOSED. If it fails, the circuit returns to OPEN with a doubled timeout (exponential backoff).

### Monitoring

The `/healthz` endpoint reports the circuit breaker state:

- Current state (CLOSED, OPEN, HALF_OPEN)
- Failure count
- Total rejections
- Time spent in the current state

## RAG Pipeline

The gateway includes a full retrieval-augmented generation pipeline backed by Qdrant and the embedding service.

### Ingestion

```
Document → Chunk (512 words, 64 word overlap) → Embed → Store in Qdrant
```

`POST /v1/rag/ingest` accepts a document text and source identifier. The text is split into overlapping chunks, each chunk is embedded using the configured embedding model, and the resulting vectors are stored in Qdrant with metadata.

Chunk IDs are deterministic (based on content hash), so re-ingesting the same document updates rather than duplicates.

### Query

```
Question → Embed → Qdrant search (top-k) → Build context → LLM generate
```

`POST /v1/rag/query` embeds the question, performs a semantic search in Qdrant, assembles the top results into a context prompt, and generates an answer using the chat model. The response includes source citations.

Streaming is supported via Server-Sent Events (SSE).

See the [RAG API Reference](../api/rag.md) for endpoint details.

## Structured Logging

Every request is logged as a structured JSON object:

```json
{
  "ts": "2026-05-07T14:23:01",
  "level": "INFO",
  "msg": "POST /v1/chat/completions 200 1234.5ms",
  "request_id": "a1b2c3d4",
  "method": "POST",
  "path": "/v1/chat/completions",
  "status": 200,
  "duration_ms": 1234.5,
  "client_ip": "10.0.0.1"
}
```

### Request ID Correlation

Every request is assigned an `X-Request-ID` header. If the client sends one, it is preserved; otherwise, a new UUID is generated. This ID appears in the logs and in the response headers, making it easy to trace a request across services.

## Authentication

When `gateway.auth` is set to `api_key` (the default), the gateway requires a Bearer token in the `Authorization` header:

```
Authorization: Bearer sk-llmstack-abc123...
```

Requests without a valid key receive `401 Unauthorized`. The `/healthz` and `/metrics` endpoints are exempt from authentication.

API keys are stored in `llmstack.yaml` under `gateway.api_keys`. If the list is empty on first `llmstack up`, a key is generated automatically.

## CORS

Cross-Origin Resource Sharing is configured via `gateway.cors` in `llmstack.yaml`. The default `["*"]` allows all origins. For production, restrict this to your application's domain:

```yaml
gateway:
  cors:
    - "https://myapp.example.com"
    - "http://localhost:3000"
```
