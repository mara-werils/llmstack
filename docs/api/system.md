# System API

System endpoints for health checking and metrics collection. These endpoints do not require authentication.

## Health Check

### `GET /healthz`

Returns the health status of the gateway and its dependencies. This endpoint is exempt from authentication and rate limiting.

**Example Request**

```bash
curl http://localhost:8000/healthz
```

**Example Response**

```json
{
  "status": "healthy",
  "version": "0.3.0",
  "inference": {
    "status": "healthy",
    "backend": "ollama",
    "model": "llama3.2"
  },
  "cache": {
    "status": "connected",
    "hits": 142,
    "misses": 58
  },
  "circuit_breaker": {
    "state": "closed",
    "failure_count": 0,
    "rejection_count": 0
  },
  "vectordb": {
    "status": "connected",
    "points": 1234
  }
}
```

**Response Fields**

| Field | Description |
|---|---|
| `status` | Overall gateway health: `healthy` or `unhealthy` |
| `version` | Gateway version |
| `inference.status` | Inference backend health |
| `inference.backend` | Active inference backend (ollama or vllm) |
| `inference.model` | Loaded model name |
| `cache.status` | Redis connection status |
| `cache.hits` | Total cache hits since startup |
| `cache.misses` | Total cache misses since startup |
| `circuit_breaker.state` | Current state: `closed`, `open`, or `half_open` |
| `circuit_breaker.failure_count` | Consecutive failures |
| `circuit_breaker.rejection_count` | Total requests rejected by the circuit breaker |
| `vectordb.status` | Qdrant connection status |
| `vectordb.points` | Total stored vectors |

### Use Cases

**Load balancer health checks**: Point your load balancer's health check at `/healthz`. It returns `200` when healthy and `503` when unhealthy.

**Monitoring**: Poll `/healthz` periodically to track cache performance and circuit breaker state without needing Prometheus.

**Debugging**: When requests fail, check `/healthz` first. If the circuit breaker is `open`, the inference backend is down. If cache status is `disconnected`, Redis may have stopped.

## Metrics

### `GET /metrics`

Exposes metrics in Prometheus exposition format. This endpoint is exempt from authentication and rate limiting.

**Example Request**

```bash
curl http://localhost:8000/metrics
```

**Example Response**

```
# HELP llmstack_requests_total Total HTTP requests
# TYPE llmstack_requests_total counter
llmstack_requests_total{method="POST",path="/v1/chat/completions",status="200"} 1542
llmstack_requests_total{method="POST",path="/v1/rag/query",status="200"} 89
llmstack_requests_total{method="GET",path="/v1/models",status="200"} 23

# HELP llmstack_request_duration_seconds Request latency histogram
# TYPE llmstack_request_duration_seconds histogram
llmstack_request_duration_seconds_bucket{le="0.1",path="/v1/chat/completions"} 142
llmstack_request_duration_seconds_bucket{le="0.5",path="/v1/chat/completions"} 890
llmstack_request_duration_seconds_bucket{le="1.0",path="/v1/chat/completions"} 1200
llmstack_request_duration_seconds_bucket{le="5.0",path="/v1/chat/completions"} 1530
llmstack_request_duration_seconds_bucket{le="+Inf",path="/v1/chat/completions"} 1542

# HELP llmstack_tokens_input_total Total input tokens
# TYPE llmstack_tokens_input_total counter
llmstack_tokens_input_total 45230

# HELP llmstack_tokens_output_total Total output tokens
# TYPE llmstack_tokens_output_total counter
llmstack_tokens_output_total 128450

# HELP llmstack_cache_hits_total Cache hits
# TYPE llmstack_cache_hits_total counter
llmstack_cache_hits_total 142

# HELP llmstack_cache_misses_total Cache misses
# TYPE llmstack_cache_misses_total counter
llmstack_cache_misses_total 1400

# HELP llmstack_rate_limit_rejections_total Rate limit rejections
# TYPE llmstack_rate_limit_rejections_total counter
llmstack_rate_limit_rejections_total 7

# HELP llmstack_circuit_breaker_state Circuit breaker state
# TYPE llmstack_circuit_breaker_state gauge
llmstack_circuit_breaker_state 0
```

### Metrics Reference

| Metric | Type | Labels | Description |
|---|---|---|---|
| `llmstack_requests_total` | Counter | method, path, status | Total HTTP requests |
| `llmstack_request_duration_seconds` | Histogram | path | Request latency |
| `llmstack_requests_in_progress` | Gauge | -- | Currently active requests |
| `llmstack_tokens_input_total` | Counter | -- | Total input tokens processed |
| `llmstack_tokens_output_total` | Counter | -- | Total output tokens generated |
| `llmstack_cache_hits_total` | Counter | -- | Cache hits |
| `llmstack_cache_misses_total` | Counter | -- | Cache misses |
| `llmstack_rate_limit_rejections_total` | Counter | -- | Requests rejected by rate limiter |
| `llmstack_circuit_breaker_state` | Gauge | -- | 0=closed, 1=open, 2=half_open |
| `llmstack_circuit_breaker_failures` | Counter | -- | Total backend failures |
| `llmstack_circuit_breaker_rejections` | Counter | -- | Requests rejected by circuit breaker |

### Scraping with Prometheus

The built-in Prometheus instance is pre-configured to scrape `/metrics` every 15 seconds. If you use your own Prometheus, add this scrape config:

```yaml
scrape_configs:
  - job_name: llmstack
    scrape_interval: 15s
    static_configs:
      - targets: ["localhost:8000"]
    metrics_path: /metrics
```
