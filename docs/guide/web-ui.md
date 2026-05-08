# Web UI

llmstack provides built-in web interfaces for monitoring and managing your stack.

## Grafana Dashboard

The primary web interface is the Grafana dashboard, which is provisioned automatically when observability is enabled.

### Accessing the Dashboard

After running `llmstack up`, open your browser to:

```
http://localhost:8080
```

Default credentials:

| Field | Value |
|---|---|
| Username | `admin` |
| Password | `llmstack` |

Anonymous read-only access is enabled by default, so you can view dashboards without logging in.

### Dashboard Overview

The pre-built dashboard provides a comprehensive view of your stack:

**Request Rate** -- Real-time requests per second, broken down by endpoint. Instantly see how much traffic your API is handling.

**Latency Percentiles** -- p50 and p99 response time histograms. The p50 shows typical performance; the p99 shows worst-case latency that some users experience.

**Token Throughput** -- Input and output tokens per second flowing through the inference backend. Useful for capacity planning.

**Error Rate** -- 4xx (client errors) and 5xx (server errors) over time. Spikes here need attention.

**Cache Hit Rate** -- Percentage of requests served from the Redis cache. Higher is better -- it means lower latency and reduced inference load.

**Circuit Breaker State** -- Timeline showing whether the inference backend is healthy (CLOSED), down (OPEN), or recovering (HALF_OPEN).

**Rate Limit Rejections** -- Count of requests rejected by the rate limiter, broken down by client.

### Customizing the Dashboard

You can modify the Grafana dashboard through the web UI. Changes made through the UI persist in the Grafana container's storage. If you want to persist changes across `llmstack down` / `llmstack up` cycles, export the dashboard JSON and mount it as a provisioned dashboard.

## Qdrant Dashboard

Qdrant provides its own web dashboard for inspecting vector collections:

```
http://localhost:6333/dashboard
```

This lets you:

- Browse collections and their configurations
- View point counts and index status
- Run similarity searches manually
- Inspect individual vectors and their payloads

This is useful for debugging RAG ingestion issues -- you can verify that documents were chunked and stored correctly.

## Prometheus UI

Prometheus has a built-in expression browser at:

```
http://localhost:9090
```

Use it to:

- Run ad-hoc PromQL queries
- Browse available metrics
- Check scrape target health
- View alerting rules (if configured)

### Useful Queries

Request rate over the last 5 minutes:

```promql
rate(llmstack_requests_total[5m])
```

p99 latency by endpoint:

```promql
histogram_quantile(0.99, rate(llmstack_request_duration_seconds_bucket[5m]))
```

Cache hit ratio:

```promql
rate(llmstack_cache_hits_total[5m]) /
(rate(llmstack_cache_hits_total[5m]) + rate(llmstack_cache_misses_total[5m]))
```

## Port Reference

| Service | URL | Description |
|---|---|---|
| Gateway API | `http://localhost:8000` | OpenAI-compatible API |
| Grafana | `http://localhost:8080` | Monitoring dashboard |
| Prometheus | `http://localhost:9090` | Metrics query UI |
| Qdrant | `http://localhost:6333/dashboard` | Vector DB dashboard |
| Ollama | `http://localhost:11434` | Inference API (direct) |

All ports are configurable in `llmstack.yaml`. The values above are the defaults.
