# OpenAI-Compatible API

llmstack exposes an OpenAI-compatible API through its gateway. Any client that works with the OpenAI API will work with llmstack by changing the `base_url` to `http://localhost:8000/v1`.

## Base URL

```
http://localhost:8000/v1
```

The port is configurable via `gateway.port` in `llmstack.yaml`.

## Authentication

All endpoints (except `/healthz` and `/metrics`) require a Bearer token when `gateway.auth` is set to `api_key`:

```
Authorization: Bearer sk-llmstack-...
```

## Chat Completions

Create a chat completion with one or more messages.

### `POST /v1/chat/completions`

**Request Body**

| Field | Type | Required | Description |
|---|---|---|---|
| `model` | string | Yes | Model name (e.g., `"llama3.2"`) |
| `messages` | array | Yes | Array of message objects with `role` and `content` |
| `stream` | boolean | No | Enable Server-Sent Events streaming (default: `false`) |
| `temperature` | float | No | Sampling temperature (default: model-dependent). Requests with temperature <= 0.1 are eligible for caching. |
| `max_tokens` | integer | No | Maximum tokens to generate |
| `top_p` | float | No | Nucleus sampling parameter |
| `stop` | string or array | No | Stop sequences |

**Message Object**

| Field | Type | Description |
|---|---|---|
| `role` | string | One of `"system"`, `"user"`, `"assistant"` |
| `content` | string | The message text |

**Example Request**

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is Docker?"}
    ]
  }'
```

**Example Response**

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1715100000,
  "model": "llama3.2",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Docker is a platform for developing, shipping, and running applications in containers..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 25,
    "completion_tokens": 150,
    "total_tokens": 175
  }
}
```

**Response Headers**

| Header | Description |
|---|---|
| `X-Cache` | `HIT` if the response came from cache, `MISS` otherwise |
| `X-Request-ID` | Unique request identifier for log correlation |
| `X-RateLimit-Limit` | Rate limit ceiling |
| `X-RateLimit-Remaining` | Remaining requests in the current window |

### Streaming

When `stream: true`, the response uses Server-Sent Events:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```

Each event is a `data:` line containing a JSON chunk:

```
data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"!"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

## Embeddings

Generate vector embeddings for text inputs.

### `POST /v1/embeddings`

**Request Body**

| Field | Type | Required | Description |
|---|---|---|---|
| `model` | string | Yes | Embedding model name (e.g., `"bge-m3"`) |
| `input` | string or array | Yes | Text(s) to embed. Can be a single string or an array of strings. |

**Example Request**

```bash
curl http://localhost:8000/v1/embeddings \
  -H "Authorization: Bearer YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "bge-m3",
    "input": ["Hello world", "How are you?"]
  }'
```

**Example Response**

```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "index": 0,
      "embedding": [0.0023, -0.0091, 0.0152, ...]
    },
    {
      "object": "embedding",
      "index": 1,
      "embedding": [0.0041, -0.0033, 0.0087, ...]
    }
  ],
  "model": "bge-m3",
  "usage": {
    "prompt_tokens": 8,
    "total_tokens": 8
  }
}
```

## Models

List available models.

### `GET /v1/models`

**Example Request**

```bash
curl http://localhost:8000/v1/models \
  -H "Authorization: Bearer YOUR_KEY"
```

**Example Response**

```json
{
  "object": "list",
  "data": [
    {
      "id": "llama3.2",
      "object": "model",
      "created": 1715100000,
      "owned_by": "library"
    },
    {
      "id": "bge-m3",
      "object": "model",
      "created": 1715100000,
      "owned_by": "library"
    }
  ]
}
```

## Error Responses

All errors follow a consistent format:

```json
{
  "error": {
    "message": "Description of the error",
    "type": "error_type",
    "code": "error_code"
  }
}
```

### Status Codes

| Code | Meaning | Common Cause |
|---|---|---|
| `401` | Unauthorized | Missing or invalid API key |
| `429` | Too Many Requests | Rate limit exceeded |
| `500` | Internal Server Error | Inference backend error |
| `503` | Service Unavailable | Circuit breaker is open (inference backend is down) |
| `504` | Gateway Timeout | Request exceeded `request_timeout` |
