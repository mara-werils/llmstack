# @llmstack/client

Official TypeScript/JavaScript SDK for [LLMStack](https://github.com/mara-werils/llmstack).

## Installation

```bash
npm install @llmstack/client
```

## Quick Start

```typescript
import { createClient } from "@llmstack/client";

const client = createClient({ baseUrl: "http://localhost:8000" });

// Chat completion
const response = await client.chat({
  messages: [{ role: "user", content: "Hello!" }],
  model: "llama3.2",
});
console.log(response.choices[0].message.content);

// Streaming
for await (const chunk of client.chatStream({
  messages: [{ role: "user", content: "Tell me a story" }],
})) {
  process.stdout.write(chunk);
}

// Embeddings
const embeddings = await client.embed({ input: "Hello world" });

// RAG
await client.ragIngest({
  documents: [{ content: "LLMStack is awesome", source: "docs.md" }],
});
const answer = await client.ragQuery({ query: "What is LLMStack?" });
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LLMSTACK_URL` | `http://localhost:8000` | Gateway URL |
| `LLMSTACK_API_KEY` | — | API key (if auth enabled) |

## API Reference

### `createClient(options?)`

Create a new client instance. Options:
- `baseUrl` — Gateway URL
- `apiKey` — Bearer token
- `timeout` — Request timeout in ms (default: 60000)

### Methods

- `client.chat(request)` — Non-streaming chat completion
- `client.chatStream(request)` — Streaming chat (AsyncGenerator)
- `client.embed(request)` — Get text embeddings
- `client.classify(request)` — Smart router classification
- `client.ragIngest(request)` — Ingest documents into RAG
- `client.ragQuery(request)` — Query RAG knowledge base
- `client.health()` — Health check
- `client.models()` — List available models
