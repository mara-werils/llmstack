# @llmstack/client

Official TypeScript/JavaScript SDK for [LLMStack](https://github.com/mara-werils/llmstack) -- an OpenAI-compatible AI gateway.

**Zero dependencies.** Uses native `fetch` (Node 18+, Bun, Deno, browsers).

## Installation

```bash
npm install @llmstack/client
```

## Quick Start

```typescript
import { LLMStackClient } from "@llmstack/client";

const client = new LLMStackClient({
  baseUrl: "http://localhost:8000",
  apiKey: "sk-...", // optional
});

// Chat completion
const response = await client.chat.completions.create({
  model: "llama3.2",
  messages: [{ role: "user", content: "Hello!" }],
});
console.log(response.choices[0].message.content);
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `LLMSTACK_URL` | `http://localhost:8000` | Gateway URL |
| `LLMSTACK_API_KEY` | -- | Bearer API key (if auth is enabled) |

When no options are provided, the client reads these automatically:

```typescript
const client = new LLMStackClient(); // uses env vars
```

## API Reference

### Constructor

```typescript
new LLMStackClient({
  baseUrl?: string,       // default: LLMSTACK_URL or "http://localhost:8000"
  apiKey?: string,        // default: LLMSTACK_API_KEY
  timeout?: number,       // default: 120000 (2 minutes)
  maxRetries?: number,    // default: 2 (retries on 5xx/network errors)
  fetch?: typeof fetch,   // custom fetch implementation
});
```

### `client.chat.completions.create(request)`

OpenAI-compatible chat completion.

```typescript
// Non-streaming
const res = await client.chat.completions.create({
  model: "llama3.2",
  messages: [
    { role: "system", content: "You are helpful." },
    { role: "user", content: "Explain quantum computing." },
  ],
  temperature: 0.7,
  max_tokens: 1024,
});

console.log(res.choices[0].message.content);
console.log(res.usage); // { prompt_tokens, completion_tokens, total_tokens }
console.log(res.cached); // true if served from cache
```

### Streaming

```typescript
const stream = await client.chat.completions.create({
  model: "llama3.2",
  messages: [{ role: "user", content: "Tell me a story" }],
  stream: true,
});

for await (const chunk of stream) {
  const content = chunk.choices[0]?.delta?.content;
  if (content) process.stdout.write(content);
}
```

### `client.embed(request)`

Generate text embeddings.

```typescript
const res = await client.embed({
  input: ["Hello world", "Goodbye world"],
  model: "bge-m3",
});

console.log(res.data[0].embedding); // number[]
```

### `client.models.list()`

List available models.

```typescript
const { data } = await client.models.list();
for (const model of data) {
  console.log(`${model.id} (${model.owned_by})`);
}
```

### `client.rag.ingest(request)`

Ingest a document into the RAG store.

```typescript
const res = await client.rag.ingest({
  text: "LLMStack is an open-source AI gateway...",
  source: "docs/overview.md",
  chunk_size: 512,
  metadata: { category: "docs" },
});

console.log(res.chunks_stored); // number of chunks created
```

### `client.rag.query(request)`

Query the RAG pipeline.

```typescript
// Non-streaming
const answer = await client.rag.query({
  question: "What is LLMStack?",
  top_k: 5,
});
console.log(answer.answer);
console.log(answer.sources);

// Streaming
const stream = await client.rag.query({
  question: "What is LLMStack?",
  stream: true,
});
for await (const delta of stream) {
  if (delta.token) process.stdout.write(delta.token);
  if (delta.done) console.log("\nSources:", delta.sources);
}
```

### `client.rag.status()`

Check RAG pipeline status.

```typescript
const status = await client.rag.status();
console.log(status.documents_count, status.chunks_count);
```

### `client.health()`

Gateway health check.

```typescript
const health = await client.health();
console.log(health.status);       // "healthy"
console.log(health.services);     // { inference: true, rag: true, ... }
console.log(health.circuit_breaker);
```

### `client.savings(plan?)`

Cumulative money saved by serving requests locally instead of paying a cloud
provider, valued against a dated cloud baseline.

```typescript
const savings = await client.savings("cursor-pro");
console.log(savings.total_saved_usd);              // 12.34
console.log(savings.subscription.months_covered);  // e.g. 0.6 months of Cursor Pro
```

### `client.onboarding(ollamaUrl?)` / `client.ready(ollamaUrl?)`

First-run readiness for zero-key local inference: whether Ollama is up, which
models are present, the recommended models, and next-step hints. `ready()` is a
boolean shortcut over `onboarding().ready`.

```typescript
if (!(await client.ready())) {
  const status = await client.onboarding();
  console.log(status.hints); // e.g. ["ollama pull llama3.2", ...]
}
```

### `client.ask(question, model?)` / `client.complete(prompt, model?, system?)`

One-liner helpers that return the reply text directly.

```typescript
const answer = await client.ask("What is machine learning?");
const code = await client.complete("Refactor this loop", "llama3.2", "You are a senior engineer.");
```

### `BatchProcessor`

Run many prompts concurrently with a bounded concurrency.

```typescript
import { LLMStackClient, BatchProcessor } from "@llmstack/client";

const processor = new BatchProcessor(new LLMStackClient(), 5);
const summary = await processor.run([
  { id: 1, prompt: "Summarize REST." },
  { id: 2, prompt: "What is a closure?" },
]);
console.log(`${summary.completed}/${summary.total} done`);
```

## Cancellation

Every request accepts an `AbortSignal`:

```typescript
const controller = new AbortController();
setTimeout(() => controller.abort(), 5000);

const res = await client.chat.completions.create(
  { model: "llama3.2", messages: [{ role: "user", content: "Hi" }] },
  { signal: controller.signal },
);
```

## Error Handling

```typescript
import { LLMStackError } from "@llmstack/client";

try {
  await client.chat.completions.create({ messages: [] });
} catch (err) {
  if (err instanceof LLMStackError) {
    console.error(err.status); // HTTP status code
    console.error(err.body);   // parsed error body
  }
}
```

## Retries

The client automatically retries on transient failures (status 408, 429, 500, 502, 503, 504) and network errors, up to `maxRetries` times with exponential back-off and jitter.

## Custom Fetch

Useful for testing or environments with a non-standard `fetch`:

```typescript
const client = new LLMStackClient({
  fetch: myCustomFetch,
});
```

## Building from Source

```bash
npm install
npm run build     # outputs to dist/
npm run typecheck  # type-check only
```

## License

Apache-2.0
