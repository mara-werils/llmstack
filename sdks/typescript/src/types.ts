// ---------------------------------------------------------------------------
// Client options
// ---------------------------------------------------------------------------

/** Configuration for the LLMStack client. */
export interface LLMStackClientOptions {
  /** Gateway base URL. Defaults to `LLMSTACK_URL` env var or `http://localhost:8000`. */
  baseUrl?: string;
  /** Bearer API key. Defaults to `LLMSTACK_API_KEY` env var. */
  apiKey?: string;
  /** Request timeout in milliseconds. Default: 120000 (2 minutes). */
  timeout?: number;
  /** Maximum number of retries on transient failures (5xx, network). Default: 2. */
  maxRetries?: number;
  /** Custom `fetch` implementation. Defaults to the global `fetch`. */
  fetch?: typeof globalThis.fetch;
}

// ---------------------------------------------------------------------------
// Chat completion
// ---------------------------------------------------------------------------

/** A single message in a chat conversation. */
export interface ChatMessage {
  role: "system" | "user" | "assistant" | "function" | "tool";
  content: string | null;
  name?: string;
}

/** Request body for `POST /v1/chat/completions`. */
export interface ChatCompletionRequest {
  messages: ChatMessage[];
  model?: string;
  stream?: boolean;
  temperature?: number;
  top_p?: number;
  max_tokens?: number;
  stop?: string | string[];
  frequency_penalty?: number;
  presence_penalty?: number;
  /** Extra fields forwarded to the backend. */
  [key: string]: unknown;
}

/** Token usage statistics. */
export interface Usage {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
}

/** One completion choice (non-streaming). */
export interface ChatChoice {
  index: number;
  message: ChatMessage;
  finish_reason: string | null;
}

/** Non-streaming response from `/v1/chat/completions`. */
export interface ChatCompletionResponse {
  id: string;
  object: "chat.completion";
  created: number;
  model: string;
  choices: ChatChoice[];
  usage?: Usage;
  /** Whether this response was served from cache. */
  cached: boolean;
  /** Age of the cached response in seconds, or 0 if not cached. */
  cache_age: number;
}

/** Delta object within a streaming chunk. */
export interface ChatDelta {
  role?: string;
  content?: string;
}

/** One streaming chunk choice. */
export interface ChatChunkChoice {
  index: number;
  delta: ChatDelta;
  finish_reason: string | null;
}

/** A single SSE chunk from a streaming chat response. */
export interface ChatCompletionChunk {
  id: string;
  object: "chat.completion.chunk";
  created: number;
  model: string;
  choices: ChatChunkChoice[];
}

// ---------------------------------------------------------------------------
// Embeddings
// ---------------------------------------------------------------------------

/** Request body for `POST /v1/embeddings`. */
export interface EmbeddingRequest {
  input: string | string[];
  model?: string;
}

/** A single embedding vector. */
export interface Embedding {
  index: number;
  embedding: number[];
  object: "embedding";
}

/** Response from `/v1/embeddings`. */
export interface EmbeddingResponse {
  object: "list";
  model: string;
  data: Embedding[];
  usage: Usage;
}

// ---------------------------------------------------------------------------
// Models
// ---------------------------------------------------------------------------

/** A single model entry. */
export interface Model {
  id: string;
  object: "model";
  owned_by: string;
  created: number;
}

/** Response from `GET /v1/models`. */
export interface ModelsResponse {
  object: "list";
  data: Model[];
}

// ---------------------------------------------------------------------------
// RAG
// ---------------------------------------------------------------------------

/** Request body for `POST /v1/rag/ingest`. */
export interface RAGIngestRequest {
  text: string;
  source: string;
  chunk_size?: number;
  metadata?: Record<string, unknown>;
}

/** Response from `/v1/rag/ingest`. */
export interface RAGIngestResponse {
  status: string;
  chunks_stored: number;
  source: string;
}

/** Request body for `POST /v1/rag/query`. */
export interface RAGQueryRequest {
  question: string;
  top_k?: number;
  stream?: boolean;
  model?: string;
  temperature?: number;
  /** Extra fields forwarded to the backend. */
  [key: string]: unknown;
}

/** Non-streaming response from `/v1/rag/query`. */
export interface RAGQueryResponse {
  answer: string;
  sources: string[];
  model: string;
  usage: Record<string, unknown>;
  latency: number;
}

/** A single SSE chunk from a streaming RAG response. */
export interface RAGStreamDelta {
  token: string | null;
  done: boolean;
  sources: string[];
}

/** Response from `GET /v1/rag/status`. */
export interface RAGStatusResponse {
  status: string;
  documents_count: number;
  chunks_count: number;
  [key: string]: unknown;
}

// ---------------------------------------------------------------------------
// Health
// ---------------------------------------------------------------------------

/** Response from `GET /healthz`. */
export interface HealthResponse {
  status: string;
  services: Record<string, boolean>;
  circuit_breaker: Record<string, unknown>;
  cache: Record<string, unknown>;
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/** Error details returned by the API. */
export interface LLMStackErrorDetail {
  message?: string;
  type?: string;
  [key: string]: unknown;
}
