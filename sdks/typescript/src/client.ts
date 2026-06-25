/**
 * LLMStack TypeScript client with OpenAI-compatible namespaced API.
 *
 * Zero runtime dependencies -- uses native `fetch` (Node 18+, Bun, Deno, browsers).
 */

import type {
  LLMStackClientOptions,
  ChatCompletionRequest,
  ChatCompletionResponse,
  ChatCompletionChunk,
  EmbeddingRequest,
  EmbeddingResponse,
  ModelsResponse,
  RAGIngestRequest,
  RAGIngestResponse,
  RAGQueryRequest,
  RAGQueryResponse,
  RAGStreamDelta,
  RAGStatusResponse,
  HealthResponse,
  SavingsSummary,
  OnboardingStatus,
} from "./types.js";
import { parseSSEStream } from "./streaming.js";

// ---------------------------------------------------------------------------
// Error
// ---------------------------------------------------------------------------

/** Error raised when the LLMStack API returns a non-2xx response. */
export class LLMStackError extends Error {
  /** HTTP status code. */
  readonly status: number;
  /** Raw response body (parsed JSON when possible, otherwise string). */
  readonly body: unknown;

  constructor(status: number, body: unknown) {
    const message =
      typeof body === "object" && body !== null && "detail" in body
        ? `HTTP ${status}: ${JSON.stringify((body as Record<string, unknown>).detail)}`
        : `HTTP ${status}: ${typeof body === "string" ? body : JSON.stringify(body)}`;
    super(message);
    this.name = "LLMStackError";
    this.status = status;
    this.body = body;
  }
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const DEFAULT_BASE_URL = "http://localhost:8000";
const DEFAULT_TIMEOUT = 120_000;
const DEFAULT_MAX_RETRIES = 2;
const RETRIABLE_STATUS_CODES = new Set([408, 429, 500, 502, 503, 504]);

// ---------------------------------------------------------------------------
// Client
// ---------------------------------------------------------------------------

/**
 * LLMStack API client.
 *
 * Exposes an OpenAI-compatible namespaced interface:
 *
 * ```ts
 * const client = new LLMStackClient({ baseUrl: "http://localhost:8000" });
 *
 * // Chat completions
 * const res = await client.chat.completions.create({ messages: [...] });
 *
 * // Streaming
 * const stream = await client.chat.completions.create({ messages: [...], stream: true });
 * for await (const chunk of stream) { ... }
 *
 * // Models
 * const models = await client.models.list();
 *
 * // RAG
 * await client.rag.ingest({ text: "...", source: "doc.md" });
 * const answer = await client.rag.query({ question: "..." });
 * const status = await client.rag.status();
 *
 * // Health
 * const health = await client.health();
 * ```
 */
export class LLMStackClient {
  private readonly _baseUrl: string;
  private readonly _apiKey: string | undefined;
  private readonly _timeout: number;
  private readonly _maxRetries: number;
  private readonly _fetch: typeof globalThis.fetch;

  /** OpenAI-compatible `chat.completions` namespace. */
  readonly chat: {
    completions: {
      create(
        request: ChatCompletionRequest & { stream: true },
        options?: RequestOptions,
      ): Promise<AsyncGenerator<ChatCompletionChunk, void, undefined>>;
      create(
        request: ChatCompletionRequest & { stream?: false | undefined },
        options?: RequestOptions,
      ): Promise<ChatCompletionResponse>;
      create(
        request: ChatCompletionRequest,
        options?: RequestOptions,
      ): Promise<ChatCompletionResponse | AsyncGenerator<ChatCompletionChunk, void, undefined>>;
    };
  };

  /** `models` namespace. */
  readonly models: {
    list(options?: RequestOptions): Promise<ModelsResponse>;
  };

  /** `rag` namespace. */
  readonly rag: {
    ingest(request: RAGIngestRequest, options?: RequestOptions): Promise<RAGIngestResponse>;
    query(
      request: RAGQueryRequest & { stream: true },
      options?: RequestOptions,
    ): Promise<AsyncGenerator<RAGStreamDelta, void, undefined>>;
    query(
      request: RAGQueryRequest & { stream?: false | undefined },
      options?: RequestOptions,
    ): Promise<RAGQueryResponse>;
    query(
      request: RAGQueryRequest,
      options?: RequestOptions,
    ): Promise<RAGQueryResponse | AsyncGenerator<RAGStreamDelta, void, undefined>>;
    status(options?: RequestOptions): Promise<RAGStatusResponse>;
  };

  constructor(options: LLMStackClientOptions = {}) {
    this._baseUrl = (
      options.baseUrl ??
      (typeof process !== "undefined" ? process.env?.["LLMSTACK_URL"] : undefined) ??
      DEFAULT_BASE_URL
    ).replace(/\/+$/, "");

    this._apiKey =
      options.apiKey ??
      (typeof process !== "undefined" ? process.env?.["LLMSTACK_API_KEY"] : undefined);

    this._timeout = options.timeout ?? DEFAULT_TIMEOUT;
    this._maxRetries = options.maxRetries ?? DEFAULT_MAX_RETRIES;
    this._fetch = options.fetch ?? globalThis.fetch;

    // Bind namespaced methods
    this.chat = {
      completions: {
        create: this._chatCreate.bind(this) as typeof this.chat.completions.create,
      },
    };

    this.models = {
      list: this._modelsList.bind(this),
    };

    this.rag = {
      ingest: this._ragIngest.bind(this),
      query: this._ragQuery.bind(this) as typeof this.rag.query,
      status: this._ragStatus.bind(this),
    };
  }

  // -----------------------------------------------------------------------
  // Public top-level methods
  // -----------------------------------------------------------------------

  /** Embed text. */
  async embed(
    request: EmbeddingRequest,
    options?: RequestOptions,
  ): Promise<EmbeddingResponse> {
    return this._request<EmbeddingResponse>("POST", "/v1/embeddings", request, options);
  }

  /** Gateway health check. */
  async health(options?: RequestOptions): Promise<HealthResponse> {
    return this._request<HealthResponse>("GET", "/healthz", undefined, options);
  }

  /**
   * Cumulative money saved by serving requests locally instead of paying a
   * cloud provider, valued against a dated cloud baseline.
   *
   * @param plan - Subscription to compare against (e.g. `"copilot-pro"`,
   *   `"cursor-pro"`). Defaults to the gateway's baseline when omitted.
   */
  async savings(plan?: string, options?: RequestOptions): Promise<SavingsSummary> {
    const path = plan
      ? `/v1/savings/summary?plan=${encodeURIComponent(plan)}`
      : "/v1/savings/summary";
    return this._request<SavingsSummary>("GET", path, undefined, options);
  }

  /**
   * First-run readiness for zero-key local inference: whether Ollama is up,
   * which models are present, the recommended chat/embed models, and the
   * concrete next-step hints to get to a working local setup.
   */
  async onboarding(
    ollamaUrl?: string,
    options?: RequestOptions,
  ): Promise<OnboardingStatus> {
    const path = ollamaUrl
      ? `/v1/onboarding?ollama_url=${encodeURIComponent(ollamaUrl)}`
      : "/v1/onboarding";
    return this._request<OnboardingStatus>("GET", path, undefined, options);
  }

  // -----------------------------------------------------------------------
  // Namespaced implementations
  // -----------------------------------------------------------------------

  private async _chatCreate(
    request: ChatCompletionRequest,
    options?: RequestOptions,
  ): Promise<ChatCompletionResponse | AsyncGenerator<ChatCompletionChunk, void, undefined>> {
    if (request.stream) {
      return this._chatStream(request, options);
    }
    const data = await this._request<Record<string, unknown>>(
      "POST",
      "/v1/chat/completions",
      { ...request, stream: false },
      options,
    );
    return this._buildChatResponse(data, options?._responseHeaders);
  }

  private async _chatStream(
    request: ChatCompletionRequest,
    options?: RequestOptions,
  ): Promise<AsyncGenerator<ChatCompletionChunk, void, undefined>> {
    const response = await this._rawRequest(
      "POST",
      "/v1/chat/completions",
      { ...request, stream: true },
      options,
    );
    return parseSSEStream<ChatCompletionChunk>(response, options?.signal);
  }

  private async _modelsList(options?: RequestOptions): Promise<ModelsResponse> {
    return this._request<ModelsResponse>("GET", "/v1/models", undefined, options);
  }

  private async _ragIngest(
    request: RAGIngestRequest,
    options?: RequestOptions,
  ): Promise<RAGIngestResponse> {
    return this._request<RAGIngestResponse>("POST", "/v1/rag/ingest", request, options);
  }

  private async _ragQuery(
    request: RAGQueryRequest,
    options?: RequestOptions,
  ): Promise<RAGQueryResponse | AsyncGenerator<RAGStreamDelta, void, undefined>> {
    if (request.stream) {
      const response = await this._rawRequest(
        "POST",
        "/v1/rag/query",
        { ...request, stream: true },
        options,
      );
      return parseSSEStream<RAGStreamDelta>(response, options?.signal);
    }
    return this._request<RAGQueryResponse>(
      "POST",
      "/v1/rag/query",
      { ...request, stream: false },
      options,
    );
  }

  private async _ragStatus(options?: RequestOptions): Promise<RAGStatusResponse> {
    return this._request<RAGStatusResponse>("GET", "/v1/rag/status", undefined, options);
  }

  // -----------------------------------------------------------------------
  // HTTP helpers
  // -----------------------------------------------------------------------

  private _headers(): Record<string, string> {
    const h: Record<string, string> = { "Content-Type": "application/json" };
    if (this._apiKey) {
      h["Authorization"] = `Bearer ${this._apiKey}`;
    }
    return h;
  }

  /**
   * Make an HTTP request with timeout, retries, and error handling.
   * Returns the parsed JSON body.
   */
  private async _request<T>(
    method: string,
    path: string,
    body?: unknown,
    options?: RequestOptions,
  ): Promise<T> {
    const response = await this._rawRequest(method, path, body, options);

    // Stash response headers for caller if they passed a holder.
    if (options?._responseHeaders) {
      response.headers.forEach((v, k) => {
        options._responseHeaders![k] = v;
      });
    }

    return (await response.json()) as T;
  }

  /**
   * Low-level request with retries. Returns the raw `Response`.
   */
  private async _rawRequest(
    method: string,
    path: string,
    body?: unknown,
    options?: RequestOptions,
  ): Promise<Response> {
    const url = `${this._baseUrl}${path}`;
    const maxAttempts = this._maxRetries + 1;
    let lastError: Error | undefined;

    for (let attempt = 0; attempt < maxAttempts; attempt++) {
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), this._timeout);

      // If the caller provides a signal, forward abort to our controller.
      const callerSignal = options?.signal;
      const onAbort = () => controller.abort();
      callerSignal?.addEventListener("abort", onAbort, { once: true });

      try {
        const init: RequestInit = {
          method,
          headers: { ...this._headers(), ...options?.headers },
          signal: controller.signal,
        };
        if (body !== undefined) {
          init.body = JSON.stringify(body);
        }

        const response = await this._fetch(url, init);

        if (!response.ok) {
          let errorBody: unknown;
          try {
            errorBody = await response.json();
          } catch {
            errorBody = await response.text().catch(() => "");
          }

          const err = new LLMStackError(response.status, errorBody);

          // Retry on transient errors, but not on the last attempt.
          if (RETRIABLE_STATUS_CODES.has(response.status) && attempt < maxAttempts - 1) {
            lastError = err;
            await sleep(retryDelay(attempt));
            continue;
          }
          throw err;
        }

        return response;
      } catch (error) {
        if (error instanceof LLMStackError) throw error;

        // Retry on network errors.
        const err = error instanceof Error ? error : new Error(String(error));
        if (attempt < maxAttempts - 1 && !callerSignal?.aborted) {
          lastError = err;
          await sleep(retryDelay(attempt));
          continue;
        }
        throw err;
      } finally {
        clearTimeout(timeout);
        callerSignal?.removeEventListener("abort", onAbort);
      }
    }

    // Should be unreachable, but just in case.
    throw lastError ?? new Error("Request failed");
  }

  /** Build a `ChatCompletionResponse` from raw JSON + response headers. */
  private _buildChatResponse(
    data: Record<string, unknown>,
    headers?: Record<string, string>,
  ): ChatCompletionResponse {
    return {
      id: (data.id as string) ?? "",
      object: "chat.completion",
      created: (data.created as number) ?? 0,
      model: (data.model as string) ?? "",
      choices: (data.choices as ChatCompletionResponse["choices"]) ?? [],
      usage: data.usage as ChatCompletionResponse["usage"],
      cached: (headers?.["x-cache"] ?? "").toUpperCase() === "HIT",
      cache_age: parseInt(headers?.["x-cache-age"] ?? "0", 10) || 0,
    };
  }
}

// ---------------------------------------------------------------------------
// Public helpers
// ---------------------------------------------------------------------------

/** Convenience factory that reads from environment variables. */
export function createClient(options?: LLMStackClientOptions): LLMStackClient {
  return new LLMStackClient(options);
}

// ---------------------------------------------------------------------------
// Request options
// ---------------------------------------------------------------------------

/** Per-request options. */
export interface RequestOptions {
  /** Abort signal to cancel the request. */
  signal?: AbortSignal;
  /** Extra headers merged into the request. */
  headers?: Record<string, string>;
  /** @internal holder for response headers (populated by _request). */
  _responseHeaders?: Record<string, string>;
}

// ---------------------------------------------------------------------------
// Internal utilities
// ---------------------------------------------------------------------------

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/** Exponential back-off with jitter. */
function retryDelay(attempt: number): number {
  const base = Math.min(1000 * 2 ** attempt, 8000);
  const jitter = Math.random() * base * 0.5;
  return base + jitter;
}
