import type {
  ChatRequest, ChatResponse, ChatChunk,
  EmbeddingRequest, EmbeddingResponse,
  RouterClassifyRequest, RouterClassifyResponse,
  RAGIngestRequest, RAGQueryRequest,
  LLMStackClientOptions, Message,
} from "./types.js";

export class LLMStackClient {
  private baseUrl: string;
  private apiKey: string | undefined;
  private timeout: number;

  constructor(options: LLMStackClientOptions = {}) {
    this.baseUrl = (options.baseUrl ?? "http://localhost:8000").replace(/\/$/, "");
    this.apiKey = options.apiKey;
    this.timeout = options.timeout ?? 60000;
  }

  private headers(): Record<string, string> {
    const h: Record<string, string> = { "Content-Type": "application/json" };
    if (this.apiKey) h["Authorization"] = `Bearer ${this.apiKey}`;
    return h;
  }

  private async fetchJSON<T>(path: string, init?: RequestInit): Promise<T> {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.timeout);
    try {
      const res = await fetch(`${this.baseUrl}${path}`, {
        ...init,
        headers: { ...this.headers(), ...(init?.headers ?? {}) },
        signal: controller.signal,
      });
      if (!res.ok) {
        const body = await res.text();
        throw new Error(`HTTP ${res.status}: ${body}`);
      }
      return res.json() as Promise<T>;
    } finally {
      clearTimeout(timer);
    }
  }

  /** Chat completion (non-streaming). */
  async chat(request: ChatRequest): Promise<ChatResponse> {
    return this.fetchJSON<ChatResponse>("/v1/chat/completions", {
      method: "POST",
      body: JSON.stringify({ ...request, stream: false }),
    });
  }

  /** Chat completion with streaming — yields text chunks. */
  async *chatStream(request: ChatRequest): AsyncGenerator<string, void, unknown> {
    const res = await fetch(`${this.baseUrl}/v1/chat/completions`, {
      method: "POST",
      headers: this.headers(),
      body: JSON.stringify({ ...request, stream: true }),
    });
    if (!res.ok || !res.body) throw new Error(`HTTP ${res.status}`);

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const lines = buf.split("\n");
      buf = lines.pop() ?? "";
      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed || trimmed === "data: [DONE]") continue;
        const data = trimmed.startsWith("data: ") ? trimmed.slice(6) : trimmed;
        try {
          const chunk = JSON.parse(data) as ChatChunk;
          const content = chunk.choices[0]?.delta?.content;
          if (content) yield content;
        } catch { /* skip malformed lines */ }
      }
    }
  }

  /** Get embeddings for text. */
  async embed(request: EmbeddingRequest): Promise<EmbeddingResponse> {
    return this.fetchJSON<EmbeddingResponse>("/v1/embeddings", {
      method: "POST",
      body: JSON.stringify(request),
    });
  }

  /** Classify a query using the smart router. */
  async classify(request: RouterClassifyRequest): Promise<RouterClassifyResponse> {
    return this.fetchJSON<RouterClassifyResponse>("/v1/router/classify", {
      method: "POST",
      body: JSON.stringify(request),
    });
  }

  /** Ingest documents into RAG. */
  async ragIngest(request: RAGIngestRequest): Promise<{ success: boolean; chunks_created: number }> {
    return this.fetchJSON("/v1/rag/ingest", {
      method: "POST",
      body: JSON.stringify(request),
    });
  }

  /** Query RAG knowledge base. */
  async ragQuery(request: RAGQueryRequest): Promise<{ answer: string; sources: string[] }> {
    return this.fetchJSON("/v1/rag/query", {
      method: "POST",
      body: JSON.stringify({ ...request, stream: false }),
    });
  }

  /** Health check. */
  async health(): Promise<Record<string, unknown>> {
    return this.fetchJSON("/healthz");
  }

  /** List available models. */
  async models(): Promise<{ data: Array<{ id: string; owned_by: string }> }> {
    return this.fetchJSON("/v1/models");
  }
}

/** Convenience: create client with environment variables. */
export function createClient(options?: LLMStackClientOptions): LLMStackClient {
  return new LLMStackClient({
    baseUrl: process.env["LLMSTACK_URL"] ?? options?.baseUrl ?? "http://localhost:8000",
    apiKey: process.env["LLMSTACK_API_KEY"] ?? options?.apiKey,
    ...options,
  });
}
