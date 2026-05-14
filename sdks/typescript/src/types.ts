export interface Message {
  role: "system" | "user" | "assistant";
  content: string;
}

export interface ChatRequest {
  messages: Message[];
  model?: string;
  stream?: boolean;
  temperature?: number;
  max_tokens?: number;
}

export interface ChatChoice {
  message: Message;
  finish_reason: string;
  index: number;
}

export interface ChatResponse {
  id: string;
  object: "chat.completion";
  created: number;
  model: string;
  choices: ChatChoice[];
  usage?: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
}

export interface ChatChunk {
  id: string;
  object: "chat.completion.chunk";
  created: number;
  model: string;
  choices: Array<{
    delta: { content?: string; role?: string };
    finish_reason: string | null;
    index: number;
  }>;
}

export interface EmbeddingRequest {
  input: string | string[];
  model?: string;
}

export interface EmbeddingResponse {
  data: Array<{
    embedding: number[];
    index: number;
    object: "embedding";
  }>;
  model: string;
  usage: { prompt_tokens: number; total_tokens: number };
}

export interface RouterClassifyRequest {
  messages: Message[];
  model?: string;
}

export interface RouterClassifyResponse {
  tier: "simple" | "medium" | "complex";
  score: number;
  recommended_model: string;
}

export interface RAGIngestRequest {
  documents: Array<{ content: string; source: string }>;
}

export interface RAGQueryRequest {
  query: string;
  top_k?: number;
  stream?: boolean;
}

export interface LLMStackClientOptions {
  baseUrl?: string;
  apiKey?: string;
  timeout?: number;
}
