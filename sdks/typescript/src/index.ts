export { LLMStackClient, LLMStackError, createClient } from "./client.js";
export type { RequestOptions } from "./client.js";
export { parseSSEStream } from "./streaming.js";
export { BatchProcessor } from "./batch.js";
export type { BatchItem, BatchResult, BatchSummary } from "./batch.js";
export type {
  LLMStackClientOptions,
  // Chat
  ChatMessage,
  ChatCompletionRequest,
  ChatCompletionResponse,
  ChatChoice,
  ChatDelta,
  ChatChunkChoice,
  ChatCompletionChunk,
  Usage,
  // Embeddings
  EmbeddingRequest,
  EmbeddingResponse,
  Embedding,
  // Models
  Model,
  ModelsResponse,
  // RAG
  RAGIngestRequest,
  RAGIngestResponse,
  RAGQueryRequest,
  RAGQueryResponse,
  RAGStreamDelta,
  RAGStatusResponse,
  // Health
  HealthResponse,
  // Savings
  SavingsSummary,
  SavingsSubscription,
  // Onboarding
  OnboardingStatus,
  OnboardingModel,
  // Errors
  LLMStackErrorDetail,
} from "./types.js";
