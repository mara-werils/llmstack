/**
 * Vercel AI SDK with llmstack — Next.js API route example.
 *
 * llmstack's OpenAI-compatible API works seamlessly with the Vercel AI SDK.
 * This example shows a Next.js API route that streams chat responses from
 * your local llmstack instance.
 *
 * Install:
 *   npm install ai openai @ai-sdk/openai
 *
 * Usage:
 *   1. Start llmstack:  llmstack up
 *   2. Add this file as app/api/chat/route.ts in your Next.js project
 *   3. Set environment variables:
 *        LLMSTACK_URL=http://localhost:8000/v1
 *        LLMSTACK_API_KEY=llmstack
 *   4. Run your Next.js app:  npm run dev
 */

import { streamText, generateText, embed, embedMany } from "ai";
import { createOpenAI } from "@ai-sdk/openai";

// ── Configure the provider ──────────────────────────────────────────
// Point the OpenAI-compatible provider at your llmstack gateway.
const llmstack = createOpenAI({
  baseURL: process.env.LLMSTACK_URL || "http://localhost:8000/v1",
  apiKey: process.env.LLMSTACK_API_KEY || "llmstack",
});

// ── 1. Streaming chat — Next.js API route (POST /api/chat) ──────────
// This is the standard pattern for Vercel AI SDK chat UIs.
export async function POST(req: Request) {
  const { messages } = await req.json();

  const result = streamText({
    model: llmstack("llama3.2"),
    messages,
    temperature: 0.4,
    maxTokens: 1024,
  });

  return result.toDataStreamResponse();
}

// ── 2. Non-streaming generation ─────────────────────────────────────
// Useful for server-side generation (e.g., background jobs, cron tasks).
async function generateExample() {
  const result = await generateText({
    model: llmstack("llama3.2"),
    prompt: "Explain the CAP theorem in 2 sentences.",
    temperature: 0.2,
    maxTokens: 256,
  });

  console.log("[Generate]");
  console.log(result.text);
  console.log(`Tokens: ${result.usage.promptTokens} in / ${result.usage.completionTokens} out`);
}

// ── 3. Embeddings ───────────────────────────────────────────────────
// Generate embeddings through llmstack's OpenAI-compatible endpoint.
async function embeddingExample() {
  // Single embedding
  const single = await embed({
    model: llmstack.embedding("bge-m3"),
    value: "What is retrieval-augmented generation?",
  });
  console.log("[Single embedding]");
  console.log(`Dimensions: ${single.embedding.length}`);

  // Batch embeddings
  const batch = await embedMany({
    model: llmstack.embedding("bge-m3"),
    values: [
      "LLMStack manages your entire LLM stack.",
      "Vector databases enable semantic search.",
      "Redis provides caching and rate limiting.",
    ],
  });
  console.log("[Batch embeddings]");
  console.log(`Generated ${batch.embeddings.length} embeddings`);
}

// ── 4. Streaming with callbacks ─────────────────────────────────────
// Process tokens as they stream in — useful for logging, analytics, etc.
async function streamWithCallbacks() {
  const result = streamText({
    model: llmstack("llama3.2"),
    prompt: "Write a short poem about APIs.",
    temperature: 0.7,
    maxTokens: 200,
    onChunk({ chunk }) {
      if (chunk.type === "text-delta") {
        process.stdout.write(chunk.textDelta);
      }
    },
    async onFinish({ text, usage }) {
      console.log("\n[Stream finished]");
      console.log(`Total tokens: ${usage.promptTokens + usage.completionTokens}`);
    },
  });

  // Consume the stream
  for await (const _chunk of result.textStream) {
    // onChunk handles output
  }
}

// ── 5. System prompts and multi-turn ─────────────────────────────────
async function multiTurnExample() {
  const result = await generateText({
    model: llmstack("llama3.2"),
    messages: [
      { role: "system", content: "You are a helpful coding assistant. Be concise." },
      { role: "user", content: "What is a closure in JavaScript?" },
      {
        role: "assistant",
        content:
          "A closure is a function that retains access to variables from its outer scope, even after the outer function has returned.",
      },
      { role: "user", content: "Give me a practical example." },
    ],
    temperature: 0.3,
    maxTokens: 512,
  });

  console.log("[Multi-turn]");
  console.log(result.text);
}

// ── Run examples (when executed directly with tsx/ts-node) ───────────
// These run when you execute this file directly, not when imported as a route.
async function main() {
  console.log("Vercel AI SDK + llmstack examples\n");
  await generateExample();
  console.log();
  await embeddingExample();
  console.log();
  await streamWithCallbacks();
  console.log();
  await multiTurnExample();
}

// Only run main() when executed directly (not imported as Next.js route)
if (typeof process !== "undefined" && process.argv[1]?.includes("index.ts")) {
  main().catch(console.error);
}
