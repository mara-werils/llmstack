/**
 * Concurrent batch processing for single-prompt completions.
 *
 * Mirrors the Python SDK's `BatchProcessor`: run many prompts through the
 * gateway with a bounded concurrency and collect per-item results.
 */
import type { LLMStackClient } from "./client.js";

export interface BatchItem {
  /** Caller-supplied identifier echoed back on the result. */
  id: string | number;
  /** The user prompt to complete. */
  prompt: string;
  /** Optional system prompt. */
  system?: string;
  /** Optional model override (defaults to the processor's model). */
  model?: string;
}

export interface BatchResult {
  id: string | number;
  prompt: string;
  response: string;
  success: boolean;
  error?: string;
}

export interface BatchSummary {
  total: number;
  completed: number;
  failed: number;
  results: BatchResult[];
}

/** Process multiple prompts concurrently against an {@link LLMStackClient}. */
export class BatchProcessor {
  constructor(
    private readonly client: LLMStackClient,
    private readonly concurrency: number = 5,
    private readonly model: string = "llama3.2",
  ) {}

  /** Run all items, at most `concurrency` in flight at once. */
  async run(items: BatchItem[]): Promise<BatchSummary> {
    const results: BatchResult[] = [];
    for (let i = 0; i < items.length; i += this.concurrency) {
      const chunk = items.slice(i, i + this.concurrency);
      const settled = await Promise.all(chunk.map((item) => this._process(item)));
      results.push(...settled);
    }
    const completed = results.filter((r) => r.success).length;
    return {
      total: items.length,
      completed,
      failed: items.length - completed,
      results,
    };
  }

  private async _process(item: BatchItem): Promise<BatchResult> {
    try {
      const response = await this.client.complete(
        item.prompt,
        item.model ?? this.model,
        item.system ?? "",
      );
      return { id: item.id, prompt: item.prompt, response, success: true };
    } catch (err) {
      return {
        id: item.id,
        prompt: item.prompt,
        response: "",
        success: false,
        error: err instanceof Error ? err.message : String(err),
      };
    }
  }
}
