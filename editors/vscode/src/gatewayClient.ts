/**
 * Minimal client for the local LLMStack gateway.
 *
 * Zero runtime dependencies — uses the native `fetch` shipped with the
 * VS Code Node runtime. Speaks the gateway's OpenAI-compatible API so the
 * same calls work against Ollama, vLLM, or any configured provider.
 */

export interface GatewayConfig {
  baseUrl: string;
  apiKey?: string;
  model: string;
}

export interface ChatMessage {
  role: "system" | "user" | "assistant";
  content: string;
}

export class GatewayError extends Error {
  readonly status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = "GatewayError";
    this.status = status;
  }
}

function headers(cfg: GatewayConfig): Record<string, string> {
  const h: Record<string, string> = { "Content-Type": "application/json" };
  if (cfg.apiKey) {
    h["Authorization"] = `Bearer ${cfg.apiKey}`;
  }
  return h;
}

/** Stream a chat completion, invoking `onToken` for each text delta. */
export async function streamChat(
  cfg: GatewayConfig,
  messages: ChatMessage[],
  onToken: (token: string) => void,
  signal?: AbortSignal,
): Promise<void> {
  const resp = await fetch(`${cfg.baseUrl}/v1/chat/completions`, {
    method: "POST",
    headers: headers(cfg),
    body: JSON.stringify({ model: cfg.model, messages, stream: true }),
    signal,
  });

  if (!resp.ok || !resp.body) {
    throw new GatewayError(resp.status, await safeText(resp));
  }

  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  for (;;) {
    const { done, value } = await reader.read();
    if (done) {
      break;
    }
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";

    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed.startsWith("data:")) {
        continue;
      }
      const data = trimmed.slice(5).trim();
      if (data === "[DONE]") {
        return;
      }
      try {
        const chunk = JSON.parse(data);
        const token = chunk?.choices?.[0]?.delta?.content;
        if (typeof token === "string" && token.length > 0) {
          onToken(token);
        }
      } catch {
        // Ignore keep-alive / non-JSON lines.
      }
    }
  }
}

/** Run a non-streaming chat completion and return the full response text. */
export async function complete(
  cfg: GatewayConfig,
  messages: ChatMessage[],
  signal?: AbortSignal,
  options?: { temperature?: number; maxTokens?: number },
): Promise<string> {
  const resp = await fetch(`${cfg.baseUrl}/v1/chat/completions`, {
    method: "POST",
    headers: headers(cfg),
    body: JSON.stringify({
      model: cfg.model,
      messages,
      stream: false,
      temperature: options?.temperature ?? 0.2,
      max_tokens: options?.maxTokens ?? 256,
    }),
    signal,
  });

  if (!resp.ok) {
    throw new GatewayError(resp.status, await safeText(resp));
  }

  const data = (await resp.json()) as {
    choices?: { message?: { content?: string } }[];
  };
  const content = data?.choices?.[0]?.message?.content;
  return typeof content === "string" ? content : "";
}

/** List model IDs the gateway exposes (OpenAI-compatible `/v1/models`). */
export async function listModels(cfg: GatewayConfig): Promise<string[]> {
  try {
    const resp = await fetch(`${cfg.baseUrl}/v1/models`, { headers: headers(cfg) });
    if (!resp.ok) {
      return [];
    }
    const data = (await resp.json()) as { data?: { id?: string }[] };
    const ids = (data?.data ?? [])
      .map((m) => m.id)
      .filter((id): id is string => typeof id === "string" && id.length > 0);
    return Array.from(new Set(ids));
  } catch {
    return [];
  }
}

/** Return true when the gateway health endpoint reports OK. */
export async function checkHealth(cfg: GatewayConfig): Promise<boolean> {
  try {
    const resp = await fetch(`${cfg.baseUrl}/health`, { headers: headers(cfg) });
    return resp.ok;
  } catch {
    return false;
  }
}

async function safeText(resp: Response): Promise<string> {
  try {
    return await resp.text();
  } catch {
    return resp.statusText;
  }
}
