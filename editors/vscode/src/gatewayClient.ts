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

export interface Feedback {
  feedbackType: "thumbs_up" | "thumbs_down";
  query: string;
  response: string;
  model: string;
}

/** Send thumbs feedback to the gateway's adaptive-learning pipeline. Best-effort. */
export async function sendFeedback(cfg: GatewayConfig, fb: Feedback): Promise<void> {
  await fetch(`${cfg.baseUrl}/v1/feedback`, {
    method: "POST",
    headers: headers(cfg),
    body: JSON.stringify({
      feedback_type: fb.feedbackType,
      query: fb.query,
      response: fb.response,
      model: fb.model,
      command: "vscode-chat",
    }),
  });
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

/**
 * Return true when the gateway is reachable. Uses the liveness probe
 * `/healthz/live` — the gateway exposes `/healthz*` and `/ping`, never `/health`,
 * and `/healthz/live` is auth-exempt so it works even with gateway auth enabled.
 */
export async function checkHealth(cfg: GatewayConfig): Promise<boolean> {
  try {
    const resp = await fetch(`${cfg.baseUrl}/healthz/live`, { headers: headers(cfg) });
    return resp.ok;
  } catch {
    return false;
  }
}

export interface SavingsSummary {
  total_requests: number;
  total_saved_usd: number;
  subscription: { name: string; monthly_usd: number; months_covered: number };
}

/**
 * Fetch the running "money saved by running locally" total from the gateway.
 * Returns `undefined` when the gateway is unreachable or the route is absent.
 */
export async function fetchSavings(
  cfg: GatewayConfig,
  plan?: string,
): Promise<SavingsSummary | undefined> {
  try {
    const url = new URL(`${cfg.baseUrl}/v1/savings/summary`);
    if (plan) {
      url.searchParams.set("plan", plan);
    }
    const resp = await fetch(url, { headers: headers(cfg) });
    if (!resp.ok) {
      return undefined;
    }
    return (await resp.json()) as SavingsSummary;
  } catch {
    return undefined;
  }
}

export interface OnboardingStatus {
  ready: boolean;
  ollama: { url: string; running: boolean; models: string[] };
  recommended: {
    chat_model: { name: string; label?: string; reason?: string };
    embed_model: { name: string; label?: string; reason?: string };
  };
  chat_model: { name: string; ready: boolean };
  embed_model: { name: string; ready: boolean };
  hints: string[];
}

/**
 * Fetch first-run readiness from the gateway (`GET /v1/onboarding`): whether
 * Ollama is up, which models are present, and next-step hints. Returns
 * `undefined` when the gateway is unreachable or the route is absent.
 */
export async function fetchOnboarding(
  cfg: GatewayConfig,
): Promise<OnboardingStatus | undefined> {
  try {
    const resp = await fetch(`${cfg.baseUrl}/v1/onboarding`, { headers: headers(cfg) });
    if (!resp.ok) {
      return undefined;
    }
    return (await resp.json()) as OnboardingStatus;
  } catch {
    return undefined;
  }
}

async function safeText(resp: Response): Promise<string> {
  try {
    return await resp.text();
  } catch {
    return resp.statusText;
  }
}
