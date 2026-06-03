/**
 * Server-Sent Events (SSE) stream parser.
 *
 * Works with the native `fetch` `ReadableStream` -- zero dependencies.
 */

/**
 * Parse an SSE stream from a `Response` body into an async iterator of
 * JSON-parsed objects of type `T`.
 *
 * Handles:
 *  - `data: <json>` lines
 *  - `data: [DONE]` sentinel (terminates the iterator)
 *  - blank lines and SSE comments (lines starting with `:`)
 *  - chunked reads where data boundaries fall mid-line
 *
 * @param response - A `Response` whose body is an SSE text/event-stream.
 * @param signal   - Optional `AbortSignal` to cancel reading.
 * @returns An `AsyncGenerator` that yields parsed JSON objects.
 */
export async function* parseSSEStream<T>(
  response: Response,
  signal?: AbortSignal,
): AsyncGenerator<T, void, undefined> {
  const body = response.body;
  if (!body) {
    return;
  }

  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (true) {
      if (signal?.aborted) {
        break;
      }

      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      // Split on newline boundaries -- SSE uses \n, \r\n, or \r.
      const lines = buffer.split(/\r?\n|\r/);
      // The last element may be an incomplete line; keep it in the buffer.
      buffer = lines.pop() ?? "";

      for (const line of lines) {
        const parsed = parseLine<T>(line);
        if (parsed === DONE_SENTINEL) {
          return;
        }
        if (parsed !== null) {
          yield parsed;
        }
      }
    }

    // Flush any remaining data in the buffer.
    if (buffer.trim()) {
      const parsed = parseLine<T>(buffer);
      if (parsed !== null && parsed !== DONE_SENTINEL) {
        yield parsed;
      }
    }
  } finally {
    reader.releaseLock();
  }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

const DONE_SENTINEL = Symbol("DONE");

/**
 * Parse a single SSE line.
 * Returns the parsed JSON object, `DONE_SENTINEL` for `[DONE]`, or `null`
 * for lines that should be skipped (blank, comments, malformed).
 */
function parseLine<T>(raw: string): T | typeof DONE_SENTINEL | null {
  const line = raw.trim();
  if (!line || line.startsWith(":")) {
    return null;
  }
  if (!line.startsWith("data:")) {
    return null;
  }

  const payload = line.slice("data:".length).trim();
  if (payload === "[DONE]") {
    return DONE_SENTINEL;
  }

  try {
    return JSON.parse(payload) as T;
  } catch {
    return null;
  }
}
