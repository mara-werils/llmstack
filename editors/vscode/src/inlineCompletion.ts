/**
 * Inline (ghost-text) code completion backed by the local LLMStack gateway.
 *
 * Opt-in via `llmstack.inlineCompletion.enabled`. Sends the surrounding code
 * as context to the configured model and renders its continuation as a
 * single inline suggestion, the same way Copilot-style completions work —
 * except the request never leaves the user's machine.
 */

import * as vscode from "vscode";

import { ChatMessage, GatewayConfig, complete } from "./gatewayClient";

const DEBOUNCE_MS = 200;

function delay(ms: number, token: vscode.CancellationToken): Promise<void> {
  return new Promise((resolve) => {
    const timer = setTimeout(resolve, ms);
    token.onCancellationRequested(() => {
      clearTimeout(timer);
      resolve();
    });
  });
}

function abortSignalFor(token: vscode.CancellationToken): AbortSignal {
  const controller = new AbortController();
  token.onCancellationRequested(() => controller.abort());
  return controller.signal;
}

/** Strip markdown code fences and trailing whitespace the model may add despite instructions. */
function sanitizeCompletion(raw: string): string {
  let text = raw.trim();
  text = text.replace(/^```[a-zA-Z0-9_-]*\n?/, "").replace(/\n?```$/, "");
  return text.trimEnd();
}

function buildPrompt(
  document: vscode.TextDocument,
  position: vscode.Position,
  contextLines: number,
): ChatMessage[] {
  const startLine = Math.max(0, position.line - contextLines);
  const prefix = document.getText(
    new vscode.Range(startLine, 0, position.line, position.character),
  );

  const endLine = Math.min(document.lineCount - 1, position.line + contextLines);
  const suffix = document.getText(
    new vscode.Range(
      position.line,
      position.character,
      endLine,
      document.lineAt(endLine).text.length,
    ),
  );

  return [
    {
      role: "system",
      content:
        "You are a code-completion engine embedded in an editor. Continue the code " +
        "exactly at the <CURSOR> marker. Output ONLY the missing code that should be " +
        "inserted there — no markdown fences, no explanations, no repeating code that " +
        "already exists before or after the marker. If nothing useful can be added, " +
        "output nothing.",
    },
    {
      role: "user",
      content: `Language: ${document.languageId}\n\n${prefix}<CURSOR>${suffix}`,
    },
  ];
}

export function createInlineCompletionProvider(
  getConfig: () => GatewayConfig,
): vscode.InlineCompletionItemProvider {
  return {
    async provideInlineCompletionItems(document, position, _context, token) {
      const settings = vscode.workspace.getConfiguration("llmstack");
      if (!settings.get<boolean>("inlineCompletion.enabled", false)) {
        return undefined;
      }

      // Allow opting specific languages out (e.g. plaintext, markdown, env files).
      const disabled = settings.get<string[]>("inlineCompletion.disabledLanguages", []);
      if (disabled.includes(document.languageId)) {
        return undefined;
      }

      // Debounce so we don't fire a request on every keystroke.
      const debounceMs = settings.get<number>("inlineCompletion.debounceMs", DEBOUNCE_MS);
      await delay(debounceMs, token);
      if (token.isCancellationRequested) {
        return undefined;
      }

      const contextLines = settings.get<number>("inlineCompletion.contextLines", 50);
      const messages = buildPrompt(document, position, contextLines);

      let raw: string;
      try {
        raw = await complete(getConfig(), messages, abortSignalFor(token), {
          temperature: 0.1,
          maxTokens: settings.get<number>("inlineCompletion.maxTokens", 200),
        });
      } catch {
        // Gateway unreachable or request aborted — fail silently, same as Copilot does offline.
        return undefined;
      }

      if (token.isCancellationRequested) {
        return undefined;
      }

      const completionText = sanitizeCompletion(raw);
      if (!completionText) {
        return undefined;
      }

      return [
        new vscode.InlineCompletionItem(completionText, new vscode.Range(position, position)),
      ];
    },
  };
}

export function registerInlineCompletionProvider(
  context: vscode.ExtensionContext,
  getConfig: () => GatewayConfig,
): void {
  const provider = createInlineCompletionProvider(getConfig);
  context.subscriptions.push(
    vscode.languages.registerInlineCompletionItemProvider({ pattern: "**" }, provider),
  );
}
