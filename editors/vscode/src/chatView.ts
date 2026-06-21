/**
 * Chat sidebar for LLMStack — a webview view that streams a conversation from the
 * user's local gateway. This is the editor-native surface most developers expect
 * (Cline/Continue style), kept zero-dependency: the webview is plain HTML/JS and
 * the host speaks the gateway's OpenAI-compatible streaming API.
 */

import * as vscode from "vscode";

import {
  ChatMessage,
  GatewayConfig,
  GatewayError,
  streamChat,
} from "./gatewayClient";

const SYSTEM_PROMPT =
  "You are a concise coding assistant running locally via LLMStack. " +
  "Answer directly and show code where helpful. Everything stays on the user's machine.";

interface WebviewMessage {
  type?: string;
  text?: string;
}

/** Provides the LLMStack chat view contributed to the activity bar. */
export class ChatViewProvider implements vscode.WebviewViewProvider {
  public static readonly viewType = "llmstack.chatView";

  private view?: vscode.WebviewView;
  private controller?: AbortController;
  private history: ChatMessage[] = [];

  constructor(
    private readonly extensionUri: vscode.Uri,
    private readonly readConfig: () => GatewayConfig,
  ) {}

  public resolveWebviewView(view: vscode.WebviewView): void {
    this.view = view;
    view.webview.options = {
      enableScripts: true,
      localResourceRoots: [vscode.Uri.joinPath(this.extensionUri, "media")],
    };
    view.webview.html = this.render(view.webview);
    view.webview.onDidReceiveMessage((msg: WebviewMessage) => {
      void this.onMessage(msg);
    });
  }

  /** Reset the conversation and clear the panel. */
  public newChat(): void {
    this.controller?.abort();
    this.history = [];
    void this.view?.webview.postMessage({ type: "clear" });
  }

  private async onMessage(msg: WebviewMessage): Promise<void> {
    if (msg.type === "send" && typeof msg.text === "string") {
      await this.handleSend(msg.text);
    } else if (msg.type === "stop") {
      this.controller?.abort();
    }
  }

  private async handleSend(text: string): Promise<void> {
    const view = this.view;
    if (!view) {
      return;
    }
    const cfg = this.readConfig();
    this.history.push({ role: "user", content: text });
    const messages: ChatMessage[] = [
      { role: "system", content: SYSTEM_PROMPT },
      ...this.history,
    ];

    const controller = new AbortController();
    this.controller = controller;
    let assistant = "";
    try {
      await streamChat(
        cfg,
        messages,
        (token) => {
          assistant += token;
          void view.webview.postMessage({ type: "token", text: token });
        },
        controller.signal,
      );
      this.history.push({ role: "assistant", content: assistant });
      void view.webview.postMessage({ type: "done" });
    } catch (err) {
      if (controller.signal.aborted) {
        if (assistant) {
          this.history.push({ role: "assistant", content: assistant });
        }
        void view.webview.postMessage({ type: "done" });
      } else {
        const message =
          err instanceof GatewayError
            ? `Gateway error ${err.status}: ${err.message}`
            : `Could not reach the gateway at ${cfg.baseUrl}. Is 'llmstack up' running?`;
        void view.webview.postMessage({ type: "error", message });
      }
    } finally {
      if (this.controller === controller) {
        this.controller = undefined;
      }
    }
  }

  private render(webview: vscode.Webview): string {
    const nonce = getNonce();
    const styleUri = webview.asWebviewUri(
      vscode.Uri.joinPath(this.extensionUri, "media", "chat", "chat.css"),
    );
    const scriptUri = webview.asWebviewUri(
      vscode.Uri.joinPath(this.extensionUri, "media", "chat", "chat.js"),
    );
    const csp = [
      "default-src 'none'",
      `img-src ${webview.cspSource} https: data:`,
      `style-src ${webview.cspSource}`,
      `script-src 'nonce-${nonce}'`,
    ].join("; ");

    return `<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta http-equiv="Content-Security-Policy" content="${csp}" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <link href="${styleUri}" rel="stylesheet" />
    <title>LLMStack Chat</title>
  </head>
  <body>
    <div id="root">
      <div id="messages"></div>
      <div id="composer">
        <textarea id="input" rows="2" placeholder="Ask your local model… (Enter to send, Shift+Enter for newline)"></textarea>
        <button id="send">Send</button>
      </div>
    </div>
    <script nonce="${nonce}" src="${scriptUri}"></script>
  </body>
</html>`;
  }
}

function getNonce(): string {
  const chars =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
  let text = "";
  for (let i = 0; i < 32; i += 1) {
    text += chars.charAt(Math.floor(Math.random() * chars.length));
  }
  return text;
}
