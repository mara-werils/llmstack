/**
 * Chat sidebar for LLMStack — a webview view that streams a conversation from the
 * user's local gateway. This is the editor-native surface most developers expect
 * (Cline/Continue style), kept zero-dependency: the webview is plain HTML/JS and
 * the host speaks the gateway's OpenAI-compatible streaming API.
 */

import * as path from "path";

import * as vscode from "vscode";

import {
  ChatMessage,
  GatewayConfig,
  GatewayError,
  checkHealth,
  listModels,
  sendFeedback,
  streamChat,
} from "./gatewayClient";

/** Fully-qualified walkthrough id (publisher.name#walkthroughId). */
const WALKTHROUGH_ID = "llmstack.llmstack-vscode#llmstack.gettingStarted";

/** Max characters of editor context sent with a message, to bound prompt size. */
const MAX_CONTEXT_CHARS = 6000;

const SYSTEM_PROMPT =
  "You are a concise coding assistant running locally via LLMStack. " +
  "Answer directly and show code where helpful. Everything stays on the user's machine.";

interface WebviewMessage {
  type?: string;
  text?: string;
  includeContext?: boolean;
  vote?: string;
  query?: string;
  response?: string;
}

/** Provides the LLMStack chat view contributed to the activity bar. */
export class ChatViewProvider implements vscode.WebviewViewProvider {
  public static readonly viewType = "llmstack.chatView";

  private view?: vscode.WebviewView;
  private controller?: AbortController;
  private history: ChatMessage[] = [];
  private contextWatchers: vscode.Disposable[] = [];
  private modelOverride?: string;

  private static readonly MODEL_KEY = "llmstack.chat.model";

  constructor(
    private readonly extensionUri: vscode.Uri,
    private readonly readConfig: () => GatewayConfig,
    private readonly state: vscode.Memento,
  ) {
    this.modelOverride = state.get<string>(ChatViewProvider.MODEL_KEY) || undefined;
  }

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
    this.registerContextWatchers();
    view.onDidDispose(() => this.disposeContextWatchers());
  }

  private registerContextWatchers(): void {
    this.disposeContextWatchers();
    this.contextWatchers.push(
      vscode.window.onDidChangeActiveTextEditor(() => this.postContextInfo()),
      vscode.window.onDidChangeTextEditorSelection(() => this.postContextInfo()),
    );
    this.postContextInfo();
  }

  private disposeContextWatchers(): void {
    this.contextWatchers.forEach((d) => d.dispose());
    this.contextWatchers = [];
  }

  /** Tell the webview whether the gateway is currently reachable. */
  private async postHealth(): Promise<void> {
    const ok = await checkHealth(this.readConfig());
    void this.view?.webview.postMessage({ type: "health", ok });
  }

  /** Fetch available models from the gateway and offer them in the panel picker. */
  private async postModels(): Promise<void> {
    const cfg = this.readConfig();
    const models = await listModels(cfg);
    const current = this.modelOverride ?? cfg.model;
    void this.view?.webview.postMessage({ type: "models", models, current });
  }

  private postContextInfo(): void {
    const editor = vscode.window.activeTextEditor;
    const file = editor ? path.basename(editor.document.fileName) : "";
    const hasSelection = !!editor && !editor.selection.isEmpty;
    void this.view?.webview.postMessage({ type: "context", file, hasSelection });
  }

  /** Build a fenced code block from the active selection (or whole file), capped in size. */
  private buildContext(): string {
    const editor = vscode.window.activeTextEditor;
    if (!editor) {
      return "";
    }
    const selection = editor.selection;
    const doc = editor.document;
    const raw = selection.isEmpty ? doc.getText() : doc.getText(selection);
    if (!raw.trim()) {
      return "";
    }
    const clipped =
      raw.length > MAX_CONTEXT_CHARS
        ? `${raw.slice(0, MAX_CONTEXT_CHARS)}\n… (truncated)`
        : raw;
    const name = path.basename(doc.fileName);
    const label = selection.isEmpty ? `Active file ${name}` : `Selection from ${name}`;
    return `${label}:\n\`\`\`${doc.languageId}\n${clipped}\n\`\`\``;
  }

  /** Reset the conversation and clear the panel. */
  public newChat(): void {
    this.controller?.abort();
    this.history = [];
    void this.view?.webview.postMessage({ type: "clear" });
  }

  private async onMessage(msg: WebviewMessage): Promise<void> {
    if (msg.type === "ready") {
      this.postContextInfo();
      await this.postModels();
      await this.postHealth();
    } else if (msg.type === "startGateway") {
      const terminal = vscode.window.createTerminal("LLMStack");
      terminal.show();
      terminal.sendText("llmstack up");
    } else if (msg.type === "openWalkthrough") {
      await vscode.commands.executeCommand(
        "workbench.action.openWalkthrough",
        WALKTHROUGH_ID,
        false,
      );
    } else if (msg.type === "send" && typeof msg.text === "string") {
      await this.handleSend(msg.text, msg.includeContext === true);
    } else if (msg.type === "stop") {
      this.controller?.abort();
    } else if (msg.type === "model") {
      this.modelOverride = msg.text || undefined;
      await this.state.update(ChatViewProvider.MODEL_KEY, this.modelOverride);
    } else if (msg.type === "feedback" && (msg.vote === "up" || msg.vote === "down")) {
      const cfg = this.readConfig();
      if (this.modelOverride) {
        cfg.model = this.modelOverride;
      }
      void sendFeedback(cfg, {
        feedbackType: msg.vote === "up" ? "thumbs_up" : "thumbs_down",
        query: typeof msg.query === "string" ? msg.query : "",
        response: typeof msg.response === "string" ? msg.response : "",
        model: cfg.model,
      }).catch(() => undefined);
      vscode.window.setStatusBarMessage("LLMStack: feedback sent — thanks!", 2000);
    } else if (msg.type === "copy" && typeof msg.text === "string") {
      await vscode.env.clipboard.writeText(msg.text);
      vscode.window.setStatusBarMessage("LLMStack: copied to clipboard", 2000);
    } else if (msg.type === "apply" && typeof msg.text === "string") {
      await this.applyToEditor(msg.text, true);
    } else if (msg.type === "insert" && typeof msg.text === "string") {
      await this.applyToEditor(msg.text, false);
    }
  }

  /** Insert generated code into the active editor, replacing the selection if one exists. */
  private async applyToEditor(text: string, replaceSelection: boolean): Promise<void> {
    const editor = vscode.window.activeTextEditor;
    if (!editor) {
      vscode.window.showInformationMessage(
        "LLMStack: open a file and place the cursor where the code should go.",
      );
      return;
    }
    const selection = editor.selection;
    await editor.edit((builder) => {
      if (replaceSelection && !selection.isEmpty) {
        builder.replace(selection, text);
      } else {
        builder.insert(selection.active, text);
      }
    });
  }

  private async handleSend(text: string, includeContext: boolean): Promise<void> {
    const view = this.view;
    if (!view) {
      return;
    }
    const cfg = this.readConfig();
    if (this.modelOverride) {
      cfg.model = this.modelOverride;
    }
    const context = includeContext ? this.buildContext() : "";
    const userContent = context ? `${text}\n\n${context}` : text;
    this.history.push({ role: "user", content: userContent });
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
        void this.postHealth();
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
      <div id="banner" hidden>
        <span class="banner-text">⚠ Gateway not reachable.</span>
        <button id="banner-start">Start gateway</button>
        <button id="banner-help" class="link">Getting started</button>
      </div>
      <div id="toolbar">
        <select id="model" title="Model used for this chat"></select>
      </div>
      <div id="messages"></div>
      <label id="ctx">
        <input type="checkbox" id="ctx-toggle" />
        <span id="ctx-label">Include editor context</span>
      </label>
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
