/**
 * LLMStack VS Code extension entry point.
 *
 * Provides "Ask" and "Explain selection" commands that stream responses from
 * the user's local LLMStack gateway into an output channel, plus a status-bar
 * indicator of gateway health. All traffic stays on the user's machine.
 */

import * as vscode from "vscode";

import {
  ChatMessage,
  GatewayConfig,
  GatewayError,
  checkHealth,
  streamChat,
} from "./gatewayClient";
import { ChatViewProvider } from "./chatView";
import { registerEditCommand } from "./editor";
import { registerInlineCompletionProvider } from "./inlineCompletion";

let output: vscode.OutputChannel;
let statusBar: vscode.StatusBarItem;

function readConfig(): GatewayConfig {
  const cfg = vscode.workspace.getConfiguration("llmstack");
  return {
    baseUrl: cfg.get<string>("gatewayUrl", "http://localhost:8000").replace(/\/$/, ""),
    apiKey: cfg.get<string>("apiKey", "") || undefined,
    model: cfg.get<string>("model", "llama3.2"),
  };
}

async function runPrompt(prompt: string, context: string): Promise<void> {
  const cfg = readConfig();
  const messages: ChatMessage[] = [
    {
      role: "system",
      content:
        "You are a concise coding assistant running locally via LLMStack. " +
        "Answer directly and show code where helpful.",
    },
    { role: "user", content: context ? `${prompt}\n\n${context}` : prompt },
  ];

  output.show(true);
  output.appendLine(`\n## ${prompt}\n`);

  await vscode.window.withProgress(
    { location: vscode.ProgressLocation.Window, title: "LLMStack: thinking…" },
    async () => {
      try {
        await streamChat(cfg, messages, (token) => output.append(token));
        output.appendLine("\n");
      } catch (err) {
        const msg =
          err instanceof GatewayError
            ? `Gateway error ${err.status}: ${err.message}`
            : `Could not reach gateway at ${cfg.baseUrl}. Is 'llmstack up' running?`;
        vscode.window.showErrorMessage(`LLMStack: ${msg}`);
      }
    },
  );
}

function selectedText(): string {
  const editor = vscode.window.activeTextEditor;
  if (!editor) {
    return "";
  }
  const text = editor.document.getText(editor.selection);
  if (!text) {
    return "";
  }
  return "```" + editor.document.languageId + "\n" + text + "\n```";
}

async function refreshHealth(): Promise<void> {
  const ok = await checkHealth(readConfig());
  statusBar.text = ok ? "$(check) LLMStack" : "$(circle-slash) LLMStack";
  statusBar.tooltip = ok
    ? "LLMStack gateway is reachable (local)"
    : "LLMStack gateway is not reachable — run 'llmstack up'";
}

export function activate(context: vscode.ExtensionContext): void {
  output = vscode.window.createOutputChannel("LLMStack");
  statusBar = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 100);
  statusBar.command = "llmstack.checkHealth";
  statusBar.show();

  context.subscriptions.push(
    output,
    statusBar,
    vscode.commands.registerCommand("llmstack.ask", async () => {
      const prompt = await vscode.window.showInputBox({
        prompt: "Ask LLMStack",
        placeHolder: "e.g. How do I read a file in Rust?",
      });
      if (prompt) {
        await runPrompt(prompt, selectedText());
      }
    }),
    vscode.commands.registerCommand("llmstack.explain", async () => {
      const code = selectedText();
      if (!code) {
        vscode.window.showInformationMessage("LLMStack: select some code first.");
        return;
      }
      await runPrompt("Explain what this code does and flag any bugs:", code);
    }),
    vscode.commands.registerCommand("llmstack.checkHealth", refreshHealth),
    vscode.commands.registerCommand("llmstack.toggleInlineCompletion", async () => {
      const cfg = vscode.workspace.getConfiguration("llmstack");
      const next = !cfg.get<boolean>("inlineCompletion.enabled", false);
      await cfg.update("inlineCompletion.enabled", next, vscode.ConfigurationTarget.Global);
      vscode.window.setStatusBarMessage(
        `LLMStack: inline completions ${next ? "enabled" : "disabled"}`,
        3000,
      );
    }),
  );

  const chat = new ChatViewProvider(context.extensionUri, readConfig, context.workspaceState);
  context.subscriptions.push(
    vscode.window.registerWebviewViewProvider(ChatViewProvider.viewType, chat),
    vscode.commands.registerCommand("llmstack.openChat", () =>
      vscode.commands.executeCommand("llmstack.chatView.focus"),
    ),
    vscode.commands.registerCommand("llmstack.newChat", () => chat.newChat()),
  );

  registerEditCommand(context, readConfig);
  registerInlineCompletionProvider(context, readConfig);

  void refreshHealth();

  // Poll periodically so the status bar reflects the gateway starting/stopping
  // without the user having to click it. Cleared on deactivate.
  const healthTimer = setInterval(() => void refreshHealth(), 15000);
  context.subscriptions.push({ dispose: () => clearInterval(healthTimer) });
}

export function deactivate(): void {
  output?.dispose();
  statusBar?.dispose();
}
