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
  fetchOnboarding,
  fetchSavings,
  listModels,
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
  const cfg = readConfig();
  const ok = await checkHealth(cfg);
  statusBar.text = ok ? "$(check) LLMStack" : "$(circle-slash) LLMStack";
  if (!ok) {
    statusBar.tooltip = "LLMStack gateway is not reachable — run 'llmstack up'";
    return;
  }
  // When reachable, fold the running savings total into the tooltip so the
  // value story is visible at a glance.
  const savings = await fetchSavings(cfg);
  statusBar.tooltip =
    savings && savings.total_saved_usd > 0
      ? `LLMStack gateway reachable (local) — saved $${savings.total_saved_usd.toFixed(
          2,
        )} so far`
      : "LLMStack gateway is reachable (local)";
}

async function showSavings(): Promise<void> {
  const cfg = readConfig();
  const savings = await fetchSavings(cfg);
  if (!savings) {
    vscode.window.showWarningMessage(
      "LLMStack: could not read savings. Is 'llmstack up' running?",
    );
    return;
  }
  const sub = savings.subscription;
  const saved = savings.total_saved_usd.toFixed(2);
  const months = sub ? sub.months_covered.toFixed(1) : "0";
  const name = sub ? sub.name : "a paid plan";
  vscode.window.showInformationMessage(
    `LLMStack has saved you $${saved} running locally — about ${months} month(s) of ${name}, ` +
      `across ${savings.total_requests} request(s).`,
  );
}

/**
 * On first reachable run, if the machine isn't ready for local inference, offer
 * a one-click path to fix it. Silent when the gateway is down (the status bar
 * already signals that) or when everything is ready.
 */
async function checkFirstRun(): Promise<void> {
  const status = await fetchOnboarding(readConfig());
  if (!status || status.ready) {
    return;
  }
  const next = status.hints[0] ?? "set up a local model";
  const choice = await vscode.window.showWarningMessage(
    `LLMStack isn't ready for local inference yet (${next}).`,
    "Run quickstart",
    "Dismiss",
  );
  if (choice === "Run quickstart") {
    await vscode.commands.executeCommand("llmstack.runQuickstart");
  }
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
    vscode.commands.registerCommand("llmstack.showSavings", showSavings),
    vscode.commands.registerCommand("llmstack.runQuickstart", () => {
      const term = vscode.window.createTerminal("LLMStack Quickstart");
      term.show();
      term.sendText("llmstack quickstart");
    }),
    vscode.commands.registerCommand("llmstack.switchModel", async () => {
      const cfg = readConfig();
      const models = await listModels(cfg);
      if (models.length === 0) {
        vscode.window.showWarningMessage(
          "LLMStack: no models found. Is the gateway running?",
        );
        return;
      }
      const pick = await vscode.window.showQuickPick(models, {
        placeHolder: `Active model: ${cfg.model}`,
      });
      if (pick) {
        await vscode.workspace
          .getConfiguration("llmstack")
          .update("model", pick, vscode.ConfigurationTarget.Global);
        vscode.window.setStatusBarMessage(`LLMStack: model set to ${pick}`, 3000);
      }
    }),
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
  void checkFirstRun();

  // Poll periodically so the status bar reflects the gateway starting/stopping
  // without the user having to click it. Cleared on deactivate.
  const healthTimer = setInterval(() => void refreshHealth(), 15000);
  context.subscriptions.push({ dispose: () => clearInterval(healthTimer) });
}

export function deactivate(): void {
  output?.dispose();
  statusBar?.dispose();
}
