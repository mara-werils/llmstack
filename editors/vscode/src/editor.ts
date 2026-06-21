/**
 * AI-assisted edit with a review step. The user selects code and describes a
 * change; LLMStack drafts a replacement, shows it as a native VS Code diff, and
 * only touches the file after the user approves. This is the diff → approve loop
 * that defines the most-adopted agentic coding tools — done locally.
 */

import * as path from "path";

import * as vscode from "vscode";

import {
  ChatMessage,
  GatewayConfig,
  GatewayError,
  complete,
} from "./gatewayClient";

const SCHEME = "llmstack-diff";

/** Serves the proposed (right-hand) side of the review diff from memory. */
class ProposedContentProvider implements vscode.TextDocumentContentProvider {
  private readonly contents = new Map<string, string>();
  private readonly emitter = new vscode.EventEmitter<vscode.Uri>();
  public readonly onDidChange = this.emitter.event;

  public set(uri: vscode.Uri, text: string): void {
    this.contents.set(uri.toString(), text);
    this.emitter.fire(uri);
  }

  public provideTextDocumentContent(uri: vscode.Uri): string {
    return this.contents.get(uri.toString()) ?? "";
  }
}

function stripFences(raw: string): string {
  return raw
    .trim()
    .replace(/^```[a-zA-Z0-9_-]*\n?/, "")
    .replace(/\n?```$/, "")
    .trimEnd();
}

let diffCounter = 0;

export function registerEditCommand(
  context: vscode.ExtensionContext,
  readConfig: () => GatewayConfig,
): void {
  const provider = new ProposedContentProvider();
  context.subscriptions.push(
    vscode.workspace.registerTextDocumentContentProvider(SCHEME, provider),
    vscode.commands.registerCommand("llmstack.editSelection", () =>
      runEdit(readConfig, provider),
    ),
  );
}

async function runEdit(
  readConfig: () => GatewayConfig,
  provider: ProposedContentProvider,
): Promise<void> {
  const editor = vscode.window.activeTextEditor;
  if (!editor || editor.selection.isEmpty) {
    vscode.window.showInformationMessage(
      "LLMStack: select the code you want to edit first.",
    );
    return;
  }

  const instruction = await vscode.window.showInputBox({
    prompt: "How should LLMStack change the selection?",
    placeHolder: "e.g. add error handling and type hints",
  });
  if (!instruction) {
    return;
  }

  const doc = editor.document;
  const selection = editor.selection;
  const original = doc.getText(selection);
  const cfg = readConfig();

  const messages: ChatMessage[] = [
    {
      role: "system",
      content:
        "You are a code editor. Rewrite the user's code to satisfy their instruction. " +
        "Return ONLY the replacement code — no explanation, no markdown fences.",
    },
    {
      role: "user",
      content: `Instruction: ${instruction}\n\nLanguage: ${doc.languageId}\n\nCode:\n${original}`,
    },
  ];

  let proposed: string;
  try {
    const raw = await vscode.window.withProgress(
      {
        location: vscode.ProgressLocation.Notification,
        title: "LLMStack: drafting edit…",
      },
      () => complete(cfg, messages, undefined, { temperature: 0.1, maxTokens: 1024 }),
    );
    proposed = stripFences(raw);
  } catch (err) {
    const message =
      err instanceof GatewayError
        ? `Gateway error ${err.status}: ${err.message}`
        : `Could not reach the gateway at ${cfg.baseUrl}. Is 'llmstack up' running?`;
    vscode.window.showErrorMessage(`LLMStack: ${message}`);
    return;
  }

  if (!proposed) {
    vscode.window.showWarningMessage("LLMStack: the model returned no replacement.");
    return;
  }

  const fullEnd = doc.lineAt(doc.lineCount - 1).range.end;
  const before = doc.getText(new vscode.Range(new vscode.Position(0, 0), selection.start));
  const after = doc.getText(new vscode.Range(selection.end, fullEnd));
  const fullProposed = before + proposed + after;

  diffCounter += 1;
  const name = path.basename(doc.fileName) || "selection";
  const proposedUri = vscode.Uri.parse(
    `${SCHEME}:/${encodeURIComponent(name)} (proposed)?${diffCounter}`,
  );
  provider.set(proposedUri, fullProposed);

  await vscode.commands.executeCommand(
    "vscode.diff",
    doc.uri,
    proposedUri,
    `LLMStack edit: ${name} (review)`,
  );

  const choice = await vscode.window.showInformationMessage(
    "Apply LLMStack's edit to your file?",
    { modal: true },
    "Apply",
  );
  if (choice === "Apply") {
    const edit = new vscode.WorkspaceEdit();
    edit.replace(doc.uri, selection, proposed);
    await vscode.workspace.applyEdit(edit);
  }
}
