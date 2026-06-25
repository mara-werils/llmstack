# LLMStack for VS Code

Local-first AI coding assistance, right inside your editor. The extension talks
to **your own** [LLMStack](https://github.com/mara-werils/llmstack) gateway, so
your code and prompts **never leave your machine**.

> Works in VS Code, Cursor, Windsurf, and any OpenVSX-compatible editor.

## Features

- **Chat sidebar** — a dedicated view in the activity bar. Streams replies with
  markdown + code blocks, a model picker, and an optional "include editor
  context" toggle. Every code block has **Apply**, **Insert**, and **Copy**.
- **Edit with AI** (`Cmd/Ctrl+Alt+E`) — select code, describe a change, and
  review it as a **native diff**. Nothing is written until you approve, and
  **Revert last AI edit** undoes it in one step.
- **👍 / 👎 feedback** — rate chat replies; feedback flows into the gateway's
  adaptive-learning pipeline so your local model improves over time.
- **Ask about selection** (`Cmd/Ctrl+Alt+A`) — quick questions with the current
  selection as context.
- **Explain selected code** — right-click any selection → _LLMStack: Explain
  selected code_.
- **Inline completions** (opt-in) — ghost-text suggestions as you type, powered
  by your local model. Toggle with `llmstack.inlineCompletion.enabled`; tune
  `debounceMs` and opt languages out with `disabledLanguages`.
- **Run quickstart** — _LLMStack: Run quickstart_ sets up a local model from
  inside the editor, and a first-run check offers it automatically when you're
  not ready yet.
- **Switch model** — _LLMStack: Switch model_ picks from the gateway's models.
- **Gateway health** — a status-bar indicator (and an in-panel banner) shows
  whether your local gateway is reachable, with a one-click **Start gateway**.

## Requirements

A running LLMStack gateway:

```bash
brew install mara-werils/llmstack/llmstack   # or: pipx install llmstack-cli
llmstack init
llmstack up
```

By default the extension connects to `http://localhost:8000`.

## Settings

| Setting | Default | Description |
| --- | --- | --- |
| `llmstack.gatewayUrl` | `http://localhost:8000` | Base URL of your gateway |
| `llmstack.apiKey` | _(empty)_ | API key (only if `gateway.auth=api_key`) |
| `llmstack.model` | `llama3.2` | Model used for completions |
| `llmstack.inlineCompletion.enabled` | `false` | Show ghost-text completions as you type |
| `llmstack.inlineCompletion.contextLines` | `50` | Lines of surrounding code sent as context |
| `llmstack.inlineCompletion.maxTokens` | `200` | Max tokens generated per completion |

## Privacy

Every request goes to the gateway URL you configure — by default a process on
`localhost`. Run `llmstack verify-private` to audit that your stack keeps all
data local.

## Development

```bash
cd editors/vscode
npm install
npm run build          # compile to dist/
npm run package        # produce a .vsix
npm run publish:ovsx   # publish to OpenVSX (CI does this on release)
```

## License

Apache-2.0
