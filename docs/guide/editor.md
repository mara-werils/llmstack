# Editor Extension (VS Code & forks)

The LLMStack extension brings your local gateway into the editor: a chat sidebar,
AI edits you review as a diff before they touch your code, inline completions, and
one-click explanations — all talking to **your** gateway. Code never leaves your
machine.

## Install

=== "VS Code"

    Install **LLMStack — Local AI** from the Visual Studio Marketplace, or:

    ```
    code --install-extension llmstack.llmstack-vscode
    ```

=== "Cursor / Windsurf / VSCodium"

    These VS Code forks cannot use Microsoft's marketplace; they use the
    [Open VSX Registry](https://open-vsx.org). Search **LLMStack** in the
    Extensions view, or:

    ```
    # Cursor / Windsurf / VSCodium all accept VSIX installs too:
    codium --install-extension llmstack.llmstack-vscode
    ```

=== "From source"

    ```
    cd editors/vscode
    npm install
    npm run package          # produces llmstack-vscode-*.vsix
    code --install-extension llmstack-vscode-*.vsix
    ```

!!! tip "First, start the gateway"
    The extension talks to a local gateway. Run `llmstack up` (or `llmstack serve`)
    first. The status-bar item shows whether the gateway is reachable.

## Features

| Surface | What it does |
| --- | --- |
| **Chat sidebar** | Streaming chat in the activity bar, with markdown + code blocks. Pick the model, optionally include the active selection/file as context. |
| **Apply / Insert / Copy** | Every code block in a reply can be applied over your selection, inserted at the cursor, or copied. |
| **Edit with AI** (`Cmd/Ctrl+Alt+E`) | Select code, describe a change, and review the result as a **native diff** — nothing is written until you approve. |
| **Revert last AI edit** | One command restores the file to its pre-edit state. |
| **Inline completion** | Opt-in ghost-text completions from your local model. |
| **Ask / Explain** (`Cmd/Ctrl+Alt+A`) | Quick questions and selection explanations. |
| **👍 / 👎 feedback** | Rate replies; feedback flows into the gateway's adaptive-learning pipeline. |

## Settings

| Setting | Default | Description |
| --- | --- | --- |
| `llmstack.gatewayUrl` | `http://localhost:8000` | Base URL of your local gateway. |
| `llmstack.apiKey` | `""` | API key, only if `gateway.auth=api_key`. |
| `llmstack.model` | `llama3.2` | Default model (override per-chat in the panel). |
| `llmstack.inlineCompletion.enabled` | `false` | Enable ghost-text completions. |
| `llmstack.inlineCompletion.contextLines` | `50` | Lines of surrounding context to send. |
| `llmstack.inlineCompletion.maxTokens` | `200` | Max tokens per completion. |

## Privacy

Every request goes to the gateway URL you configure — by default `localhost`.
Run [`llmstack verify-private`](gateway.md) to audit that nothing in your
configuration sends data off the machine.
