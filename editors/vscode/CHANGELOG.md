# Changelog

All notable changes to the LLMStack VS Code extension are documented here.

## [0.3.0] — Unreleased

### Added
- **`LLMStack: Run quickstart`** — set up a local model from inside the editor.
- **First-run readiness check**: when the machine isn't ready for local inference,
  a one-click prompt offers to run quickstart (backed by the gateway's
  `/v1/onboarding`).
- **`LLMStack: Switch model`** — quick pick over the gateway's `/v1/models`.
- Inline completions: **configurable debounce** (`inlineCompletion.debounceMs`)
  and **per-language opt-out** (`inlineCompletion.disabledLanguages`).

## [0.2.0] — Unreleased

### Added
- **Chat sidebar** in the activity bar: streaming replies, markdown + code-block
  rendering, an in-panel model picker, and an "include editor context" toggle.
- **Apply / Insert / Copy** actions on every chat code block.
- **`LLMStack: Edit selection with AI…`** (`Cmd/Ctrl+Alt+E`): drafts an edit and
  shows it as a native diff for review before applying.
- **`LLMStack: Revert last AI edit`**: one-step checkpoint undo.
- **👍 / 👎 feedback** on chat replies, sent to the gateway learning pipeline.
- **Getting-started walkthrough** and an in-panel "gateway not reachable" banner
  with a one-click **Start gateway**.
- **`LLMStack: Show savings vs cloud`** command, and a running "saved $X so far"
  total folded into the status-bar tooltip when the gateway is reachable.

### Fixed
- Gateway health check now hits `/healthz/live` instead of the nonexistent
  `/health`, so the status bar reflects reality.

## [0.1.0] — Unreleased

### Added
- Initial release.
- `LLMStack: Ask about selection` command with keybinding (`Cmd/Ctrl+Alt+A`).
- `LLMStack: Explain selected code` editor context-menu command.
- `LLMStack: Check gateway health` command + status-bar indicator.
- Token-by-token streaming into the LLMStack output channel.
- Settings: `llmstack.gatewayUrl`, `llmstack.apiKey`, `llmstack.model`.
- Opt-in inline (ghost-text) code completion, toggled via
  `LLMStack: Toggle inline code completion` or `llmstack.inlineCompletion.enabled`.
- Extension icon and marketplace listing metadata (gallery banner, keywords).
