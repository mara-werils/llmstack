# llmstack — 2026 Adoption Execution Plan

> Goal: become the local-first AI coding tool a large share of developers actually
> use. This plan is derived from a fact-checked market study of the most-adopted
> open/local AI coding tools (Cline, Continue, Tabby, Aider, Roo Code) and an audit
> of llmstack's current surface. It supersedes the high-level direction in
> `PRODUCT_STRATEGY.md` with concrete, buildable workstreams.

## What the evidence says

Adversarially-verified findings (high confidence) on what drives adoption:

1. **Agentic editing is the #1 editor-UX pattern.** Winners surface AI edits as an
   **in-editor diff → per-step human approval → checkpoint-based undo** loop
   (Cline's Plan/Act). Chat-only or autocomplete-only is not enough.
2. **Multi-channel distribution is the #1 scale lever.** Publish to the **VS Code
   Marketplace AND the Open VSX Registry** (and JetBrains). VS Code forks
   (Cursor, Windsurf, VSCodium) cannot reach Microsoft's marketplace and default
   to Open VSX — absence there = invisibility in those editors.
3. **Local-model support is table stakes, not a moat.** Cline/Continue/Tabby all
   run Ollama + any OpenAI-compatible endpoint. llmstack must win on **UX,
   distribution, and onboarding**, not on "it runs locally."
4. **The competitive feature bar is four modes:** Chat, Autocomplete, Edit
   (targeted natural-language edits), and Agent (multi-step). (Continue.)
5. **One command to first value wins** (Aider `curl | sh`; Tabby "Run in 1 minute").
   A **zero-key local default** — first value with no API key — is a concrete
   differentiation a local-first tool can own.
6. **Trust signals beyond "local" matter:** audited no-telemetry defaults,
   supply-chain hardening, and a *reproducible* privacy proof (not just a claim).

Honest caveats from the research: benchmark numbers (SWE-bench etc.) are
under-evidenced and self-reported install counts overstate active use, so we do
**not** lead with benchmark or install-count claims we cannot reproduce.

## Workstreams

### A. Editor-native agentic UX (the #1 gap) — VS Code extension
The extension today has Ask/Explain (output channel) + opt-in inline completion,
but **no chat panel and no edit/approve/undo loop**. Build:
- Chat sidebar (webview) with streaming, markdown + code-block rendering.
- Per-code-block actions: Apply to editor / Insert at cursor / Copy.
- Context inclusion: active selection / file, with a toggle.
- In-panel model picker (from `/v1/models`).
- **Edit mode:** selection → proposed edit shown as a **native VS Code diff** with
  **Apply / Reject** (the verified diff-review + approval pattern).
- **Checkpoint + revert:** snapshot before an AI edit; one-click undo.
- **Feedback (👍/👎)** wired to the gateway `/feedback` route — closes the loop to
  llmstack's adaptive-learning pipeline (a real differentiator).
- Persist chat across reloads; new/clear/stop generation.

### B. Onboarding to first value (single command, zero-key local default)
- One-line `curl | sh` installer script.
- Make first value reachable **without any API key** via a small bundled local
  default; emphasize this everywhere.
- VS Code getting-started walkthrough + a gateway-not-running welcome state.

### C. Multi-channel distribution + a real docs site
- Wire both publish channels (Open VSX + VS Marketplace) in CI (token-gated, so it
  no-ops safely until secrets exist).
- **Deploy the docs site** to GitHub Pages (mkdocs is configured but never
  deployed) — uses `GITHUB_TOKEN`, no external secret needed.
- Document install across editors (Cursor/Windsurf/VSCodium via Open VSX).
- README badges + install matrix refresh.

### D. Trust / privacy proof (the moat, evidence-informed)
- An **egress benchmark**: prove zero external network connections under a real
  workload (extends `verify-private --live`), with a reproducible methodology doc.
- `SECURITY.md` + a no-telemetry / supply-chain trust page.
- An honest comparison page vs Cline/Continue/Tabby on privacy + local.

### E. Quality
- Unit tests for all new Python; keep the 95% coverage gate green.
- Keep the extension `tsc` typecheck + build green at every commit.

## Execution

Shipped as a series of focused, independently-green commits to `main` — each with
tests/docs where applicable and `ruff` clean. Progress is reflected in
`ROADMAP.md`. Items blocked on external accounts (publishing tokens, JetBrains
Marketplace) are prepared up to the point a human must push the button.
