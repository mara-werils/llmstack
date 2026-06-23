# How LLMStack Compares

An honest comparison with the most-adopted open / local AI coding tools. The goal
here is accuracy, not marketing — where a competitor is ahead, this page says so.

!!! note "Local models are table stakes"
    Running on Ollama / LM Studio / any OpenAI-compatible endpoint is **not** a
    differentiator anymore — Cline, Continue, and Tabby all support local models.
    What differs is the surrounding platform, the editor UX, and how *verifiable*
    the privacy claim is.

## Feature matrix

| Capability | LLMStack | Cline | Continue | Tabby | Aider |
| --- | --- | --- | --- | --- | --- |
| Runs fully local | ✅ | ✅ | ✅ | ✅ | ⚠️ needs a key by default |
| Editor chat panel | ✅ | ✅ | ✅ | ✅ | ❌ (terminal) |
| AI edit with diff + approval | ✅ | ✅ | ✅ | ⚠️ | ✅ (diff in terminal) |
| One-step revert / checkpoint | ✅ | ✅ | ⚠️ | ❌ | ✅ (git) |
| Inline (ghost-text) completion | ✅ | ❌ | ✅ | ✅ | ❌ |
| Model gateway + smart routing | ✅ | ❌ | ❌ | ❌ | ❌ |
| Built-in RAG over your code | ✅ | ⚠️ | ✅ | ✅ | ✅ (repo map) |
| Fine-tuning (QLoRA/LoRA) | ✅ | ❌ | ❌ | ❌ | ❌ |
| Adaptive learning from feedback | ✅ | ❌ | ❌ | ❌ | ❌ |
| MCP support | ✅ | ✅ | ✅ | ❌ | ❌ |
| Verifiable no-egress proof | ✅ | ❌ | ❌ | ⚠️ | ❌ |
| No telemetry by default | ✅ | ⚠️ | ❌ opt-out | ✅ | ✅ |
| VS Code Marketplace + Open VSX | 🚧 in progress | ✅ | ✅ | ✅ | n/a (CLI) |
| JetBrains plugin | ❌ planned | ✅ | ✅ | ✅ | n/a |

Legend: ✅ yes · ⚠️ partial · ❌ no · 🚧 in progress

## Where LLMStack leads

- **It's a platform, not just an extension.** A production gateway sits under the
  editor: smart routing (cheap model for easy prompts), semantic caching, rate
  limiting, cost tracking, A/B testing, and observability.
- **The privacy claim is verifiable.** `llmstack verify-private` (static) plus a
  runtime egress monitor let you *prove* no data leaves the machine — in CI. See
  the [Privacy guide](guide/privacy.md).
- **The value claims are reproducible.** `llmstack savings` tallies the cloud bill
  you didn't pay (vs [dated, sourced pricing](guide/savings.md)), and
  `llmstack benchmark` produces a cost + latency + zero-egress report carrying a
  methodology hash, so anyone can [reproduce it](guide/benchmarks.md) rather than
  take a marketing number on faith.
- **It learns from you.** Thumbs feedback and fine-tuning let the stack adapt to
  your codebase — locally, on your own data.

## Where LLMStack is catching up

- **Distribution.** The editor extension isn't on the marketplaces yet (it's
  built and CI-ready; publishing is a tagging step). Cline/Continue/Tabby are
  already broadly installed.
- **JetBrains.** No JetBrains plugin yet — it's on the [roadmap](https://github.com/mara-werils/llmstack/blob/main/ROADMAP.md).

## A note on numbers

Public install and star counts (e.g. Cline's millions of installs) are
point-in-time, vendor-reported, and measure *cumulative installs* — not retained,
active users. We avoid quoting adoption figures we can't reproduce, and we don't
claim benchmark wins without a reproducible methodology. That's why the numbers
llmstack does report — savings and benchmarks — ship as tools you run yourself
(`llmstack savings`, `llmstack benchmark`), each valued against dated, sourced
pricing and fingerprinted with a methodology hash. The figures are yours, not ours.
