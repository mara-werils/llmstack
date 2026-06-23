# llmstack — Value-Proof Plan (provable savings + reproducible benchmarks)

> Goal: turn llmstack's two headline claims — *"the open-source alternative to
> Cursor and Copilot"* and *"actually saves you money"* — from marketing into
> **numbers a skeptic can reproduce on their own machine**. This is the natural
> sequel to the privacy work: that made *privacy* provable (static audit +
> runtime egress monitor + CI gate); this makes **cost and local performance**
> provable too.

## Why this, why now

`ROADMAP.md` lists exactly two unbuilt items that block v1.0, and both are about
proof, not features:

- v0.9: **"Open benchmarks vs Cursor/Copilot on privacy + local latency"**
- v1.0: **"Performance benchmarks vs alternatives"**

The research behind `ADOPTION_PLAN.md` was explicit that self-reported benchmark
numbers are under-evidenced and that we must **not** lead with claims we cannot
reproduce. The answer is not to drop the claim — it is to ship a **reproducible
methodology** so the numbers are the user's own, generated locally, with the
environment captured and a zero-egress proof attached.

## Two pillars

### 1. Savings engine — make "saves you money" a live number

Today the gateway prices *cloud API* calls (`gateway/cost_tracker.py`'s
`MODEL_PRICING`). It has no notion of **what you would have paid** on the paid
alternatives had you not run locally. We add:

- `core/pricing.py` — a dated, sourced catalog of the paid alternatives:
  per-seat subscriptions (Copilot, Cursor, ChatGPT Plus) **and** per-token API
  prices (OpenAI, Anthropic, Google). Pure data + lookup, no network.
- `core/savings.py` — a `SavingsCalculator` (given usage → equivalent cloud
  cost → dollars saved running locally) and a persistent `SavingsLedger`
  (`~/.llmstack/savings.json`) that accrues savings over time. Deterministic:
  injected clock + path, no I/O surprises.
- `gateway/savings.py` + `routes/savings.py` — accrue savings on every local
  request and expose `/v1/savings/summary`. Wired into the chat route.
- `llmstack savings` CLI — a shareable "llmstack has saved you $X vs Copilot"
  report. The viral hook.

### 2. Reproducible benchmark suite — make "alternative to Cursor/Copilot" measurable

A provider-agnostic harness under `benchmark/` that anyone can run:

- `spec.py` — a versioned, deterministic task suite (latency/coding/reasoning).
- `metrics.py` — TTFT, tokens/sec, latency percentiles (pure stats).
- `environment.py` — capture hardware/OS/python/model/version for reproducibility.
- `runner.py` — drive any `generate` callable; no network in tests (fake generator).
- `baselines.py` — published, sourced cloud latency/cost baselines to compare against.
- `compare.py` — local-vs-cloud table on **cost, latency, and privacy**.
- `privacy.py` — run the suite under the egress monitor; attach a zero-egress proof.
- `report.py` — deterministic JSON + Markdown report with a content hash so two
  runs of the same inputs produce the same artifact (reproducibility you can diff).
- `llmstack benchmark` CLI + a CI job that runs the harness in deterministic mock
  mode (and proves zero egress) on every push, uploading the report as an artifact.

## Quality bar (unchanged)

- Every new non-CLI module ships with tests in the same commit; keep coverage
  green (CI floor + the 95% local gate). CLI command modules are coverage-omitted.
- `ruff check` + `ruff format` clean; one focused, independently-green commit per
  unit of work, pushed to `main`.
- No invented numbers: pricing and baselines are **dated and sourced**, and the
  benchmark only ever reports what it measured on the host it ran on.

## Out of scope (kept honest)

- We do **not** hardcode a "llmstack is 10x faster" headline. The harness reports
  the user's measured numbers next to dated cloud baselines and lets them judge.
- JetBrains plugin and marketplace publishing remain blocked on external accounts.
