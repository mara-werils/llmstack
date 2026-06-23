# Reproducible Benchmarks

Most "X is faster/cheaper than Cursor" claims are unverifiable — you cannot
re-run them. llmstack ships the opposite: a **reproducible benchmark harness** you
run on your own machine, against your own model, that reports cost, latency, and a
zero-egress privacy proof in one shareable artifact.

```bash
llmstack benchmark                      # run the default suite against local Ollama
llmstack benchmark --mock               # deterministic demo, no model required
llmstack benchmark -b gpt-4o -o out.md  # compare cost vs GPT-4o, save report
```

## What it measures

- **Latency** — per-task wall time, summarised as mean / p50 / p95 / p99 / min / max.
- **Throughput** — output tokens per second across the run.
- **Cost vs cloud** — the dollars a metered cloud baseline would have charged for
  the exact same token volume, priced from a [dated, sourced catalog](savings.md).
- **Privacy** — the run executes under the runtime egress monitor, so the report
  states whether it made *zero* external connections (see
  [Privacy & the No-Egress Proof](privacy.md)).

## What makes it reproducible

Every report carries a **methodology hash** — a SHA-256 fingerprint of the
benchmark *definition*: the suite name and version, the exact task ids, the cloud
baseline, and the pricing snapshot. Crucially, it does **not** include the measured
latencies (which depend on your hardware). So two people who run the same suite
version against the same baseline get the **same methodology hash** and can confirm
they benchmarked an identical definition — even though their latency numbers differ.

The suite itself (`llmstack.benchmark.spec`) is fixed and versioned: the prompts
never change for a given version. Bump the version and the hash changes, by design.

## Honest by construction

The harness never prints a cloud latency number we cannot reproduce. It compares
only what can be stated fairly: the cloud **cost** for the same tokens (from list
pricing) and the **privacy** difference (a metered cloud API receives your prompt
off-device; your local run, per the attached proof, sent nothing). Latency is
reported as *measured on your machine* — your number, not ours.

## The report

`llmstack benchmark -o report.md` writes both `report.md` (human-readable) and
`report.json` (machine-readable, with the full methodology hash, environment,
per-task results, comparison, and egress proof). The environment block records
only non-identifying facts — OS, architecture, core count, RAM, GPU — never
hostname, username, or IP.

## In CI

The `Benchmark` workflow runs the harness in deterministic `--mock` mode on every
change to the benchmark code, **fails the build on any external egress**, and
uploads the report as an artifact. You can reproduce that gate locally:

```bash
python examples/benchmark_proof.py    # exits non-zero if any external connection is seen
```

## Programmatic use

```python
from llmstack.benchmark import get_suite, run_benchmark, Generation

def generate(prompt: str) -> Generation:
    # wrap Ollama, the gateway, or anything that returns text + token counts
    ...

report = run_benchmark(get_suite("default"), generate, model="llama3.2", baseline="gpt-4o")
print(report.to_markdown())
print(report.methodology_hash)
```
