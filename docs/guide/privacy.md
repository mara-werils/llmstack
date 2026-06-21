# Privacy & the No-Egress Proof

LLMStack's core promise is simple: **your code and prompts never leave your
machine.** Unlike cloud assistants, that claim is something you can *verify* —
two ways, both reproducible.

## Layer 1 — static audit (`llmstack verify-private`)

`verify-private` inspects your `llmstack.yaml` and flags anything that could send
data off the machine: cloud providers, webhooks, network-capable agent/MCP tools,
wide-open CORS, or disabled gateway auth.

```bash
llmstack verify-private              # human-readable report
llmstack verify-private --json       # machine-readable, exits non-zero if not private
llmstack verify-private --live       # also probe the running gateway for env-var overrides
```

It exits non-zero when the local-only guarantee is broken, so you can gate CI on
it:

```yaml
# .github/workflows/privacy.yml (example)
- run: llmstack verify-private --json
```

Verdicts are `PRIVATE`, `PRIVATE (with warnings)`, or `NOT PRIVATE`.

## Layer 2 — runtime egress monitor

The static audit checks *configuration*. The runtime monitor checks *behavior*:
it records every outbound socket connection a block of code makes and flags any
that leave the local machine or private network. Loopback, RFC1918 private,
link-local, and the Docker network all count as local; anything else is external.

```python
from llmstack.core.egress import monitor_egress, assert_local_only

with monitor_egress() as mon:
    run_my_local_workload()          # e.g. llmstack ask over local files

assert_local_only(mon)               # raises ExternalEgressError if anything left the box
print(mon.connections)               # every (host, port) that was contacted
```

`is_local_host()` classifies hosts (IP literals via `ipaddress`, hostnames via
the same rules as the audit), so the proof is deterministic and dependency-free.

### Use it as a CI guarantee

Wrap a representative local workload in `monitor_egress()` and assert
`is_local_only` in a test. If a future change ever introduces an unexpected
outbound call — telemetry, a phone-home, a misrouted provider — the test fails.

!!! note "Scope"
    The monitor observes connections made by **the current process**. For the
    gateway's own upstream calls, run `verify-private --live` against the running
    gateway. Together the two layers cover both configuration and behavior.

## What LLMStack never does

- **No telemetry.** The CLI and gateway do not phone home. There is no analytics
  endpoint, no usage beacon, no opt-out required.
- **No hidden network calls.** Model inference goes to the local backend you
  configure (Ollama/vLLM/llama.cpp) or an explicit endpoint you set.
- **No account required.** Everything works offline after the model is pulled.
