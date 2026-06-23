# Savings

llmstack runs on your own hardware, so every request it serves is one you did not
pay a cloud provider for. The savings engine turns that into a concrete, auditable
number — *"llmstack has saved you $X vs Copilot"* — instead of a vague promise.

```bash
llmstack savings                  # cumulative savings vs the default baseline
llmstack savings --plan cursor-pro
llmstack savings --json           # raw summary for scripts/dashboards
llmstack savings --reset          # start the ledger over
```

`llmstack savings` reads a local ledger (`~/.llmstack/savings.json`) directly — no
gateway, no network required. It is local-first, like everything else here.

## How the number is computed

When the gateway serves a request from a local backend, that request cost you
nothing. The savings engine values it against a **dated cloud baseline** and books
the difference as a saving:

- The default baseline is **`gpt-4o-mini`** — deliberately one of the *cheapest*
  mainstream cloud models, so the figure is conservative and defensible rather
  than inflated.
- Pricing comes from a dated, sourced catalog (`llmstack.core.pricing`). Every row
  carries an `as_of` month and a `source` URL. Inspect it any time:

```bash
curl localhost:8000/v1/savings/pricing | jq      # what the math is based on
curl localhost:8000/v1/savings/summary | jq      # the running total
```

The summary also tells you how many **months of a paid subscription** your savings
would have covered (Copilot Pro, Cursor Pro, ChatGPT Plus, Claude Pro).

## What is *not* counted

Requests routed to a metered cloud provider cost you real money, so they are
**never** booked as savings — only locally-served (free) requests count. The
saving for a request is also clamped at zero: if a local path somehow cost more
than the cloud baseline, it is reported as no saving, never as a loss.

## API

| Endpoint | Description |
| --- | --- |
| `GET /v1/savings/summary?plan=copilot-pro` | Cumulative savings + subscription-months covered |
| `GET /v1/savings/pricing` | The dated, sourced pricing the figure is derived from |
| `POST /v1/savings/reset` | Reset the ledger to zero |

## Programmatic use

```python
from llmstack.core.savings import SavingsCalculator, get_ledger

calc = SavingsCalculator()                      # default gpt-4o-mini baseline
est = calc.estimate(input_tokens=1200, output_tokens=400)
print(est.saved_usd)                            # what this one request saved

print(get_ledger().summary("cursor-pro"))       # the running total + Cursor-months
```
