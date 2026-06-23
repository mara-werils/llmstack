"""Compare a local benchmark run against a cloud baseline.

The comparison is deliberately limited to claims we can defend: the *cost* the
baseline would have charged for the same token volume (from the dated pricing
catalog) and the *privacy* difference (the cloud option sends prompts off-device;
the local run, per the egress proof, did not). Latency is reported as measured —
never compared against an unsourced cloud number.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from llmstack.benchmark.baselines import CloudBaseline, get_baseline
from llmstack.benchmark.privacy import EgressProof
from llmstack.benchmark.runner import RunResult


@dataclass(frozen=True)
class Comparison:
    """Local-vs-cloud comparison on cost and privacy for one run."""

    baseline_key: str
    baseline_name: str
    total_input_tokens: int
    total_output_tokens: int
    cloud_cost_usd: float
    local_cost_usd: float
    saved_usd: float
    local_sends_offdevice: bool
    cloud_sends_offdevice: bool

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def compare(
    run: RunResult,
    *,
    baseline: str | None = None,
    egress_proof: EgressProof | None = None,
    local_cost_usd: float = 0.0,
) -> Comparison:
    """Build a :class:`Comparison` of ``run`` against a cloud ``baseline``."""
    b: CloudBaseline = get_baseline(baseline)
    cloud_cost = b.cost_usd(run.total_input_tokens, run.total_output_tokens)
    saved = max(0.0, cloud_cost - local_cost_usd)
    # Without a proof we make no privacy claim about the local run (assume the
    # weaker statement that it might have sent data), so default to True only when
    # we actually proved local-only.
    local_offdevice = (not egress_proof.is_local_only) if egress_proof is not None else False
    return Comparison(
        baseline_key=b.key,
        baseline_name=b.name,
        total_input_tokens=run.total_input_tokens,
        total_output_tokens=run.total_output_tokens,
        cloud_cost_usd=cloud_cost,
        local_cost_usd=local_cost_usd,
        saved_usd=saved,
        local_sends_offdevice=local_offdevice,
        cloud_sends_offdevice=b.sends_prompt_offdevice,
    )
