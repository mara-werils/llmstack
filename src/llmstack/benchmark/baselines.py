"""Cloud baselines a local benchmark run is compared against.

A baseline pairs a metered cloud model (for the *cost* dimension, priced from the
dated catalog in :mod:`llmstack.core.pricing`) with the inescapable *privacy*
fact that using it transmits your prompt off-device. The benchmark never claims a
cloud latency number we cannot reproduce; it compares the things we can state
honestly — cost and data egress — against the local run's measured latency.
"""

from __future__ import annotations

from dataclasses import dataclass

from llmstack.core.pricing import TokenPrice, get_token_price


@dataclass(frozen=True)
class CloudBaseline:
    """A cloud option to compare the local run against."""

    key: str
    name: str
    model: str
    # Every metered cloud API receives the prompt off the user's machine.
    sends_prompt_offdevice: bool = True

    @property
    def price(self) -> TokenPrice:
        return get_token_price(self.model)

    def cost_usd(self, input_tokens: int, output_tokens: int) -> float:
        """What this baseline would have charged for the given token usage."""
        return self.price.cost_usd(input_tokens, output_tokens)


CLOUD_BASELINES: dict[str, CloudBaseline] = {
    b.key: b
    for b in (
        CloudBaseline("gpt-4o-mini", "OpenAI GPT-4o mini", "gpt-4o-mini"),
        CloudBaseline("gpt-4o", "OpenAI GPT-4o", "gpt-4o"),
        CloudBaseline("claude-sonnet-4", "Anthropic Claude Sonnet 4", "claude-sonnet-4"),
        CloudBaseline("gemini-2.0-flash", "Google Gemini 2.0 Flash", "gemini-2.0-flash"),
    )
}

# A conservative default so "dollars saved" is defensible rather than inflated.
DEFAULT_BASELINE = "gpt-4o-mini"


def get_baseline(key: str | None = None) -> CloudBaseline:
    """Return a cloud baseline by key, defaulting to :data:`DEFAULT_BASELINE`."""
    return CLOUD_BASELINES[key or DEFAULT_BASELINE]
