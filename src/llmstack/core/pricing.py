"""Dated, sourced pricing catalog for the paid alternatives to llmstack.

llmstack's headline claim is that it *saves you money* by running locally instead
of paying for Cursor, GitHub Copilot, or a metered cloud API. To make that claim
**provable** rather than asserted, we keep an explicit, dated, and sourced catalog
of what those alternatives actually cost:

* :data:`SUBSCRIPTIONS` — per-seat monthly plans (Copilot, Cursor, ChatGPT Plus, …).
* :data:`API_PRICING` — metered per-token API prices (OpenAI, Anthropic, Google).

Every entry carries an ``as_of`` month and a ``source`` URL so a reader can verify
the number. Nothing here touches the network; this is static reference data that
:mod:`llmstack.core.savings` turns into a concrete "dollars saved" figure.

These are public list prices captured for comparison only and may change; update
:data:`PRICING_AS_OF` and the affected rows when they do.
"""

from __future__ import annotations

from dataclasses import dataclass

# The month this catalog was last reviewed. Bump when any row changes.
PRICING_AS_OF = "2026-06"


@dataclass(frozen=True)
class SubscriptionPlan:
    """A flat per-seat subscription for a paid AI coding/chat tool."""

    key: str
    name: str
    vendor: str
    monthly_usd: float
    source: str
    annual_usd: float | None = None
    as_of: str = PRICING_AS_OF

    def __post_init__(self) -> None:
        # A non-positive price is a catalog data-entry error: it makes the
        # "months of subscription covered" comparison meaningless (and risks a
        # divide-by-zero), so reject it at construction rather than silently
        # producing a bogus savings figure.
        if self.monthly_usd <= 0:
            raise ValueError(f"monthly_usd must be positive (got {self.monthly_usd})")
        if self.annual_usd is not None and self.annual_usd <= 0:
            raise ValueError(f"annual_usd must be positive when set (got {self.annual_usd})")

    @property
    def effective_monthly_usd(self) -> float:
        """Monthly cost, using the annual plan's per-month rate when cheaper."""
        if self.annual_usd is not None:
            annual_monthly = self.annual_usd / 12.0
            return min(self.monthly_usd, annual_monthly)
        return self.monthly_usd


@dataclass(frozen=True)
class TokenPrice:
    """Metered API price for a model, in USD per 1,000,000 tokens."""

    model: str
    vendor: str
    input_per_million: float
    output_per_million: float
    source: str
    as_of: str = PRICING_AS_OF

    def cost_usd(self, input_tokens: int, output_tokens: int) -> float:
        """Cost in USD of a call with the given token counts."""
        return (
            input_tokens * self.input_per_million + output_tokens * self.output_per_million
        ) / 1_000_000.0


_OPENAI_SRC = "https://openai.com/api/pricing/"
_ANTHROPIC_SRC = "https://www.anthropic.com/pricing"
_GOOGLE_SRC = "https://ai.google.dev/pricing"
_COPILOT_SRC = "https://github.com/features/copilot/plans"
_CURSOR_SRC = "https://cursor.com/pricing"
_OPENAI_PLUS_SRC = "https://openai.com/chatgpt/pricing/"
_CLAUDE_PRO_SRC = "https://www.anthropic.com/pricing"


# Per-seat subscription plans for the paid alternatives, keyed by a stable slug.
SUBSCRIPTIONS: dict[str, SubscriptionPlan] = {
    plan.key: plan
    for plan in (
        SubscriptionPlan("copilot-pro", "GitHub Copilot Pro", "GitHub", 10.0, _COPILOT_SRC, 100.0),
        SubscriptionPlan(
            "copilot-business", "GitHub Copilot Business", "GitHub", 19.0, _COPILOT_SRC
        ),
        SubscriptionPlan("cursor-pro", "Cursor Pro", "Anysphere", 20.0, _CURSOR_SRC, 192.0),
        SubscriptionPlan("chatgpt-plus", "ChatGPT Plus", "OpenAI", 20.0, _OPENAI_PLUS_SRC),
        SubscriptionPlan("claude-pro", "Claude Pro", "Anthropic", 20.0, _CLAUDE_PRO_SRC, 200.0),
    )
}


# Metered API prices (USD per 1M tokens), keyed by model id.
API_PRICING: dict[str, TokenPrice] = {
    price.model: price
    for price in (
        TokenPrice("gpt-4o", "OpenAI", 2.50, 10.00, _OPENAI_SRC),
        TokenPrice("gpt-4o-mini", "OpenAI", 0.15, 0.60, _OPENAI_SRC),
        TokenPrice("gpt-4-turbo", "OpenAI", 10.00, 30.00, _OPENAI_SRC),
        TokenPrice("claude-sonnet-4", "Anthropic", 3.00, 15.00, _ANTHROPIC_SRC),
        TokenPrice("claude-haiku-4-5", "Anthropic", 1.00, 5.00, _ANTHROPIC_SRC),
        TokenPrice("gemini-2.0-flash", "Google", 0.075, 0.30, _GOOGLE_SRC),
        TokenPrice("gemini-2.5-pro", "Google", 1.25, 10.00, _GOOGLE_SRC),
    )
}


# The default metered baseline used to value a local request: a cheap, mainstream
# cloud model, chosen so "dollars saved" is a conservative figure we can defend.
DEFAULT_API_BASELINE = "gpt-4o-mini"

# The default subscription baseline for the per-seat comparison.
DEFAULT_SUBSCRIPTION_BASELINE = "copilot-pro"


def get_subscription(key: str) -> SubscriptionPlan:
    """Return the subscription plan for ``key`` or raise ``KeyError``."""
    return SUBSCRIPTIONS[key]


def get_token_price(model: str) -> TokenPrice:
    """Return the metered price for ``model`` or raise ``KeyError``."""
    return API_PRICING[model]


def baseline_token_price(model: str | None = None) -> TokenPrice:
    """Return the metered baseline price, defaulting to :data:`DEFAULT_API_BASELINE`."""
    return API_PRICING[model or DEFAULT_API_BASELINE]


def baseline_subscription(key: str | None = None) -> SubscriptionPlan:
    """Return the subscription baseline, defaulting to the Copilot Pro seat."""
    return SUBSCRIPTIONS[key or DEFAULT_SUBSCRIPTION_BASELINE]


def cheapest_subscription() -> SubscriptionPlan:
    """Return the least expensive subscription by effective monthly cost."""
    return min(SUBSCRIPTIONS.values(), key=lambda p: p.effective_monthly_usd)
