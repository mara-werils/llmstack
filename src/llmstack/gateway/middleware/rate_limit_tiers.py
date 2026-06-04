"""Tiered rate limiting — different rate limits per API key tier.

Extends the base rate limiter with tier-based limits so enterprise
keys get higher throughput than free-tier keys.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from threading import Lock


@dataclass
class TierConfig:
    """Rate limit configuration for a specific tier."""

    name: str
    requests_per_minute: int = 100
    requests_per_hour: int = 0  # 0 = unlimited
    tokens_per_minute: int = 0  # 0 = unlimited
    burst_size: int = 0  # 0 = no burst allowance
    concurrent_requests: int = 0  # 0 = unlimited


# Default tier configurations
DEFAULT_TIERS: dict[str, TierConfig] = {
    "enterprise": TierConfig(
        name="enterprise",
        requests_per_minute=1000,
        requests_per_hour=50000,
        tokens_per_minute=2_000_000,
        burst_size=50,
        concurrent_requests=100,
    ),
    "pro": TierConfig(
        name="pro",
        requests_per_minute=300,
        requests_per_hour=10000,
        tokens_per_minute=500_000,
        burst_size=20,
        concurrent_requests=30,
    ),
    "standard": TierConfig(
        name="standard",
        requests_per_minute=100,
        requests_per_hour=3000,
        tokens_per_minute=100_000,
        burst_size=10,
        concurrent_requests=10,
    ),
    "free": TierConfig(
        name="free",
        requests_per_minute=20,
        requests_per_hour=500,
        tokens_per_minute=20_000,
        burst_size=5,
        concurrent_requests=3,
    ),
}


@dataclass
class KeyUsage:
    """Tracks usage for a single API key."""

    timestamps: list[float] = field(default_factory=list)
    tokens_used: list[tuple[float, int]] = field(default_factory=list)
    active_requests: int = 0

    def record_request(self) -> None:
        self.timestamps.append(time.time())

    def record_tokens(self, tokens: int) -> None:
        self.tokens_used.append((time.time(), tokens))

    def requests_in_window(self, window_seconds: float) -> int:
        cutoff = time.time() - window_seconds
        return sum(1 for ts in self.timestamps if ts >= cutoff)

    def tokens_in_window(self, window_seconds: float) -> int:
        cutoff = time.time() - window_seconds
        return sum(t for ts, t in self.tokens_used if ts >= cutoff)

    def cleanup(self, max_age: float = 7200) -> None:
        """Remove entries older than max_age seconds."""
        cutoff = time.time() - max_age
        self.timestamps = [ts for ts in self.timestamps if ts >= cutoff]
        self.tokens_used = [(ts, t) for ts, t in self.tokens_used if ts >= cutoff]


class TieredRateLimiter:
    """Rate limiter with per-tier and per-key configurations."""

    def __init__(self, tiers: dict[str, TierConfig] | None = None):
        self._lock = Lock()
        self._tiers = tiers or dict(DEFAULT_TIERS)
        self._usage: dict[str, KeyUsage] = defaultdict(KeyUsage)
        self._key_tiers: dict[str, str] = {}  # api_key -> tier name

    def set_key_tier(self, api_key: str, tier: str) -> None:
        """Assign an API key to a tier."""
        with self._lock:
            self._key_tiers[api_key] = tier

    def add_tier(self, config: TierConfig) -> None:
        """Add or update a tier configuration."""
        with self._lock:
            self._tiers[config.name] = config

    def check(self, api_key: str) -> tuple[bool, str]:
        """Check if a request is allowed for this API key.

        Returns (allowed, reason).
        """
        with self._lock:
            tier_name = self._key_tiers.get(api_key, "standard")
            tier = self._tiers.get(tier_name)
            if tier is None:
                return True, ""

            usage = self._usage[api_key]

            # Check requests per minute
            if tier.requests_per_minute > 0:
                rpm = usage.requests_in_window(60)
                if rpm >= tier.requests_per_minute:
                    return False, f"Rate limit exceeded: {rpm}/{tier.requests_per_minute} req/min"

            # Check requests per hour
            if tier.requests_per_hour > 0:
                rph = usage.requests_in_window(3600)
                if rph >= tier.requests_per_hour:
                    return False, f"Rate limit exceeded: {rph}/{tier.requests_per_hour} req/hour"

            # Check concurrent requests
            if tier.concurrent_requests > 0:
                if usage.active_requests >= tier.concurrent_requests:
                    return (
                        False,
                        f"Concurrent limit: {usage.active_requests}/{tier.concurrent_requests}",
                    )

            # Check tokens per minute
            if tier.tokens_per_minute > 0:
                tpm = usage.tokens_in_window(60)
                if tpm >= tier.tokens_per_minute:
                    return False, f"Token limit exceeded: {tpm}/{tier.tokens_per_minute} tok/min"

            return True, ""

    def record_request(self, api_key: str) -> None:
        """Record a request for rate tracking."""
        with self._lock:
            self._usage[api_key].record_request()
            self._usage[api_key].active_requests += 1

    def record_completion(self, api_key: str, tokens: int = 0) -> None:
        """Record request completion."""
        with self._lock:
            usage = self._usage[api_key]
            usage.active_requests = max(0, usage.active_requests - 1)
            if tokens > 0:
                usage.record_tokens(tokens)

    def get_limits(self, api_key: str) -> dict:
        """Get the rate limits for an API key."""
        with self._lock:
            tier_name = self._key_tiers.get(api_key, "standard")
            tier = self._tiers.get(tier_name)
            if tier is None:
                return {"tier": "unknown"}

            usage = self._usage.get(api_key)
            return {
                "tier": tier_name,
                "requests_per_minute": tier.requests_per_minute,
                "requests_per_hour": tier.requests_per_hour,
                "tokens_per_minute": tier.tokens_per_minute,
                "concurrent_requests": tier.concurrent_requests,
                "current_rpm": usage.requests_in_window(60) if usage else 0,
                "current_active": usage.active_requests if usage else 0,
            }

    def get_all_tiers(self) -> list[dict]:
        """Get all tier configurations."""
        with self._lock:
            return [
                {
                    "name": t.name,
                    "requests_per_minute": t.requests_per_minute,
                    "requests_per_hour": t.requests_per_hour,
                    "tokens_per_minute": t.tokens_per_minute,
                    "burst_size": t.burst_size,
                    "concurrent_requests": t.concurrent_requests,
                }
                for t in self._tiers.values()
            ]

    def cleanup(self) -> None:
        """Clean up old usage entries."""
        with self._lock:
            for usage in self._usage.values():
                usage.cleanup()
