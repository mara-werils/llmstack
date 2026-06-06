"""Advanced rate limiter v2 — sliding window with burst, per-endpoint limits, and quotas."""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum


SECONDS_PER_MINUTE = 60
SECONDS_PER_HOUR = 3600
SECONDS_PER_DAY = 86400


class RateLimitTier(str, Enum):
    FREE = "free"
    STANDARD = "standard"
    PRO = "pro"
    ENTERPRISE = "enterprise"


@dataclass
class TierConfig:
    """Rate limit configuration per tier."""

    requests_per_minute: int
    requests_per_hour: int
    requests_per_day: int
    burst_size: int  # Max burst above normal rate
    max_tokens_per_request: int
    daily_token_quota: int
    concurrent_requests: int


TIER_CONFIGS = {
    RateLimitTier.FREE: TierConfig(
        requests_per_minute=10,
        requests_per_hour=100,
        requests_per_day=500,
        burst_size=5,
        max_tokens_per_request=2048,
        daily_token_quota=50000,
        concurrent_requests=2,
    ),
    RateLimitTier.STANDARD: TierConfig(
        requests_per_minute=30,
        requests_per_hour=500,
        requests_per_day=5000,
        burst_size=15,
        max_tokens_per_request=4096,
        daily_token_quota=500000,
        concurrent_requests=5,
    ),
    RateLimitTier.PRO: TierConfig(
        requests_per_minute=100,
        requests_per_hour=2000,
        requests_per_day=20000,
        burst_size=50,
        max_tokens_per_request=8192,
        daily_token_quota=2000000,
        concurrent_requests=10,
    ),
    RateLimitTier.ENTERPRISE: TierConfig(
        requests_per_minute=500,
        requests_per_hour=10000,
        requests_per_day=100000,
        burst_size=200,
        max_tokens_per_request=32768,
        daily_token_quota=10000000,
        concurrent_requests=50,
    ),
}


@dataclass
class RateLimitResult:
    """Result of a rate limit check."""

    allowed: bool
    limit: int
    remaining: int
    reset_at: float
    retry_after: float = 0
    reason: str = ""


@dataclass
class SlidingWindowCounter:
    """Sliding window rate counter."""

    window_seconds: float
    max_requests: int
    timestamps: list[float] = field(default_factory=list)

    def check_and_record(self, now: float | None = None) -> RateLimitResult:
        now = now or time.time()
        cutoff = now - self.window_seconds

        # Remove expired timestamps
        self.timestamps = [t for t in self.timestamps if t > cutoff]

        remaining = self.max_requests - len(self.timestamps)
        reset_at = (
            (self.timestamps[0] + self.window_seconds)
            if self.timestamps
            else now + self.window_seconds
        )

        if len(self.timestamps) >= self.max_requests:
            retry_after = self.timestamps[0] + self.window_seconds - now
            return RateLimitResult(
                allowed=False,
                limit=self.max_requests,
                remaining=0,
                reset_at=reset_at,
                retry_after=max(0, retry_after),
                reason=f"Rate limit exceeded ({self.max_requests}/{self.window_seconds}s)",
            )

        self.timestamps.append(now)
        return RateLimitResult(
            allowed=True,
            limit=self.max_requests,
            remaining=remaining - 1,
            reset_at=reset_at,
        )


class AdvancedRateLimiter:
    """Per-key, per-endpoint rate limiter with tiers and quotas."""

    def __init__(self):
        # key -> {window_name -> SlidingWindowCounter}
        self._windows: dict[str, dict[str, SlidingWindowCounter]] = defaultdict(dict)
        self._token_usage: dict[str, int] = defaultdict(int)  # key -> daily tokens
        self._token_reset: dict[str, float] = {}  # key -> next reset time
        self._concurrent: dict[str, int] = defaultdict(int)  # key -> active requests
        self._key_tiers: dict[str, RateLimitTier] = {}

    def set_tier(self, key: str, tier: RateLimitTier) -> None:
        """Set rate limit tier for a key."""
        self._key_tiers[key] = tier

    def get_tier(self, key: str) -> RateLimitTier:
        """Get tier for a key."""
        return self._key_tiers.get(key, RateLimitTier.FREE)

    def check(self, key: str, endpoint: str = "default", tokens: int = 0) -> RateLimitResult:
        """Check if request is allowed."""
        tier = self.get_tier(key)
        config = TIER_CONFIGS[tier]
        now = time.time()

        # Initialize windows if needed
        composite_key = f"{key}:{endpoint}"
        if "minute" not in self._windows[composite_key]:
            self._windows[composite_key] = {
                "minute": SlidingWindowCounter(SECONDS_PER_MINUTE, config.requests_per_minute + config.burst_size),
                "hour": SlidingWindowCounter(SECONDS_PER_HOUR, config.requests_per_hour),
                "day": SlidingWindowCounter(SECONDS_PER_DAY, config.requests_per_day),
            }

        # Check concurrent requests
        if self._concurrent[key] >= config.concurrent_requests:
            return RateLimitResult(
                allowed=False,
                limit=config.concurrent_requests,
                remaining=0,
                reset_at=now + 1,
                retry_after=1,
                reason=f"Max concurrent requests ({config.concurrent_requests}) exceeded",
            )

        # Check each window
        for window_name in ["minute", "hour", "day"]:
            result = self._windows[composite_key][window_name].check_and_record(now)
            if not result.allowed:
                return result

        # Check token quota
        reset_time = self._token_reset.get(key, 0)
        if now > reset_time:
            self._token_usage[key] = 0
            self._token_reset[key] = now + SECONDS_PER_DAY  # Reset daily

        if tokens > 0:
            if tokens > config.max_tokens_per_request:
                return RateLimitResult(
                    allowed=False,
                    limit=config.max_tokens_per_request,
                    remaining=0,
                    reset_at=now,
                    reason=f"Token limit per request ({config.max_tokens_per_request}) exceeded",
                )

            if self._token_usage[key] + tokens > config.daily_token_quota:
                return RateLimitResult(
                    allowed=False,
                    limit=config.daily_token_quota,
                    remaining=config.daily_token_quota - self._token_usage[key],
                    reset_at=self._token_reset.get(key, now + 86400),
                    reason="Daily token quota exceeded",
                )

            self._token_usage[key] += tokens

        self._concurrent[key] += 1
        return RateLimitResult(
            allowed=True,
            limit=config.requests_per_minute,
            remaining=config.requests_per_minute
            - len(self._windows[composite_key]["minute"].timestamps),
            reset_at=now + 60,
        )

    def is_rate_limited(self, key: str, endpoint: str = "default") -> bool:
        """Return True if the key is currently rate-limited for the endpoint."""
        tier = self.get_tier(key)
        config = TIER_CONFIGS[tier]
        now = time.time()
        composite_key = f"{key}:{endpoint}"

        if self._concurrent.get(key, 0) >= config.concurrent_requests:
            return True

        windows = self._windows.get(composite_key)
        if not windows:
            return False

        for window in windows.values():
            cutoff = now - window.window_seconds
            active = [t for t in window.timestamps if t > cutoff]
            if len(active) >= window.max_requests:
                return True

        return False

    def reset_user(self, key: str) -> None:
        """Clear all counters for a given API key (e.g. after account upgrade)."""
        self._windows.pop(key, None)
        self._token_usage.pop(key, None)
        self._token_reset.pop(key, None)
        self._concurrent[key] = 0

    def release(self, key: str) -> None:
        """Release a concurrent request slot."""
        if self._concurrent[key] > 0:
            self._concurrent[key] -= 1

    def get_usage(self, key: str) -> dict:
        """Get usage statistics for a key."""
        tier = self.get_tier(key)
        config = TIER_CONFIGS[tier]

        return {
            "tier": tier.value,
            "token_usage": self._token_usage.get(key, 0),
            "token_quota": config.daily_token_quota,
            "token_remaining": config.daily_token_quota - self._token_usage.get(key, 0),
            "concurrent_active": self._concurrent.get(key, 0),
            "concurrent_limit": config.concurrent_requests,
        }
