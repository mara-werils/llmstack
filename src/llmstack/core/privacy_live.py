"""Live privacy probe — checks a *running* gateway, not just static config.

``audit_privacy`` (in :mod:`llmstack.core.privacy`) is intentionally pure: it
only ever reads ``llmstack.yaml``. But the gateway process can be started
with environment-variable overrides (``LLMSTACK_PROVIDERS_CONFIG``,
``LLMSTACK_CORS_ORIGINS``, ``LLMSTACK_API_KEYS``, ...) that diverge from the
YAML on disk — a static-only audit can miss real egress that way.

This module makes a few unauthenticated HTTP requests against a *running*
gateway to verify what it is actually doing right now: which providers it is
really serving models from, whether it actually enforces auth, and whether
it actually sends a wide-open CORS header.
"""

from __future__ import annotations

import httpx

from llmstack.core.privacy import INFO, WARNING, CRITICAL, Finding, LOCAL_PROVIDERS

DEFAULT_TIMEOUT = 5.0


async def probe_live_gateway(base_url: str, timeout: float = DEFAULT_TIMEOUT) -> list[Finding]:
    """Probe a running gateway for live privacy issues.

    Returns a list of :class:`Finding`. Network failures (gateway not
    running, wrong URL, ...) produce a single INFO finding rather than
    raising, so callers can always merge this into a :class:`PrivacyReport`.
    """
    base_url = base_url.rstrip("/")
    findings: list[Finding] = []

    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            await client.get(f"{base_url}/healthz")
        except httpx.HTTPError:
            findings.append(
                Finding(
                    INFO,
                    "live-probe",
                    f"Could not reach gateway at {base_url} — live probe skipped.",
                    "Run 'llmstack up' and re-run with --live to verify the running gateway.",
                )
            )
            return findings

        findings.extend(await _probe_live_providers(client, base_url))
        findings.extend(await _probe_live_cors(client, base_url))
        findings.extend(await _probe_live_auth(client, base_url))

    return findings


async def _probe_live_providers(client: httpx.AsyncClient, base_url: str) -> list[Finding]:
    """Check which providers the running gateway is actually serving models from."""
    try:
        resp = await client.get(f"{base_url}/v1/models")
        resp.raise_for_status()
        data = resp.json()
    except (httpx.HTTPError, ValueError):
        return []

    external: set[str] = set()
    for model in data.get("data", []):
        # A malformed /v1/models payload (non-dict entries) must not crash the
        # live privacy probe with an AttributeError -- skip anything unexpected.
        if not isinstance(model, dict):
            continue
        x_llmstack = model.get("x_llmstack")
        x_provider = x_llmstack.get("provider") if isinstance(x_llmstack, dict) else None
        provider = (model.get("owned_by") or x_provider or "").lower()
        if provider and provider not in LOCAL_PROVIDERS:
            external.add(provider)

    return [
        Finding(
            CRITICAL,
            "live-providers",
            f"Running gateway is actually serving models from external provider '{provider}'.",
            f"Disable provider '{provider}', or check whether LLMSTACK_PROVIDERS_CONFIG "
            "is overriding llmstack.yaml at runtime.",
        )
        for provider in sorted(external)
    ]


async def _probe_live_cors(client: httpx.AsyncClient, base_url: str) -> list[Finding]:
    """Check whether the running gateway actually echoes a wide-open CORS header."""
    try:
        resp = await client.options(
            f"{base_url}/v1/models",
            headers={
                "Origin": "https://example.com",
                "Access-Control-Request-Method": "GET",
            },
        )
    except httpx.HTTPError:
        return []

    if resp.headers.get("access-control-allow-origin", "") == "*":
        return [
            Finding(
                WARNING,
                "live-cors",
                "Running gateway echoes 'Access-Control-Allow-Origin: *' — any website "
                "can call it from a browser.",
                "Restrict gateway.cors to trusted origins and restart the gateway.",
            )
        ]
    return []


async def _probe_live_auth(client: httpx.AsyncClient, base_url: str) -> list[Finding]:
    """Check whether the running gateway actually rejects unauthenticated requests."""
    try:
        resp = await client.get(f"{base_url}/v1/models")
    except httpx.HTTPError:
        return []

    if resp.status_code == 200:
        return [
            Finding(
                WARNING,
                "live-auth",
                "Running gateway accepted an unauthenticated request to /v1/models.",
                "Set gateway.auth=api_key and configure api_keys, then restart the gateway.",
            )
        ]
    return []
