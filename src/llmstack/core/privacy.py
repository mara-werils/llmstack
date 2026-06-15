"""Privacy audit — verify the stack runs 100% locally with no external egress.

This powers ``llmstack verify-private``: it inspects a :class:`StackConfig`
and reports every place where data could leave the machine (cloud providers,
webhooks, agent tools that reach the network, wide-open CORS, …).

The audit is intentionally pure — it takes a config object and returns a
:class:`PrivacyReport`. No I/O, no network, fully unit-testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from urllib.parse import urlparse

from llmstack.config.schema import StackConfig

# Severity ordering for sorting/aggregation.
CRITICAL = "CRITICAL"
WARNING = "WARNING"
INFO = "INFO"

# Provider names that are inherently local (never leave the machine).
_LOCAL_PROVIDERS = {"local", "ollama", "vllm", "llamacpp", "llama.cpp", "tgi", "lmstudio"}

# Hostnames that resolve to this machine / the local Docker network.
_LOCAL_HOSTS = {
    "localhost",
    "127.0.0.1",
    "0.0.0.0",  # noqa: S104 — used only for host classification, not binding
    "::1",
    "host.docker.internal",
}

# Agent/MCP tools that can move data off the machine.
_EGRESS_TOOLS = {"http_get", "http_post", "web_search", "fetch_url", "browser"}


def is_local_url(url: str) -> bool:
    """Return True if ``url`` points at this machine or the local Docker network."""
    if not url:
        return True  # empty == provider default, treated as local backend
    parsed = urlparse(url if "://" in url else f"//{url}")
    host = (parsed.hostname or "").lower()
    if not host:
        return True
    if host in _LOCAL_HOSTS:
        return True
    # Docker-compose service names used by llmstack (e.g. llmstack-ollama).
    if host.startswith("llmstack") or host.endswith(".local"):
        return True
    return False


@dataclass
class Finding:
    """A single privacy finding."""

    severity: str
    category: str
    detail: str
    recommendation: str

    def to_dict(self) -> dict:
        return {
            "severity": self.severity,
            "category": self.category,
            "detail": self.detail,
            "recommendation": self.recommendation,
        }


@dataclass
class PrivacyReport:
    """Result of a privacy audit."""

    findings: list[Finding] = field(default_factory=list)

    @property
    def critical(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == CRITICAL]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == WARNING]

    @property
    def is_private(self) -> bool:
        """True when nothing can send data off the machine (no CRITICAL findings)."""
        return not self.critical

    @property
    def verdict(self) -> str:
        if self.critical:
            return "NOT PRIVATE"
        if self.warnings:
            return "PRIVATE (with warnings)"
        return "PRIVATE"

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "is_private": self.is_private,
            "critical": len(self.critical),
            "warnings": len(self.warnings),
            "findings": [f.to_dict() for f in self.findings],
        }


def _audit_providers(config: StackConfig, findings: list[Finding]) -> None:
    providers_cfg = config.providers
    if not providers_cfg.enabled:
        findings.append(
            Finding(
                INFO,
                "providers",
                "External provider routing is disabled — all inference is local.",
                "No action needed.",
            )
        )
        return

    external = []
    for provider in providers_cfg.providers:
        if not provider.enabled:
            continue
        if provider.name.lower() in _LOCAL_PROVIDERS:
            continue
        # A non-local provider is external unless it is explicitly pointed at a
        # local base_url. An empty base_url means it uses its public cloud API.
        if not provider.base_url or not is_local_url(provider.base_url):
            external.append(provider.name)

    for name in external:
        findings.append(
            Finding(
                CRITICAL,
                "providers",
                f"Provider '{name}' sends prompts to an external API — code leaves the machine.",
                f"Disable provider '{name}' or set providers.enabled=false for full privacy.",
            )
        )


def _audit_webhooks(config: StackConfig, findings: list[Finding]) -> None:
    webhooks = config.gateway.webhooks
    if not webhooks.enabled:
        return
    for endpoint in webhooks.endpoints:
        url = endpoint.get("url", "") if isinstance(endpoint, dict) else ""
        if not is_local_url(url):
            findings.append(
                Finding(
                    CRITICAL,
                    "webhooks",
                    f"Webhook forwards request data to an external URL: {url}",
                    "Point webhooks at a local endpoint or disable gateway.webhooks.",
                )
            )


def _audit_agent_tools(config: StackConfig, findings: list[Finding]) -> None:
    tool_sources: list[tuple[str, list[str]]] = []
    for profile in config.agents.profiles:
        tool_sources.append((f"agent profile '{profile.name}'", profile.tools))
    if config.mcp.enabled:
        tool_sources.append(("mcp server", config.mcp.tools))

    for source, tools in tool_sources:
        egress = sorted(set(tools) & _EGRESS_TOOLS)
        if egress:
            findings.append(
                Finding(
                    WARNING,
                    "agent-tools",
                    f"{source} enables network tool(s) {egress} that can reach external hosts.",
                    "Remove network tools if you need an airtight offline guarantee.",
                )
            )


def _audit_gateway(config: StackConfig, findings: list[Finding]) -> None:
    gateway = config.gateway
    if "*" in gateway.cors:
        findings.append(
            Finding(
                WARNING,
                "gateway-cors",
                "CORS allows any origin ('*') — any website can call your local gateway.",
                "Restrict gateway.cors to trusted origins (e.g. http://localhost:3000).",
            )
        )
    if gateway.auth == "none":
        findings.append(
            Finding(
                WARNING,
                "gateway-auth",
                "Gateway authentication is disabled — anyone on the network can use it.",
                "Set gateway.auth=api_key and configure api_keys.",
            )
        )


def _audit_guardrails(config: StackConfig, findings: list[Finding]) -> None:
    guardrails = config.gateway.guardrails
    if guardrails.enabled and guardrails.pii_detection:
        findings.append(
            Finding(
                INFO,
                "guardrails",
                "PII detection guardrail is enabled.",
                "No action needed.",
            )
        )


def audit_privacy(config: StackConfig) -> PrivacyReport:
    """Audit a :class:`StackConfig` for anything that breaks the local-only guarantee."""
    findings: list[Finding] = []
    _audit_providers(config, findings)
    _audit_webhooks(config, findings)
    _audit_agent_tools(config, findings)
    _audit_gateway(config, findings)
    _audit_guardrails(config, findings)
    return PrivacyReport(findings=findings)
