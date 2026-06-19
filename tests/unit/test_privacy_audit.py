"""Tests for the privacy audit (powers `llmstack verify-private`)."""

from __future__ import annotations

import pytest

from llmstack.config.schema import (
    AgentProfileConfig,
    AgentsConfig,
    GuardrailsConfig,
    MCPConfig,
    ProviderConfig,
    ProvidersConfig,
    StackConfig,
    WebhooksConfig,
)
from llmstack.core.privacy import CRITICAL, WARNING, audit_privacy, is_local_url


class TestIsLocalUrl:
    @pytest.mark.parametrize(
        "url",
        [
            "",
            "http://localhost:11434/v1",
            "http://127.0.0.1:8000",
            "http://host.docker.internal:11434",
            "http://llmstack-ollama:11434/v1",
            "http://myhost.local",
            "localhost:1234",
        ],
    )
    def test_local(self, url):
        assert is_local_url(url) is True

    @pytest.mark.parametrize(
        "url",
        [
            "https://api.openai.com/v1",
            "https://api.anthropic.com",
            "http://10.0.0.5:8000",
            "https://example.com",
            # A dotted host beginning with "llmstack" is external, not local.
            "http://llmstack.evil.com",
            "http://llmstack-attacker.example.com:8000",
        ],
    )
    def test_external(self, url):
        assert is_local_url(url) is False


def _private_config() -> StackConfig:
    """A config that is fully private with no warnings (locked-down CORS)."""
    config = StackConfig()
    config.gateway.cors = ["http://localhost:3000"]
    return config


class TestDefaultConfig:
    def test_default_is_private_with_cors_warning(self):
        # The shipped default opens CORS to '*', so it is private-with-warnings.
        report = audit_privacy(StackConfig())
        assert report.is_private is True
        assert report.verdict == "PRIVATE (with warnings)"
        assert report.critical == []

    def test_locked_down_config_is_fully_private(self):
        report = audit_privacy(_private_config())
        assert report.verdict == "PRIVATE"

    def test_to_dict_shape(self):
        d = audit_privacy(_private_config()).to_dict()
        assert d["verdict"] == "PRIVATE"
        assert d["critical"] == 0
        assert "findings" in d


class TestProviders:
    def test_cloud_provider_enabled_is_critical(self):
        config = StackConfig(
            providers=ProvidersConfig(
                enabled=True,
                providers=[ProviderConfig(name="openai", api_key_env="OPENAI_API_KEY")],
            )
        )
        report = audit_privacy(config)
        assert report.is_private is False
        assert report.verdict == "NOT PRIVATE"
        assert any(f.category == "providers" and f.severity == CRITICAL for f in report.critical)

    def test_disabled_provider_routing_is_fine(self):
        config = StackConfig(
            providers=ProvidersConfig(
                enabled=False,
                providers=[ProviderConfig(name="openai")],
            )
        )
        assert audit_privacy(config).is_private is True

    def test_disabled_individual_provider_skipped(self):
        config = StackConfig(
            providers=ProvidersConfig(
                enabled=True,
                providers=[ProviderConfig(name="openai", enabled=False)],
            )
        )
        assert audit_privacy(config).is_private is True

    def test_local_provider_name_is_fine(self):
        config = StackConfig(
            providers=ProvidersConfig(
                enabled=True,
                providers=[ProviderConfig(name="ollama")],
            )
        )
        assert audit_privacy(config).is_private is True

    def test_cloud_provider_with_local_base_url_is_fine(self):
        config = StackConfig(
            providers=ProvidersConfig(
                enabled=True,
                providers=[ProviderConfig(name="openai", base_url="http://localhost:1234/v1")],
            )
        )
        assert audit_privacy(config).is_private is True


class TestWebhooks:
    def test_external_webhook_is_critical(self):
        config = StackConfig()
        config.gateway.webhooks = WebhooksConfig(
            enabled=True, endpoints=[{"url": "https://hooks.example.com/x"}]
        )
        assert audit_privacy(config).is_private is False

    def test_local_webhook_is_fine(self):
        config = StackConfig()
        config.gateway.webhooks = WebhooksConfig(
            enabled=True, endpoints=[{"url": "http://localhost:9000/hook"}]
        )
        assert audit_privacy(config).is_private is True

    def test_disabled_webhooks_ignored(self):
        config = StackConfig()
        config.gateway.webhooks = WebhooksConfig(
            enabled=False, endpoints=[{"url": "https://hooks.example.com/x"}]
        )
        assert audit_privacy(config).is_private is True


class TestAgentTools:
    def test_http_tool_is_warning(self):
        config = StackConfig(
            agents=AgentsConfig(
                profiles=[AgentProfileConfig(name="default", tools=["read_file", "http_get"])]
            )
        )
        report = audit_privacy(config)
        assert report.is_private is True  # warning, not critical
        assert any(f.category == "agent-tools" and f.severity == WARNING for f in report.warnings)

    def test_mcp_http_tool_is_warning(self):
        config = StackConfig(mcp=MCPConfig(enabled=True, tools=["read_file", "http_get"]))
        assert any(f.category == "agent-tools" for f in audit_privacy(config).warnings)

    def test_no_network_tools_no_finding(self):
        config = StackConfig(
            agents=AgentsConfig(
                profiles=[AgentProfileConfig(name="default", tools=["read_file", "grep"])]
            )
        )
        assert not [f for f in audit_privacy(config).findings if f.category == "agent-tools"]


class TestGateway:
    def test_wildcard_cors_is_warning(self):
        report = audit_privacy(StackConfig())  # default cors is ["*"]
        assert any(f.category == "gateway-cors" for f in report.warnings)
        assert report.verdict == "PRIVATE (with warnings)"

    def test_auth_none_is_warning(self):
        config = StackConfig()
        config.gateway.auth = "none"
        config.gateway.cors = ["http://localhost:3000"]
        assert any(f.category == "gateway-auth" for f in audit_privacy(config).warnings)


class TestGuardrails:
    def test_enabled_guardrails_reported_as_info(self):
        config = StackConfig()
        config.gateway.guardrails = GuardrailsConfig(enabled=True, pii_detection=True)
        assert any(f.category == "guardrails" for f in audit_privacy(config).findings)
