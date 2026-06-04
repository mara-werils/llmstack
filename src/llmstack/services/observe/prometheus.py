"""Prometheus + Grafana observability services."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import yaml

from llmstack.config.schema import ObserveConfig
from llmstack.services.base import ServiceBase

CONFIG_DIR = Path.home() / ".llmstack" / "config"

# Prometheus config that scrapes the gateway /metrics endpoint
PROMETHEUS_CONFIG = {
    "global": {
        "scrape_interval": "15s",
        "evaluation_interval": "15s",
    },
    "scrape_configs": [
        {
            "job_name": "llmstack-gateway",
            "metrics_path": "/metrics",
            "static_configs": [{"targets": ["llmstack-gateway:8000"]}],
            "scrape_interval": "5s",
        },
        {
            "job_name": "qdrant",
            "metrics_path": "/metrics",
            "static_configs": [{"targets": ["llmstack-qdrant:6333"]}],
            "scrape_interval": "15s",
        },
    ],
}

# Grafana datasource provisioning
GRAFANA_DATASOURCE = {
    "apiVersion": 1,
    "datasources": [
        {
            "name": "Prometheus",
            "type": "prometheus",
            "access": "proxy",
            "url": "http://llmstack-prometheus:9090",
            "isDefault": True,
        }
    ],
}

# Grafana dashboard provisioning config
GRAFANA_DASHBOARD_PROVIDER = {
    "apiVersion": 1,
    "providers": [
        {
            "name": "LLMStack",
            "type": "file",
            "options": {"path": "/opt/grafana/dashboards"},
        }
    ],
}

# Pre-built Grafana dashboard JSON
GRAFANA_DASHBOARD = {
    "dashboard": {
        "title": "LLMStack Overview",
        "uid": "llmstack-overview",
        "timezone": "browser",
        "refresh": "10s",
        "panels": [
            {
                "title": "Request Rate",
                "type": "timeseries",
                "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
                "targets": [
                    {"expr": "rate(llmstack_requests_total[1m])", "legendFormat": "{{path}}"}
                ],
            },
            {
                "title": "Latency (p50 / p99)",
                "type": "timeseries",
                "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0},
                "targets": [
                    {
                        "expr": "histogram_quantile(0.5, rate(llmstack_request_duration_seconds_bucket[5m]))",
                        "legendFormat": "p50",
                    },
                    {
                        "expr": "histogram_quantile(0.99, rate(llmstack_request_duration_seconds_bucket[5m]))",
                        "legendFormat": "p99",
                    },
                ],
            },
            {
                "title": "Error Rate",
                "type": "stat",
                "gridPos": {"h": 4, "w": 6, "x": 0, "y": 8},
                "targets": [{"expr": "sum(rate(llmstack_errors_total[5m]))"}],
            },
            {
                "title": "Active Services",
                "type": "stat",
                "gridPos": {"h": 4, "w": 6, "x": 6, "y": 8},
                "targets": [{"expr": "up"}],
            },
            {
                "title": "Token Throughput",
                "type": "timeseries",
                "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8},
                "targets": [
                    {"expr": "rate(llmstack_tokens_total[1m])", "legendFormat": "{{type}}"}
                ],
            },
        ],
    },
}


def _write_file(path: Path, content: str) -> None:
    """Write content to file, creating parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


class PrometheusService(ServiceBase):
    name = "prometheus"
    category = "observe"

    def __init__(self, config: ObserveConfig):
        self.config = config
        self.host_port = 9090

    def _prepare_config(self) -> str:
        """Write prometheus.yml to ~/.llmstack/config/prometheus/ and return the dir path."""
        config_dir = CONFIG_DIR / "prometheus"
        _write_file(
            config_dir / "prometheus.yml", yaml.dump(PROMETHEUS_CONFIG, default_flow_style=False)
        )
        return str(config_dir)

    def container_spec(self) -> dict[str, Any]:
        config_dir = self._prepare_config()
        return {
            "image": "prom/prometheus:latest",
            "name": "llmstack-prometheus",
            "ports": {"9090/tcp": self.host_port},
            "command": [
                "--config.file=/etc/prometheus/prometheus.yml",
                f"--storage.tsdb.retention.time={self.config.retention}",
                "--web.enable-lifecycle",
            ],
            "volumes": {
                config_dir: {"bind": "/etc/prometheus", "mode": "ro"},
                "llmstack_prometheus_data": {"bind": "/prometheus", "mode": "rw"},
            },
            "environment": {},
        }

    def health_url(self) -> str:
        return f"http://localhost:{self.host_port}/-/healthy"

    def get_config_yaml(self) -> str:
        """Return the prometheus.yml content."""
        return yaml.dump(PROMETHEUS_CONFIG, default_flow_style=False)


class GrafanaService(ServiceBase):
    name = "grafana"
    category = "observe"

    def __init__(self, config: ObserveConfig):
        self.config = config
        self.host_port = config.dashboard_port

    def _prepare_provisioning(self) -> str:
        """Write Grafana provisioning files to ~/.llmstack/config/grafana/ and return the dir."""
        base = CONFIG_DIR / "grafana"

        # Datasource provisioning
        _write_file(
            base / "provisioning" / "datasources" / "datasource.yml",
            yaml.dump(GRAFANA_DATASOURCE, default_flow_style=False),
        )

        # Dashboard provider provisioning
        _write_file(
            base / "provisioning" / "dashboards" / "provider.yml",
            yaml.dump(GRAFANA_DASHBOARD_PROVIDER, default_flow_style=False),
        )

        # Dashboard JSON
        _write_file(
            base / "dashboards" / "llmstack.json",
            json.dumps(GRAFANA_DASHBOARD, indent=2),
        )

        return str(base)

    def container_spec(self) -> dict[str, Any]:
        base = self._prepare_provisioning()
        return {
            "image": "grafana/grafana:latest",
            "name": "llmstack-grafana",
            "ports": {"3000/tcp": self.host_port},
            "environment": {
                "GF_SECURITY_ADMIN_USER": "admin",
                "GF_SECURITY_ADMIN_PASSWORD": "llmstack",
                "GF_AUTH_ANONYMOUS_ENABLED": "true",
                "GF_AUTH_ANONYMOUS_ORG_ROLE": "Viewer",
                "GF_DASHBOARDS_DEFAULT_HOME_DASHBOARD_PATH": "/opt/grafana/dashboards/llmstack.json",
            },
            "volumes": {
                os.path.join(base, "provisioning", "datasources"): {
                    "bind": "/etc/grafana/provisioning/datasources",
                    "mode": "ro",
                },
                os.path.join(base, "provisioning", "dashboards"): {
                    "bind": "/etc/grafana/provisioning/dashboards",
                    "mode": "ro",
                },
                os.path.join(base, "dashboards"): {
                    "bind": "/opt/grafana/dashboards",
                    "mode": "ro",
                },
            },
        }

    def health_url(self) -> str:
        return f"http://localhost:{self.host_port}/api/health"

    def get_datasource_json(self) -> str:
        return json.dumps(GRAFANA_DATASOURCE, indent=2)

    def get_dashboard_provider_json(self) -> str:
        return json.dumps(GRAFANA_DASHBOARD_PROVIDER, indent=2)

    def get_dashboard_json(self) -> str:
        return json.dumps(GRAFANA_DASHBOARD, indent=2)
