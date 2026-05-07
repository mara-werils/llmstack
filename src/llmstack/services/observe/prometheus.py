"""Prometheus + Grafana observability services."""

from __future__ import annotations

import json
from typing import Any

from llmstack.config.schema import ObserveConfig
from llmstack.services.base import ServiceBase


# Prometheus config that scrapes the gateway /metrics endpoint
PROMETHEUS_CONFIG = """
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'llmstack-gateway'
    metrics_path: '/metrics'
    static_configs:
      - targets: ['llmstack-gateway:8000']
    scrape_interval: 5s

  - job_name: 'qdrant'
    metrics_path: '/metrics'
    static_configs:
      - targets: ['llmstack-qdrant:6333']
    scrape_interval: 15s
"""

# Grafana datasource provisioning
GRAFANA_DATASOURCE = {
    "apiVersion": 1,
    "datasources": [{
        "name": "Prometheus",
        "type": "prometheus",
        "access": "proxy",
        "url": "http://llmstack-prometheus:9090",
        "isDefault": True,
    }],
}

# Grafana dashboard provisioning config
GRAFANA_DASHBOARD_PROVIDER = {
    "apiVersion": 1,
    "providers": [{
        "name": "LLMStack",
        "type": "file",
        "options": {"path": "/var/lib/grafana/dashboards"},
    }],
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
                "targets": [{"expr": "rate(llmstack_requests_total[1m])", "legendFormat": "{{path}}"}],
            },
            {
                "title": "Latency (p50 / p99)",
                "type": "timeseries",
                "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0},
                "targets": [
                    {"expr": "histogram_quantile(0.5, rate(llmstack_request_duration_seconds_bucket[5m]))", "legendFormat": "p50"},
                    {"expr": "histogram_quantile(0.99, rate(llmstack_request_duration_seconds_bucket[5m]))", "legendFormat": "p99"},
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
                "targets": [{"expr": "rate(llmstack_tokens_total[1m])", "legendFormat": "{{type}}"}],
            },
        ],
    },
}


class PrometheusService(ServiceBase):
    name = "prometheus"
    category = "observe"

    def __init__(self, config: ObserveConfig):
        self.config = config
        self.host_port = 9090

    def container_spec(self) -> dict[str, Any]:
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
                "llmstack_prometheus_config": {"bind": "/etc/prometheus", "mode": "rw"},
                "llmstack_prometheus_data": {"bind": "/prometheus", "mode": "rw"},
            },
            "environment": {},
        }

    def health_url(self) -> str:
        return f"http://localhost:{self.host_port}/-/healthy"

    def get_config_yaml(self) -> str:
        """Return the prometheus.yml content."""
        return PROMETHEUS_CONFIG


class GrafanaService(ServiceBase):
    name = "grafana"
    category = "observe"

    def __init__(self, config: ObserveConfig):
        self.config = config
        self.host_port = config.dashboard_port

    def container_spec(self) -> dict[str, Any]:
        return {
            "image": "grafana/grafana:latest",
            "name": "llmstack-grafana",
            "ports": {"3000/tcp": self.host_port},
            "environment": {
                "GF_SECURITY_ADMIN_USER": "admin",
                "GF_SECURITY_ADMIN_PASSWORD": "llmstack",
                "GF_AUTH_ANONYMOUS_ENABLED": "true",
                "GF_AUTH_ANONYMOUS_ORG_ROLE": "Viewer",
                "GF_DASHBOARDS_DEFAULT_HOME_DASHBOARD_PATH": "/var/lib/grafana/dashboards/llmstack.json",
            },
            "volumes": {
                "llmstack_grafana_data": {"bind": "/var/lib/grafana", "mode": "rw"},
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
