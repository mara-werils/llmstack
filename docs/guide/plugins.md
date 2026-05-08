# Plugins

llmstack supports a plugin system for adding new service backends. Plugins are distributed as Python packages and discovered via `entry_points`.

## Installing Plugins

Plugins are installed with pip:

```bash
pip install llmstack-cli-plugin-chromadb
```

Once installed, the new provider is available in `llmstack.yaml`:

```yaml
services:
  vectors:
    provider: chromadb    # Now available after installing the plugin
```

No code changes to llmstack are needed. The plugin is discovered automatically at runtime.

## How Plugins Work

llmstack uses Python's `entry_points` mechanism (defined in `pyproject.toml`) to discover plugins. At startup, the plugin loader scans the `llmstack.services` entry point group and registers any service classes it finds.

The lookup chain is:

1. Check built-in services (Ollama, vLLM, Qdrant, Redis, TEI, Gateway, Prometheus, Grafana)
2. Check installed plugins via `entry_points`
3. Raise an error if no provider matches

## Creating a Plugin

To create a plugin, you need to:

1. Create a Python package
2. Implement the `ServiceBase` interface
3. Register it via entry points

### Step 1: Implement `ServiceBase`

Every service in llmstack extends `ServiceBase`. Your plugin must implement the required methods:

```python
from llmstack.services.base import ServiceBase


class ChromaDBService(ServiceBase):
    name = "chromadb"
    category = "vectordb"

    def __init__(self, config):
        self.config = config

    def container_spec(self) -> dict:
        """Return the Docker container specification."""
        return {
            "image": "chromadb/chroma:latest",
            "name": "llmstack-chromadb",
            "ports": {
                "8000/tcp": self.config.port,
            },
            "volumes": {
                "chromadb_data": {"bind": "/chroma/chroma", "mode": "rw"},
            },
            "environment": {},
        }

    def health_url(self) -> str:
        """Return the URL used to check if the service is healthy."""
        return f"http://localhost:{self.config.port}/api/v1/heartbeat"

    async def post_start(self) -> None:
        """Run after the container is healthy. Optional."""
        pass

    def openai_base_url(self) -> str | None:
        """Return an OpenAI-compatible base URL, if applicable."""
        return None  # ChromaDB is not an OpenAI-compatible service
```

### Step 2: Register via Entry Points

In your plugin's `pyproject.toml`:

```toml
[project]
name = "llmstack-cli-plugin-chromadb"
version = "0.1.0"
dependencies = ["llmstack-cli"]

[project.entry-points."llmstack.services"]
chromadb = "llmstack_plugin_chromadb:ChromaDBService"
```

The key (`chromadb`) is the provider name used in `llmstack.yaml`. The value is the import path to your service class.

### Step 3: Package and Publish

```bash
pip install build twine
python -m build
twine upload dist/*
```

Users can then install your plugin:

```bash
pip install llmstack-cli-plugin-chromadb
```

## ServiceBase Interface Reference

| Method | Required | Description |
|---|---|---|
| `container_spec()` | Yes | Returns a dict with Docker container configuration (image, ports, volumes, environment, etc.) |
| `health_url()` | Yes | Returns an HTTP URL that llmstack polls to determine when the service is ready |
| `post_start()` | No | Async method called after the health check passes. Use for initialization tasks like pulling models. |
| `openai_base_url()` | No | Returns an OpenAI-compatible base URL if the service provides one. Used by the gateway to route requests. |

### Class Attributes

| Attribute | Type | Description |
|---|---|---|
| `name` | string | Unique service name (e.g., `"chromadb"`) |
| `category` | string | Service category: `"inference"`, `"embeddings"`, `"vectordb"`, `"cache"`, `"gateway"`, or `"observe"` |

## Plugin Guidelines

- **Name your package** with the `llmstack-cli-plugin-` prefix for discoverability
- **Pin your llmstack dependency** to a compatible version range
- **Include health checks** -- your `health_url()` must return a URL that responds with HTTP 200 when the service is ready
- **Use Docker volumes** for persistent data to survive container restarts
- **Add tests** -- test your `container_spec()` output and health check logic
- **Document the configuration** -- tell users what YAML fields your plugin expects
