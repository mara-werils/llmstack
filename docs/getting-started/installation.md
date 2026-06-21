# Installation

## Prerequisites

Before installing llmstack, make sure you have:

- **Python 3.11 or later** -- check with `python --version`
- **Docker** -- the Docker daemon must be running (`docker info` should succeed)

### GPU Support (optional)

For GPU-accelerated inference:

| Hardware | Additional Requirement |
|---|---|
| NVIDIA GPU | [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) |
| Apple Silicon (M1--M4) | No additional setup needed (Metal acceleration via Ollama) |
| CPU only | No additional setup needed |

## Quick install (one-liner)

The fastest path — auto-picks `uv`, `pipx`, or `pip`:

```bash
curl -LsSf https://raw.githubusercontent.com/mara-werils/llmstack/main/install.sh | sh
```

Prefer an isolated install? Use `uv tool install llmstack-cli` or
`pipx install llmstack-cli`.

## Install from PyPI

```bash
pip install llmstack-cli
```

This installs the `llmstack` CLI and its core dependencies:

- [Typer](https://typer.tiangolo.com/) -- CLI framework
- [Rich](https://rich.readthedocs.io/) -- terminal formatting
- [Pydantic v2](https://docs.pydantic.dev/) -- config validation
- [Docker SDK for Python](https://docker-py.readthedocs.io/) -- container orchestration
- [httpx](https://www.python-httpx.org/) -- HTTP client for health checks
- [psutil](https://psutil.readthedocs.io/) -- hardware detection

### Install with Gateway Dependencies

If you plan to develop or run the gateway outside of Docker:

```bash
pip install "llmstack-cli[gateway]"
```

This adds FastAPI, Uvicorn, Starlette, and Redis client libraries.

### Install with Documentation Dependencies

```bash
pip install "llmstack-cli[docs]"
```

This adds MkDocs Material and the minify plugin for building the documentation site.

## Install from Source

```bash
git clone https://github.com/mara-werils/llmstack.git
cd llmstack
pip install -e ".[dev]"
```

The `dev` extra includes testing and linting tools (pytest, ruff) in addition to gateway dependencies.

## Docker Setup

llmstack uses Docker to manage all services. It does not require `docker-compose` -- it uses the Docker SDK for Python directly.

### Verify Docker Is Working

```bash
docker info
```

If you see an error like "Cannot connect to the Docker daemon", start Docker Desktop or the Docker service:

=== "macOS"

    Open Docker Desktop from your Applications folder.

=== "Linux"

    ```bash
    sudo systemctl start docker
    ```

=== "Windows (WSL2)"

    Open Docker Desktop. Make sure WSL2 integration is enabled in Settings.

### Verify GPU Passthrough (NVIDIA only)

```bash
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
```

If this prints your GPU info, you are ready for GPU-accelerated inference. If it fails, install the [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

## Verify Installation

Run the built-in diagnostic tool:

```bash
llmstack doctor
```

This checks:

- Docker is installed and the daemon is reachable
- GPU detection (NVIDIA, Apple Silicon, or CPU)
- Available RAM and CPU cores
- Required ports are free (11434, 6333, 6379, 8000)
- `llmstack.yaml` is valid (if present)

Example output:

```
LLMStack Doctor

  PASS Docker is installed
  PASS Docker daemon is running
  PASS GPU detected: Apple M2 Pro
  INFO RAM: 32 GB, CPU: 10 cores
  PASS Port 11434 (Ollama) is available
  PASS Port 6333 (Qdrant) is available
  PASS Port 6379 (Redis) is available
  PASS Port 8000 (Gateway) is available
  WARN No llmstack.yaml found (run 'llmstack init')

All checks passed!
```

## Install the Editor Extension

Want AI in your editor? Install **LLMStack** from the VS Code Marketplace, or from
the [Open VSX Registry](https://open-vsx.org) for Cursor / Windsurf / VSCodium.
See the [editor extension guide](../guide/editor.md).

## Next Steps

Once installed, head to the [Quickstart](quickstart.md) guide to launch your first stack.
