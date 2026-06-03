# Contributing to LLMStack

Thanks for your interest in contributing! This guide covers everything you
need to get started.

## Development Setup

### Prerequisites

- Python 3.11 or newer
- [Ollama](https://ollama.com/download) (for running local models)
- Docker (optional, for full-stack testing)
- Git

### Clone and Install

```bash
git clone https://github.com/mara-werils/llmstack.git
cd llmstack

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows

# Install in editable mode with dev dependencies
pip install -e ".[dev]"
```

### Verify the installation

```bash
llmstack --version
llmstack doctor
```

## Running Tests

```bash
# Run the full test suite
pytest

# Run with verbose output
pytest -v

# Run a specific test file
pytest tests/unit/test_schema.py

# Run only fast tests (skip slow/integration)
pytest -m "not slow"

# Check code style
ruff check .

# Auto-format code
ruff format .
```

### Writing Tests

- Place unit tests in `tests/unit/` and integration tests in `tests/integration/`.
- Use `pytest-asyncio` for async tests (the project uses `asyncio_mode = "auto"`).
- Name test files `test_<module>.py` and test functions `test_<behaviour>`.
- Use `fakeredis` instead of a real Redis server in unit tests.

## Project Structure

```
src/llmstack/
├── cli/          # Typer CLI commands
│   ├── app.py    # Main Typer app and command registration
│   └── commands/ # One file per CLI command
├── config/       # Pydantic config schema + presets + loader
├── core/         # Stack orchestrator, hardware detection, resolver
├── gateway/      # FastAPI gateway (OpenAI-compatible proxy)
│   ├── main.py   # App factory
│   ├── routes/   # One router per API group
│   └── middleware/  # Auth, logging, rate limiting, correlation ID
├── docker/       # Docker SDK wrapper
├── finetune/     # Fine-tuning pipeline (LoRA / QLoRA)
├── observe/      # Observability: traces, scoring, alerts
├── agent/        # Tool-using AI agent
├── mcp/          # MCP server for AI client integration
├── sdk/          # Python SDK (Client / AsyncClient)
└── plugins/      # Plugin interface + loader
```

## Code Style Guide

- **Python 3.11+** -- use modern syntax (`X | Y` unions, etc.).
- **Ruff** for linting and formatting. Config is in `pyproject.toml`.
- **Type hints** on all function signatures.
- **Docstrings** for every public function, class, and module (Google style).
- **Line length** 100 characters (`tool.ruff.line-length`).
- Prefer `from __future__ import annotations` at the top of every module.
- Use `Field(default_factory=...)` for mutable Pydantic defaults.

## Adding a New CLI Command

1. Create `src/llmstack/cli/commands/<name>.py` with a top-level function.
2. Register it in `src/llmstack/cli/app.py` as a `@app.command()`.
3. Add tests in `tests/unit/test_<name>.py`.

## Adding a New Service

1. Create a class extending `ServiceBase` in `src/llmstack/services/`.
2. Implement `container_spec()`, `health_url()`.
3. Optionally implement `post_start()`, `openai_base_url()`.
4. Register in `services/registry.py` or as a plugin via `entry_points`.
5. Add tests in `tests/unit/`.

## Creating a Plugin

See [plugins/spec.py](src/llmstack/plugins/spec.py) for the plugin interface.

```toml
# In your plugin's pyproject.toml:
[project.entry-points."llmstack.services"]
my_service = "my_package:MyServiceClass"
```

## Pull Request Process

1. **Branch** -- create a feature branch from `main` (e.g. `feat/my-feature`).
2. **Commits** -- use [Conventional Commits](https://www.conventionalcommits.org/)
   (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `build:`, `ci:`).
3. **Tests** -- add or update tests. All CI checks must pass.
4. **Lint** -- run `ruff check . && ruff format --check .` before pushing.
5. **PR description** -- explain *what* and *why*. Reference any related issues.
6. **Review** -- a maintainer will review and may request changes.

### PR Size Guidelines

- Keep PRs focused on a single concern.
- Aim for < 400 lines changed per PR.
- If a change is large, break it into a stack of smaller PRs.

## Reporting Bugs

Open an issue with:

- LLMStack version (`llmstack --version`)
- OS and Python version
- Steps to reproduce
- Expected vs actual behaviour
- Output of `llmstack doctor`
