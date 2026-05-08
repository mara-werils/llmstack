# Contributing

Thanks for your interest in contributing to llmstack. Here is how to get started.

## Development Setup

```bash
git clone https://github.com/mara-werils/llmstack.git
cd llmstack
pip install -e ".[dev]"
```

The `dev` extra installs testing and linting dependencies: pytest, pytest-asyncio, ruff, FastAPI, Starlette, Redis, and fakeredis.

## Running Tests

```bash
make test        # Run all tests
make lint        # Check code style
make format      # Auto-format code
```

## Project Structure

```
src/llmstack/
├── cli/          # Typer CLI commands
├── config/       # Pydantic config schema + presets
├── core/         # Stack orchestrator, hardware detection, resolver
├── services/     # Service implementations (Ollama, vLLM, Qdrant, Redis, etc.)
├── gateway/      # FastAPI gateway (OpenAI-compatible proxy)
├── docker/       # Docker SDK wrapper
└── plugins/      # Plugin interface + loader
```

## Adding a New Service

1. Create a class extending `ServiceBase` in `services/`
2. Implement `container_spec()` and `health_url()`
3. Optionally implement `post_start()` and `openai_base_url()`
4. Register in `services/registry.py` or as a plugin via `entry_points`
5. Add tests in `tests/unit/`

## Creating a Plugin

See the [Plugins guide](guide/plugins.md) for detailed instructions, or refer to `plugins/spec.py` for the interface.

```toml
# In your plugin's pyproject.toml:
[project.entry-points."llmstack.services"]
my_service = "my_package:MyServiceClass"
```

## Pull Requests

- Keep PRs focused on a single change
- Add tests for new functionality
- Run `make lint` before submitting
- Use conventional commit messages (`feat:`, `fix:`, `docs:`)

## Code Style

- Python 3.11+
- Ruff for linting and formatting
- Type hints everywhere
- Docstrings for public APIs
