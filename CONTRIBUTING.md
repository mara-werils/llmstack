# Contributing to llmstack

Thanks for your interest in contributing! Here's how to get started.

## Development setup

```bash
git clone https://github.com/mara-werils/llmstack.git
cd llmstack
pip install -e ".[dev]"
```

## Running tests

```bash
make test        # run all tests
make lint        # check code style
make format      # auto-format code
```

## Project structure

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

## Adding a new service

1. Create a class extending `ServiceBase` in `services/`
2. Implement `container_spec()`, `health_url()`
3. Optionally implement `post_start()`, `openai_base_url()`
4. Register in `services/registry.py` or as a plugin via `entry_points`
5. Add tests in `tests/unit/`

## Creating a plugin

See [plugins/spec.py](src/llmstack/plugins/spec.py) for the plugin interface.

```toml
# In your plugin's pyproject.toml:
[project.entry-points."llmstack.services"]
my_service = "my_package:MyServiceClass"
```

## Pull requests

- Keep PRs focused on a single change
- Add tests for new functionality
- Run `make lint` before submitting
- Use conventional commit messages (`feat:`, `fix:`, `docs:`)

## Code style

- Python 3.11+
- Ruff for linting and formatting
- Type hints everywhere
- Docstrings for public APIs
