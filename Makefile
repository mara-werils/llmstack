.PHONY: install dev test test-cov lint format type-check security \
       docker-build docker-up docker-down docs clean \
       release-patch release-minor release-major help

# ── Development ──────────────────────────────────────────────────────

install: ## Install the package
	pip install -e .

dev: ## Install with dev dependencies
	pip install -e ".[dev]"
	pip install pytest-cov mypy pre-commit
	pre-commit install

# ── Quality ──────────────────────────────────────────────────────────

test: ## Run tests
	pytest tests/ -v --tb=short --timeout=30

test-cov: ## Run tests with coverage
	pytest tests/ -v --tb=short --timeout=30 --cov=llmstack --cov-report=term-missing --cov-report=html

lint: ## Lint code with ruff
	ruff check src/ tests/

format: ## Format code with ruff
	ruff format src/ tests/

type-check: ## Run type checking with mypy
	mypy src/llmstack --ignore-missing-imports

security: ## Run security checks with bandit
	bandit -r src/llmstack -s B101

# ── Docker ───────────────────────────────────────────────────────────

docker-build: ## Build Docker image
	docker build -t llmstack:latest .

docker-up: ## Start services with docker compose
	docker compose up -d

docker-down: ## Stop services with docker compose
	docker compose down

# ── Documentation ────────────────────────────────────────────────────

docs: ## Build documentation
	mkdocs build --strict

docs-serve: ## Serve documentation locally
	mkdocs serve

# ── Cleanup ──────────────────────────────────────────────────────────

clean: ## Remove build artifacts
	rm -rf build/ dist/ *.egg-info .pytest_cache .ruff_cache .mypy_cache htmlcov/ coverage.xml
	find . -type d -name __pycache__ -exec rm -rf {} +

# ── Releases ─────────────────────────────────────────────────────────

release-patch: ## Release a patch version (x.y.Z)
	@VERSION=$$(python -c "import re; f=open('pyproject.toml').read(); v=re.search(r'version=\"(.+?)\"',f).group(1); parts=v.split('.'); parts[2]=str(int(parts[2])+1); print('.'.join(parts))") && \
	sed -i.bak "s/version=\"[^\"]*\"/version=\"$$VERSION\"/" pyproject.toml && rm -f pyproject.toml.bak && \
	git add pyproject.toml && git commit -m "release: v$$VERSION" && \
	git tag "v$$VERSION" && \
	echo "Tagged v$$VERSION. Run 'git push && git push --tags' to publish."

release-minor: ## Release a minor version (x.Y.0)
	@VERSION=$$(python -c "import re; f=open('pyproject.toml').read(); v=re.search(r'version=\"(.+?)\"',f).group(1); parts=v.split('.'); parts[1]=str(int(parts[1])+1); parts[2]='0'; print('.'.join(parts))") && \
	sed -i.bak "s/version=\"[^\"]*\"/version=\"$$VERSION\"/" pyproject.toml && rm -f pyproject.toml.bak && \
	git add pyproject.toml && git commit -m "release: v$$VERSION" && \
	git tag "v$$VERSION" && \
	echo "Tagged v$$VERSION. Run 'git push && git push --tags' to publish."

release-major: ## Release a major version (X.0.0)
	@VERSION=$$(python -c "import re; f=open('pyproject.toml').read(); v=re.search(r'version=\"(.+?)\"',f).group(1); parts=v.split('.'); parts[0]=str(int(parts[0])+1); parts[1]='0'; parts[2]='0'; print('.'.join(parts))") && \
	sed -i.bak "s/version=\"[^\"]*\"/version=\"$$VERSION\"/" pyproject.toml && rm -f pyproject.toml.bak && \
	git add pyproject.toml && git commit -m "release: v$$VERSION" && \
	git tag "v$$VERSION" && \
	echo "Tagged v$$VERSION. Run 'git push && git push --tags' to publish."

# ── Help ─────────────────────────────────────────────────────────────

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
