#!/usr/bin/env bash
set -euo pipefail

BOLD='\033[1m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m'

info()  { echo -e "${GREEN}✓${NC} $1"; }
warn()  { echo -e "${YELLOW}!${NC} $1"; }
error() { echo -e "${RED}✗${NC} $1"; }

echo -e "\n${BOLD}LLMStack Installer${NC}\n"

# Check Python
if ! command -v python3 &>/dev/null; then
    error "Python 3 is required but not installed."
    echo "  Install: https://www.python.org/downloads/"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)
if [ "$MAJOR" -lt 3 ] || ([ "$MAJOR" -eq 3 ] && [ "$MINOR" -lt 11 ]); then
    error "Python 3.11+ required (found $PYTHON_VERSION)"
    exit 1
fi
info "Python $PYTHON_VERSION"

# Install llmstack with the best available tool. Isolated installs (uv/pipx) are
# preferred — a bare `pip install` fails on PEP 668 "externally-managed" Pythons
# (modern Homebrew / Debian / Ubuntu).
echo -e "\nInstalling llmstack..."
if command -v uv &>/dev/null; then
    info "Using uv (isolated tool install)"
    uv tool install --upgrade llmstack-cli
elif command -v pipx &>/dev/null; then
    info "Using pipx (isolated install)"
    pipx install --force llmstack-cli
else
    warn "uv/pipx not found — installing with 'pip install --user'."
    warn "For an isolated install, see https://docs.astral.sh/uv/"
    python3 -m pip install --user --upgrade llmstack-cli
fi
info "llmstack installed"

# Check Ollama
if command -v ollama &>/dev/null; then
    info "Ollama found"
else
    warn "Ollama not found — install from https://ollama.com"
fi

# Run quickstart only if llmstack landed on PATH
if command -v llmstack &>/dev/null; then
    echo -e "\nRunning quickstart..."
    llmstack quickstart --skip-pull 2>/dev/null || true
else
    warn "llmstack is installed but not on your PATH yet."
    warn "Add your tool bin dir (e.g. ~/.local/bin) to PATH and re-open your shell."
fi

echo -e "\n${BOLD}Done!${NC} Get started:"
echo "  llmstack quickstart    # zero-key local AI, proves it works (~30s)"
echo "  llmstack chat          # start chatting"
echo "  llmstack doctor        # check first-run readiness"
echo ""
echo "  Editor (VS Code / Cursor / VSCodium):"
echo "    code --install-extension llmstack.llmstack-vscode"
echo ""
