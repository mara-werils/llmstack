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

# Install llmstack
echo -e "\nInstalling llmstack..."
pip install --quiet llmstack-cli 2>/dev/null || pip3 install --quiet llmstack-cli
info "llmstack installed"

# Check Ollama
if command -v ollama &>/dev/null; then
    info "Ollama found"
else
    warn "Ollama not found — install from https://ollama.com"
fi

# Run quickstart
echo -e "\nRunning quickstart..."
llmstack quickstart --skip-pull 2>/dev/null || true

echo -e "\n${BOLD}Done!${NC} Get started:"
echo "  llmstack quickstart    # setup model + config"
echo "  llmstack chat          # start chatting"
echo "  llmstack doctor        # check system health"
echo ""
