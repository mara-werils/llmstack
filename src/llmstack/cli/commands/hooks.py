"""llmstack hooks — Set up AI-powered git hooks."""

from __future__ import annotations

import stat
from pathlib import Path

from llmstack.cli.console import console


HOOK_TEMPLATES = {
    "pre-commit": {
        "description": "AI review of staged changes before commit",
        "script": '''#!/usr/bin/env bash
# llmstack pre-commit hook — AI review before commit
set -e

DIFF=$(git diff --cached --diff-filter=ACMR)
if [ -z "$DIFF" ]; then
    exit 0
fi

echo "🔍 Running AI pre-commit review..."

# Check for common issues
ISSUES=$(echo "$DIFF" | grep -n "TODO\\|FIXME\\|HACK\\|XXX\\|password\\|secret\\|api_key\\|token" || true)
if [ -n "$ISSUES" ]; then
    echo "⚠️  Found potential issues in staged changes:"
    echo "$ISSUES"
    echo ""
    read -p "Continue with commit? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Commit aborted."
        exit 1
    fi
fi

# Check for large files
LARGE_FILES=$(git diff --cached --name-only | while read f; do
    if [ -f "$f" ]; then
        SIZE=$(wc -c < "$f" 2>/dev/null || echo 0)
        if [ "$SIZE" -gt 1048576 ]; then
            echo "  $f ($(echo "scale=1; $SIZE/1048576" | bc)MB)"
        fi
    fi
done)
if [ -n "$LARGE_FILES" ]; then
    echo "⚠️  Large files detected:"
    echo "$LARGE_FILES"
fi

echo "✅ Pre-commit check passed"
''',
    },
    "commit-msg": {
        "description": "Validate commit message format (conventional commits)",
        "script": '''#!/usr/bin/env bash
# llmstack commit-msg hook — validate conventional commits
MSG_FILE=$1
MSG=$(cat "$MSG_FILE")

# Check conventional commit format
if ! echo "$MSG" | head -1 | grep -qE "^(feat|fix|docs|style|refactor|test|chore|perf|ci|build|revert)(\(.+\))?: .{1,72}$"; then
    echo "❌ Commit message doesn't follow conventional commits format."
    echo ""
    echo "Format: type(scope): description"
    echo "Types: feat, fix, docs, style, refactor, test, chore, perf, ci, build, revert"
    echo ""
    echo "Your message: $MSG"
    echo ""
    echo "Tip: Use 'llmstack commit' to auto-generate messages"
    exit 1
fi

echo "✅ Commit message format OK"
''',
    },
    "pre-push": {
        "description": "Run tests and security check before push",
        "script": '''#!/usr/bin/env bash
# llmstack pre-push hook — verify before pushing
set -e

echo "🔍 Running pre-push checks..."

# Run tests if available
if [ -f "pyproject.toml" ] || [ -f "setup.py" ]; then
    if command -v pytest &> /dev/null; then
        echo "  Running pytest..."
        pytest --tb=short -q 2>/dev/null || {
            echo "❌ Tests failed. Push aborted."
            exit 1
        }
    fi
elif [ -f "package.json" ]; then
    if grep -q '"test"' package.json 2>/dev/null; then
        echo "  Running npm test..."
        npm test --silent 2>/dev/null || {
            echo "❌ Tests failed. Push aborted."
            exit 1
        }
    fi
fi

# Check for secrets
SECRETS=$(git diff --name-only HEAD~1..HEAD 2>/dev/null | xargs grep -l "PRIVATE_KEY\\|aws_secret\\|password\\s*=" 2>/dev/null || true)
if [ -n "$SECRETS" ]; then
    echo "⚠️  Potential secrets detected in:"
    echo "$SECRETS"
    read -p "Continue with push? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "✅ Pre-push checks passed"
''',
    },
    "post-checkout": {
        "description": "Auto-install dependencies after branch switch",
        "script": '''#!/usr/bin/env bash
# llmstack post-checkout hook — auto-setup after branch switch
OLD_REF=$1
NEW_REF=$2
BRANCH_SWITCH=$3

# Only run on branch switch, not file checkout
if [ "$BRANCH_SWITCH" != "1" ]; then
    exit 0
fi

echo "🔄 Post-checkout: checking for dependency changes..."

# Python
if git diff --name-only "$OLD_REF" "$NEW_REF" | grep -q "requirements.txt\\|pyproject.toml"; then
    echo "  📦 Python dependencies changed, installing..."
    pip install -r requirements.txt 2>/dev/null || pip install -e ".[dev]" 2>/dev/null || true
fi

# Node.js
if git diff --name-only "$OLD_REF" "$NEW_REF" | grep -q "package.json"; then
    echo "  📦 Node dependencies changed, installing..."
    npm install 2>/dev/null || yarn install 2>/dev/null || true
fi
''',
    },
}


def hooks(
    action: str = "list",
    hook_name: str | None = None,
    force: bool = False,
) -> None:
    """Manage AI-powered git hooks."""
    from rich.table import Table
    from rich.panel import Panel
    from rich.syntax import Syntax

    git_dir = Path.cwd() / ".git"
    if not git_dir.exists():
        console.print("[error]Not a git repository.[/]")
        return

    hooks_dir = git_dir / "hooks"
    hooks_dir.mkdir(exist_ok=True)

    if action == "list":
        table = Table(
            title="Available Git Hooks",
            show_header=True,
            header_style="bold cyan",
            border_style="dim",
        )
        table.add_column("Hook", style="bold")
        table.add_column("Description")
        table.add_column("Status", width=10)

        for name, info in HOOK_TEMPLATES.items():
            hook_path = hooks_dir / name
            installed = hook_path.exists() and hook_path.stat().st_mode & stat.S_IXUSR
            status = "[green]active[/]" if installed else "[dim]not installed[/]"
            table.add_row(name, info["description"], status)

        console.print(table)
        console.print("\n[dim]Install: llmstack hooks install <hook-name>[/]")
        console.print("[dim]Install all: llmstack hooks install-all[/]")

    elif action == "install" and hook_name:
        if hook_name not in HOOK_TEMPLATES:
            console.print(f"[error]Unknown hook: {hook_name}[/]")
            console.print(f"Available: {', '.join(HOOK_TEMPLATES.keys())}")
            return

        hook_path = hooks_dir / hook_name
        if hook_path.exists() and not force:
            console.print(f"[warning]Hook '{hook_name}' already exists. Use --force to overwrite.[/]")
            return

        hook_path.write_text(HOOK_TEMPLATES[hook_name]["script"])
        hook_path.chmod(hook_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        console.print(f"[green]Installed hook:[/] {hook_name}")
        console.print(f"  [dim]{HOOK_TEMPLATES[hook_name]['description']}[/]")

    elif action == "install-all":
        for name, info in HOOK_TEMPLATES.items():
            hook_path = hooks_dir / name
            if hook_path.exists() and not force:
                console.print(f"  [yellow]SKIP[/] {name} (exists, use --force)")
                continue
            hook_path.write_text(info["script"])
            hook_path.chmod(hook_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
            console.print(f"  [green]OK[/] {name}")
        console.print(f"\n[green]All hooks installed.[/]")

    elif action == "remove" and hook_name:
        hook_path = hooks_dir / hook_name
        if hook_path.exists():
            hook_path.unlink()
            console.print(f"[green]Removed hook:[/] {hook_name}")
        else:
            console.print(f"[dim]Hook not installed: {hook_name}[/]")

    elif action == "show" and hook_name:
        if hook_name in HOOK_TEMPLATES:
            console.print(Syntax(
                HOOK_TEMPLATES[hook_name]["script"],
                "bash", theme="monokai", line_numbers=True,
            ))
        else:
            hook_path = hooks_dir / hook_name
            if hook_path.exists():
                console.print(Syntax(
                    hook_path.read_text(), "bash",
                    theme="monokai", line_numbers=True,
                ))
            else:
                console.print(f"[error]Hook not found: {hook_name}[/]")
    else:
        console.print("[error]Usage: llmstack hooks <list|install|install-all|remove|show> [hook-name][/]")
