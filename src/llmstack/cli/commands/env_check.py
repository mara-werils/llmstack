"""llmstack env-check — Validate development environment and .env files."""

from __future__ import annotations

import os
import re
from pathlib import Path

from llmstack.cli.console import console


# Common env var patterns that should NOT be committed
SECRET_PATTERNS = [
    (r"(?i)(password|passwd|pwd)\s*=\s*\S+", "Password detected"),
    (r"(?i)(secret|private.?key)\s*=\s*\S+", "Secret/private key detected"),
    (r"(?i)api.?key\s*=\s*sk-\S+", "OpenAI API key detected"),
    (r"(?i)api.?key\s*=\s*\S{20,}", "API key detected"),
    (r"(?i)token\s*=\s*gh[ps]_\S+", "GitHub token detected"),
    (r"(?i)aws_secret\s*=\s*\S+", "AWS secret detected"),
    (r"(?i)(database.?url|db.?url)\s*=\s*\S+://\S+:\S+@", "Database URL with credentials"),
    (r"-----BEGIN (RSA |EC )?PRIVATE KEY-----", "Private key in file"),
]

# Common required env vars by framework
FRAMEWORK_ENV_VARS = {
    "django": ["SECRET_KEY", "DEBUG", "DATABASE_URL", "ALLOWED_HOSTS"],
    "fastapi": ["DATABASE_URL"],
    "nextjs": ["NEXT_PUBLIC_API_URL"],
    "react": ["REACT_APP_API_URL"],
    "flask": ["SECRET_KEY", "FLASK_ENV"],
    "rails": ["SECRET_KEY_BASE", "DATABASE_URL", "RAILS_ENV"],
    "node": ["NODE_ENV", "PORT"],
}


def env_check(
    target: str | None = None,
    fix: bool = False,
) -> None:
    """Validate environment configuration."""
    from rich.table import Table
    from rich.panel import Panel

    directory = Path(target) if target else Path.cwd()
    issues = []
    info = []

    # Find .env files
    env_files = list(directory.glob(".env*"))
    env_files.extend(directory.glob("**/.env*"))
    env_files = [f for f in env_files if f.is_file() and ".git" not in str(f)]

    console.print()
    console.print(f"[bold]llmstack env-check[/]  directory=[dim]{directory}[/]")
    console.print()

    # Check .gitignore
    gitignore = directory / ".gitignore"
    env_in_gitignore = False
    if gitignore.exists():
        gitignore_content = gitignore.read_text()
        env_in_gitignore = ".env" in gitignore_content
        if not env_in_gitignore:
            issues.append({
                "severity": "CRITICAL",
                "file": ".gitignore",
                "issue": ".env is NOT in .gitignore — secrets may be committed!",
                "fix": 'Add ".env" to .gitignore',
            })
        else:
            info.append(".env is properly in .gitignore")
    else:
        issues.append({
            "severity": "WARNING",
            "file": ".gitignore",
            "issue": "No .gitignore file found",
            "fix": "Create .gitignore with .env entry",
        })

    # Check for .env.example
    env_example = directory / ".env.example"
    if not env_example.exists() and env_files:
        issues.append({
            "severity": "INFO",
            "file": ".env.example",
            "issue": "No .env.example file — new developers won't know required vars",
            "fix": "Create .env.example with placeholder values",
        })

    # Scan env files for secrets
    for env_file in env_files:
        if env_file.name == ".env.example":
            continue

        try:
            content = env_file.read_text(errors="replace")
        except OSError:
            continue

        rel_path = str(env_file.relative_to(directory))

        for pattern, desc in SECRET_PATTERNS:
            if re.search(pattern, content):
                issues.append({
                    "severity": "CRITICAL",
                    "file": rel_path,
                    "issue": desc,
                    "fix": "Use environment variables or a secrets manager",
                })

        # Check for empty values
        for line in content.split("\n"):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if not value:
                    issues.append({
                        "severity": "WARNING",
                        "file": rel_path,
                        "issue": f"Empty value for {key}",
                        "fix": f"Set a value for {key} or remove the line",
                    })

    # Check if .env is tracked by git
    try:
        import subprocess
        result = subprocess.run(
            ["git", "ls-files", "--error-unmatch", ".env"],
            capture_output=True, text=True, cwd=str(directory), timeout=5,
        )
        if result.returncode == 0:
            issues.append({
                "severity": "CRITICAL",
                "file": ".env",
                "issue": ".env is tracked by git — secrets are in version history!",
                "fix": "git rm --cached .env && add to .gitignore",
            })
    except Exception:
        pass

    # Detect framework and check required vars
    detected_framework = _detect_framework(directory)
    if detected_framework:
        info.append(f"Detected framework: {detected_framework}")
        required_vars = FRAMEWORK_ENV_VARS.get(detected_framework, [])
        if required_vars and env_files:
            env_content = ""
            for ef in env_files:
                if ef.name == ".env":
                    try:
                        env_content = ef.read_text()
                    except OSError:
                        pass
                    break

            existing_vars = set()
            for line in env_content.split("\n"):
                if "=" in line and not line.strip().startswith("#"):
                    existing_vars.add(line.split("=", 1)[0].strip())

            for var in required_vars:
                if var not in existing_vars and var not in os.environ:
                    issues.append({
                        "severity": "WARNING",
                        "file": ".env",
                        "issue": f"Missing {detected_framework} variable: {var}",
                        "fix": f"Add {var}=<value> to .env",
                    })

    # Display results
    sev_colors = {"CRITICAL": "bold red", "WARNING": "yellow", "INFO": "cyan"}
    sev_icons = {"CRITICAL": "✖", "WARNING": "⚠", "INFO": "ℹ"}

    if issues:
        table = Table(
            title=f"Environment Issues ({len(issues)})",
            show_header=True,
            header_style="bold cyan",
            border_style="dim",
        )
        table.add_column("Sev", width=10)
        table.add_column("File")
        table.add_column("Issue")
        table.add_column("Fix", style="dim")

        for issue in issues:
            sev = issue["severity"]
            color = sev_colors.get(sev, "white")
            icon = sev_icons.get(sev, "•")
            table.add_row(
                f"[{color}]{icon} {sev}[/]",
                issue["file"],
                issue["issue"],
                issue["fix"],
            )

        console.print(table)

    # Summary
    critical = sum(1 for i in issues if i["severity"] == "CRITICAL")
    warnings = sum(1 for i in issues if i["severity"] == "WARNING")
    status_color = "red" if critical > 0 else "yellow" if warnings > 0 else "green"
    status_text = "ISSUES FOUND" if issues else "ALL CLEAR"

    console.print()
    console.print(Panel(
        f"[bold]Status:[/] [{status_color}]{status_text}[/]\n"
        f"[bold]Env files:[/] {len(env_files)}\n"
        f"[bold]Critical:[/] {critical}\n"
        f"[bold]Warnings:[/] {warnings}\n" +
        ("\n".join(f"[dim]• {i}[/]" for i in info) if info else ""),
        title="Environment Check",
        border_style=status_color,
    ))

    # Auto-fix
    if fix and not env_in_gitignore and gitignore:
        with open(gitignore, "a") as f:
            f.write("\n# Environment\n.env\n.env.local\n.env.*.local\n")
        console.print("[green]Fixed: Added .env to .gitignore[/]")


def _detect_framework(directory: Path) -> str | None:
    """Detect project framework."""
    if (directory / "manage.py").exists():
        return "django"
    if (directory / "next.config.js").exists() or (directory / "next.config.mjs").exists():
        return "nextjs"
    if (directory / "Gemfile").exists():
        return "rails"

    pyproject = directory / "pyproject.toml"
    if pyproject.exists():
        content = pyproject.read_text()
        if "fastapi" in content:
            return "fastapi"
        if "flask" in content:
            return "flask"

    package_json = directory / "package.json"
    if package_json.exists():
        try:
            import json
            data = json.loads(package_json.read_text())
            deps = {**data.get("dependencies", {}), **data.get("devDependencies", {})}
            if "react" in deps:
                return "react"
            if "express" in deps:
                return "node"
        except Exception:
            pass

    return None
