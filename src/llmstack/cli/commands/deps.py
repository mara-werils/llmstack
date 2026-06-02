"""llmstack deps — Analyze project dependencies for security and updates."""

from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path

from llmstack.cli.console import console


# Supported package managers
MANIFEST_FILES = {
    "requirements.txt": "pip",
    "pyproject.toml": "pip",
    "setup.py": "pip",
    "Pipfile": "pipenv",
    "package.json": "npm",
    "yarn.lock": "yarn",
    "pnpm-lock.yaml": "pnpm",
    "go.mod": "go",
    "Cargo.toml": "cargo",
    "Gemfile": "bundler",
    "composer.json": "composer",
}


def deps(
    target: str | None = None,
    check_updates: bool = True,
    check_security: bool = True,
    model: str = "llama3.2",
    ollama_url: str = "http://localhost:11434",
    output: str | None = None,
) -> None:
    """Analyze project dependencies."""
    asyncio.run(_deps_async(
        target=target, check_updates=check_updates,
        check_security=check_security, model=model,
        ollama_url=ollama_url, output=output,
    ))


def _detect_manifests(directory: Path) -> dict[str, Path]:
    """Find dependency manifest files."""
    found = {}
    for name, manager in MANIFEST_FILES.items():
        path = directory / name
        if path.exists():
            found[manager] = path
    return found


def _parse_requirements_txt(path: Path) -> list[dict]:
    """Parse requirements.txt."""
    deps = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        # Parse name==version, name>=version, name
        for op in ["==", ">=", "<=", "~=", "!=", ">", "<"]:
            if op in line:
                name, version = line.split(op, 1)
                deps.append({"name": name.strip(), "version": version.strip(), "constraint": op})
                break
        else:
            deps.append({"name": line, "version": "any", "constraint": ""})
    return deps


def _parse_package_json(path: Path) -> list[dict]:
    """Parse package.json dependencies."""
    deps = []
    try:
        data = json.loads(path.read_text())
        for section in ("dependencies", "devDependencies"):
            for name, version in data.get(section, {}).items():
                deps.append({
                    "name": name, "version": version,
                    "dev": section == "devDependencies",
                })
    except json.JSONDecodeError:
        pass
    return deps


def _parse_go_mod(path: Path) -> list[dict]:
    """Parse go.mod."""
    deps = []
    in_require = False
    for line in path.read_text().splitlines():
        line = line.strip()
        if line.startswith("require ("):
            in_require = True
            continue
        if line == ")":
            in_require = False
            continue
        if in_require and line:
            parts = line.split()
            if len(parts) >= 2:
                deps.append({"name": parts[0], "version": parts[1]})
    return deps


def _parse_cargo_toml(path: Path) -> list[dict]:
    """Parse Cargo.toml dependencies."""
    deps = []
    in_deps = False
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("[dependencies]") or stripped.startswith("[dev-dependencies]"):
            in_deps = True
            continue
        if stripped.startswith("[") and in_deps:
            in_deps = False
            continue
        if in_deps and "=" in stripped:
            parts = stripped.split("=", 1)
            name = parts[0].strip()
            version = parts[1].strip().strip('"').strip("'")
            if not name.startswith("#"):
                deps.append({"name": name, "version": version})
    return deps


def _parse_pyproject_toml(path: Path) -> list[dict]:
    """Parse pyproject.toml dependencies."""
    deps = []
    in_deps = False
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if "dependencies" in stripped and "=" in stripped:
            in_deps = True
            continue
        if stripped.startswith("[") and in_deps:
            in_deps = False
            continue
        if in_deps and stripped.startswith('"'):
            dep = stripped.strip('",')
            for op in [">=", "<=", "==", "~=", ">", "<"]:
                if op in dep:
                    name, ver = dep.split(op, 1)
                    deps.append({"name": name.strip(), "version": ver.strip(), "constraint": op})
                    break
            else:
                deps.append({"name": dep.strip(), "version": "any", "constraint": ""})
    return deps


async def _deps_async(
    target: str | None,
    check_updates: bool,
    check_security: bool,
    model: str,
    ollama_url: str,
    output: str | None,
) -> None:
    import httpx
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn

    directory = Path(target) if target else Path.cwd()
    manifests = _detect_manifests(directory)

    if not manifests:
        console.print("[warning]No dependency manifest files found.[/]")
        console.print("[dim]Supported: requirements.txt, package.json, go.mod, Cargo.toml, pyproject.toml[/]")
        return

    console.print()
    console.print(f"[bold]llmstack deps[/]  directory=[dim]{directory}[/]")
    console.print(f"  [dim]Found: {', '.join(manifests.keys())}[/]")
    console.print()

    all_deps = []
    parsers = {
        "pip": lambda p: _parse_requirements_txt(p) if p.name == "requirements.txt" else _parse_pyproject_toml(p),
        "npm": _parse_package_json,
        "yarn": lambda p: _parse_package_json(directory / "package.json") if (directory / "package.json").exists() else [],
        "go": _parse_go_mod,
        "cargo": _parse_cargo_toml,
    }

    for manager, path in manifests.items():
        parser = parsers.get(manager)
        if parser:
            deps_list = parser(path)
            for d in deps_list:
                d["manager"] = manager
            all_deps.extend(deps_list)

    if not all_deps:
        console.print("[dim]No dependencies found in manifest files.[/]")
        return

    # Display dependencies table
    table = Table(
        title=f"Dependencies ({len(all_deps)} total)",
        show_header=True,
        header_style="bold cyan",
        border_style="dim",
    )
    table.add_column("Package", style="bold")
    table.add_column("Version")
    table.add_column("Manager", width=10)
    table.add_column("Type", width=6)

    for dep in sorted(all_deps, key=lambda x: x["name"]):
        dep_type = "dev" if dep.get("dev") else "prod"
        table.add_row(
            dep["name"],
            dep.get("version", ""),
            dep.get("manager", ""),
            dep_type,
        )

    console.print(table)

    # AI security analysis
    if check_security and model:
        ollama_url = ollama_url.rstrip("/")
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"{ollama_url}/api/version")
                if resp.status_code != 200:
                    return
        except httpx.ConnectError:
            console.print("[dim]Ollama not available, skipping AI analysis.[/]")
            return

        dep_summary = "\n".join(
            f"- {d['name']} {d.get('version', '')} ({d.get('manager', '')})"
            for d in all_deps[:50]
        )

        prompt = f"""Analyze these project dependencies for potential issues:

{dep_summary}

Check for:
1. Known security vulnerabilities in these package versions
2. Deprecated or unmaintained packages
3. Version pinning issues
4. Unnecessary or redundant dependencies
5. License compatibility concerns

Output a brief, actionable analysis."""

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]AI analyzing dependencies..."),
            console=console,
        ) as progress:
            task = progress.add_task("Analyzing", total=None)

            analysis = ""
            timeout = httpx.Timeout(120, connect=10, read=120, write=30)
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream(
                    "POST", f"{ollama_url}/api/chat",
                    json={
                        "model": model,
                        "messages": [
                            {"role": "system", "content": "You are a software security expert analyzing project dependencies."},
                            {"role": "user", "content": prompt},
                        ],
                        "stream": True,
                    },
                ) as resp:
                    if resp.status_code == 200:
                        async for line in resp.aiter_lines():
                            if not line.strip():
                                continue
                            try:
                                data = json.loads(line)
                                token = data.get("message", {}).get("content", "")
                                if token:
                                    analysis += token
                                if data.get("done", False):
                                    break
                            except json.JSONDecodeError:
                                continue

            progress.update(task, completed=True)

        if analysis:
            from rich.markdown import Markdown
            console.print()
            console.print(Panel(
                Markdown(analysis),
                title="Dependency Analysis",
                border_style="cyan",
            ))

    if output:
        out_data = {"dependencies": all_deps, "manifests": {k: str(v) for k, v in manifests.items()}}
        Path(output).write_text(json.dumps(out_data, indent=2))
        console.print(f"\n[green]Report saved to {output}[/]")
