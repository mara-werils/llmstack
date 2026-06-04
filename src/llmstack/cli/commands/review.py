"""llmstack review — AI-powered code review for git diffs and GitHub PRs."""

from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path

from llmstack.cli.console import console


REVIEW_SYSTEM_PROMPT = """You are an expert code reviewer. Analyze the provided git diff and give a structured code review.

For each issue found, output a JSON line in this format:
{"severity": "CRITICAL|WARNING|INFO", "file": "filename", "line": 42, "message": "description", "suggestion": "fix suggestion"}

After all issues, output a summary line:
{"type": "summary", "total_issues": 3, "critical": 1, "warnings": 1, "info": 1, "verdict": "NEEDS_CHANGES|APPROVED", "summary": "overall assessment"}

Focus on:
- Security vulnerabilities (CRITICAL)
- Logic errors and bugs (CRITICAL/WARNING)
- Performance issues (WARNING)
- Code style and best practices (INFO)
- Missing error handling (WARNING)
- Documentation (INFO)

Be specific with file names and line numbers from the diff."""


def review(
    target: str = "",
    pr_url: str | None = None,
    model: str = "llama3.2",
    ollama_url: str = "http://localhost:11434",
    output_format: str = "terminal",
    severity: str | None = None,
    output_file: str | None = None,
    staged: bool = False,
    commits: int = 1,
) -> None:
    """Run AI code review."""
    asyncio.run(
        _review_async(
            target=target,
            pr_url=pr_url,
            model=model,
            ollama_url=ollama_url,
            output_format=output_format,
            severity=severity,
            output_file=output_file,
            staged=staged,
            commits=commits,
        )
    )


def _get_git_diff(target: str, staged: bool, commits: int) -> str:
    """Get git diff based on options."""
    cwd = str(Path.cwd())

    def run_git(*args):
        try:
            result = subprocess.run(
                ["git", *args],
                capture_output=True,
                text=True,
                cwd=cwd,
                timeout=30,
            )
            return result.stdout if result.returncode == 0 else ""
        except Exception:
            return ""

    if staged:
        return run_git("diff", "--staged")
    elif ".." in target or target.startswith("origin/") or target.startswith("HEAD"):
        return run_git("diff", target)
    elif commits > 0:
        return run_git("diff", f"HEAD~{commits}..HEAD")
    else:
        # Unstaged + staged
        diff = run_git("diff", "HEAD") or run_git("diff")
        if not diff:
            diff = run_git("diff", "HEAD~1..HEAD")
        return diff


async def _fetch_pr_diff(pr_url: str) -> str:
    """Fetch PR diff from GitHub API."""
    import httpx
    import re
    import os

    # Parse github.com/owner/repo/pull/123
    match = re.search(r"github\.com/([^/]+)/([^/]+)/pull/(\d+)", pr_url)
    if not match:
        console.print("[error]Invalid GitHub PR URL. Expected: github.com/owner/repo/pull/123[/]")
        return ""

    owner, repo, pr_num = match.groups()
    api_url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_num}"

    headers = {"Accept": "application/vnd.github.v3.diff"}

    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(api_url, headers=headers)
        if resp.status_code == 200:
            return resp.text
        elif resp.status_code == 404:
            console.print(f"[error]PR not found: {pr_url}[/]")
        else:
            console.print(f"[error]GitHub API error: {resp.status_code}[/]")
    return ""


async def _review_async(
    target: str,
    pr_url: str | None,
    model: str,
    ollama_url: str,
    output_format: str,
    severity: str | None,
    output_file: str | None,
    staged: bool,
    commits: int,
) -> None:
    import httpx
    import typer
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn

    ollama_url = ollama_url.rstrip("/")

    # Check Ollama
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{ollama_url}/api/version")
            if resp.status_code != 200:
                console.print("[error]Ollama is not responding.[/]")
                raise typer.Exit(1)
    except httpx.ConnectError:
        console.print(
            Panel(
                "[error]Cannot connect to Ollama.[/]\n\nMake sure Ollama is running:\n  [bold cyan]ollama serve[/]",
                title="Connection Error",
                border_style="red",
            )
        )
        raise typer.Exit(1)

    # Get diff
    if pr_url:
        console.print(f"  [dim]Fetching PR diff from {pr_url}...[/]")
        diff = await _fetch_pr_diff(pr_url)
        source_label = f"PR: {pr_url}"
    else:
        diff = _get_git_diff(target, staged, commits)
        source_label = f"HEAD~{commits}..HEAD" if commits > 0 else "working tree"

    if not diff:
        console.print("[warning]No diff found. Make sure there are changes to review.[/]")
        raise typer.Exit(0)

    # Truncate very large diffs
    MAX_DIFF = 12000
    if len(diff) > MAX_DIFF:
        diff = diff[:MAX_DIFF] + f"\n\n... (diff truncated, showing first {MAX_DIFF} chars)"

    console.print()
    console.print(
        f"[bold]llmstack review[/]  model=[cyan]{model}[/]  source=[dim]{source_label}[/]"
    )
    console.print(f"  [dim]Diff size: {len(diff)} chars[/]")
    console.print()

    prompt = f"""Review this git diff and provide structured feedback.

{diff}

Output each issue as a JSON object on its own line, followed by a summary JSON object.
Start directly with JSON output, no preamble."""

    # Stream LLM response
    issues = []
    summary_data = None
    raw_lines = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]Reviewing with AI..."),
        console=console,
    ) as progress:
        task = progress.add_task("Reviewing", total=None)

        timeout = httpx.Timeout(300, connect=10, read=300, write=30)
        async with httpx.AsyncClient(timeout=timeout) as client:
            full_response = ""
            async with client.stream(
                "POST",
                f"{ollama_url}/api/chat",
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": REVIEW_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    "stream": True,
                },
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                        token = data.get("message", {}).get("content", "")
                        if token:
                            full_response += token
                        if data.get("done", False):
                            break
                    except json.JSONDecodeError:
                        continue

        progress.update(task, completed=True)

    # Parse response — handle both clean JSON lines and JSON embedded in prose
    for line in full_response.splitlines():
        line = line.strip()
        if not line:
            continue
        # Strip markdown code fences if present
        if line.startswith("```"):
            continue
        # Find first { in line for embedded JSON
        brace_idx = line.find("{")
        if brace_idx == -1:
            continue
        candidate = line[brace_idx:]
        try:
            obj = json.loads(candidate)
            if obj.get("type") == "summary":
                summary_data = obj
            elif "severity" in obj:
                issues.append(obj)
            raw_lines.append(candidate)
        except json.JSONDecodeError:
            raw_lines.append(line)

    # Filter by severity
    if severity:
        sev_upper = severity.upper()
        issues = [i for i in issues if i.get("severity", "").upper() == sev_upper]

    # Display results
    if output_format == "terminal" or not output_format:
        _display_terminal(issues, summary_data, diff)
    elif output_format == "markdown":
        md = _format_markdown(issues, summary_data, source_label)
        if output_file:
            Path(output_file).write_text(md)
            console.print(f"[green]Report saved to {output_file}[/]")
        else:
            from rich.markdown import Markdown

            console.print(Markdown(md))
    elif output_format == "json":
        output_data = {"source": source_label, "issues": issues, "summary": summary_data}
        json_str = json.dumps(output_data, indent=2)
        if output_file:
            Path(output_file).write_text(json_str)
            console.print(f"[green]Report saved to {output_file}[/]")
        else:
            console.print(json_str)

    # If no structured output, fallback to raw
    if not issues and not summary_data:
        console.print()
        console.print(Panel(full_response, title="Review", border_style="cyan"))


def _display_terminal(issues: list, summary_data: dict | None, diff: str) -> None:
    """Display review results in terminal."""
    from rich.panel import Panel
    from rich.table import Table

    if not issues and not summary_data:
        return

    sev_colors = {"CRITICAL": "bold red", "WARNING": "yellow", "INFO": "cyan"}
    sev_icons = {"CRITICAL": "✖", "WARNING": "⚠", "INFO": "ℹ"}

    if issues:
        table = Table(
            title="Code Review Issues",
            show_header=True,
            header_style="bold cyan",
            border_style="dim",
            padding=(0, 1),
        )
        table.add_column("Sev", style="bold", width=8)
        table.add_column("File", style="bold")
        table.add_column("Line", justify="right", width=6)
        table.add_column("Issue")
        table.add_column("Suggestion", style="dim")

        for issue in issues:
            sev = issue.get("severity", "INFO")
            color = sev_colors.get(sev, "white")
            icon = sev_icons.get(sev, "•")
            table.add_row(
                f"[{color}]{icon} {sev}[/]",
                issue.get("file", ""),
                str(issue.get("line", "")),
                issue.get("message", ""),
                issue.get("suggestion", ""),
            )

        console.print()
        console.print(table)

    if summary_data:
        verdict = summary_data.get("verdict", "")
        verdict_color = "green" if verdict == "APPROVED" else "red"
        console.print()
        console.print(
            Panel(
                f"[bold]Verdict:[/] [{verdict_color}]{verdict}[/]\n\n"
                f"{summary_data.get('summary', '')}\n\n"
                f"[dim]Total: {summary_data.get('total_issues', 0)} issues | "
                f"Critical: {summary_data.get('critical', 0)} | "
                f"Warnings: {summary_data.get('warnings', 0)} | "
                f"Info: {summary_data.get('info', 0)}[/]",
                title="Review Summary",
                border_style=verdict_color,
            )
        )


def _format_markdown(issues: list, summary_data: dict | None, source: str) -> str:
    """Format review as Markdown."""
    lines = [f"# Code Review Report\n\nSource: `{source}`\n"]

    if summary_data:
        verdict = summary_data.get("verdict", "")
        lines.append(f"## Verdict: {verdict}\n")
        lines.append(f"{summary_data.get('summary', '')}\n")
        lines.append(
            f"| Metric | Count |\n|--------|-------|\n"
            f"| Critical | {summary_data.get('critical', 0)} |\n"
            f"| Warnings | {summary_data.get('warnings', 0)} |\n"
            f"| Info | {summary_data.get('info', 0)} |\n"
        )

    if issues:
        lines.append("## Issues\n")
        for i, issue in enumerate(issues, 1):
            sev = issue.get("severity", "INFO")
            lines.append(f"### {i}. [{sev}] {issue.get('file', '')}:{issue.get('line', '')}")
            lines.append(f"\n**Issue:** {issue.get('message', '')}\n")
            lines.append(f"**Suggestion:** {issue.get('suggestion', '')}\n")

    return "\n".join(lines)
