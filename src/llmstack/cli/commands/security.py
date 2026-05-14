"""llmstack security — AI-powered security audit."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from llmstack.cli.console import console


SECURITY_SYSTEM_PROMPT = """You are a security expert performing a code security audit.

Find vulnerabilities based on OWASP Top 10 and common security issues:
1. Injection (SQL, Command, LDAP)
2. Broken Authentication
3. Sensitive Data Exposure (hardcoded secrets, API keys, passwords)
4. Security Misconfiguration
5. Cross-Site Scripting (XSS)
6. Insecure Deserialization
7. Using Components with Known Vulnerabilities
8. Insufficient Logging

For each finding, output a JSON line:
{"severity": "CRITICAL|HIGH|MEDIUM|LOW", "category": "OWASP category", "file": "filename", "line": 42, "description": "what the vulnerability is", "recommendation": "how to fix it", "cwe": "CWE-89"}

After all findings, output:
{"type": "summary", "total": 5, "critical": 1, "high": 2, "medium": 1, "low": 1, "risk_score": 7.5}

Be thorough but avoid false positives."""


def security(
    target: str | None = None,
    model: str = "llama3.2",
    ollama_url: str = "http://localhost:11434",
    output_format: str = "terminal",
    output_file: str | None = None,
    severity: str | None = None,
) -> None:
    """Run an AI-powered security audit on your code."""
    asyncio.run(_security_async(
        target=target, model=model, ollama_url=ollama_url,
        output_format=output_format, output_file=output_file, severity=severity,
    ))


async def _security_async(
    target: str | None,
    model: str,
    ollama_url: str,
    output_format: str,
    output_file: str | None,
    severity: str | None,
) -> None:
    import httpx
    import typer
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn

    ollama_url = ollama_url.rstrip("/")

    # Check Ollama
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{ollama_url}/api/version")
            if resp.status_code != 200:
                console.print("[error]Ollama is not responding.[/]")
                raise typer.Exit(1)
    except httpx.ConnectError:
        console.print(Panel("[error]Cannot connect to Ollama.[/]", border_style="red"))
        raise typer.Exit(1)

    target_path = Path(target) if target else Path.cwd()

    # Collect files to audit
    code_exts = {".py", ".js", ".ts", ".go", ".java", ".rb", ".php", ".rs"}
    if target_path.is_file():
        files = [target_path]
    else:
        files = [
            p for p in target_path.rglob("*")
            if p.suffix.lower() in code_exts
            and not any(x in str(p) for x in ["__pycache__", ".git", "node_modules", "venv"])
        ][:10]  # Limit for safety

    if not files:
        console.print("[warning]No source files found.[/]")
        return

    console.print()
    console.print(f"[bold]llmstack security[/]  model=[cyan]{model}[/]  files=[dim]{len(files)}[/]")
    console.print()

    all_issues = []
    summary_data = None
    timeout = httpx.Timeout(300, connect=10, read=300, write=30)

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]Scanning..."),
        BarColumn(),
        MofNCompleteColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Scanning", total=len(files))

        for fpath in files:
            try:
                content = fpath.read_text(errors="replace")
                if len(content) > 8000:
                    content = content[:8000] + "\n... (truncated)"
            except OSError:
                progress.advance(task)
                continue

            prompt = f"""Perform a security audit on this file. Find ALL security vulnerabilities.

File: {fpath}

```
{content}
```

Output JSON findings, then summary."""

            async with httpx.AsyncClient(timeout=timeout) as client:
                full_response = ""
                async with client.stream(
                    "POST", f"{ollama_url}/api/chat",
                    json={
                        "model": model,
                        "messages": [
                            {"role": "system", "content": SECURITY_SYSTEM_PROMPT},
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
                                    full_response += token
                                if data.get("done", False):
                                    break
                            except json.JSONDecodeError:
                                continue

            for line in full_response.splitlines():
                line = line.strip()
                if not line.startswith("{"):
                    continue
                try:
                    obj = json.loads(line)
                    if obj.get("type") == "summary":
                        summary_data = obj
                    elif "severity" in obj:
                        obj["_file"] = str(fpath)
                        all_issues.append(obj)
                except json.JSONDecodeError:
                    pass

            progress.advance(task)

    # Filter by severity
    if severity:
        sev_upper = severity.upper()
        all_issues = [i for i in all_issues if i.get("severity", "").upper() == sev_upper]

    # Display
    if output_format == "terminal":
        _display_security_terminal(all_issues, summary_data)
    elif output_format in ("markdown", "json"):
        data = {"issues": all_issues, "summary": summary_data}
        if output_format == "json":
            out = json.dumps(data, indent=2)
        else:
            out = _format_security_markdown(all_issues, summary_data)

        if output_file:
            Path(output_file).write_text(out)
            console.print(f"[green]Report saved to {output_file}[/]")
        else:
            console.print(out)


def _display_security_terminal(issues: list, summary_data: dict | None) -> None:
    from rich.table import Table
    from rich.panel import Panel

    sev_order = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
    sev_colors = {"CRITICAL": "bold red", "HIGH": "red", "MEDIUM": "yellow", "LOW": "cyan"}
    sev_icons = {"CRITICAL": "✖", "HIGH": "!", "MEDIUM": "⚠", "LOW": "i"}

    if issues:
        table = Table(
            title="Security Findings",
            show_header=True,
            header_style="bold red",
            border_style="dim",
            padding=(0, 1),
        )
        table.add_column("Sev", width=10)
        table.add_column("Category", width=20)
        table.add_column("File:Line")
        table.add_column("Issue")
        table.add_column("CWE", width=10)

        def sort_key(x):
            sev = x.get("severity", "LOW")
            return sev_order.index(sev) if sev in sev_order else len(sev_order)

        for issue in sorted(issues, key=sort_key):
            sev = issue.get("severity", "LOW")
            color = sev_colors.get(sev, "white")
            icon = sev_icons.get(sev, "•")
            file_line = f"{issue.get('file', '')}"
            if issue.get("line"):
                file_line += f":{issue['line']}"
            table.add_row(
                f"[{color}]{icon} {sev}[/]",
                issue.get("category", ""),
                file_line,
                issue.get("description", ""),
                issue.get("cwe", ""),
            )

        console.print()
        console.print(table)

    if summary_data:
        risk = summary_data.get("risk_score", 0)
        risk_color = "red" if risk >= 7 else "yellow" if risk >= 4 else "green"
        console.print()
        console.print(Panel(
            f"[bold]Risk Score:[/] [{risk_color}]{risk}/10[/]\n"
            f"[dim]Critical: {summary_data.get('critical', 0)} | "
            f"High: {summary_data.get('high', 0)} | "
            f"Medium: {summary_data.get('medium', 0)} | "
            f"Low: {summary_data.get('low', 0)}[/]",
            title="Security Summary",
            border_style=risk_color,
        ))
    elif not issues:
        console.print()
        console.print(Panel("[green]No security issues found.[/]", border_style="green"))


def _format_security_markdown(issues: list, summary_data: dict | None) -> str:
    lines = ["# Security Audit Report\n"]
    if summary_data:
        risk = summary_data.get("risk_score", 0)
        lines.append(f"**Risk Score:** {risk}/10\n")
        lines.append("| Severity | Count |\n|----------|-------|\n")
        for sev in ["critical", "high", "medium", "low"]:
            lines.append(f"| {sev.title()} | {summary_data.get(sev, 0)} |\n")

    if issues:
        lines.append("\n## Findings\n")
        for i, issue in enumerate(issues, 1):
            lines.append(f"### {i}. [{issue.get('severity', 'LOW')}] {issue.get('category', '')}")
            lines.append(f"\n**File:** `{issue.get('file', '')}:{issue.get('line', '')}`")
            lines.append(f"\n**CWE:** {issue.get('cwe', 'N/A')}")
            lines.append(f"\n**Description:** {issue.get('description', '')}")
            lines.append(f"\n**Recommendation:** {issue.get('recommendation', '')}\n")

    return "\n".join(lines)
