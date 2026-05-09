"""Git-aware context — enrich answers with git history, diffs, and blame.

Provides additional context to the LLM about recent changes, authorship,
and file history. All operations use subprocess to call git directly.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class GitInfo:
    """Git context for a project."""

    is_repo: bool = False
    branch: str = ""
    recent_commits: list[dict[str, str]] = field(default_factory=list)
    changed_files: list[str] = field(default_factory=list)
    diff_summary: str = ""

    def to_context(self) -> str:
        """Format git info as context for the LLM prompt."""
        if not self.is_repo:
            return ""

        parts = [f"Git branch: {self.branch}"]

        if self.recent_commits:
            parts.append("\nRecent commits:")
            for c in self.recent_commits[:10]:
                parts.append(f"  {c.get('hash', '')[:8]} {c.get('message', '')} ({c.get('author', '')})")

        if self.changed_files:
            parts.append(f"\nUncommitted changes ({len(self.changed_files)} files):")
            for f in self.changed_files[:20]:
                parts.append(f"  {f}")

        if self.diff_summary:
            parts.append(f"\nRecent diff summary:\n{self.diff_summary[:1000]}")

        return "\n".join(parts)


def get_git_info(project_dir: str | Path, max_commits: int = 15) -> GitInfo:
    """Gather git context for a project directory."""
    cwd = str(Path(project_dir).resolve())
    info = GitInfo()

    # Check if git repo
    if not _run_git(["rev-parse", "--is-inside-work-tree"], cwd):
        return info
    info.is_repo = True

    # Current branch
    info.branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], cwd).strip()

    # Recent commits
    log_output = _run_git(
        ["log", f"--max-count={max_commits}", "--format=%H|%an|%s|%cr"],
        cwd,
    )
    for line in log_output.strip().splitlines():
        parts = line.split("|", 3)
        if len(parts) >= 4:
            info.recent_commits.append({
                "hash": parts[0],
                "author": parts[1],
                "message": parts[2],
                "when": parts[3],
            })

    # Changed files (unstaged + staged)
    status = _run_git(["status", "--porcelain"], cwd)
    info.changed_files = [
        line[3:].strip() for line in status.splitlines() if line.strip()
    ]

    # Diff summary (last commit)
    info.diff_summary = _run_git(["diff", "--stat", "HEAD~1..HEAD"], cwd)

    return info


def get_file_blame(file_path: str | Path, project_dir: str | Path) -> str:
    """Get git blame for a specific file (summarized)."""
    cwd = str(Path(project_dir).resolve())
    blame = _run_git(["blame", "--line-porcelain", str(file_path)], cwd)
    if not blame:
        return ""

    # Summarize: count lines per author
    authors: dict[str, int] = {}
    for line in blame.splitlines():
        if line.startswith("author "):
            name = line[7:]
            authors[name] = authors.get(name, 0) + 1

    if not authors:
        return ""

    total = sum(authors.values())
    parts = []
    for author, count in sorted(authors.items(), key=lambda x: -x[1]):
        pct = round(count / total * 100)
        parts.append(f"  {author}: {count} lines ({pct}%)")

    return "File ownership:\n" + "\n".join(parts)


def get_file_log(file_path: str | Path, project_dir: str | Path, max_entries: int = 10) -> str:
    """Get recent git log for a specific file."""
    cwd = str(Path(project_dir).resolve())
    log = _run_git(
        ["log", f"--max-count={max_entries}", "--format=%h %s (%cr)", "--", str(file_path)],
        cwd,
    )
    return log.strip()


def get_recent_diff(project_dir: str | Path, commits_back: int = 1) -> str:
    """Get the diff of recent changes (truncated for context window)."""
    cwd = str(Path(project_dir).resolve())
    diff = _run_git(["diff", f"HEAD~{commits_back}..HEAD", "--stat"], cwd)
    return diff[:2000] if diff else ""


def _run_git(args: list[str], cwd: str) -> str:
    """Run a git command and return stdout, or empty string on error."""
    try:
        result = subprocess.run(
            ["git", *args],
            capture_output=True, text=True, timeout=10, cwd=cwd,
        )
        return result.stdout if result.returncode == 0 else ""
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return ""
