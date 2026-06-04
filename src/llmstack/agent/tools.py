"""Tool system — registry of callable tools for the agent.

Each tool has a name, description, parameter schema, and an execute method.
Tools are designed to be safe by default with configurable sandboxing.
"""

from __future__ import annotations

import os
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ToolResult:
    """Result of a tool execution."""

    output: str
    success: bool = True
    error: str | None = None

    def to_message(self) -> str:
        if self.success:
            return self.output
        return f"Error: {self.error or self.output}"


@dataclass
class ToolParam:
    """A parameter for a tool."""

    name: str
    type: str = "string"
    description: str = ""
    required: bool = True


class Tool(ABC):
    """Base class for agent tools."""

    name: str = "base"
    description: str = ""
    parameters: list[ToolParam] = []

    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with the given parameters."""

    def schema(self) -> dict:
        """Return OpenAI-compatible function schema for this tool."""
        properties = {}
        required = []
        for p in self.parameters:
            properties[p.name] = {"type": p.type, "description": p.description}
            if p.required:
                required.append(p.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }


# ---------------------------------------------------------------------------
# Built-in tools
# ---------------------------------------------------------------------------


class ReadFileTool(Tool):
    name = "read_file"
    description = "Read the contents of a file. Returns the file content with line numbers."
    parameters = [
        ToolParam(name="path", description="Path to the file to read"),
        ToolParam(
            name="start_line",
            type="integer",
            description="Start reading from this line (1-based)",
            required=False,
        ),
        ToolParam(
            name="end_line", type="integer", description="Stop reading at this line", required=False
        ),
    ]

    def __init__(self, working_dir: str = "."):
        self._cwd = Path(working_dir).resolve()

    async def execute(
        self, path: str = "", start_line: int = 0, end_line: int = 0, **kw
    ) -> ToolResult:
        target = self._resolve(path)
        if not target.is_file():
            return ToolResult(output="", success=False, error=f"File not found: {path}")
        try:
            text = target.read_text(errors="replace")
            lines = text.splitlines()
            if start_line > 0:
                start_idx = max(0, start_line - 1)
                end_idx = end_line if end_line > 0 else len(lines)
                lines = lines[start_idx:end_idx]
                numbered = [f"{start_idx + i + 1}\t{ln}" for i, ln in enumerate(lines)]
            else:
                numbered = [f"{i + 1}\t{ln}" for i, ln in enumerate(lines)]
            return ToolResult(output="\n".join(numbered))
        except Exception as exc:
            return ToolResult(output="", success=False, error=str(exc))

    def _resolve(self, path: str) -> Path:
        p = Path(path)
        if p.is_absolute():
            return p
        return self._cwd / p


class WriteFileTool(Tool):
    name = "write_file"
    description = "Write content to a file. Creates parent directories if needed."
    parameters = [
        ToolParam(name="path", description="Path to the file to write"),
        ToolParam(name="content", description="Content to write to the file"),
    ]

    def __init__(self, working_dir: str = "."):
        self._cwd = Path(working_dir).resolve()

    async def execute(self, path: str = "", content: str = "", **kw) -> ToolResult:
        target = self._resolve(path)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)
            return ToolResult(output=f"Written {len(content)} bytes to {path}")
        except Exception as exc:
            return ToolResult(output="", success=False, error=str(exc))

    def _resolve(self, path: str) -> Path:
        p = Path(path)
        if p.is_absolute():
            return p
        return self._cwd / p


class ListDirectoryTool(Tool):
    name = "list_directory"
    description = "List files and directories in a path. Shows type, size, and name."
    parameters = [
        ToolParam(name="path", description="Directory path to list", required=False),
    ]

    def __init__(self, working_dir: str = "."):
        self._cwd = Path(working_dir).resolve()

    async def execute(self, path: str = ".", **kw) -> ToolResult:
        target = self._resolve(path)
        if not target.is_dir():
            return ToolResult(output="", success=False, error=f"Not a directory: {path}")
        try:
            entries = sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name))
            lines = []
            for entry in entries:
                if entry.name.startswith("."):
                    continue
                kind = "dir" if entry.is_dir() else "file"
                size = ""
                if entry.is_file():
                    sz = entry.stat().st_size
                    if sz < 1024:
                        size = f"{sz}B"
                    elif sz < 1024 * 1024:
                        size = f"{sz // 1024}KB"
                    else:
                        size = f"{sz // (1024 * 1024)}MB"
                lines.append(f"  {kind:4s}  {size:>6s}  {entry.name}")
            return ToolResult(output="\n".join(lines) if lines else "(empty directory)")
        except Exception as exc:
            return ToolResult(output="", success=False, error=str(exc))

    def _resolve(self, path: str) -> Path:
        p = Path(path)
        if p.is_absolute():
            return p
        return self._cwd / p


class GrepTool(Tool):
    name = "grep"
    description = (
        "Search for a pattern in files. Returns matching lines with file paths and line numbers."
    )
    parameters = [
        ToolParam(name="pattern", description="Regex pattern to search for"),
        ToolParam(name="path", description="File or directory to search in", required=False),
        ToolParam(
            name="include", description="Glob pattern for file names (e.g. '*.py')", required=False
        ),
    ]

    def __init__(self, working_dir: str = "."):
        self._cwd = Path(working_dir).resolve()

    async def execute(
        self, pattern: str = "", path: str = ".", include: str = "", **kw
    ) -> ToolResult:
        target = self._resolve(path)
        cmd = ["grep", "-rn", "--color=never"]
        if include:
            cmd.extend(["--include", include])
        cmd.extend([pattern, str(target)])

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=30, cwd=str(self._cwd)
            )
            output = result.stdout.strip()
            if not output:
                return ToolResult(output="No matches found.")
            # Limit output
            lines = output.splitlines()
            if len(lines) > 50:
                output = "\n".join(lines[:50]) + f"\n... ({len(lines) - 50} more matches)"
            return ToolResult(output=output)
        except subprocess.TimeoutExpired:
            return ToolResult(output="", success=False, error="Search timed out after 30s")
        except Exception as exc:
            return ToolResult(output="", success=False, error=str(exc))

    def _resolve(self, path: str) -> Path:
        p = Path(path)
        if p.is_absolute():
            return p
        return self._cwd / p


class ShellTool(Tool):
    name = "shell"
    description = (
        "Execute a shell command and return stdout/stderr. Use for git, tests, builds, etc."
    )
    parameters = [
        ToolParam(name="command", description="Shell command to execute"),
    ]

    def __init__(self, working_dir: str = ".", timeout: int = 60):
        self._cwd = Path(working_dir).resolve()
        self._timeout = timeout

    async def execute(self, command: str = "", **kw) -> ToolResult:
        # Safety: block destructive commands
        blocked = ["rm -rf /", "mkfs", "dd if=", ":(){", "fork bomb"]
        for b in blocked:
            if b in command:
                return ToolResult(output="", success=False, error=f"Blocked dangerous command: {b}")

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=self._timeout,
                cwd=str(self._cwd),
                env={**os.environ, "TERM": "dumb"},
            )
            output = result.stdout
            if result.stderr:
                output += f"\nSTDERR:\n{result.stderr}" if output else result.stderr
            output = output.strip()

            # Limit output
            lines = output.splitlines()
            if len(lines) > 100:
                output = "\n".join(lines[:100]) + f"\n... ({len(lines) - 100} more lines)"

            return ToolResult(
                output=output or "(no output)",
                success=result.returncode == 0,
                error=f"Exit code: {result.returncode}" if result.returncode != 0 else None,
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                output="", success=False, error=f"Command timed out after {self._timeout}s"
            )
        except Exception as exc:
            return ToolResult(output="", success=False, error=str(exc))


class HttpGetTool(Tool):
    name = "http_get"
    description = "Fetch a URL and return the response body. Useful for checking APIs and docs."
    parameters = [
        ToolParam(name="url", description="URL to fetch"),
    ]

    async def execute(self, url: str = "", **kw) -> ToolResult:
        import httpx

        try:
            async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                text = resp.text
                if len(text) > 5000:
                    text = text[:5000] + f"\n... (truncated, {len(resp.text)} total chars)"
                return ToolResult(output=text)
        except Exception as exc:
            return ToolResult(output="", success=False, error=str(exc))


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------


class ToolRegistry:
    """Manages available tools for agents."""

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def all_tools(self) -> list[Tool]:
        return list(self._tools.values())

    def schemas(self) -> list[dict]:
        """Return OpenAI-compatible tool schemas for all registered tools."""
        return [t.schema() for t in self._tools.values()]

    def names(self) -> list[str]:
        return list(self._tools.keys())


def create_default_registry(working_dir: str = ".") -> ToolRegistry:
    """Create a registry with all built-in tools."""
    registry = ToolRegistry()
    registry.register(ReadFileTool(working_dir))
    registry.register(WriteFileTool(working_dir))
    registry.register(ListDirectoryTool(working_dir))
    registry.register(GrepTool(working_dir))
    registry.register(ShellTool(working_dir))
    registry.register(HttpGetTool())
    return registry
