"""Comprehensive tests for agent tools — registry, built-in tools, schemas."""

from __future__ import annotations

from pathlib import Path

import pytest

from llmstack.agent.tools import (
    GrepTool,
    ListDirectoryTool,
    ReadFileTool,
    ShellTool,
    Tool,
    ToolParam,
    ToolRegistry,
    ToolResult,
    WriteFileTool,
    create_default_registry,
)


# ---------------------------------------------------------------------------
# ToolResult
# ---------------------------------------------------------------------------


class TestToolResult:
    def test_success_message(self) -> None:
        r = ToolResult(output="hello", success=True)
        assert r.to_message() == "hello"

    def test_error_message_with_error_field(self) -> None:
        r = ToolResult(output="", success=False, error="something broke")
        assert r.to_message() == "Error: something broke"

    def test_error_message_fallback_to_output(self) -> None:
        r = ToolResult(output="fallback text", success=False)
        assert r.to_message() == "Error: fallback text"

    def test_defaults(self) -> None:
        r = ToolResult(output="x")
        assert r.success is True
        assert r.error is None


# ---------------------------------------------------------------------------
# ToolParam
# ---------------------------------------------------------------------------


class TestToolParam:
    def test_defaults(self) -> None:
        p = ToolParam(name="path")
        assert p.type == "string"
        assert p.required is True

    def test_custom(self) -> None:
        p = ToolParam(name="count", type="integer", description="How many", required=False)
        assert p.type == "integer"
        assert p.required is False


# ---------------------------------------------------------------------------
# Tool.schema
# ---------------------------------------------------------------------------


class TestToolSchema:
    def test_schema_structure(self) -> None:
        class MyTool(Tool):
            name = "my_tool"
            description = "Does things"
            parameters = [
                ToolParam(name="x", type="string", description="X value"),
                ToolParam(name="y", type="integer", description="Y value", required=False),
            ]

            async def execute(self, **kwargs) -> ToolResult:
                return ToolResult(output="ok")

        schema = MyTool().schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "my_tool"
        assert schema["function"]["description"] == "Does things"
        props = schema["function"]["parameters"]["properties"]
        assert "x" in props
        assert "y" in props
        assert props["x"]["type"] == "string"
        assert "x" in schema["function"]["parameters"]["required"]
        assert "y" not in schema["function"]["parameters"]["required"]


# ---------------------------------------------------------------------------
# ToolRegistry
# ---------------------------------------------------------------------------


class TestToolRegistry:
    def test_register_and_get(self) -> None:
        reg = ToolRegistry()

        class Dummy(Tool):
            name = "dummy"
            description = "test"
            parameters = []

            async def execute(self, **kwargs) -> ToolResult:
                return ToolResult(output="ok")

        tool = Dummy()
        reg.register(tool)
        assert reg.get("dummy") is tool
        assert reg.get("nonexistent") is None

    def test_all_tools(self) -> None:
        reg = create_default_registry()
        tools = reg.all_tools()
        assert len(tools) >= 5

    def test_schemas(self) -> None:
        reg = create_default_registry()
        schemas = reg.schemas()
        assert all(s["type"] == "function" for s in schemas)

    def test_names(self) -> None:
        reg = create_default_registry()
        names = reg.names()
        assert "read_file" in names
        assert "write_file" in names
        assert "list_directory" in names
        assert "grep" in names
        assert "shell" in names
        assert "http_get" in names


# ---------------------------------------------------------------------------
# ReadFileTool
# ---------------------------------------------------------------------------


class TestReadFileTool:
    @pytest.mark.asyncio
    async def test_read_existing_file(self, tmp_path: Path) -> None:
        f = tmp_path / "test.txt"
        f.write_text("line1\nline2\nline3")
        tool = ReadFileTool(str(tmp_path))
        result = await tool.execute(path="test.txt")
        assert result.success is True
        assert "line1" in result.output
        assert "line2" in result.output

    @pytest.mark.asyncio
    async def test_read_nonexistent_file(self, tmp_path: Path) -> None:
        tool = ReadFileTool(str(tmp_path))
        result = await tool.execute(path="nope.txt")
        assert result.success is False
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_read_with_line_range(self, tmp_path: Path) -> None:
        f = tmp_path / "test.txt"
        f.write_text("line1\nline2\nline3\nline4\nline5")
        tool = ReadFileTool(str(tmp_path))
        result = await tool.execute(path="test.txt", start_line=2, end_line=4)
        assert result.success is True
        assert "line2" in result.output
        assert "line3" in result.output
        assert "line4" in result.output
        # line1 should NOT be there
        assert "1\tline1" not in result.output

    @pytest.mark.asyncio
    async def test_read_absolute_path(self, tmp_path: Path) -> None:
        f = tmp_path / "abs.txt"
        f.write_text("absolute content")
        tool = ReadFileTool(str(tmp_path))
        result = await tool.execute(path=str(f))
        assert result.success is True
        assert "absolute content" in result.output

    @pytest.mark.asyncio
    async def test_line_numbers_in_output(self, tmp_path: Path) -> None:
        f = tmp_path / "test.txt"
        f.write_text("aaa\nbbb")
        tool = ReadFileTool(str(tmp_path))
        result = await tool.execute(path="test.txt")
        assert "1\t" in result.output


# ---------------------------------------------------------------------------
# WriteFileTool
# ---------------------------------------------------------------------------


class TestWriteFileTool:
    @pytest.mark.asyncio
    async def test_write_new_file(self, tmp_path: Path) -> None:
        tool = WriteFileTool(str(tmp_path))
        result = await tool.execute(path="new.txt", content="hello world")
        assert result.success is True
        assert (tmp_path / "new.txt").read_text() == "hello world"

    @pytest.mark.asyncio
    async def test_write_creates_parent_dirs(self, tmp_path: Path) -> None:
        tool = WriteFileTool(str(tmp_path))
        result = await tool.execute(path="sub/dir/file.txt", content="deep")
        assert result.success is True
        assert (tmp_path / "sub" / "dir" / "file.txt").read_text() == "deep"

    @pytest.mark.asyncio
    async def test_write_overwrites(self, tmp_path: Path) -> None:
        f = tmp_path / "existing.txt"
        f.write_text("old")
        tool = WriteFileTool(str(tmp_path))
        await tool.execute(path="existing.txt", content="new")
        assert f.read_text() == "new"

    @pytest.mark.asyncio
    async def test_write_reports_bytes(self, tmp_path: Path) -> None:
        tool = WriteFileTool(str(tmp_path))
        result = await tool.execute(path="f.txt", content="12345")
        assert "5 bytes" in result.output


# ---------------------------------------------------------------------------
# ListDirectoryTool
# ---------------------------------------------------------------------------


class TestListDirectoryTool:
    @pytest.mark.asyncio
    async def test_list_directory(self, tmp_path: Path) -> None:
        (tmp_path / "file.txt").write_text("x")
        sub = tmp_path / "subdir"
        sub.mkdir()
        tool = ListDirectoryTool(str(tmp_path))
        result = await tool.execute(path=".")
        assert result.success is True
        assert "file.txt" in result.output
        assert "subdir" in result.output

    @pytest.mark.asyncio
    async def test_list_nonexistent_dir(self, tmp_path: Path) -> None:
        tool = ListDirectoryTool(str(tmp_path))
        result = await tool.execute(path="nope")
        assert result.success is False

    @pytest.mark.asyncio
    async def test_hidden_files_excluded(self, tmp_path: Path) -> None:
        (tmp_path / ".hidden").write_text("secret")
        (tmp_path / "visible.txt").write_text("public")
        tool = ListDirectoryTool(str(tmp_path))
        result = await tool.execute(path=".")
        assert ".hidden" not in result.output
        assert "visible.txt" in result.output

    @pytest.mark.asyncio
    async def test_empty_dir(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        tool = ListDirectoryTool(str(tmp_path))
        result = await tool.execute(path="empty")
        assert result.success is True
        assert "empty" in result.output.lower()

    @pytest.mark.asyncio
    async def test_file_sizes_shown(self, tmp_path: Path) -> None:
        (tmp_path / "small.txt").write_text("x")
        tool = ListDirectoryTool(str(tmp_path))
        result = await tool.execute(path=".")
        assert "B" in result.output  # size suffix

    @pytest.mark.asyncio
    async def test_dirs_sorted_first(self, tmp_path: Path) -> None:
        (tmp_path / "z_file.txt").write_text("x")
        (tmp_path / "a_dir").mkdir()
        tool = ListDirectoryTool(str(tmp_path))
        result = await tool.execute(path=".")
        lines = result.output.strip().split("\n")
        assert "dir" in lines[0]


# ---------------------------------------------------------------------------
# ShellTool
# ---------------------------------------------------------------------------


class TestShellTool:
    @pytest.mark.asyncio
    async def test_simple_command(self, tmp_path: Path) -> None:
        tool = ShellTool(str(tmp_path))
        result = await tool.execute(command="echo hello")
        assert result.success is True
        assert "hello" in result.output

    @pytest.mark.asyncio
    async def test_failed_command(self, tmp_path: Path) -> None:
        tool = ShellTool(str(tmp_path))
        result = await tool.execute(command="false")
        assert result.success is False

    @pytest.mark.asyncio
    async def test_blocked_dangerous_command(self, tmp_path: Path) -> None:
        tool = ShellTool(str(tmp_path))
        result = await tool.execute(command="rm -rf /")
        assert result.success is False
        assert "Blocked" in result.error

    @pytest.mark.asyncio
    async def test_timeout(self, tmp_path: Path) -> None:
        tool = ShellTool(str(tmp_path), timeout=1)
        result = await tool.execute(command="sleep 10")
        assert result.success is False
        assert "timed out" in result.error.lower()

    @pytest.mark.asyncio
    async def test_stderr_included(self, tmp_path: Path) -> None:
        tool = ShellTool(str(tmp_path))
        result = await tool.execute(command="echo err >&2")
        assert "err" in result.output

    @pytest.mark.asyncio
    async def test_output_truncation(self, tmp_path: Path) -> None:
        tool = ShellTool(str(tmp_path))
        # Generate >100 lines
        result = await tool.execute(command="seq 1 200")
        assert "more lines" in result.output


# ---------------------------------------------------------------------------
# GrepTool
# ---------------------------------------------------------------------------


class TestGrepTool:
    @pytest.mark.asyncio
    async def test_grep_finds_match(self, tmp_path: Path) -> None:
        (tmp_path / "test.py").write_text("def hello():\n    pass\n")
        tool = GrepTool(str(tmp_path))
        result = await tool.execute(pattern="hello", path=".")
        assert result.success is True
        assert "hello" in result.output

    @pytest.mark.asyncio
    async def test_grep_no_match(self, tmp_path: Path) -> None:
        (tmp_path / "test.py").write_text("x = 1\n")
        tool = GrepTool(str(tmp_path))
        result = await tool.execute(pattern="zzz_nonexistent")
        assert "No matches" in result.output

    @pytest.mark.asyncio
    async def test_grep_with_include(self, tmp_path: Path) -> None:
        (tmp_path / "a.py").write_text("target\n")
        (tmp_path / "a.txt").write_text("target\n")
        tool = GrepTool(str(tmp_path))
        result = await tool.execute(pattern="target", include="*.py")
        assert "a.py" in result.output


# ---------------------------------------------------------------------------
# create_default_registry
# ---------------------------------------------------------------------------


class TestCreateDefaultRegistry:
    def test_all_tools_registered(self) -> None:
        reg = create_default_registry()
        expected = {"read_file", "write_file", "list_directory", "grep", "shell", "http_get"}
        assert set(reg.names()) == expected

    def test_custom_working_dir(self, tmp_path: Path) -> None:
        reg = create_default_registry(str(tmp_path))
        tool = reg.get("read_file")
        assert tool is not None
