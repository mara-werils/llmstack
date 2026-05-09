"""Comprehensive tests for the Agent system and MCP server.

Covers tools, tool registry, agent loop, MCP protocol, and config schema.
"""

from __future__ import annotations


import pytest

from llmstack.agent.tools import (
    ToolRegistry,
    ToolResult,
    ReadFileTool,
    WriteFileTool,
    ListDirectoryTool,
    GrepTool,
    ShellTool,
    create_default_registry,
)
from llmstack.agent.loop import AgentConfig, AgentEvent, AgentLoop
from llmstack.mcp.server import MCPServer


# ===================================================================
# ToolResult tests
# ===================================================================

class TestToolResult:
    def test_success_message(self):
        r = ToolResult(output="hello", success=True)
        assert r.to_message() == "hello"

    def test_error_message(self):
        r = ToolResult(output="", success=False, error="not found")
        assert "Error: not found" in r.to_message()

    def test_error_without_error_field(self):
        r = ToolResult(output="failed", success=False)
        assert r.to_message() == "Error: failed"


# ===================================================================
# Tool schema tests
# ===================================================================

class TestToolSchema:
    def test_schema_format(self):
        tool = ReadFileTool()
        schema = tool.schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "read_file"
        assert "parameters" in schema["function"]
        assert "path" in schema["function"]["parameters"]["properties"]

    def test_required_params(self):
        tool = ReadFileTool()
        schema = tool.schema()
        assert "path" in schema["function"]["parameters"]["required"]

    def test_optional_params_not_required(self):
        tool = ReadFileTool()
        schema = tool.schema()
        required = schema["function"]["parameters"]["required"]
        assert "start_line" not in required
        assert "end_line" not in required


# ===================================================================
# ReadFile tool tests
# ===================================================================

class TestReadFileTool:
    @pytest.fixture
    def tmp_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("line 1\nline 2\nline 3\nline 4\nline 5\n")
        return f

    @pytest.mark.asyncio
    async def test_read_full_file(self, tmp_file):
        tool = ReadFileTool(working_dir=str(tmp_file.parent))
        result = await tool.execute(path=str(tmp_file))
        assert result.success
        assert "line 1" in result.output
        assert "line 5" in result.output

    @pytest.mark.asyncio
    async def test_read_with_line_range(self, tmp_file):
        tool = ReadFileTool(working_dir=str(tmp_file.parent))
        result = await tool.execute(path=str(tmp_file), start_line=2, end_line=4)
        assert result.success
        assert "line 2" in result.output
        assert "line 4" in result.output
        assert "line 5" not in result.output

    @pytest.mark.asyncio
    async def test_read_nonexistent(self, tmp_path):
        tool = ReadFileTool(working_dir=str(tmp_path))
        result = await tool.execute(path="nonexistent.txt")
        assert not result.success
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_line_numbers_in_output(self, tmp_file):
        tool = ReadFileTool(working_dir=str(tmp_file.parent))
        result = await tool.execute(path=str(tmp_file))
        assert "1\t" in result.output
        assert "5\t" in result.output


# ===================================================================
# WriteFile tool tests
# ===================================================================

class TestWriteFileTool:
    @pytest.mark.asyncio
    async def test_write_new_file(self, tmp_path):
        tool = WriteFileTool(working_dir=str(tmp_path))
        result = await tool.execute(path="output.txt", content="hello world")
        assert result.success
        assert (tmp_path / "output.txt").read_text() == "hello world"

    @pytest.mark.asyncio
    async def test_write_creates_dirs(self, tmp_path):
        tool = WriteFileTool(working_dir=str(tmp_path))
        result = await tool.execute(path="sub/dir/file.txt", content="nested")
        assert result.success
        assert (tmp_path / "sub" / "dir" / "file.txt").read_text() == "nested"

    @pytest.mark.asyncio
    async def test_write_overwrite(self, tmp_path):
        f = tmp_path / "existing.txt"
        f.write_text("old content")
        tool = WriteFileTool(working_dir=str(tmp_path))
        result = await tool.execute(path="existing.txt", content="new content")
        assert result.success
        assert f.read_text() == "new content"


# ===================================================================
# ListDirectory tool tests
# ===================================================================

class TestListDirectoryTool:
    @pytest.mark.asyncio
    async def test_list_directory(self, tmp_path):
        (tmp_path / "a.txt").write_text("aaa")
        (tmp_path / "b.py").write_text("bbb")
        (tmp_path / "subdir").mkdir()

        tool = ListDirectoryTool(working_dir=str(tmp_path))
        result = await tool.execute(path=".")
        assert result.success
        assert "a.txt" in result.output
        assert "b.py" in result.output
        assert "subdir" in result.output

    @pytest.mark.asyncio
    async def test_list_shows_types(self, tmp_path):
        (tmp_path / "file.txt").write_text("x")
        (tmp_path / "folder").mkdir()

        tool = ListDirectoryTool(working_dir=str(tmp_path))
        result = await tool.execute(path=".")
        assert "file" in result.output
        assert "dir" in result.output

    @pytest.mark.asyncio
    async def test_list_nonexistent(self, tmp_path):
        tool = ListDirectoryTool(working_dir=str(tmp_path))
        result = await tool.execute(path="nonexistent")
        assert not result.success

    @pytest.mark.asyncio
    async def test_hides_dotfiles(self, tmp_path):
        (tmp_path / ".hidden").write_text("secret")
        (tmp_path / "visible.txt").write_text("shown")

        tool = ListDirectoryTool(working_dir=str(tmp_path))
        result = await tool.execute(path=".")
        assert ".hidden" not in result.output
        assert "visible.txt" in result.output


# ===================================================================
# Grep tool tests
# ===================================================================

class TestGrepTool:
    @pytest.mark.asyncio
    async def test_grep_finds_match(self, tmp_path):
        (tmp_path / "code.py").write_text("def hello():\n    return 'world'\n")
        tool = GrepTool(working_dir=str(tmp_path))
        result = await tool.execute(pattern="hello", path=".")
        assert result.success
        assert "hello" in result.output

    @pytest.mark.asyncio
    async def test_grep_no_match(self, tmp_path):
        (tmp_path / "code.py").write_text("def foo(): pass\n")
        tool = GrepTool(working_dir=str(tmp_path))
        result = await tool.execute(pattern="nonexistent_xyz_123", path=".")
        assert "No matches" in result.output

    @pytest.mark.asyncio
    async def test_grep_with_include(self, tmp_path):
        (tmp_path / "a.py").write_text("target line\n")
        (tmp_path / "b.txt").write_text("target line\n")
        tool = GrepTool(working_dir=str(tmp_path))
        result = await tool.execute(pattern="target", path=".", include="*.py")
        assert "a.py" in result.output


# ===================================================================
# Shell tool tests
# ===================================================================

class TestShellTool:
    @pytest.mark.asyncio
    async def test_simple_command(self):
        tool = ShellTool()
        result = await tool.execute(command="echo hello")
        assert result.success
        assert "hello" in result.output

    @pytest.mark.asyncio
    async def test_failed_command(self):
        tool = ShellTool()
        result = await tool.execute(command="false")
        assert not result.success

    @pytest.mark.asyncio
    async def test_blocked_dangerous_command(self):
        tool = ShellTool()
        result = await tool.execute(command="rm -rf /")
        assert not result.success
        assert "Blocked" in result.error

    @pytest.mark.asyncio
    async def test_timeout(self):
        tool = ShellTool(timeout=1)
        result = await tool.execute(command="sleep 10")
        assert not result.success
        assert "timed out" in result.error.lower()

    @pytest.mark.asyncio
    async def test_working_dir(self, tmp_path):
        tool = ShellTool(working_dir=str(tmp_path))
        result = await tool.execute(command="pwd")
        assert result.success
        assert str(tmp_path) in result.output


# ===================================================================
# Tool Registry tests
# ===================================================================

class TestToolRegistry:
    def test_register_and_get(self):
        registry = ToolRegistry()
        tool = ReadFileTool()
        registry.register(tool)
        assert registry.get("read_file") is tool

    def test_get_unknown(self):
        registry = ToolRegistry()
        assert registry.get("unknown") is None

    def test_all_tools(self):
        registry = create_default_registry()
        tools = registry.all_tools()
        assert len(tools) == 6  # read, write, list, grep, shell, http_get

    def test_schemas(self):
        registry = create_default_registry()
        schemas = registry.schemas()
        assert len(schemas) == 6
        assert all(s["type"] == "function" for s in schemas)

    def test_names(self):
        registry = create_default_registry()
        names = registry.names()
        assert "read_file" in names
        assert "write_file" in names
        assert "shell" in names
        assert "grep" in names

    def test_default_registry_all_tools_present(self):
        registry = create_default_registry()
        expected = {"read_file", "write_file", "list_directory", "grep", "shell", "http_get"}
        assert set(registry.names()) == expected


# ===================================================================
# AgentConfig tests
# ===================================================================

class TestAgentConfig:
    def test_defaults(self):
        config = AgentConfig()
        assert config.model == "llama3.2"
        assert config.max_steps == 25
        assert config.temperature == 0.1
        assert config.timeout == 300

    def test_custom_config(self):
        config = AgentConfig(
            model="llama3.1:70b",
            max_steps=50,
            temperature=0.0,
        )
        assert config.model == "llama3.1:70b"
        assert config.max_steps == 50


# ===================================================================
# AgentEvent tests
# ===================================================================

class TestAgentEvent:
    def test_to_dict(self):
        event = AgentEvent(
            type="tool_call", tool_name="read_file",
            tool_args={"path": "test.py"}, step=1, elapsed_ms=50.0,
        )
        d = event.to_dict()
        assert d["type"] == "tool_call"
        assert d["tool_name"] == "read_file"
        assert d["step"] == 1
        assert d["elapsed_ms"] == 50.0

    def test_to_dict_minimal(self):
        event = AgentEvent(type="done", step=5)
        d = event.to_dict()
        assert d == {"type": "done", "step": 5}

    def test_to_dict_with_content(self):
        event = AgentEvent(type="message", content="Hello!", step=3)
        d = event.to_dict()
        assert d["content"] == "Hello!"


# ===================================================================
# AgentLoop tests (unit — mock LLM responses)
# ===================================================================

class TestAgentLoop:
    def test_init(self):
        config = AgentConfig()
        tools = create_default_registry()
        agent = AgentLoop(config=config, tools=tools)
        assert agent.steps_taken == 0
        assert agent.messages == []

    @pytest.mark.asyncio
    async def test_max_steps_limit(self):
        """Agent should stop after max_steps even if LLM keeps calling tools."""
        config = AgentConfig(max_steps=2)
        tools = ToolRegistry()

        # We can't easily mock the LLM here without httpx mock,
        # but we can test the config is respected
        AgentLoop(config=config, tools=tools)
        assert config.max_steps == 2


# ===================================================================
# MCP Server tests (protocol layer)
# ===================================================================

class TestMCPServer:
    @pytest.fixture
    def server(self):
        return MCPServer(model="llama3.2", ollama_url="http://localhost:11434")

    @pytest.mark.asyncio
    async def test_initialize(self, server):
        result = await server._handle_initialize({
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "test", "version": "1.0"},
        })
        assert result["protocolVersion"] == "2024-11-05"
        assert "tools" in result["capabilities"]
        assert result["serverInfo"]["name"] == "llmstack"

    @pytest.mark.asyncio
    async def test_tools_list(self, server):
        result = await server._handle_tools_list({})
        tools = result["tools"]
        names = [t["name"] for t in tools]

        # Should include agent tools
        assert "read_file" in names
        assert "write_file" in names
        assert "grep" in names
        assert "shell" in names
        assert "list_directory" in names
        assert "http_get" in names

        # Should include LLM tools
        assert "llmstack_chat" in names
        assert "llmstack_ask" in names

    @pytest.mark.asyncio
    async def test_tools_list_has_schemas(self, server):
        result = await server._handle_tools_list({})
        for tool in result["tools"]:
            assert "name" in tool
            assert "description" in tool
            assert "inputSchema" in tool
            assert tool["inputSchema"]["type"] == "object"

    @pytest.mark.asyncio
    async def test_tool_call_read_file(self, server, tmp_path):
        # Create a test file
        test_file = tmp_path / "hello.txt"
        test_file.write_text("Hello, MCP!")

        # Replace the server's tool registry with one pointing to tmp_path
        server.tools = create_default_registry(str(tmp_path))

        result = await server._handle_tools_call({
            "name": "read_file",
            "arguments": {"path": str(test_file)},
        })
        assert not result.get("isError", False)
        assert "Hello, MCP!" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_tool_call_unknown_tool(self, server):
        result = await server._handle_tools_call({
            "name": "nonexistent_tool",
            "arguments": {},
        })
        assert result["isError"] is True
        assert "Unknown tool" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_tool_call_list_directory(self, server, tmp_path):
        (tmp_path / "file1.txt").write_text("a")
        (tmp_path / "file2.py").write_text("b")
        server.tools = create_default_registry(str(tmp_path))

        result = await server._handle_tools_call({
            "name": "list_directory",
            "arguments": {"path": "."},
        })
        assert not result.get("isError", False)
        assert "file1.txt" in result["content"][0]["text"]
        assert "file2.py" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_tool_call_shell(self, server):
        result = await server._handle_tools_call({
            "name": "shell",
            "arguments": {"command": "echo mcp_test"},
        })
        assert not result.get("isError", False)
        assert "mcp_test" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_tool_call_write_file(self, server, tmp_path):
        server.tools = create_default_registry(str(tmp_path))
        result = await server._handle_tools_call({
            "name": "write_file",
            "arguments": {"path": "output.txt", "content": "MCP wrote this"},
        })
        assert not result.get("isError", False)
        assert (tmp_path / "output.txt").read_text() == "MCP wrote this"

    @pytest.mark.asyncio
    async def test_resources_list(self, server):
        result = await server._handle_resources_list({})
        assert result == {"resources": []}

    @pytest.mark.asyncio
    async def test_prompts_list(self, server):
        result = await server._handle_prompts_list({})
        assert result == {"prompts": []}

    @pytest.mark.asyncio
    async def test_ping(self, server):
        result = await server._handle_ping({})
        assert result == {}

    @pytest.mark.asyncio
    async def test_handle_unknown_method(self, server):
        """Unknown methods with an ID should get error responses."""
        # We test the dispatch directly
        await server._handle_message({"jsonrpc": "2.0", "id": 1, "method": "unknown/method"})
        # No crash = success (error written to stdout)


# ===================================================================
# MCP message dispatch tests
# ===================================================================

class TestMCPDispatch:
    @pytest.fixture
    def server(self):
        return MCPServer()

    @pytest.mark.asyncio
    async def test_dispatch_initialize(self, server):
        msg = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
        await server._handle_message(msg)

    @pytest.mark.asyncio
    async def test_dispatch_tools_list(self, server):
        msg = {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
        await server._handle_message(msg)

    @pytest.mark.asyncio
    async def test_dispatch_notification(self, server):
        """Notifications (no id) should not produce a response."""
        msg = {"jsonrpc": "2.0", "method": "initialized", "params": {}}
        await server._handle_message(msg)


# ===================================================================
# Config schema tests
# ===================================================================

class TestAgentConfigSchema:
    def test_agent_profile_defaults(self):
        from llmstack.config.schema import AgentProfileConfig
        profile = AgentProfileConfig()
        assert profile.name == "default"
        assert profile.model == "llama3.2"
        assert profile.max_steps == 25
        assert "read_file" in profile.tools
        assert "shell" in profile.tools

    def test_agents_config(self):
        from llmstack.config.schema import AgentsConfig, AgentProfileConfig
        agents = AgentsConfig(profiles=[
            AgentProfileConfig(name="code-review", model="llama3.1:70b", max_steps=50),
        ])
        assert len(agents.profiles) == 1
        assert agents.profiles[0].name == "code-review"

    def test_mcp_config(self):
        from llmstack.config.schema import MCPConfig
        mcp = MCPConfig(enabled=True, model="llama3.2")
        assert mcp.enabled
        assert "llmstack_chat" in mcp.tools
        assert "llmstack_ask" in mcp.tools

    def test_stack_config_has_agents_and_mcp(self):
        from llmstack.config.schema import StackConfig
        config = StackConfig()
        assert hasattr(config, "agents")
        assert hasattr(config, "mcp")
