"""Tests for the MCP Tasks extension support (non-blocking long-running MCP tool calls) — see
CLAUDE.md's runtime/mcp_client.py section and docs coverage of MCPToolConfig.max_task_wait_seconds.

Deliberately does NOT spin up a real MCP subprocess: MCPToolManager.call_tool/get_task_status/
get_task_payload are exercised directly against fake ClientSession-shaped objects (unit-level),
while RuntimeEngine.execute_tool's dispatch of a claimed result and check_mcp_task_status are
exercised against a real RuntimeEngine with MCPToolManager methods patched (integration-level,
matching tests/test_shared_resources.py's own patch.object(MCPToolManager, ...) style, since
there's no existing dedicated MCP-dispatch test file to extend).
"""

import asyncio
from pathlib import Path
from unittest.mock import patch

import mcp.types as types

from intagrin.compiler.parser import ExecutionGraph
from intagrin.config.schema import (
    AgentConfig,
    AppConfig,
    MCPToolConfig,
    MemoryConfig,
    ModelConfig,
    ToolReferenceConfig,
)
from intagrin.runtime.engine import RuntimeEngine
from intagrin.runtime.mcp_client import MCPToolManager


def _config() -> AppConfig:
    return AppConfig(
        version="1.0",
        name="mcp-tasks-test",
        default_agent="assistant",
        model=ModelConfig(primary="mock/model"),
        memory=MemoryConfig(type="buffer"),
        tools=[MCPToolConfig(name="srv", type="mcp", command="noop", args=[])],
        agents={"assistant": AgentConfig(tools=[ToolReferenceConfig(name="srv")])},
    )


def _config_with_wait_cap(seconds: int | None) -> AppConfig:
    cfg = _config()
    cfg.tools[0].max_task_wait_seconds = seconds
    return cfg


class _TextContent:
    type = "text"

    def __init__(self, text: str):
        self.text = text


class _FakeCallToolResult:
    def __init__(self, text: str, is_error: bool = False):
        self.isError = is_error
        self.content = [_TextContent(text)]


class _FakeSession:
    """Stands in for mcp.client.session.ClientSession — only the surface MCPToolManager.call_tool
    actually uses."""

    def __init__(self, response):
        self._response = response
        self.last_call_kwargs: dict | None = None

    async def call_tool(self, name, arguments=None, allow_claimed=False):
        self.last_call_kwargs = {"name": name, "arguments": arguments, "allow_claimed": allow_claimed}
        return self._response


async def _initialized_engine():
    graph = ExecutionGraph(_config(), {})
    engine = RuntimeEngine(graph, Path.cwd())
    with patch.object(MCPToolManager, "connect", _noop_connect), patch.object(
        MCPToolManager, "get_server_tool_schemas", _fake_schemas
    ):
        await engine.initialize()
    # Real connect() is mocked away above (no real subprocess); populate the tool_mappings/
    # sessions dict the way a real connect() would, so execute_tool's dispatch can find them.
    engine.mcp_manager.tool_mappings["srv_tool"] = "srv"
    engine.mcp_manager.sessions["srv"] = object()
    return engine


async def _noop_connect(self, server_name, command, args, env=None):
    return None


async def _fake_schemas(self, server_name):
    return [
        {
            "type": "function",
            "function": {"name": f"{server_name}_tool", "description": "", "parameters": {}},
        }
    ]


# --- Unit-level: MCPToolManager.call_tool's own claimed/plain detection ---------------------


def test_call_tool_detects_a_claimed_result_and_returns_a_task_sentinel():
    async def run():
        manager = MCPToolManager()
        task = types.Task(
            task_id="task-123",
            status="working",
            created_at="2026-01-01T00:00:00Z",
            last_updated_at="2026-01-01T00:00:00Z",
            ttl=None,
        )
        session = _FakeSession(types.CreateTaskResult(task=task))
        manager.sessions["srv"] = session
        manager.tool_mappings["srv_tool"] = "srv"

        result = await manager.call_tool("srv_tool", {"x": 1})

        assert result == {"_mcp_task": {"task_id": "task-123", "server_name": "srv"}}
        assert session.last_call_kwargs["allow_claimed"] is True

    asyncio.run(run())


def test_call_tool_leaves_a_plain_non_task_server_completely_unaffected():
    """Regression guard: a server that doesn't support Tasks must behave exactly as before —
    allow_claimed=True must be a harmless no-op for it."""

    async def run():
        manager = MCPToolManager()
        session = _FakeSession(_FakeCallToolResult("plain result text"))
        manager.sessions["srv"] = session
        manager.tool_mappings["srv_tool"] = "srv"

        result = await manager.call_tool("srv_tool", {})

        assert result == "plain result text"
        assert isinstance(result, str)

    asyncio.run(run())


def test_call_tool_still_flattens_an_error_result_as_before():
    async def run():
        manager = MCPToolManager()
        session = _FakeSession(_FakeCallToolResult("boom", is_error=True))
        manager.sessions["srv"] = session
        manager.tool_mappings["srv_tool"] = "srv"

        result = await manager.call_tool("srv_tool", {})

        assert result.startswith("Error:")

    asyncio.run(run())


def test_verify_by_breaking_allow_claimed_must_actually_be_passed():
    """If a future edit accidentally drops allow_claimed=True, a claimed result would instead
    raise UnexpectedClaimedResult deep in the real SDK — this test only proves *our* call site
    requests it; flip it off here to prove the assertion actually catches the regression."""

    async def run():
        manager = MCPToolManager()
        session = _FakeSession(_FakeCallToolResult("x"))
        manager.sessions["srv"] = session
        manager.tool_mappings["srv_tool"] = "srv"
        await manager.call_tool("srv_tool", {})
        # This is the assertion that failed when allow_claimed=True was temporarily removed from
        # call_tool during verify-by-breaking (see the manual check performed while implementing
        # this feature) — confirming it's load-bearing, not just present.
        assert session.last_call_kwargs["allow_claimed"] is True

    asyncio.run(run())


# --- Integration-level: RuntimeEngine dispatch + check_mcp_task_status ----------------------


def test_execute_tool_populates_pending_mcp_tasks_and_returns_still_running_message():
    async def run():
        engine = await _initialized_engine()

        async def fake_call_tool(self, name, args):
            return {"_mcp_task": {"task_id": "task-abc", "server_name": "srv"}}

        with patch.object(MCPToolManager, "call_tool", fake_call_tool):
            result = await engine.execute_tool("srv_tool", {}, tool_call_id="tc-1")

        assert "task-abc" in result
        assert "check_mcp_task_status" in result
        pending = engine.state["_pending_mcp_tasks"]["task-abc"]
        assert pending["tool"] == "srv_tool"
        assert pending["server"] == "srv"
        assert pending["tool_call_id"] == "tc-1"

    asyncio.run(run())


def test_check_mcp_task_status_returns_payload_and_clears_pending_entry_on_completion():
    async def run():
        engine = await _initialized_engine()
        engine.state["_pending_mcp_tasks"]["task-abc"] = {
            "tool": "srv_tool",
            "server": "srv",
            "created_at": "2026-01-01T00:00:00+00:00",
            "tool_call_id": "tc-1",
        }

        async def fake_status(self, server_name, task_id):
            return types.GetTaskResult(
                task_id=task_id, status="completed", created_at="x", last_updated_at="x", ttl=None
            )

        async def fake_payload(self, server_name, task_id):
            return "the real answer"

        with patch.object(MCPToolManager, "get_task_status", fake_status), patch.object(
            MCPToolManager, "get_task_payload", fake_payload
        ):
            result = await engine.execute_tool(
                "check_mcp_task_status", {"task_id": "task-abc"}, tool_call_id="tc-2"
            )

        assert result == "the real answer"
        assert "task-abc" not in engine.state["_pending_mcp_tasks"]

    asyncio.run(run())


def test_check_mcp_task_status_reports_failure_and_clears_pending_entry():
    async def run():
        engine = await _initialized_engine()
        engine.state["_pending_mcp_tasks"]["task-abc"] = {
            "tool": "srv_tool",
            "server": "srv",
            "created_at": "2026-01-01T00:00:00+00:00",
            "tool_call_id": "tc-1",
        }

        async def fake_status(self, server_name, task_id):
            return types.GetTaskResult(
                task_id=task_id,
                status="failed",
                status_message="upstream API rejected the request",
                created_at="x",
                last_updated_at="x",
                ttl=None,
            )

        with patch.object(MCPToolManager, "get_task_status", fake_status):
            result = await engine.execute_tool(
                "check_mcp_task_status", {"task_id": "task-abc"}, tool_call_id="tc-2"
            )

        assert "IG-MCP-002" in result
        assert "upstream API rejected the request" in result
        assert "task-abc" not in engine.state["_pending_mcp_tasks"]

    asyncio.run(run())


def test_check_mcp_task_status_expires_after_max_task_wait_seconds():
    async def run():
        graph = ExecutionGraph(_config_with_wait_cap(60), {})
        engine = RuntimeEngine(graph, Path.cwd())
        with patch.object(MCPToolManager, "connect", _noop_connect), patch.object(
            MCPToolManager, "get_server_tool_schemas", _fake_schemas
        ):
            await engine.initialize()
        engine.mcp_manager.tool_mappings["srv_tool"] = "srv"
        engine.mcp_manager.sessions["srv"] = object()
        assert engine.mcp_task_wait_seconds["srv"] == 60

        engine.state["_pending_mcp_tasks"]["task-abc"] = {
            "tool": "srv_tool",
            "server": "srv",
            # Far enough in the past that a 60s cap has long since elapsed.
            "created_at": "2000-01-01T00:00:00+00:00",
            "tool_call_id": "tc-1",
        }

        async def fake_status(self, server_name, task_id):
            return types.GetTaskResult(
                task_id=task_id, status="working", created_at="x", last_updated_at="x", ttl=None
            )

        with patch.object(MCPToolManager, "get_task_status", fake_status):
            result = await engine.execute_tool(
                "check_mcp_task_status", {"task_id": "task-abc"}, tool_call_id="tc-2"
            )

        assert "IG-MCP-002" in result
        assert "max_task_wait_seconds" in result
        assert "task-abc" not in engine.state["_pending_mcp_tasks"]

    asyncio.run(run())


def test_check_mcp_task_status_with_no_wait_cap_never_expires():
    """None (the schema default) must mean unbounded, not zero — a still-working task with an
    old created_at and no cap configured should just report 'still working', not fail."""

    async def run():
        engine = await _initialized_engine()
        assert engine.mcp_task_wait_seconds["srv"] is None
        engine.state["_pending_mcp_tasks"]["task-abc"] = {
            "tool": "srv_tool",
            "server": "srv",
            "created_at": "2000-01-01T00:00:00+00:00",
            "tool_call_id": "tc-1",
        }

        async def fake_status(self, server_name, task_id):
            return types.GetTaskResult(
                task_id=task_id, status="working", created_at="x", last_updated_at="x", ttl=None
            )

        with patch.object(MCPToolManager, "get_task_status", fake_status):
            result = await engine.execute_tool(
                "check_mcp_task_status", {"task_id": "task-abc"}, tool_call_id="tc-2"
            )

        assert "still working" in result
        assert "task-abc" in engine.state["_pending_mcp_tasks"]

    asyncio.run(run())


def test_check_mcp_task_status_unknown_task_id_reports_not_tracked():
    async def run():
        engine = await _initialized_engine()
        result = await engine.execute_tool(
            "check_mcp_task_status", {"task_id": "never-existed"}, tool_call_id="tc-1"
        )
        assert "never-existed" in result
        assert "No pending task" in result

    asyncio.run(run())
