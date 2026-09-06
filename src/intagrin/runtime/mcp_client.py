from contextlib import AsyncExitStack
from typing import Any

import mcp.types as types
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from ..errors import IntaGrinError


def _flatten_call_tool_result(result: types.CallToolResult) -> str:
    """Text-only flattening of a CallToolResult — shared by the immediate call_tool path and
    get_task_payload's polled-completion path, since both ultimately produce the same shape of
    result for a tool call (an immediate CallToolResult, or one fetched via tasks/result after a
    claimed call finishes). Non-text content parts are dropped, matching pre-existing behavior."""
    if result.isError:
        return f"Error: {result.content}"
    output = [content.text for content in result.content if content.type == "text"]
    return "\n".join(output)


class MCPToolManager:
    def __init__(self):
        self.exit_stack = AsyncExitStack()
        self.sessions: dict[str, ClientSession] = {}
        self.tool_mappings: dict[str, str] = {}  # tool_name -> server_name

    async def connect(
        self, server_name: str, command: str, args: list[str], env: dict[str, str] | None = None
    ):
        params = StdioServerParameters(command=command, args=args, env=env or None)
        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(params)
        )
        read, write = stdio_transport
        session = await self.exit_stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        self.sessions[server_name] = session

        # Load tools
        mcp_tools = await session.list_tools()
        for t in mcp_tools.tools:
            self.tool_mappings[t.name] = server_name

    async def get_all_tool_schemas(self) -> list[dict[str, Any]]:
        schemas = []
        for server_name in self.sessions:
            schemas.extend(await self.get_server_tool_schemas(server_name))
        return schemas

    async def get_server_tool_schemas(self, server_name: str) -> list[dict[str, Any]]:
        """Return schemas for one MCP server without exposing other servers' tools."""
        session = self.sessions[server_name]
        mcp_tools = await session.list_tools()
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_schema,
                },
            }
            for tool in mcp_tools.tools
        ]

    async def call_tool(self, name: str, args: dict) -> str | dict:
        """Dispatch a tool call. Passes allow_claimed=True so a server that supports the MCP
        Tasks extension (long-running calls) can claim the call instead of blocking until it
        finishes — harmless for any server that doesn't support Tasks, which just returns an
        ordinary CallToolResult as before. A claimed call returns a small sentinel dict instead
        of a flattened string; the caller (RuntimeEngine.execute_tool) is responsible for turning
        that into a non-blocking "task started" tool result and offering check_mcp_task_status."""
        if name not in self.tool_mappings:
            raise IntaGrinError("IG-MCP-001", f"Tool {name} not found in MCP servers")
        server_name = self.tool_mappings[name]
        session = self.sessions[server_name]
        result = await session.call_tool(name, arguments=args, allow_claimed=True)

        if isinstance(result, types.CreateTaskResult):
            return {"_mcp_task": {"task_id": result.task.task_id, "server_name": server_name}}

        return _flatten_call_tool_result(result)

    async def get_task_status(self, server_name: str, task_id: str) -> types.GetTaskResult:
        """Poll a claimed task's status on the same already-connected session — never
        reconnects, since the shared AsyncExitStack ties every server's lifetime together."""
        session = self.sessions[server_name]
        return await session.send_request(
            types.GetTaskRequest(params=types.GetTaskRequestParams(task_id=task_id)),
            types.GetTaskResult,
        )

    async def get_task_payload(self, server_name: str, task_id: str) -> str:
        """Fetch a completed task's result and flatten it the same way an immediate call_tool
        result would be. Per GetTaskPayloadResult's own docstring, the wire payload must be
        validated into the *original* request's result type (CallToolResult, since every task
        this manager creates comes from a tools/call), not GetTaskPayloadResult itself."""
        session = self.sessions[server_name]
        result = await session.send_request(
            types.GetTaskPayloadRequest(params=types.GetTaskPayloadRequestParams(task_id=task_id)),
            types.CallToolResult,
        )
        return _flatten_call_tool_result(result)

    async def cleanup(self):
        await self.exit_stack.aclose()
