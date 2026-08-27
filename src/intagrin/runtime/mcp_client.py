from contextlib import AsyncExitStack
from typing import Any

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from ..errors import IntaGrinError


class MCPToolManager:
    def __init__(self):
        self.exit_stack = AsyncExitStack()
        self.sessions: dict[str, ClientSession] = {}
        self.tool_mappings: dict[str, str] = {}  # tool_name -> server_name

    async def connect(self, server_name: str, command: str, args: list[str]):
        params = StdioServerParameters(command=command, args=args)
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

    async def call_tool(self, name: str, args: dict) -> str:
        if name not in self.tool_mappings:
            raise IntaGrinError("IG-MCP-001", f"Tool {name} not found in MCP servers")
        server_name = self.tool_mappings[name]
        session = self.sessions[server_name]
        result = await session.call_tool(name, arguments=args)

        if result.isError:
            return f"Error: {result.content}"

        # Format the result content
        output = []
        for content in result.content:
            if content.type == "text":
                output.append(content.text)
        return "\n".join(output)

    async def cleanup(self):
        await self.exit_stack.aclose()
