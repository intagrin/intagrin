import asyncio
import os
import time
from pathlib import Path
from unittest.mock import patch

from intagrin.compiler.parser import ExecutionGraph
from intagrin.config.schema import (
    AgentConfig,
    AppConfig,
    MCPToolConfig,
    MemoryConfig,
    ModelConfig,
)
from intagrin.runtime.engine import RuntimeEngine
from intagrin.runtime.mcp_client import MCPToolManager
from intagrin.runtime.shared_resources import SharedResourcesCache


def _config(name: str) -> AppConfig:
    return AppConfig(
        version="1.0",
        name=name,
        default_agent="assistant",
        model=ModelConfig(primary="mock/model"),
        memory=MemoryConfig(type="buffer"),
        agents={"assistant": AgentConfig()},
    )


def test_shared_resources_cache_reuses_pooled_engine_across_requests(tmp_path):
    """Two sequential /chat-style requests against the same project must not rebuild MCP
    connections/tool schemas/prompts twice — the second call reuses the SharedResources built
    by the first, verified by counting real RuntimeEngine.initialize() invocations."""

    async def run():
        graph = ExecutionGraph(_config("pool-test"), {})
        cache = SharedResourcesCache()

        original_initialize = RuntimeEngine.initialize
        call_count = {"n": 0}

        async def counting_initialize(self):
            call_count["n"] += 1
            await original_initialize(self)

        with patch.object(RuntimeEngine, "initialize", counting_initialize):
            first = await cache.get(graph, tmp_path)
            second = await cache.get(graph, tmp_path)

        assert call_count["n"] == 1
        assert first is second

    asyncio.run(run())


def test_shared_resources_cache_rebuilds_after_ai_yaml_changes(tmp_path):
    """Editing ai.yaml must invalidate the pool so config changes take effect — the cache can't
    be allowed to serve stale tool schemas/prompts forever."""

    async def run():
        graph = ExecutionGraph(_config("pool-invalidate-test"), {})
        cache = SharedResourcesCache()

        ai_yaml = tmp_path / "ai.yaml"
        ai_yaml.write_text("version: '1.0'\n")

        first = await cache.get(graph, tmp_path)

        newer = os.path.getmtime(ai_yaml) + 5
        os.utime(ai_yaml, (newer, newer))

        second = await cache.get(graph, tmp_path)

        assert first is not second

    asyncio.run(run())


def test_engine_reusing_shared_resources_still_binds_state_tools_per_session():
    """read_state/write_state must stay bound to *this* engine's own state even when the rest of
    local_tools comes from a pooled builder engine — otherwise two sessions sharing a pool would
    read/write each other's state."""

    async def run():
        graph = ExecutionGraph(_config("state-binding-test"), {})
        builder = RuntimeEngine(graph, Path.cwd())
        await builder.initialize()
        shared = builder._as_shared_resources()

        engine = RuntimeEngine(graph, Path.cwd(), shared_resources=shared)
        await engine.initialize()

        assert engine.local_tools["read_state"] == engine.read_state
        assert engine.local_tools["write_state"] == engine.write_state
        assert engine.local_tools["read_state"] != builder.read_state

        engine.state["k"] = "engine-value"
        assert engine.read_state("k") == "engine-value"
        assert builder.read_state("k") == "Key not found"

    asyncio.run(run())


def test_initialize_connects_mcp_servers_concurrently_not_sequentially():
    """initialize() previously awaited mcp_manager.connect() one server at a time, so N slow
    providers cost sum(all) instead of max(slowest). Simulate 3 servers that each take ~0.2s to
    connect and assert total initialize() time is well under the 0.6s a sequential loop would
    take — proving the loads now run concurrently via asyncio.gather."""

    async def run():
        config = _config("concurrency-test")
        config.tools = [
            MCPToolConfig(name=f"srv{i}", type="mcp", command="noop", args=[])
            for i in range(3)
        ]
        graph = ExecutionGraph(config, {})
        engine = RuntimeEngine(graph, Path.cwd())

        async def slow_connect(self, server_name, command, args, env=None):
            await asyncio.sleep(0.2)
            self.sessions[server_name] = object()
            self.tool_mappings[f"{server_name}_tool"] = server_name

        async def fake_schemas(self, server_name):
            return [
                {
                    "type": "function",
                    "function": {"name": f"{server_name}_tool", "description": "", "parameters": {}},
                }
            ]

        with patch.object(MCPToolManager, "connect", slow_connect), patch.object(
            MCPToolManager, "get_server_tool_schemas", fake_schemas
        ):
            start = time.monotonic()
            await engine.initialize()
            elapsed = time.monotonic() - start

        assert elapsed < 0.45, f"expected concurrent connects (~0.2s), took {elapsed:.2f}s"
        connected_tools = {s["function"]["name"] for s in engine.global_tool_schemas}
        assert {"srv0_tool", "srv1_tool", "srv2_tool"} <= connected_tools

    asyncio.run(run())
