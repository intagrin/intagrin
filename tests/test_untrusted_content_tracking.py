"""The lethal-trifecta guardrail: untrusted content (RAG retrieval, most MCP/OpenAPI tool
results) + access to state + a way to act on it is the concrete shape most real prompt-injection
incidents take. IntaGrin can't detect injection itself, but it can track provenance — the moment
any tool call flagged untrusted_output=true succeeds, state["_untrusted_content_ingested"] goes
true for the rest of the session, and any tool's `available_when` can reference that bare state
key directly (e.g. "not _untrusted_content_ingested") to withhold itself once contamination has
happened. See LocalToolConfig.untrusted_output / MCPToolConfig.untrusted_output /
OpenAPIToolConfig.untrusted_output in config/schema.py, and the corresponding
compiler/verifier.py advisory check (tests/test_verifier.py).
"""

import asyncio
from unittest.mock import AsyncMock, patch

from intagrin.compiler.parser import ExecutionGraph
from intagrin.config.schema import (
    AgentConfig,
    AppConfig,
    LocalToolConfig,
    MCPToolConfig,
    MemoryConfig,
    ModelConfig,
    OpenAPIToolConfig,
)
from intagrin.runtime.engine import RuntimeEngine


def test_untrusted_output_defaults_to_false_for_local_and_true_for_mcp_and_openapi():
    assert LocalToolConfig(name="t", module="m").untrusted_output is False
    assert MCPToolConfig(name="t", type="mcp", command="npx", args=[]).untrusted_output is True
    assert OpenAPIToolConfig(name="t", type="openapi", url="https://example.com/openapi.json").untrusted_output is True


def _graph(tools):
    config = AppConfig(
        version="1.0",
        name="trifecta-engine-test",
        default_agent="assistant",
        model=ModelConfig(primary="mock/model"),
        memory=MemoryConfig(type="sqlite"),
        agents={"assistant": AgentConfig(tools=tools)},
    )
    return ExecutionGraph(config, {})


def test_untrusted_output_local_tool_call_sets_the_session_flag(tmp_path):
    def fetch_external(url: str) -> str:
        return "<content from the outside world>"

    async def _run():
        graph = _graph(
            [LocalToolConfig(name="fetch_external", module="unused", untrusted_output=True)]
        )
        engine = RuntimeEngine(graph=graph, project_dir=tmp_path, session_id="s1")
        engine.local_tools["fetch_external"] = fetch_external
        engine.untrusted_tools.add("fetch_external")
        await engine.initialize()
        engine.active_agent_name = "assistant"

        assert engine.state["_untrusted_content_ingested"] is False
        await engine.execute_tool("fetch_external", {"url": "http://x"}, interactive=False)
        assert engine.state["_untrusted_content_ingested"] is True

    asyncio.run(_run())


def test_trusted_local_tool_call_does_not_set_the_flag(tmp_path):
    def add(a: int, b: int) -> str:
        return str(a + b)

    async def _run():
        graph = _graph([LocalToolConfig(name="add", module="unused")])
        engine = RuntimeEngine(graph=graph, project_dir=tmp_path, session_id="s2")
        engine.local_tools["add"] = add
        await engine.initialize()
        engine.active_agent_name = "assistant"

        await engine.execute_tool("add", {"a": 1, "b": 2}, interactive=False)
        assert engine.state["_untrusted_content_ingested"] is False

    asyncio.run(_run())


def test_available_when_gate_hides_a_sensitive_tool_once_untrusted_content_is_ingested(tmp_path):
    """End-to-end: a send_email tool gated behind `not _untrusted_content_ingested` is offered
    to the model up front, then structurally disappears from the tool schema list the moment a
    prior turn's untrusted-output tool call has succeeded — not just a documented convention, an
    enforced one."""

    def fetch_external(url: str) -> str:
        return "<content from the outside world>"

    def send_email(to: str, body: str) -> str:
        return f"sent to {to}"

    async def _run():
        graph = _graph(
            [
                LocalToolConfig(name="fetch_external", module="unused", untrusted_output=True),
                LocalToolConfig(
                    name="send_email",
                    module="unused",
                    available_when="not _untrusted_content_ingested",
                ),
            ]
        )
        engine = RuntimeEngine(graph=graph, project_dir=tmp_path, session_id="s3")
        await engine.initialize()
        engine.active_agent_name = "assistant"
        agent_cfg = engine.graph.config.agents["assistant"]

        # _load_tool_config's real module load fails for these test stand-ins (module="unused");
        # inject the callables and their schemas directly, same as test_available_when.py's
        # _inject_tools helper.
        for tool_name, func in (
            ("fetch_external", fetch_external),
            ("send_email", send_email),
        ):
            engine.local_tools[tool_name] = func
            engine.global_tool_schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "description": "test tool",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            )
        engine.untrusted_tools.add("fetch_external")

        active_before = await engine._get_active_tools(agent_cfg)
        assert "send_email" in {t["function"]["name"] for t in active_before}

        await engine.execute_tool("fetch_external", {"url": "http://x"}, interactive=False)

        active_after = await engine._get_active_tools(agent_cfg)
        assert "send_email" not in {t["function"]["name"] for t in active_after}

    asyncio.run(_run())


def test_load_tool_config_marks_mcp_tools_untrusted_by_default(tmp_path):
    async def _run():
        graph = _graph([])
        engine = RuntimeEngine(graph=graph, project_dir=tmp_path, session_id="s4")
        with (
            patch.object(engine.mcp_manager, "connect", new_callable=AsyncMock),
            patch.object(
                engine.mcp_manager,
                "get_server_tool_schemas",
                new_callable=AsyncMock,
                return_value=[{"type": "function", "function": {"name": "remote_search"}}],
            ),
        ):
            await engine._load_tool_config(
                MCPToolConfig(name="docs_server", type="mcp", command="npx", args=["-y", "docs-mcp"])
            )
        assert "remote_search" in engine.untrusted_tools

    asyncio.run(_run())


def test_load_tool_config_respects_mcp_untrusted_output_false(tmp_path):
    async def _run():
        graph = _graph([])
        engine = RuntimeEngine(graph=graph, project_dir=tmp_path, session_id="s5")
        with (
            patch.object(engine.mcp_manager, "connect", new_callable=AsyncMock),
            patch.object(
                engine.mcp_manager,
                "get_server_tool_schemas",
                new_callable=AsyncMock,
                return_value=[{"type": "function", "function": {"name": "internal_search"}}],
            ),
        ):
            await engine._load_tool_config(
                MCPToolConfig(
                    name="internal_server",
                    type="mcp",
                    command="npx",
                    args=["-y", "internal-mcp"],
                    untrusted_output=False,
                )
            )
        assert "internal_search" not in engine.untrusted_tools

    asyncio.run(_run())


def test_load_tool_config_marks_openapi_tools_untrusted_by_default(tmp_path):
    async def _run():
        graph = _graph([])
        engine = RuntimeEngine(graph=graph, project_dir=tmp_path, session_id="s6")
        fake_schemas = [{"type": "function", "function": {"name": "weather_lookup"}}]
        fake_funcs = {"weather_lookup": lambda **kw: "sunny"}
        with patch(
            "intagrin.runtime.engine.load_openapi_tools",
            new_callable=AsyncMock,
            return_value=(fake_schemas, fake_funcs),
        ):
            await engine._load_tool_config(
                OpenAPIToolConfig(name="weather", type="openapi", url="https://example.com/openapi.json")
            )
        assert "weather_lookup" in engine.untrusted_tools

    asyncio.run(_run())
