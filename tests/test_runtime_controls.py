import asyncio
from pathlib import Path
from unittest.mock import patch

from intagrin.compiler.parser import ExecutionGraph
from intagrin.config.schema import (
    AgentConfig,
    AppConfig,
    CircuitBreakersConfig,
    MemoryConfig,
    ModelConfig,
    RAGConfig,
    RouterConfig,
)
from intagrin.runtime.engine import RuntimeEngine
from intagrin.tracing.console import EventStreamer


def _schema(name: str) -> dict:
    return {
        "type": "function",
        "function": {"name": name, "description": name, "parameters": {}},
    }


def test_tools_are_scoped_and_handoffs_are_exposed():
    async def run():
        config = AppConfig(
            version="1.0",
            name="scope-test",
            default_agent="support",
            model=ModelConfig(primary="mock/model"),
            memory=MemoryConfig(type="buffer"),
            agents={
                "support": AgentConfig(
                    tools=[{"name": "lookup_account"}], handoffs=["billing"]
                ),
                "billing": AgentConfig(tools=[{"name": "issue_refund"}]),
            },
        )
        engine = RuntimeEngine(ExecutionGraph(config, {}), Path.cwd())
        engine.global_tool_schemas = [_schema("lookup_account"), _schema("issue_refund")]
        engine.local_tools = {
            "lookup_account": lambda: "account",
            "issue_refund": lambda: "refund",
        }

        active_names = {
            schema["function"]["name"]
            for schema in await engine._get_active_tools(config.agents["support"])
        }
        assert active_names == {"lookup_account", "transfer_agent"}

        result = await engine.execute_tool("issue_refund", {}, interactive=False)
        assert "not authorized" in result

    asyncio.run(run())


def test_yaml_hyde_configuration_reaches_the_rag_engine(tmp_path):
    async def run():
        config = AppConfig(
            version="1.0",
            name="hyde-test",
            default_agent="assistant",
            model=ModelConfig(primary="mock/model"),
            memory=MemoryConfig(type="buffer"),
            rag=RAGConfig(hyde=True),
            agents={"assistant": AgentConfig()},
        )
        engine = RuntimeEngine(ExecutionGraph(config, {}), tmp_path)
        with patch("intagrin.runtime.rag.VectorRAGEngine") as rag_engine:
            await engine.initialize()
        assert rag_engine.call_args.kwargs["hyde"] is True

    asyncio.run(run())


def test_conditional_router_honours_configured_handoff_limit():
    async def run():
        config = AppConfig(
            version="1.0",
            name="router-limit-test",
            default_agent="triage",
            model=ModelConfig(primary="mock/model"),
            memory=MemoryConfig(type="buffer"),
            circuit_breakers=CircuitBreakersConfig(max_handoffs_per_session=0),
            agents={
                "triage": AgentConfig(
                    routers=[RouterConfig(condition="balance < 0", target="billing")]
                ),
                "billing": AgentConfig(),
            },
        )
        engine = RuntimeEngine(ExecutionGraph(config, {}), Path.cwd())
        await engine.initialize()
        engine.state["balance"] = -1

        await engine._run_agent_turn(interactive=False)

        assert engine.active_agent_name == "triage"
        assert engine.is_transferring is False

    asyncio.run(run())


def test_conditional_router_leaves_a_message_breadcrumb_and_emits_handoff_event():
    """Previously a firing conditional/root router updated active_agent_name directly with zero
    trace in the message history — inta replay and the live monitor dashboard couldn't tell a
    router-driven handoff happened at all, unlike transfer_agent/auto_route which both do leave a
    trace. This is the regression test for both the message breadcrumb and the SSE 'handoff' event."""

    async def run():
        config = AppConfig(
            version="1.0",
            name="router-breadcrumb-test",
            default_agent="triage",
            model=ModelConfig(primary="mock/model"),
            memory=MemoryConfig(type="buffer"),
            agents={
                "triage": AgentConfig(
                    routers=[RouterConfig(condition="balance < 0", target="billing")]
                ),
                "billing": AgentConfig(),
            },
        )
        engine = RuntimeEngine(ExecutionGraph(config, {}), Path.cwd())
        await engine.initialize()
        engine.state["balance"] = -1

        q = EventStreamer.subscribe()
        try:
            route_err = await engine._resolve_routing(config.agents["triage"])
        finally:
            EventStreamer.unsubscribe(q)

        assert route_err is None
        assert engine.active_agent_name == "billing"
        assert engine.is_transferring is True

        breadcrumbs = [
            m for m in engine.messages
            if m.get("role") == "system" and "conditional router" in str(m.get("content", ""))
        ]
        assert len(breadcrumbs) == 1
        assert "billing" in breadcrumbs[0]["content"]

        handoff_events = []
        while not q.empty():
            handoff_events.append(q.get_nowait())
        handoff_events = [e for e in handoff_events if e["type"] == "handoff"]
        assert len(handoff_events) == 1
        assert handoff_events[0]["data"] == {
            "from": "triage",
            "to": "billing",
            "mechanism": "conditional_router",
            "condition": "balance < 0",
        }

    asyncio.run(run())
