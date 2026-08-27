import asyncio

from intagrin.compiler.parser import ExecutionGraph
from intagrin.config.schema import (
    AgentConfig,
    AppConfig,
    LocalToolConfig,
    MemoryConfig,
    ModelConfig,
    ToolReferenceConfig,
)
from intagrin.runtime.engine import RuntimeEngine


def _mock_graph(*, available_when="research_done == True", use_tool_reference=False):
    """A `planner` agent with an unrestricted `create_itinerary` tool and a `book_flight` tool
    gated behind `available_when` — mirrors the real research-then-book pattern this feature was
    built for. `use_tool_reference` switches book_flight to the ToolReferenceConfig shape (a
    root-level tool referenced by name), since that's the shape actually used in practice for a
    per-agent `tools:` list."""
    book_flight_entry = (
        ToolReferenceConfig(name="book_flight", available_when=available_when)
        if use_tool_reference
        else LocalToolConfig(
            name="book_flight", module="unused", available_when=available_when
        )
    )
    config = AppConfig(
        version="1.0",
        name="available-when-test",
        default_agent="planner",
        model=ModelConfig(primary="mock/model"),
        memory=MemoryConfig(type="buffer"),
        tools=(
            [LocalToolConfig(name="book_flight", module="unused")]
            if use_tool_reference
            else []
        ),
        agents={
            "planner": AgentConfig(
                description="Plans trips",
                tools=[
                    LocalToolConfig(name="create_itinerary", module="unused"),
                    book_flight_entry,
                ],
            ),
        },
    )
    return ExecutionGraph(config, {})


def _inject_tools(engine, names):
    for name in names:
        engine.local_tools[name] = lambda **kwargs: f"{kwargs}"
        engine.global_tool_schemas.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": "test tool",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        )


async def _init_engine(tmp_path, graph):
    engine = RuntimeEngine(graph=graph, project_dir=tmp_path, session_id="s1")
    await engine.initialize()
    engine.active_agent_name = "planner"
    _inject_tools(engine, ["create_itinerary", "book_flight"])
    return engine


def test_tool_hidden_from_schema_when_condition_is_false(tmp_path):
    async def _run():
        engine = await _init_engine(tmp_path, _mock_graph())
        agent_cfg = engine.graph.config.agents["planner"]

        active_tools = await engine._get_active_tools(agent_cfg)
        names = {t["function"]["name"] for t in active_tools}

        assert "create_itinerary" in names  # unrestricted, always present
        assert "book_flight" not in names  # gated, condition not yet true

    asyncio.run(_run())


def test_tool_appears_once_condition_becomes_true(tmp_path):
    async def _run():
        engine = await _init_engine(tmp_path, _mock_graph())
        agent_cfg = engine.graph.config.agents["planner"]
        engine.state["research_done"] = True

        active_tools = await engine._get_active_tools(agent_cfg)
        names = {t["function"]["name"] for t in active_tools}

        assert "book_flight" in names

    asyncio.run(_run())


def test_tool_reference_config_shape_also_respects_available_when(tmp_path):
    """The practically-important case: a per-agent tools: entry is usually a name-reference to a
    root-level tool declaration, not an inline LocalToolConfig — available_when must work there
    too, not only when a tool is declared inline."""

    async def _run():
        engine = await _init_engine(tmp_path, _mock_graph(use_tool_reference=True))
        agent_cfg = engine.graph.config.agents["planner"]

        names_before = {
            t["function"]["name"] for t in await engine._get_active_tools(agent_cfg)
        }
        assert "book_flight" not in names_before

        engine.state["research_done"] = True
        names_after = {
            t["function"]["name"] for t in await engine._get_active_tools(agent_cfg)
        }
        assert "book_flight" in names_after

    asyncio.run(_run())


def test_execute_tool_still_rejects_a_gated_call_even_if_the_schema_check_is_bypassed(tmp_path):
    """Defense in depth: even a direct execute_tool call (bypassing whatever built the schema
    list the model saw) must be rejected while the condition is false — the schema filter is
    never trusted alone, the same principle already applied to tool_pool and every other
    schema-driven gate in this codebase."""

    async def _run():
        engine = await _init_engine(tmp_path, _mock_graph())

        assert not engine._is_tool_allowed_for_active_agent("book_flight")
        result = await engine.execute_tool("book_flight", {}, interactive=False)
        assert "not authorized" in result

        engine.state["research_done"] = True
        assert engine._is_tool_allowed_for_active_agent("book_flight")
        result = await engine.execute_tool("book_flight", {}, interactive=False)
        assert "not authorized" not in result

    asyncio.run(_run())


def test_malformed_available_when_condition_fails_closed(tmp_path):
    """An unparseable condition must hide/reject the tool, not silently grant access — the
    opposite default from routers[].condition, which fails open and just skips a broken router.
    The whole point of this gate is to withhold a tool until it should be used."""

    async def _run():
        engine = await _init_engine(
            tmp_path, _mock_graph(available_when="state.get('research_done')")
        )
        agent_cfg = engine.graph.config.agents["planner"]

        active_tools = await engine._get_active_tools(agent_cfg)
        names = {t["function"]["name"] for t in active_tools}
        assert "book_flight" not in names

        assert not engine._is_tool_allowed_for_active_agent("book_flight")

    asyncio.run(_run())


def test_no_available_when_means_always_available_todays_behavior(tmp_path):
    async def _run():
        engine = await _init_engine(tmp_path, _mock_graph(available_when=None))
        agent_cfg = engine.graph.config.agents["planner"]

        names = {t["function"]["name"] for t in await engine._get_active_tools(agent_cfg)}
        assert "book_flight" in names
        assert engine._is_tool_allowed_for_active_agent("book_flight")

    asyncio.run(_run())


def test_missing_state_key_hides_the_tool_quietly_not_as_a_logged_error(tmp_path):
    """Regression test found live against a real project: a condition referencing a state key
    that simply hasn't been set yet (the normal, expected shape of an available_when condition's
    early turns — e.g. no research has happened yet in a fresh session) must not spam
    Tracer.log_error on every single turn until it's set. Genuine syntax errors still must."""
    from unittest.mock import patch

    async def _run():
        engine = await _init_engine(tmp_path, _mock_graph())  # research_done never set
        agent_cfg = engine.graph.config.agents["planner"]

        with patch("intagrin.runtime.engine.Tracer.log_error") as mock_log_error:
            names = {t["function"]["name"] for t in await engine._get_active_tools(agent_cfg)}
            assert "book_flight" not in names
            mock_log_error.assert_not_called()

    asyncio.run(_run())


def test_genuinely_malformed_condition_still_logs_an_error(tmp_path):
    """Contrast with the test above — an actual grammar violation (not just a not-yet-set key)
    must still be logged, since that one really is a misconfiguration worth a developer's
    attention."""
    from unittest.mock import patch

    async def _run():
        engine = await _init_engine(
            tmp_path, _mock_graph(available_when="state.get('research_done')")
        )
        agent_cfg = engine.graph.config.agents["planner"]

        with patch("intagrin.runtime.engine.Tracer.log_error") as mock_log_error:
            names = {t["function"]["name"] for t in await engine._get_active_tools(agent_cfg)}
            assert "book_flight" not in names
            mock_log_error.assert_called()

    asyncio.run(_run())
