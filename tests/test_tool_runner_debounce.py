import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch

from intagrin.compiler.parser import ExecutionGraph
from intagrin.config.schema import AgentConfig, AppConfig, MemoryConfig, ModelConfig
from intagrin.runtime.engine import RuntimeEngine
from intagrin.runtime.tool_runner import ToolRunner


def _schemas(n: int) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {"name": f"tool_{i}", "description": f"does thing {i}", "parameters": {}},
        }
        for i in range(n)
    ]


def _fake_llm_response(names: list[str]):
    import json

    resp = MagicMock()
    resp.choices = [MagicMock(message=MagicMock(content=json.dumps({"tools": names})))]
    return resp


def test_repeated_trajectory_reuses_prior_selection_without_a_new_llm_call():
    """Two get_active_tools() calls with the same recent trajectory and schema set must only
    call the router LLM once — the second is served from the engine's per-turn debounce cache."""

    async def run():
        config = AppConfig(
            version="1.0",
            name="debounce-test",
            default_agent="assistant",
            model=ModelConfig(primary="mock/model"),
            memory=MemoryConfig(type="buffer"),
            agents={"assistant": AgentConfig(lazy_load_tools=True)},
        )
        graph = ExecutionGraph(config, {})
        engine = RuntimeEngine(graph, Path.cwd())
        engine.messages = [{"role": "user", "content": "please use tool_2"}]

        schemas = _schemas(10)
        agent_cfg = config.agents["assistant"]

        call_count = {"n": 0}

        async def fake_acompletion(**kwargs):
            call_count["n"] += 1
            return _fake_llm_response(["tool_2"])

        with patch("intagrin.runtime.tool_runner.litellm.acompletion", side_effect=fake_acompletion):
            first = await ToolRunner.get_active_tools(engine, agent_cfg, schemas)
            second = await ToolRunner.get_active_tools(engine, agent_cfg, schemas)

        assert call_count["n"] == 1, "second call with an unchanged trajectory must not re-query"
        first_names = {s["function"]["name"] for s in first}
        second_names = {s["function"]["name"] for s in second}
        assert first_names == second_names
        assert "tool_2" in first_names

    asyncio.run(run())


def test_changed_trajectory_triggers_a_fresh_selection():
    """A new message changes the trajectory, so the debounce cache must miss and re-query."""

    async def run():
        config = AppConfig(
            version="1.0",
            name="debounce-invalidate-test",
            default_agent="assistant",
            model=ModelConfig(primary="mock/model"),
            memory=MemoryConfig(type="buffer"),
            agents={"assistant": AgentConfig(lazy_load_tools=True)},
        )
        graph = ExecutionGraph(config, {})
        engine = RuntimeEngine(graph, Path.cwd())
        engine.messages = [{"role": "user", "content": "please use tool_2"}]

        schemas = _schemas(10)
        agent_cfg = config.agents["assistant"]

        call_count = {"n": 0}

        async def fake_acompletion(**kwargs):
            call_count["n"] += 1
            return _fake_llm_response(["tool_3"])

        with patch("intagrin.runtime.tool_runner.litellm.acompletion", side_effect=fake_acompletion):
            await ToolRunner.get_active_tools(engine, agent_cfg, schemas)
            engine.messages.append({"role": "assistant", "content": "using tool_2 now"})
            await ToolRunner.get_active_tools(engine, agent_cfg, schemas)

        assert call_count["n"] == 2, "a changed trajectory must trigger a new selection query"

    asyncio.run(run())
