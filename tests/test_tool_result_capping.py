"""Tests for circuit_breakers.max_tool_result_chars (oversized tool results silently ballooning
context — see docs/13_Configuration_Reference.md) and model.enable_prompt_caching (wiring
litellm.enable_anthropic_prompt_caching from ai.yaml)."""

import asyncio
from types import SimpleNamespace

import litellm

from intagrin.compiler.parser import ExecutionGraph
from intagrin.config.schema import (
    AgentConfig,
    AppConfig,
    CircuitBreakersConfig,
    LocalToolConfig,
    MemoryConfig,
    ModelConfig,
)
from intagrin.runtime.engine import RuntimeEngine


def _graph(circuit_breakers: CircuitBreakersConfig | None = None, enable_prompt_caching: bool = True):
    config = AppConfig(
        version="1.0",
        name="tool-result-capping-test",
        default_agent="assistant",
        model=ModelConfig(primary="mock/model", enable_prompt_caching=enable_prompt_caching),
        memory=MemoryConfig(type="sqlite"),
        circuit_breakers=circuit_breakers or CircuitBreakersConfig(),
        agents={
            "assistant": AgentConfig(tools=[LocalToolConfig(name="big_tool", module="unused")])
        },
    )
    return ExecutionGraph(config, {})


def _tool_call():
    return [SimpleNamespace(id="call_1", function=SimpleNamespace(name="big_tool", arguments="{}"))]


# --- circuit_breakers.max_tool_result_chars -------------------------------------------------


def test_oversized_tool_result_is_truncated_with_a_notice(tmp_path):
    huge = "x" * 30_000

    def big_tool() -> str:
        return huge

    async def _run():
        engine = RuntimeEngine(graph=_graph(), project_dir=tmp_path, session_id="s1")
        engine.local_tools["big_tool"] = big_tool
        await engine.initialize()
        engine.active_agent_name = "assistant"

        results = await engine._execute_tool_calls_with_healing(_tool_call(), interactive=False)
        content = results[0]["content"]

        # Default cap is 20,000 chars — the kept prefix plus notice must be well short of the
        # original 30,000, and the notice must say how much was cut and point at the setting.
        assert len(content) < 30_000
        assert content.startswith("x" * 20_000)
        assert "truncated" in content
        assert "30,000" in content
        assert "max_tool_result_chars" in content

    asyncio.run(_run())


def test_small_tool_result_is_not_touched(tmp_path):
    def small_tool() -> str:
        return "a short result"

    async def _run():
        engine = RuntimeEngine(graph=_graph(), project_dir=tmp_path, session_id="s2")
        engine.local_tools["big_tool"] = small_tool
        await engine.initialize()
        engine.active_agent_name = "assistant"

        results = await engine._execute_tool_calls_with_healing(_tool_call(), interactive=False)
        assert results[0]["content"] == "a short result"

    asyncio.run(_run())


def test_max_tool_result_chars_null_disables_the_cap(tmp_path):
    huge = "y" * 30_000

    def big_tool() -> str:
        return huge

    async def _run():
        graph = _graph(circuit_breakers=CircuitBreakersConfig(max_tool_result_chars=None))
        engine = RuntimeEngine(graph=graph, project_dir=tmp_path, session_id="s3")
        engine.local_tools["big_tool"] = big_tool
        await engine.initialize()
        engine.active_agent_name = "assistant"

        results = await engine._execute_tool_calls_with_healing(_tool_call(), interactive=False)
        assert results[0]["content"] == huge

    asyncio.run(_run())


# --- model.enable_prompt_caching -------------------------------------------------------------


def test_enable_prompt_caching_true_by_default_sets_the_litellm_flag(tmp_path):
    original = litellm.enable_anthropic_prompt_caching
    try:
        litellm.enable_anthropic_prompt_caching = False
        RuntimeEngine(graph=_graph(enable_prompt_caching=True), project_dir=tmp_path, session_id="s4")
        assert litellm.enable_anthropic_prompt_caching is True
    finally:
        litellm.enable_anthropic_prompt_caching = original


def test_enable_prompt_caching_false_clears_the_litellm_flag(tmp_path):
    original = litellm.enable_anthropic_prompt_caching
    try:
        litellm.enable_anthropic_prompt_caching = True
        RuntimeEngine(graph=_graph(enable_prompt_caching=False), project_dir=tmp_path, session_id="s5")
        assert litellm.enable_anthropic_prompt_caching is False
    finally:
        litellm.enable_anthropic_prompt_caching = original
