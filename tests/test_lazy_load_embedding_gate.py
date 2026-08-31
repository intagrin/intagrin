"""Tests for ToolRunner._select_tools_by_embedding — the cosine-similarity fast path in front of
lazy_load_tools' original LLM-based selection (runtime/tool_runner.py). Every embedding call is
faked with hand-picked vectors so the selection math is checked exactly, not just "something got
picked" — real embedding-provider behavior is out of scope here (episodic_memory.embed_text
already has its own coverage for that)."""

import asyncio
from pathlib import Path
from unittest.mock import patch

from intagrin.compiler.parser import ExecutionGraph
from intagrin.config.schema import AgentConfig, AppConfig, MemoryConfig, ModelConfig
from intagrin.runtime.engine import RuntimeEngine
from intagrin.runtime.tool_runner import ToolRunner


def _schemas(names: list[str]) -> list[dict]:
    return [
        {"type": "function", "function": {"name": name, "description": f"does {name}", "parameters": {}}}
        for name in names
    ]


def _engine():
    config = AppConfig(
        version="1.0",
        name="embedding-gate-test",
        default_agent="assistant",
        model=ModelConfig(primary="mock/model"),
        memory=MemoryConfig(type="buffer"),
        agents={"assistant": AgentConfig(lazy_load_tools=True)},
    )
    graph = ExecutionGraph(config, {})
    return RuntimeEngine(graph, Path.cwd()), config.agents["assistant"]


# Trajectory embedding is [1, 0, 0]. tool_0 matches it exactly (sim=1.0), tool_1 is close
# (sim≈0.994), tool_2..tool_5 are orthogonal (sim=0.0) — a clean two-tool cluster above any
# reasonable relative threshold, with the rest clearly excluded.
_VECTORS = {
    "trajectory": [1.0, 0.0, 0.0],
    "tool_0": [1.0, 0.0, 0.0],
    "tool_1": [0.9, 0.1, 0.0],
    "tool_2": [0.0, 1.0, 0.0],
    "tool_3": [0.0, 1.0, 0.0],
    "tool_4": [0.0, 1.0, 0.0],
    "tool_5": [0.0, 1.0, 0.0],
}


async def _fake_embed(model, text):
    """Any text that isn't a tool's own "name: description" string is treated as the trajectory
    — deliberately not an exact string match, since get_active_tools builds the real trajectory
    as "role: content" (e.g. "user: trajectory"), not the bare word "trajectory" the standalone
    _select_tools_by_embedding tests pass directly."""
    for key in ("tool_0", "tool_1", "tool_2", "tool_3", "tool_4", "tool_5"):
        if text.startswith(f"{key}:"):
            return _VECTORS[key]
    return _VECTORS["trajectory"]


def test_embedding_gate_selects_the_similar_cluster_plus_the_floor(tmp_path):
    """tool_0/tool_1 clear the relative threshold on their own; the floor of 3 pulls in exactly
    one more (tool_2, the first of the tied-zero group in original schema order) — not the whole
    remaining set."""
    engine, agent_cfg = _engine()
    engine.messages = [{"role": "user", "content": "trajectory"}]
    schemas = _schemas(["tool_0", "tool_1", "tool_2", "tool_3", "tool_4", "tool_5"])

    async def run():
        with patch("intagrin.runtime.episodic_memory.embed_text", side_effect=_fake_embed):
            return await ToolRunner._select_tools_by_embedding(engine, schemas, "trajectory")

    selected = asyncio.run(run())
    assert selected == {"tool_0", "tool_1", "tool_2"}


def test_embedding_gate_end_to_end_never_calls_the_llm_when_it_succeeds(tmp_path):
    """get_active_tools must use the embedding selection outright — no fallback to the LLM
    router — when the embedding path returns a confident answer."""
    engine, agent_cfg = _engine()
    engine.messages = [{"role": "user", "content": "trajectory"}]
    schemas = _schemas(["tool_0", "tool_1", "tool_2", "tool_3", "tool_4", "tool_5"])

    llm_calls = {"n": 0}

    async def fake_acompletion(**kwargs):
        llm_calls["n"] += 1
        raise AssertionError("LLM router must not be called when the embedding path succeeds")

    async def run():
        with (
            patch("intagrin.runtime.episodic_memory.embed_text", side_effect=_fake_embed),
            patch("intagrin.runtime.tool_runner.litellm.acompletion", side_effect=fake_acompletion),
        ):
            return await ToolRunner.get_active_tools(engine, agent_cfg, schemas)

    result = asyncio.run(run())
    assert llm_calls["n"] == 0
    names = {s["function"]["name"] for s in result}
    # The always-include control-flow tools aren't in this schema set, so the result is exactly
    # the embedding selection.
    assert names == {"tool_0", "tool_1", "tool_2"}


def test_embedding_gate_falls_back_to_none_when_a_tool_embedding_fails(tmp_path):
    """A single tool failing to embed (provider hiccup, unmapped model) must abandon the whole
    embedding attempt rather than silently scoring that tool as maximally dissimilar."""
    engine, agent_cfg = _engine()
    schemas = _schemas(["tool_0", "tool_1"])

    async def flaky_embed(model, text):
        if text.startswith("tool_1:"):
            return None
        return [1.0, 0.0, 0.0]

    async def run():
        with patch("intagrin.runtime.episodic_memory.embed_text", side_effect=flaky_embed):
            return await ToolRunner._select_tools_by_embedding(engine, schemas, "trajectory")

    assert asyncio.run(run()) is None


def test_embedding_gate_falls_back_to_none_when_the_trajectory_embedding_fails(tmp_path):
    engine, agent_cfg = _engine()
    schemas = _schemas(["tool_0", "tool_1"])

    async def trajectory_fails(model, text):
        if text == "trajectory":
            return None
        return await _fake_embed(model, text)

    async def run():
        with patch("intagrin.runtime.episodic_memory.embed_text", side_effect=trajectory_fails):
            return await ToolRunner._select_tools_by_embedding(engine, schemas, "trajectory")

    assert asyncio.run(run()) is None


def test_embedding_gate_caches_tool_embeddings_across_calls(tmp_path):
    """Tool embeddings are computed once per (embedding_model, tool-name-set) and reused — only
    the trajectory should be re-embedded on a second call with the same tool set."""
    engine, agent_cfg = _engine()
    schemas = _schemas(["tool_0", "tool_1", "tool_2", "tool_3", "tool_4", "tool_5"])

    embed_calls = []

    async def counting_embed(model, text):
        embed_calls.append(text)
        return await _fake_embed(model, text)

    async def run():
        with patch("intagrin.runtime.episodic_memory.embed_text", side_effect=counting_embed):
            await ToolRunner._select_tools_by_embedding(engine, schemas, "trajectory")
            first_call_count = len(embed_calls)
            await ToolRunner._select_tools_by_embedding(engine, schemas, "trajectory")
            second_call_batch = len(embed_calls) - first_call_count
        return first_call_count, second_call_batch

    first_count, second_batch = asyncio.run(run())
    assert first_count == 7  # 6 tools + 1 trajectory
    assert second_batch == 1  # only the trajectory re-embedded
