"""Tests for the Reflexion-style connector between _compress_error_loops (already-existing
repeated-identical-tool-failure detection) and episodic memory (engine.py's
_reflect_on_error_loop) — a known-bad tool-call pattern gets remembered across sessions, not just
interrupted within the one it happened in."""

import asyncio
from unittest.mock import AsyncMock, patch

from intagrin.compiler.parser import ExecutionGraph
from intagrin.config.schema import (
    AgentConfig,
    AppConfig,
    EpisodicMemoryConfig,
    MemoryConfig,
    ModelConfig,
)
from intagrin.runtime.engine import RuntimeEngine
from intagrin.runtime.episodic_memory import query_episodes


def _mock_graph(episodic_memory=None, memory_type="sqlite"):
    config = AppConfig(
        version="1.0",
        name="reflexion-test",
        default_agent="assistant",
        model=ModelConfig(primary="mock/model"),
        memory=MemoryConfig(type=memory_type),
        agents={"assistant": AgentConfig(description="Assistant")},
        episodic_memory=episodic_memory,
    )
    return ExecutionGraph(config, {})


def _append_identical_failures(engine, tool_name="bad_tool", error="System Error: Invalid syntax", n=3):
    for _ in range(n):
        engine.messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": tool_name, "arguments": "{}"}}
                ],
            }
        )
        engine.messages.append({"role": "tool", "name": tool_name, "content": error})


def test_error_loop_writes_a_reflection_episode_when_episodic_memory_is_configured(tmp_path):
    async def _run():
        engine = RuntimeEngine(
            graph=_mock_graph(EpisodicMemoryConfig(scope="session")),
            project_dir=tmp_path,
            session_id="s1",
        )
        await engine.initialize()
        _append_identical_failures(engine)

        with patch("litellm.aembedding", new_callable=AsyncMock) as mock_aembedding:
            mock_aembedding.side_effect = RuntimeError("no embedding provider in this test")
            engine._compress_error_loops()
            assert engine._pending_reflection_tasks, "a reflection task must be scheduled"
            await asyncio.gather(*engine._pending_reflection_tasks)

        # Task must remove itself from the tracking set once done (see the done-callback).
        assert engine._pending_reflection_tasks == set()

        rows = query_episodes(
            engine.graph.config.memory, tmp_path, engine._episodic_scope_key(), "failure_pattern", None, 10
        )
        assert len(rows) == 1
        assert "bad_tool" in rows[0]["content"]
        assert "3 times" in rows[0]["content"]
        assert "System Error: Invalid syntax" in rows[0]["content"]
        assert "reflexion" in rows[0]["tags"]
        assert "bad_tool" in rows[0]["tags"]
        assert rows[0]["importance"] == 7  # min(10, 4 + 3)

    asyncio.run(_run())


def test_reflection_importance_scales_with_repeat_count(tmp_path):
    async def _run():
        engine = RuntimeEngine(
            graph=_mock_graph(EpisodicMemoryConfig(scope="session")),
            project_dir=tmp_path,
            session_id="s1",
        )
        await engine.initialize()
        _append_identical_failures(engine, n=8)  # well beyond the 3-repeat detection threshold

        with patch("litellm.aembedding", new_callable=AsyncMock) as mock_aembedding:
            mock_aembedding.side_effect = RuntimeError("no embedding provider in this test")
            engine._compress_error_loops()
            await asyncio.gather(*engine._pending_reflection_tasks)

        rows = query_episodes(
            engine.graph.config.memory, tmp_path, engine._episodic_scope_key(), "failure_pattern", None, 10
        )
        assert rows[0]["importance"] == 10  # min(10, 4 + 8) clamped

    asyncio.run(_run())


def test_no_episodic_memory_configured_schedules_nothing_and_needs_no_event_loop(tmp_path):
    """Regression guard for the exact shape tests/test_engine_advanced.py's
    test_error_loop_compression already exercises: _compress_error_loops must remain safe to
    call synchronously, with no running asyncio event loop, when episodic_memory isn't
    configured — asyncio.create_task() would raise RuntimeError outside a running loop, so the
    early-return guard in _reflect_on_error_loop must fire before ever reaching it. Uses
    buffer-type memory (no real checkpointer), matching test_error_loop_compression's own setup
    — a real sqlite checkpointer's _save_checkpoint has its own, unrelated pre-existing habit of
    also calling asyncio.create_task() outside a running loop, which isn't what this test is
    about."""
    engine = RuntimeEngine(graph=_mock_graph(None, memory_type="buffer"), project_dir=tmp_path, session_id="s1")
    _append_identical_failures(engine)

    engine._compress_error_loops()  # no asyncio.run(...) wrapper — must not raise

    assert engine._pending_reflection_tasks == set()
    assert "YOU ARE STUCK IN A LOOP" in engine.messages[-1]["content"]


def test_system_guard_mentions_recall_episodes_only_when_episodic_memory_is_configured(tmp_path):
    async def _run():
        with_episodic = RuntimeEngine(
            graph=_mock_graph(EpisodicMemoryConfig()), project_dir=tmp_path, session_id="s1"
        )
        await with_episodic.initialize()
        _append_identical_failures(with_episodic)
        with patch("litellm.aembedding", new_callable=AsyncMock) as mock_aembedding:
            mock_aembedding.side_effect = RuntimeError("no embedding provider in this test")
            with_episodic._compress_error_loops()
            await asyncio.gather(*with_episodic._pending_reflection_tasks)
        assert "recall_episodes" in with_episodic.messages[-1]["content"]

    asyncio.run(_run())

    without_episodic = RuntimeEngine(
        graph=_mock_graph(None, memory_type="buffer"), project_dir=tmp_path, session_id="s2"
    )
    _append_identical_failures(without_episodic)
    without_episodic._compress_error_loops()  # sync, no event loop — must still work (see the test above)
    assert "recall_episodes" not in without_episodic.messages[-1]["content"]
