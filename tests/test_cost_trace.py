import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from intagrin.compiler.parser import ExecutionGraph
from intagrin.config.schema import AgentConfig, AppConfig, MemoryConfig, ModelConfig
from intagrin.runtime.engine import RuntimeEngine


def _engine():
    config = AppConfig(
        version="1.0",
        name="cost-trace-test",
        default_agent="assistant",
        model=ModelConfig(primary="mock/model"),
        memory=MemoryConfig(type="buffer"),
        agents={"assistant": AgentConfig()},
    )
    graph = ExecutionGraph(config, {})
    return RuntimeEngine(graph, Path.cwd())


def _response(total_tokens=100, prompt_tokens=60, completion_tokens=40):
    usage = MagicMock(
        total_tokens=total_tokens, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
    )
    return MagicMock(usage=usage)


def test_record_usage_appends_a_per_turn_cost_trace_entry_additively():
    """_cost_trace must be additive alongside the existing running totals in _metrics — not a
    replacement — so inta replay/simulate can reconstruct a session's cost trajectory without
    breaking anything that already reads _metrics.total_cost/total_tokens."""
    engine = _engine()
    engine.messages = [{"role": "user", "content": "hi"}]

    with patch("intagrin.runtime.engine.litellm.completion_cost", return_value=0.01):
        engine._record_usage(_response(total_tokens=100))

    assert engine.state["_metrics"]["total_tokens"] == 100
    assert engine.state["_metrics"]["total_cost"] == 0.01
    assert engine.state["_cost_trace"] == [{"turn": 1, "tokens": 100, "cost": 0.01}]

    engine.messages.append({"role": "assistant", "content": "hello"})
    with patch("intagrin.runtime.engine.litellm.completion_cost", return_value=0.02):
        engine._record_usage(_response(total_tokens=50))

    assert engine.state["_metrics"]["total_tokens"] == 150
    assert engine.state["_metrics"]["total_cost"] == pytest.approx(0.03)
    assert engine.state["_cost_trace"] == [
        {"turn": 1, "tokens": 100, "cost": 0.01},
        {"turn": 2, "tokens": 50, "cost": 0.02},
    ]


def test_record_usage_without_usage_data_does_not_append_a_trace_entry():
    engine = _engine()
    response = MagicMock(usage=None)
    engine._record_usage(response)
    assert "_cost_trace" not in engine.state


def test_run_agent_turn_records_usage_before_appending_the_assistant_message():
    """Regression test for a real bug: _record_usage's own docstring says `turn` is "the index
    this response's assistant message will land at once appended... called before that append at
    every call site" — _run_agent_turn_stream honored this (record, then append), but the
    blocking _run_agent_turn called _record_usage AFTER the append, so `turn` was always one past
    where the message actually landed. `inta replay` builds `cost_by_turn = {entry["turn"]: entry
    for entry in _cost_trace}` and looks up the assistant message's own index — for every session
    produced via the blocking loop (/chat, /resume, `inta run`/`inta dev`, nested spawned-child
    continuations) that lookup silently missed or misattributed cost to the wrong message."""

    class _Msg:
        def __init__(self, content):
            self.content = content
            self.tool_calls = None

        def model_dump(self, exclude_none=True):
            return {"role": "assistant", "content": self.content}

    async def _run():
        engine = _engine()
        await engine.initialize()
        engine.active_agent_name = "assistant"
        engine.messages.append({"role": "user", "content": "hello"})

        response = _response(total_tokens=42)
        response.choices = [MagicMock(message=_Msg("hi there"))]

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion, patch(
            "intagrin.runtime.engine.litellm.completion_cost", return_value=0.01
        ):
            mock_acompletion.return_value = response
            await engine._run_agent_turn(interactive=False)

        turn = engine.state["_cost_trace"][-1]["turn"]
        landed_index = next(
            i for i, m in enumerate(engine.messages)
            if m.get("role") == "assistant" and m.get("content") == "hi there"
        )
        assert turn == landed_index

    asyncio.run(_run())
