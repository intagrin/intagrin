import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from intagrin.compiler.parser import ExecutionGraph
from intagrin.config.schema import AgentConfig, AppConfig, MemoryConfig, ModelConfig
from intagrin.errors import IntaGrinError
from intagrin.runtime.engine import RuntimeEngine


def _config():
    return AppConfig(
        version="1.0",
        name="stream-loop-test",
        default_agent="assistant",
        model=ModelConfig(primary="mock/model"),
        memory=MemoryConfig(type="buffer"),
        agents={"assistant": AgentConfig()},
    )


class _FakeDelta:
    def __init__(self, content=None):
        self.content = content
        self.tool_calls = None


class _FakeChunk:
    def __init__(self, content=None):
        self.choices = [MagicMock(delta=_FakeDelta(content))]


async def _fake_stream():
    yield _FakeChunk("")


class _Msg:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls
        self.role = "assistant"

    def model_dump(self, exclude_none=True):
        d = {"role": "assistant"}
        if self.content is not None:
            d["content"] = self.content
        if self.tool_calls:
            d["tool_calls"] = self.tool_calls
        return d


def test_streaming_turn_builds_system_prompt_once_per_turn_not_per_round():
    """A multi-round tool-calling turn must call _build_system_prompt exactly once (at the top of
    the turn), not once per tool-call round — the Jinja2-render-per-round cost the Systems Review
    flagged in the old recursive implementation of _run_agent_turn_stream."""

    async def run():
        graph = ExecutionGraph(_config(), {})
        engine = RuntimeEngine(graph, Path.cwd())
        await engine.initialize()

        tool_call = MagicMock()
        tool_call.function.name = "noop_tool"
        tool_call.id = "call_1"

        responses = [
            MagicMock(choices=[MagicMock(message=_Msg(tool_calls=[tool_call]))], usage=None),
            MagicMock(choices=[MagicMock(message=_Msg(content="All done."))], usage=None),
        ]

        build_calls = {"n": 0}
        original_build = engine._build_system_prompt

        def counting_build(agent_cfg):
            build_calls["n"] += 1
            return original_build(agent_cfg)

        with patch.object(
            engine, "_build_system_prompt", side_effect=counting_build
        ), patch(
            "intagrin.runtime.engine.litellm.acompletion",
            new=AsyncMock(return_value=_fake_stream()),
        ), patch(
            "intagrin.runtime.engine.litellm.stream_chunk_builder",
            side_effect=lambda chunks, messages: responses.pop(0),
        ), patch.object(
            RuntimeEngine,
            "_execute_tool_calls_with_healing",
            AsyncMock(
                return_value=[
                    {"role": "tool", "tool_call_id": "call_1", "name": "noop_tool", "content": "ok"}
                ]
            ),
        ), patch.object(
            RuntimeEngine,
            "_get_active_tools",
            AsyncMock(return_value=[{"function": {"name": "noop_tool"}}]),
        ):
            # acompletion is called twice (once per round); AsyncMock's single return_value would
            # be reused for both, which is fine since the stream itself is empty either way — the
            # actual round-specific message comes from stream_chunk_builder's side_effect above.
            events = [ev async for ev in engine._run_agent_turn_stream(interactive=False)]

        assert build_calls["n"] == 1, f"expected system prompt built once, got {build_calls['n']}"
        assert any(e.get("content") == "\n[Executing 1 tools...]" for e in events)
        assert engine.messages[-1]["content"] == "All done."

    asyncio.run(run())


def test_streaming_turn_circuit_breaker_trip_leaves_no_orphaned_tool_call():
    """Same regression as test_circuit_breaker_trip_mid_turn_leaves_no_orphaned_tool_call in
    test_agent_spawning.py, for the streaming loop specifically — this is the code path a real
    production incident actually went through (/stream, not /chat). If
    _execute_tool_calls_with_healing raises IntaGrinError (e.g. a circuit breaker tripping
    mid-execution), the assistant's tool_calls message — already appended and checkpointed before
    execution started — must still get a paired role:"tool" response for every tool_call_id, or
    the checkpointed history is left permanently invalid for strict providers (Gemini/Vertex)."""

    async def run():
        graph = ExecutionGraph(_config(), {})
        engine = RuntimeEngine(graph, Path.cwd())
        await engine.initialize()

        tool_call = MagicMock()
        tool_call.function.name = "some_tool"
        tool_call.id = "call_orphan_check_stream"

        response = MagicMock(
            choices=[MagicMock(message=_Msg(tool_calls=[tool_call]))], usage=None
        )

        with patch(
            "intagrin.runtime.engine.litellm.acompletion",
            new=AsyncMock(return_value=_fake_stream()),
        ), patch(
            "intagrin.runtime.engine.litellm.stream_chunk_builder",
            side_effect=lambda chunks, messages: response,
        ), patch.object(
            RuntimeEngine,
            "_execute_tool_calls_with_healing",
            AsyncMock(side_effect=IntaGrinError("IG-RT-007", "Circuit Breaker Triggered: test.")),
        ), patch.object(
            RuntimeEngine,
            "_get_active_tools",
            AsyncMock(return_value=[{"function": {"name": "some_tool"}}]),
        ):
            events = [ev async for ev in engine._run_agent_turn_stream(interactive=False)]

        assert any("IG-RT-007" in (e.get("content") or "") for e in events)

        assistant_idx = next(
            i
            for i, m in enumerate(engine.messages)
            if m.get("role") == "assistant" and m.get("tool_calls")
        )
        next_msg = engine.messages[assistant_idx + 1]
        assert next_msg.get("role") == "tool", (
            f"expected a role:'tool' response immediately after the tool_calls message, got "
            f"{next_msg.get('role')!r}"
        )
        assert next_msg.get("tool_call_id") == "call_orphan_check_stream"

    asyncio.run(run())
