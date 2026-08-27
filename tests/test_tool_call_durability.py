import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

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


def _graph():
    config = AppConfig(
        version="1.0",
        name="tool-durability-test",
        default_agent="assistant",
        model=ModelConfig(primary="mock/model"),
        memory=MemoryConfig(type="sqlite"),
        agents={
            "assistant": AgentConfig(tools=[LocalToolConfig(name="refund", module="unused")])
        },
    )
    return ExecutionGraph(config, {})


def _dangling_assistant_message():
    """An assistant message with tool_calls and no matching tool-role responses — the exact
    shape a process interruption leaves behind between that message being checkpointed and the
    batch's results being checkpointed."""
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "refund", "arguments": '{"order_id": "o1"}'},
            },
            {
                "id": "call_2",
                "type": "function",
                "function": {"name": "refund", "arguments": '{"order_id": "o2"}'},
            },
        ],
    }


def test_recovery_does_not_re_execute_a_tool_call_already_cached_before_interruption(tmp_path):
    """Regression test: if the process is interrupted after call_1 finished (and was write-ahead
    cached in state["_tool_call_scratch"]) but before the whole batch's results were appended and
    checkpointed, resuming must NOT re-invoke call_1's side-effecting tool a second time — only
    call_2, which never got the chance to run at all."""
    calls = []

    def refund(order_id: str) -> str:
        calls.append(order_id)
        return f"refunded {order_id}"

    async def _run():
        engine = RuntimeEngine(graph=_graph(), project_dir=tmp_path, session_id="crash-1")
        engine.local_tools["refund"] = refund
        await engine.initialize()
        engine.active_agent_name = "assistant"

        engine.messages.append(_dangling_assistant_message())
        engine.state["_tool_call_scratch"] = {
            "call_1": {
                "result": {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "name": "refund",
                    "content": "refunded o1 (from before interruption)",
                },
                "deferred_merge": None,
            }
        }

        await engine._recover_dangling_tool_calls()

        # call_1 must NOT be re-invoked — only call_2, the genuinely-unresolved one.
        assert calls == ["o2"]

        tool_msgs = {m["tool_call_id"]: m for m in engine.messages if m.get("role") == "tool"}
        assert tool_msgs["call_1"]["content"] == "refunded o1 (from before interruption)"
        assert tool_msgs["call_2"]["content"] == "refunded o2"
        assert engine.state["_tool_call_scratch"] == {}

    asyncio.run(_run())


def test_a_fresh_engine_loading_a_crashed_checkpoint_auto_resolves_the_dangling_batch(tmp_path):
    """End-to-end: engine1 leaves a dangling assistant tool_calls message checkpointed, as if the
    process died right after that message was appended and before any tool call ran. A brand new
    RuntimeEngine instance (simulating a server restart) loading that same session must resolve it
    automatically during initialize() instead of leaving the conversation permanently broken."""
    calls = []

    def refund(order_id: str) -> str:
        calls.append(order_id)
        return f"refunded {order_id}"

    async def _run():
        engine1 = RuntimeEngine(graph=_graph(), project_dir=tmp_path, session_id="crash-2")
        await engine1.initialize()
        engine1.active_agent_name = "assistant"
        engine1.messages.append(_dangling_assistant_message())
        engine1._save_checkpoint()
        await engine1._await_last_checkpoint()

        # A brand new engine instance, same session, same on-disk checkpoint — as if the server
        # restarted. It's never told about the dangling batch except via the loaded checkpoint.
        engine2 = RuntimeEngine(graph=_graph(), project_dir=tmp_path, session_id="crash-2")
        engine2.local_tools["refund"] = refund
        await engine2.initialize()

        assert sorted(calls) == ["o1", "o2"]
        tool_msgs = [m for m in engine2.messages if m.get("role") == "tool"]
        assert {m["tool_call_id"] for m in tool_msgs} == {"call_1", "call_2"}
        assert engine2.messages[-1]["role"] == "tool"
        assert engine2.state.get("_tool_call_scratch", {}) == {}

    asyncio.run(_run())


def test_normal_completion_leaves_no_scratch_residue(tmp_path):
    """A fully-completed (non-crashed) tool round must not leave anything behind in
    state["_tool_call_scratch"] — otherwise the cache would grow unboundedly over a long-running
    session."""
    def refund(order_id: str) -> str:
        return f"refunded {order_id}"

    async def _run():
        engine = RuntimeEngine(graph=_graph(), project_dir=tmp_path, session_id="normal-1")
        engine.local_tools["refund"] = refund
        await engine.initialize()
        engine.active_agent_name = "assistant"

        from types import SimpleNamespace

        tool_calls = [
            SimpleNamespace(
                id="call_1",
                function=SimpleNamespace(name="refund", arguments='{"order_id": "o1"}'),
            )
        ]
        results = await engine._execute_tool_calls_with_healing(tool_calls, interactive=False)
        assert results[0]["content"] == "refunded o1"
        assert engine.state.get("_tool_call_scratch", {}) == {}

    asyncio.run(_run())


def test_parallel_tool_calls_beyond_the_cap_are_rejected_not_executed(tmp_path):
    """circuit_breakers.max_parallel_tool_calls_per_turn caps how many ordinary tool calls run
    concurrently in one turn (inta verify used to flag this as unbounded). A call beyond the cap
    must NOT be executed — side effects matter, e.g. a refund — but must still get a tool-role
    response so the message sequence stays valid."""

    calls = []

    def refund(order_id: str) -> str:
        calls.append(order_id)
        return f"refunded {order_id}"

    async def _run():
        config = AppConfig(
            version="1.0",
            name="tool-durability-test",
            default_agent="assistant",
            model=ModelConfig(primary="mock/model"),
            memory=MemoryConfig(type="sqlite"),
            circuit_breakers=CircuitBreakersConfig(max_parallel_tool_calls_per_turn=1),
            agents={
                "assistant": AgentConfig(tools=[LocalToolConfig(name="refund", module="unused")])
            },
        )
        engine = RuntimeEngine(graph=ExecutionGraph(config, {}), project_dir=tmp_path, session_id="cap-1")
        engine.local_tools["refund"] = refund
        await engine.initialize()
        engine.active_agent_name = "assistant"

        tool_calls = [
            SimpleNamespace(id="call_1", function=SimpleNamespace(name="refund", arguments='{"order_id": "o1"}')),
            SimpleNamespace(id="call_2", function=SimpleNamespace(name="refund", arguments='{"order_id": "o2"}')),
        ]
        return await engine._execute_tool_calls_with_healing(tool_calls, interactive=False)

    results = asyncio.run(_run())
    # Only the first call, within the cap, actually ran the side-effecting tool.
    assert calls == ["o1"]
    result_map = {r["tool_call_id"]: r for r in results}
    assert result_map["call_1"]["content"] == "refunded o1"
    assert "max_parallel_tool_calls_per_turn" in result_map["call_2"]["content"]
    # Every original tool_call still gets a tool-role response — no dangling batch.
    assert {r["tool_call_id"] for r in results} == {"call_1", "call_2"}


def test_self_healing_corrector_call_respects_max_corrector_tokens(tmp_path):
    """The corrector-model call that repairs malformed tool-call arguments previously had no
    max_tokens set at all (inta verify used to flag this as unbounded cost) — it must now pass
    circuit_breakers.max_corrector_tokens."""

    def refund(order_id: str) -> str:
        return f"refunded {order_id}"

    async def _run():
        config = AppConfig(
            version="1.0",
            name="tool-durability-test",
            default_agent="assistant",
            model=ModelConfig(primary="mock/model"),
            memory=MemoryConfig(type="sqlite"),
            circuit_breakers=CircuitBreakersConfig(max_corrector_tokens=77),
            agents={
                "assistant": AgentConfig(tools=[LocalToolConfig(name="refund", module="unused")])
            },
        )
        engine = RuntimeEngine(
            graph=ExecutionGraph(config, {}), project_dir=tmp_path, session_id="corrector-1"
        )
        engine.local_tools["refund"] = refund
        await engine.initialize()
        engine.active_agent_name = "assistant"

        # Malformed (truncated) JSON arguments trigger the self-healing corrector-model path.
        tool_calls = [
            SimpleNamespace(
                id="call_1", function=SimpleNamespace(name="refund", arguments='{"order_id": "o1"')
            )
        ]
        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
            mock_acompletion.return_value = MagicMock(
                choices=[MagicMock(message=MagicMock(content='{"order_id": "o1"}'))]
            )
            await engine._execute_tool_calls_with_healing(tool_calls, interactive=False)
        return mock_acompletion

    mock_acompletion = asyncio.run(_run())
    mock_acompletion.assert_awaited_once()
    assert mock_acompletion.await_args.kwargs["max_tokens"] == 77
