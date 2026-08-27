"""_approved_tool_calls (the one-time exemption POST /resume grants a paused tool call once a
human approves it) used to be keyed by tool *name*, not tool_call_id. Two concurrent calls to the
same requires_approval tool with different arguments (e.g. two "refund" calls for different
orders in one batch) are different actions — approving one must never let the *other*, unapproved
one execute just because they share a name. Fixed in both halves of the mechanism: the append
site (server/api.py's _execute_approved_tool_and_replace_placeholder) and the consume site
(runtime/engine.py's execute_tool)."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from intagrin.compiler.parser import ExecutionGraph
from intagrin.config.schema import (
    AgentConfig,
    AppConfig,
    LocalToolConfig,
    MemoryConfig,
    ModelConfig,
)
from intagrin.runtime.engine import RuntimeEngine


def _graph():
    config = AppConfig(
        version="1.0",
        name="approval-exemption-test",
        default_agent="assistant",
        model=ModelConfig(primary="mock/model"),
        memory=MemoryConfig(type="sqlite"),
        agents={
            "assistant": AgentConfig(
                tools=[LocalToolConfig(name="refund", module="unused", requires_approval=True)]
            )
        },
    )
    return ExecutionGraph(config, {})


async def _engine(tmp_path, session_id):
    def refund(order_id: str) -> str:
        refund.calls.append(order_id)
        return f"refunded {order_id}"

    refund.calls = []

    engine = RuntimeEngine(graph=_graph(), project_dir=tmp_path, session_id=session_id)
    engine.local_tools["refund"] = refund
    engine.tools_requiring_approval["refund"] = {
        "required_approvals": 1,
        "required_approvers": None,
    }
    await engine.initialize()
    engine.active_agent_name = "assistant"
    return engine, refund


def test_approving_one_call_does_not_exempt_a_different_call_of_the_same_tool(tmp_path):
    """The core regression: an exemption entry shaped like the tool name (what the old,
    over-broad mechanism would have stored) must not grant a free pass to an arbitrary
    differently-identified call of that tool."""

    async def _run():
        engine, refund = await _engine(tmp_path, "s1")

        # Simulate a name-keyed entry — exactly what the old (buggy) /resume append produced.
        engine.state.setdefault("_approved_tool_calls", []).append("refund")

        # A call this exemption was never actually meant for must still be rejected/paused.
        result = await engine.execute_tool(
            "refund", {"order_id": "order_B"}, interactive=False, tool_call_id="call_B"
        )
        assert "order_B" not in refund.calls
        assert "paused" in result.lower()

    asyncio.run(_run())


def test_approved_call_id_executes_and_a_concurrent_unapproved_one_does_not(tmp_path):
    """Two concurrent refund calls for different orders both pause. Only call_A's id is granted
    the exemption. Retrying both: call_A executes, call_B (same tool, no exemption of its own)
    must still be refused."""

    async def _run():
        engine, refund = await _engine(tmp_path, "s2")

        result_a = await engine.execute_tool(
            "refund", {"order_id": "order_A"}, interactive=False, tool_call_id="call_A"
        )
        assert "paused" in result_a.lower()
        result_b = await engine.execute_tool(
            "refund", {"order_id": "order_B"}, interactive=False, tool_call_id="call_B"
        )
        assert "paused" in result_b.lower()

        # Only call_A gets approved.
        engine.state.setdefault("_approved_tool_calls", []).append("call_A")

        retry_b = await engine.execute_tool(
            "refund", {"order_id": "order_B"}, interactive=False, tool_call_id="call_B"
        )
        assert "order_B" not in refund.calls
        assert "paused" in retry_b.lower()

        retry_a = await engine.execute_tool(
            "refund", {"order_id": "order_A"}, interactive=False, tool_call_id="call_A"
        )
        assert refund.calls == ["order_A"]
        assert "refunded order_A" in retry_a

        # The exemption is one-time: call_A can't be replayed a second time either.
        assert "call_A" not in engine.state.get("_approved_tool_calls", [])

    asyncio.run(_run())


def test_resume_endpoint_records_the_approved_tool_call_id_not_the_tool_name():
    """Unit-level proof of the api.py half of the fix: POST /resume must append the paused call's
    tool_call_id to _approved_tool_calls, not its tool name."""
    from intagrin.server.api import ResumeRequest, resume_endpoint

    graph = MagicMock()
    graph.config.server.auth.approver_env_var = None

    # Captures _approved_tool_calls exactly as it stood the moment execute_tool was invoked —
    # append-then-call-then-cleanup all happen within the same request, so asserting on
    # mock_engine.state after resume_endpoint returns would see it already popped.
    captured: dict = {}

    async def _capture_and_return(*args, **kwargs):
        captured["approved_tool_calls"] = list(mock_engine.state.get("_approved_tool_calls", []))
        return "Refunded $100."

    mock_engine = MagicMock()
    mock_engine._await_last_checkpoint = AsyncMock()
    mock_engine.initialize = AsyncMock()
    mock_engine._run_agent_turn = AsyncMock()
    mock_engine.execute_tool = AsyncMock(side_effect=_capture_and_return)
    mock_engine._promote_next_queued_approval = MagicMock(return_value=False)
    mock_engine.state = {
        "_pending_approval": {
            "tool": "refund",
            "args": {"order_id": "o1", "amount": 100.0},
            "status": "awaiting_approval",
            "tool_call_id": "call_1",
        }
    }
    mock_engine.messages = []
    mock_engine.is_transferring = False
    mock_engine.active_agent_name = "assistant"

    with patch("intagrin.server.api.parse_project", return_value=graph), patch(
        "intagrin.server.api.get_shared_resources_cache",
        return_value=MagicMock(get=AsyncMock(return_value=MagicMock())),
    ), patch("intagrin.server.api.RuntimeEngine", return_value=mock_engine):
        asyncio.run(
            resume_endpoint(
                ResumeRequest(session_id="s1", approved=True),
                request=MagicMock(headers={}),
                user_context="tenant",
            )
        )

    assert captured["approved_tool_calls"] == ["call_1"]
