import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from intagrin.compiler.parser import ExecutionGraph
from intagrin.config.schema import (
    AgentConfig,
    AppConfig,
    LocalToolConfig,
    MemoryConfig,
    ModelConfig,
)
from intagrin.errors import AwaitingHumanInput
from intagrin.runtime.engine import RuntimeEngine


def _mock_graph(tool_names=()):
    """`tool_names` are declared on the agent (module path is never actually loaded — the test
    injects the function directly into engine.local_tools after initialize()) so that
    _is_tool_allowed_for_active_agent permits calling them."""
    config = AppConfig(
        version="1.0",
        name="dynamic-approval-test",
        default_agent="assistant",
        model=ModelConfig(primary="mock/model"),
        memory=MemoryConfig(type="buffer"),
        agents={
            "assistant": AgentConfig(
                tools=[LocalToolConfig(name=n, module="unused") for n in tool_names]
            )
        },
    )
    return ExecutionGraph(config, {})


def test_awaiting_human_input_sets_pending_approval_with_prompt_and_context():
    """A local tool raising AwaitingHumanInput mid-execution pauses the session exactly like a
    statically requires_approval-gated tool does — same _pending_approval shape, plus additive
    prompt/context keys carrying what the tool wants a human to see."""

    async def _run():
        engine = RuntimeEngine(
            graph=_mock_graph(["refund"]), project_dir=Path.cwd(), session_id="s1"
        )
        await engine.initialize()
        engine.active_agent_name = "assistant"

        def refund(order_id: str, amount: float) -> str:
            if amount > 500:
                raise AwaitingHumanInput(
                    prompt=f"Refund of ${amount} for order {order_id} exceeds the auto-approve limit.",
                    context={"order_id": order_id, "amount": amount},
                )
            return f"Refunded ${amount}."

        engine.local_tools["refund"] = refund

        result = await engine.execute_tool(
            "refund", {"order_id": "o1", "amount": 999.0}, interactive=False, tool_call_id="call_1"
        )

        pending = engine.state["_pending_approval"]
        assert "exceeds the auto-approve limit" in result
        assert pending["tool"] == "refund"
        assert pending["args"] == {"order_id": "o1", "amount": 999.0}
        assert pending["status"] == "awaiting_approval"
        assert pending["tool_call_id"] == "call_1"
        assert "exceeds the auto-approve limit" in pending["prompt"]
        assert pending["context"] == {"order_id": "o1", "amount": 999.0}

    asyncio.run(_run())


def test_pause_for_human_stamps_a_created_at_timestamp():
    """With no expiry/escalation mechanism, created_at is the only way a caller (or GET /sessions)
    can tell a pause apart from one that's been silently stuck for days versus seconds."""
    import datetime as dt

    async def _run():
        engine = RuntimeEngine(
            graph=_mock_graph(["send_email"]), project_dir=Path.cwd(), session_id="s1b"
        )
        await engine.initialize()
        engine.active_agent_name = "assistant"
        engine.tools_requiring_approval["send_email"] = {
            "required_approvals": 1,
            "required_approvers": None,
        }
        engine.local_tools["send_email"] = lambda **kwargs: "sent"

        before = dt.datetime.now(dt.UTC)
        await engine.execute_tool(
            "send_email", {"to": "a@b.com"}, interactive=False, tool_call_id="call_ts"
        )
        after = dt.datetime.now(dt.UTC)

        created_at = engine.state["_pending_approval"]["created_at"]
        stamp = dt.datetime.fromisoformat(created_at)
        assert before <= stamp <= after

    asyncio.run(_run())


def test_static_pending_approval_still_omits_prompt_and_context():
    """Regression guard: the existing static requires_approval: true headless path's
    _pending_approval dict must keep its original shape — no prompt/context keys — so existing
    checkpoints/clients that only know tool/args/agent/status/tool_call_id are unaffected."""

    async def _run():
        engine = RuntimeEngine(
            graph=_mock_graph(["send_email"]), project_dir=Path.cwd(), session_id="s2"
        )
        await engine.initialize()
        engine.active_agent_name = "assistant"
        engine.tools_requiring_approval["send_email"] = {"required_approvals": 1, "required_approvers": None}
        engine.local_tools["send_email"] = lambda **kwargs: "sent"

        await engine.execute_tool(
            "send_email", {"to": "a@b.com"}, interactive=False, tool_call_id="call_2"
        )

        pending = engine.state["_pending_approval"]
        assert "prompt" not in pending
        assert "context" not in pending

    asyncio.run(_run())


def test_resume_endpoint_handles_dynamically_paused_tool():
    """POST /resume's existing approved + edited_args contract already generically covers
    "human supplies the corrected/missing value, retry" — proves it works unchanged for a
    dynamically-paused (AwaitingHumanInput) call, not just a statically-gated one."""
    from intagrin.server.api import ResumeRequest, resume_endpoint

    graph = MagicMock()
    graph.config.server.auth.approver_env_var = None

    mock_engine = MagicMock()
    mock_engine._await_last_checkpoint = AsyncMock()
    mock_engine.initialize = AsyncMock()
    mock_engine._run_agent_turn = AsyncMock()
    mock_engine.execute_tool = AsyncMock(return_value="Refunded $100.")
    mock_engine._promote_next_queued_approval = MagicMock(return_value=False)
    mock_engine.state = {
        "_pending_approval": {
            "tool": "refund",
            "args": {"order_id": "o1", "amount": 999.0},
            "status": "awaiting_approval",
            "tool_call_id": "call_1",
            "prompt": "Refund exceeds limit.",
            "context": {"order_id": "o1", "amount": 999.0},
        }
    }
    mock_engine.messages = []
    mock_engine.is_transferring = False
    mock_engine.active_agent_name = "assistant"

    with patch("intagrin.server.api.parse_project", return_value=graph), patch(
        "intagrin.server.api.get_shared_resources_cache",
        return_value=MagicMock(get=AsyncMock(return_value=MagicMock())),
    ), patch("intagrin.server.api.RuntimeEngine", return_value=mock_engine):
        response = asyncio.run(
            resume_endpoint(
                ResumeRequest(
                    session_id="s1",
                    approved=True,
                    edited_args={"order_id": "o1", "amount": 100.0},
                ),
                request=MagicMock(headers={}),
                user_context="tenant",
            )
        )

    assert response.status == "completed"
    mock_engine.execute_tool.assert_awaited_once_with(
        "refund", {"order_id": "o1", "amount": 100.0}, interactive=False, tool_call_id="call_1"
    )


def test_run_agent_turn_breaks_on_dynamic_pause_in_interactive_mode():
    """A tool raising AwaitingHumanInput must halt _run_agent_turn even when interactive=True —
    there's no synchronous continuation possible for a mid-function raise, unlike the static
    requires_approval path's Confirm.ask, which resolves before the tool ever runs."""

    async def _run():
        graph = _mock_graph(["needs_human"])
        engine = RuntimeEngine(graph=graph, project_dir=Path.cwd(), session_id="s3")
        await engine.initialize()
        engine.active_agent_name = "assistant"

        def needs_human(**kwargs) -> str:
            raise AwaitingHumanInput(prompt="need a human")

        engine.local_tools["needs_human"] = needs_human
        # The declared tool's module ("unused") never actually loads, so global_tool_schemas
        # never got an entry for it — append one directly, matching what a successful load would
        # produce, so the turn loop's tool-hallucination guard doesn't drop the call.
        from intagrin.runtime.tools_loader import get_tool_schema

        engine.global_tool_schemas.append(get_tool_schema(needs_human))

        tool_call = MagicMock()
        tool_call.function.name = "needs_human"
        tool_call.function.arguments = "{}"
        tool_call.id = "call_1"

        class _Msg:
            def __init__(self):
                self.role = "assistant"
                self.content = None
                self.tool_calls = [tool_call]

            def model_dump(self, exclude_none=True):
                return {"role": "assistant", "tool_calls": [tool_call]}

        response = MagicMock(
            choices=[MagicMock(message=_Msg())],
            usage=MagicMock(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )

        acompletion_calls = {"n": 0}

        async def counting_acompletion(*args, **kwargs):
            acompletion_calls["n"] += 1
            return response

        with patch("intagrin.runtime.engine.litellm.acompletion", side_effect=counting_acompletion):
            await engine._run_agent_turn(interactive=True)

        assert "_pending_approval" in engine.state
        assert acompletion_calls["n"] == 1, "the loop must halt, not call the LLM again"

    asyncio.run(_run())


def test_stream_turn_halts_on_pending_approval():
    """_run_agent_turn_stream must halt after a tool sets _pending_approval instead of continuing
    to a second LLM round — pre-existing gap fixed alongside the dynamic-suspend feature: this
    loop previously had no check for _pending_approval at all."""

    async def _run():
        engine = RuntimeEngine(
            graph=_mock_graph(["send_email"]), project_dir=Path.cwd(), session_id="s4"
        )
        await engine.initialize()
        engine.active_agent_name = "assistant"
        engine.tools_requiring_approval["send_email"] = {"required_approvals": 1, "required_approvers": None}

        def send_email(**kwargs) -> str:
            return "sent"

        engine.local_tools["send_email"] = send_email
        from intagrin.runtime.tools_loader import get_tool_schema

        engine.global_tool_schemas.append(get_tool_schema(send_email))

        class _FakeDelta:
            def __init__(self, content=None):
                self.content = content
                self.tool_calls = None

        class _FakeChunk:
            def __init__(self, content=None):
                self.choices = [MagicMock(delta=_FakeDelta(content))]

        async def _fake_stream():
            yield _FakeChunk("")

        tool_call = MagicMock()
        tool_call.function.name = "send_email"
        tool_call.function.arguments = "{}"
        tool_call.id = "call_1"

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

        response = MagicMock(
            choices=[MagicMock(message=_Msg(tool_calls=[tool_call]))], usage=None
        )

        acompletion_calls = {"n": 0}

        async def counting_acompletion(*args, **kwargs):
            acompletion_calls["n"] += 1
            return _fake_stream()

        with patch(
            "intagrin.runtime.engine.litellm.acompletion", side_effect=counting_acompletion
        ), patch(
            "intagrin.runtime.engine.litellm.stream_chunk_builder",
            return_value=response,
        ):
            events = [ev async for ev in engine._run_agent_turn_stream(interactive=False)]

        assert acompletion_calls["n"] == 1, "must halt, not start a second LLM round"
        assert "_pending_approval" in engine.state
        assert any("paused" in str(e.get("content", "")) for e in events)

    asyncio.run(_run())


def test_set_pending_approval_does_not_let_a_new_pause_jump_an_already_queued_one():
    """Regression test for a real bug found live: _set_pending_approval only checked whether
    _pending_approval itself was occupied, not whether anything was already waiting in
    _pending_approval_queue. So a pause that arrived exactly when _pending_approval had *just*
    been popped (but an older pause was still sitting in the queue behind it — e.g. a /resume
    call that resolves the current one and, in the same turn-loop continuation, immediately
    triggers a brand-new pause) would claim the now-empty slot directly, permanently stranding
    the older queued pause: nothing ever promotes an item that isn't at the front of the queue,
    so it would sit there forever, invisible to /resume. FIFO order must be preserved regardless
    of whether _pending_approval happens to be empty at the exact moment a new pause arrives."""

    async def _run():
        engine = RuntimeEngine(
            graph=_mock_graph(), project_dir=Path.cwd(), session_id="s5"
        )
        await engine.initialize()

        # Simulate: an older pause already sitting in the queue (as if _pending_approval was
        # occupied by something else when it was queued), then _pending_approval itself becomes
        # empty (as /resume's own pop does right before its turn-loop continuation runs).
        engine.state["_pending_approval_queue"] = [{"tool": "older_paused_tool"}]

        engine._set_pending_approval({"tool": "brand_new_tool"})

        # The new pause must NOT have claimed the empty slot directly — it must have gone to the
        # back of the queue, behind the older one.
        assert "_pending_approval" not in engine.state
        assert [item["tool"] for item in engine.state["_pending_approval_queue"]] == [
            "older_paused_tool",
            "brand_new_tool",
        ]

        # Promotion must surface the OLDER one first, not the new one.
        promoted = engine._promote_next_queued_approval()
        assert promoted is True
        assert engine.state["_pending_approval"]["tool"] == "older_paused_tool"
        assert [item["tool"] for item in engine.state["_pending_approval_queue"]] == [
            "brand_new_tool"
        ]

    asyncio.run(_run())
