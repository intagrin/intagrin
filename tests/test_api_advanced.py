import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from intagrin.runtime.engine import RuntimeEngine
from intagrin.server.api import (
    ChatRequest,
    ChatResponse,
    ResumeRequest,
    chat_endpoint,
    resume_endpoint,
    stream_endpoint,
)


def _mock_graph_with_no_approver():
    """A parse_project()-shaped mock with server.auth.approver_env_var explicitly None — a bare
    MagicMock() attribute is truthy, which would make verify_approver think an approver key is
    configured and reject every mocked /resume call regardless of what's being tested."""
    graph = MagicMock()
    graph.config.server.auth.approver_env_var = None
    return graph


def test_tenant_idor_isolation():
    """Test that the API securely prefixes the session_id with the authenticated Tenant ID."""
    with patch(
        "intagrin.server.api.parse_project", return_value=_mock_graph_with_no_approver()
    ), patch(
        "intagrin.server.api.get_shared_resources_cache",
        return_value=MagicMock(get=AsyncMock(return_value=MagicMock())),
    ):
        mock_engine = MagicMock()
        mock_engine._await_last_checkpoint = AsyncMock()
        mock_engine.initialize = AsyncMock()
        mock_engine._run_agent_turn = AsyncMock()
        mock_engine._apply_guardrails.return_value = "Hello"
        mock_engine._compress_memory = AsyncMock()
        mock_engine._save_checkpoint = MagicMock()
        mock_engine.state = {}
        mock_engine.messages = []
        mock_engine.is_transferring = False
        mock_engine.active_agent_name = "triage"

        with patch("intagrin.server.api.RuntimeEngine", return_value=mock_engine) as runtime:
            response = asyncio.run(
                chat_endpoint(
                    ChatRequest(message="Hello", session_id="session_99"),
                    user_context="tenant_xyz123",
                )
            )

        assert response.status == "completed"
        assert runtime.call_args.kwargs["session_id"] == "tenant_xyz123:session_99"


def test_chat_endpoint_logs_success_run():
    """A successful /chat call must write one run-log row with status='completed' — the
    regression guard for the Logs page's data source, not just its rendering."""
    with patch(
        "intagrin.server.api.parse_project", return_value=_mock_graph_with_no_approver()
    ), patch(
        "intagrin.server.api.get_shared_resources_cache",
        return_value=MagicMock(get=AsyncMock(return_value=MagicMock())),
    ), patch("intagrin.server.api.record_run_log") as mock_log:
        mock_engine = MagicMock()
        mock_engine._await_last_checkpoint = AsyncMock()
        mock_engine.initialize = AsyncMock()
        mock_engine._run_agent_turn = AsyncMock()
        mock_engine._apply_guardrails.return_value = "Hello"
        mock_engine._compress_memory = AsyncMock()
        mock_engine._save_checkpoint = MagicMock()
        mock_engine.state = {"_metrics": {"total_tokens": 50, "total_cost": 0.005}}
        mock_engine.messages = [{"role": "assistant", "content": "Hi there"}]
        mock_engine.is_transferring = False
        mock_engine.active_agent_name = "triage"

        with patch("intagrin.server.api.RuntimeEngine", return_value=mock_engine):
            asyncio.run(
                chat_endpoint(
                    ChatRequest(message="Hello", session_id="session_99"),
                    user_context="tenant_xyz123",
                )
            )

        mock_log.assert_called_once()
        assert mock_log.call_args.kwargs["session_id"] == "tenant_xyz123:session_99"
        assert mock_log.call_args.kwargs["endpoint"] == "/chat"
        assert mock_log.call_args.kwargs["status"] == "completed"
        assert mock_log.call_args.kwargs["error"] is None
        # chat_endpoint appends the user's message before the turn loop runs, so the seeded
        # 1-message list becomes 2 by the time this is logged.
        assert mock_log.call_args.kwargs["message_count"] == 2


def test_chat_endpoint_logs_error_run():
    """When the turn loop raises, /chat must still write a run-log row (status='error', the
    exception message) before re-raising as an HTTPException — debugging a failed run is exactly
    what this feature is for."""
    import pytest
    from fastapi import HTTPException

    with patch(
        "intagrin.server.api.parse_project", return_value=_mock_graph_with_no_approver()
    ), patch(
        "intagrin.server.api.get_shared_resources_cache",
        return_value=MagicMock(get=AsyncMock(return_value=MagicMock())),
    ), patch("intagrin.server.api.record_run_log") as mock_log:
        mock_engine = MagicMock()
        mock_engine._await_last_checkpoint = AsyncMock()
        mock_engine.initialize = AsyncMock()
        mock_engine._run_agent_turn = AsyncMock(side_effect=RuntimeError("boom"))
        mock_engine._apply_guardrails.return_value = "Hello"
        mock_engine._compress_memory = AsyncMock()
        mock_engine._save_checkpoint = MagicMock()
        mock_engine.state = {"_metrics": {"total_tokens": 0, "total_cost": 0.0}}
        mock_engine.messages = []
        mock_engine.is_transferring = False
        mock_engine.active_agent_name = "triage"

        with patch(
            "intagrin.server.api.RuntimeEngine", return_value=mock_engine
        ), pytest.raises(HTTPException) as exc_info:
            asyncio.run(
                chat_endpoint(
                    ChatRequest(message="Hello", session_id="session_99"),
                    user_context="tenant_xyz123",
                )
            )

        assert exc_info.value.status_code == 500
        mock_log.assert_called_once()
        assert mock_log.call_args.kwargs["status"] == "error"
        assert "boom" in mock_log.call_args.kwargs["error"]


def test_resume_draft_and_review():
    """Test that resuming an agent applies edited arguments to the pending tool call, and that
    the resumed result replaces the paused placeholder in place (same tool_call_id) rather than
    appending a second, orphaned response — the message-threading bug found in the Systems Review."""
    with patch(
        "intagrin.server.api.parse_project", return_value=_mock_graph_with_no_approver()
    ), patch(
        "intagrin.server.api.get_shared_resources_cache",
        return_value=MagicMock(get=AsyncMock(return_value=MagicMock())),
    ):
        # Mock the internal state of the engine as if it was paused awaiting approval
        mock_engine_instance = MagicMock()
        mock_engine_instance._await_last_checkpoint = AsyncMock()
        mock_engine_instance.initialize = AsyncMock()
        mock_engine_instance._run_agent_turn = AsyncMock()
        mock_engine_instance.execute_tool = AsyncMock(return_value="Email sent")
        mock_engine_instance._promote_next_queued_approval = MagicMock(return_value=False)
        mock_engine_instance.state = {
            "_pending_approval": {
                "tool": "send_email",
                "args": {"body": "Bad text"},
                "status": "awaiting_approval",
                "tool_call_id": "call_send_email_1",
            }
        }
        # The placeholder tool message left behind when the pause happened.
        mock_engine_instance.messages = [
            {"role": "user", "content": "send an email"},
            {"role": "assistant", "tool_calls": [{"id": "call_send_email_1"}]},
            {
                "role": "tool",
                "tool_call_id": "call_send_email_1",
                "name": "send_email",
                "content": "Operation 'send_email' is paused awaiting human approval.",
            },
        ]
        mock_engine_instance.is_transferring = False
        mock_engine_instance.active_agent_name = "triage"

        with patch("intagrin.server.api.RuntimeEngine", return_value=mock_engine_instance):
            response = asyncio.run(
                resume_endpoint(
                    ResumeRequest(
                        session_id="session_99",
                        approved=True,
                        edited_args={"body": "Good text"},
                    ),
                    request=MagicMock(headers={}),
                    user_context="tenant_xyz123",
                )
            )

        assert response.status == "completed"
        mock_engine_instance.execute_tool.assert_awaited_once_with(
            "send_email", {"body": "Good text"}, interactive=False, tool_call_id="call_send_email_1"
        )

        # The placeholder must be replaced in place, not duplicated.
        tool_msgs = [m for m in mock_engine_instance.messages if m.get("role") == "tool"]
        assert len(tool_msgs) == 1, f"expected exactly one tool message, got {tool_msgs}"
        assert tool_msgs[0]["tool_call_id"] == "call_send_email_1"
        assert tool_msgs[0]["content"] == "Email sent"


def test_resume_endpoint_preserves_a_fresh_pause_that_happens_during_the_same_turn_loop():
    """Regression test for a real bug found live: approving book_flight via /resume, whose turn
    loop then immediately has the model call book_hotel (also requires_approval: true), reported
    status="awaiting_approval" to the caller but PERSISTED a checkpoint with no _pending_approval
    at all — engine.state.pop("_pending_approval", None) removed the freshly-set book_hotel pause
    before _save_checkpoint(), instead of leaving it (get(), like /chat and /chat/stream already
    do at their equivalent line). The very next plain chat message then sailed straight through
    _pending_approval_block (nothing pending, as far as persisted state was concerned), silently
    orphaning book_hotel's tool_call and letting the model confabulate a "booked" summary for a
    hotel that was never actually reserved."""
    with patch(
        "intagrin.server.api.parse_project", return_value=_mock_graph_with_no_approver()
    ), patch(
        "intagrin.server.api.get_shared_resources_cache",
        return_value=MagicMock(get=AsyncMock(return_value=MagicMock())),
    ):
        mock_engine_instance = MagicMock()
        mock_engine_instance._await_last_checkpoint = AsyncMock()
        mock_engine_instance.initialize = AsyncMock()
        mock_engine_instance.execute_tool = AsyncMock(return_value="Flight booked")
        mock_engine_instance._promote_next_queued_approval = MagicMock(return_value=False)
        mock_engine_instance.state = {
            "_pending_approval": {
                "tool": "book_flight",
                "args": {"destination": "Malaysia"},
                "status": "awaiting_approval",
                "tool_call_id": "call_flight_1",
            }
        }
        mock_engine_instance.messages = [
            {"role": "user", "content": "book my trip"},
            {"role": "assistant", "tool_calls": [{"id": "call_flight_1"}]},
            {
                "role": "tool",
                "tool_call_id": "call_flight_1",
                "name": "book_flight",
                "content": "Operation 'book_flight' is paused awaiting human approval.",
            },
        ]
        mock_engine_instance.is_transferring = False
        mock_engine_instance.active_agent_name = "planner"

        async def _turn(interactive=False):
            # The model immediately proposes book_hotel next, which also requires approval —
            # simulating _pause_for_human's _set_pending_approval writing a brand-new pause into
            # the same freed slot mid-turn-loop.
            mock_engine_instance.messages.append(
                {"role": "assistant", "tool_calls": [{"id": "call_hotel_1"}]}
            )
            mock_engine_instance.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": "call_hotel_1",
                    "name": "book_hotel",
                    "content": "Operation 'book_hotel' is paused awaiting human approval.",
                }
            )
            mock_engine_instance.state["_pending_approval"] = {
                "tool": "book_hotel",
                "args": {"destination": "Malaysia"},
                "status": "awaiting_approval",
                "tool_call_id": "call_hotel_1",
            }
            mock_engine_instance.is_transferring = False

        mock_engine_instance._run_agent_turn = AsyncMock(side_effect=_turn)

        with patch("intagrin.server.api.RuntimeEngine", return_value=mock_engine_instance):
            response = asyncio.run(
                resume_endpoint(
                    ResumeRequest(session_id="session_99", approved=True),
                    request=MagicMock(headers={}),
                    user_context="tenant_xyz123",
                )
            )

        assert response.status == "awaiting_approval"
        assert response.pending_action["tool"] == "book_hotel"
        # The checkpoint that actually gets persisted must still carry the fresh pause — this is
        # the part the bug broke: the JSON response claimed "awaiting_approval" while the
        # persisted engine.state had already had it popped out from under it.
        assert mock_engine_instance.state.get("_pending_approval", {}).get("tool") == "book_hotel"


def test_resume_endpoint_logs_run():
    """A successful /resume call must write one run-log row for the /resume endpoint."""
    with patch(
        "intagrin.server.api.parse_project", return_value=_mock_graph_with_no_approver()
    ), patch(
        "intagrin.server.api.get_shared_resources_cache",
        return_value=MagicMock(get=AsyncMock(return_value=MagicMock())),
    ), patch("intagrin.server.api.record_run_log") as mock_log:
        mock_engine_instance = MagicMock()
        mock_engine_instance._await_last_checkpoint = AsyncMock()
        mock_engine_instance.initialize = AsyncMock()
        mock_engine_instance._run_agent_turn = AsyncMock()
        mock_engine_instance.execute_tool = AsyncMock(return_value="Email sent")
        mock_engine_instance._promote_next_queued_approval = MagicMock(return_value=False)
        mock_engine_instance.state = {
            "_pending_approval": {
                "tool": "send_email",
                "args": {"body": "Bad text"},
                "status": "awaiting_approval",
                "tool_call_id": "call_send_email_1",
            },
            "_metrics": {"total_tokens": 10, "total_cost": 0.001},
        }
        mock_engine_instance.messages = []
        mock_engine_instance.is_transferring = False
        mock_engine_instance.active_agent_name = "triage"

        with patch("intagrin.server.api.RuntimeEngine", return_value=mock_engine_instance):
            asyncio.run(
                resume_endpoint(
                    ResumeRequest(session_id="session_99", approved=True),
                    request=MagicMock(headers={}),
                    user_context="tenant_xyz123",
                )
            )

        mock_log.assert_called_once()
        assert mock_log.call_args.kwargs["endpoint"] == "/resume"
        assert mock_log.call_args.kwargs["status"] == "completed"


def test_concurrent_chat_requests_to_same_session_are_serialized():
    """Two overlapping requests for the same session_id must not run concurrently — before the
    session-lock fix, both would independently load/mutate/save the same session state, and
    last-write-wins on save could silently drop one entire turn. Proven deterministically here by
    recording execution order rather than relying on real timing: a slow "LLM call" that isn't
    serialized would interleave (start, start, end, end); a serialized one won't
    (start, end, start, end)."""
    execution_order = []

    async def slow_turn(*args, **kwargs):
        execution_order.append("start")
        await asyncio.sleep(0.05)
        execution_order.append("end")

    with patch(
        "intagrin.server.api.parse_project", return_value=_mock_graph_with_no_approver()
    ), patch(
        "intagrin.server.api.get_shared_resources_cache",
        return_value=MagicMock(get=AsyncMock(return_value=MagicMock())),
    ):
        mock_engine = MagicMock()
        mock_engine._await_last_checkpoint = AsyncMock()
        mock_engine.initialize = AsyncMock()
        mock_engine._run_agent_turn = AsyncMock(side_effect=slow_turn)
        mock_engine._apply_guardrails.return_value = "Hello"
        mock_engine._compress_memory = AsyncMock()
        mock_engine._save_checkpoint = MagicMock()
        mock_engine.state = {}
        mock_engine.messages = []
        mock_engine.is_transferring = False
        mock_engine.active_agent_name = "triage"

        with patch("intagrin.server.api.RuntimeEngine", return_value=mock_engine):

            async def run_both():
                await asyncio.gather(
                    chat_endpoint(
                        ChatRequest(message="Hi", session_id="shared_session"),
                        user_context="tenant",
                    ),
                    chat_endpoint(
                        ChatRequest(message="Hi", session_id="shared_session"),
                        user_context="tenant",
                    ),
                )

            asyncio.run(run_both())

    assert execution_order == ["start", "end", "start", "end"], execution_order


def test_resume_approval_requires_separate_approver_key_when_configured():
    """When server.auth.approver_env_var is set, approving a gated tool call must require a
    distinct X-Approver-Key header — the session's own auth credential is not enough. Direct
    regression test for the fixed self-approval gap (previously, the same credential that
    triggered a requires_approval tool call could immediately approve it too)."""
    import os

    import pytest
    from fastapi import HTTPException

    graph = MagicMock()
    graph.config.server.auth.approver_env_var = "APPROVER_KEY"

    mock_engine_instance = MagicMock()
    mock_engine_instance._await_last_checkpoint = AsyncMock()
    mock_engine_instance.initialize = AsyncMock()
    mock_engine_instance._run_agent_turn = AsyncMock()
    mock_engine_instance.execute_tool = AsyncMock(return_value="done")
    mock_engine_instance._promote_next_queued_approval = MagicMock(return_value=False)
    mock_engine_instance.state = {
        "_pending_approval": {"tool": "t", "args": {}, "tool_call_id": "c1"}
    }
    mock_engine_instance.messages = []
    mock_engine_instance.is_transferring = False
    mock_engine_instance.active_agent_name = "triage"

    with patch("intagrin.server.api.parse_project", return_value=graph), patch(
        "intagrin.server.api.get_shared_resources_cache",
        return_value=MagicMock(get=AsyncMock(return_value=MagicMock())),
    ), patch("intagrin.server.api.RuntimeEngine", return_value=mock_engine_instance), patch.dict(
        os.environ, {"APPROVER_KEY": "s3cr3t"}
    ):
        # Missing header -> rejected before the engine even resumes anything.
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(
                resume_endpoint(
                    ResumeRequest(session_id="s1", approved=True),
                    request=MagicMock(headers={}),
                    user_context="tenant",
                )
            )
        assert exc_info.value.status_code == 403
        mock_engine_instance.execute_tool.assert_not_awaited()

        # Wrong header value -> also rejected.
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(
                resume_endpoint(
                    ResumeRequest(session_id="s1", approved=True),
                    request=MagicMock(headers={"X-Approver-Key": "wrong"}),
                    user_context="tenant",
                )
            )
        assert exc_info.value.status_code == 403

        # Correct header -> proceeds normally.
        response = asyncio.run(
            resume_endpoint(
                ResumeRequest(session_id="s1", approved=True),
                request=MagicMock(headers={"X-Approver-Key": "s3cr3t"}),
                user_context="tenant",
            )
        )
        assert response.status == "completed"
        mock_engine_instance.execute_tool.assert_awaited_once()


def test_resume_denial_does_not_require_approver_key():
    """Rejecting a gated tool call isn't a privilege escalation the way approving it is — denial
    must stay usable with only the requester's own session credential."""
    import os

    graph = MagicMock()
    graph.config.server.auth.approver_env_var = "APPROVER_KEY"

    mock_engine_instance = MagicMock()
    mock_engine_instance._await_last_checkpoint = AsyncMock()
    mock_engine_instance.initialize = AsyncMock()
    mock_engine_instance._run_agent_turn = AsyncMock()
    mock_engine_instance._promote_next_queued_approval = MagicMock(return_value=False)
    mock_engine_instance.state = {
        "_pending_approval": {"tool": "t", "args": {}, "tool_call_id": "c1"}
    }
    mock_engine_instance.messages = []
    mock_engine_instance.is_transferring = False
    mock_engine_instance.active_agent_name = "triage"

    with patch("intagrin.server.api.parse_project", return_value=graph), patch(
        "intagrin.server.api.get_shared_resources_cache",
        return_value=MagicMock(get=AsyncMock(return_value=MagicMock())),
    ), patch("intagrin.server.api.RuntimeEngine", return_value=mock_engine_instance), patch.dict(
        os.environ, {"APPROVER_KEY": "s3cr3t"}
    ):
        response = asyncio.run(
            resume_endpoint(
                ResumeRequest(session_id="s1", approved=False),
                request=MagicMock(headers={}),
                user_context="tenant",
            )
        )
        assert response.status == "completed"


def test_stream_endpoint_error_path_yields_error_event_and_logs():
    """Pre-existing gap fixed alongside the run-log feature: stream_endpoint's event_generator
    had no except clause at all, so an in-stream error died silently with no client-visible event
    and no chance to log it. Now it must yield an SSE error event and write a run-log row."""

    async def raising_stream(*args, **kwargs):
        yield {"type": "content", "content": "partial..."}
        raise RuntimeError("stream broke")

    with patch(
        "intagrin.server.api.parse_project", return_value=_mock_graph_with_no_approver()
    ), patch(
        "intagrin.server.api.get_shared_resources_cache",
        return_value=MagicMock(get=AsyncMock(return_value=MagicMock())),
    ), patch("intagrin.server.api.record_run_log") as mock_log:
        mock_engine = MagicMock()
        mock_engine._await_last_checkpoint = AsyncMock()
        mock_engine.initialize = AsyncMock()
        mock_engine._run_agent_turn_stream = raising_stream
        mock_engine._apply_guardrails.return_value = "Hello"
        mock_engine._compress_memory = AsyncMock()
        mock_engine._save_checkpoint = MagicMock()
        mock_engine.state = {"_metrics": {"total_tokens": 0, "total_cost": 0.0}}
        mock_engine.messages = []
        mock_engine.is_transferring = False
        mock_engine.active_agent_name = "triage"

        with patch("intagrin.server.api.RuntimeEngine", return_value=mock_engine):

            async def collect():
                response = await stream_endpoint(
                    ChatRequest(message="Hello", session_id="session_99"),
                    user_context="tenant_xyz123",
                )
                chunks = []
                async for chunk in response.body_iterator:
                    chunks.append(chunk)
                return chunks

            chunks = asyncio.run(collect())

        body = "".join(chunks)
        assert '"type": "error"' in body
        assert "stream broke" in body

        mock_log.assert_called_once()
        assert mock_log.call_args.kwargs["endpoint"] == "/stream"
        assert mock_log.call_args.kwargs["status"] == "error"
        assert "stream broke" in mock_log.call_args.kwargs["error"]


def _nested_pending_action(**overrides):
    pending = {
        "tool": "book_flight",
        "args": {"flight": "AF123"},
        "agent": "orchestrator_dyn_abc",
        "status": "awaiting_approval",
        "required_approvals": 1,
        "required_approvers": None,
        "approvals_received": [],
        "child_session_id": "tenant:s1_spawn_orchestrator_dyn_abc",
        "parent_tool_call_id": "spawn_call_1",
        "pre_state": {"trip_plan": {}},
    }
    pending.update(overrides)
    return pending


def test_resume_endpoint_continues_a_spawned_childs_pending_approval():
    """Regression test for the real fix: spawn_agent no longer silently discards a spawned
    child's pending approval — it persists the child under its own session_id and leaves a
    pointer (child_session_id) in the parent's own _pending_approval. /resume must detect that
    pointer, resolve the *child's* pending approval (not try to execute the tool directly on the
    parent, which has none of the child's own context), continue the child to completion, merge
    its result back into the parent, and replace the parent's original spawn_agent tool-result
    placeholder with the real outcome — not leave it reading "paused" forever."""
    graph = MagicMock()
    graph.config.server.auth.approver_env_var = None
    graph.config.circuit_breakers.max_delegation_turns = 15

    parent_engine = MagicMock()
    parent_engine._await_last_checkpoint = AsyncMock()
    parent_engine.initialize = AsyncMock()
    parent_engine._save_checkpoint = MagicMock()
    parent_engine._promote_next_queued_approval = MagicMock(return_value=False)
    parent_engine.session_id = "tenant:s1"
    parent_engine.active_agent_name = "orchestrator"
    parent_engine.is_transferring = False
    parent_engine.state = {"_pending_approval": _nested_pending_action()}
    parent_engine.messages = [
        {
            "role": "tool",
            "tool_call_id": "spawn_call_1",
            "name": "spawn_agent",
            "content": "Sub-agent 'orchestrator_dyn_abc' is paused awaiting human approval for tool 'book_flight'.",
        }
    ]

    async def _parent_turn(interactive=False):
        parent_engine.messages.append(
            {"role": "assistant", "content": "Booked Paris flight, moving to Tokyo."}
        )
        parent_engine.is_transferring = False

    parent_engine._run_agent_turn = AsyncMock(side_effect=_parent_turn)

    child_engine = MagicMock()
    child_engine._await_last_checkpoint = AsyncMock()
    child_engine.initialize = AsyncMock()
    child_engine._save_checkpoint = MagicMock()
    child_engine.session_id = "tenant:s1_spawn_orchestrator_dyn_abc"
    child_engine.active_agent_name = "orchestrator_dyn_abc"
    child_engine.is_transferring = False
    child_engine.state = {
        "_pending_approval": {
            "tool": "book_flight",
            "args": {"flight": "AF123"},
            "agent": "orchestrator_dyn_abc",
            "status": "awaiting_approval",
            "tool_call_id": "child_tool_call_1",
            "required_approvals": 1,
            "required_approvers": None,
            "approvals_received": [],
        }
    }
    child_engine.messages = [
        {
            "role": "tool",
            "tool_call_id": "child_tool_call_1",
            "name": "book_flight",
            "content": "Operation 'book_flight' is paused awaiting human approval.",
        }
    ]
    child_engine.execute_tool = AsyncMock(return_value="Flight AF123 booked.")

    async def _child_turn(interactive=False):
        child_engine.messages.append(
            {"role": "assistant", "content": "Booked your flight to Paris!"}
        )
        child_engine.is_transferring = False

    child_engine._run_agent_turn = AsyncMock(side_effect=_child_turn)

    engines_by_session = {
        "tenant:s1": parent_engine,
        "tenant:s1_spawn_orchestrator_dyn_abc": child_engine,
    }

    def _make_engine(*, graph, project_dir, session_id, shared_resources=None, initial_state=None):
        return engines_by_session[session_id]

    with patch(
        "intagrin.server.api.parse_project", return_value=graph
    ), patch(
        "intagrin.server.api.get_shared_resources_cache",
        return_value=MagicMock(get=AsyncMock(return_value=MagicMock())),
    ), patch(
        "intagrin.server.api.RuntimeEngine", side_effect=_make_engine
    ):
        response = asyncio.run(
            resume_endpoint(
                ResumeRequest(session_id="s1", approved=True),
                request=MagicMock(headers={}),
                user_context="tenant",
            )
        )

    # The child's tool actually ran, with the child's own pending args/tool_call_id — not
    # something invented on the parent.
    child_engine.execute_tool.assert_awaited_once_with(
        "book_flight", {"flight": "AF123"}, interactive=False, tool_call_id="child_tool_call_1"
    )

    # The parent's original spawn_agent placeholder was replaced in place with the real result.
    spawn_msg = next(m for m in parent_engine.messages if m.get("tool_call_id") == "spawn_call_1")
    assert "Booked your flight to Paris!" in spawn_msg["content"]

    # The parent's state was merged from the child, using the pre-spawn snapshot, and its own
    # pending pointer cleared.
    parent_engine._merge_child_state.assert_called_once_with(
        {"trip_plan": {}}, child_engine.state
    )
    assert "_pending_approval" not in parent_engine.state

    # The parent's own turn continued past the resolved spawn — that's what /resume reports.
    assert response.status == "completed"
    assert "Booked Paris flight, moving to Tokyo." in response.response


def test_resume_endpoint_reports_child_pausing_again_without_merging():
    """If the resumed child hits a *second* approval gate (a different tool call further into its
    task) instead of finishing, /resume must not treat it as done — no merge into the parent, and
    the parent's _pending_approval must be updated to point at the new pending tool so a follow-up
    /resume on the same (parent) session_id continues from exactly where this one left off."""
    graph = MagicMock()
    graph.config.server.auth.approver_env_var = None
    graph.config.circuit_breakers.max_delegation_turns = 15

    parent_engine = MagicMock()
    parent_engine._await_last_checkpoint = AsyncMock()
    parent_engine.initialize = AsyncMock()
    parent_engine._save_checkpoint = MagicMock()
    parent_engine.session_id = "tenant:s1"
    parent_engine.active_agent_name = "orchestrator"
    parent_engine.state = {"_pending_approval": _nested_pending_action()}
    parent_engine.messages = []

    child_engine = MagicMock()
    child_engine._await_last_checkpoint = AsyncMock()
    child_engine.initialize = AsyncMock()
    child_engine._save_checkpoint = MagicMock()
    child_engine.session_id = "tenant:s1_spawn_orchestrator_dyn_abc"
    child_engine.active_agent_name = "orchestrator_dyn_abc"
    child_engine.is_transferring = False
    child_engine.state = {
        "_pending_approval": {
            "tool": "book_flight",
            "args": {"flight": "AF123"},
            "agent": "orchestrator_dyn_abc",
            "status": "awaiting_approval",
            "tool_call_id": "child_tool_call_1",
            "required_approvals": 1,
            "required_approvers": None,
            "approvals_received": [],
        }
    }
    child_engine.messages = []
    child_engine.execute_tool = AsyncMock(return_value="Flight AF123 booked.")

    async def _child_turn(interactive=False):
        # Instead of finishing, the child's very next action is a second approval-gated tool.
        child_engine.state["_pending_approval"] = {
            "tool": "book_hotel",
            "args": {"hotel": "Ritz"},
            "agent": "orchestrator_dyn_abc",
            "status": "awaiting_approval",
            "tool_call_id": "child_tool_call_2",
            "required_approvals": 1,
            "required_approvers": None,
            "approvals_received": [],
        }

    child_engine._run_agent_turn = AsyncMock(side_effect=_child_turn)

    engines_by_session = {
        "tenant:s1": parent_engine,
        "tenant:s1_spawn_orchestrator_dyn_abc": child_engine,
    }

    def _make_engine(*, graph, project_dir, session_id, shared_resources=None, initial_state=None):
        return engines_by_session[session_id]

    with patch(
        "intagrin.server.api.parse_project", return_value=graph
    ), patch(
        "intagrin.server.api.get_shared_resources_cache",
        return_value=MagicMock(get=AsyncMock(return_value=MagicMock())),
    ), patch(
        "intagrin.server.api.RuntimeEngine", side_effect=_make_engine
    ):
        response = asyncio.run(
            resume_endpoint(
                ResumeRequest(session_id="s1", approved=True),
                request=MagicMock(headers={}),
                user_context="tenant",
            )
        )

    assert response.status == "awaiting_approval"
    assert parent_engine.state["_pending_approval"]["tool"] == "book_hotel"
    # child_session_id/parent_tool_call_id/pre_state must survive so a follow-up resume on the
    # same parent session still knows which child (and which pre-spawn state) to continue.
    assert parent_engine.state["_pending_approval"]["child_session_id"] == child_engine.session_id
    assert parent_engine.state["_pending_approval"]["parent_tool_call_id"] == "spawn_call_1"
    parent_engine._merge_child_state.assert_not_called()


def test_resume_endpoint_surfaces_a_queued_pending_approval_after_resolving_the_current_one():
    """End-to-end proof (through the public /resume entry point, not just the engine-level unit
    test) that a second sub-agent's pause — queued behind the first because both paused in the
    same original turn (see _set_pending_approval) — is not silently dropped once the first is
    resolved. /resume must report status="awaiting_approval" again, pointing at the queued one,
    instead of "completed" as if nothing else were pending."""
    graph = MagicMock()
    graph.config.server.auth.approver_env_var = None
    graph.config.circuit_breakers.max_delegation_turns = 15

    queued_pending = _nested_pending_action(
        tool="book_flight",
        agent="orchestrator_dyn_tokyo",
        child_session_id="tenant:s1_spawn_orchestrator_dyn_tokyo",
        parent_tool_call_id="spawn_call_2",
    )

    parent_engine = MagicMock()
    parent_engine._await_last_checkpoint = AsyncMock()
    parent_engine.initialize = AsyncMock()
    parent_engine._save_checkpoint = MagicMock()
    # Bind the real implementation (not a stub) — this test specifically exercises how its
    # result is consumed by /resume, so it needs to actually pop/promote against parent_engine's
    # own state the way a real RuntimeEngine instance would.
    parent_engine._promote_next_queued_approval = lambda: RuntimeEngine._promote_next_queued_approval(
        parent_engine
    )
    parent_engine.session_id = "tenant:s1"
    parent_engine.active_agent_name = "orchestrator"
    parent_engine.is_transferring = False
    parent_engine.state = {
        "_pending_approval": _nested_pending_action(),
        "_pending_approval_queue": [queued_pending],
    }
    parent_engine.messages = [
        {
            "role": "tool",
            "tool_call_id": "spawn_call_1",
            "name": "spawn_agent",
            "content": "Sub-agent 'orchestrator_dyn_abc' is paused awaiting human approval for tool 'book_flight'.",
        }
    ]

    async def _parent_turn(interactive=False):
        # The orchestrator finishes its own reaction to the resolved Paris spawn — it does not
        # naturally re-pause on its own; only the queued Tokyo pause should bring it back to
        # awaiting_approval.
        parent_engine.messages.append(
            {"role": "assistant", "content": "Booked Paris flight."}
        )
        parent_engine.is_transferring = False

    parent_engine._run_agent_turn = AsyncMock(side_effect=_parent_turn)

    child_engine = MagicMock()
    child_engine._await_last_checkpoint = AsyncMock()
    child_engine.initialize = AsyncMock()
    child_engine._save_checkpoint = MagicMock()
    child_engine.session_id = "tenant:s1_spawn_orchestrator_dyn_abc"
    child_engine.active_agent_name = "orchestrator_dyn_abc"
    child_engine.is_transferring = False
    child_engine.state = {
        "_pending_approval": {
            "tool": "book_flight",
            "args": {"flight": "AF123"},
            "agent": "orchestrator_dyn_abc",
            "status": "awaiting_approval",
            "tool_call_id": "child_tool_call_1",
            "required_approvals": 1,
            "required_approvers": None,
            "approvals_received": [],
        }
    }
    child_engine.messages = [
        {
            "role": "tool",
            "tool_call_id": "child_tool_call_1",
            "name": "book_flight",
            "content": "Operation 'book_flight' is paused awaiting human approval.",
        }
    ]
    child_engine.execute_tool = AsyncMock(return_value="Flight AF123 booked.")

    async def _child_turn(interactive=False):
        child_engine.messages.append(
            {"role": "assistant", "content": "Booked your flight to Paris!"}
        )
        child_engine.is_transferring = False

    child_engine._run_agent_turn = AsyncMock(side_effect=_child_turn)

    engines_by_session = {
        "tenant:s1": parent_engine,
        "tenant:s1_spawn_orchestrator_dyn_abc": child_engine,
    }

    def _make_engine(*, graph, project_dir, session_id, shared_resources=None, initial_state=None):
        return engines_by_session[session_id]

    with patch(
        "intagrin.server.api.parse_project", return_value=graph
    ), patch(
        "intagrin.server.api.get_shared_resources_cache",
        return_value=MagicMock(get=AsyncMock(return_value=MagicMock())),
    ), patch(
        "intagrin.server.api.RuntimeEngine", side_effect=_make_engine
    ):
        response = asyncio.run(
            resume_endpoint(
                ResumeRequest(session_id="s1", approved=True),
                request=MagicMock(headers={}),
                user_context="tenant",
            )
        )

    # The Paris child actually resolved and merged — this isn't a no-op.
    child_engine.execute_tool.assert_awaited_once()
    parent_engine._merge_child_state.assert_called_once()

    # The queued Tokyo pause was promoted and reported back to the caller, not dropped.
    assert response.status == "awaiting_approval"
    assert response.pending_action["agent"] == "orchestrator_dyn_tokyo"
    assert response.pending_action["child_session_id"] == "tenant:s1_spawn_orchestrator_dyn_tokyo"
    assert parent_engine.state["_pending_approval"]["agent"] == "orchestrator_dyn_tokyo"
    assert "_pending_approval_queue" not in parent_engine.state


def test_chat_endpoint_refuses_a_new_message_while_an_approval_is_pending():
    """Regression test for a real bug found live: /chat never checked for an existing unresolved
    _pending_approval before running a brand-new LLM turn. A client sending a plain chat message
    while a session was paused (instead of calling /resume) would just plow ahead — risking the
    orchestrator spawning yet another sub-agent (spending circuit-breaker budget) while the
    original pause sat un-actioned, and even letting a second pause silently jump the queue ahead
    of it. /chat must now refuse and point the caller at /resume, without running any turn."""
    pending = {
        "tool": "book_flight",
        "args": {"flight": "AF123"},
        "status": "awaiting_approval",
        "tool_call_id": "call_1",
    }
    with patch(
        "intagrin.server.api.parse_project", return_value=_mock_graph_with_no_approver()
    ), patch(
        "intagrin.server.api.get_shared_resources_cache",
        return_value=MagicMock(get=AsyncMock(return_value=MagicMock())),
    ):
        mock_engine = MagicMock()
        mock_engine._await_last_checkpoint = AsyncMock()
        mock_engine.initialize = AsyncMock()
        mock_engine._run_agent_turn = AsyncMock()
        mock_engine.state = {"_pending_approval": pending}
        mock_engine.messages = []
        mock_engine.active_agent_name = "triage"

        with patch("intagrin.server.api.RuntimeEngine", return_value=mock_engine):
            response = asyncio.run(
                chat_endpoint(
                    ChatRequest(message="please continue", session_id="session_99"),
                    user_context="tenant_xyz123",
                )
            )

    assert response.status == "awaiting_approval"
    assert response.pending_action == pending
    mock_engine._run_agent_turn.assert_not_awaited()
    assert mock_engine.messages == []  # the new message was never appended either


def test_chat_endpoint_stops_advancing_into_a_transferred_agent_when_the_same_round_also_paused():
    """Regression test for a real bug: _execute_tool_calls_with_healing runs a round's non-
    transfer tool calls (including any requires_approval tool, which pauses via
    _pause_for_human/_set_pending_approval) BEFORE its transfer tool (transfer_agent/
    return_to_creator), so a single round can legitimately set BOTH _pending_approval and
    is_transferring at once. chat_endpoint's outer `while True: ... if not
    engine.is_transferring: break` loop only checked is_transferring — it kept calling
    _run_agent_turn again for the newly-transferred-to agent while the earlier pause sat
    unresolved, instead of halting immediately. The pause must win: the loop must stop after the
    very first _run_agent_turn call."""
    pending = {
        "tool": "issue_refund",
        "args": {"amount": 50},
        "status": "awaiting_approval",
        "tool_call_id": "call_1",
    }
    with patch(
        "intagrin.server.api.parse_project", return_value=_mock_graph_with_no_approver()
    ), patch(
        "intagrin.server.api.get_shared_resources_cache",
        return_value=MagicMock(get=AsyncMock(return_value=MagicMock())),
    ):
        mock_engine = MagicMock()
        mock_engine._await_last_checkpoint = AsyncMock()
        mock_engine.initialize = AsyncMock()
        mock_engine._apply_guardrails.return_value = "please continue"
        mock_engine._compress_memory = AsyncMock()
        mock_engine._save_checkpoint = MagicMock()
        mock_engine.state = {}
        mock_engine.messages = []
        mock_engine.active_agent_name = "triage"

        call_count = {"n": 0}

        async def _turn(interactive=False):
            call_count["n"] += 1
            if call_count["n"] > 1:
                # A buggy loop that ignores _pending_approval would call this repeatedly forever
                # (is_transferring is set True below on every call) — fail fast with a clear
                # assertion instead of hanging the test suite.
                raise AssertionError(
                    "the outer loop must not call _run_agent_turn again once a pause exists"
                )
            # Both a pause AND a transfer happened in this one round.
            mock_engine.state["_pending_approval"] = pending
            mock_engine.is_transferring = True
            mock_engine.active_agent_name = "specialist"
            mock_engine.messages.append({"role": "assistant", "content": "working on it"})

        mock_engine._run_agent_turn = AsyncMock(side_effect=_turn)

        with patch("intagrin.server.api.RuntimeEngine", return_value=mock_engine):
            response = asyncio.run(
                chat_endpoint(
                    ChatRequest(message="please continue", session_id="session_99"),
                    user_context="tenant_xyz123",
                )
            )

    mock_engine._run_agent_turn.assert_awaited_once()
    assert response.status == "awaiting_approval"
    assert response.pending_action == pending


def test_stream_endpoint_refuses_a_new_message_while_an_approval_is_pending():
    """Same guard as /chat, for the /stream SSE endpoint — refuses before ever acquiring a turn,
    yielding an awaiting_approval event instead of silently starting a new one."""
    pending = {"tool": "book_flight", "args": {"flight": "AF123"}, "status": "awaiting_approval"}
    with patch(
        "intagrin.server.api.parse_project", return_value=_mock_graph_with_no_approver()
    ), patch(
        "intagrin.server.api.get_shared_resources_cache",
        return_value=MagicMock(get=AsyncMock(return_value=MagicMock())),
    ):
        mock_engine = MagicMock()
        mock_engine._await_last_checkpoint = AsyncMock()
        mock_engine.initialize = AsyncMock()
        mock_engine._run_agent_turn_stream = MagicMock(
            side_effect=AssertionError("must not run a turn while awaiting approval")
        )
        mock_engine.state = {"_metrics": {}, "_pending_approval": pending}
        mock_engine.messages = []
        mock_engine.active_agent_name = "triage"

        with patch("intagrin.server.api.RuntimeEngine", return_value=mock_engine):

            async def collect():
                response = await stream_endpoint(
                    ChatRequest(message="please continue", session_id="session_99"),
                    user_context="tenant_xyz123",
                )
                chunks = []
                async for chunk in response.body_iterator:
                    chunks.append(chunk)
                return chunks

            chunks = asyncio.run(collect())

    body = "".join(chunks)
    assert '"type": "awaiting_approval"' in body
    assert "[DONE]" in body
    mock_engine._run_agent_turn_stream.assert_not_called()
    assert mock_engine.messages == []


def test_stream_endpoint_reports_awaiting_approval_status_when_a_turn_pauses():
    """Regression test for a real bug found live: stream_endpoint's event_generator hardcoded
    status="completed" unconditionally, unlike /chat and /chat/stream — a turn that ended because
    a tool paused for human approval was logged and reported identically to one that actually
    finished, with no client-visible signal that anything needed /resume."""
    with patch(
        "intagrin.server.api.parse_project", return_value=_mock_graph_with_no_approver()
    ), patch(
        "intagrin.server.api.get_shared_resources_cache",
        return_value=MagicMock(get=AsyncMock(return_value=MagicMock())),
    ), patch("intagrin.server.api.record_run_log") as mock_log:
        mock_engine = MagicMock()
        mock_engine._await_last_checkpoint = AsyncMock()
        mock_engine.initialize = AsyncMock()
        mock_engine._apply_guardrails.return_value = "Hello"
        mock_engine._compress_memory = AsyncMock()
        mock_engine._save_checkpoint = MagicMock()
        mock_engine.messages = []
        mock_engine.is_transferring = False
        mock_engine.active_agent_name = "triage"
        mock_engine.state = {"_metrics": {"total_tokens": 0, "total_cost": 0.0}}

        async def paused_stream(*args, **kwargs):
            # The turn itself sets _pending_approval, same as a real requires_approval tool call.
            mock_engine.state["_pending_approval"] = {
                "tool": "book_flight",
                "args": {},
                "status": "awaiting_approval",
            }
            yield {"type": "tool_result", "content": "paused"}

        mock_engine._run_agent_turn_stream = paused_stream

        with patch("intagrin.server.api.RuntimeEngine", return_value=mock_engine):

            async def collect():
                response = await stream_endpoint(
                    ChatRequest(message="Hello", session_id="session_99"),
                    user_context="tenant_xyz123",
                )
                chunks = []
                async for chunk in response.body_iterator:
                    chunks.append(chunk)
                return chunks

            chunks = asyncio.run(collect())

    body = "".join(chunks)
    assert '"type": "awaiting_approval"' in body
    assert "[DONE]" in body
    assert mock_log.call_args.kwargs["endpoint"] == "/stream"
    assert mock_log.call_args.kwargs["status"] == "awaiting_approval"


def test_chat_endpoint_surfaces_how_many_approvals_are_queued():
    """queued_approvals must reflect _pending_approval_queue's real depth — previously a caller
    had no way to know more pauses were coming behind the current one short of reading the raw
    checkpoint state directly, which is what made the live-session orphaning bug invisible until
    the database was inspected by hand."""
    with patch(
        "intagrin.server.api.parse_project", return_value=_mock_graph_with_no_approver()
    ), patch(
        "intagrin.server.api.get_shared_resources_cache",
        return_value=MagicMock(get=AsyncMock(return_value=MagicMock())),
    ):
        mock_engine = MagicMock()
        mock_engine._await_last_checkpoint = AsyncMock()
        mock_engine.initialize = AsyncMock()
        mock_engine._run_agent_turn = AsyncMock()
        mock_engine.state = {
            "_pending_approval": {"tool": "book_flight", "args": {}, "status": "awaiting_approval"},
            "_pending_approval_queue": [
                {"tool": "book_hotel", "args": {}},
                {"tool": "create_itinerary", "args": {}},
            ],
        }
        mock_engine.messages = []
        mock_engine.active_agent_name = "triage"

        with patch("intagrin.server.api.RuntimeEngine", return_value=mock_engine):
            response = asyncio.run(
                chat_endpoint(
                    ChatRequest(message="please continue", session_id="session_99"),
                    user_context="tenant_xyz123",
                )
            )

    assert response.status == "awaiting_approval"
    assert response.queued_approvals == 2

    # The default stays 0 for the ordinary, nothing-pending case — every pre-existing
    # ChatResponse(...) construction across this file that never set it relies on that default.
    assert ChatResponse(response="ok", active_agent="triage").queued_approvals == 0
