import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from intagrin.server.api import ResumeRequest, resume_endpoint


def _mock_graph_with_approvers(approvers: dict[str, str]):
    graph = MagicMock()
    graph.config.server.auth.approver_env_var = None
    graph.config.server.auth.approvers = approvers
    return graph


def _mock_engine(pending_action: dict):
    engine = MagicMock()
    engine._await_last_checkpoint = AsyncMock()
    engine.initialize = AsyncMock()
    engine._run_agent_turn = AsyncMock()
    engine.execute_tool = AsyncMock(return_value="done")
    engine._promote_next_queued_approval = MagicMock(return_value=False)
    engine.state = {"_pending_approval": pending_action}
    engine.messages = []
    engine.is_transferring = False
    engine.active_agent_name = "triage"
    return engine


def _pending_two_of_two():
    return {
        "tool": "wire_transfer",
        "args": {"amount": 10000},
        "tool_call_id": "call_1",
        "required_approvals": 2,
        "required_approvers": ["finance", "security"],
        "approvals_received": [],
    }


def test_first_of_two_named_approvers_leaves_it_pending():
    graph = _mock_graph_with_approvers({"finance": "FINANCE_KEY", "security": "SECURITY_KEY"})
    engine = _mock_engine(_pending_two_of_two())

    with patch("intagrin.server.api.parse_project", return_value=graph), patch(
        "intagrin.server.api.get_shared_resources_cache",
        return_value=MagicMock(get=AsyncMock(return_value=MagicMock())),
    ), patch("intagrin.server.api.RuntimeEngine", return_value=engine), patch.dict(
        os.environ, {"FINANCE_KEY": "f-secret", "SECURITY_KEY": "s-secret"}
    ):
        response = asyncio.run(
            resume_endpoint(
                ResumeRequest(session_id="s1", approved=True),
                request=MagicMock(headers={"X-Approver-Key": "f-secret"}),
                user_context="tenant",
            )
        )

    assert response.status == "awaiting_approval"
    engine.execute_tool.assert_not_awaited()
    engine._run_agent_turn.assert_not_awaited()
    assert engine.state["_pending_approval"]["approvals_received"] == ["finance"]


def test_second_distinct_approver_resolves_and_executes():
    graph = _mock_graph_with_approvers({"finance": "FINANCE_KEY", "security": "SECURITY_KEY"})
    pending = _pending_two_of_two()
    pending["approvals_received"] = ["finance"]
    engine = _mock_engine(pending)

    with patch("intagrin.server.api.parse_project", return_value=graph), patch(
        "intagrin.server.api.get_shared_resources_cache",
        return_value=MagicMock(get=AsyncMock(return_value=MagicMock())),
    ), patch("intagrin.server.api.RuntimeEngine", return_value=engine), patch.dict(
        os.environ, {"FINANCE_KEY": "f-secret", "SECURITY_KEY": "s-secret"}
    ):
        response = asyncio.run(
            resume_endpoint(
                ResumeRequest(session_id="s1", approved=True),
                request=MagicMock(headers={"X-Approver-Key": "s-secret"}),
                user_context="tenant",
            )
        )

    assert response.status == "completed"
    engine.execute_tool.assert_awaited_once_with(
        "wire_transfer", {"amount": 10000}, interactive=False, tool_call_id="call_1"
    )
    assert "_pending_approval" not in engine.state


def test_duplicate_approval_from_same_approver_does_not_double_count():
    graph = _mock_graph_with_approvers({"finance": "FINANCE_KEY", "security": "SECURITY_KEY"})
    pending = _pending_two_of_two()
    pending["approvals_received"] = ["finance"]
    engine = _mock_engine(pending)

    with patch("intagrin.server.api.parse_project", return_value=graph), patch(
        "intagrin.server.api.get_shared_resources_cache",
        return_value=MagicMock(get=AsyncMock(return_value=MagicMock())),
    ), patch("intagrin.server.api.RuntimeEngine", return_value=engine), patch.dict(
        os.environ, {"FINANCE_KEY": "f-secret", "SECURITY_KEY": "s-secret"}
    ):
        response = asyncio.run(
            resume_endpoint(
                ResumeRequest(session_id="s1", approved=True),
                request=MagicMock(headers={"X-Approver-Key": "f-secret"}),
                user_context="tenant",
            )
        )

    assert response.status == "awaiting_approval"
    engine.execute_tool.assert_not_awaited()
    assert engine.state["_pending_approval"]["approvals_received"] == ["finance"]


def test_wrong_or_missing_approver_credential_still_403s_in_multi_approver_mode():
    graph = _mock_graph_with_approvers({"finance": "FINANCE_KEY", "security": "SECURITY_KEY"})
    engine = _mock_engine(_pending_two_of_two())

    with patch("intagrin.server.api.parse_project", return_value=graph), patch(
        "intagrin.server.api.get_shared_resources_cache",
        return_value=MagicMock(get=AsyncMock(return_value=MagicMock())),
    ), patch("intagrin.server.api.RuntimeEngine", return_value=engine), patch.dict(
        os.environ, {"FINANCE_KEY": "f-secret", "SECURITY_KEY": "s-secret"}
    ):
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(
                resume_endpoint(
                    ResumeRequest(session_id="s1", approved=True),
                    request=MagicMock(headers={"X-Approver-Key": "wrong"}),
                    user_context="tenant",
                )
            )
        assert exc_info.value.status_code == 403

        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(
                resume_endpoint(
                    ResumeRequest(session_id="s1", approved=True),
                    request=MagicMock(headers={}),
                    user_context="tenant",
                )
            )
        assert exc_info.value.status_code == 403

    engine.execute_tool.assert_not_awaited()


def test_single_default_approver_unaffected_by_multi_approver_support():
    """required_approvals defaults to 1 with no required_approvers — the exact single-approval
    behavior that existed before this feature, now proven with the new fields present on the
    pending dict rather than absent (as in the older-checkpoint-shape tests in
    test_api_advanced.py)."""
    graph = _mock_graph_with_approvers({})
    graph.config.server.auth.approver_env_var = "APPROVER_KEY"
    pending = {
        "tool": "send_email",
        "args": {},
        "tool_call_id": "call_1",
        "required_approvals": 1,
        "required_approvers": None,
        "approvals_received": [],
    }
    engine = _mock_engine(pending)

    with patch("intagrin.server.api.parse_project", return_value=graph), patch(
        "intagrin.server.api.get_shared_resources_cache",
        return_value=MagicMock(get=AsyncMock(return_value=MagicMock())),
    ), patch("intagrin.server.api.RuntimeEngine", return_value=engine), patch.dict(
        os.environ, {"APPROVER_KEY": "s3cr3t"}
    ):
        response = asyncio.run(
            resume_endpoint(
                ResumeRequest(session_id="s1", approved=True),
                request=MagicMock(headers={"X-Approver-Key": "s3cr3t"}),
                user_context="tenant",
            )
        )

    assert response.status == "completed"
    engine.execute_tool.assert_awaited_once()
