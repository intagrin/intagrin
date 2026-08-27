"""Regression tests for the IntaGrinError passthrough fix in server/api.py: each endpoint's own
broad `except Exception` must not swallow a rate-limit rejection (or any other IntaGrinError) and
rewrap it into a misleading 500 — the same class of bug previously fixed for identify_approver's
403 in resume_endpoint. Patches check_rate_limit itself (already covered directly by
tests/test_rate_limiter.py) so these tests isolate the wiring, not the enforcement logic.
"""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from intagrin.errors import IntaGrinError
from intagrin.server.api import (
    ChatRequest,
    ResumeRequest,
    chat_endpoint,
    resume_endpoint,
    stream_endpoint,
)


def _mock_graph_with_no_approver():
    graph = MagicMock()
    graph.config.server.auth.approver_env_var = None
    graph.config.server.auth.approvers = None
    return graph


def _rate_limited():
    return IntaGrinError("IG-RT-008", "Rate limit exceeded: 5/5 requests in the last 60s.")


def test_chat_endpoint_propagates_rate_limit_as_429_not_500():
    with patch(
        "intagrin.server.api.parse_project", return_value=_mock_graph_with_no_approver()
    ), patch(
        "intagrin.server.api.check_rate_limit", side_effect=_rate_limited()
    ), pytest.raises(IntaGrinError) as exc_info:
        asyncio.run(
            chat_endpoint(
                ChatRequest(message="Hello", session_id="session_99"),
                user_context="tenant_xyz123",
            )
        )
    assert exc_info.value.http_status == 429
    assert exc_info.value.code == "IG-RT-008"


def test_stream_endpoint_propagates_rate_limit_as_429_not_500():
    with patch(
        "intagrin.server.api.parse_project", return_value=_mock_graph_with_no_approver()
    ), patch(
        "intagrin.server.api.check_rate_limit", side_effect=_rate_limited()
    ), pytest.raises(IntaGrinError) as exc_info:
        asyncio.run(
            stream_endpoint(
                ChatRequest(message="Hello", session_id="session_99"),
                user_context="tenant_xyz123",
            )
        )
    assert exc_info.value.http_status == 429
    assert exc_info.value.code == "IG-RT-008"


def test_resume_endpoint_propagates_rate_limit_as_429_not_500():
    with patch(
        "intagrin.server.api.parse_project", return_value=_mock_graph_with_no_approver()
    ), patch(
        "intagrin.server.api.check_rate_limit", side_effect=_rate_limited()
    ), pytest.raises(IntaGrinError) as exc_info:
        asyncio.run(
            resume_endpoint(
                ResumeRequest(session_id="session_99", approved=False),
                request=MagicMock(headers={}),
                user_context="tenant_xyz123",
            )
        )
    assert exc_info.value.http_status == 429
    assert exc_info.value.code == "IG-RT-008"


def test_chat_stream_endpoint_propagates_rate_limit_as_429_not_500():
    from intagrin.server.api import chat_stream_endpoint

    with patch(
        "intagrin.server.api.parse_project", return_value=_mock_graph_with_no_approver()
    ), patch(
        "intagrin.server.api.check_rate_limit", side_effect=_rate_limited()
    ), pytest.raises(IntaGrinError) as exc_info:
        asyncio.run(
            chat_stream_endpoint(
                ChatRequest(message="Hello", session_id="session_99"),
                user_context="tenant_xyz123",
            )
        )
    assert exc_info.value.http_status == 429
    assert exc_info.value.code == "IG-RT-008"
