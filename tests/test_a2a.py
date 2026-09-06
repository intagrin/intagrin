import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient

from intagrin.server.api import app

VALID_AI_YAML = """name: t
version: "1.0"
default_agent: triage
description: A test agent
model:
  primary: mock/model
memory:
  type: sqlite
server:
  auth:
    type: none
tools:
  - name: lookup_order
    module: unused
agents:
  triage:
    description: hi
    tools:
      - name: lookup_order
"""


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "ai.yaml").write_text(VALID_AI_YAML)
    return tmp_path


def _mock_graph_with_no_approver():
    """Same shape as test_api_advanced.py's helper, extended with an explicit auth.type — unlike
    those tests (which call chat_endpoint directly, bypassing Depends(verify_auth) entirely),
    these tests go through TestClient/real HTTP, so verify_auth's own parse_project(...) call
    (patched to return this same mock) must see a real 'none' string, not a truthy MagicMock, or
    every request 401s before a2a_endpoint's body ever runs."""
    graph = MagicMock()
    graph.config.server.auth.approver_env_var = None
    graph.config.server.auth.type = "none"
    return graph


def _mock_engine(*, messages, state, active_agent="triage"):
    mock_engine = MagicMock()
    mock_engine._await_last_checkpoint = AsyncMock()
    mock_engine.initialize = AsyncMock()
    mock_engine._run_agent_turn = AsyncMock()
    mock_engine._apply_guardrails.side_effect = lambda m: m
    mock_engine._compress_memory = AsyncMock()
    mock_engine._save_checkpoint = MagicMock()
    mock_engine.state = state
    mock_engine.messages = messages
    mock_engine.is_transferring = False
    mock_engine.active_agent_name = active_agent
    return mock_engine


# --- GET /.well-known/agent-card.json ------------------------------------------------------


def test_agent_card_returns_required_fields(project):
    client = TestClient(app)
    resp = client.get("/.well-known/agent-card.json")
    assert resp.status_code == 200
    card = resp.json()
    assert card["name"] == "t"
    assert card["description"] == "A test agent"
    assert card["protocolVersion"]
    assert card["url"] == "/a2a"
    assert card["capabilities"]["streaming"] is True
    names = {s["id"] for s in card["skills"]}
    assert "lookup_order" in names
    # auth.type: none -> no security scheme published, nothing required to call this agent.
    assert card["securitySchemes"] == {}
    assert card["security"] == []


def test_agent_card_maps_api_key_auth_to_http_bearer_scheme(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    yaml_text = VALID_AI_YAML.replace(
        "server:\n  auth:\n    type: none",
        "server:\n  auth:\n    type: api_key\n    env_var: TEST_A2A_KEY",
    )
    (tmp_path / "ai.yaml").write_text(yaml_text)
    client = TestClient(app)
    resp = client.get("/.well-known/agent-card.json")
    card = resp.json()
    # IntaGrin's api_key auth is `Authorization: Bearer <token>` (HTTPBearer) — the correct
    # OpenAPI/A2A scheme for that is HTTP bearer, not a named apiKey header.
    assert card["securitySchemes"] == {"bearerAuth": {"type": "http", "scheme": "bearer"}}
    assert card["security"] == [{"bearerAuth": []}]


# --- POST /a2a: message/send ---------------------------------------------------------------


def test_message_send_matches_direct_chat_endpoint_response(project):
    """A message/send JSON-RPC call must produce the same effective response chat_endpoint
    itself would for an equivalent direct /chat call — a genuine end-to-end equivalence
    assertion, not just 'returns 200'."""
    engine = _mock_engine(
        messages=[{"role": "assistant", "content": "Order #123 ships tomorrow."}],
        state={"_metrics": {}},
    )
    with patch(
        "intagrin.server.api.parse_project", return_value=_mock_graph_with_no_approver()
    ), patch(
        "intagrin.server.api.get_shared_resources_cache",
        return_value=MagicMock(get=AsyncMock(return_value=MagicMock())),
    ), patch("intagrin.server.api.RuntimeEngine", return_value=engine):
        client = TestClient(app)
        resp = client.post(
            "/a2a",
            json={
                "jsonrpc": "2.0",
                "id": "req-1",
                "method": "message/send",
                "params": {
                    "message": {
                        "contextId": "ctx-1",
                        "parts": [{"kind": "text", "text": "Where is my order?"}],
                    }
                },
            },
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["jsonrpc"] == "2.0"
    assert body["id"] == "req-1"
    result = body["result"]
    assert result["status"]["state"] == "completed"
    assert result["contextId"] == "ctx-1"
    assert result["history"][0]["parts"][0]["text"] == "Order #123 ships tomorrow."


def test_message_send_requires_context_id(project):
    client = TestClient(app)
    resp = client.post(
        "/a2a",
        json={
            "jsonrpc": "2.0",
            "id": "req-2",
            "method": "message/send",
            "params": {"message": {"parts": [{"kind": "text", "text": "hi"}]}},
        },
    )
    body = resp.json()
    assert body["error"]["code"] == -32600
    assert body["error"]["data"]["intagrin_code"] == "IG-A2A-001"


# --- POST /a2a: message/stream -------------------------------------------------------------


def test_message_stream_reframes_events_as_a2a_status_updates(project):
    async def fake_inner_generator():
        yield f"data: {json.dumps({'type': 'agent', 'agent': 'triage'})}\n\n"
        yield f"data: {json.dumps({'type': 'content', 'content': 'Hello'})}\n\n"
        done_event = {
            "type": "done",
            "active_agent": "triage",
            "status": "completed",
            "pending_action": None,
            "queued_approvals": 0,
        }
        yield f"data: {json.dumps(done_event)}\n\n"
        yield "data: [DONE]\n\n"

    fake_response = StreamingResponse(fake_inner_generator(), media_type="text/event-stream")

    with patch(
        "intagrin.server.a2a.stream_endpoint", new_callable=AsyncMock, return_value=fake_response
    ):
        client = TestClient(app)
        resp = client.post(
            "/a2a",
            json={
                "jsonrpc": "2.0",
                "id": "req-8",
                "method": "message/stream",
                "params": {
                    "message": {"contextId": "ctx-2", "parts": [{"kind": "text", "text": "hi"}]}
                },
            },
        )

    assert resp.status_code == 200
    frames = [line for line in resp.text.split("\n\n") if line.strip()]
    events = [json.loads(line[len("data:"):].strip())["result"] for line in frames]
    # The "agent" handoff-narration event has no A2A equivalent and is dropped; "content" becomes
    # one working status-update, "done" becomes the final completed status-update.
    states = [e["status"]["state"] for e in events]
    assert states == ["working", "completed"]
    assert events[0]["status"]["message"]["parts"][0]["text"] == "Hello"
    assert events[-1]["final"] is True


# --- POST /a2a: tasks/get -------------------------------------------------------------------


def test_tasks_get_reports_input_required_when_pending_approval(project):
    engine = _mock_engine(messages=[], state={"_pending_approval": {"tool": "refund"}})
    with patch(
        "intagrin.server.a2a.parse_project", return_value=_mock_graph_with_no_approver()
    ), patch(
        "intagrin.server.a2a.get_shared_resources_cache",
        return_value=MagicMock(get=AsyncMock(return_value=MagicMock())),
    ), patch("intagrin.server.a2a.RuntimeEngine", return_value=engine):
        client = TestClient(app)
        resp = client.post(
            "/a2a",
            json={"jsonrpc": "2.0", "id": "req-3", "method": "tasks/get", "params": {"id": "ctx-1"}},
        )
    assert resp.json()["result"]["status"]["state"] == "input-required"


def test_tasks_get_reports_working_for_pending_mcp_task(project):
    engine = _mock_engine(messages=[], state={"_pending_mcp_tasks": {"t1": {}}})
    with patch(
        "intagrin.server.a2a.parse_project", return_value=_mock_graph_with_no_approver()
    ), patch(
        "intagrin.server.a2a.get_shared_resources_cache",
        return_value=MagicMock(get=AsyncMock(return_value=MagicMock())),
    ), patch("intagrin.server.a2a.RuntimeEngine", return_value=engine):
        client = TestClient(app)
        resp = client.post(
            "/a2a",
            json={"jsonrpc": "2.0", "id": "req-4", "method": "tasks/get", "params": {"id": "ctx-1"}},
        )
    assert resp.json()["result"]["status"]["state"] == "working"


def test_tasks_get_reports_completed_with_no_pending_state(project):
    engine = _mock_engine(messages=[], state={})
    with patch(
        "intagrin.server.a2a.parse_project", return_value=_mock_graph_with_no_approver()
    ), patch(
        "intagrin.server.a2a.get_shared_resources_cache",
        return_value=MagicMock(get=AsyncMock(return_value=MagicMock())),
    ), patch("intagrin.server.a2a.RuntimeEngine", return_value=engine):
        client = TestClient(app)
        resp = client.post(
            "/a2a",
            json={"jsonrpc": "2.0", "id": "req-5", "method": "tasks/get", "params": {"id": "ctx-1"}},
        )
    assert resp.json()["result"]["status"]["state"] == "completed"


# --- Malformed / unsupported JSON-RPC --------------------------------------------------------


def test_malformed_jsonrpc_missing_method_returns_ig_a2a_001(project):
    client = TestClient(app)
    resp = client.post("/a2a", json={"jsonrpc": "2.0", "id": "req-6", "params": {}})
    assert resp.status_code == 200
    body = resp.json()
    assert body["error"]["code"] == -32600
    assert body["error"]["data"]["intagrin_code"] == "IG-A2A-001"


def test_non_jsonrpc_body_returns_ig_a2a_001(project):
    client = TestClient(app)
    resp = client.post("/a2a", json={"not": "jsonrpc"})
    body = resp.json()
    assert body["error"]["code"] == -32600
    assert body["error"]["data"]["intagrin_code"] == "IG-A2A-001"


def test_unsupported_method_returns_ig_a2a_002(project):
    client = TestClient(app)
    resp = client.post(
        "/a2a", json={"jsonrpc": "2.0", "id": "req-7", "method": "tasks/cancel", "params": {}}
    )
    body = resp.json()
    assert body["error"]["code"] == -32601
    assert body["error"]["data"]["intagrin_code"] == "IG-A2A-002"


# --- Auth enforcement -------------------------------------------------------------------------


def test_a2a_endpoint_rejects_missing_api_key(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    yaml_text = VALID_AI_YAML.replace(
        "server:\n  auth:\n    type: none",
        "server:\n  auth:\n    type: api_key\n    env_var: TEST_A2A_KEY2",
    )
    (tmp_path / "ai.yaml").write_text(yaml_text)
    monkeypatch.setenv("TEST_A2A_KEY2", "supersecret")
    client = TestClient(app)
    resp = client.post(
        "/a2a", json={"jsonrpc": "2.0", "id": "1", "method": "tasks/get", "params": {"id": "x"}}
    )
    assert resp.status_code == 401


def test_a2a_endpoint_rejects_wrong_api_key(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    yaml_text = VALID_AI_YAML.replace(
        "server:\n  auth:\n    type: none",
        "server:\n  auth:\n    type: api_key\n    env_var: TEST_A2A_KEY4",
    )
    (tmp_path / "ai.yaml").write_text(yaml_text)
    monkeypatch.setenv("TEST_A2A_KEY4", "supersecret")
    client = TestClient(app)
    resp = client.post(
        "/a2a",
        headers={"Authorization": "Bearer wrong-key"},
        json={"jsonrpc": "2.0", "id": "1", "method": "tasks/get", "params": {"id": "x"}},
    )
    assert resp.status_code == 401


def test_a2a_endpoint_accepts_valid_api_key(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    yaml_text = VALID_AI_YAML.replace(
        "server:\n  auth:\n    type: none",
        "server:\n  auth:\n    type: api_key\n    env_var: TEST_A2A_KEY3",
    )
    (tmp_path / "ai.yaml").write_text(yaml_text)
    monkeypatch.setenv("TEST_A2A_KEY3", "supersecret")
    engine = _mock_engine(messages=[], state={})
    with patch(
        "intagrin.server.a2a.parse_project", return_value=_mock_graph_with_no_approver()
    ), patch(
        "intagrin.server.a2a.get_shared_resources_cache",
        return_value=MagicMock(get=AsyncMock(return_value=MagicMock())),
    ), patch("intagrin.server.a2a.RuntimeEngine", return_value=engine):
        client = TestClient(app)
        resp = client.post(
            "/a2a",
            headers={"Authorization": "Bearer supersecret"},
            json={"jsonrpc": "2.0", "id": "1", "method": "tasks/get", "params": {"id": "x"}},
        )
    assert resp.status_code == 200
    assert resp.json()["result"]["status"]["state"] == "completed"
