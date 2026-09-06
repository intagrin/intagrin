"""Regression tests for GET /pending-approvals (server/api.py) — a narrow, poll-friendly view of
which sessions are paused on a human-in-the-loop approval. Built for a caller that needs to check
many deployed projects at once (a fleet console) without pulling every session's full transcript
via GET /sessions."""
import yaml
from fastapi.testclient import TestClient

from intagrin.runtime.memory import SQLiteCheckpointer
from intagrin.server.api import app, get_pending_approvals

client = TestClient(app)


def _write_project(tmp_path):
    (tmp_path / "prompts").mkdir(exist_ok=True)
    (tmp_path / "prompts" / "triage.jinja2").write_text("You are a triage agent.")
    data = {
        "version": "1.0",
        "name": "test_project",
        "default_agent": "triage",
        "memory": {"type": "sqlite", "db_path": ".ai/test_mem.db"},
        "model": {"primary": "test-model"},
        "agents": {"triage": {"system_prompt_file": "prompts/triage.jinja2"}},
    }
    (tmp_path / "ai.yaml").write_text(yaml.dump(data))


def test_pending_approvals_empty_for_a_fresh_project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_project(tmp_path)

    resp = client.get("/pending-approvals")
    assert resp.status_code == 200
    assert resp.json() == []


def test_pending_approvals_lists_a_session_with_an_active_pause(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_project(tmp_path)

    checkpointer = SQLiteCheckpointer(str(tmp_path / ".ai" / "test_mem.db"))
    checkpointer.save_checkpoint(
        "global_tenant:sess-1",
        [],
        {
            "_pending_approval": {
                "tool": "refund",
                "agent": "triage",
                "status": "awaiting_approval",
                "tool_call_id": "call_1",
                "required_approvals": 1,
                "required_approvers": None,
                "approvals_received": [],
                "created_at": "2026-01-01T00:00:00+00:00",
            }
        },
    )
    checkpointer.save_checkpoint("global_tenant:sess-2", [], {"foo": "bar"})

    resp = client.get("/pending-approvals")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 1
    assert body[0]["session_id"] == "sess-1"
    assert body[0]["pending"]["tool"] == "refund"
    assert body[0]["queued_count"] == 0


def test_pending_approvals_reports_queued_count(tmp_path, monkeypatch):
    """A session can have more than one pause queued up behind the claimed _pending_approval slot
    (see RuntimeEngine._set_pending_approval) — queued_count must surface that, not silently drop
    it."""
    monkeypatch.chdir(tmp_path)
    _write_project(tmp_path)

    checkpointer = SQLiteCheckpointer(str(tmp_path / ".ai" / "test_mem.db"))
    checkpointer.save_checkpoint(
        "global_tenant:sess-1",
        [],
        {
            "_pending_approval": {"tool": "refund", "status": "awaiting_approval"},
            "_pending_approval_queue": [
                {"tool": "cancel_order", "status": "awaiting_approval"},
            ],
        },
    )

    resp = client.get("/pending-approvals")
    assert resp.json()[0]["queued_count"] == 1


def test_pending_approvals_is_tenant_isolated(tmp_path, monkeypatch):
    """Same IDOR-isolation guarantee GET /sessions already provides — a caller must never see
    another tenant's pending approvals, mirroring test_tenant_idor_isolation's direct-call style
    (server.auth.type doesn't distinguish callers by itself; the session_id tenant prefix does)."""
    monkeypatch.chdir(tmp_path)
    _write_project(tmp_path)

    checkpointer = SQLiteCheckpointer(str(tmp_path / ".ai" / "test_mem.db"))
    checkpointer.save_checkpoint(
        "tenant_a:sess-1",
        [],
        {"_pending_approval": {"tool": "refund", "status": "awaiting_approval"}},
    )
    checkpointer.save_checkpoint(
        "tenant_b:sess-2",
        [],
        {"_pending_approval": {"tool": "cancel_order", "status": "awaiting_approval"}},
    )

    tenant_a_view = get_pending_approvals(user_context="tenant_a")
    assert [s["session_id"] for s in tenant_a_view] == ["sess-1"]

    tenant_b_view = get_pending_approvals(user_context="tenant_b")
    assert [s["session_id"] for s in tenant_b_view] == ["sess-2"]
