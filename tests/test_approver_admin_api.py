"""Regression tests for the /approvers HTTP endpoints (server/api.py) — issuing, listing, and
revoking DB-backed reviewer credentials (runtime/approvers.py) over HTTP for a consumer's own
admin site/tooling, instead of only via `inta approvers` locally. Gated by a deliberately separate
admin credential tier (server.auth.admin_env_var) from both the main session auth and any
individual approver's own X-Approver-Key."""
import yaml
from fastapi.testclient import TestClient

from intagrin.server.api import app

client = TestClient(app)


def _write_project(tmp_path, admin_env_var=None):
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
    if admin_env_var:
        data["server"] = {"auth": {"admin_env_var": admin_env_var}}
    (tmp_path / "ai.yaml").write_text(yaml.dump(data))


def test_approver_endpoints_disabled_without_admin_env_var(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_project(tmp_path)

    resp = client.post("/approvers", json={"approver_id": "finance"})
    assert resp.status_code == 503

    resp = client.get("/approvers")
    assert resp.status_code == 503

    resp = client.delete("/approvers/finance")
    assert resp.status_code == 503


def test_approver_endpoints_reject_wrong_or_missing_admin_key(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_project(tmp_path, admin_env_var="ADMIN_KEY")
    monkeypatch.setenv("ADMIN_KEY", "correct-secret")

    resp = client.post("/approvers", json={"approver_id": "finance"})
    assert resp.status_code == 401  # no Authorization header at all

    resp = client.post(
        "/approvers",
        json={"approver_id": "finance"},
        headers={"Authorization": "Bearer wrong"},
    )
    assert resp.status_code == 401


def test_admin_key_cannot_double_as_the_main_session_api_key(tmp_path, monkeypatch):
    """The whole point of a separate admin_env_var is that the requester's own session key must
    not also be able to mint approver credentials — otherwise the same credential that triggers a
    gated tool call could immediately approve it via a self-issued approver key."""
    monkeypatch.chdir(tmp_path)
    _write_project(tmp_path, admin_env_var="ADMIN_KEY")
    monkeypatch.setenv("ADMIN_KEY", "admin-secret")
    monkeypatch.setenv("INTAGRIN_API_KEY", "session-secret")

    resp = client.post(
        "/approvers",
        json={"approver_id": "finance"},
        headers={"Authorization": "Bearer session-secret"},
    )
    assert resp.status_code == 401


def test_create_list_revoke_round_trip(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_project(tmp_path, admin_env_var="ADMIN_KEY")
    monkeypatch.setenv("ADMIN_KEY", "correct-secret")
    headers = {"Authorization": "Bearer correct-secret"}

    resp = client.post("/approvers", json={"approver_id": "finance"}, headers=headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body["approver_id"] == "finance"
    secret = body["secret"]
    assert secret

    resp = client.get("/approvers", headers=headers)
    assert resp.status_code == 200
    approvers = resp.json()["approvers"]
    assert len(approvers) == 1
    assert approvers[0]["approver_id"] == "finance"
    assert approvers[0]["revoked_at"] is None
    assert "secret" not in approvers[0]
    assert "secret_hash" not in approvers[0]

    from intagrin.compiler.parser import parse_project
    from intagrin.runtime.approvers import verify_secret

    graph = parse_project(tmp_path)
    assert verify_secret(graph.config.memory, tmp_path, secret) == "finance"
    assert verify_secret(graph.config.memory, tmp_path, "wrong-secret") is None

    resp = client.delete("/approvers/finance", headers=headers)
    assert resp.status_code == 200
    assert resp.json() == {"approver_id": "finance", "revoked": True}

    resp = client.delete("/approvers/finance", headers=headers)
    assert resp.status_code == 404

    assert verify_secret(graph.config.memory, tmp_path, secret) is None
