from typer.testing import CliRunner

from intagrin.cli import app
from intagrin.runtime.memory import SQLiteCheckpointer

runner = CliRunner()

_AI_YAML = """
name: "replay-router-trace-test"
version: "1.0"
default_agent: "triage"
model:
  primary: "mock/model"
memory:
  type: "sqlite"
agents:
  triage:
    description: "Routes tickets."
  billing:
    description: "Handles billing."
"""


def _seed(project_dir, session_id, messages, state):
    cp = SQLiteCheckpointer(str(project_dir / ".ai" / "memory.db"))
    cp.save_checkpoint(session_id, messages, state)


def test_replay_shows_a_router_that_did_not_fire(tmp_path, monkeypatch):
    """Regression test for the actual gap: a router that's evaluated but doesn't fire previously
    left zero trace anywhere a human could see after the fact — only a live SSE event nobody may
    have been watching. state["_router_trace"] (see RuntimeEngine._record_router_trace) must
    surface it in `inta replay`."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "ai.yaml").write_text(_AI_YAML)
    _seed(
        tmp_path,
        "sess1",
        [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}],
        {
            "_router_trace": [
                {
                    "turn": 1,
                    "kind": "conditional",
                    "description": "triage -> billing if 'balance > 0'",
                    "fired": False,
                    "target": "billing",
                    "error": None,
                }
            ]
        },
    )

    result = runner.invoke(app, ["replay", "sess1"])

    assert result.exit_code == 0, result.output
    assert "did not fire" in result.output
    assert "triage -> billing" in result.output


def test_replay_shows_a_router_that_raised_distinctly_from_a_plain_non_fire(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "ai.yaml").write_text(_AI_YAML)
    _seed(
        tmp_path,
        "sess1",
        [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}],
        {
            "_router_trace": [
                {
                    "turn": 1,
                    "kind": "conditional",
                    "description": "triage -> billing if 'typo_balance > 0'",
                    "fired": False,
                    "target": "billing",
                    "error": "Unknown variable: typo_balance",
                }
            ]
        },
    )

    result = runner.invoke(app, ["replay", "sess1"])

    assert result.exit_code == 0, result.output
    assert "condition raised" in result.output
    assert "typo_balance" in result.output


def test_replay_does_not_duplicate_a_router_that_fired(tmp_path, monkeypatch):
    """A router that DID fire already has a 'Router: Transferred to...' system message — the
    trace-derived note must not also print for it, or every successful route would show twice."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "ai.yaml").write_text(_AI_YAML)
    _seed(
        tmp_path,
        "sess1",
        [
            {"role": "user", "content": "hi"},
            {
                "role": "system",
                "content": "Router: Transferred to billing via conditional router ('balance > 0').",
            },
        ],
        {
            "_router_trace": [
                {
                    "turn": 1,
                    "kind": "conditional",
                    "description": "triage -> billing if 'balance > 0'",
                    "fired": True,
                    "target": "billing",
                    "error": None,
                }
            ]
        },
    )

    result = runner.invoke(app, ["replay", "sess1"])

    assert result.exit_code == 0, result.output
    assert "did not fire" not in result.output
    assert "condition raised" not in result.output
    assert "Transferred to billing" in result.output
