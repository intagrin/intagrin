from typer.testing import CliRunner

from intagrin.cli import app
from intagrin.runtime.memory import SQLiteCheckpointer

runner = CliRunner()

_BASE_AI_YAML = """
name: "simulate-cli-test"
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

_ROUTER_TIGHTENED_YAML = """
name: "simulate-cli-test"
version: "1.0"
default_agent: "triage"
model:
  primary: "mock/model"
memory:
  type: "sqlite"
agents:
  triage:
    description: "Routes tickets."
    routers:
      - condition: "True"
        target: "billing"
  billing:
    description: "Handles billing."
"""

_MODEL_CHANGED_YAML = """
name: "simulate-cli-test"
version: "1.0"
default_agent: "triage"
model:
  primary: "openai/gpt-4o"
memory:
  type: "sqlite"
agents:
  triage:
    description: "Routes tickets."
  billing:
    description: "Handles billing."
"""


def _seed(project_dir, session_id, messages):
    cp = SQLiteCheckpointer(str(project_dir / ".ai" / "memory.db"))
    cp.save_checkpoint(session_id, messages, {})


def test_simulate_cli_reports_routing_divergence(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "ai.yaml").write_text(_BASE_AI_YAML)
    (tmp_path / "candidate.yaml").write_text(_ROUTER_TIGHTENED_YAML)
    _seed(
        tmp_path,
        "sess1",
        [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello, how can I help?"},
        ],
    )

    result = runner.invoke(app, ["simulate", "--config", "candidate.yaml"])

    assert result.exit_code == 0, result.output
    assert "1 session(s) checked" in result.output
    assert "ROUTING_DIVERGES" in result.output


def test_simulate_cli_refuses_unsafe_diff(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "ai.yaml").write_text(_BASE_AI_YAML)
    (tmp_path / "candidate.yaml").write_text(_MODEL_CHANGED_YAML)
    _seed(tmp_path, "sess1", [{"role": "user", "content": "hi"}])

    result = runner.invoke(app, ["simulate", "--config", "candidate.yaml"])

    assert result.exit_code != 0
    assert "Not simulatable" in result.output
    assert "model" in result.output


def test_simulate_cli_missing_candidate_file_errors_cleanly(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "ai.yaml").write_text(_BASE_AI_YAML)

    result = runner.invoke(app, ["simulate", "--config", "does_not_exist.yaml"])

    assert result.exit_code != 0
    assert "not found" in result.output


def test_simulate_cli_rejects_bad_since_duration(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "ai.yaml").write_text(_BASE_AI_YAML)
    (tmp_path / "candidate.yaml").write_text(_ROUTER_TIGHTENED_YAML)

    result = runner.invoke(app, ["simulate", "--config", "candidate.yaml", "--since", "banana"])

    assert result.exit_code != 0
    assert "isn't a recognized duration" in result.output
