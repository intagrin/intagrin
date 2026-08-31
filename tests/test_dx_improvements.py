"""Tests for the DX-audit follow-through: the mock/echo model path, `inta doctor`, `inta why`,
`inta verify --watch`, and `inta dev --once`."""

import asyncio

from typer.testing import CliRunner

from intagrin.cli import _project_fingerprint, app
from intagrin.compiler.parser import parse_project
from intagrin.runtime.engine import RuntimeEngine
from intagrin.runtime.router import condition_state_keys

runner = CliRunner()

_MOCK_AI_YAML = """
name: "dx-test"
version: "1.0"
default_agent: "triage"
model:
  primary: "mock/echo"
memory:
  type: "sqlite"
agents:
  triage:
    description: "Routes tickets."
    routers:
      - condition: "user_status == 'banned'"
        target: "billing"
  billing:
    description: "Handles billing."
"""

_REAL_MODEL_AI_YAML = _MOCK_AI_YAML.replace('primary: "mock/echo"', 'primary: "openai/gpt-4o-mini"')

_MCP_AI_YAML = """
name: "dx-test"
version: "1.0"
default_agent: "triage"
model:
  primary: "mock/echo"
memory:
  type: "sqlite"
agents:
  triage:
    description: "Routes tickets."
    tools:
      - name: "definitely_not_a_real_binary_xyz"
        type: "mcp"
        command: "definitely-not-a-real-binary-xyz"
        args: []
"""


# --- runtime/router.py: condition_state_keys -------------------------------------------------


def test_condition_state_keys_extracts_bare_identifiers():
    assert condition_state_keys("user_status == 'banned'") == {"user_status"}


def test_condition_state_keys_ignores_string_literals_and_keywords():
    assert condition_state_keys("'refund' in intent and not resolved") == {"intent", "resolved"}


def test_condition_state_keys_empty_for_unparseable_expression():
    assert condition_state_keys("not a valid ( expression") == set()


# --- runtime/engine.py: mock model + run_once + router-trace printing ------------------------


def test_mock_model_run_once_replies_with_zero_cost(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "ai.yaml").write_text(_MOCK_AI_YAML)

    graph = parse_project(tmp_path)

    async def _run():
        engine = RuntimeEngine(graph=graph, project_dir=tmp_path, session_id="s1")
        await engine.initialize()
        reply = await engine.run_once("hello from a test")
        return engine, reply

    engine, reply = asyncio.run(_run())

    assert "hello from a test" in reply
    assert "mock model" in reply.lower()
    assert engine.state.get("_metrics", {}).get("total_cost", 0.0) == 0.0

    # The scaffolded router references user_status, which nothing has written — it should have
    # been recorded as a non-fired evaluation and printed inline, not silently swallowed.
    trace = engine.state.get("_router_trace", [])
    assert any(not entry["fired"] for entry in trace)
    printed = capsys.readouterr().out
    assert "did not fire" in printed


def test_run_once_drives_multi_agent_handoff_to_settled_state(tmp_path, monkeypatch):
    """A handoff-capable mock reply isn't attempted (see engine.py's _mock_reply docstring) —
    this just proves run_once terminates and returns control cleanly for a plain single-agent
    turn instead of hanging in the transfer loop."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "ai.yaml").write_text(_MOCK_AI_YAML)
    graph = parse_project(tmp_path)

    async def _run():
        engine = RuntimeEngine(graph=graph, project_dir=tmp_path, session_id="s2")
        await engine.initialize()
        return await engine.run_once("hi")

    reply = asyncio.run(_run())
    assert reply


# --- cli.py: inta dev --once ------------------------------------------------------------------


def test_dev_once_prints_reply_and_exits_zero(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "ai.yaml").write_text(_MOCK_AI_YAML)

    result = runner.invoke(app, ["dev", "--once", "ping"])

    assert result.exit_code == 0, result.output
    assert "ping" in result.output
    assert "mock model" in result.output.lower()


def test_dev_once_warns_but_does_not_fail_without_an_api_key(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    (tmp_path / "ai.yaml").write_text(_REAL_MODEL_AI_YAML)

    result = runner.invoke(app, ["dev", "--once", "ping"])

    assert "Warning" in result.output
    assert "mock/echo" in result.output  # points the user at the escape hatch


# --- cli.py: inta doctor -----------------------------------------------------------------------


def test_doctor_passes_clean_on_a_mock_model_project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "ai.yaml").write_text(_MOCK_AI_YAML)

    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 0, result.output
    assert "ai.yaml parses" in result.output
    assert "no key required" in result.output


def test_doctor_fails_on_invalid_ai_yaml(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "ai.yaml").write_text("not: [valid, ai.yaml")

    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 1


def test_doctor_fails_when_an_mcp_command_is_not_on_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "ai.yaml").write_text(_MCP_AI_YAML)

    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 1
    assert "not found on PATH" in result.output


def test_doctor_warns_without_failing_when_state_schema_is_unset(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    (tmp_path / "ai.yaml").write_text(_REAL_MODEL_AI_YAML)

    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 0
    assert "warnings" in result.output.lower()


# --- cli.py: inta why ---------------------------------------------------------------------------


def test_why_finds_a_router_that_reads_the_key(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "ai.yaml").write_text(_MOCK_AI_YAML)

    result = runner.invoke(app, ["why", "user_status"])

    assert result.exit_code == 0, result.output
    assert "router on 'triage'" in result.output
    assert "billing" in result.output


def test_why_reports_nothing_reads_an_unreferenced_key(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "ai.yaml").write_text(_MOCK_AI_YAML)

    result = runner.invoke(app, ["why", "totally_unreferenced_key"])

    assert result.exit_code == 0, result.output
    assert "nothing currently reads this key" in result.output


# --- cli.py: inta doctor's description nudges ---------------------------------------------------


def test_doctor_warns_on_missing_app_and_agent_descriptions(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    undescribed = _MOCK_AI_YAML.replace(
        'description: "Routes tickets."', "description: null"
    ).replace('description: "Handles billing."', "description: null")
    (tmp_path / "ai.yaml").write_text(undescribed)

    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 0, result.output
    assert "app description" in result.output
    assert "agent descriptions" in result.output
    assert "billing" in result.output and "triage" in result.output


def test_doctor_passes_description_checks_when_all_are_set(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    described = _MOCK_AI_YAML.replace(
        'name: "dx-test"\n', 'name: "dx-test"\ndescription: "A ticket triage/billing app."\n'
    )
    (tmp_path / "ai.yaml").write_text(described)

    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 0, result.output
    assert "app description" in result.output
    assert "every agent has one" in result.output


# --- cli.py: inta explain -------------------------------------------------------------------------


def test_explain_narrates_agents_tools_and_routers(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "ai.yaml").write_text(_MCP_AI_YAML)

    result = runner.invoke(app, ["explain"])

    assert result.exit_code == 0, result.output
    assert "dx-test" in result.output
    assert "triage" in result.output
    assert "Routes tickets." in result.output
    assert "an MCP server tool" in result.output
    assert "Safety limits" in result.output


def test_explain_notes_missing_descriptions_instead_of_failing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    undescribed = _MOCK_AI_YAML.replace(
        'description: "Routes tickets."', "description: null"
    ).replace('description: "Handles billing."', "description: null")
    (tmp_path / "ai.yaml").write_text(undescribed)

    result = runner.invoke(app, ["explain"])

    assert result.exit_code == 0, result.output
    assert "no description set" in result.output


def test_why_finds_write_state_mentions_in_prompts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "ai.yaml").write_text(_MOCK_AI_YAML)
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "triage.jinja2").write_text(
        'Call write_state("user_status", "banned") once you confirm the account is banned.'
    )

    result = runner.invoke(app, ["why", "user_status"])

    assert result.exit_code == 0, result.output
    assert "prompts/triage.jinja2" in result.output


# --- cli.py: inta verify --watch (fingerprint helper — the watch loop itself is exercised
# manually, since it never returns by design and isn't worth a sleep-based test here) -----------


def test_project_fingerprint_changes_when_ai_yaml_changes(tmp_path):
    (tmp_path / "ai.yaml").write_text(_MOCK_AI_YAML)
    before = _project_fingerprint(tmp_path)

    # Force a detectably later mtime instead of racing a same-tick rewrite.
    import os

    (tmp_path / "ai.yaml").write_text(_MOCK_AI_YAML + "\n# changed\n")
    os.utime(tmp_path / "ai.yaml", (before + 5, before + 5))

    after = _project_fingerprint(tmp_path)
    assert after > before


def test_project_fingerprint_is_zero_for_an_empty_directory(tmp_path):
    assert _project_fingerprint(tmp_path) == 0.0
