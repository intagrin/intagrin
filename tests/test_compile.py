import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from intagrin.cli import app

runner = CliRunner()

VALID_MINIMAL_CONFIG = {
    "version": "1.0",
    "name": "compiled",
    "default_agent": "triage",
    "model": {"primary": "mock/model"},
    "memory": {"type": "sqlite"},
    "agents": {"triage": {"description": "Routes tickets."}},
}


class _MockMessage:
    def __init__(self, content):
        self.content = content


class _MockChoice:
    def __init__(self, content):
        self.message = _MockMessage(content)


class _MockResponse:
    def __init__(self, content):
        self.choices = [_MockChoice(content)]


@pytest.fixture
def mock_blueprint(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # compile_command pre-flight-checks for a provider API key before ever calling litellm (see
    # IG-CLI-008) — set a dummy one so these tests exercise the mocked-completion path below,
    # not the pre-flight check itself (that has its own dedicated test).
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    bp = tmp_path / "blueprint.md"
    bp.write_text("# Vision\nCreate a triage and support agent.")
    return bp


def test_compile_command_no_file():
    result = runner.invoke(app, ["compile", "missing.md"])
    assert result.exit_code == 1
    assert "not found" in result.stdout


def test_compile_command_valid_config_writes_ai_yaml(mock_blueprint, monkeypatch):
    def mock_completion(*args, **kwargs):
        return _MockResponse(json.dumps(VALID_MINIMAL_CONFIG))

    import litellm

    monkeypatch.setattr(litellm, "completion", mock_completion)

    result = runner.invoke(app, ["compile", "blueprint.md"])

    assert result.exit_code == 0, result.stdout
    assert "Successfully compiled" in result.stdout
    ai_yaml = Path("ai.yaml")
    assert ai_yaml.exists()
    written = yaml.safe_load(ai_yaml.read_text())
    assert written["name"] == "compiled"
    assert written["default_agent"] == "triage"


def test_compile_command_invalid_config_never_writes_ai_yaml(mock_blueprint, monkeypatch):
    """The exact scenario the old fixture in this file accidentally proved was broken: a config
    missing required fields (model/memory/default_agent) must never reach disk, however many
    times the model insists it's done."""
    invalid = {"version": "1.0", "name": "compiled", "agents": {"triage": {}}}

    def mock_completion(*args, **kwargs):
        return _MockResponse(json.dumps(invalid))

    import litellm

    monkeypatch.setattr(litellm, "completion", mock_completion)

    result = runner.invoke(app, ["compile", "blueprint.md"])

    assert result.exit_code == 1
    assert "NOT written" in result.stdout
    assert not Path("ai.yaml").exists()


def test_compile_command_stops_and_asks_instead_of_guessing_a_tech_choice(
    mock_blueprint, monkeypatch
):
    """When the blueprint doesn't specify a consequential, requirements-dependent decision (which
    memory backend, auth, RAG), the compiler must stop and print the questions rather than
    guessing a default and writing a config anyway — and must not spend a self-heal retry doing
    it, since this isn't a validation failure."""
    clarification_response = {
        "clarifications_needed": [
            "Which memory backend do you want — sqlite (local/dev) or postgres (production)?",
            "Should inta serve/inta monitor require authentication?",
        ]
    }
    call_count = {"n": 0}

    def mock_completion(*args, **kwargs):
        call_count["n"] += 1
        return _MockResponse(json.dumps(clarification_response))

    import litellm

    monkeypatch.setattr(litellm, "completion", mock_completion)

    result = runner.invoke(app, ["compile", "blueprint.md"])

    assert result.exit_code == 1
    assert "Which memory backend" in result.stdout
    assert "authentication" in result.stdout
    assert not Path("ai.yaml").exists()
    assert call_count["n"] == 1  # no self-heal retries spent on a clarification request


def test_compile_command_self_heals_after_one_bad_attempt(mock_blueprint, monkeypatch):
    invalid = {"version": "1.0", "name": "compiled", "agents": {"triage": {}}}
    responses = [json.dumps(invalid), json.dumps(VALID_MINIMAL_CONFIG)]
    call_count = {"n": 0}

    def mock_completion(*args, **kwargs):
        call_count["n"] += 1
        return _MockResponse(responses[call_count["n"] - 1])

    import litellm

    monkeypatch.setattr(litellm, "completion", mock_completion)

    result = runner.invoke(app, ["compile", "blueprint.md"])

    assert result.exit_code == 0, result.stdout
    assert call_count["n"] == 2
    assert Path("ai.yaml").exists()


def test_compile_command_gives_up_after_max_attempts(mock_blueprint, monkeypatch):
    invalid = {"version": "1.0", "name": "compiled", "agents": {"triage": {}}}
    call_count = {"n": 0}

    def mock_completion(*args, **kwargs):
        call_count["n"] += 1
        return _MockResponse(json.dumps(invalid))

    import litellm

    monkeypatch.setattr(litellm, "completion", mock_completion)

    result = runner.invoke(app, ["compile", "blueprint.md"])

    assert result.exit_code == 1
    assert call_count["n"] == 3  # 1 initial + 2 self-heal retries, then give up
    assert not Path("ai.yaml").exists()


def test_compile_command_rejects_unsupported_router_condition_syntax(mock_blueprint, monkeypatch):
    bad_condition_config = dict(VALID_MINIMAL_CONFIG)
    bad_condition_config["agents"] = {
        "triage": {
            "description": "Routes tickets.",
            "routers": [
                {"condition": "state.get('user_status', '') == 'banned'", "target": "triage"}
            ],
        }
    }

    def mock_completion(*args, **kwargs):
        return _MockResponse(json.dumps(bad_condition_config))

    import litellm

    monkeypatch.setattr(litellm, "completion", mock_completion)

    result = runner.invoke(app, ["compile", "blueprint.md"])

    assert result.exit_code == 1
    assert not Path("ai.yaml").exists()


def test_compile_command_rejects_on_complete_writing_to_a_reserved_key(mock_blueprint, monkeypatch):
    """The same class of bug as the router-condition case above: a compiled config that would
    silently fail at runtime (write_state rejects any key starting with `_`) must be caught before
    ai.yaml is ever written, not discovered the first time a spawn actually completes."""
    bad_config = dict(VALID_MINIMAL_CONFIG)
    bad_config["agents"] = {
        "triage": {
            "description": "Routes tickets.",
            "tools": [{"name": "search", "module": "tools.custom"}],
            "spawns": {
                "tool_pool": ["search"],
                "on_complete": [{"key": "_pending_approval", "value": True}],
            },
        }
    }

    def mock_completion(*args, **kwargs):
        return _MockResponse(json.dumps(bad_config))

    import litellm

    monkeypatch.setattr(litellm, "completion", mock_completion)

    result = runner.invoke(app, ["compile", "blueprint.md"])

    assert result.exit_code == 1
    assert not Path("ai.yaml").exists()


def test_compile_command_scaffolds_missing_prompt_and_tool_then_leaves_them_alone(
    mock_blueprint, monkeypatch
):
    config = dict(VALID_MINIMAL_CONFIG)
    config["agents"] = {
        "triage": {
            "description": "Routes tickets.",
            "system_prompt_file": "prompts/triage.jinja2",
            "tools": [
                {"name": "compile_scaffold_lookup", "module": "compile_scaffold_test_module"}
            ],
        }
    }

    def mock_completion(*args, **kwargs):
        return _MockResponse(json.dumps(config))

    import litellm

    monkeypatch.setattr(litellm, "completion", mock_completion)

    result = runner.invoke(app, ["compile", "blueprint.md"])
    assert result.exit_code == 0, result.stdout

    prompt_path = Path("prompts/triage.jinja2")
    tool_path = Path("compile_scaffold_test_module.py")
    assert prompt_path.exists()
    assert tool_path.exists()
    assert "compile_scaffold_lookup" in tool_path.read_text()
    assert "Scaffolded" in result.stdout

    # A user fills in the placeholders with real content.
    prompt_path.write_text("You are the real triage agent.\n")
    original_tool_content = tool_path.read_text()

    result2 = runner.invoke(app, ["compile", "blueprint.md"], input="y\n")
    assert result2.exit_code == 0, result2.stdout
    assert prompt_path.read_text() == "You are the real triage agent.\n"
    assert tool_path.read_text() == original_tool_content


def test_compile_command_reports_missing_api_key_without_a_raw_traceback(tmp_path, monkeypatch):
    """Regression test: `inta compile` in a fresh directory with no API key configured anywhere
    used to crash with a raw litellm traceback deep in a provider SDK. It must instead fail with
    one clear, coded, actionable message — and never reach litellm at all."""
    monkeypatch.chdir(tmp_path)
    for var in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    bp = tmp_path / "blueprint.md"
    bp.write_text("# Vision\nCreate a triage and support agent.")

    import litellm

    def mock_completion(*args, **kwargs):
        raise AssertionError("litellm.completion should never be reached without an API key")

    monkeypatch.setattr(litellm, "completion", mock_completion)

    result = runner.invoke(app, ["compile", "blueprint.md"])

    assert result.exit_code == 1
    assert "IG-CLI-008" in result.stdout
    assert "GEMINI_API_KEY" in result.stdout
    assert "Traceback" not in result.stdout


def test_compile_command_loads_dot_env_from_the_project_directory(mock_blueprint, monkeypatch):
    """Regression test: ai.yaml doesn't exist yet during `inta compile`, so parse_project() (the
    usual place .env gets loaded) never runs. A key placed in a `.env` next to the blueprint must
    still be picked up, not just a key already exported in the shell."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    (mock_blueprint.parent / ".env").write_text("GEMINI_API_KEY=from-dot-env-file\n")

    seen_env_values = []

    def mock_completion(*args, **kwargs):
        import os

        seen_env_values.append(os.environ.get("GEMINI_API_KEY"))
        return _MockResponse(json.dumps(VALID_MINIMAL_CONFIG))

    import litellm

    monkeypatch.setattr(litellm, "completion", mock_completion)

    result = runner.invoke(app, ["compile", "blueprint.md"])

    assert result.exit_code == 0, result.stdout
    assert seen_env_values == ["from-dot-env-file"]
