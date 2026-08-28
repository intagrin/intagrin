import json
import os
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

    result = runner.invoke(app, ["compile", "blueprint.md", "--model", "gemini/gemini-2.5-flash"])

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

    result = runner.invoke(app, ["compile", "blueprint.md", "--model", "gemini/gemini-2.5-flash"])

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

    result = runner.invoke(app, ["compile", "blueprint.md", "--model", "gemini/gemini-2.5-flash"])

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

    result = runner.invoke(app, ["compile", "blueprint.md", "--model", "gemini/gemini-2.5-flash"])

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

    result = runner.invoke(app, ["compile", "blueprint.md", "--model", "gemini/gemini-2.5-flash"])

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

    result = runner.invoke(app, ["compile", "blueprint.md", "--model", "gemini/gemini-2.5-flash"])

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

    result = runner.invoke(app, ["compile", "blueprint.md", "--model", "gemini/gemini-2.5-flash"])

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

    result = runner.invoke(app, ["compile", "blueprint.md", "--model", "gemini/gemini-2.5-flash"])
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

    result2 = runner.invoke(
        app, ["compile", "blueprint.md", "--model", "gemini/gemini-2.5-flash"], input="y\n"
    )
    assert result2.exit_code == 0, result2.stdout
    assert prompt_path.read_text() == "You are the real triage agent.\n"
    assert tool_path.read_text() == original_tool_content


def test_compile_command_drafts_a_real_system_prompt_not_a_placeholder(mock_blueprint, monkeypatch):
    """The scaffolded prompt file used to always be the literal one-liner
    f"You are {description}.\\n" — must now be genuinely drafted by the compile model, with the
    agent's tools/handoffs and the blueprint itself as context, not a fixed template. The earlier
    version of this test only checked prompt_path.exists(), which would have passed even if the
    scaffolded content were nonsense — asserting on actual content this time."""
    config = dict(VALID_MINIMAL_CONFIG)
    config["agents"] = {
        "triage": {
            "description": "Routes tickets.",
            "system_prompt_file": "prompts/triage.jinja2",
            "handoffs": ["billing"],
            "tools": [{"name": "lookup_ticket", "module": "draft_prompt_test_module"}],
        },
        "billing": {"description": "Handles billing."},
    }
    seen_draft_user_prompts = []

    def mock_completion(*args, **kwargs):
        sys_content = kwargs["messages"][0]["content"]
        if "expert prompt engineer" in sys_content:
            seen_draft_user_prompts.append(kwargs["messages"][1]["content"])
            return _MockResponse("You are the ticket triage specialist. Your job is to read "
                                  "incoming support tickets, look them up with lookup_ticket, "
                                  "and hand off to billing whenever the issue is payment-related.")
        return _MockResponse(json.dumps(config))

    import litellm

    monkeypatch.setattr(litellm, "completion", mock_completion)

    result = runner.invoke(app, ["compile", "blueprint.md", "--model", "gemini/gemini-2.5-flash"])

    assert result.exit_code == 0, result.stdout
    drafted = Path("prompts/triage.jinja2").read_text()
    assert drafted != "You are Routes tickets.\n"
    assert "lookup_ticket" in drafted or "triage" in drafted.lower()
    # The draft call must actually have been given the tool/handoff/blueprint context, not just
    # the bare description.
    assert len(seen_draft_user_prompts) == 1
    assert "lookup_ticket" in seen_draft_user_prompts[0]
    assert "billing" in seen_draft_user_prompts[0]
    assert "triage and support agent" in seen_draft_user_prompts[0]  # from mock_blueprint's text


def test_compile_command_falls_back_to_a_placeholder_when_prompt_drafting_fails(
    mock_blueprint, monkeypatch
):
    """Prompt drafting is best-effort — a failure there (network error, bad response) must never
    block the compile itself. Falls back to the old one-liner, not a crash."""
    config = dict(VALID_MINIMAL_CONFIG)
    config["agents"] = {
        "triage": {"description": "Routes tickets.", "system_prompt_file": "prompts/triage.jinja2"}
    }

    def mock_completion(*args, **kwargs):
        sys_content = kwargs["messages"][0]["content"]
        if "expert prompt engineer" in sys_content:
            raise RuntimeError("simulated network failure")
        return _MockResponse(json.dumps(config))

    import litellm

    monkeypatch.setattr(litellm, "completion", mock_completion)

    result = runner.invoke(app, ["compile", "blueprint.md", "--model", "gemini/gemini-2.5-flash"])

    assert result.exit_code == 0, result.stdout
    assert Path("prompts/triage.jinja2").read_text() == "You are Routes tickets.\n"


def test_compile_command_offers_to_update_a_previously_drafted_prompt_when_blueprint_changes(
    mock_blueprint, monkeypatch
):
    """Once a prompt file carries the intagrin:blueprint-hash marker (i.e. this command drafted
    it), a later compile against a *changed* blueprint.md must offer to redraft it — diffed and
    confirmed just like the ai.yaml change above — instead of leaving it stale forever just
    because the file already exists on disk."""
    config = dict(VALID_MINIMAL_CONFIG)
    config["agents"] = {
        "triage": {"description": "Routes tickets.", "system_prompt_file": "prompts/triage.jinja2"}
    }
    draft_calls = []

    def mock_completion(*args, **kwargs):
        sys_content = kwargs["messages"][0]["content"]
        if "expert prompt engineer" in sys_content:
            draft_calls.append(kwargs["messages"][1]["content"])
            if len(draft_calls) == 1:
                return _MockResponse("You are the triage agent. Handle incoming tickets.")
            return _MockResponse("You are the triage agent. Handle tickets and escalate refunds.")
        return _MockResponse(json.dumps(config))

    import litellm

    monkeypatch.setattr(litellm, "completion", mock_completion)

    result1 = runner.invoke(app, ["compile", "blueprint.md", "--model", "gemini/gemini-2.5-flash"])
    assert result1.exit_code == 0, result1.stdout
    prompt_path = Path("prompts/triage.jinja2")
    first_draft = prompt_path.read_text()
    assert "Handle incoming tickets" in first_draft
    assert "intagrin:blueprint-hash" in first_draft

    mock_blueprint.write_text(
        "# Vision\nCreate a triage agent that also handles refund escalations."
    )

    result2 = runner.invoke(
        app, ["compile", "blueprint.md", "--model", "gemini/gemini-2.5-flash"], input="y\ny\n"
    )
    assert result2.exit_code == 0, result2.stdout
    assert "blueprint.md changed since" in result2.stdout
    assert "Updated prompts/triage.jinja2" in result2.stdout

    updated_draft = prompt_path.read_text()
    assert "escalate refunds" in updated_draft
    assert len(draft_calls) == 2


def test_compile_command_declining_a_prompt_update_keeps_the_old_text_but_stops_reasking(
    mock_blueprint, monkeypatch
):
    """Declining the redraft diff must preserve the existing prompt text exactly, but still needs
    to record that this blueprint version was already reviewed — otherwise every future compile
    against the same unchanged blueprint would re-ask the identical question forever."""
    config = dict(VALID_MINIMAL_CONFIG)
    config["agents"] = {
        "triage": {"description": "Routes tickets.", "system_prompt_file": "prompts/triage.jinja2"}
    }
    draft_calls = []

    def mock_completion(*args, **kwargs):
        sys_content = kwargs["messages"][0]["content"]
        if "expert prompt engineer" in sys_content:
            draft_calls.append(kwargs["messages"][1]["content"])
            return _MockResponse(f"Drafted version {len(draft_calls)} of the triage prompt.")
        return _MockResponse(json.dumps(config))

    import litellm

    monkeypatch.setattr(litellm, "completion", mock_completion)

    runner.invoke(app, ["compile", "blueprint.md", "--model", "gemini/gemini-2.5-flash"])
    prompt_path = Path("prompts/triage.jinja2")
    original_draft = prompt_path.read_text()

    mock_blueprint.write_text("# Vision\nCreate a triage agent with a slightly different scope.")

    # "y" accepts the (no-op) ai.yaml diff, "n" declines the offered prompt update.
    result2 = runner.invoke(
        app, ["compile", "blueprint.md", "--model", "gemini/gemini-2.5-flash"], input="y\nn\n"
    )
    assert result2.exit_code == 0, result2.stdout
    assert prompt_path.read_text().splitlines()[0] == original_draft.splitlines()[0]
    assert len(draft_calls) == 2

    # Recompiling again against the SAME (now-declined) blueprint must not ask a second time.
    result3 = runner.invoke(
        app, ["compile", "blueprint.md", "--model", "gemini/gemini-2.5-flash"], input="y\n"
    )
    assert result3.exit_code == 0, result3.stdout
    assert "blueprint.md changed since" not in result3.stdout
    assert len(draft_calls) == 2


def test_compile_command_skips_redrafting_when_blueprint_is_unchanged(mock_blueprint, monkeypatch):
    """The common case: recompiling against the exact same blueprint.md must not re-draft (and
    re-ask about) prompts that have nothing to update — no wasted LLM call, no confirm prompt."""
    config = dict(VALID_MINIMAL_CONFIG)
    config["agents"] = {
        "triage": {"description": "Routes tickets.", "system_prompt_file": "prompts/triage.jinja2"}
    }
    draft_calls = []

    def mock_completion(*args, **kwargs):
        sys_content = kwargs["messages"][0]["content"]
        if "expert prompt engineer" in sys_content:
            draft_calls.append(kwargs["messages"][1]["content"])
            return _MockResponse("You are the triage agent.")
        return _MockResponse(json.dumps(config))

    import litellm

    monkeypatch.setattr(litellm, "completion", mock_completion)

    runner.invoke(app, ["compile", "blueprint.md", "--model", "gemini/gemini-2.5-flash"])
    assert len(draft_calls) == 1

    result2 = runner.invoke(
        app, ["compile", "blueprint.md", "--model", "gemini/gemini-2.5-flash"], input="y\n"
    )
    assert result2.exit_code == 0, result2.stdout
    assert len(draft_calls) == 1
    assert "Updated" not in result2.stdout


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

    result = runner.invoke(app, ["compile", "blueprint.md", "--model", "gemini/gemini-2.5-flash"])

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

    result = runner.invoke(app, ["compile", "blueprint.md", "--model", "gemini/gemini-2.5-flash"])

    assert result.exit_code == 0, result.stdout
    assert seen_env_values == ["from-dot-env-file"]


def test_compile_command_reuses_an_already_configured_projects_own_model(mock_blueprint, monkeypatch):
    """Re-compiling a blueprint into a project that already has an ai.yaml must reuse that
    project's own model.primary for the compile call itself, not silently force a different
    provider — the same reasoning server/monitor.py's run_architect already applies."""
    (mock_blueprint.parent / "ai.yaml").write_text(
        "version: '1.0'\nname: existing\ndefault_agent: triage\n"
        "model:\n  primary: anthropic/claude-sonnet-4-5\nmemory:\n  type: sqlite\n"
        "agents:\n  triage:\n    description: Routes tickets.\n"
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    seen_models = []

    def mock_completion(*args, **kwargs):
        seen_models.append(kwargs.get("model"))
        return _MockResponse(json.dumps(VALID_MINIMAL_CONFIG))

    import litellm

    monkeypatch.setattr(litellm, "completion", mock_completion)

    result = runner.invoke(app, ["compile", "blueprint.md"], input="y\n")

    assert result.exit_code == 0, result.stdout
    assert seen_models == ["anthropic/claude-sonnet-4-5"]


def test_compile_command_explicit_model_flag_overrides_an_existing_projects_model(
    mock_blueprint, monkeypatch
):
    """--model always wins, even over an already-configured project's own model.primary — the
    explicit, scripted-use escape hatch should never be silently second-guessed. Deliberately
    uses a third, distinct provider (neither the project's own anthropic/... nor the compiler's
    gemini/... hardcoded fallback) so this can't coincidentally pass for the wrong reason."""
    (mock_blueprint.parent / "ai.yaml").write_text(
        "version: '1.0'\nname: existing\ndefault_agent: triage\n"
        "model:\n  primary: anthropic/claude-sonnet-4-5\nmemory:\n  type: sqlite\n"
        "agents:\n  triage:\n    description: Routes tickets.\n"
    )
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-real")
    seen_models = []

    def mock_completion(*args, **kwargs):
        seen_models.append(kwargs.get("model"))
        return _MockResponse(json.dumps(VALID_MINIMAL_CONFIG))

    import litellm

    monkeypatch.setattr(litellm, "completion", mock_completion)

    result = runner.invoke(
        app, ["compile", "blueprint.md", "--model", "openai/gpt-4o"], input="y\n"
    )

    assert result.exit_code == 0, result.stdout
    assert seen_models == ["openai/gpt-4o"]


def test_compile_command_prompts_interactively_when_nothing_to_infer_from(tmp_path, monkeypatch):
    """First-ever compile: no ai.yaml yet and no --model given. Must ask which provider rather
    than silently forcing one specific provider on someone who may not even have that key."""
    monkeypatch.chdir(tmp_path)
    bp = tmp_path / "blueprint.md"
    bp.write_text("# Vision\nCreate a triage and support agent.")

    seen_models = []

    def mock_completion(*args, **kwargs):
        seen_models.append(kwargs.get("model"))
        return _MockResponse(json.dumps(VALID_MINIMAL_CONFIG))

    import litellm

    monkeypatch.setattr(litellm, "completion", mock_completion)

    # Provider menu choice "1" (OpenAI), then the API key prompt, then the model prompt —
    # accepted with an empty line, i.e. the suggested default.
    result = runner.invoke(app, ["compile", "blueprint.md"], input="1\nsk-test-not-real\n\n")

    assert result.exit_code == 0, result.stdout
    assert seen_models == ["openai/gpt-4o"]
    assert os.environ.get("OPENAI_API_KEY") == "sk-test-not-real"
    # The freshly-collected key must be persisted for next time, not just this process.
    assert "OPENAI_API_KEY=sk-test-not-real" in (tmp_path / ".env").read_text()


def test_compile_command_interactive_prompt_lets_you_override_the_suggested_model(
    tmp_path, monkeypatch
):
    """The actual point of the model sub-prompt: picking a provider must not lock you into one
    specific model from it. gpt-4o is only offered as a *default* — typing a different model for
    the same provider (e.g. a cheaper/faster variant) must be honored, not silently ignored."""
    monkeypatch.chdir(tmp_path)
    bp = tmp_path / "blueprint.md"
    bp.write_text("# Vision\nCreate a triage and support agent.")

    seen_models = []

    def mock_completion(*args, **kwargs):
        seen_models.append(kwargs.get("model"))
        return _MockResponse(json.dumps(VALID_MINIMAL_CONFIG))

    import litellm

    monkeypatch.setattr(litellm, "completion", mock_completion)

    # Provider "1" (OpenAI), a key, then an explicit model that is NOT the suggested default.
    result = runner.invoke(
        app, ["compile", "blueprint.md"], input="1\nsk-test-not-real\nopenai/gpt-4o-mini\n"
    )

    assert result.exit_code == 0, result.stdout
    assert seen_models == ["openai/gpt-4o-mini"]
