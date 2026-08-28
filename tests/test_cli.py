import asyncio
import json
import re
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from intagrin.cli import (
    AI_SCHEMA_JSON_PATH,
    AI_YAML_TEMPLATE,
    _generate_and_validate_wizard_config,
    app,
)
from intagrin.compiler.parser import parse_project
from intagrin.runtime.engine import RuntimeEngine

runner = CliRunner()


def test_version_flag_prints_the_installed_version_and_exits_zero():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    # Must actually print a version number and exit early — not just contain the word
    # "intagrin" somewhere, which a broken callback could still satisfy via the --version
    # option's own help text if it fell through to printing full --help instead.
    assert re.search(r"intagrin \d+\.\d+\.\d+", result.stdout.lower())
    assert "usage:" not in result.stdout.lower()


def test_no_args_still_shows_help_not_a_version_prompt():
    """--version must not change the pre-existing no-args-shows-help behavior (main()'s
    invoke_without_command=True path)."""
    result = runner.invoke(app, [])
    assert result.exit_code == 0
    assert "Usage" in result.stdout or "usage" in result.stdout.lower()


def test_scaffolded_ai_yaml_has_no_live_router_referencing_unset_state(tmp_path):
    """Regression guard: the default scaffold used to ship a conditional router
    (`user_status == 'banned'`) that could never fire — nothing in the scaffold ever sets
    `user_status`, so it silently logged 'Unknown variable' on every single session's first
    turn. The pattern is still shown, but commented out, not live."""
    (tmp_path / "ai.yaml").write_text(AI_YAML_TEMPLATE.format(project_name="demo"))
    graph = parse_project(tmp_path)
    assert graph.config.agents["triage"].routers == []


def test_scaffolded_ai_yaml_has_no_mcp_tool_requiring_external_setup(tmp_path):
    """Regression guard: the default scaffold used to ship a live MCP GitHub tool needing `npx`
    and a GitHub token with zero setup guidance — the first thing a brand-new user's `inta dev`
    would trip on. The MCP pattern is still shown, but commented out, not live."""
    (tmp_path / "ai.yaml").write_text(AI_YAML_TEMPLATE.format(project_name="demo"))
    graph = parse_project(tmp_path)
    tool_names = {t.name for t in graph.config.agents["support"].tools}
    assert tool_names == {"get_user_account"}


def test_new_scaffolds_a_yaml_language_server_schema_directive_and_the_schema_file(tmp_path):
    """`inta new` must wire up editor autocomplete/validation for ai.yaml out of the box: the
    yaml-language-server modeline as ai.yaml's first line, and a matching ai.schema.json the
    relative `./ai.schema.json` $schema reference actually resolves to."""
    project_dir = tmp_path / "demo"
    result = runner.invoke(app, ["new", str(project_dir)])
    assert result.exit_code == 0, result.output

    ai_yaml_text = (project_dir / "ai.yaml").read_text()
    assert ai_yaml_text.startswith("# yaml-language-server: $schema=./ai.schema.json\n")

    schema_path = project_dir / "ai.schema.json"
    assert schema_path.exists()
    schema = json.loads(schema_path.read_text())
    assert schema == json.loads(AI_SCHEMA_JSON_PATH.read_text(encoding="utf-8"))

    # The whole point of the schema is catching a typo'd field before `inta verify` does — every
    # actual model fragment must reject unknown properties, not just validate YAML syntax.
    assert schema["$defs"]["AgentConfig"]["additionalProperties"] is False
    # ...but a genuine dict[str, X] map (agents: dict[str, AgentConfig]) must still accept any
    # agent name as a key — its additionalProperties is a value *schema*, not a strictness flag,
    # and must not have been clobbered by the same pass.
    assert schema["properties"]["agents"]["additionalProperties"] == {
        "$ref": "#/$defs/AgentConfig"
    }

    # The scaffolded ai.yaml itself must still parse through the real pipeline with the new
    # leading comment line present — a schema directive is only worth adding if it doesn't
    # break the file it's decorating.
    graph = parse_project(project_dir)
    assert "triage" in graph.config.agents


def test_scaffolded_project_initializes_without_attempting_any_external_connection(tmp_path):
    """End-to-end proof, not just a config-shape check: a fresh scaffold's engine must
    initialize cleanly with no MCP subprocess spawn attempt and no router-evaluation error."""
    (tmp_path / "ai.yaml").write_text(AI_YAML_TEMPLATE.format(project_name="demo"))
    (tmp_path / "tools").mkdir()
    (tmp_path / "tools" / "__init__.py").touch()
    (tmp_path / "tools" / "custom_tools.py").write_text(
        "def get_user_account(user_id: str) -> str:\n"
        '    """Fetch user account details.\n\n    Args:\n        user_id: The user id.\n    """\n'
        "    return f'Account for {user_id}'\n"
    )

    import sys

    sys.path.insert(0, str(tmp_path))
    try:

        async def _run():
            graph = parse_project(tmp_path)
            engine = RuntimeEngine(graph=graph, project_dir=tmp_path, session_id="s1")
            await engine.initialize()
            return engine

        engine = asyncio.run(_run())
        assert "get_user_account" in engine.local_tools
    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.pop("tools.custom_tools", None)
        sys.modules.pop("tools", None)


def test_new_scaffolds_a_state_schema_by_default(tmp_path):
    """Regression guard for the state_schema default nudge: a fresh scaffold should ship a
    state_schema out of the box (every field optional, so this costs nothing even unfilled)
    rather than leaving write_state completely unchecked by default."""
    project_dir = tmp_path / "demo"
    result = runner.invoke(app, ["new", str(project_dir)])
    assert result.exit_code == 0, result.output

    schemas_path = project_dir / "schemas.py"
    assert schemas_path.exists()
    assert "class AppState" in schemas_path.read_text()

    graph = parse_project(project_dir)
    assert graph.config.state_schema == "schemas.AppState"


def _mock_completion_response(content: str):
    resp = MagicMock()
    resp.choices = [MagicMock(message=MagicMock(content=content))]
    return resp


def test_wizard_config_generation_accepts_a_valid_config_on_first_try():
    valid_json = json.dumps(
        {
            "ai_yaml": (
                'version: "1.0"\nname: "demo"\ndefault_agent: "triage"\n'
                'model:\n  primary: "mock/model"\nmemory:\n  type: "sqlite"\n'
                "agents:\n  triage: {}\n"
            ),
            "custom_tools_py": "# nothing yet",
        }
    )
    with patch("litellm.completion", return_value=_mock_completion_response(valid_json)) as mock_completion:
        result = _generate_and_validate_wizard_config("mock/model", "sys prompt", "a demo bot")

    assert result is not None
    ai_yaml_text, custom_tools_py = result
    assert "name: \"demo\"" in ai_yaml_text
    assert custom_tools_py == "# nothing yet"
    assert mock_completion.call_count == 1


def test_wizard_config_generation_self_heals_an_invalid_router_condition():
    """Regression test for the actual bug: the wizard used to write whatever JSON parsed, with no
    schema check at all. A router condition using unsupported syntax (state.get(...)) must be
    caught and sent back to the model for a fix, not written straight to disk."""
    bad_json = json.dumps(
        {
            "ai_yaml": (
                'version: "1.0"\nname: "demo"\ndefault_agent: "triage"\n'
                'model:\n  primary: "mock/model"\nmemory:\n  type: "sqlite"\n'
                "agents:\n  triage:\n    routers:\n"
                "      - condition: \"state.get('x') == 1\"\n        target: \"billing\"\n"
                "  billing: {}\n"
            ),
            "custom_tools_py": "",
        }
    )
    fixed_json = json.dumps(
        {
            "ai_yaml": (
                'version: "1.0"\nname: "demo"\ndefault_agent: "triage"\n'
                'model:\n  primary: "mock/model"\nmemory:\n  type: "sqlite"\n'
                "agents:\n  triage:\n    routers:\n"
                "      - condition: \"x == 1\"\n        target: \"billing\"\n"
                "  billing: {}\n"
            ),
            "custom_tools_py": "",
        }
    )
    with patch(
        "litellm.completion",
        side_effect=[
            _mock_completion_response(bad_json),
            _mock_completion_response(fixed_json),
        ],
    ) as mock_completion:
        result = _generate_and_validate_wizard_config("mock/model", "sys prompt", "a demo bot")

    assert result is not None
    ai_yaml_text, _ = result
    assert "state.get" not in ai_yaml_text
    assert mock_completion.call_count == 2


def test_wizard_config_generation_returns_none_after_exhausting_retries():
    bad_json = json.dumps(
        {
            "ai_yaml": (
                'version: "1.0"\nname: "demo"\ndefault_agent: "triage"\n'
                'model:\n  primary: "mock/model"\nmemory:\n  type: "sqlite"\n'
                "agents:\n  triage:\n    routers:\n"
                "      - condition: \"state.get('x') == 1\"\n        target: \"billing\"\n"
                "  billing: {}\n"
            ),
            "custom_tools_py": "",
        }
    )
    with patch(
        "litellm.completion", return_value=_mock_completion_response(bad_json)
    ) as mock_completion:
        result = _generate_and_validate_wizard_config(
            "mock/model", "sys prompt", "a demo bot", max_retries=2
        )

    assert result is None
    assert mock_completion.call_count == 3  # initial attempt + 2 retries


def test_wizard_config_generation_treats_malformed_json_as_a_healable_error():
    with patch(
        "litellm.completion",
        side_effect=[
            _mock_completion_response("not even json"),
            _mock_completion_response(
                json.dumps(
                    {
                        "ai_yaml": (
                            'version: "1.0"\nname: "demo"\ndefault_agent: "triage"\n'
                            'model:\n  primary: "mock/model"\nmemory:\n  type: "sqlite"\n'
                            "agents:\n  triage: {}\n"
                        ),
                        "custom_tools_py": "",
                    }
                )
            ),
        ],
    ) as mock_completion:
        result = _generate_and_validate_wizard_config("mock/model", "sys prompt", "a demo bot")

    assert result is not None
    assert mock_completion.call_count == 2


def test_new_template_coding_agent_scaffolds_a_valid_plan_code_verify_loop(tmp_path):
    """`inta new --template coding-agent` must produce a real, working project — not just prose
    to copy from by hand — with the architect -> coder -> verifier handoff loop from
    docs/05_Example_Coding_Agent.md, schema-validated before anything is written to disk."""
    project_dir = tmp_path / "coding-demo"
    result = runner.invoke(app, ["new", str(project_dir), "--template", "coding-agent"])
    assert result.exit_code == 0, result.output

    graph = parse_project(project_dir)
    assert set(graph.config.agents) == {"architect_agent", "coder_agent", "verifier_agent"}
    assert graph.config.default_agent == "architect_agent"
    assert graph.config.agents["architect_agent"].handoffs == ["coder_agent"]
    assert graph.config.agents["coder_agent"].handoffs == ["verifier_agent"]
    assert graph.config.agents["verifier_agent"].handoffs == ["coder_agent"]  # the self-heal loop

    verifier_tools = {t.name: t for t in graph.config.agents["verifier_agent"].tools}
    assert verifier_tools["run_bash_command"].requires_approval is True

    for prompt_name in ("architect_prompt.jinja2", "coder_prompt.jinja2", "verifier_prompt.jinja2"):
        assert (project_dir / "prompts" / prompt_name).exists()
    assert (project_dir / "schemas.py").exists()


def test_new_template_coding_agent_tools_load_and_initialize_cleanly(tmp_path):
    """End-to-end proof the scaffolded tools/*.py is real, importable Python that RuntimeEngine
    can actually load — not just that the YAML shape is valid."""
    project_dir = tmp_path / "coding-demo"
    result = runner.invoke(app, ["new", str(project_dir), "--template", "coding-agent"])
    assert result.exit_code == 0, result.output

    import sys

    sys.path.insert(0, str(project_dir))
    try:
        async def _run():
            graph = parse_project(project_dir)
            engine = RuntimeEngine(graph=graph, project_dir=project_dir, session_id="s1")
            await engine.initialize()
            return engine

        engine = asyncio.run(_run())
        assert {"grep_search", "list_directory"} <= engine.local_tools.keys()
    finally:
        sys.path.remove(str(project_dir))
        sys.modules.pop("tools.custom_tools", None)
        sys.modules.pop("tools", None)


def test_new_rejects_an_unrecognized_template_value(tmp_path):
    project_dir = tmp_path / "bad-template-demo"
    result = runner.invoke(app, ["new", str(project_dir), "--template", "bogus"])
    assert result.exit_code != 0
    assert "bogus" in result.output
    assert not project_dir.exists()


def test_new_withagent_end_to_end_through_the_interactive_provider_picker(tmp_path):
    """run_agent_wizard (the --withagent path) has no other test exercising its actual
    interactive prompt sequence — every other wizard test calls
    _generate_and_validate_wizard_config directly, bypassing the picker entirely. This is the one
    test that would have caught the picker's own refactor breaking (the removed api_key/provider
    variables, the new model sub-prompt needing an extra line of input) rather than just the
    LLM-call/validation logic downstream of it."""
    project_dir = tmp_path / "withagent-demo"
    valid_json = json.dumps(
        {
            "ai_yaml": (
                'version: "1.0"\nname: "demo"\ndefault_agent: "triage"\n'
                'model:\n  primary: "mock/model"\nmemory:\n  type: "sqlite"\n'
                "agents:\n  triage: {}\n"
            ),
            "custom_tools_py": "# nothing yet",
        }
    )
    seen_models = []

    def mock_completion(*args, **kwargs):
        seen_models.append(kwargs.get("model"))
        return _mock_completion_response(valid_json)

    with patch("litellm.completion", side_effect=mock_completion):
        # Provider "1" (OpenAI) -> API key -> model sub-prompt (accept the default) -> the idea.
        result = runner.invoke(
            app,
            ["new", str(project_dir), "--withagent"],
            input="1\nsk-test-not-real\n\nA demo triage bot\n",
        )

    assert result.exit_code == 0, result.output
    assert seen_models == ["openai/gpt-4o"]
    # Must be the AI-generated config, not the exception-fallback default template — an earlier
    # draft of this test passed even against a reintroduced bug because both paths satisfy
    # "ai.yaml exists" and the .env write (now done earlier, inside the picker itself) happens
    # regardless of what fails afterward. The generated project's actual name is the one thing
    # only the real path produces.
    assert "Project generated successfully via AI!" in result.output
    assert "Falling back to default template" not in result.output
    assert 'name: "demo"' in (project_dir / "ai.yaml").read_text()
    assert "OPENAI_API_KEY=sk-test-not-real" in (project_dir / ".env").read_text()
