import pytest
from typer.testing import CliRunner

from intagrin.cli import app

runner = CliRunner()

# Regression coverage for a real bug found in production: a botched "DefinAI" -> "IntaGrin" rename
# corrupted words containing "Defin" as a substring (Defined -> IntaGrined, Defines -> IntaGrines)
# across every template `inta copilot` writes, the compile skill's file reused the implement
# skill's filename, and the reference doc referred to a nonexistent `intagrin` command instead of
# the actual `inta` console script.
CORRUPTION_PATTERNS = ["IntaGrined", "IntaGrines", "IntaGrine ", "DefinAI", "defin-implement", "defin-compile"]


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _run_copilot(agent: str):
    return runner.invoke(app, ["copilot", "--agent", agent])


@pytest.mark.parametrize("agent", ["cursor", "claude", "copilot", "antigravity", "factory"])
def test_generated_content_has_no_corruption_artifacts(project, agent):
    result = _run_copilot(agent)
    assert result.exit_code == 0, result.stdout

    all_text = "\n".join(p.read_text(encoding="utf-8") for p in project.rglob("*") if p.is_file())
    for pattern in CORRUPTION_PATTERNS:
        assert pattern not in all_text, f"found corrupted artifact {pattern!r} in generated output"

    # The generated docs must reference the real `inta` console script, never a bare `intagrin`
    # command (which isn't registered in pyproject.toml and doesn't exist).
    assert "`intagrin serve`" not in all_text
    assert "`intagrin verify`" not in all_text
    assert "`inta serve`" in all_text


@pytest.mark.parametrize("agent", ["cursor", "claude", "copilot", "antigravity", "factory"])
def test_generated_content_asks_about_auth_and_rag_instead_of_assuming(project, agent):
    """Regression test for the requirement that the Architect/IDE agents must ask about
    consequential tech decisions (auth mechanism, RAG/vector-DB choice) rather than silently
    picking a default — confirms the clarification protocol wasn't silently dropped in a future
    edit to these templates."""
    result = _run_copilot(agent)
    assert result.exit_code == 0, result.stdout

    all_text = "\n".join(p.read_text(encoding="utf-8") for p in project.rglob("*") if p.is_file())
    assert "authentication" in all_text.lower()
    assert "server.auth.type" in all_text
    assert "rag" in all_text.lower()


def test_cursor_compile_skill_gets_its_own_filename_not_the_implement_skills(project):
    result = _run_copilot("cursor")
    assert result.exit_code == 0, result.stdout

    implement_file = project / ".cursor" / "skills" / "intagrin-implement" / "intagrin-implement.mdc"
    compile_file = project / ".cursor" / "skills" / "intagrin-compile" / "intagrin-compile.mdc"
    stale_misnamed_file = project / ".cursor" / "skills" / "intagrin-compile" / "intagrin-implement.mdc"

    assert implement_file.exists()
    assert compile_file.exists()
    assert not stale_misnamed_file.exists()
    assert "intagrin-compile" in compile_file.read_text()
    assert "Bidirectional Architecture Sync" in compile_file.read_text()


def test_rerunning_copilot_is_idempotent_no_duplicated_content(project):
    """The exact failure mode found in the repo's own checked-in files: content got concatenated
    twice instead of cleanly overwritten. Running twice must produce byte-identical output."""
    _run_copilot("cursor")
    rules_file = project / ".cursor" / "rules" / "intagrin-agent.mdc"
    first_content = rules_file.read_text()
    assert first_content.count("# IntaGrin Architect Instructions") == 1

    _run_copilot("cursor")
    second_content = rules_file.read_text()

    assert second_content == first_content
    assert second_content.count("# IntaGrin Architect Instructions") == 1


def test_rerun_warns_when_overwriting_differing_content(project):
    _run_copilot("cursor")
    rules_file = project / ".cursor" / "rules" / "intagrin-agent.mdc"
    rules_file.write_text(rules_file.read_text() + "\n<!-- my manual customization -->")

    result = _run_copilot("cursor")

    assert "Overwriting existing" in result.stdout
    assert "my manual customization" not in rules_file.read_text()


@pytest.mark.parametrize("agent", ["cursor", "claude", "copilot", "antigravity", "factory"])
def test_agents_md_is_written_regardless_of_chosen_ide(project, agent):
    """AGENTS.md is the tool-agnostic file read by Codex, Cursor, Aider, Copilot, Gemini CLI, and
    Windsurf alike — it must land at the project root no matter which IDE-specific agent flavor
    was picked, and it must not carry that IDE's own frontmatter (e.g. Cursor's MDC header)."""
    result = _run_copilot(agent)
    assert result.exit_code == 0, result.stdout

    agents_md = project / "AGENTS.md"
    assert agents_md.exists()
    content = agents_md.read_text()
    assert "# IntaGrin Architect Instructions" in content
    assert "---" not in content.splitlines()[0]  # no YAML frontmatter fence up front


def test_rerunning_copilot_leaves_agents_md_byte_identical(project):
    _run_copilot("claude")
    first = (project / "AGENTS.md").read_text()

    _run_copilot("claude")
    second = (project / "AGENTS.md").read_text()

    assert first == second


def test_invalid_agent_choice_errors_cleanly(project):
    result = _run_copilot("not-a-real-ide")
    assert result.exit_code == 1
    assert "must be one of" in result.stdout
