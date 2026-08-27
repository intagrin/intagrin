import ast
import tempfile
from pathlib import Path

import pytest

from intagrin.compiler.exporter import CodeExporter


def test_code_exporter_standalone():
    with tempfile.TemporaryDirectory() as tmpdir:
        p_dir = Path(tmpdir)
        (p_dir / "tools").mkdir(parents=True, exist_ok=True)
        (p_dir / "prompts").mkdir(parents=True, exist_ok=True)

        (p_dir / "tools" / "custom.py").write_text("def test_tool(): return 'ok'")
        (p_dir / "prompts" / "agent.jinja2").write_text("You are test agent.")

        ai_yaml = """version: "1.0"
name: "export-test"
default_agent: "test_agent"
model:
  primary: "openai/gpt-4o-mini"
memory:
  type: "sqlite"
agents:
  test_agent:
    system_prompt_file: "prompts/agent.jinja2"
    tools:
      - name: "test_tool"
        module: "tools.custom"
"""
        (p_dir / "ai.yaml").write_text(ai_yaml)

        out_file = "standalone_app.py"
        exporter = CodeExporter(project_dir=p_dir, output_file=out_file)
        exporter.export_fastapi()

        generated_code = (p_dir / out_file).read_text()
        assert "FastAPI(title=\"export-test (Standalone)\"" in generated_code
        assert "from tools.custom import test_tool" in generated_code
        assert "@app.post(\"/chat\"" in generated_code


@pytest.fixture
def exported_project(tmp_path):
    (tmp_path / "tools").mkdir()
    (tmp_path / "prompts").mkdir()
    # A module name distinct from other tests' `tools/custom.py` — Python caches imported
    # submodules process-wide, and pytest runs every test in one process.
    (tmp_path / "tools" / "lookup_tools.py").write_text(
        'def lookup(query: str) -> str:\n'
        '    """Look something up.\n\n'
        '    Args:\n'
        '        query: What to look up.\n'
        '    """\n'
        '    return f"result for {query}"\n'
    )
    (tmp_path / "prompts" / "agent.jinja2").write_text("You are the export test agent.")
    (tmp_path / "ai.yaml").write_text(
        'version: "1.0"\n'
        'name: "export-tools-test"\n'
        'default_agent: "worker"\n'
        'model:\n'
        '  primary: "openai/gpt-4o-mini"\n'
        'memory:\n'
        '  type: "sqlite"\n'
        'agents:\n'
        '  worker:\n'
        '    system_prompt_file: "prompts/agent.jinja2"\n'
        '    tools:\n'
        '      - name: "lookup"\n'
        '        module: "tools.lookup_tools"\n'
        '      - name: "remote_mcp"\n'
        '        type: "mcp"\n'
        '        command: "npx"\n'
        '        args: []\n'
    )
    out_file = tmp_path / "standalone_app.py"
    CodeExporter(project_dir=tmp_path, output_file="standalone_app.py").export_fastapi()
    return out_file


def test_export_bakes_a_real_tool_schema_and_function_mapping(exported_project):
    code = exported_project.read_text()
    assert '"lookup"' in code
    assert "TOOL_FUNCTIONS" in code
    assert '"lookup": lookup,' in code
    # MCP tools have no local implementation to export — must not be silently pretended into
    # TOOL_FUNCTIONS.
    assert "remote_mcp" not in code.split("TOOL_FUNCTIONS")[1].split("}")[0]


def test_export_scope_is_documented_in_the_generated_file(exported_project):
    code = exported_project.read_text()
    assert "SCOPE:" in code
    assert "Handoffs" in code
    assert "NOT included" in code


def test_exported_app_is_valid_python_and_actually_callable(exported_project, monkeypatch):
    code = exported_project.read_text()
    # Full syntax validation, not just "no SyntaxError on compile" — parses the whole module.
    ast.parse(code, filename=str(exported_project))

    project_dir = str(exported_project.parent)
    monkeypatch.syspath_prepend(project_dir)
    monkeypatch.chdir(exported_project.parent)

    import importlib.util

    spec = importlib.util.spec_from_file_location("standalone_app_under_test", exported_project)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert len(module.TOOL_SCHEMAS) == 1
    assert module.TOOL_SCHEMAS[0]["function"]["name"] == "lookup"
    assert "lookup" in module.TOOL_FUNCTIONS

    import asyncio

    result = asyncio.run(module.call_tool("lookup", {"query": "widgets"}))
    assert result == "result for widgets"
