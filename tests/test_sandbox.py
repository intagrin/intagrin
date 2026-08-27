"""runtime/sandbox.py: isolated subprocess execution for agent-generated code (SandboxToolConfig,
tools[].type: "sandbox"). Tests what the module's own docstring claims it provides — process
isolation, a timeout, and a secret-free environment — not filesystem/network isolation, which it
explicitly does not claim."""

import asyncio
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

from intagrin.compiler.parser import ExecutionGraph
from intagrin.compiler.verifier import GraphVerifier, console as verifier_console
from intagrin.config.schema import (
    AgentConfig,
    AppConfig,
    MemoryConfig,
    ModelConfig,
    SandboxToolConfig,
)
from intagrin.runtime.engine import RuntimeEngine
from intagrin.runtime.sandbox import run_sandboxed_code


def test_run_sandboxed_code_captures_stdout_and_exit_code():
    result = asyncio.run(
        run_sandboxed_code("print('hello from sandbox')", "python", 10, 256)
    )
    assert "Exit code: 0" in result
    assert "hello from sandbox" in result


def test_run_sandboxed_code_captures_stderr_and_nonzero_exit():
    result = asyncio.run(
        run_sandboxed_code("import sys; sys.exit(3)", "python", 10, 256)
    )
    assert "Exit code: 3" in result


def test_run_sandboxed_code_reports_a_traceback_in_stderr():
    result = asyncio.run(run_sandboxed_code("raise ValueError('boom')", "python", 10, 256))
    assert "Exit code: 1" in result
    assert "ValueError: boom" in result


def test_run_sandboxed_code_times_out_instead_of_hanging():
    # time.sleep (not a CPU-burning loop) isolates the wall-clock subprocess timeout path from
    # the separate CPU-time rlimit — both are real limits, but a CPU-burning loop can race the
    # two and get killed by the rlimit first, which is covered by the signal-kill test below.
    result = asyncio.run(
        run_sandboxed_code("import time; time.sleep(100)", "python", 1, 256)
    )
    assert "timed out" in result.lower()


def test_run_sandboxed_code_reports_a_killed_process_honestly():
    """A CPU-burning infinite loop gets killed by the CPU-time rlimit (a real OS signal), which
    races the wall-clock timeout — either way, the result must say the process was killed, not
    just print a bare, uninformative negative exit code."""
    result = asyncio.run(run_sandboxed_code("while True: pass", "python", 1, 256))
    assert "timed out" in result.lower() or "terminated by signal" in result.lower()


def test_run_sandboxed_code_gets_no_inherited_environment_secrets():
    """The subprocess must get an explicit minimal env, never a copy of the parent's os.environ —
    otherwise a real API key sitting in the engine's environment would be readable by whatever
    code the LLM decided to run."""
    with patch.dict(os.environ, {"SUPER_SECRET_API_KEY": "sk-do-not-leak"}):
        result = asyncio.run(
            run_sandboxed_code(
                "import os; print(repr(os.environ.get('SUPER_SECRET_API_KEY')))",
                "python",
                10,
                256,
            )
        )
    assert "sk-do-not-leak" not in result
    assert "None" in result


def test_run_sandboxed_code_supports_bash():
    result = asyncio.run(run_sandboxed_code("echo hello-from-bash", "bash", 10, 256))
    assert "Exit code: 0" in result
    assert "hello-from-bash" in result


def _sandbox_graph(**sandbox_kwargs):
    config = AppConfig(
        version="1.0",
        name="sandbox-engine-test",
        default_agent="coder",
        model=ModelConfig(primary="mock/model"),
        memory=MemoryConfig(type="sqlite"),
        agents={
            "coder": AgentConfig(
                tools=[
                    SandboxToolConfig(name="run_code", type="sandbox", **sandbox_kwargs)
                ]
            )
        },
    )
    return ExecutionGraph(config, {})


def test_load_tool_config_registers_a_working_sandbox_tool(tmp_path):
    async def _run():
        engine = RuntimeEngine(graph=_sandbox_graph(), project_dir=tmp_path, session_id="s1")
        await engine.initialize()
        engine.active_agent_name = "coder"

        assert "run_code" in engine.local_tools
        assert "run_code" in engine.untrusted_tools
        assert any(
            s["function"]["name"] == "run_code" for s in engine.global_tool_schemas
        )

        result = await engine.execute_tool(
            "run_code", {"code": "print(1 + 1)"}, interactive=False
        )
        assert "2" in result

    asyncio.run(_run())


def test_sandbox_tool_respects_requires_approval(tmp_path):
    async def _run():
        engine = RuntimeEngine(
            graph=_sandbox_graph(requires_approval=True), project_dir=tmp_path, session_id="s2"
        )
        await engine.initialize()
        engine.active_agent_name = "coder"

        result = await engine.execute_tool(
            "run_code",
            {"code": "print('should not run yet')"},
            interactive=False,
            tool_call_id="call_1",
        )
        assert "paused" in result.lower()
        assert "_pending_approval" in engine.state

    asyncio.run(_run())


def test_verifier_advises_on_a_sandbox_tool_without_requires_approval():
    ai_yaml = """version: "1.0"
name: "sandbox-verify-app"
default_agent: "coder"
model:
  primary: "gemini/gemini-2.5-flash"
memory:
  type: "sqlite"
agents:
  coder:
    tools:
      - name: "run_code"
        type: "sandbox"
"""
    with tempfile.TemporaryDirectory() as tmpdir:
        p_dir = Path(tmpdir)
        (p_dir / "ai.yaml").write_text(ai_yaml)
        verifier = GraphVerifier(project_dir=p_dir)

        with verifier_console.capture() as capture:
            verifier.verify()
        output = capture.get()

        assert "Sandbox tool(s) without requires_approval" in output
        assert "coder.run_code" in output


def test_verifier_does_not_advise_when_sandbox_tool_requires_approval():
    ai_yaml = """version: "1.0"
name: "sandbox-verify-gated-app"
default_agent: "coder"
model:
  primary: "gemini/gemini-2.5-flash"
memory:
  type: "sqlite"
agents:
  coder:
    tools:
      - name: "run_code"
        type: "sandbox"
        requires_approval: true
"""
    with tempfile.TemporaryDirectory() as tmpdir:
        p_dir = Path(tmpdir)
        (p_dir / "ai.yaml").write_text(ai_yaml)
        verifier = GraphVerifier(project_dir=p_dir)

        with verifier_console.capture() as capture:
            verifier.verify()
        output = capture.get()

        assert "Sandbox tool(s) without requires_approval" not in output
