import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from intagrin.testing.eval_runner import load_eval_cases
from intagrin.testing.synthesizer import SyntheticEvalSynthesizer

AI_YAML = """version: "1.0"
name: "pipeline-app"
default_agent: "support"
model:
  primary: "gemini/gemini-2.5-flash"
memory:
  type: "sqlite"
agents:
  support:
    handoffs: ["billing"]
    tools:
      - name: "query_user"
        module: "tools.user_tools"
  billing:
    handoffs: []
"""

TOOL_MODULE = '''
def query_user(user_id: int, note: str = "") -> str:
    """Query user info."""
    return f"User {user_id}"
'''


def _scaffold_project(p_dir: Path):
    (p_dir / "tools").mkdir(parents=True, exist_ok=True)
    (p_dir / "tools" / "user_tools.py").write_text(TOOL_MODULE)
    (p_dir / "ai.yaml").write_text(AI_YAML)


def test_synth_output_is_readable_by_eval_runner():
    """Regression test for the exact bug the audit found: `inta synth` used to write cases under
    the `evaluations` key while `inta eval` only read `evals`, so the documented synth -> eval
    workflow silently produced zero test cases. This exercises both commands against the same
    project dir and asserts the generated cases are actually found and runnable."""
    with tempfile.TemporaryDirectory() as tmpdir:
        p_dir = Path(tmpdir)
        _scaffold_project(p_dir)

        synth = SyntheticEvalSynthesizer(project_dir=p_dir, count=10)
        synth.evolve()

        evals_path = p_dir / "tests" / "evals.yaml"
        assert evals_path.exists()

        # What run_evals() actually reads must be non-empty
        cases = load_eval_cases(p_dir)
        assert len(cases) > 0

        # Every generated case carries the fields the shared runner (and hence both `inta eval`
        # and `inta tune`) depend on.
        for case in cases:
            assert case.get("input")
            assert case.get("starting_agent") in {"support", "billing"}


def test_run_evals_executes_synthesized_cases_end_to_end():
    with tempfile.TemporaryDirectory() as tmpdir:
        p_dir = Path(tmpdir)
        _scaffold_project(p_dir)

        synth = SyntheticEvalSynthesizer(project_dir=p_dir, count=3)
        synth.evolve()

        from intagrin.testing.evaluator import run_evals

        fake_message = MagicMock(content="Done.", tool_calls=None)
        fake_response = MagicMock()
        fake_response.choices = [MagicMock(message=fake_message)]
        fake_response.usage = None

        with patch("litellm.acompletion", new=AsyncMock(return_value=fake_response)):
            # Must not raise, and must actually attempt to run the synthesized cases rather than
            # reporting "no evaluation cases found".
            with patch(
                "intagrin.testing.evaluator.console.print"
            ) as mock_print:
                asyncio.run(run_evals(p_dir))

        printed = " ".join(str(call.args[0]) for call in mock_print.call_args_list if call.args)
        assert "No evaluation cases found" not in printed
