"""Tests for AutoTuner (`inta tune`) — previously had zero coverage, including its core safety
claim: a non-converging tuning run must never leave a project's prompts worse off than when it
started. Follows the established monkeypatch.setattr(litellm, "completion"/"acompletion", ...)
mocking pattern already used in tests/test_compile.py, plus monkeypatching the `run_case` name
tuner.py imported directly from eval_runner (patching eval_runner.run_case itself wouldn't affect
tuner.py's own already-bound reference)."""

import asyncio

import litellm
import pytest

from intagrin.testing import tuner as tuner_module
from intagrin.testing.eval_runner import EvalCaseResult
from intagrin.testing.tuner import AutoTuner

AI_YAML = """version: "1.0"
name: tuning-test
default_agent: triage
model:
  primary: mock/model
memory:
  type: sqlite
agents:
  triage:
    description: Routes tickets.
    system_prompt_file: prompts/triage.jinja2
"""

EVALS_YAML = """version: "1.0"
evaluations:
  - name: case1
    input: "Hello"
    expected_agent: triage
"""

ORIGINAL_PROMPT = "You are the original triage agent.\n"


@pytest.fixture
def tuning_project(tmp_path):
    (tmp_path / "ai.yaml").write_text(AI_YAML)
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "evals.yaml").write_text(EVALS_YAML)
    (tmp_path / "prompts").mkdir()
    (tmp_path / "prompts" / "triage.jinja2").write_text(ORIGINAL_PROMPT)
    return tmp_path


def _result(passed: bool) -> EvalCaseResult:
    return EvalCaseResult(
        name="case1",
        input="Hello",
        starting_agent="triage",
        final_agent="triage",
        deterministic_pass=passed,
        reasons=[] if passed else ["expected_agent mismatch"],
        final_answer="some answer",
    )


def _mock_completion_response(text: str):
    class _Msg:
        content = text

    class _Choice:
        message = _Msg()

    class _Resp:
        choices = [_Choice()]

    return _Resp()


def test_tune_rolls_back_every_touched_prompt_when_it_never_converges(
    tuning_project, monkeypatch
):
    """The core safety claim in AutoTuner's own docstring: if tuning exhausts max_iterations
    without reaching zero failures, every prompt file it touched must be restored to its exact
    original content — a non-converging run can never leave the project worse than it started."""
    async def always_fails(graph, project_dir, case):
        return _result(passed=False)

    monkeypatch.setattr(tuner_module, "run_case", always_fails)

    async def mock_acompletion(*args, **kwargs):
        return _mock_completion_response("You are a REPAIRED (but still failing) triage agent.")

    monkeypatch.setattr(litellm, "acompletion", mock_acompletion)

    tuner = AutoTuner(project_dir=tuning_project, max_iterations=2)
    asyncio.run(tuner.tune())

    prompt_path = tuning_project / "prompts" / "triage.jinja2"
    assert prompt_path.read_text(encoding="utf-8") == ORIGINAL_PROMPT
    assert prompt_path in tuner._original_prompts


def test_tune_converges_and_keeps_the_repaired_prompt_on_success(tuning_project, monkeypatch):
    """A case that fails once then passes must converge without rolling back — the repaired
    prompt content should be what's left on disk, not the original."""
    call_count = {"n": 0}

    async def fails_then_passes(graph, project_dir, case):
        call_count["n"] += 1
        return _result(passed=call_count["n"] > 1)

    monkeypatch.setattr(tuner_module, "run_case", fails_then_passes)

    repaired_text = "You are the REPAIRED and now-passing triage agent."

    async def mock_acompletion(*args, **kwargs):
        return _mock_completion_response(repaired_text)

    monkeypatch.setattr(litellm, "acompletion", mock_acompletion)

    tuner = AutoTuner(project_dir=tuning_project, max_iterations=3)
    asyncio.run(tuner.tune())

    prompt_path = tuning_project / "prompts" / "triage.jinja2"
    assert prompt_path.read_text(encoding="utf-8") == repaired_text
    assert call_count["n"] == 2  # one failing pass, one passing pass — no wasted iterations


def test_tune_with_no_eval_cases_does_nothing(tmp_path, monkeypatch):
    """No tests/evals.yaml at all — tune() should report it and return, not crash or touch
    anything."""
    (tmp_path / "ai.yaml").write_text(AI_YAML)

    called = {"run_case": False}

    async def should_not_be_called(*args, **kwargs):
        called["run_case"] = True
        return _result(passed=True)

    monkeypatch.setattr(tuner_module, "run_case", should_not_be_called)

    tuner = AutoTuner(project_dir=tmp_path, max_iterations=2)
    asyncio.run(tuner.tune())

    assert called["run_case"] is False
    assert tuner._original_prompts == {}
