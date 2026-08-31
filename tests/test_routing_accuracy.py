"""Tests for testing/eval_runner.py's compute_routing_accuracy — the prerequisite baseline for
ever considering replacing auto_route's per-turn LLM routing call with a cheaper heuristic (see
RoutingAccuracy's own docstring). Previously there was no way to measure this number at all."""

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from intagrin.compiler.parser import ExecutionGraph
from intagrin.config.schema import AgentConfig, AppConfig, MemoryConfig, ModelConfig
from intagrin.testing.eval_runner import EvalCaseResult, compute_routing_accuracy


def _graph():
    config = AppConfig(
        version="1.0",
        name="routing-accuracy-test",
        default_agent="triage",
        model=ModelConfig(primary="mock/model"),
        memory=MemoryConfig(type="buffer"),
        agents={
            "triage": AgentConfig(auto_route=True),
            "support": AgentConfig(auto_route=True),
            "billing": AgentConfig(auto_route=False),
        },
    )
    return ExecutionGraph(config, {})


def _result(starting_agent: str, final_agent: str) -> EvalCaseResult:
    return EvalCaseResult(name="c", input="x", starting_agent=starting_agent, final_agent=final_agent)


def test_counts_correct_and_incorrect_routing_decisions():
    graph = _graph()
    cases = [
        {"starting_agent": "triage", "expected_agent": "support"},
        {"starting_agent": "triage", "expected_agent": "billing"},
    ]
    results = [
        _result("triage", "support"),  # correct
        _result("triage", "support"),  # wrong — expected billing
    ]

    acc = compute_routing_accuracy(graph, cases, results)
    assert acc.total == 2
    assert acc.correct == 1
    assert acc.accuracy == 0.5


def test_ignores_cases_without_expected_agent():
    graph = _graph()
    cases = [
        {"starting_agent": "triage"},  # no expected_agent — must not count
        {"starting_agent": "triage", "expected_agent": "support"},
    ]
    results = [
        _result("triage", "billing"),
        _result("triage", "support"),
    ]

    acc = compute_routing_accuracy(graph, cases, results)
    assert acc.total == 1
    assert acc.correct == 1


def test_splits_out_the_semantic_auto_route_subset():
    graph = _graph()
    cases = [
        {"starting_agent": "triage", "expected_agent": "support"},  # auto_route=True, correct
        {"starting_agent": "triage", "expected_agent": "billing"},  # auto_route=True, wrong
        {"starting_agent": "billing", "expected_agent": "support"},  # auto_route=False
    ]
    results = [
        _result("triage", "support"),
        _result("triage", "support"),
        _result("billing", "support"),
    ]

    acc = compute_routing_accuracy(graph, cases, results)
    assert acc.total == 3
    assert acc.correct == 2
    assert acc.semantic_total == 2  # only the two "triage" (auto_route=True) cases
    assert acc.semantic_correct == 1
    assert acc.semantic_accuracy == 0.5


def test_accuracy_properties_are_none_with_no_cases():
    acc = compute_routing_accuracy(_graph(), [], [])
    assert acc.accuracy is None
    assert acc.semantic_accuracy is None


def test_unknown_starting_agent_does_not_crash_and_is_excluded_from_semantic_subset():
    """A case referencing an agent name that isn't (or is no longer) in the config must not
    raise — just fall outside the semantic subset, same as any non-auto_route agent."""
    graph = _graph()
    cases = [{"starting_agent": "ghost_agent", "expected_agent": "support"}]
    results = [_result("ghost_agent", "support")]

    acc = compute_routing_accuracy(graph, cases, results)
    assert acc.total == 1
    assert acc.correct == 1
    assert acc.semantic_total == 0


# --- integration: inta eval's printed summary -----------------------------------------------


AI_YAML = """version: "1.0"
name: "routing-eval-app"
default_agent: "triage"
model:
  primary: "mock/model"
memory:
  type: "sqlite"
agents:
  triage:
    auto_route: true
    description: "Routes."
  billing:
    description: "Handles billing."
"""


def test_run_evals_prints_routing_accuracy_when_expected_agent_cases_exist():
    with tempfile.TemporaryDirectory() as tmpdir:
        p_dir = Path(tmpdir)
        (p_dir / "ai.yaml").write_text(AI_YAML)
        (p_dir / "tests").mkdir()
        (p_dir / "tests" / "evals.yaml").write_text(
            """version: "1.0"
evaluations:
  - name: "routes to billing"
    input: "I have a billing question"
    starting_agent: "triage"
    expected_agent: "billing"
"""
        )

        from intagrin.testing.evaluator import run_evals

        fake_message = MagicMock(content="Done.", tool_calls=None)
        fake_response = MagicMock()
        fake_response.choices = [MagicMock(message=fake_message)]
        fake_response.usage = None

        with patch("litellm.acompletion", new=AsyncMock(return_value=fake_response)):
            with patch("intagrin.testing.evaluator.console.print") as mock_print:
                asyncio.run(run_evals(p_dir))

        printed = " ".join(str(call.args[0]) for call in mock_print.call_args_list if call.args)
        assert "Routing accuracy:" in printed
        assert "auto_route (semantic) subset" in printed


def test_run_evals_omits_routing_summary_without_expected_agent_cases():
    with tempfile.TemporaryDirectory() as tmpdir:
        p_dir = Path(tmpdir)
        (p_dir / "ai.yaml").write_text(AI_YAML)
        (p_dir / "tests").mkdir()
        (p_dir / "tests" / "evals.yaml").write_text(
            """version: "1.0"
evaluations:
  - name: "no routing assertion"
    input: "hello"
    starting_agent: "triage"
"""
        )

        from intagrin.testing.evaluator import run_evals

        fake_message = MagicMock(content="Done.", tool_calls=None)
        fake_response = MagicMock()
        fake_response.choices = [MagicMock(message=fake_message)]
        fake_response.usage = None

        with patch("litellm.acompletion", new=AsyncMock(return_value=fake_response)):
            with patch("intagrin.testing.evaluator.console.print") as mock_print:
                asyncio.run(run_evals(p_dir))

        printed = " ".join(str(call.args[0]) for call in mock_print.call_args_list if call.args)
        assert "Routing accuracy:" not in printed
