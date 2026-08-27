import tempfile
from pathlib import Path

from intagrin.compiler.parser import parse_project
from intagrin.testing.eval_runner import EvalCaseResult
from intagrin.testing.icu import (
    CPI_ELEVATED_THRESHOLD,
    LATENCY_SLA_SECONDS,
    TOOL_ERROR_RATE_THRESHOLD,
    AgentICUDiagnostics,
)

AI_YAML = """version: "1.0"
name: "icu-app"
default_agent: "assistant"
model:
  primary: "gemini/gemini-2.5-flash"
memory:
  type: "sqlite"
agents:
  assistant: {}
"""


def _graph(tmp_path: Path):
    (tmp_path / "ai.yaml").write_text(AI_YAML)
    return parse_project(tmp_path)


def test_aggregate_computes_real_numbers_not_placeholders():
    with tempfile.TemporaryDirectory() as tmpdir:
        graph = _graph(Path(tmpdir))
        icu = AgentICUDiagnostics(project_dir=Path(tmpdir))

        results = [
            EvalCaseResult(
                name="a", input="x", starting_agent="assistant",
                tokens=1000, cost=0.01, duration_seconds=1.0,
                called_tools=["t1"], tool_error_count=0,
            ),
            EvalCaseResult(
                name="b", input="y", starting_agent="assistant",
                tokens=3000, cost=0.02, duration_seconds=2.0,
                called_tools=["t1", "t2"], tool_error_count=1,
            ),
        ]

        m = icu._aggregate(graph, results)

        assert m["n"] == 2
        assert m["total_tokens"] == 4000
        assert m["avg_tokens_per_run"] == 2000
        assert m["total_cost"] == 0.01 + 0.02
        assert m["total_tool_calls"] == 3
        assert m["total_tool_errors"] == 1
        assert m["tool_error_rate"] == 1 / 3
        assert m["avg_latency"] == 1.5
        assert m["max_latency"] == 2.0
        assert m["crash_rate"] == 0.0
        # cpi is a real ratio of measured tokens to the model's context window, not a hardcoded value
        assert 0 < m["cpi"] < 1
        assert m["cpi"] == m["avg_tokens_per_run"] / m["context_window"]


def test_aggregate_flags_crashes_and_tool_errors_above_threshold():
    with tempfile.TemporaryDirectory() as tmpdir:
        graph = _graph(Path(tmpdir))
        icu = AgentICUDiagnostics(project_dir=Path(tmpdir))

        results = [
            EvalCaseResult(
                name="a", input="x", starting_agent="assistant",
                crashed=True, crash_error="boom", duration_seconds=1.0,
            ),
            EvalCaseResult(
                name="b", input="y", starting_agent="assistant",
                called_tools=["t1"], tool_error_count=1, duration_seconds=1.0,
            ),
        ]

        m = icu._aggregate(graph, results)

        assert m["crash_rate"] == 0.5
        assert len(m["crashed"]) == 1
        assert m["crashed"][0].crash_error == "boom"
        assert m["tool_error_rate"] == 1.0
        assert m["tool_error_rate"] >= TOOL_ERROR_RATE_THRESHOLD


def test_diagnosis_reports_no_pathology_when_all_metrics_are_healthy(capsys):
    with tempfile.TemporaryDirectory() as tmpdir:
        graph = _graph(Path(tmpdir))
        icu = AgentICUDiagnostics(project_dir=Path(tmpdir))

        results = [
            EvalCaseResult(
                name="a", input="x", starting_agent="assistant",
                tokens=100, cost=0.001, duration_seconds=0.5,
                called_tools=["t1"], tool_error_count=0,
            )
        ]
        m = icu._aggregate(graph, results)
        assert m["cpi"] < CPI_ELEVATED_THRESHOLD
        assert m["avg_latency"] < LATENCY_SLA_SECONDS

        icu._render_diagnosis(m, results)
        out = capsys.readouterr().out
        assert "No pathology detected" in out
        assert "Pathology detected" not in out
