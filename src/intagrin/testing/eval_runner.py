"""Shared eval-case loading and execution for `inta synth` / `inta eval` / `inta tune` / `inta
diagnose`. All four commands previously carried near-duplicate copies of this logic with subtly
incompatible file formats (synth wrote "evaluations", eval read "evals") — this module is now the
single source of truth for both the on-disk schema and how a case gets executed.

Canonical `tests/evals.yaml` shape:
    version: "1.0"
    evaluations:
      - name: "..."                     # optional, defaults to the input text
        input: "..."                    # required — the user message to send
        starting_agent: "agent_name"    # optional, defaults to the project's default_agent
        expected_agent: "agent_name"    # optional — assert the FINAL active agent after the run
        expected_tool: "tool_name"      # optional — assert this tool was called at least once
        expected_output_contains: "..."  # optional — substring match on the final assistant reply
        evaluators:                      # optional — LLM-judge checks, evaluator.py only
          - type: llm_judge
            criteria: "..."
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ..compiler.parser import ExecutionGraph
from ..runtime.engine import RuntimeEngine

EVALS_RELATIVE_PATH = Path("tests") / "evals.yaml"


def load_eval_cases(project_dir: Path) -> list[dict[str, Any]]:
    """Reads tests/evals.yaml and returns the list of cases under the canonical `evaluations` key.
    Falls back to the legacy `evals` key so hand-written files from before the schema was unified
    still work."""
    evals_file = project_dir / EVALS_RELATIVE_PATH
    if not evals_file.exists():
        return []
    with open(evals_file, "r") as f:
        data = yaml.safe_load(f) or {}
    return data.get("evaluations") or data.get("evals") or []


@dataclass
class EvalCaseResult:
    name: str
    input: str
    starting_agent: str
    final_agent: str = ""
    called_tools: list[str] = field(default_factory=list)
    tool_error_count: int = 0
    final_answer: str = ""
    tokens: int = 0
    cost: float = 0.0
    duration_seconds: float = 0.0
    crashed: bool = False
    crash_error: str | None = None
    deterministic_pass: bool = True
    reasons: list[str] = field(default_factory=list)


async def run_case(
    graph: ExecutionGraph, project_dir: Path, case: dict[str, Any]
) -> EvalCaseResult:
    """Runs one eval case headlessly to completion against a fresh engine and returns a structured
    result covering both correctness checks (expected_agent/tool/output) and health metrics
    (tokens, cost, latency, tool error count) — shared by inta eval, inta tune, and inta diagnose
    so all three agree on what "ran this case" means."""
    starting_agent = case.get("starting_agent", graph.config.default_agent)
    user_input = case.get("input", "")
    name = case.get("name", user_input[:40] or "unnamed case")

    result = EvalCaseResult(name=name, input=user_input, starting_agent=starting_agent)

    engine = RuntimeEngine(graph=graph, project_dir=project_dir, session_id=f"eval_{id(case)}")
    await engine.initialize()
    engine.active_agent_name = starting_agent

    safe_input = engine._apply_guardrails(user_input)
    engine.messages.append({"role": "user", "content": safe_input})

    start = time.monotonic()
    try:
        while True:
            engine.is_transferring = False
            await engine._run_agent_turn(interactive=False)
            if not engine.is_transferring:
                break
    except Exception as e:
        result.crashed = True
        result.crash_error = str(e)
    finally:
        result.duration_seconds = time.monotonic() - start
        await engine.mcp_manager.cleanup()

    result.final_agent = engine.active_agent_name
    result.called_tools = [m.get("name") for m in engine.messages if m.get("role") == "tool"]
    result.tool_error_count = sum(
        1
        for m in engine.messages
        if m.get("role") == "tool" and "error" in str(m.get("content", "")).lower()
    )
    result.final_answer = next(
        (
            m.get("content")
            for m in reversed(engine.messages)
            if m.get("role") == "assistant" and m.get("content")
        ),
        "",
    )
    metrics = engine.state.get("_metrics", {})
    result.tokens = metrics.get("total_tokens", 0)
    result.cost = metrics.get("total_cost", 0.0)

    reasons = []
    if result.crashed:
        reasons.append(f"Engine crashed: {result.crash_error}")
    expected_agent = case.get("expected_agent")
    if expected_agent and result.final_agent != expected_agent:
        reasons.append(f"Expected final agent '{expected_agent}' but got '{result.final_agent}'")
    expected_tool = case.get("expected_tool")
    if expected_tool and expected_tool not in result.called_tools:
        reasons.append(f"Expected tool call '{expected_tool}' was not called")
    expected_contains = case.get("expected_output_contains")
    if expected_contains and expected_contains.lower() not in result.final_answer.lower():
        reasons.append(f"Output missing expected keyword '{expected_contains}'")

    result.reasons = reasons
    result.deterministic_pass = not reasons
    return result
