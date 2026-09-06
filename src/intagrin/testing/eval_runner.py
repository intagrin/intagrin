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

import json
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
    # (tool_name, json.dumps(args, sort_keys=True)) per call, in call order — lets icu.py detect
    # an agent redoing the exact same call within one case (a context-distraction proxy), which
    # a bare tool-name list can't distinguish from legitimately calling the same tool twice with
    # different arguments.
    tool_call_log: list[tuple[str, str]] = field(default_factory=list)
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
    for m in engine.messages:
        if m.get("role") != "assistant":
            continue
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function", {})
            raw_args = fn.get("arguments", "{}")
            try:
                normalized_args = json.dumps(json.loads(raw_args), sort_keys=True)
            except (TypeError, ValueError):
                normalized_args = str(raw_args)
            result.tool_call_log.append((fn.get("name", ""), normalized_args))
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


@dataclass
class RoutingAccuracy:
    """Aggregate routing-correctness summary over a batch of eval cases that set
    `expected_agent` — see compute_routing_accuracy. `semantic_total`/`semantic_correct` are the
    subset whose `starting_agent` has `auto_route: true`: the specific number this framework
    doesn't currently have any other way to measure. A prerequisite baseline for ever considering
    replacing auto_route's LLM-based routing decision (runtime/router.py's
    evaluate_semantic_routing) with a cheaper heuristic (e.g. an embedding-similarity gate, the
    same fast-path pattern already applied to lazy_load_tools) — that swap must not regress this
    number, and there was previously no way to even know what "this number" was."""

    total: int = 0
    correct: int = 0
    semantic_total: int = 0
    semantic_correct: int = 0

    @property
    def accuracy(self) -> float | None:
        return (self.correct / self.total) if self.total else None

    @property
    def semantic_accuracy(self) -> float | None:
        return (self.semantic_correct / self.semantic_total) if self.semantic_total else None


def compute_routing_accuracy(
    graph: ExecutionGraph, cases: list[dict[str, Any]], results: list[EvalCaseResult]
) -> RoutingAccuracy:
    """Aggregates over every case that set `expected_agent`, in the same order as `cases`/
    `results` (both produced by iterating the same case list — callers must not reorder one
    without the other). A case's `starting_agent` counts toward the `semantic_*` subset when that
    agent has `auto_route: true` — a reasonable proxy for "semantic routing decided this," since
    eval cases are single-hop probes (one starting agent, one message) by convention, not
    multi-turn transcripts where routing could span several agents."""
    acc = RoutingAccuracy()
    for case, result in zip(cases, results):
        expected_agent = case.get("expected_agent")
        if not expected_agent:
            continue
        is_correct = result.final_agent == expected_agent
        acc.total += 1
        acc.correct += int(is_correct)

        starting_cfg = graph.config.agents.get(result.starting_agent)
        if starting_cfg is not None and getattr(starting_cfg, "auto_route", False):
            acc.semantic_total += 1
            acc.semantic_correct += int(is_correct)
    return acc
