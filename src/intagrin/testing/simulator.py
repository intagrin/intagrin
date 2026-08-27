"""`inta simulate` — Phase 1 (deterministic-only) shadow replay.

Answers "if I change this ai.yaml, what happens to conversations I've already had?" for the slice
of a config that's safe to evaluate without a network call or re-executing a tool: routers,
circuit breakers, and requires_approval flags on already-declared local tools. Any other change
(prompts, models, tool identity, lazy_load_tools, auto_route, handoffs/delegations) could alter
what the LLM itself generates, which Phase 1 deliberately does not attempt to predict — see
diff_reasons() below. That's Phase 2 (live divergence replay), not implemented here.
"""

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from ..compiler.parser import ExecutionGraph
from ..config.schema import AppConfig, LocalToolConfig
from ..runtime.memory import build_checkpointer
from ..runtime.router import SwarmRouter
from ..runtime.state_reconstruction import TurnState, reconstruct_turn_states

_DETERMINISTIC_BREADCRUMB_RE = re.compile(
    r"^Router: Transferred to (\S+) via (?:root|conditional) router"
)


@dataclass
class SessionVerdict:
    kind: str  # ROUTING_DIVERGES | NEW_CIRCUIT_BREAKER_TRIP | TRIP_REMOVED |
    # NEW_APPROVAL_GATE | APPROVAL_GATE_REMOVED
    turn: int
    detail: str


@dataclass
class SessionResult:
    session_id: str
    verdicts: list[SessionVerdict] = field(default_factory=list)

    @property
    def unchanged(self) -> bool:
        return not self.verdicts


@dataclass
class SimulationReport:
    sessions_checked: int
    results: list[SessionResult] = field(default_factory=list)
    not_simulatable_reasons: list[str] = field(default_factory=list)

    @property
    def simulatable(self) -> bool:
        return not self.not_simulatable_reasons


def _tools_identity_equal(old_tools: list, new_tools: list) -> bool:
    """True if the two tool lists are identical except possibly for requires_approval."""
    if len(old_tools) != len(new_tools):
        return False
    for ot, nt in zip(old_tools, new_tools):
        if type(ot) is not type(nt):
            return False
        od, nd = ot.model_dump(), nt.model_dump()
        od.pop("requires_approval", None)
        nd.pop("requires_approval", None)
        if od != nd:
            return False
    return True


_BEHAVIOR_AGENT_FIELDS = (
    "system_prompt_file",
    "system_prompt_module",
    "prompt_key",
    "system_prompt_langfuse",
    "model_override",
    "lazy_load_tools",
    "auto_route",
    "response_schema",
    "handoffs",
    "delegations",
)


def diff_reasons(old_cfg: AppConfig, new_cfg: AppConfig) -> list[str]:
    """Reasons a config diff can't be Phase-1-simulated. Deliberately conservative — deny by
    default: routers, circuit_breakers, max_session_budget_usd, and requires_approval-only tool
    changes are the only diffs considered safe; anything else that could change what the LLM
    generates (or even which agents/tools exist) blocks the whole run rather than being silently
    ignored."""
    reasons = []

    if old_cfg.default_agent != new_cfg.default_agent:
        reasons.append(
            f"default_agent changed ('{old_cfg.default_agent}' -> '{new_cfg.default_agent}')"
        )

    if old_cfg.model.model_dump() != new_cfg.model.model_dump():
        reasons.append("model config changed (primary/fallback/temperature/guardrails/etc.)")

    if old_cfg.rag != new_cfg.rag:
        reasons.append("rag config changed")

    if not _tools_identity_equal(old_cfg.tools, new_cfg.tools):
        reasons.append(
            "global tools changed (added/removed/reconfigured — not just requires_approval)"
        )

    old_agents, new_agents = old_cfg.agents, new_cfg.agents
    if set(old_agents) != set(new_agents):
        reasons.append("agent set changed (an agent was added or removed)")

    for name in sorted(set(old_agents) & set(new_agents)):
        oa, na = old_agents[name], new_agents[name]
        for f in _BEHAVIOR_AGENT_FIELDS:
            if getattr(oa, f, None) != getattr(na, f, None):
                reasons.append(f"agents.{name}.{f} changed")
        if not _tools_identity_equal(oa.tools, na.tools):
            reasons.append(
                f"agents.{name}.tools changed (added/removed/reconfigured — not just requires_approval)"
            )

    return reasons


def _local_approval_set(cfg: AppConfig, agent_name: str) -> set[str]:
    """Tool names requiring approval, resolvable statically. Scoped to LocalToolConfig only —
    MCP-sourced tool names require a live server connection to enumerate and OpenAPI-sourced names
    require a live spec fetch, both out of scope for Phase 1's approval-gate diff."""
    names = set()
    agent_cfg = cfg.agents.get(agent_name)
    tool_lists = [cfg.tools] + ([agent_cfg.tools] if agent_cfg else [])
    for tools in tool_lists:
        for t in tools:
            if isinstance(t, LocalToolConfig) and getattr(t, "requires_approval", False):
                names.add(t.name)
    return names


def _known_local_tool_names(cfg: AppConfig, agent_name: str) -> set[str]:
    names = set()
    agent_cfg = cfg.agents.get(agent_name)
    tool_lists = [cfg.tools] + ([agent_cfg.tools] if agent_cfg else [])
    for tools in tool_lists:
        for t in tools:
            if isinstance(t, LocalToolConfig):
                names.add(t.name)
    return names


def _round_boundaries(messages: list[dict]) -> list[int]:
    """Indices of the message that *is* a routing decision's outcome: the response (or router
    breadcrumb) right after a user message, or right after the last tool result of a completed
    tool-call round. This is where the real engine evaluates routers — at the top of a turn and
    again after each tool-execution round — so it's where a *new* router's decision needs to be
    compared against what actually happened. Index 0 is never a boundary itself: a session always
    starts with the incoming user message at index 0, and the first real decision point is
    whatever comes after it (naturally captured by the `prev_role == "user"` check at i=1)."""
    boundaries = []
    for i, msg in enumerate(messages):
        if i == 0:
            continue
        prev_role = messages[i - 1].get("role")
        if prev_role == "user" or prev_role == "tool" and msg.get("role") != "tool":
            boundaries.append(i)
    return boundaries


def _empty_turn_state(starting_agent: str, cost_tracked: bool) -> TurnState:
    return TurnState(
        turn=-1,
        active_agent=starting_agent,
        state={},
        handoff_count=0,
        tool_failure_streak=0,
        tokens_so_far=0 if cost_tracked else None,
        cost_so_far=0.0 if cost_tracked else None,
    )


def _load_condition_functions(graph: ExecutionGraph) -> dict:
    """Same loading a live RuntimeEngine.initialize() does for AppConfig.condition_functions —
    duplicated here (rather than imported) because it's genuinely this cheap (an import + getattr,
    no I/O) and pulling in RuntimeEngine here would be a much heavier dependency than the three
    lines it'd save. A function that fails to load is skipped, not fatal — a router condition that
    calls it will simply record an "Unknown condition function" evaluation error and fail open,
    same as any other broken condition."""
    from ..runtime.tools_loader import load_local_tool

    functions = {}
    for cf in graph.config.condition_functions:
        try:
            functions[cf.name] = load_local_tool(cf.module, cf.name)
        except Exception:
            pass
    return functions


def _check_new_router(new_graph: ExecutionGraph, ts: TurnState) -> tuple[bool, str | None]:
    """Would the *new* config's router fire against this reconstructed historical state? Reuses
    SwarmRouter's pure, side-effect-free evaluators — the same primitives the live engine's
    routing is built on — rather than re-deriving condition evaluation here."""
    fired, target, _err = SwarmRouter.evaluate_root_router(
        new_graph, ts.active_agent, ts.state
    )
    if fired:
        return True, target
    agent_cfg = new_graph.config.agents.get(ts.active_agent)
    functions = _load_condition_functions(new_graph)
    fired, target, _evaluations = SwarmRouter.evaluate_conditional_routers(
        agent_cfg, ts.state, functions
    )
    return fired, target


def _guess_starting_agent(session_id: str, default_agent: str) -> str:
    """Top-level sessions always start at default_agent. Delegated sub-sessions
    (`{parent}_sub_{target}`) start at their delegation target, recoverable from the session_id
    itself. Parallel-workflow branch sessions (`{parent}_branch_{subtask_name}`) are keyed by
    subtask *name*, not agent — their true starting agent can't be recovered from the id alone, so
    this falls back to default_agent for those (a known, accepted Phase 1 limitation)."""
    if "_sub_" in session_id:
        return session_id.rsplit("_sub_", 1)[-1]
    return default_agent


def _simulate_session(
    session_id: str,
    messages: list[dict],
    final_state: dict,
    old_graph: ExecutionGraph,
    new_graph: ExecutionGraph,
    project_dir: Path,
) -> SessionResult:
    old_cfg, new_cfg = old_graph.config, new_graph.config
    starting_agent = _guess_starting_agent(session_id, old_cfg.default_agent)
    cost_trace = final_state.get("_cost_trace")

    turn_states = reconstruct_turn_states(
        messages, old_graph, project_dir, starting_agent=starting_agent, cost_trace=cost_trace
    )

    result = SessionResult(session_id=session_id)

    old_cb, new_cb = old_cfg.circuit_breakers, new_cfg.circuit_breakers
    old_budget = old_cb.max_usd_cost_per_session or old_cfg.max_session_budget_usd
    new_budget = new_cb.max_usd_cost_per_session or new_cfg.max_session_budget_usd

    seen = {"handoffs": False, "failures": False, "cost": False}

    for ts in turn_states:
        if (
            new_cb.max_handoffs_per_session is not None
            and ts.handoff_count > new_cb.max_handoffs_per_session
            and not seen["handoffs"]
            and (
                old_cb.max_handoffs_per_session is None
                or ts.handoff_count <= old_cb.max_handoffs_per_session
            )
        ):
            result.verdicts.append(
                SessionVerdict(
                    "NEW_CIRCUIT_BREAKER_TRIP",
                    ts.turn,
                    f"max_handoffs_per_session: reached {ts.handoff_count}, new limit is "
                    f"{new_cb.max_handoffs_per_session} (old limit: {old_cb.max_handoffs_per_session})",
                )
            )
            seen["handoffs"] = True

        if (
            new_cb.max_tool_failures_in_a_row is not None
            and ts.tool_failure_streak >= new_cb.max_tool_failures_in_a_row
            and not seen["failures"]
            and (
                old_cb.max_tool_failures_in_a_row is None
                or ts.tool_failure_streak < old_cb.max_tool_failures_in_a_row
            )
        ):
            result.verdicts.append(
                SessionVerdict(
                    "NEW_CIRCUIT_BREAKER_TRIP",
                    ts.turn,
                    f"max_tool_failures_in_a_row: streak reached {ts.tool_failure_streak}, new "
                    f"limit is {new_cb.max_tool_failures_in_a_row} "
                    f"(old limit: {old_cb.max_tool_failures_in_a_row})",
                )
            )
            seen["failures"] = True

        if (
            new_budget is not None
            and ts.cost_so_far is not None
            and ts.cost_so_far >= new_budget
            and not seen["cost"]
            and (old_budget is None or ts.cost_so_far < old_budget)
        ):
            result.verdicts.append(
                SessionVerdict(
                    "NEW_CIRCUIT_BREAKER_TRIP",
                    ts.turn,
                    f"max_usd_cost_per_session: reached ${ts.cost_so_far:.4f}, new limit is "
                    f"${new_budget:.2f} (old limit: {old_budget})",
                )
            )
            seen["cost"] = True

        # Approval gates: only for tool calls right after this snapshot's agent, resolvable
        # statically (LocalToolConfig only — see _local_approval_set).
        if ts.turn + 1 < len(messages):
            nxt = messages[ts.turn + 1]
            if nxt.get("role") == "tool":
                tool_name = nxt.get("name")
                if tool_name and tool_name in _known_local_tool_names(old_cfg, ts.active_agent):
                    was_gated = tool_name in _local_approval_set(old_cfg, ts.active_agent)
                    is_gated = tool_name in _local_approval_set(new_cfg, ts.active_agent)
                    if is_gated and not was_gated:
                        result.verdicts.append(
                            SessionVerdict(
                                "NEW_APPROVAL_GATE",
                                ts.turn + 1,
                                f"'{tool_name}' now requires human approval (agent: {ts.active_agent})",
                            )
                        )
                    elif was_gated and not is_gated:
                        result.verdicts.append(
                            SessionVerdict(
                                "APPROVAL_GATE_REMOVED",
                                ts.turn + 1,
                                f"'{tool_name}' no longer requires human approval (agent: {ts.active_agent})",
                            )
                        )

    cost_tracked = cost_trace is not None
    empty = _empty_turn_state(starting_agent, cost_tracked)
    prior_by_boundary = {0: empty}
    for ts in turn_states:
        prior_by_boundary[ts.turn + 1] = ts

    for boundary in _round_boundaries(messages):
        pre_state = prior_by_boundary.get(boundary, empty) if boundary > 0 else empty
        actual_target = None
        actual_msg = messages[boundary]
        if actual_msg.get("role") == "system":
            m = _DETERMINISTIC_BREADCRUMB_RE.match(str(actual_msg.get("content", "")))
            if m:
                actual_target = m.group(1)

        new_fired, new_target = _check_new_router(new_graph, pre_state)

        if new_fired and new_target != actual_target:
            result.verdicts.append(
                SessionVerdict(
                    "ROUTING_DIVERGES",
                    boundary,
                    f"new router would transfer '{pre_state.active_agent}' -> '{new_target}' "
                    f"here; originally: "
                    + (f"routed to '{actual_target}'" if actual_target else "no deterministic router fired"),
                )
            )
        elif not new_fired and actual_target is not None:
            result.verdicts.append(
                SessionVerdict(
                    "ROUTING_DIVERGES",
                    boundary,
                    f"originally routed '{pre_state.active_agent}' -> '{actual_target}' here; "
                    "the new router config no longer fires",
                )
            )

    return result


async def simulate(
    project_dir: Path,
    old_graph: ExecutionGraph,
    new_graph: ExecutionGraph,
    since: datetime | None = None,
    limit: int = 200,
    session_ids: list[str] | None = None,
) -> SimulationReport:
    """Replays real historical sessions through `new_graph`'s routers/circuit-breakers/approval
    gates, using `old_graph` (the config that was actually in effect) to faithfully reconstruct
    what state really was at each turn. Returns a report with either per-session verdicts, or (if
    the diff touches anything Phase 1 can't evaluate) a list of blocking reasons and zero sessions
    checked."""
    reasons = diff_reasons(old_graph.config, new_graph.config)
    if reasons:
        return SimulationReport(sessions_checked=0, not_simulatable_reasons=reasons)

    checkpointer = build_checkpointer(old_graph.config.memory, project_dir, strict=True)
    ids = session_ids or checkpointer.list_sessions(since=since, limit=limit)

    results = []
    for session_id in ids:
        messages, final_state = checkpointer.load_checkpoint(session_id)
        if not messages:
            continue
        results.append(
            _simulate_session(session_id, messages, final_state, old_graph, new_graph, project_dir)
        )

    return SimulationReport(sessions_checked=len(results), results=results)
