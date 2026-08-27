"""Reconstructs a faithful state-at-each-turn timeline for a historical session from its
checkpointed `messages` list alone.

Checkpoints only ever retain the *final* `state` snapshot — `save_checkpoint` overwrites it on
every save — so there is no per-turn state history to read directly. `messages`, by contrast, is
append-only and never overwritten, so it retains full fidelity. This module rebuilds state at each
turn by replaying every `write_state` call that actually succeeded through
`engine.apply_state_write` — the exact same pure reducer (overwrite/append/deep_merge) +
state_schema validation logic RuntimeEngine.write_state uses, using the config that was truly in
effect when the session ran. Calling the extracted pure function (rather than constructing a full
RuntimeEngine) avoids also constructing a *real* checkpointer — a DB connection, or a hard
dependency on `psycopg`/`redis` being installed — for what only needs a plain dict.

Used by `inta simulate` to check whether a *different* config's routers/circuit-breakers would
behave differently against real historical trajectories, without a network call or re-executing a
tool.
"""

import copy
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..compiler.parser import ExecutionGraph
from .engine import apply_state_write

# Deterministic/root routers leave a system-message breadcrumb with no structured data alongside
# it (routing doesn't go through a tool call) — this is the only durable record of which agent a
# router transferred to. Coupled to the exact strings RuntimeEngine._resolve_routing emits; also
# exercised by test_conditional_router_leaves_a_message_breadcrumb_and_emits_handoff_event.
_ROUTER_BREADCRUMB_RE = re.compile(
    r"^Router: Transferred to (\S+) via (?:root|conditional) router"
)
_SEMANTIC_BREADCRUMB_RE = re.compile(r"^Semantic Swarm Router: Control transferred to (\S+)\.")


@dataclass
class TurnState:
    """A snapshot of everything routers/circuit-breakers can depend on, as of right after
    `messages[turn]`."""

    turn: int
    active_agent: str
    state: dict[str, Any]
    handoff_count: int
    tool_failure_streak: int
    tokens_so_far: int | None
    cost_so_far: float | None


def reconstruct_turn_states(
    messages: list[dict],
    old_graph: ExecutionGraph,
    project_dir: Path,
    starting_agent: str,
    cost_trace: list[dict] | None = None,
) -> list[TurnState]:
    """Walks `messages` in order, rebuilding state/active-agent/handoff-count/tool-failure-streak
    at each point. `cost_trace` is `state["_cost_trace"]` from the session's final checkpoint
    (see RuntimeEngine._record_usage) — when absent, `tokens_so_far`/`cost_so_far` are `None` for
    every snapshot rather than silently reporting a wrong number.
    """
    state: dict[str, Any] = {}

    active_agent = starting_agent
    handoff_count = 0
    failure_streak = 0
    pending_write_state_args: dict[str, dict] = {}
    pending_transfer_targets: dict[str, str | None] = {}

    cost_by_turn: dict[int, tuple[int, float]] = {}
    if cost_trace:
        for entry in cost_trace:
            cost_by_turn[entry["turn"]] = (entry.get("tokens", 0), entry.get("cost", 0.0))
    tokens_so_far: int | None = 0 if cost_trace is not None else None
    cost_so_far: float | None = 0.0 if cost_trace is not None else None

    snapshots: list[TurnState] = []

    for idx, msg in enumerate(messages):
        role = msg.get("role")

        if role == "assistant":
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function", {})
                tc_id = tc.get("id")
                if not tc_id:
                    continue
                fname = fn.get("name")
                if fname == "write_state":
                    try:
                        args = json.loads(fn.get("arguments") or "{}")
                    except Exception:
                        args = None
                    if isinstance(args, dict):
                        pending_write_state_args[tc_id] = args
                elif fname == "transfer_agent":
                    try:
                        args = json.loads(fn.get("arguments") or "{}")
                    except Exception:
                        args = None
                    pending_transfer_targets[tc_id] = (
                        args.get("target_agent") if isinstance(args, dict) else None
                    )

        elif role == "tool":
            content = str(msg.get("content", ""))
            tc_id = msg.get("tool_call_id")
            tool_name = msg.get("name")

            if tool_name == "transfer_agent":
                # Matches RuntimeEngine.execute_tool's transfer_agent branch, which returns
                # before the generic tool_failures counter is touched either way.
                handoff_count += 1
                target = pending_transfer_targets.get(tc_id)
                if target:
                    active_agent = target
            elif tool_name == "delegate_task":
                # Same: delegate_task's branch also returns before tool_failures is touched.
                pass
            elif tool_name == "write_state":
                # write_state never raises — success or schema-rejection, execute_tool's normal
                # (non-exception) return path always resets the failure streak to 0.
                if content.startswith("Wrote '"):
                    wargs = pending_write_state_args.get(tc_id)
                    if wargs and wargs.get("key") is not None:
                        state, _ = apply_state_write(
                            state,
                            wargs["key"],
                            wargs.get("value"),
                            old_graph.config.reducers,
                            old_graph.config.state_schema,
                            project_dir,
                        )
                failure_streak = 0
            else:
                # Matches execute_tool's exception handler, which formats failures as
                # "Tool '<name>' execution failed: <error>" and increments tool_failures; any
                # other (non-exception) result resets it to 0.
                is_error = "execution failed:" in content
                failure_streak = failure_streak + 1 if is_error else 0

        elif role == "system":
            text = str(msg.get("content", ""))
            m = _ROUTER_BREADCRUMB_RE.match(text) or _SEMANTIC_BREADCRUMB_RE.match(text)
            if m:
                handoff_count += 1
                active_agent = m.group(1)

        if idx in cost_by_turn:
            turn_tokens, turn_cost = cost_by_turn[idx]
            tokens_so_far = (tokens_so_far or 0) + turn_tokens
            cost_so_far = (cost_so_far or 0.0) + turn_cost

        snapshots.append(
            TurnState(
                turn=idx,
                active_agent=active_agent,
                state=copy.deepcopy(state),
                handoff_count=handoff_count,
                tool_failure_streak=failure_streak,
                tokens_so_far=tokens_so_far,
                cost_so_far=cost_so_far,
            )
        )

    return snapshots
