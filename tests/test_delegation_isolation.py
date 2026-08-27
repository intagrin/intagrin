import asyncio
from unittest.mock import MagicMock, patch

import pytest

from intagrin.compiler.parser import ExecutionGraph
from intagrin.config.schema import (
    AgentConfig,
    AppConfig,
    MemoryConfig,
    ModelConfig,
    StateReducerConfig,
)
from intagrin.runtime.engine import RuntimeEngine


@pytest.fixture
def mock_graph():
    config = AppConfig(
        version="1.0",
        name="test_app",
        default_agent="manager",
        model=ModelConfig(primary="mock/model"),
        memory=MemoryConfig(type="buffer"),
        reducers=[
            StateReducerConfig(key="notes", strategy="append"),
        ],
        agents={
            "manager": AgentConfig(description="Manager agent", delegations=["worker"]),
            "worker": AgentConfig(description="Worker agent"),
        },
    )
    return ExecutionGraph(config=config, env_vars={})


def _is_child(engine) -> bool:
    return "_sub_" in engine.session_id


@pytest.mark.anyio
async def test_delegation_isolates_state_during_execution(mock_graph, tmp_path):
    """A delegated sub-agent's write_state during execution must not be visible on the parent's
    self.state until delegation actually returns — proves isolation during execution, not just
    eventual consistency after merge."""
    engine = RuntimeEngine(graph=mock_graph, project_dir=tmp_path, session_id="parent")
    await engine.initialize()
    engine.active_agent_name = "manager"
    engine.state["ticket_status"] = "open"

    observed_parent_status_mid_flight = []

    async def mock_run_agent_turn(self, interactive=False):
        if _is_child(self):
            # The child mutates its own state directly (equivalent to a write_state tool call).
            self.state["ticket_status"] = "resolved"
            # Snapshot what the PARENT sees while the child is still executing.
            observed_parent_status_mid_flight.append(engine.state["ticket_status"])
        self.is_transferring = False
        self.messages.append({"role": "assistant", "content": "done"})

    with patch.object(RuntimeEngine, "_run_agent_turn", new=mock_run_agent_turn):
        result = await engine.execute_tool(
            "delegate_task",
            {"target_agent": "worker", "task": "Resolve the ticket"},
            interactive=False,
        )

    assert "Delegated task completed by worker" in result
    # While the child was executing, the parent's state must still show the original value.
    assert observed_parent_status_mid_flight == ["open"]
    # After delegation returns, the change has merged back (default overwrite).
    assert engine.state["ticket_status"] == "resolved"


@pytest.mark.anyio
async def test_delegation_default_overwrites_undeclared_key(mock_graph, tmp_path):
    """A state key the sub-agent changed with no declared reducer ends up overwritten on the
    parent by default."""
    engine = RuntimeEngine(graph=mock_graph, project_dir=tmp_path, session_id="parent")
    await engine.initialize()
    engine.active_agent_name = "manager"
    engine.state["summary"] = "old"

    async def mock_run_agent_turn(self, interactive=False):
        if _is_child(self):
            self.state["summary"] = "new_from_child"
        self.is_transferring = False
        self.messages.append({"role": "assistant", "content": "done"})

    with patch.object(RuntimeEngine, "_run_agent_turn", new=mock_run_agent_turn):
        await engine.execute_tool(
            "delegate_task",
            {"target_agent": "worker", "task": "Summarize"},
            interactive=False,
        )

    assert engine.state["summary"] == "new_from_child"


@pytest.mark.anyio
async def test_delegation_reducer_strategy_and_unchanged_key_not_remerged(mock_graph, tmp_path):
    """A key with a declared `append` reducer uses that strategy; an unchanged key with an
    `append` reducer is NOT re-appended to itself (proves the pre_state-vs-child_state diffing
    works — both the unchanged-key skip and, for lists specifically, stripping the child's own
    pre-existing prefix before extending — since a delegated child starts as a full copy of the
    parent's state, list values included)."""
    engine = RuntimeEngine(graph=mock_graph, project_dir=tmp_path, session_id="parent")
    await engine.initialize()
    engine.active_agent_name = "manager"
    engine.state["notes"] = ["first"]

    call_count = {"n": 0}

    async def mock_run_agent_turn(self, interactive=False):
        if _is_child(self):
            call_count["n"] += 1
            if call_count["n"] == 1:
                # First delegation: child appends a new note.
                self.state["notes"].append("second")
            # Second delegation (called separately below): child leaves notes untouched.
        self.is_transferring = False
        self.messages.append({"role": "assistant", "content": "done"})

    with patch.object(RuntimeEngine, "_run_agent_turn", new=mock_run_agent_turn):
        await engine.execute_tool(
            "delegate_task",
            {"target_agent": "worker", "task": "Add a note"},
            interactive=False,
        )
        assert engine.state["notes"] == ["first", "second"]

        # Delegate again without the child touching `notes` at all.
        await engine.execute_tool(
            "delegate_task",
            {"target_agent": "worker", "task": "Do nothing to notes"},
            interactive=False,
        )

    # If the unchanged-key skip didn't work, the append reducer would have re-appended
    # ["first", "second"] onto itself.
    assert engine.state["notes"] == ["first", "second"]


@pytest.mark.anyio
async def test_delegation_merges_cost_as_a_delta_not_double_counted(mock_graph, tmp_path):
    """Delegated LLM usage cost/tokens reach the parent's `_metrics` as a correct delta — not
    double-counted (the child starts as a full copy of the parent's already-incurred metrics)
    and not zero (the cost-tracking leak this fix closes)."""
    engine = RuntimeEngine(graph=mock_graph, project_dir=tmp_path, session_id="parent")
    await engine.initialize()
    engine.active_agent_name = "manager"
    engine.state["_metrics"] = {"total_tokens": 100, "total_cost": 1.00}

    async def mock_run_agent_turn(self, interactive=False):
        if _is_child(self):
            # The child starts as a full copy (100 tokens / $1.00 already) and incurs its own
            # usage on top of that during execution.
            self.state["_metrics"]["total_tokens"] += 50
            self.state["_metrics"]["total_cost"] += 0.25
        self.is_transferring = False
        self.messages.append({"role": "assistant", "content": "done"})

    with patch.object(RuntimeEngine, "_run_agent_turn", new=mock_run_agent_turn):
        await engine.execute_tool(
            "delegate_task",
            {"target_agent": "worker", "task": "Do some work"},
            interactive=False,
        )

    assert engine.state["_metrics"]["total_tokens"] == 150
    assert engine.state["_metrics"]["total_cost"] == pytest.approx(1.25)


@pytest.mark.anyio
async def test_delegation_persists_a_child_pause_instead_of_reporting_false_completion(
    mock_graph, tmp_path
):
    """Regression test for a real bug: when a delegated child hits a human-approval gate mid-task
    (a requires_approval tool, or a dynamic AwaitingHumanInput raise), the loop only checked
    `is_transferring` — a paused child never sets that, so it looked identical to a child that
    simply finished with no final message. delegate_task reported "Delegated task completed" and
    the pause was never surfaced anywhere: no /resume path knew a delegation sub-session existed,
    so it was stuck forever, unreachable. This mirrors spawn_agent's own identical fix for the
    same shape of bug — the pause must instead propagate onto the PARENT's own _pending_approval
    (with a child_session_id pointer), which resume_endpoint's existing generic dispatch already
    knows how to continue."""
    engine = RuntimeEngine(graph=mock_graph, project_dir=tmp_path, session_id="parent")
    await engine.initialize()
    engine.active_agent_name = "manager"

    async def mock_run_agent_turn(self, interactive=False):
        if _is_child(self):
            # The child pauses mid-task — never sets is_transferring, and never appends a final
            # assistant message either (exactly what a real pause looks like: _run_agent_turn
            # just returns after _pause_for_human sets _pending_approval).
            self.state["_pending_approval"] = {
                "tool": "issue_refund",
                "args": {"amount": 50},
                "status": "awaiting_approval",
                "tool_call_id": "child_tool_call_1",
                "required_approvals": 1,
                "required_approvers": None,
                "approvals_received": [],
            }
        self.is_transferring = False

    with patch.object(RuntimeEngine, "_run_agent_turn", new=mock_run_agent_turn):
        result = await engine.execute_tool(
            "delegate_task",
            {"target_agent": "worker", "task": "Resolve the ticket"},
            interactive=False,
            tool_call_id="delegate_call_1",
        )

    assert "paused" in result.lower()
    assert "completed" not in result.lower()

    # The pause must land on the PARENT's own state, not be silently discarded.
    pending = engine.state["_pending_approval"]
    assert pending["tool"] == "issue_refund"
    assert pending["agent"] == "worker"
    assert pending["parent_tool_call_id"] == "delegate_call_1"
    assert pending["child_session_id"] == "parent_sub_worker"


def _tool_call(call_id: str, target_agent: str, task: str):
    tc = MagicMock()
    tc.id = call_id
    tc.function.name = "delegate_task"
    tc.function.arguments = f'{{"target_agent": "{target_agent}", "task": "{task}"}}'
    return tc


@pytest.mark.anyio
async def test_concurrent_delegate_task_calls_merge_in_declared_order_not_completion_order(
    mock_graph, tmp_path
):
    """Regression test for a real bug: two delegate_task calls issued by the model in the same
    turn run concurrently (_execute_tool_calls_with_healing gathers a round's non-transfer tool
    calls) — each delegate_task call used to merge its child's state back into the parent inline,
    the moment its OWN child execution finished, so which merge landed last (and therefore won,
    for an undeclared-reducer key) depended on real wall-clock completion timing, not on what
    order the model actually issued the calls in. Here the FIRST-declared call's child is made
    deliberately slower than the SECOND-declared call's — if merge order followed completion
    order (the bug), the first call's value would win (it finishes and merges last); the fix
    (deferred merge, applied in original tool-call order via asyncio.gather's own ordering
    guarantee, mirroring run_workflow's parallel/vote branch merging) must make the SECOND call's
    value win instead, matching the order the calls were actually declared in."""
    engine = RuntimeEngine(graph=mock_graph, project_dir=tmp_path, session_id="parent")
    await engine.initialize()
    engine.active_agent_name = "manager"

    async def mock_run_agent_turn(self, interactive=False):
        if _is_child(self):
            if "first" in self.messages[-1]["content"]:
                await asyncio.sleep(0.1)  # First-declared call's child finishes LAST.
                self.state["winner"] = "first"
            else:
                self.state["winner"] = "second"
        self.is_transferring = False
        self.messages.append({"role": "assistant", "content": "done"})

    tool_calls = [
        _tool_call("call_1", "worker", "first task"),
        _tool_call("call_2", "worker", "second task"),
    ]

    with patch.object(RuntimeEngine, "_run_agent_turn", new=mock_run_agent_turn):
        await engine._execute_tool_calls_with_healing(tool_calls, interactive=False)

    assert not engine._deferred_child_merges
    assert engine.state["winner"] == "second"


def _fan_out_tool_call(call_id: str, target_agent: str, instructions: list):
    import json

    tc = MagicMock()
    tc.id = call_id
    tc.function.name = "delegate_to_many"
    tc.function.arguments = json.dumps({"target_agent": target_agent, "instructions": instructions})
    return tc


@pytest.mark.anyio
async def test_delegate_to_many_runs_one_child_per_instruction_and_merges_all(mock_graph, tmp_path):
    engine = RuntimeEngine(graph=mock_graph, project_dir=tmp_path, session_id="parent")
    await engine.initialize()
    engine.active_agent_name = "manager"

    async def mock_run_agent_turn(self, interactive=False):
        if _is_child(self):
            task = self.messages[-1]["content"]
            self.state[f"result_{task}"] = f"done_{task}"
            self.messages.append({"role": "assistant", "content": f"Handled {task}"})
        self.is_transferring = False

    with patch.object(RuntimeEngine, "_run_agent_turn", new=mock_run_agent_turn):
        result = await engine.execute_tool(
            "delegate_to_many",
            {"target_agent": "worker", "instructions": ["Paris", "Tokyo", "NYC"]},
            interactive=False,
        )

    assert engine.state["result_Paris"] == "done_Paris"
    assert engine.state["result_Tokyo"] == "done_Tokyo"
    assert engine.state["result_NYC"] == "done_NYC"
    for city in ("Paris", "Tokyo", "NYC"):
        assert city in result


@pytest.mark.anyio
async def test_delegate_to_many_merges_in_declared_order_not_completion_order(mock_graph, tmp_path):
    """Same regression shape as the delegate_task concurrent-merge-order test above: the
    first-declared instruction's child is made deliberately slower — the last-merged (and
    therefore winning, for an undeclared-reducer key) value must still be the one from the
    LAST-declared instruction, not whichever child happened to finish first in real time."""
    engine = RuntimeEngine(graph=mock_graph, project_dir=tmp_path, session_id="parent")
    await engine.initialize()
    engine.active_agent_name = "manager"

    async def mock_run_agent_turn(self, interactive=False):
        if _is_child(self):
            task = self.messages[-1]["content"]
            if task == "first":
                await asyncio.sleep(0.1)  # First-declared finishes LAST.
            self.state["winner"] = task
        self.is_transferring = False
        self.messages.append({"role": "assistant", "content": "done"})

    with patch.object(RuntimeEngine, "_run_agent_turn", new=mock_run_agent_turn):
        await engine.execute_tool(
            "delegate_to_many",
            {"target_agent": "worker", "instructions": ["first", "second"]},
            interactive=False,
        )

    assert engine.state["winner"] == "second"


@pytest.mark.anyio
async def test_delegate_to_many_rejects_over_the_max_parallel_fan_out_limit(mock_graph, tmp_path):
    engine = RuntimeEngine(graph=mock_graph, project_dir=tmp_path, session_id="parent")
    await engine.initialize()
    engine.active_agent_name = "manager"
    engine.graph.config.circuit_breakers.max_parallel_fan_out = 2

    called = {"n": 0}

    async def mock_run_agent_turn(self, interactive=False):
        called["n"] += 1
        self.is_transferring = False
        self.messages.append({"role": "assistant", "content": "done"})

    with patch.object(RuntimeEngine, "_run_agent_turn", new=mock_run_agent_turn):
        result = await engine.execute_tool(
            "delegate_to_many",
            {"target_agent": "worker", "instructions": ["a", "b", "c"]},
            interactive=False,
        )

    assert "max_parallel_fan_out" in result
    assert called["n"] == 0  # rejected before spawning any child


@pytest.mark.anyio
async def test_delegate_to_many_rejects_unauthorized_target(mock_graph, tmp_path):
    engine = RuntimeEngine(graph=mock_graph, project_dir=tmp_path, session_id="parent")
    await engine.initialize()
    engine.active_agent_name = "manager"

    result = await engine.execute_tool(
        "delegate_to_many",
        {"target_agent": "not_a_real_agent", "instructions": ["x"]},
        interactive=False,
    )
    assert "Unauthorized" in result


@pytest.mark.anyio
async def test_delegate_to_many_surfaces_one_childs_pause_while_others_complete(mock_graph, tmp_path):
    """One instruction's child hits a human-approval gate; the others still run to completion and
    merge normally — a single paused branch must not block or discard the rest of the fan-out."""
    engine = RuntimeEngine(graph=mock_graph, project_dir=tmp_path, session_id="parent")
    await engine.initialize()
    engine.active_agent_name = "manager"

    async def mock_run_agent_turn(self, interactive=False):
        if _is_child(self):
            task = self.messages[-1]["content"]
            if task == "needs_approval":
                self.state["_pending_approval"] = {
                    "tool": "issue_refund",
                    "args": {"amount": 50},
                    "status": "awaiting_approval",
                    "tool_call_id": "child_tool_call_1",
                    "required_approvals": 1,
                    "required_approvers": None,
                    "approvals_received": [],
                }
                self.is_transferring = False
                return
            self.state[f"done_{task}"] = True
        self.is_transferring = False
        self.messages.append({"role": "assistant", "content": "done"})

    with patch.object(RuntimeEngine, "_run_agent_turn", new=mock_run_agent_turn):
        result = await engine.execute_tool(
            "delegate_to_many",
            {"target_agent": "worker", "instructions": ["needs_approval", "plain_task"]},
            interactive=False,
            tool_call_id="fanout_call_1",
        )

    assert "paused" in result.lower()
    assert engine.state["done_plain_task"] is True
    pending = engine.state["_pending_approval"]
    assert pending["tool"] == "issue_refund"
    assert pending["parent_tool_call_id"] == "fanout_call_1"
