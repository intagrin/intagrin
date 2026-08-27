import json
from pathlib import Path

import pytest

from intagrin.compiler.parser import ExecutionGraph
from intagrin.config.schema import (
    AgentConfig,
    AppConfig,
    MemoryConfig,
    ModelConfig,
    StateReducerConfig,
)
from intagrin.runtime.state_reconstruction import reconstruct_turn_states


def _graph(reducers=None):
    config = AppConfig(
        version="1.0",
        name="reconstruct-test",
        default_agent="triage",
        model=ModelConfig(primary="mock/model"),
        memory=MemoryConfig(type="buffer"),
        agents={"triage": AgentConfig(), "billing": AgentConfig(), "support": AgentConfig()},
        reducers=reducers or [],
    )
    return ExecutionGraph(config, {})


def _write_state_call(tc_id: str, key: str, value):
    return {
        "role": "assistant",
        "tool_calls": [
            {
                "id": tc_id,
                "type": "function",
                "function": {"name": "write_state", "arguments": json.dumps({"key": key, "value": value})},
            }
        ],
    }


def _write_state_result(tc_id: str, key: str, strategy: str = "overwrite"):
    return {
        "role": "tool",
        "tool_call_id": tc_id,
        "name": "write_state",
        "content": f"Wrote '{key}' to state successfully using '{strategy}' strategy.",
    }


def test_overwrite_reducer_reconstructs_final_value():
    messages = [
        {"role": "user", "content": "hi"},
        _write_state_call("c1", "x", "1"),
        _write_state_result("c1", "x"),
        _write_state_call("c2", "x", "2"),
        _write_state_result("c2", "x"),
    ]
    snapshots = reconstruct_turn_states(messages, _graph(), Path.cwd(), starting_agent="triage")
    assert snapshots[-1].state == {"x": 2}
    # The write only lands once the matching tool *result* (index 2) is processed, not at the
    # assistant tool_call itself (index 1) — value is parsed from the numeric-looking string,
    # matching apply_state_write's json.loads coercion.
    assert snapshots[2].state == {"x": 1}


def test_append_reducer_reconstructs_accumulated_list():
    reducers = [StateReducerConfig(key="tags", strategy="append")]
    messages = [
        _write_state_call("c1", "tags", "a"),
        _write_state_result("c1", "tags", "append"),
        _write_state_call("c2", "tags", "b"),
        _write_state_result("c2", "tags", "append"),
    ]
    snapshots = reconstruct_turn_states(messages, _graph(reducers), Path.cwd(), starting_agent="triage")
    assert snapshots[-1].state == {"tags": ["a", "b"]}


def test_deep_merge_reducer_reconstructs_merged_dict():
    reducers = [StateReducerConfig(key="ctx", strategy="deep_merge")]
    messages = [
        _write_state_call("c1", "ctx", json.dumps({"a": 1})),
        _write_state_result("c1", "ctx", "deep_merge"),
        _write_state_call("c2", "ctx", json.dumps({"b": 2})),
        _write_state_result("c2", "ctx", "deep_merge"),
    ]
    snapshots = reconstruct_turn_states(messages, _graph(reducers), Path.cwd(), starting_agent="triage")
    assert snapshots[-1].state == {"ctx": {"a": 1, "b": 2}}


def test_rejected_write_state_does_not_change_state_but_resets_failure_streak():
    messages = [
        {
            "role": "tool",
            "tool_call_id": "c0",
            "name": "some_tool",
            "content": "Tool 'some_tool' execution failed: boom",
        },
        _write_state_call("c1", "x", "1"),
        {
            "role": "tool",
            "tool_call_id": "c1",
            "name": "write_state",
            "content": "Write to 'x' rejected: does not satisfy state_schema 'schemas.S'. err",
        },
    ]
    snapshots = reconstruct_turn_states(messages, _graph(), Path.cwd(), starting_agent="triage")
    assert snapshots[0].tool_failure_streak == 1
    assert snapshots[-1].state == {}
    assert snapshots[-1].tool_failure_streak == 0


def test_transfer_agent_updates_active_agent_and_handoff_count():
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "t1",
                    "type": "function",
                    "function": {
                        "name": "transfer_agent",
                        "arguments": json.dumps({"target_agent": "billing", "reason": "refund"}),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "t1",
            "name": "transfer_agent",
            "content": "Transferred to billing. Context/Reason: refund",
        },
    ]
    snapshots = reconstruct_turn_states(messages, _graph(), Path.cwd(), starting_agent="triage")
    assert snapshots[-1].active_agent == "billing"
    assert snapshots[-1].handoff_count == 1


def test_router_breadcrumb_updates_active_agent_and_handoff_count():
    messages = [
        {
            "role": "system",
            "content": "Router: Transferred to support via conditional router ('balance < 0').",
        }
    ]
    snapshots = reconstruct_turn_states(messages, _graph(), Path.cwd(), starting_agent="triage")
    assert snapshots[-1].active_agent == "support"
    assert snapshots[-1].handoff_count == 1


def test_semantic_router_breadcrumb_updates_active_agent():
    messages = [
        {"role": "system", "content": "Semantic Swarm Router: Control transferred to billing."}
    ]
    snapshots = reconstruct_turn_states(messages, _graph(), Path.cwd(), starting_agent="triage")
    assert snapshots[-1].active_agent == "billing"
    assert snapshots[-1].handoff_count == 1


def test_consecutive_tool_failures_streak_and_reset():
    messages = [
        {"role": "tool", "tool_call_id": "a", "name": "t", "content": "Tool 't' execution failed: x"},
        {"role": "tool", "tool_call_id": "b", "name": "t", "content": "Tool 't' execution failed: x"},
        {"role": "tool", "tool_call_id": "c", "name": "t", "content": "ok result"},
        {"role": "tool", "tool_call_id": "d", "name": "t", "content": "Tool 't' execution failed: x"},
    ]
    snapshots = reconstruct_turn_states(messages, _graph(), Path.cwd(), starting_agent="triage")
    assert [s.tool_failure_streak for s in snapshots] == [1, 2, 0, 1]


def test_cost_trace_reconstructs_running_totals_at_the_right_turns():
    messages = [
        {"role": "user", "content": "hi"},          # idx 0
        {"role": "assistant", "content": "hello"},   # idx 1 -- cost trace entry lands here
        {"role": "user", "content": "more"},         # idx 2
        {"role": "assistant", "content": "done"},    # idx 3 -- second cost trace entry
    ]
    cost_trace = [
        {"turn": 1, "tokens": 100, "cost": 0.01},
        {"turn": 3, "tokens": 50, "cost": 0.02},
    ]
    snapshots = reconstruct_turn_states(
        messages, _graph(), Path.cwd(), starting_agent="triage", cost_trace=cost_trace
    )
    assert [s.tokens_so_far for s in snapshots] == [0, 100, 100, 150]
    assert [s.cost_so_far for s in snapshots] == [0.0, 0.01, 0.01, pytest.approx(0.03)]


def test_missing_cost_trace_leaves_cost_fields_as_none_not_a_guess():
    messages = [{"role": "user", "content": "hi"}]
    snapshots = reconstruct_turn_states(messages, _graph(), Path.cwd(), starting_agent="triage", cost_trace=None)
    assert snapshots[0].tokens_so_far is None
    assert snapshots[0].cost_so_far is None
