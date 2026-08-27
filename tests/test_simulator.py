import asyncio

from intagrin.compiler.parser import ExecutionGraph
from intagrin.config.schema import (
    AgentConfig,
    AppConfig,
    CircuitBreakersConfig,
    LocalToolConfig,
    MemoryConfig,
    ModelConfig,
    RouterConfig,
)
from intagrin.runtime.memory import SQLiteCheckpointer
from intagrin.testing.simulator import diff_reasons, simulate


def _config(**overrides) -> AppConfig:
    base = dict(
        version="1.0",
        name="sim-test",
        default_agent="triage",
        model=ModelConfig(primary="mock/model"),
        memory=MemoryConfig(type="sqlite"),
        agents={"triage": AgentConfig(), "billing": AgentConfig()},
    )
    base.update(overrides)
    return AppConfig(**base)


def _seed_session(tmp_path, session_id: str, messages: list[dict], state: dict | None = None):
    cp = SQLiteCheckpointer(str(tmp_path / ".ai" / "memory.db"))
    cp.save_checkpoint(session_id, messages, state or {})


def test_diff_reasons_flags_model_change_but_allows_router_and_approval_only_changes():
    old_cfg = _config()
    new_router_cfg = _config(
        agents={
            "triage": AgentConfig(routers=[RouterConfig(condition="True", target="billing")]),
            "billing": AgentConfig(),
        }
    )
    assert diff_reasons(old_cfg, new_router_cfg) == []

    old_with_tool = _config(tools=[LocalToolConfig(name="t", module="tools.x")])
    new_with_approval = _config(
        tools=[LocalToolConfig(name="t", module="tools.x", requires_approval=True)]
    )
    assert diff_reasons(old_with_tool, new_with_approval) == []

    new_model_cfg = _config(model=ModelConfig(primary="openai/gpt-4o"))
    reasons = diff_reasons(old_cfg, new_model_cfg)
    assert reasons and "model" in reasons[0]

    new_tool_added = _config(tools=[LocalToolConfig(name="new_tool", module="tools.y")])
    reasons2 = diff_reasons(old_cfg, new_tool_added)
    assert reasons2 and "tools changed" in reasons2[0]


def test_simulate_reports_not_simulatable_for_an_unsafe_diff(tmp_path):
    old_graph = ExecutionGraph(_config(), {})
    new_graph = ExecutionGraph(_config(model=ModelConfig(primary="openai/gpt-4o")), {})
    _seed_session(tmp_path, "s1", [{"role": "user", "content": "hi"}])

    report = asyncio.run(simulate(tmp_path, old_graph, new_graph))
    assert report.simulatable is False
    assert report.sessions_checked == 0
    assert any("model" in r for r in report.not_simulatable_reasons)


def test_simulate_flags_a_new_router_that_would_now_fire(tmp_path):
    old_graph = ExecutionGraph(_config(), {})
    new_cfg = _config(
        agents={
            "triage": AgentConfig(routers=[RouterConfig(condition="True", target="billing")]),
            "billing": AgentConfig(),
        }
    )
    new_graph = ExecutionGraph(new_cfg, {})

    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello, how can I help?"},
    ]
    _seed_session(tmp_path, "s1", messages)

    report = asyncio.run(simulate(tmp_path, old_graph, new_graph))
    assert report.simulatable is True
    assert report.sessions_checked == 1
    result = report.results[0]
    assert not result.unchanged
    kinds = {v.kind for v in result.verdicts}
    assert "ROUTING_DIVERGES" in kinds
    diverge = next(v for v in result.verdicts if v.kind == "ROUTING_DIVERGES")
    assert diverge.turn == 1
    assert "billing" in diverge.detail


def test_simulate_reports_unchanged_when_new_router_never_fires(tmp_path):
    old_graph = ExecutionGraph(_config(), {})
    new_cfg = _config(
        agents={
            "triage": AgentConfig(routers=[RouterConfig(condition="False", target="billing")]),
            "billing": AgentConfig(),
        }
    )
    new_graph = ExecutionGraph(new_cfg, {})

    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello, how can I help?"},
    ]
    _seed_session(tmp_path, "s1", messages)

    report = asyncio.run(simulate(tmp_path, old_graph, new_graph))
    assert report.results[0].unchanged


def test_simulate_flags_a_new_circuit_breaker_trip_on_real_handoff_history(tmp_path):
    old_cfg = _config(circuit_breakers=CircuitBreakersConfig(max_handoffs_per_session=25))
    new_cfg = _config(circuit_breakers=CircuitBreakersConfig(max_handoffs_per_session=1))
    old_graph = ExecutionGraph(old_cfg, {})
    new_graph = ExecutionGraph(new_cfg, {})

    messages = [
        {"role": "user", "content": "hi"},
        {"role": "system", "content": "Router: Transferred to billing via conditional router ('True')."},
        {"role": "user", "content": "again"},
        {"role": "system", "content": "Router: Transferred to triage via conditional router ('True')."},
    ]
    _seed_session(tmp_path, "s1", messages)

    report = asyncio.run(simulate(tmp_path, old_graph, new_graph))
    result = report.results[0]
    trip = next(v for v in result.verdicts if v.kind == "NEW_CIRCUIT_BREAKER_TRIP")
    assert "max_handoffs_per_session" in trip.detail


def test_simulate_flags_new_approval_gate_on_a_tool_that_was_actually_called(tmp_path):
    old_cfg = _config(tools=[LocalToolConfig(name="send_email", module="tools.mail")])
    new_cfg = _config(
        tools=[LocalToolConfig(name="send_email", module="tools.mail", requires_approval=True)]
    )
    old_graph = ExecutionGraph(old_cfg, {})
    new_graph = ExecutionGraph(new_cfg, {})

    messages = [
        {"role": "user", "content": "email the customer"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "send_email", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "name": "send_email", "content": "Email sent"},
    ]
    _seed_session(tmp_path, "s1", messages)

    report = asyncio.run(simulate(tmp_path, old_graph, new_graph))
    result = report.results[0]
    gate = next(v for v in result.verdicts if v.kind == "NEW_APPROVAL_GATE")
    assert "send_email" in gate.detail


def test_simulate_skips_sessions_with_no_messages(tmp_path):
    old_graph = ExecutionGraph(_config(), {})
    new_graph = ExecutionGraph(_config(), {})
    cp = SQLiteCheckpointer(str(tmp_path / ".ai" / "memory.db"))
    # Seeding via list_sessions requires a real row; save an empty-messages checkpoint directly.
    cp.save_checkpoint("empty1", [], {})
    _seed_session(tmp_path, "real1", [{"role": "user", "content": "hi"}])

    report = asyncio.run(simulate(tmp_path, old_graph, new_graph))
    assert report.sessions_checked == 1
    assert report.results[0].session_id == "real1"
