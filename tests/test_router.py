import asyncio
from pathlib import Path

from intagrin.compiler.parser import ExecutionGraph
from intagrin.config.schema import (
    AgentConfig,
    AppConfig,
    ConditionFunctionConfig,
    LocalToolConfig,
    MemoryConfig,
    ModelConfig,
    RootRouterConfig,
    RouterConfig,
    validate_config_dict,
)
from intagrin.runtime.engine import RuntimeEngine
from intagrin.runtime.router import SwarmRouter, safe_eval, validate_condition_syntax
from intagrin.tracing.console import EventStreamer


def test_evaluate_conditional_routers_stops_at_first_fired_and_records_every_evaluation():
    agent_cfg = AgentConfig(
        routers=[
            RouterConfig(condition="False", target="a"),
            RouterConfig(condition="True", target="b"),
            RouterConfig(condition="True", target="c"),  # never reached
        ]
    )
    fired, target, evaluations = SwarmRouter.evaluate_conditional_routers(agent_cfg, {})

    assert fired is True
    assert target == "b"
    # Only the first two routers were actually checked — the third is unreachable once one fires.
    assert [t for _, t, _e in evaluations] == [False, True]
    assert [r.target for r, _, _e in evaluations] == ["a", "b"]


def test_evaluate_conditional_routers_records_a_raising_condition_with_its_error_and_continues():
    """Regression test for a real observability gap: a router whose condition raises (most
    commonly a typo'd state-key name) used to vanish from `evaluations` entirely — indistinguishable
    from a router that was never declared at all, visible only via a separate log line. It must
    still be recorded (fired=None, error=<the exception>) so a broken condition stays visible in
    the same trace a fired/not-fired decision would show up in, while still being skipped (fails
    open, doesn't block later routers from being checked)."""
    agent_cfg = AgentConfig(
        routers=[
            RouterConfig(condition="undefined_var > 0", target="a"),
            RouterConfig(condition="True", target="b"),
        ]
    )
    fired, target, evaluations = SwarmRouter.evaluate_conditional_routers(agent_cfg, {})

    assert fired is True
    assert target == "b"
    assert [r.target for r, _f, _e in evaluations] == ["a", "b"]
    raising_fired, raising_error = evaluations[0][1], evaluations[0][2]
    assert raising_fired is None
    assert "undefined_var" in raising_error
    fired_flag, no_error = evaluations[1][1], evaluations[1][2]
    assert fired_flag is True
    assert no_error is None


def test_evaluate_conditional_routers_no_router_fires():
    agent_cfg = AgentConfig(routers=[RouterConfig(condition="False", target="a")])
    fired, target, evaluations = SwarmRouter.evaluate_conditional_routers(agent_cfg, {})
    assert fired is False
    assert target is None
    assert len(evaluations) == 1


def test_evaluate_root_router_no_config_returns_false_none_none():
    config = AppConfig(
        version="1.0", name="t", default_agent="a",
        model=ModelConfig(primary="mock/model"), memory=MemoryConfig(type="buffer"),
        agents={"a": AgentConfig()},
    )
    graph = ExecutionGraph(config, {})
    assert SwarmRouter.evaluate_root_router(graph, "a", {}) == (False, None, None)


def test_engine_logs_a_router_decision_event_for_every_conditional_router_checked():
    """Regression guard: _resolve_routing used to inline-evaluate each conditional router and log
    a router_decision event per router as it went. After delegating evaluation to
    SwarmRouter.evaluate_conditional_routers, that per-router tracing must still happen for every
    router actually checked, not just the one that fired."""

    async def run():
        config = AppConfig(
            version="1.0",
            name="router-trace-test",
            default_agent="triage",
            model=ModelConfig(primary="mock/model"),
            memory=MemoryConfig(type="buffer"),
            agents={
                "triage": AgentConfig(
                    routers=[
                        RouterConfig(condition="balance > 0", target="billing"),  # false
                        RouterConfig(condition="balance < 0", target="support"),  # true
                    ]
                ),
                "billing": AgentConfig(),
                "support": AgentConfig(),
            },
        )
        engine = RuntimeEngine(ExecutionGraph(config, {}), Path.cwd())
        await engine.initialize()
        engine.state["balance"] = -1

        q = EventStreamer.subscribe()
        try:
            route_err = await engine._resolve_routing(config.agents["triage"])
        finally:
            EventStreamer.unsubscribe(q)

        assert route_err is None
        assert engine.active_agent_name == "support"

        decisions = []
        while not q.empty():
            ev = q.get_nowait()
            if ev["type"] == "router_decision":
                decisions.append(ev["data"])

        assert len(decisions) == 2
        assert decisions[0]["fired"] is False
        assert decisions[0]["target"] == "billing"
        assert decisions[1]["fired"] is True
        assert decisions[1]["target"] == "support"

    asyncio.run(run())


def test_engine_router_decision_event_surfaces_a_broken_conditions_error():
    """Regression test: a typo'd router condition must show up in the SAME router_decision trace
    a normal fired/not-fired decision does, carrying its error — previously it was omitted from
    `evaluations` entirely (see test_evaluate_conditional_routers_records_a_raising_condition_
    with_its_error_and_continues), so _resolve_routing's per-router event loop never even ran for
    it, and a config typo was invisible to the Monitor dashboard's live trace / `inta simulate`,
    discoverable only by grepping raw logs."""

    async def run():
        config = AppConfig(
            version="1.0",
            name="router-error-trace-test",
            default_agent="triage",
            model=ModelConfig(primary="mock/model"),
            memory=MemoryConfig(type="buffer"),
            agents={
                "triage": AgentConfig(
                    routers=[
                        RouterConfig(condition="typo_balance < 0", target="support"),
                    ]
                ),
                "support": AgentConfig(),
            },
        )
        engine = RuntimeEngine(ExecutionGraph(config, {}), Path.cwd())
        await engine.initialize()

        q = EventStreamer.subscribe()
        try:
            route_err = await engine._resolve_routing(config.agents["triage"])
        finally:
            EventStreamer.unsubscribe(q)

        assert route_err is None
        assert engine.active_agent_name == "triage"  # never routed — the broken condition fails open

        decisions = []
        while not q.empty():
            ev = q.get_nowait()
            if ev["type"] == "router_decision":
                decisions.append(ev["data"])

        assert len(decisions) == 1
        assert decisions[0]["fired"] is False
        assert decisions[0]["error"] is not None
        assert "typo_balance" in decisions[0]["error"]

    asyncio.run(run())


def test_engine_root_router_error_message_format_is_preserved():
    """The wrapped 'Root router '<agent>' error: ...' format is depended on by anything
    surfacing router errors to the user/dashboard — must survive the evaluate_root_router
    delegation unchanged."""

    async def run():
        config = AppConfig(
            version="1.0",
            name="root-router-error-test",
            default_agent="triage",
            model=ModelConfig(primary="mock/model"),
            memory=MemoryConfig(type="buffer"),
            routers={
                "triage": RootRouterConfig(
                    module="tests._fixtures_missing_module_xyz", possible_targets=["x"]
                )
            },
            agents={"triage": AgentConfig()},
        )
        engine = RuntimeEngine(ExecutionGraph(config, {}), Path.cwd())
        await engine.initialize()

        err = await engine._resolve_routing(config.agents["triage"])
        assert err is not None
        assert err.startswith("Root router 'triage' error:")

    asyncio.run(run())


def test_validate_condition_syntax_accepts_everything_safe_eval_supports():
    valid_conditions = [
        "user_status == 'banned'",
        "balance < 0",
        "balance <= 0 and tier != 'gold'",
        "'refund' in intent",
        "not is_verified",
        "a < b or (c == d and not e)",
        "1 == 1",
    ]
    for cond in valid_conditions:
        assert validate_condition_syntax(cond) is None, cond
        # Every condition validate_condition_syntax accepts must actually be evaluable by
        # safe_eval given a state that defines every referenced name — the two must agree.
        names = [
            n
            for n in ("user_status", "balance", "tier", "intent", "is_verified", "a", "b", "c", "d", "e")
            if n in cond
        ]
        state = {n: "" for n in names} | {n: 0 for n in ("balance", "a", "b") if n in names} | {
            n: True for n in ("is_verified",) if n in names
        }
        safe_eval(cond, state)  # must not raise


def test_validate_condition_syntax_rejects_method_calls_and_attribute_access():
    reason = validate_condition_syntax("state.get('user_status', '') == 'banned'")
    assert reason is not None
    assert "state.get" in "state.get('user_status', '') == 'banned'"  # sanity: this is the real bug shape
    assert "Call" in reason or "not supported" in reason


def test_validate_condition_syntax_rejects_bad_python_syntax():
    reason = validate_condition_syntax("balance <")
    assert reason is not None
    assert "syntax" in reason.lower()


def test_validate_condition_syntax_rejects_unsupported_operators():
    # Power (**) isn't in safe_eval's supported comparison/boolean grammar at all.
    reason = validate_condition_syntax("balance is None")
    assert reason is not None

    reason2 = validate_condition_syntax("balance ** 2 == 4")
    assert reason2 is not None


def test_safe_eval_calls_a_registered_condition_function():
    functions = {"is_high_value": lambda total: total > 1000}
    assert safe_eval("is_high_value(order_total)", {"order_total": 5000}, functions) is True
    assert safe_eval("is_high_value(order_total)", {"order_total": 10}, functions) is False


def test_safe_eval_rejects_a_call_to_an_unregistered_function():
    try:
        safe_eval("is_high_value(order_total)", {"order_total": 5000}, {})
        assert False, "expected a ValueError for an unregistered condition function"
    except ValueError as e:
        assert "is_high_value" in str(e)


def test_safe_eval_rejects_attribute_based_calls_even_with_functions_registered():
    """A registered condition_functions whitelist must not become a backdoor for state.get(...)
    or any other attribute-based call — only bare, whitelisted names are ever callable."""
    functions = {"get": lambda *a: "banned"}
    try:
        safe_eval("state.get('user_status') == 'banned'", {}, functions)
        assert False, "expected a ValueError for an attribute-based call target"
    except ValueError as e:
        assert "call target" in str(e).lower()


def test_safe_eval_evaluates_condition_function_arguments_through_the_same_restricted_grammar():
    """Arguments to a condition function are themselves state-key names/literals/comparisons —
    not raw, unvalidated expressions — so nesting a call doesn't smuggle in unsupported syntax."""
    functions = {"combine": lambda a, b: a and b}
    assert safe_eval("combine(tier == 'gold', balance > 0)", {"tier": "gold", "balance": 5}, functions) is True


def test_validate_condition_syntax_accepts_a_declared_condition_function_call():
    reason = validate_condition_syntax("is_high_value(order_total)", {"is_high_value"})
    assert reason is None


def test_validate_condition_syntax_rejects_an_undeclared_condition_function_call():
    reason = validate_condition_syntax("is_high_value(order_total)", set())
    assert reason is not None
    assert "is_high_value" in reason


def test_evaluate_conditional_routers_threads_functions_through_to_safe_eval():
    agent_cfg = AgentConfig(
        routers=[RouterConfig(condition="is_vip(tier)", target="vip_agent")]
    )
    functions = {"is_vip": lambda tier: tier == "platinum"}

    fired, target, _evaluations = SwarmRouter.evaluate_conditional_routers(
        agent_cfg, {"tier": "platinum"}, functions
    )
    assert fired is True
    assert target == "vip_agent"

    fired, target, _evaluations = SwarmRouter.evaluate_conditional_routers(
        agent_cfg, {"tier": "bronze"}, functions
    )
    assert fired is False


def test_validate_config_dict_flags_an_undeclared_condition_function_in_a_router():
    data = {
        "version": "1.0",
        "name": "condfn-test",
        "default_agent": "assistant",
        "model": {"primary": "mock/model"},
        "memory": {"type": "buffer"},
        "agents": {
            "assistant": {
                "routers": [{"condition": "is_vip(tier)", "target": "other"}],
            },
            "other": {},
        },
    }
    config, errors = validate_config_dict(data)
    assert any("is_vip" in e for e in errors)

    data["condition_functions"] = [{"name": "is_vip", "module": "tools.condition_functions"}]
    config, errors = validate_config_dict(data)
    assert errors == []
    assert config is not None


def test_engine_loads_and_evaluates_a_declared_condition_function_end_to_end():
    """condition_functions is loaded by RuntimeEngine.initialize() and actually reaches routing
    decisions and available_when gates — not just the pure router.py helpers exercised above."""
    import sys
    import types

    mod = types.ModuleType("_test_condition_functions_mod")

    def is_vip(tier: str) -> bool:
        return tier == "platinum"

    mod.is_vip = is_vip
    sys.modules["_test_condition_functions_mod"] = mod

    config = AppConfig(
        version="1.0",
        name="condfn-engine-test",
        default_agent="assistant",
        model=ModelConfig(primary="mock/model"),
        memory=MemoryConfig(type="buffer"),
        condition_functions=[
            ConditionFunctionConfig(name="is_vip", module="_test_condition_functions_mod")
        ],
        agents={
            "assistant": AgentConfig(
                routers=[RouterConfig(condition="is_vip(tier)", target="vip_desk")],
                tools=[
                    LocalToolConfig(
                        name="escalate", module="unused", available_when="is_vip(tier)"
                    )
                ],
            ),
            "vip_desk": AgentConfig(),
        },
    )
    graph = ExecutionGraph(config, {})

    async def _run():
        engine = RuntimeEngine(graph=graph, project_dir=Path.cwd(), session_id="condfn-1")
        await engine.initialize()
        assert engine._condition_functions["is_vip"] is is_vip

        engine.active_agent_name = "assistant"
        engine.state["tier"] = "bronze"
        assert engine._tool_currently_available("escalate", config.agents["assistant"]) is False

        engine.state["tier"] = "platinum"
        assert engine._tool_currently_available("escalate", config.agents["assistant"]) is True

        route_err = await engine._resolve_routing(config.agents["assistant"])
        assert route_err is None
        assert engine.active_agent_name == "vip_desk"

    asyncio.run(_run())


def test_resolve_routing_records_a_router_trace_entry_for_a_non_firing_router():
    """Regression test for a real gap: a router that's evaluated but doesn't fire (or raises)
    previously only reached a live SSE event — invisible the instant nobody was watching the
    Monitor dashboard. state["_router_trace"] must carry this forward into the checkpoint."""
    config = AppConfig(
        version="1.0",
        name="router-trace-persist-test",
        default_agent="triage",
        model=ModelConfig(primary="mock/model"),
        memory=MemoryConfig(type="buffer"),
        agents={
            "triage": AgentConfig(
                routers=[RouterConfig(condition="balance > 0", target="billing")]
            ),
            "billing": AgentConfig(),
        },
    )
    graph = ExecutionGraph(config, {})

    async def _run():
        engine = RuntimeEngine(graph=graph, project_dir=Path.cwd(), session_id="trace-1")
        await engine.initialize()
        engine.state["balance"] = -5
        engine.messages.append({"role": "user", "content": "hi"})

        route_err = await engine._resolve_routing(config.agents["triage"])
        assert route_err is None
        assert engine.active_agent_name == "triage"  # did not fire

        trace = engine.state["_router_trace"]
        assert len(trace) == 1
        assert trace[0]["kind"] == "conditional"
        assert trace[0]["fired"] is False
        assert trace[0]["target"] == "billing"
        assert trace[0]["error"] is None
        assert trace[0]["turn"] == 1  # len(self.messages) at evaluation time

    asyncio.run(_run())


def test_resolve_routing_records_a_router_trace_entry_with_its_error():
    config = AppConfig(
        version="1.0",
        name="router-trace-error-test",
        default_agent="triage",
        model=ModelConfig(primary="mock/model"),
        memory=MemoryConfig(type="buffer"),
        agents={
            "triage": AgentConfig(
                routers=[RouterConfig(condition="typo_balance > 0", target="billing")]
            ),
            "billing": AgentConfig(),
        },
    )
    graph = ExecutionGraph(config, {})

    async def _run():
        engine = RuntimeEngine(graph=graph, project_dir=Path.cwd(), session_id="trace-2")
        await engine.initialize()

        await engine._resolve_routing(config.agents["triage"])

        trace = engine.state["_router_trace"]
        assert len(trace) == 1
        assert trace[0]["fired"] is False
        assert "typo_balance" in trace[0]["error"]

    asyncio.run(_run())


def test_router_trace_is_bounded_to_the_last_50_entries():
    config = AppConfig(
        version="1.0",
        name="router-trace-bound-test",
        default_agent="triage",
        model=ModelConfig(primary="mock/model"),
        memory=MemoryConfig(type="buffer"),
        agents={
            "triage": AgentConfig(
                routers=[RouterConfig(condition="False", target="billing")]
            ),
            "billing": AgentConfig(),
        },
    )
    graph = ExecutionGraph(config, {})

    async def _run():
        engine = RuntimeEngine(graph=graph, project_dir=Path.cwd(), session_id="trace-3")
        await engine.initialize()

        for _ in range(60):
            await engine._resolve_routing(config.agents["triage"])

        assert len(engine.state["_router_trace"]) == 50

    asyncio.run(_run())
