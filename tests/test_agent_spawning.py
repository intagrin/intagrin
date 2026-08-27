import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from intagrin.compiler.parser import ExecutionGraph
from intagrin.config.schema import (
    AgentConfig,
    AgentSpawningConfig,
    AppConfig,
    LocalToolConfig,
    MemoryConfig,
    ModelConfig,
    StateWriteAction,
)
from intagrin.errors import IntaGrinError
from intagrin.runtime.engine import RuntimeEngine


def _mock_graph(
    *,
    max_creations_per_session=3,
    requires_approval_on_first_action=False,
    tool_pool=None,
    allow_recursive_spawning=False,
    max_spawn_depth=1,
    model_pool=None,
    memory_type="buffer",
    result_schema=None,
    on_complete=None,
    max_delegation_turns=None,
):
    config = AppConfig(
        version="1.0",
        name="test-swarm",
        default_agent="orchestrator",
        model=ModelConfig(primary="mock/model"),
        memory=MemoryConfig(type=memory_type),
        agents={
            "orchestrator": AgentConfig(
                description="Orchestrator",
                tools=[
                    LocalToolConfig(name="search", module="unused"),
                    LocalToolConfig(name="summarize", module="unused"),
                    LocalToolConfig(name="issue_refund", module="unused", requires_approval=True),
                ],
                spawns=AgentSpawningConfig(
                    tool_pool=tool_pool or ["search", "summarize"],
                    max_creations_per_session=max_creations_per_session,
                    requires_approval_on_first_action=requires_approval_on_first_action,
                    allow_recursive_spawning=allow_recursive_spawning,
                    max_spawn_depth=max_spawn_depth,
                    model_pool=model_pool,
                    result_schema=result_schema,
                    on_complete=on_complete or [],
                ),
            ),
        },
    )
    if max_delegation_turns is not None:
        config.circuit_breakers.max_delegation_turns = max_delegation_turns
    return ExecutionGraph(config, {})


def _inject_tools(engine, names):
    for name in names:
        engine.local_tools[name] = lambda **kwargs: f"{kwargs}"
        engine.global_tool_schemas.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": "test tool",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        )


async def _init_engine(tmp_path, graph, session_id="s1"):
    engine = RuntimeEngine(graph=graph, project_dir=tmp_path, session_id=session_id)
    await engine.initialize()
    engine.active_agent_name = "orchestrator"
    _inject_tools(engine, ["search", "summarize", "issue_refund"])
    return engine


def _register_dynamic_agent(engine, name, tools, *, created_by="orchestrator", **extra):
    """Directly seeds the state a real spawn_agent call would have produced, without running an
    isolated child engine (and therefore without needing to mock an LLM call) — for tests that
    only care about tool-authorization/approval-gate behavior once an agent is dynamic, not about
    spawn_agent's own child-engine execution (covered separately below)."""
    dynamic_agents = engine.state.setdefault("_dynamic_agents", {})
    dynamic_agents[name] = {
        "role": "Specialist",
        "instruction": "test",
        "tools": tools,
        "model": "mock/model",
        "created_by": created_by,
        "depth": 0,
        "pending_first_action_approval": False,
        "allow_recursive_spawning": False,
        "max_spawn_depth": 1,
        **extra,
    }
    return name


def _mock_llm_response(*, content=None, tool_calls=None):
    message = MagicMock(content=content, tool_calls=tool_calls)
    dumped = {"role": "assistant"}
    if content is not None:
        dumped["content"] = content
    if tool_calls is not None:
        dumped["tool_calls"] = [{"id": tc.id} for tc in tool_calls]
    message.model_dump.return_value = dumped
    return MagicMock(
        choices=[MagicMock(message=message)],
        usage=MagicMock(prompt_tokens=5, completion_tokens=5, total_tokens=10),
    )


def test_spawn_agent_runs_the_child_to_completion_and_returns_a_result_without_transferring(
    tmp_path,
):
    """spawn_agent no longer transfers the parent's own control (that used to race when the LLM
    spawned multiple sub-agents in one turn — every spawn_agent call mutated the same shared
    active_agent_name/is_transferring concurrently via asyncio.gather, so only one spawn ever
    reliably "won"). It now runs the dynamic agent in an isolated child engine to completion and
    returns its result as an ordinary tool result — orchestrator stays active throughout, so
    multiple spawns in one turn can run concurrently with no shared mutable state to race on."""

    async def _run():
        engine = await _init_engine(tmp_path, _mock_graph())
        child_final = _mock_llm_response(content="Found X.")

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
            mock_acompletion.return_value = child_final
            result = await engine.execute_tool(
                "spawn_agent",
                {"role": "Search specialist", "instruction": "Find X", "tools": ["search"]},
                interactive=False,
            )

        assert "completed" in result
        assert "Found X." in result
        # The parent is never transferred — this is what fixes the concurrent-spawn race.
        assert engine.active_agent_name == "orchestrator"
        assert engine.is_transferring is False

        dynamic_names = list(engine.state["_dynamic_agents"].keys())
        assert len(dynamic_names) == 1
        assert dynamic_names[0].startswith("orchestrator_dyn_")
        dynamic = engine.state["_dynamic_agents"][dynamic_names[0]]
        assert dynamic["tools"] == ["search"]
        assert dynamic["created_by"] == "orchestrator"

    asyncio.run(_run())


def test_spawned_agent_can_use_its_granted_tool_and_return_to_creator(tmp_path):
    async def _run():
        engine = await _init_engine(tmp_path, _mock_graph())
        dyn_name = _register_dynamic_agent(engine, "orchestrator_dyn_test1", ["search"])
        engine.active_agent_name = dyn_name

        # Granted tool works.
        result = await engine.execute_tool("search", {"query": "x"}, interactive=False)
        assert "query" in result

        # return_to_creator only exists/works for a dynamic agent.
        assert engine._is_tool_allowed_for_active_agent("return_to_creator")
        result = await engine.execute_tool(
            "return_to_creator", {"summary": "done"}, interactive=False
        )
        assert "returned control to orchestrator" in result
        assert engine.active_agent_name == "orchestrator"
        assert not engine._is_tool_allowed_for_active_agent("return_to_creator")

        # A static agent never had access to a tool named after the dynamic agent.
        assert not engine._is_tool_allowed_for_active_agent(dyn_name)

    asyncio.run(_run())


def test_spawned_agent_cannot_use_a_tool_outside_its_granted_subset(tmp_path):
    """Even though 'summarize' is in orchestrator's tool_pool and orchestrator's own tools, the
    spawned agent wasn't granted it — _is_tool_allowed_for_active_agent must still block it."""

    async def _run():
        engine = await _init_engine(tmp_path, _mock_graph())
        dyn_name = _register_dynamic_agent(engine, "orchestrator_dyn_test2", ["search"])
        engine.active_agent_name = dyn_name

        assert not engine._is_tool_allowed_for_active_agent("summarize")
        result = await engine.execute_tool("summarize", {}, interactive=False)
        assert "not authorized" in result

    asyncio.run(_run())


def test_spawn_agent_rejects_a_tool_outside_the_declared_tool_pool(tmp_path):
    """Defense in depth: even if something upstream let a request for an out-of-pool tool
    through to execute_tool (the schema enum is not trusted alone), the server-side re-check
    must reject it — 'issue_refund' is one of orchestrator's own tools but NOT in spawns.tool_pool."""

    async def _run():
        engine = await _init_engine(tmp_path, _mock_graph(tool_pool=["search"]))
        result = await engine.execute_tool(
            "spawn_agent",
            {"role": "x", "instruction": "y", "tools": ["search", "issue_refund"]},
            interactive=False,
        )
        assert "rejected" in result
        assert "issue_refund" in result
        assert not engine.is_transferring
        assert engine.state.get("_dynamic_agents", {}) == {}

    asyncio.run(_run())


def test_max_creations_per_session_trips(tmp_path):
    async def _run():
        engine = await _init_engine(tmp_path, _mock_graph(max_creations_per_session=2))

        for _ in range(2):
            engine.active_agent_name = "orchestrator"
            await engine.execute_tool(
                "spawn_agent",
                {"role": "x", "instruction": "y", "tools": ["search"]},
                interactive=False,
            )

        engine.active_agent_name = "orchestrator"
        with pytest.raises(IntaGrinError) as exc_info:
            await engine.execute_tool(
                "spawn_agent",
                {"role": "x", "instruction": "y", "tools": ["search"]},
                interactive=False,
            )
        assert exc_info.value.code == "IG-RT-007"
        assert "dynamic agent creations" in str(exc_info.value).lower()

    asyncio.run(_run())


def test_circuit_breaker_trip_mid_turn_leaves_no_orphaned_tool_call(tmp_path):
    """Regression test for a real production incident: a spawn_agent call that trips
    max_creations_per_session mid-turn used to leave the assistant's tool_calls message
    permanently unanswered in the checkpointed history — Gemini/Vertex (and other strict
    providers) reject every future completion built from a history containing an unpaired
    function call, so the bug didn't just fail one turn, it permanently poisoned the session.
    _run_agent_turn's `except IntaGrinError` handler must give every pending tool_call_id its own
    `role: tool` response before the turn aborts, not just a free-floating assistant message."""

    async def _run():
        engine = await _init_engine(tmp_path, _mock_graph(max_creations_per_session=1))
        # Pre-load the counter at the cap so the very first spawn_agent call this turn trips
        # IG-RT-007 immediately, without needing a real child engine run first.
        engine.state.setdefault("_circuit_breakers", {})["dynamic_agents_created"] = 1
        engine.active_agent_name = "orchestrator"
        engine.messages.append({"role": "user", "content": "please spawn a helper"})

        tool_call = MagicMock()
        tool_call.id = "call_orphan_check"
        tool_call.function.name = "spawn_agent"
        tool_call.function.arguments = (
            '{"role": "x", "instruction": "y", "tools": ["search"]}'
        )
        spawn_response = _mock_llm_response(tool_calls=[tool_call])

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
            mock_acompletion.side_effect = [spawn_response]
            await engine._run_agent_turn(interactive=False)

        # The assistant's tool_calls message (from spawn_response) must be immediately followed
        # by a role:"tool" response carrying the exact same tool_call_id — not just an assistant
        # message with no paired response, which is the shape that poisons the checkpoint.
        assistant_idx = next(
            i
            for i, m in enumerate(engine.messages)
            if m.get("role") == "assistant" and m.get("tool_calls")
        )
        next_msg = engine.messages[assistant_idx + 1]
        assert next_msg.get("role") == "tool", (
            f"expected a role:'tool' response immediately after the tool_calls message, got "
            f"{next_msg.get('role')!r} — this is exactly the shape that breaks every future "
            f"turn on strict providers (Gemini/Vertex)"
        )
        assert next_msg.get("tool_call_id") == "call_orphan_check"
        assert "IG-RT-007" in next_msg.get("content", "")

    asyncio.run(_run())


def test_spawn_agent_forewarns_when_it_consumes_the_last_creation_slot(tmp_path):
    """The circuit breaker previously had no visible signal before it hard-failed the *next*
    spawn attempt with IG-RT-007 — a session could run out of budget mid-task with zero
    forewarning. The spawn that uses up the last slot must say so in its own result message, not
    leave the LLM to only discover the cap once it's already too late to plan around."""

    async def _run():
        engine = await _init_engine(tmp_path, _mock_graph(max_creations_per_session=2))
        engine.active_agent_name = "orchestrator"

        first_result = await engine.execute_tool(
            "spawn_agent",
            {"role": "x", "instruction": "y", "tools": ["search"]},
            interactive=False,
        )
        assert "last available specialist-creation slot" not in first_result

        engine.active_agent_name = "orchestrator"
        second_result = await engine.execute_tool(
            "spawn_agent",
            {"role": "y", "instruction": "z", "tools": ["search"]},
            interactive=False,
        )
        assert "last available specialist-creation slot" in second_result
        assert "2/2 used" in second_result

    asyncio.run(_run())


def test_requires_approval_on_first_action_pauses_and_resume_completes_it(tmp_path):
    async def _run():
        engine = await _init_engine(
            tmp_path, _mock_graph(requires_approval_on_first_action=True)
        )
        dyn_name = _register_dynamic_agent(
            engine,
            "orchestrator_dyn_test3",
            ["search"],
            pending_first_action_approval=True,
        )
        engine.active_agent_name = dyn_name

        # First tool call by the dynamic agent must pause, not execute.
        result = await engine.execute_tool(
            "search", {"query": "x"}, interactive=False, tool_call_id="c1"
        )
        assert "_pending_approval" in engine.state
        assert engine.state["_pending_approval"]["tool"] == "search"
        assert "paused" in result.lower()

        # Simulate /resume's approval flow: grant the one-time exemption, keyed by the same
        # tool_call_id the pause carried (not the tool name — see IG's approval-scoping fix), and
        # re-call with that same id.
        engine.state.setdefault("_approved_tool_calls", []).append("c1")
        result = await engine.execute_tool(
            "search", {"query": "x"}, interactive=False, tool_call_id="c1"
        )
        assert "query" in result
        assert engine.state["_dynamic_agents"][dyn_name]["pending_first_action_approval"] is False

        # A second tool call (different id, no exemption of its own) must not pause again either
        # — but that's because the first-action gate itself is one-time and already consumed
        # above, not because it's riding the c1 exemption (which _approved_tool_calls no longer
        # has after being consumed just now).
        result = await engine.execute_tool(
            "search", {"query": "y"}, interactive=False, tool_call_id="c2"
        )
        assert "query" in result

    asyncio.run(_run())


def test_dynamic_agent_still_respects_a_pre_existing_requires_approval_gate_on_its_tool(tmp_path):
    """Two independent gates: even with requires_approval_on_first_action off, a granted tool
    that's itself statically requires_approval-gated must still pause — defense in depth. Uses a
    real importable module (unlike the other tests' inject-after-initialize shortcut) so
    _load_tool_config actually populates tools_requiring_approval from the real requires_approval
    flag, not a test double."""
    tools_dir = tmp_path / "tools"
    tools_dir.mkdir()
    (tools_dir / "__init__.py").write_text("")
    (tools_dir / "custom.py").write_text(
        "def issue_refund(amount: float) -> str:\n    return f'refunded {amount}'\n"
    )
    import sys

    sys.path.insert(0, str(tmp_path))
    try:

        async def _run():
            config = AppConfig(
                version="1.0",
                name="test-swarm",
                default_agent="orchestrator",
                model=ModelConfig(primary="mock/model"),
                memory=MemoryConfig(type="buffer"),
                agents={
                    "orchestrator": AgentConfig(
                        description="Orchestrator",
                        tools=[
                            LocalToolConfig(
                                name="issue_refund",
                                module="tools.custom",
                                requires_approval=True,
                            ),
                        ],
                        spawns=AgentSpawningConfig(tool_pool=["issue_refund"]),
                    ),
                },
            )
            graph = ExecutionGraph(config, {})
            engine = RuntimeEngine(graph=graph, project_dir=tmp_path, session_id="s3")
            await engine.initialize()
            engine.active_agent_name = "orchestrator"
            assert "issue_refund" in engine.tools_requiring_approval

            await engine.execute_tool(
                "spawn_agent",
                {"role": "refund bot", "instruction": "y", "tools": ["issue_refund"]},
                interactive=False,
            )
            result = await engine.execute_tool(
                "issue_refund", {"amount": 10}, interactive=False
            )
            assert "paused" in result.lower()
            assert "_pending_approval" in engine.state

        asyncio.run(_run())
    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.pop("tools.custom", None)
        sys.modules.pop("tools", None)


def test_run_agent_turn_end_to_end_spawns_via_a_real_mocked_llm_tool_call(tmp_path):
    """Proves the schema/dispatch wiring end to end through _run_agent_turn, not just direct
    execute_tool calls — one turn, one child engine run inline, orchestrator resumes and finishes
    on its own without ever handing off control."""

    async def _run():
        graph = _mock_graph()
        engine = RuntimeEngine(graph=graph, project_dir=tmp_path, session_id="s2")
        await engine.initialize()
        engine.active_agent_name = "orchestrator"
        _inject_tools(engine, ["search", "summarize"])
        engine.messages.append({"role": "user", "content": "please spawn a helper"})

        tool_call = MagicMock()
        tool_call.id = "call_1"
        tool_call.function.name = "spawn_agent"
        tool_call.function.arguments = (
            '{"role": "Search specialist", "instruction": "find x", "tools": ["search"]}'
        )
        spawn_response = _mock_llm_response(tool_calls=[tool_call])
        # The isolated child engine's own single turn, run synchronously inside spawn_agent's
        # execute_tool call — this is what proves resolving the dynamic agent's config (model,
        # prompt, tools) doesn't crash, exactly the path that would raise AttributeError on a bare
        # `self.graph.config.agents.get(...)` lookup, since the dynamic agent's name was never in
        # that static dict.
        child_response = _mock_llm_response(content="I found x.")
        # orchestrator's own follow-up turn after seeing spawn_agent's tool result.
        final_response = _mock_llm_response(content="Done — spawned a helper.")

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
            mock_acompletion.side_effect = [spawn_response, child_response, final_response]
            await engine._run_agent_turn(interactive=False)

        # spawn_agent never transfers the parent's own control.
        assert engine.is_transferring is False
        assert engine.active_agent_name == "orchestrator"

        dynamic_names = list(engine.state.get("_dynamic_agents", {}).keys())
        assert len(dynamic_names) == 1
        assert dynamic_names[0].startswith("orchestrator_dyn_")

        tool_result = next(
            m
            for m in engine.messages
            if m.get("role") == "tool" and m.get("name") == "spawn_agent"
        )
        assert "I found x." in tool_result["content"]
        assert any(
            m.get("content") == "Done — spawned a helper."
            for m in engine.messages
            if m.get("role") == "assistant"
        )

    asyncio.run(_run())


def test_write_state_cannot_hijack_active_agent_name(tmp_path):
    """Regression test: making active_agent_name a property backed by
    self.state["_active_agent_name"] (so a handoff survives a checkpoint reload) means write_state
    — which has no key restriction and is unconditionally available to every agent, dynamic ones
    included — could otherwise directly overwrite it and transfer control to any agent in the
    graph, completely bypassing transfer_agent's/delegate_task's own target-authorization checks.
    Any leading-`_` key must be rejected, not just this one, since the same gap would apply to
    _dynamic_agents, _circuit_breakers, _pending_approval, etc."""

    async def _run():
        engine = await _init_engine(tmp_path, _mock_graph())
        assert engine.active_agent_name == "orchestrator"

        result = engine.write_state("_active_agent_name", "some_other_agent")

        assert "rejected" in result.lower()
        assert engine.active_agent_name == "orchestrator"
        assert engine.state["_active_agent_name"] == "orchestrator"

    asyncio.run(_run())


def test_spawn_agent_persists_a_child_pause_instead_of_discarding_it(tmp_path):
    """Regression test for the real fix: when the spawned child hits a human-approval gate mid-
    task (here, requires_approval_on_first_action), spawn_agent must not discard the child and
    report a false "completed... No response" — it must checkpoint the child under its own
    session_id (so /resume can reload and continue it later) and surface the pause on the PARENT
    too, so _run_agent_turn's existing `if "_pending_approval" in self.state: break` halts this
    turn instead of letting the orchestrator wrongly believe the sub-task finished."""

    async def _run():
        engine = await _init_engine(
            tmp_path,
            _mock_graph(
                requires_approval_on_first_action=True,
                tool_pool=["search"],
                memory_type="sqlite",
            ),
        )

        search_call = MagicMock()
        search_call.function.name = "search"
        search_call.function.arguments = '{"query": "flights"}'
        search_call.id = "child_tool_call_1"
        child_response = _mock_llm_response(tool_calls=[search_call])

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
            mock_acompletion.return_value = child_response
            result = await engine.execute_tool(
                "spawn_agent",
                {"role": "Search specialist", "instruction": "Find X", "tools": ["search"]},
                interactive=False,
                tool_call_id="spawn_call_1",
            )

        assert "paused" in result.lower()
        # The parent is not discarded/finished — it must actually be paused, not "completed".
        assert engine.active_agent_name == "orchestrator"

        pending = engine.state["_pending_approval"]
        assert pending["tool"] == "search"
        assert pending["parent_tool_call_id"] == "spawn_call_1"
        dyn_name = pending["agent"]
        assert dyn_name.startswith("orchestrator_dyn_")
        child_session_id = pending["child_session_id"]
        assert child_session_id == f"{engine.session_id}_spawn_{dyn_name}"

        # _save_checkpoint fires the actual DB write in a background thread (fire-and-forget, by
        # design — see its docstring) rather than awaiting it — poll for it to land (a fixed
        # sleep flaked under full-suite load) before asserting the pause is reloadable, not just
        # present on the in-memory child object that execute_tool is about to let go of.
        reloaded_child = None
        for _ in range(50):
            await asyncio.sleep(0.1)
            reloaded_child = RuntimeEngine(
                graph=engine.graph, project_dir=tmp_path, session_id=child_session_id
            )
            await reloaded_child.initialize()
            if "_pending_approval" in reloaded_child.state:
                break
        assert "_pending_approval" in reloaded_child.state
        assert reloaded_child.state["_pending_approval"]["tool"] == "search"
        assert reloaded_child.active_agent_name == dyn_name

    asyncio.run(_run())


def test_resolve_agent_cfg_propagates_requires_approval_on_first_action_to_a_recursive_grandchild(
    tmp_path,
):
    """Regression test for a real bug: _resolve_agent_cfg synthesizes a recursively-spawning
    dynamic agent's own AgentSpawningConfig by reading dynamic.get("requires_approval_on_first_
    action", True) — but the key actually stored per dynamic agent (see _register_dynamic_agent
    above and spawn_agent's own state write) is "pending_first_action_approval". Since the read
    key was never written, this always silently fell back to True regardless of what
    spawns.requires_approval_on_first_action was actually configured to — a recursively-spawned
    grandchild agent would pause for human approval on its first tool call even when the user's
    ai.yaml explicitly set requires_approval_on_first_action: false."""

    async def _run():
        engine = await _init_engine(tmp_path, _mock_graph(tool_pool=["search"]))
        _register_dynamic_agent(
            engine,
            "orchestrator_dyn_a",
            tools=["search"],
            # The user explicitly turned this off for the dynamic agent's own further spawning.
            allow_recursive_spawning=True,
            max_spawn_depth=2,
            pending_first_action_approval=False,
        )

        synthesized = engine._resolve_agent_cfg("orchestrator_dyn_a")

        assert synthesized.spawns is not None
        assert synthesized.spawns.requires_approval_on_first_action is False

    asyncio.run(_run())


def test_only_the_first_of_several_concurrent_first_action_tool_calls_pauses(tmp_path):
    """Regression test: requires_approval_on_first_action gates the dynamic agent's very first
    tool call, but a first turn can legitimately request several tools at once
    (_execute_tool_calls_with_healing runs them concurrently). Before this fix,
    pending_first_action_approval was only cleared once an approval was actually granted — never
    when a call merely *tripped* the gate — so every concurrently-run tool call independently read
    it as still-True, each called _pause_for_human, and the single self.state["_pending_approval"]
    slot ended up holding whichever call happened to write last, silently losing the others. Only
    the first tool call in the batch should pause; the rest should execute normally."""

    async def _run():
        engine = await _init_engine(
            tmp_path,
            _mock_graph(requires_approval_on_first_action=True, tool_pool=["search", "summarize"]),
        )

        search_call = MagicMock()
        search_call.function.name = "search"
        search_call.function.arguments = "{}"
        search_call.id = "t1"
        summarize_call = MagicMock()
        summarize_call.function.name = "summarize"
        summarize_call.function.arguments = "{}"
        summarize_call.id = "t2"
        child_response = _mock_llm_response(tool_calls=[search_call, summarize_call])

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
            mock_acompletion.return_value = child_response
            result = await engine.execute_tool(
                "spawn_agent",
                {"role": "x", "instruction": "y", "tools": ["search", "summarize"]},
                interactive=False,
                tool_call_id="spawn_call_1",
            )

        assert "paused" in result.lower()
        # Only one of the two tool calls is reflected in the surviving pending approval —
        # deterministically the first one in the batch, not whichever happened to run last.
        assert engine.state["_pending_approval"]["tool"] == "search"

    asyncio.run(_run())


def test_two_concurrently_spawned_children_that_both_pause_do_not_lose_either_pause(tmp_path):
    """Regression test for _set_pending_approval's queueing fix. Two spawn_agent calls in the
    same turn (_execute_tool_calls_with_healing runs non-transfer tool calls, spawn_agent
    included, concurrently) whose isolated children BOTH pause used to race on the single
    self.state["_pending_approval"] slot — last write wins, and the other pause was gone for
    good, with no way to ever resume it (this is the "3 parallel city specialists" scenario from
    the live TravelPlanner run). Both must now be recoverable: one is surfaced as
    _pending_approval, the other queued in _pending_approval_queue and promoted once the first is
    resolved via /resume."""

    async def _run():
        engine = await _init_engine(
            tmp_path,
            _mock_graph(
                requires_approval_on_first_action=True,
                tool_pool=["search"],
                max_creations_per_session=5,
            ),
        )

        spawn_call_1 = MagicMock()
        spawn_call_1.function.name = "spawn_agent"
        spawn_call_1.function.arguments = (
            '{"role": "Paris specialist", "instruction": "plan paris", "tools": ["search"]}'
        )
        spawn_call_1.id = "spawn_1"
        spawn_call_2 = MagicMock()
        spawn_call_2.function.name = "spawn_agent"
        spawn_call_2.function.arguments = (
            '{"role": "Tokyo specialist", "instruction": "plan tokyo", "tools": ["search"]}'
        )
        spawn_call_2.id = "spawn_2"

        search_call = MagicMock()
        search_call.function.name = "search"
        search_call.function.arguments = "{}"
        search_call.id = "child_search"
        child_response = _mock_llm_response(tool_calls=[search_call])

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
            mock_acompletion.return_value = child_response
            results = await engine._execute_tool_calls_with_healing(
                [spawn_call_1, spawn_call_2], interactive=False
            )

        assert all("paused" in r["content"].lower() for r in results)

        # Both pauses survive — one surfaced, one queued — matching the two agents actually
        # created, not silently collapsed down to just one.
        assert "_pending_approval" in engine.state
        queue = engine.state.get("_pending_approval_queue", [])
        assert len(queue) == 1

        surfaced_agent = engine.state["_pending_approval"]["agent"]
        queued_agent = queue[0]["agent"]
        assert {surfaced_agent, queued_agent} == set(engine.state["_dynamic_agents"].keys())

        # Resolving the surfaced one promotes the queued one instead of leaving it forgotten.
        engine.state.pop("_pending_approval")
        promoted = engine._promote_next_queued_approval()
        assert promoted is True
        assert engine.state["_pending_approval"]["agent"] == queued_agent
        assert "_pending_approval_queue" not in engine.state  # cleaned up once drained

    asyncio.run(_run())


def test_spawn_agent_captures_the_return_to_creators_summary_not_no_response(tmp_path):
    """Regression test for a real bug found while designing spawns.result_schema: a child that
    exits via return_to_creator — the clean, framework-suggested way to signal a dynamic agent's
    task is done — used to be reported to the orchestrator as "No response from sub-agent." even
    when it explicitly summarized what it did. return_to_creator's result lands in a role="tool"
    message, not a role="assistant" one, so the old assistant-content-only scan silently missed
    it. This is a plausible real driver of the wasted-spawn-budget confusion seen in the live
    TravelPlanner session — the orchestrator, seeing "No response," had no way to know a
    destination was already handled and could reasonably decide to spawn it again."""

    async def _run():
        engine = await _init_engine(tmp_path, _mock_graph())

        return_call = MagicMock()
        return_call.function.name = "return_to_creator"
        return_call.function.arguments = (
            '{"summary": "Created a 7-day itinerary for Paris and booked the hotel."}'
        )
        return_call.id = "return_call_1"
        child_response = _mock_llm_response(tool_calls=[return_call])

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
            mock_acompletion.return_value = child_response
            result = await engine.execute_tool(
                "spawn_agent",
                {"role": "Paris specialist", "instruction": "plan paris", "tools": ["search"]},
                interactive=False,
                tool_call_id="spawn_call_1",
            )

        assert "No response from sub-agent" not in result
        assert "Created a 7-day itinerary for Paris and booked the hotel." in result

    asyncio.run(_run())


def test_result_schema_derives_return_to_creators_tool_schema_and_validates_the_result(tmp_path):
    """spawns.result_schema replaces return_to_creator's generic free-text `summary` field with
    a tool-call schema derived directly from the Pydantic model — steering the model via
    constrained tool-call decoding, not just prompt instructions — and the structured, validated
    result flows back as spawn_agent's own tool result without any write_state/read_state
    choreography needed."""
    (tmp_path / "itinerary_result.py").write_text(
        "from pydantic import BaseModel\n\n"
        "class ItineraryResult(BaseModel):\n"
        "    destination: str\n"
        "    days: int\n"
    )

    async def _run():
        engine = await _init_engine(
            tmp_path, _mock_graph(result_schema="itinerary_result.ItineraryResult")
        )

        return_call = MagicMock()
        return_call.function.name = "return_to_creator"
        return_call.function.arguments = '{"destination": "Paris", "days": 7}'
        return_call.id = "return_call_1"
        child_response = _mock_llm_response(tool_calls=[return_call])

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
            mock_acompletion.return_value = child_response
            result = await engine.execute_tool(
                "spawn_agent",
                {"role": "Paris specialist", "instruction": "plan paris", "tools": ["search"]},
                interactive=False,
                tool_call_id="spawn_call_1",
            )

        sent_tools = mock_acompletion.call_args.kwargs["tools"]
        return_to_creator_schema = next(
            t for t in sent_tools if t["function"]["name"] == "return_to_creator"
        )
        assert set(return_to_creator_schema["function"]["parameters"]["properties"]) == {
            "destination",
            "days",
        }

        assert '"destination": "Paris"' in result
        assert '"days": 7' in result

    asyncio.run(_run())


def test_result_schema_self_heals_a_malformed_return_to_creators_call(tmp_path):
    """A return_to_creator call missing a required result_schema field must go through the same
    corrector-model self-heal already used for malformed tool arguments elsewhere, not crash or
    silently accept invalid data."""
    (tmp_path / "itinerary_result_heal.py").write_text(
        "from pydantic import BaseModel\n\n"
        "class ItineraryResult(BaseModel):\n"
        "    destination: str\n"
        "    days: int\n"
    )

    async def _run():
        engine = await _init_engine(
            tmp_path, _mock_graph(result_schema="itinerary_result_heal.ItineraryResult")
        )

        bad_call = MagicMock()
        bad_call.function.name = "return_to_creator"
        bad_call.function.arguments = '{"destination": "Paris"}'  # missing required "days"
        bad_call.id = "return_call_1"
        child_response = _mock_llm_response(tool_calls=[bad_call])

        healed_response = MagicMock()
        healed_response.choices = [
            MagicMock(message=MagicMock(content='{"destination": "Paris", "days": 7}'))
        ]

        call_log = []

        async def fake_acompletion(*args, **kwargs):
            call_log.append(kwargs)
            return child_response if len(call_log) == 1 else healed_response

        with patch("litellm.acompletion", side_effect=fake_acompletion):
            result = await engine.execute_tool(
                "spawn_agent",
                {"role": "Paris specialist", "instruction": "plan paris", "tools": ["search"]},
                interactive=False,
                tool_call_id="spawn_call_1",
            )

        assert len(call_log) == 2  # the child's own turn, then exactly one corrector retry
        assert '"destination": "Paris"' in result
        assert '"days": 7' in result

    asyncio.run(_run())


def test_result_schema_misconfigured_falls_back_to_generic_summary_field(tmp_path):
    """A bad result_schema module path must not crash the turn — fall back to the ordinary
    free-text `summary` field/behavior, the same as if result_schema were never set."""

    async def _run():
        engine = await _init_engine(
            tmp_path, _mock_graph(result_schema="does_not_exist.NoSuchModel")
        )

        return_call = MagicMock()
        return_call.function.name = "return_to_creator"
        return_call.function.arguments = '{"summary": "Done with Paris."}'
        return_call.id = "return_call_1"
        child_response = _mock_llm_response(tool_calls=[return_call])

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
            mock_acompletion.return_value = child_response
            result = await engine.execute_tool(
                "spawn_agent",
                {"role": "Paris specialist", "instruction": "plan paris", "tools": ["search"]},
                interactive=False,
                tool_call_id="spawn_call_1",
            )

        assert "Done with Paris." in result
        sent_tools = mock_acompletion.call_args.kwargs["tools"]
        return_to_creator_schema = next(
            t for t in sent_tools if t["function"]["name"] == "return_to_creator"
        )
        assert "summary" in return_to_creator_schema["function"]["parameters"]["properties"]

    asyncio.run(_run())


def test_on_complete_writes_state_when_spawn_genuinely_completes(tmp_path):
    """spawns.on_complete unlocks a state-driven gate (e.g. tools[].available_when) automatically
    once a spawned agent genuinely finishes — no write_state instruction needed in the spawning
    agent's own prompt to the sub-agent."""

    async def _run():
        engine = await _init_engine(
            tmp_path,
            _mock_graph(
                on_complete=[StateWriteAction(key="research_done", value=True)],
            ),
        )
        assert "research_done" not in engine.state

        text_response = _mock_llm_response(content="Found X.")
        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
            mock_acompletion.return_value = text_response
            await engine.execute_tool(
                "spawn_agent",
                {"role": "x", "instruction": "y", "tools": ["search"]},
                interactive=False,
                tool_call_id="spawn_call_1",
            )

        assert engine.state.get("research_done") is True

    asyncio.run(_run())


def test_on_complete_does_not_fire_while_the_child_is_paused(tmp_path):
    """A spawn that pauses for approval hasn't completed — on_complete must not fire yet, or a
    tool gated on the flag it sets could become available before the thing it was supposed to
    gate on has actually finished."""

    async def _run():
        engine = await _init_engine(
            tmp_path,
            _mock_graph(
                requires_approval_on_first_action=True,
                tool_pool=["search"],
                on_complete=[StateWriteAction(key="research_done", value=True)],
            ),
        )

        search_call = MagicMock()
        search_call.function.name = "search"
        search_call.function.arguments = "{}"
        search_call.id = "child_search"
        child_response = _mock_llm_response(tool_calls=[search_call])

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
            mock_acompletion.return_value = child_response
            result = await engine.execute_tool(
                "spawn_agent",
                {"role": "x", "instruction": "y", "tools": ["search"]},
                interactive=False,
                tool_call_id="spawn_call_1",
            )

        assert "paused" in result.lower()
        assert "research_done" not in engine.state

    asyncio.run(_run())


def test_on_complete_does_not_fire_on_a_forced_max_turns_abort(tmp_path):
    """A child whose active_agent_name keeps changing away and back (is_transferring stays True
    every turn, so spawn_agent's own outer loop never finds a natural exit — the same shape as a
    handoff ping-pong that would otherwise run forever) must be cut off by max_delegation_turns
    without ever counting as a genuine completion. on_complete must not fire for it.

    (Note: a child that just keeps calling ordinary tools without transferring is bounded by
    _run_agent_turn's own internal 10-iteration cap instead, and returns with is_transferring
    still False — indistinguishable from "finished" at this outer loop, a separate pre-existing
    ambiguity this test doesn't exercise. This test targets max_delegation_turns specifically.)"""

    async def _run():
        engine = await _init_engine(
            tmp_path,
            _mock_graph(
                max_delegation_turns=2,
                on_complete=[StateWriteAction(key="research_done", value=True)],
            ),
        )

        async def never_settling_turn(self, interactive=True):
            # Simulates a child stuck perpetually transferring — is_transferring stays True and
            # active_agent_name never changes, so the outer while loop's own break conditions
            # never fire and it must be cut off purely by turn_count reaching max_turns.
            self.is_transferring = True

        with patch.object(RuntimeEngine, "_run_agent_turn", new=never_settling_turn):
            await engine.execute_tool(
                "spawn_agent",
                {"role": "x", "instruction": "y", "tools": ["search"]},
                interactive=False,
                tool_call_id="spawn_call_1",
            )

        assert "research_done" not in engine.state

    asyncio.run(_run())


def test_on_complete_fires_after_the_child_resolves_via_resume(tmp_path):
    """The same guarantee must hold on the /resume nested-child-approval path, not just
    spawn_agent's own synchronous completion — a spawn that pauses, gets approved, and then
    genuinely finishes must still fire on_complete exactly once, at the point it actually
    completes."""
    from intagrin.runtime.shared_resources import SharedResources
    from intagrin.server.api import ResumeRequest, _resume_nested_child_approval

    async def _run():
        engine = await _init_engine(
            tmp_path,
            _mock_graph(
                requires_approval_on_first_action=True,
                tool_pool=["search"],
                on_complete=[StateWriteAction(key="research_done", value=True)],
                memory_type="sqlite",
            ),
        )

        search_call = MagicMock()
        search_call.function.name = "search"
        search_call.function.arguments = "{}"
        search_call.id = "child_search"
        child_response = _mock_llm_response(tool_calls=[search_call])

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
            mock_acompletion.return_value = child_response
            await engine.execute_tool(
                "spawn_agent",
                {"role": "x", "instruction": "y", "tools": ["search"]},
                interactive=False,
                tool_call_id="spawn_call_1",
            )

        assert "research_done" not in engine.state
        pending_action = engine.state["_pending_approval"]

        # _save_checkpoint fires the child's DB write in a background thread
        # (fire-and-forget by design) — poll for it to actually land before
        # _resume_nested_child_approval tries to reload the child from it.
        for _ in range(50):
            await asyncio.sleep(0.1)
            probe = RuntimeEngine(
                graph=engine.graph,
                project_dir=tmp_path,
                session_id=pending_action["child_session_id"],
            )
            await probe.initialize()
            if "_pending_approval" in probe.state:
                break

        shared = SharedResources(
            mcp_manager=engine.mcp_manager,
            global_tool_schemas=engine.global_tool_schemas,
            local_tools=engine.local_tools,
            agent_prompts=engine.agent_prompts,
            tools_requiring_approval=engine.tools_requiring_approval,
        )

        final_response = _mock_llm_response(content="Found it!")
        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
            mock_acompletion.return_value = final_response
            await _resume_nested_child_approval(
                graph=engine.graph,
                project_dir=tmp_path,
                shared=shared,
                engine=engine,
                namespaced_session=engine.session_id,
                pending_action=pending_action,
                req=ResumeRequest(session_id=engine.session_id, approved=True),
                approver_id="default",
                pre_metrics={},
                run_start=0.0,
            )

        assert engine.state.get("research_done") is True

    asyncio.run(_run())
