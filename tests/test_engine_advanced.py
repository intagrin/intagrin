import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from intagrin.compiler.parser import ExecutionGraph
from intagrin.config.schema import (
    AgentConfig,
    AppConfig,
    MemoryConfig,
    ModelConfig,
    RouterConfig,
    ServerConfig,
)
from intagrin.runtime.engine import RuntimeEngine


@pytest.fixture
def mock_graph():
    config = AppConfig(
        version="1.0",
        name="test-swarm",
        default_agent="triage",
        state_schema="schemas.UserState",
        max_session_budget_usd=5.00,
        model=ModelConfig(primary="mock/model"),
        memory=MemoryConfig(type="buffer"),
        server=ServerConfig(webhook_url="http://mock.com", webhook_secret_env_var="MOCK_SECRET"),
        agents={
            "triage": AgentConfig(
                description="Triage agent",
                routers=[
                    RouterConfig(condition="balance < 0", target="collections")
                ]
            ),
            "collections": AgentConfig(description="Collections agent")
        }
    )
    return ExecutionGraph(config, {})


def test_deterministic_routing(mock_graph):
    """Test that if the python condition is met, the engine transfers without LLM intervention."""
    async def _run():
        engine = RuntimeEngine(graph=mock_graph, project_dir=Path.cwd(), session_id="test_1")
        await engine.initialize()
        
        # Set state so balance < 0
        engine.state["balance"] = -50
        engine.active_agent_name = "triage"
        
        # Run a single turn
        await engine._run_agent_turn(interactive=False)
        
        # Because condition was met, active agent should instantly swap to 'collections'
        assert engine.active_agent_name == "collections"
        assert engine.is_transferring
    asyncio.run(_run())

def test_run_agent_turn_reevaluates_conditional_routers_after_a_tool_call(mock_graph):
    """Regression test for a real bug: _run_agent_turn_stream re-evaluates conditional/root
    routers after every tool-call round (a tool can change state a router condition depends on —
    see its own docstring), but the blocking _run_agent_turn never had the equivalent call. A
    conditional router meant to fire once a tool's write_state satisfied its condition only ever
    actually routed on /chat/stream — the same config, same input, on /chat, /resume, or `inta
    run` would keep calling the LLM instead of transferring immediately. Proven here by only ever
    queuing ONE mocked LLM response: if the blocking loop doesn't re-check the router after the
    tool call and instead loops back to the LLM again, this raises IndexError on the second
    (unavailable) response — the fixed loop must transfer and stop after the first round."""

    class _Msg:
        def __init__(self, tool_calls=None):
            self.tool_calls = tool_calls
            self.content = None

        def model_dump(self, exclude_none=True):
            d = {"role": "assistant"}
            if self.tool_calls:
                d["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in self.tool_calls
                ]
            return d

    async def _run():
        engine = RuntimeEngine(graph=mock_graph, project_dir=Path.cwd(), session_id="test_router_midturn")
        await engine.initialize()
        engine.active_agent_name = "triage"
        engine.messages.append({"role": "user", "content": "my balance seems off"})

        tool_call = MagicMock()
        tool_call.id = "call_1"
        tool_call.function.name = "write_state"
        tool_call.function.arguments = '{"key": "balance", "value": "-50"}'

        response = MagicMock(choices=[MagicMock(message=_Msg(tool_calls=[tool_call]))], usage=None)
        # Only ONE response queued on purpose — see docstring.
        responses = [response]

        async def _fake_acompletion(*args, **kwargs):
            return responses.pop(0)

        async def _fake_execute_tool_calls(tool_calls, interactive):
            # Stand-in for the real write_state execution — actually mutates state the way the
            # real tool call would, without needing full JSON-schema tool-call plumbing.
            engine.state["balance"] = -50
            return [
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "name": "write_state",
                    "content": "State updated.",
                }
            ]

        with patch("litellm.acompletion", side_effect=_fake_acompletion), patch.object(
            RuntimeEngine,
            "_execute_tool_calls_with_healing",
            AsyncMock(side_effect=_fake_execute_tool_calls),
        ):
            await engine._run_agent_turn(interactive=False)

        assert engine.active_agent_name == "collections"
        assert engine.is_transferring

    asyncio.run(_run())


def test_budget_circuit_breaker(mock_graph, capsys):
    """Test that engine aborts if budget is exceeded."""
    async def _run():
        engine = RuntimeEngine(graph=mock_graph, project_dir=Path.cwd(), session_id="test_2")
        await engine.initialize()
        
        engine.active_agent_name = "triage"
        # Mock usage past budget
        engine.state["_metrics"] = {"total_cost": 6.00}
        
        await engine._run_agent_turn(interactive=False)
        
        # The messages should contain the hard abort error
        assert any("Exceeded maximum session budget" in msg.get("content", "") for msg in engine.messages if msg.get("role") == "assistant")
    asyncio.run(_run())

def test_shared_state_injection(mock_graph):
    """Test that JIT Shared Typed State is actually injected into the system prompt — previously
    this test set up state and asserted nothing, so it always passed regardless of whether
    injection worked. _build_system_prompt is sync and injects unconditionally whenever
    state_schema is set (which mock_graph's config does), so no LLM call is needed to verify it."""
    async def _run():
        engine = RuntimeEngine(graph=mock_graph, project_dir=Path.cwd(), session_id="test_3")
        await engine.initialize()

        engine.state["balance"] = -50
        engine.active_agent_name = "triage"

        agent_cfg = mock_graph.config.agents["triage"]
        prompt = engine._build_system_prompt(agent_cfg)

        assert "SHARED TYPED STATE" in prompt
        assert "balance" in prompt
        assert "-50" in prompt
    asyncio.run(_run())

def test_error_loop_compression(mock_graph):
    """Test that 3 identical consecutive tool errors trigger the GC barrier."""
    engine = RuntimeEngine(graph=mock_graph, project_dir=Path.cwd(), session_id="test_4")
    
    # Simulate a loop: 3 identical tool calls that fail exactly the same way
    for _ in range(3):
        engine.messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "bad_tool", "arguments": "{}"}}]
        })
        engine.messages.append({
            "role": "tool",
            "name": "bad_tool",
            "content": "System Error: Invalid syntax"
        })
        
    engine._compress_error_loops()
    
    # GC should have triggered. Messages should be truncated and a system barrier injected.
    assert len(engine.messages) > 0
    last_msg = engine.messages[-1]
    assert last_msg["role"] == "system"
    assert "YOU ARE STUCK IN A LOOP" in last_msg["content"]
