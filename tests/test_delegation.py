import asyncio
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from intagrin.config.schema import AgentConfig, AppConfig, MemoryConfig, ModelConfig
from intagrin.runtime.engine import RuntimeEngine
from intagrin.runtime.tool_runner import ToolRunner


@pytest.fixture
def mock_graph_delegation():
    """Returns a mock parsed graph with two agents: a manager and a worker."""
    mock = MagicMock()
    mock.config = AppConfig(
        version="1.0",
        name="test_app",
        default_agent="manager",
        model=ModelConfig(primary="mock/model"),
        memory=MemoryConfig(type="buffer"),
        agents={
            "manager": AgentConfig(
                description="Manager agent",
                delegations=["worker"]
            ),
            "worker": AgentConfig(
                description="Worker agent",
                system_prompt_file=None
            )
        }
    )
    return mock

@pytest.mark.skip(reason="Refactored to ToolRunner")
def test_native_delegation(mock_graph_delegation):
    """Test that a Manager agent can dynamically delegate a task to a Worker sub-agent."""
    
    async def _run():
        engine = RuntimeEngine(graph=mock_graph_delegation, project_dir=".", session_id="test_session")
        await engine.initialize()
        
        engine.active_agent_name = "manager"
        
        # Verify the schema was dynamically injected!
        schemas = await ToolRunner.get_active_tools(engine, engine.graph.config.agents["manager"], engine.global_tool_schemas)
        tool_names = [s["function"]["name"] for s in schemas]
        assert "delegate_task" in tool_names
        
        # Execute the delegation tool
        # We need to mock acompletion so the child engine's loop actually does something and exits
        with patch("litellm.acompletion") as mock_acompletion:
            # When the child engine (worker) runs, it should return a mock answer
            # We'll make it return a normal text response (no tool calls) so the while loop in child_engine terminates
            mock_response = MagicMock()
            mock_response.choices = [
                MagicMock(message=MagicMock(role="assistant", content="I have completed the subtask sir!", tool_calls=None, model_dump=lambda exclude_none: {"role": "assistant", "content": "I have completed the subtask sir!"}))
            ]
            mock_response.usage = MagicMock(total_tokens=100, prompt_tokens=50, completion_tokens=50)
            mock_acompletion.return_value = mock_response
            
            # Fire the tool!
            result = await engine.execute_tool("delegate_task", {
                "target_agent": "worker",
                "task": "Please calculate the meaning of life."
            }, interactive=False)
            
            # Assert the tool successfully extracted the subagent's answer and returned it
            assert "Delegated task completed by worker" in result
            assert "I have completed the subtask sir!" in result
            
            # Assert that litellm was actually invoked for the child agent!
            assert mock_acompletion.call_count == 1
            
            # Finally, check that the subagent shares the same state dictionary as the parent (Typed Shared State)
            engine.state["foo"] = "bar"
            # We can't directly check the child's state from here, but in the code we passed `initial_state=self.state`
            # which passes the dictionary reference.
            
    asyncio.run(_run())


def _delegation_app(strict: bool, parent_tools: list, child_tools: list) -> AppConfig:
    """A manager delegating to a worker, with each side's tools: controllable."""
    return AppConfig(
        version="1.0",
        name="deleg_privilege_app",
        default_agent="manager",
        model=ModelConfig(primary="mock/model"),
        memory=MemoryConfig(type="buffer"),
        circuit_breakers={"strict_delegation_privilege": strict},
        agents={
            "manager": AgentConfig(delegations=["worker"], tools=parent_tools),
            "worker": AgentConfig(tools=child_tools),
        },
    )


_LOOKUP = {"name": "lookup_order", "module": "tools.custom"}
_REFUND = {"name": "issue_refund", "module": "tools.custom"}


def test_strict_delegation_privilege_rejects_a_delegate_holding_extra_tools():
    """spawns.tool_pool already makes privilege escalation through agent *creation* a validation
    error. Delegation was the open path: a delegated child runs as a full agent with its own
    tools:, so without this flag a manager reaches issue_refund just by delegating."""
    with pytest.raises(ValidationError) as excinfo:
        _delegation_app(strict=True, parent_tools=[_LOOKUP], child_tools=[_LOOKUP, _REFUND])

    message = str(excinfo.value)
    assert "issue_refund" in message
    assert "strict_delegation_privilege" in message


def test_strict_delegation_privilege_allows_a_delegate_with_a_subset_of_the_callers_tools():
    config = _delegation_app(
        strict=True, parent_tools=[_LOOKUP, _REFUND], child_tools=[_LOOKUP]
    )
    assert config.agents["manager"].delegations == ["worker"]


def test_delegation_privilege_is_not_enforced_by_default():
    """Delegating to a deliberately more capable specialist is a legitimate, common shape — the
    subset rule must stay opt-in so this doesn't break existing projects."""
    config = _delegation_app(
        strict=False, parent_tools=[_LOOKUP], child_tools=[_LOOKUP, _REFUND]
    )
    assert {t.name for t in config.agents["worker"].tools} == {"lookup_order", "issue_refund"}
