import asyncio
from unittest.mock import MagicMock, patch

import pytest

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
