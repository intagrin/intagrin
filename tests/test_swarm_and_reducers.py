from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from intagrin.compiler.parser import ExecutionGraph
from intagrin.config.schema import (
    AgentConfig,
    AppConfig,
    MemoryConfig,
    ModelConfig,
    StateReducerConfig,
    WorkflowTask,
)
from intagrin.runtime.engine import RuntimeEngine


@pytest.fixture
def mock_graph():
    config = AppConfig(
        version="1.0",
        name="test_app",
        default_agent="manager",
        model=ModelConfig(primary="openai/gpt-4o-mini", fallback="openai/gpt-4o-mini"),
        memory=MemoryConfig(type="sqlite"),
        reducers=[
            StateReducerConfig(key="reports", strategy="append"),
            StateReducerConfig(key="summary", strategy="overwrite"),
            StateReducerConfig(key="stats", strategy="deep_merge")
        ],
        agents={
            "manager": AgentConfig(
                description="Manager agent",
                auto_route=True
            ),
            "researcher": AgentConfig(
                description="Research agent",
                auto_route=True
            )
        },
        workflows={
            "parallel_research": [
                WorkflowTask(
                    name="parallel_gather",
                    type="parallel",
                    tasks=[
                        WorkflowTask(name="task1", agent="researcher", instruction="Do research 1"),
                        WorkflowTask(name="task2", agent="researcher", instruction="Do research 2")
                    ]
                )
            ]
        }
    )
    return ExecutionGraph(config=config, env_vars={})

@pytest.mark.anyio
async def test_declarative_state_reducers(mock_graph, tmp_path):
    """Test that parallel workflows correctly reduce their branch states into the parent state."""
    engine = RuntimeEngine(graph=mock_graph, project_dir=tmp_path)
    await engine.initialize()
    
    # Initialize some state
    engine.state["reports"] = ["initial_report"]
    engine.state["summary"] = "old_summary"
    engine.state["stats"] = {"count": 1}
    
    # We will mock RuntimeEngine so that the child branches return our expected states
    original_init = RuntimeEngine.__init__
    def mock_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        # Check if this is a branch engine by its session ID
        if "branch_task1" in getattr(self, "session_id", ""):
            self.state = {
                "reports": ["task1_report"],
                "summary": "task1_summary",
                "stats": {"task1_done": True},
                "_metrics": {"total_tokens": 10, "total_cost": 0.1}
            }
        elif "branch_task2" in getattr(self, "session_id", ""):
            self.state = {
                "reports": "task2_report_string", # Testing appending a string instead of a list
                "summary": "task2_summary",
                "stats": {"task2_done": True},
                "_metrics": {"total_tokens": 20, "total_cost": 0.2}
            }
            
    async def mock_run_agent_turn(self):
        # Prevent infinite loop, pretend it finished
        self.is_transferring = False
        self.messages.append({"role": "assistant", "content": "Done!"})
            
    with patch.object(RuntimeEngine, '__init__', new=mock_init):
        with patch.object(RuntimeEngine, 'initialize', new_callable=AsyncMock):
            with patch.object(RuntimeEngine, '_run_agent_turn', new=mock_run_agent_turn):
                await engine._execute_task(mock_graph.config.workflows["parallel_research"][0], 0)
        
        # Now verify the state was reduced correctly according to the declarative YAML strategies!
        
        # 1. Append Strategy:
        assert engine.state["reports"] == ["initial_report", "task1_report", "task2_report_string"]
        
        # 2. Overwrite Strategy (task2 runs second in gather, so it wins)
        assert engine.state["summary"] == "task2_summary"
        
        # 3. Deep Merge Strategy
        assert engine.state["stats"] == {"count": 1, "task1_done": True, "task2_done": True}

@pytest.mark.anyio
async def test_semantic_swarm_routing(mock_graph, tmp_path):
    """Test that when an agent finishes and auto_route=True, it dynamically routes to the next best agent."""
    engine = RuntimeEngine(graph=mock_graph, project_dir=tmp_path)
    await engine.initialize()
    
    engine.active_agent_name = "manager"
    
    class MockMessage:
        def __init__(self, content, tool_calls=None):
            self.content = content
            self.tool_calls = tool_calls
        def model_dump(self, exclude_none=True):
            return {"role": "assistant", "content": self.content}
            
    class MockChoice:
        def __init__(self, message):
            self.message = message
            
    class MockResponse:
        def __init__(self, message):
            self.choices = [MockChoice(message)]
            self.usage = MagicMock(prompt_tokens=10, completion_tokens=10, total_tokens=20)
    
    # We will mock the two LLM calls
    # Call 1: The manager generates a response
    # Call 2: The Swarm Router (tiny model) selects "researcher"
    mock_responses = [
        MockResponse(MockMessage("I need someone to research this.")), # Manager response
        MockResponse(MockMessage("researcher")) # Swarm Router selects researcher
    ]
    
    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
        mock_acompletion.side_effect = mock_responses
        
        engine.is_transferring = False
        await engine._run_agent_turn()
        
        # Check that it triggered a transfer dynamically
        assert engine.is_transferring is True
        assert engine.active_agent_name == "researcher"
        
        # Verify the system message was injected
        assert engine.messages[-1]["content"] == "Semantic Swarm Router: Control transferred to researcher."
