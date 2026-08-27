from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from intagrin.compiler.parser import ExecutionGraph
from intagrin.config.schema import (
    AgentConfig,
    AppConfig,
    MemoryConfig,
    ModelConfig,
    StateReducerConfig,
    VoteConfig,
    WorkflowTask,
)
from intagrin.runtime.engine import RuntimeEngine


def _mock_graph(vote_task: WorkflowTask):
    config = AppConfig(
        version="1.0",
        name="test_vote_app",
        default_agent="manager",
        model=ModelConfig(primary="openai/gpt-4o-mini", fallback="openai/gpt-4o-mini"),
        memory=MemoryConfig(type="sqlite"),
        reducers=[StateReducerConfig(key="reports", strategy="append")],
        agents={
            "manager": AgentConfig(description="Manager agent"),
            "checker": AgentConfig(description="Checker agent"),
        },
        workflows={"vote_flow": [vote_task]},
    )
    return ExecutionGraph(config=config, env_vars={})


def _branch_task(name: str, vote: VoteConfig | None = None) -> WorkflowTask:
    return WorkflowTask(
        name=name,
        type="vote",
        vote=vote,
        tasks=[
            WorkflowTask(name="b1", agent="checker", instruction="check"),
            WorkflowTask(name="b2", agent="checker", instruction="check"),
            WorkflowTask(name="b3", agent="checker", instruction="check"),
        ],
    )


def _patched_branches(answers: dict[str, str]):
    """Patches RuntimeEngine so each branch (by session-id substring, e.g. 'branch_b1') ends its
    turn with the given final assistant content — mirrors test_swarm_and_reducers.py's pattern."""
    original_init = RuntimeEngine.__init__

    def mock_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        for branch_name, answer in answers.items():
            if f"branch_{branch_name}" in getattr(self, "session_id", ""):
                self.state = {"_metrics": {"total_tokens": 1, "total_cost": 0.001}}
                self._pending_answer = answer

    async def mock_run_agent_turn(self):
        self.is_transferring = False
        self.messages.append(
            {"role": "assistant", "content": getattr(self, "_pending_answer", "No response")}
        )

    return patch.object(RuntimeEngine, "__init__", new=mock_init), patch.object(
        RuntimeEngine, "initialize", new_callable=AsyncMock
    ), patch.object(RuntimeEngine, "_run_agent_turn", new=mock_run_agent_turn)


@pytest.mark.anyio
async def test_vote_majority_picks_winner(tmp_path):
    """2 of 3 branches agree ('Paris') — majority wins, and zero extra LLM calls are made."""
    task = _branch_task("capital_check")
    graph = _mock_graph(task)
    engine = RuntimeEngine(graph=graph, project_dir=tmp_path)
    await engine.initialize()

    p1, p2, p3 = _patched_branches({"b1": "Paris", "b2": "Paris", "b3": "London"})
    with p1, p2, p3, patch(
        "intagrin.runtime.engine.litellm.acompletion", new_callable=AsyncMock
    ) as mock_llm:
        await engine._execute_task(task, 0)

    content = engine.messages[-1]["content"]
    assert "result (majority, 2/3 agreed): Paris" in content
    mock_llm.assert_not_called()


@pytest.mark.anyio
async def test_vote_no_consensus_below_min_agreement(tmp_path):
    """All 3 branches disagree — default min_agreement=0.5 isn't met, so the task must not guess
    a winner; it reports 'no consensus reached' plus every branch's output, with no LLM call."""
    task = _branch_task("capital_check")
    graph = _mock_graph(task)
    engine = RuntimeEngine(graph=graph, project_dir=tmp_path)
    await engine.initialize()

    p1, p2, p3 = _patched_branches({"b1": "Paris", "b2": "London", "b3": "Berlin"})
    with p1, p2, p3, patch(
        "intagrin.runtime.engine.litellm.acompletion", new_callable=AsyncMock
    ) as mock_llm:
        await engine._execute_task(task, 0)

    content = engine.messages[-1]["content"]
    assert "no consensus reached" in content
    assert "Paris" in content and "London" in content and "Berlin" in content
    mock_llm.assert_not_called()


@pytest.mark.anyio
async def test_vote_llm_judge_calls_model_and_uses_result(tmp_path):
    """llm_judge makes exactly one litellm.acompletion call using model.fallback at
    temperature=0, and its (mocked) return value becomes the vote result."""
    task = _branch_task("capital_check", vote=VoteConfig(strategy="llm_judge"))
    graph = _mock_graph(task)
    engine = RuntimeEngine(graph=graph, project_dir=tmp_path)
    await engine.initialize()

    judge_response = MagicMock()
    judge_response.choices = [MagicMock(message=MagicMock(content="Paris (consensus)"))]

    p1, p2, p3 = _patched_branches({"b1": "Paris", "b2": "London", "b3": "Berlin"})
    with p1, p2, p3, patch(
        "intagrin.runtime.engine.litellm.acompletion",
        new_callable=AsyncMock,
        return_value=judge_response,
    ) as mock_llm:
        await engine._execute_task(task, 0)

    content = engine.messages[-1]["content"]
    assert "result (llm_judge): Paris (consensus)" in content
    mock_llm.assert_called_once()
    _, kwargs = mock_llm.call_args
    assert kwargs["model"] == "openai/gpt-4o-mini"
    assert kwargs["temperature"] == 0


@pytest.mark.anyio
async def test_vote_state_merge_back_still_works(tmp_path):
    """State merge-back for a 'vote' task reuses _merge_child_state exactly like 'parallel' does
    — proves reuse, not duplication, of the merge path."""
    task = _branch_task("capital_check")
    graph = _mock_graph(task)
    engine = RuntimeEngine(graph=graph, project_dir=tmp_path)
    await engine.initialize()
    engine.state["reports"] = ["initial_report"]

    original_init = RuntimeEngine.__init__

    def mock_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if "branch_b1" in getattr(self, "session_id", ""):
            self.state = {"reports": ["b1_report"], "_metrics": {"total_tokens": 0, "total_cost": 0.0}}
            self._pending_answer = "Paris"
        elif "branch_b2" in getattr(self, "session_id", ""):
            self.state = {"reports": ["b2_report"], "_metrics": {"total_tokens": 0, "total_cost": 0.0}}
            self._pending_answer = "Paris"
        elif "branch_b3" in getattr(self, "session_id", ""):
            self.state = {"reports": ["b3_report"], "_metrics": {"total_tokens": 0, "total_cost": 0.0}}
            self._pending_answer = "London"

    async def mock_run_agent_turn(self):
        self.is_transferring = False
        self.messages.append(
            {"role": "assistant", "content": getattr(self, "_pending_answer", "No response")}
        )

    with patch.object(RuntimeEngine, "__init__", new=mock_init), patch.object(
        RuntimeEngine, "initialize", new_callable=AsyncMock
    ), patch.object(RuntimeEngine, "_run_agent_turn", new=mock_run_agent_turn):
        await engine._execute_task(task, 0)

    assert engine.state["reports"] == [
        "initial_report",
        "b1_report",
        "b2_report",
        "b3_report",
    ]


@pytest.mark.anyio
async def test_vote_majority_tie_break_is_deterministic_by_branch_order(tmp_path):
    """A 2-vs-2 tie across 4 branches resolves to the first-declared branch's answer."""
    task = WorkflowTask(
        name="tie_check",
        type="vote",
        tasks=[
            WorkflowTask(name="b1", agent="checker", instruction="check"),
            WorkflowTask(name="b2", agent="checker", instruction="check"),
            WorkflowTask(name="b3", agent="checker", instruction="check"),
            WorkflowTask(name="b4", agent="checker", instruction="check"),
        ],
    )
    graph = _mock_graph(task)
    engine = RuntimeEngine(graph=graph, project_dir=tmp_path)
    await engine.initialize()

    p1, p2, p3 = _patched_branches(
        {"b1": "Paris", "b2": "London", "b3": "London", "b4": "Paris"}
    )
    with p1, p2, p3, patch(
        "intagrin.runtime.engine.litellm.acompletion", new_callable=AsyncMock
    ) as mock_llm:
        await engine._execute_task(task, 0)

    content = engine.messages[-1]["content"]
    assert "result (majority, 2/4 agreed): Paris" in content
    mock_llm.assert_not_called()
