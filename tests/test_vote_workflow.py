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


# --- vote.debate_rounds: multi-agent debate ----------------------------------------------------


def _patched_debating_branches(round_answers: dict[str, list[str]]):
    """Like _patched_branches, but each branch's _run_agent_turn returns a DIFFERENT answer per
    call — round_answers[branch_name][call_index] — so a test can prove a branch's answer
    actually changes across debate rounds, not just that the mechanism ran without crashing.
    Calling past the end of a branch's answer list repeats its last answer. Also exposes every
    branch engine instance created, in creation order, via the returned `engines` list."""
    original_init = RuntimeEngine.__init__
    call_counts: dict[str, int] = {}
    engines: list[RuntimeEngine] = []

    def mock_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        for branch_name in round_answers:
            if f"branch_{branch_name}" in getattr(self, "session_id", ""):
                self.state = {"_metrics": {"total_tokens": 1, "total_cost": 0.001}}
                self._branch_name = branch_name
                engines.append(self)

    async def mock_run_agent_turn(self):
        self.is_transferring = False
        branch_name = getattr(self, "_branch_name", None)
        answers = round_answers.get(branch_name, ["No response"])
        idx = call_counts.get(branch_name, 0)
        call_counts[branch_name] = idx + 1
        self.messages.append({"role": "assistant", "content": answers[min(idx, len(answers) - 1)]})

    patches = (
        patch.object(RuntimeEngine, "__init__", new=mock_init),
        patch.object(RuntimeEngine, "initialize", new_callable=AsyncMock),
        patch.object(RuntimeEngine, "_run_agent_turn", new=mock_run_agent_turn),
    )
    return patches, call_counts, engines


@pytest.mark.anyio
async def test_debate_rounds_default_is_one_no_extra_llm_turns(tmp_path):
    """debate_rounds defaults to 1 — today's exact original behavior, zero extra cost for any
    project not opting in. Each branch's _run_agent_turn must be called exactly once."""
    task = _branch_task("capital_check")  # no vote= override, so VoteConfig() defaults apply
    graph = _mock_graph(task)
    engine = RuntimeEngine(graph=graph, project_dir=tmp_path)
    await engine.initialize()

    (p1, p2, p3), call_counts, _ = _patched_debating_branches(
        {"b1": ["Paris"], "b2": ["Paris"], "b3": ["London"]}
    )
    with p1, p2, p3, patch("intagrin.runtime.engine.litellm.acompletion", new_callable=AsyncMock):
        await engine._execute_task(task, 0)

    assert call_counts == {"b1": 1, "b2": 1, "b3": 1}


@pytest.mark.anyio
async def test_debate_runs_the_configured_number_of_extra_rounds(tmp_path):
    task = _branch_task("capital_check", vote=VoteConfig(debate_rounds=3))
    graph = _mock_graph(task)
    engine = RuntimeEngine(graph=graph, project_dir=tmp_path)
    await engine.initialize()

    (p1, p2, p3), call_counts, _ = _patched_debating_branches(
        {
            "b1": ["Paris", "Paris", "Paris"],
            "b2": ["London", "Paris", "Paris"],
            "b3": ["Berlin", "Berlin", "Paris"],
        }
    )
    with p1, p2, p3, patch("intagrin.runtime.engine.litellm.acompletion", new_callable=AsyncMock):
        await engine._execute_task(task, 0)

    assert call_counts == {"b1": 3, "b2": 3, "b3": 3}


@pytest.mark.anyio
async def test_debate_aggregation_reflects_the_final_round_not_the_first(tmp_path):
    """Round 1 has no consensus (3-way split); by round 3 every branch has converged on 'Paris'
    after seeing the others' answers. The vote result must reflect round 3, proving the
    aggregation step runs on debate's final answers, not the initial independent ones."""
    task = _branch_task("capital_check", vote=VoteConfig(debate_rounds=3))
    graph = _mock_graph(task)
    engine = RuntimeEngine(graph=graph, project_dir=tmp_path)
    await engine.initialize()

    (p1, p2, p3), _, _ = _patched_debating_branches(
        {
            "b1": ["Paris", "Paris", "Paris"],
            "b2": ["London", "Paris", "Paris"],
            "b3": ["Berlin", "Berlin", "Paris"],
        }
    )
    with p1, p2, p3, patch("intagrin.runtime.engine.litellm.acompletion", new_callable=AsyncMock) as mock_llm:
        await engine._execute_task(task, 0)

    content = engine.messages[-1]["content"]
    assert "result (majority, 3/3 agreed): Paris" in content
    mock_llm.assert_not_called()  # majority strategy — debate itself makes no extra LLM calls beyond the branch turns


@pytest.mark.anyio
async def test_debate_shows_each_branch_the_others_current_answers(tmp_path):
    """The revision prompt appended to a branch's own conversation must actually contain the
    OTHER branches' current answers, not just a generic "reconsider" nudge with nothing to
    reconsider against."""
    task = _branch_task("capital_check", vote=VoteConfig(debate_rounds=2))
    graph = _mock_graph(task)
    engine = RuntimeEngine(graph=graph, project_dir=tmp_path)
    await engine.initialize()

    (p1, p2, p3), _, engines = _patched_debating_branches(
        {"b1": ["Paris", "Paris"], "b2": ["London", "Paris"], "b3": ["Berlin", "Paris"]}
    )
    with p1, p2, p3, patch("intagrin.runtime.engine.litellm.acompletion", new_callable=AsyncMock):
        await engine._execute_task(task, 0)

    b1_engine = next(e for e in engines if e._branch_name == "b1")
    revision_messages = [m["content"] for m in b1_engine.messages if m.get("role") == "user"]
    # b1's revision prompt (the second user message — the first is the original TASK INSTRUCTION)
    # must show it b2's and b3's round-1 answers, not its own.
    assert len(revision_messages) >= 2
    revision_prompt = revision_messages[1]
    assert "London" in revision_prompt
    assert "Berlin" in revision_prompt


@pytest.mark.anyio
async def test_debate_state_merge_uses_the_final_rounds_state(tmp_path):
    """State merge-back must reflect the LAST debate round's state, not round 1's — proves
    _merge_child_state runs after every round completes, not right after the first."""
    task = _branch_task("capital_check", vote=VoteConfig(debate_rounds=2))
    graph = _mock_graph(task)
    engine = RuntimeEngine(graph=graph, project_dir=tmp_path)
    await engine.initialize()
    engine.state["reports"] = ["initial_report"]

    original_init = RuntimeEngine.__init__
    call_counts: dict[str, int] = {}

    def mock_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        for branch_name in ("b1", "b2", "b3"):
            if f"branch_{branch_name}" in getattr(self, "session_id", ""):
                self._branch_name = branch_name
                self.state = {"reports": [f"{branch_name}_round1"], "_metrics": {"total_tokens": 0, "total_cost": 0.0}}

    async def mock_run_agent_turn(self):
        self.is_transferring = False
        idx = call_counts.get(self._branch_name, 0)
        call_counts[self._branch_name] = idx + 1
        # On the second call (the debate round), the branch's state changes — proves the merge
        # uses this later state, not the round-1 state captured in mock_init.
        if idx == 1:
            self.state = {"reports": [f"{self._branch_name}_round2"], "_metrics": {"total_tokens": 0, "total_cost": 0.0}}
        self.messages.append({"role": "assistant", "content": "Paris"})

    with (
        patch.object(RuntimeEngine, "__init__", new=mock_init),
        patch.object(RuntimeEngine, "initialize", new_callable=AsyncMock),
        patch.object(RuntimeEngine, "_run_agent_turn", new=mock_run_agent_turn),
    ):
        await engine._execute_task(task, 0)

    assert engine.state["reports"] == [
        "initial_report",
        "b1_round2",
        "b2_round2",
        "b3_round2",
    ]


@pytest.mark.anyio
async def test_debate_rounds_ignored_for_a_parallel_task_even_if_vote_config_is_set(tmp_path):
    """vote.debate_rounds only has meaning for type: 'vote' — a 'parallel' task setting it
    (nonsensical, but not schema-invalid) must not run extra rounds."""
    task = WorkflowTask(
        name="parallel_check",
        type="parallel",
        vote=VoteConfig(debate_rounds=3),
        tasks=[
            WorkflowTask(name="b1", agent="checker", instruction="check"),
            WorkflowTask(name="b2", agent="checker", instruction="check"),
        ],
    )
    graph = _mock_graph(task)
    engine = RuntimeEngine(graph=graph, project_dir=tmp_path)
    await engine.initialize()

    (p1, p2, p3), call_counts, _ = _patched_debating_branches({"b1": ["A", "B", "C"], "b2": ["X", "Y", "Z"]})
    with p1, p2, p3, patch("intagrin.runtime.engine.litellm.acompletion", new_callable=AsyncMock):
        await engine._execute_task(task, 0)

    assert call_counts == {"b1": 1, "b2": 1}


def test_debate_rounds_schema_rejects_a_value_above_the_cap():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        VoteConfig(debate_rounds=6)


def test_debate_rounds_schema_rejects_a_value_below_one():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        VoteConfig(debate_rounds=0)
