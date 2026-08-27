"""_compress_memory's eviction slice must never split a function-call/function-response pair, and
must never start the kept window on a function-call turn with nothing before it either.

Real-world bug report #1: a live social-media-manager session hit
    litellm.BadRequestError: ... "Please ensure that function response turn comes
    immediately after a function call turn."
from Gemini/Vertex. The checkpointed history showed exactly `memory.max_messages` (20) messages,
with message[0] a dangling "tool" role reply — its "assistant" tool_calls message had been
evicted by the old `self.messages[-max_msgs:]` naive positional slice, which has zero awareness
of call/response pairing. OpenAI tolerates this kind of split; Gemini/Vertex reject it outright.

Real-world bug report #2 (same class, other half): the fix for #1 only walked the cut point back
past leading "tool" messages, so it could still land the kept window's first message on an
"assistant" tool_calls message with nothing evicted before it to be its required preceding
user/response turn — Gemini/Vertex reject that too ("Please ensure that function call turn comes
immediately after a user turn or after a function response turn"), a *different* error message
than #1's, which is why `_no_dangling_tool_messages` below didn't catch it: every "tool" message
was still correctly preceded by its own call, so nothing was "dangling" by that check's
definition — the problem was the call itself, not its response. Walking back to the nearest
"user" message (instead of just past "tool" messages) fixes both at once.
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from intagrin.compiler.parser import ExecutionGraph
from intagrin.config.schema import AgentConfig, AppConfig, MemoryConfig, ModelConfig
from intagrin.runtime.engine import RuntimeEngine


@pytest.fixture
def mock_graph():
    config = AppConfig(
        version="1.0",
        name="test-swarm",
        default_agent="triage",
        model=ModelConfig(primary="mock/model"),
        memory=MemoryConfig(type="buffer", max_messages=4),
        agents={"triage": AgentConfig(description="Triage agent")},
    )
    return ExecutionGraph(config, {})


def _no_dangling_tool_messages(messages: list[dict]) -> bool:
    """A "tool" role message must always be immediately preceded (possibly after other "tool"
    messages answering the same multi-call turn) by an "assistant" message with tool_calls."""
    for i, m in enumerate(messages):
        if m.get("role") != "tool":
            continue
        j = i - 1
        while j >= 0 and messages[j].get("role") == "tool":
            j -= 1
        if j < 0 or messages[j].get("role") != "assistant" or "tool_calls" not in messages[j]:
            return False
    return True


def _mock_llm_response(content: str = "summary"):
    return MagicMock(choices=[MagicMock(message=MagicMock(content=content))])


def test_naive_slice_would_have_produced_a_dangling_tool_message(mock_graph, tmp_path):
    """Sanity check the reproduction itself: with max_messages=4 and this 6-message history, the
    old `messages[-max_msgs:]` slice lands squarely on a dangling tool reply."""
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "tool_calls": [{"id": "c1"}]},
        {"role": "tool", "tool_call_id": "c1", "content": "result1"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "next"},
        {"role": "assistant", "content": "final"},
    ]
    naive_kept = messages[-4:]
    assert naive_kept[0]["role"] == "tool"
    assert not _no_dangling_tool_messages(naive_kept)


def test_compress_memory_never_splits_a_call_response_pair(mock_graph, tmp_path):
    async def _run():
        engine = RuntimeEngine(graph=mock_graph, project_dir=tmp_path, session_id="s1")
        await engine.initialize()
        engine.messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "tool_calls": [{"id": "c1"}]},
            {"role": "tool", "tool_call_id": "c1", "content": "result1"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "next"},
            {"role": "assistant", "content": "final"},
        ]
        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
            mock_acompletion.return_value = _mock_llm_response()
            await engine._compress_memory()
        return engine

    engine = asyncio.run(_run())
    assert _no_dangling_tool_messages(engine.messages)
    # The call+response pair must be evicted or kept *together*, never split.
    has_call = any(m.get("role") == "assistant" and "tool_calls" in m for m in engine.messages)
    has_response = any(m.get("role") == "tool" for m in engine.messages)
    assert has_call == has_response


def test_compress_memory_keeps_extra_messages_rather_than_split_a_pair(mock_graph, tmp_path):
    """The fix necessarily keeps all 6 messages here, not exactly max_messages=4 — correctness
    over hitting the exact budget is the point. (Previously asserted the walk-back stopped at
    index 1, keeping 5 messages starting on the "assistant" tool_calls message itself — that was
    bug report #2 above, not a correct outcome; walking back to the nearest "user" message
    continues past it to index 0, so nothing is evicted at all in this particular history.)"""
    async def _run():
        engine = RuntimeEngine(graph=mock_graph, project_dir=tmp_path, session_id="s2")
        await engine.initialize()
        engine.messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "tool_calls": [{"id": "c1"}]},
            {"role": "tool", "tool_call_id": "c1", "content": "result1"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "next"},
            {"role": "assistant", "content": "final"},
        ]
        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
            mock_acompletion.return_value = _mock_llm_response()
            await engine._compress_memory()
        return engine

    engine = asyncio.run(_run())
    assert len(engine.messages) == 6
    assert engine.messages[0]["role"] == "user"


def test_compress_memory_never_starts_the_kept_window_on_an_unpreceded_function_call(
    mock_graph, tmp_path
):
    """Reproduces bug report #2 exactly: the naive cut point lands squarely on an "assistant"
    tool_calls message that itself is immediately followed by its own "tool" response — so
    `_no_dangling_tool_messages` sees nothing wrong (every response has its call) — but the kept
    window's very first turn is a function call with no preceding user/response turn, which
    Gemini/Vertex reject with a different error message than the dangling-response case."""
    async def _run():
        engine = RuntimeEngine(graph=mock_graph, project_dir=tmp_path, session_id="s5")
        await engine.initialize()
        engine.messages = [
            {"role": "user", "content": "m0"},
            {"role": "assistant", "content": "m1"},
            {"role": "user", "content": "m2"},
            {"role": "assistant", "tool_calls": [{"id": "c1"}]},
            {"role": "tool", "tool_call_id": "c1", "content": "r1"},
            {"role": "assistant", "content": "m5"},
        ]
        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
            mock_acompletion.return_value = _mock_llm_response()
            await engine._compress_memory()
        return engine

    engine = asyncio.run(_run())
    assert engine.messages[0]["role"] == "user", (
        "the kept window must start on a user turn, not directly on the assistant's tool_calls "
        "message — even though that message's own tool response is present and correctly paired"
    )
    assert engine.messages[0]["content"] == "m2"
    assert len(engine.messages) == 4


def test_compress_memory_ordinary_case_unaffected_when_cut_lands_cleanly(mock_graph, tmp_path):
    """No tool messages anywhere near the cut boundary — behavior must be identical to the old
    plain slice (exactly max_messages kept)."""
    async def _run():
        engine = RuntimeEngine(graph=mock_graph, project_dir=tmp_path, session_id="s3")
        await engine.initialize()
        engine.messages = [
            {"role": "user", "content": "m0"},
            {"role": "assistant", "content": "m1"},
            {"role": "user", "content": "m2"},
            {"role": "assistant", "content": "m3"},
            {"role": "user", "content": "m4"},
            {"role": "assistant", "content": "m5"},
        ]
        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
            mock_acompletion.return_value = _mock_llm_response()
            await engine._compress_memory()
        return engine

    engine = asyncio.run(_run())
    assert len(engine.messages) == 4
    assert [m["content"] for m in engine.messages] == ["m2", "m3", "m4", "m5"]


def test_compress_memory_extreme_case_all_leading_messages_are_tool_skips_compression(
    mock_graph, tmp_path
):
    """If walking back from the naive cut point reaches index 0 still on a "tool" message (a
    pathological/corrupt history), compression must no-op for this round rather than ever
    producing an invalid split — never crash, never guess."""
    async def _run():
        engine = RuntimeEngine(graph=mock_graph, project_dir=tmp_path, session_id="s4")
        await engine.initialize()
        engine.messages = [
            {"role": "tool", "tool_call_id": "orphan", "content": "orphaned from a prior bug"},
            {"role": "tool", "tool_call_id": "orphan2", "content": "also orphaned"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "next"},
            {"role": "assistant", "content": "final"},
        ]
        original = list(engine.messages)
        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
            mock_acompletion.return_value = _mock_llm_response()
            await engine._compress_memory()
        return engine, original

    engine, original = asyncio.run(_run())
    assert engine.messages == original


def test_compress_memory_batches_large_evictions_into_multiple_corrector_calls(mock_graph, tmp_path):
    """circuit_breakers.max_compression_batch_messages bounds how many evicted messages go into a
    single compression prompt — inta verify used to flag this input side as unbounded. A large
    eviction must be folded in successive batches rather than dumped into one giant prompt."""
    mock_graph.config.circuit_breakers.max_compression_batch_messages = 2

    async def _run():
        engine = RuntimeEngine(graph=mock_graph, project_dir=tmp_path, session_id="s5")
        await engine.initialize()
        # 10 plain alternating turns; max_messages=4 evicts the first 6 (cut lands cleanly on a
        # "user" message, no walk-back needed).
        engine.messages = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"msg{i}"}
            for i in range(10)
        ]
        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
            mock_acompletion.return_value = _mock_llm_response("summary")
            await engine._compress_memory()
        return mock_acompletion, engine

    mock_acompletion, engine = asyncio.run(_run())
    # 6 evicted messages / batch size 2 = 3 corrector calls, not 1.
    assert mock_acompletion.await_count == 3
    for call in mock_acompletion.await_args_list:
        prompt = call.kwargs["messages"][1]["content"]
        logs_json = prompt.split("NEW LOGS TO COMPRESS:\n", 1)[1]
        assert len(json.loads(logs_json)) <= 2
    assert engine.state["long_term_memory"] == "summary"
