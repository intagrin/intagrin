"""memory.context_edit_keep_recent_messages: a rule-based, zero-LLM-cost complement to
_compress_memory (see tests/test_memory_compression.py for that mechanism) that clears an old
'tool' message's content body in place, well before memory.max_messages is ever reached, instead
of waiting for the whole history to hit the limit before an expensive LLM summarization pass.
"""

import asyncio

import pytest

from intagrin.compiler.parser import ExecutionGraph
from intagrin.config.schema import AgentConfig, AppConfig, MemoryConfig, ModelConfig
from intagrin.runtime.engine import RuntimeEngine


def _make_graph(keep_recent):
    config = AppConfig(
        version="1.0",
        name="test-swarm",
        default_agent="triage",
        model=ModelConfig(primary="mock/model"),
        memory=MemoryConfig(
            type="buffer", max_messages=100, context_edit_keep_recent_messages=keep_recent
        ),
        agents={"triage": AgentConfig(description="Triage agent")},
    )
    return ExecutionGraph(config, {})


def _make_engine(graph, tmp_path, messages, session_id="s1"):
    async def _run():
        engine = RuntimeEngine(graph=graph, project_dir=tmp_path, session_id=session_id)
        await engine.initialize()
        engine.messages = messages
        return engine

    return asyncio.run(_run())


def test_disabled_by_default_leaves_messages_untouched(tmp_path):
    graph = _make_graph(None)
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "tool_calls": [{"id": "c1"}]},
        {"role": "tool", "tool_call_id": "c1", "name": "search", "content": "a" * 500},
        {"role": "assistant", "content": "ok"},
    ]
    engine = _make_engine(graph, tmp_path, list(messages))

    engine._apply_context_editing()

    assert engine.messages == messages


def test_clears_old_tool_result_beyond_keep_recent(tmp_path):
    graph = _make_graph(2)
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "tool_calls": [{"id": "c1"}]},
        {"role": "tool", "tool_call_id": "c1", "name": "search", "content": "a" * 500},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "next"},
    ]
    engine = _make_engine(graph, tmp_path, messages)

    engine._apply_context_editing()

    tool_msg = engine.messages[2]
    assert tool_msg["content"] == "[tool result cleared to reduce context size — original call: search()]"
    # The message itself (and its tool_call_id pairing) must survive — only content is touched.
    assert tool_msg["role"] == "tool"
    assert tool_msg["tool_call_id"] == "c1"
    assert engine.messages[1]["tool_calls"] == [{"id": "c1"}]


def test_never_touches_tool_results_within_the_recent_window(tmp_path):
    graph = _make_graph(2)
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "tool_calls": [{"id": "c1"}]},
        {"role": "tool", "tool_call_id": "c1", "name": "search", "content": "a" * 500},
    ]
    engine = _make_engine(graph, tmp_path, messages)

    engine._apply_context_editing()

    # Only 3 messages total and keep_recent=2 means cutoff <= 0 — nothing is old enough to clear.
    assert engine.messages[2]["content"] == "a" * 500


def test_idempotent_and_never_grows_content(tmp_path):
    graph = _make_graph(2)
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "tool_calls": [{"id": "c1"}]},
        {"role": "tool", "tool_call_id": "c1", "name": "search", "content": "a" * 500},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "next"},
    ]
    engine = _make_engine(graph, tmp_path, messages)

    engine._apply_context_editing()
    first_pass = engine.messages[2]["content"]
    engine._apply_context_editing()
    second_pass = engine.messages[2]["content"]

    assert first_pass == second_pass


def test_never_replaces_a_short_result_with_a_longer_placeholder(tmp_path):
    graph = _make_graph(2)
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "tool_calls": [{"id": "c1"}]},
        {"role": "tool", "tool_call_id": "c1", "name": "search", "content": "ok"},
        {"role": "assistant", "content": "done"},
        {"role": "user", "content": "next"},
    ]
    engine = _make_engine(graph, tmp_path, messages)

    engine._apply_context_editing()

    assert engine.messages[2]["content"] == "ok"


def test_ge_validator_rejects_values_below_two():
    with pytest.raises(Exception):
        MemoryConfig(type="buffer", context_edit_keep_recent_messages=1)
