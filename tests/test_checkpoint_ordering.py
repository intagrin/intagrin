import asyncio
import time

import pytest

from intagrin.compiler.parser import ExecutionGraph
from intagrin.config.schema import AgentConfig, AppConfig, MemoryConfig, ModelConfig
from intagrin.runtime.engine import RuntimeEngine
from intagrin.runtime.memory import SQLiteCheckpointer


def _graph():
    config = AppConfig(
        version="1.0",
        name="checkpoint-order-test",
        default_agent="assistant",
        model=ModelConfig(primary="mock/model"),
        memory=MemoryConfig(type="sqlite"),
        agents={"assistant": AgentConfig()},
    )
    return ExecutionGraph(config, {})


@pytest.mark.anyio
async def test_save_checkpoint_calls_are_ordered_even_when_an_earlier_one_is_slow(tmp_path):
    """Regression test for a real bug: _save_checkpoint fires each write as an independent,
    unawaited asyncio.create_task(asyncio.to_thread(...)) — with no ordering guarantee across
    calls, an earlier call's background thread finishing AFTER a later call's would silently
    clobber newer state (a tool-result placeholder, a freshly-set _pending_approval, ...) with
    stale state, since SQLiteCheckpointer.save_checkpoint is a last-write-wins upsert. A single
    request routinely calls _save_checkpoint() several times in quick succession (before the turn
    loop, after the assistant message, after tool results, ...), so this reproduces within one
    request with no multi-request timing needed — this is the actual mechanism behind a real
    corrupted production session traced this session, where a tool-call's placeholder tool-role
    message was entirely absent from the persisted checkpoint despite having been appended and
    saved in memory.

    Verifies the fix: chaining each save onto the previously scheduled one via
    self._pending_checkpoint_task, and awaiting the last one (_await_last_checkpoint) before
    trusting the checkpoint reflects the latest state."""
    engine = RuntimeEngine(graph=_graph(), project_dir=tmp_path, session_id="order_test")
    await engine.initialize()

    real_save = SQLiteCheckpointer.save_checkpoint
    call_order = []

    def slow_first_save(self, session_id, messages, state):
        # Deliberately delay only the call carrying the FIRST message list's physical write, so
        # it would land AFTER the second call's write if the two saves were unordered/concurrent
        # — proving the fix actually serializes them rather than just getting lucky on timing.
        # Keyed off the message content itself (not a shared counter) so the delay assignment
        # can't itself race across the two background threads.
        if len(messages) == 1:
            time.sleep(0.3)
        call_order.append(messages)
        real_save(self, session_id, messages, state)

    engine.checkpointer.save_checkpoint = slow_first_save.__get__(
        engine.checkpointer, SQLiteCheckpointer
    )

    engine.messages = [{"role": "user", "content": "first"}]
    engine._save_checkpoint()
    # Give the first save a moment to actually start running in its background thread before the
    # second is scheduled — otherwise the chaining fix would trivially "work" just because the
    # first save hadn't been picked up by the thread pool yet.
    await asyncio.sleep(0.05)

    engine.messages = [{"role": "user", "content": "first"}, {"role": "assistant", "content": "second"}]
    engine._save_checkpoint()

    await engine._await_last_checkpoint()
    # _await_last_checkpoint only guarantees the MOST RECENTLY scheduled save has landed. If saves
    # aren't actually chained (the bug), the still-sleeping first save is an orphaned task nothing
    # is waiting on — it would land later and silently clobber the second save's write with stale
    # data. Give it time to do that (proving the fix truly serializes them, not just that
    # _await_last_checkpoint happened to return after the fast second write).
    await asyncio.sleep(0.5)

    # Both physical writes must have completed in the order they were scheduled, not the order
    # their background threads happened to finish.
    assert len(call_order) == 2
    assert call_order[0] == [{"role": "user", "content": "first"}]
    assert call_order[1] == [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "second"},
    ]

    # And critically: what's actually persisted is the LATEST state, not clobbered by the slow
    # first save landing after the second.
    reloaded = SQLiteCheckpointer(str(tmp_path / ".ai" / "memory.db"))
    persisted_messages, _ = reloaded.load_checkpoint("order_test")
    assert persisted_messages == [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "second"},
    ]
