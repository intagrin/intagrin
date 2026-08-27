import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from intagrin.compiler.parser import ExecutionGraph
from intagrin.config.schema import AgentConfig, AppConfig, MemoryConfig, ModelConfig
from intagrin.runtime.engine import RuntimeEngine
from intagrin.runtime.shared_memory import load_shared_memory, save_shared_memory


def _sqlite_mem_cfg(shared_scope="tenant"):
    cfg = MagicMock()
    cfg.type = "sqlite"
    cfg.db_path = None
    cfg.shared_scope = shared_scope
    return cfg


def test_save_and_load_round_trip(tmp_path):
    save_shared_memory(_sqlite_mem_cfg(), tmp_path, "acme_corp", "The user prefers dark mode.")
    content = load_shared_memory(_sqlite_mem_cfg(), tmp_path, "acme_corp")
    assert content == "The user prefers dark mode."


def test_load_returns_none_for_an_unknown_scope_key(tmp_path):
    assert load_shared_memory(_sqlite_mem_cfg(), tmp_path, "nobody_wrote_this") is None


def test_save_overwrites_last_write_wins(tmp_path):
    save_shared_memory(_sqlite_mem_cfg(), tmp_path, "acme_corp", "v1")
    save_shared_memory(_sqlite_mem_cfg(), tmp_path, "acme_corp", "v2")
    assert load_shared_memory(_sqlite_mem_cfg(), tmp_path, "acme_corp") == "v2"


def test_noop_for_non_sqlite_postgres_memory_types(tmp_path):
    cfg = MagicMock()
    cfg.type = "buffer"
    save_shared_memory(cfg, tmp_path, "acme_corp", "should not persist")
    assert load_shared_memory(cfg, tmp_path, "acme_corp") is None
    assert not (tmp_path / ".ai").exists()


def test_save_never_raises_on_write_failure(tmp_path):
    with patch(
        "intagrin.runtime.shared_memory.sqlite3.connect", side_effect=OSError("disk full")
    ), patch("intagrin.runtime.shared_memory.Tracer.log_error") as mock_log_error:
        save_shared_memory(_sqlite_mem_cfg(), tmp_path, "acme_corp", "content")
    mock_log_error.assert_called_once()


def test_load_never_raises_on_read_failure(tmp_path):
    with patch(
        "intagrin.runtime.shared_memory.sqlite3.connect", side_effect=OSError("disk full")
    ), patch("intagrin.runtime.shared_memory.Tracer.log_error") as mock_log_error:
        result = load_shared_memory(_sqlite_mem_cfg(), tmp_path, "acme_corp")
    assert result is None
    mock_log_error.assert_called_once()


# --- Engine integration -----------------------------------------------------------------------


def _mock_graph(shared_scope: str):
    config = AppConfig(
        version="1.0",
        name="test-swarm",
        default_agent="triage",
        model=ModelConfig(primary="mock/model"),
        memory=MemoryConfig(type="sqlite", shared_scope=shared_scope),
        agents={"triage": AgentConfig(description="Triage agent")},
    )
    return ExecutionGraph(config, {})


class _MockMessage:
    def __init__(self, content):
        self.content = content


class _MockChoice:
    def __init__(self, message):
        self.message = message


class _MockResponse:
    def __init__(self, content):
        self.choices = [_MockChoice(_MockMessage(content))]


def test_session_scope_default_has_zero_shared_memory_behavior(tmp_path):
    """shared_scope defaults to 'session' — no load/save call should happen at all."""

    async def _run():
        engine = RuntimeEngine(
            graph=_mock_graph("session"), project_dir=tmp_path, session_id="tenantA:s1"
        )
        with patch(
            "intagrin.runtime.engine.load_shared_memory"
        ) as mock_load:
            await engine.initialize()
        mock_load.assert_not_called()

    asyncio.run(_run())


def test_tenant_scope_shares_compressed_memory_across_sessions_under_the_same_tenant(tmp_path):
    """Two different session_ids under the same tenant prefix, shared_scope: tenant — session
    B's initialize() must pick up session A's summarized long_term_memory after A compresses."""

    async def _run():
        engine_a = RuntimeEngine(
            graph=_mock_graph("tenant"), project_dir=tmp_path, session_id="tenantA:session1"
        )
        await engine_a.initialize()
        engine_a.messages = [
            {"role": "user", "content": f"message {i}"} for i in range(engine_a.graph.config.memory.max_messages + 5)
        ]

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
            mock_acompletion.return_value = _MockResponse("User's name is Alex and prefers email.")
            await engine_a._compress_memory()

        assert engine_a.state["long_term_memory"] == "User's name is Alex and prefers email."

        # A second, distinct session under the SAME tenant prefix.
        engine_b = RuntimeEngine(
            graph=_mock_graph("tenant"), project_dir=tmp_path, session_id="tenantA:session2"
        )
        await engine_b.initialize()

        assert "User's name is Alex and prefers email." in engine_b.state.get(
            "long_term_memory", ""
        )

        # A third session under a DIFFERENT tenant must NOT see tenant A's shared memory.
        engine_c = RuntimeEngine(
            graph=_mock_graph("tenant"), project_dir=tmp_path, session_id="tenantB:session1"
        )
        await engine_c.initialize()
        assert "Alex" not in engine_c.state.get("long_term_memory", "")

    asyncio.run(_run())


def test_merge_is_idempotent_and_does_not_duplicate_on_repeated_initialize(tmp_path):
    """initialize() re-checks shared memory on every call (every request) — re-running with
    unchanged shared content must not keep re-appending it."""

    async def _run():
        from intagrin.runtime.shared_memory import save_shared_memory

        save_shared_memory(
            _sqlite_mem_cfg(), tmp_path, "tenantA", "Shared fact: user is on the Pro plan."
        )

        engine = RuntimeEngine(
            graph=_mock_graph("tenant"), project_dir=tmp_path, session_id="tenantA:s1"
        )
        await engine.initialize()
        first_ltm = engine.state.get("long_term_memory", "")
        assert first_ltm.count("Shared fact: user is on the Pro plan.") == 1

        await engine.initialize()
        second_ltm = engine.state.get("long_term_memory", "")
        assert second_ltm.count("Shared fact: user is on the Pro plan.") == 1
        assert second_ltm == first_ltm

    asyncio.run(_run())
