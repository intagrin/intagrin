import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from intagrin.compiler.parser import ExecutionGraph
from intagrin.config.schema import (
    AgentConfig,
    AppConfig,
    EpisodicMemoryConfig,
    MemoryConfig,
    ModelConfig,
)
from intagrin.runtime.engine import RuntimeEngine
from intagrin.runtime.episodic_memory import (
    query_episodes,
    save_episode,
    semantic_search_episodes,
)


def _sqlite_mem_cfg():
    cfg = MagicMock()
    cfg.type = "sqlite"
    cfg.db_path = None
    return cfg


def _embed_response(vector):
    return MagicMock(data=[{"embedding": vector}])


def test_save_and_query_round_trip(tmp_path):
    save_episode(_sqlite_mem_cfg(), tmp_path, "acme", "s1", "preference", "likes window seats", None, None)
    save_episode(_sqlite_mem_cfg(), tmp_path, "acme", "s1", "failure", "card declined", None, None)

    prefs = query_episodes(_sqlite_mem_cfg(), tmp_path, "acme", "preference", None, 10)
    assert len(prefs) == 1
    assert prefs[0]["content"] == "likes window seats"
    assert prefs[0]["event_type"] == "preference"

    everything = query_episodes(_sqlite_mem_cfg(), tmp_path, "acme", None, None, 10)
    assert len(everything) == 2


def test_episodes_are_append_only_not_upserted(tmp_path):
    """Structural difference from shared_memory.py's single-row UPSERT: saving twice under the
    same scope_key must produce two rows, not one overwritten row."""
    save_episode(_sqlite_mem_cfg(), tmp_path, "acme", "s1", "note", "first", None, None)
    save_episode(_sqlite_mem_cfg(), tmp_path, "acme", "s1", "note", "second", None, None)

    rows = query_episodes(_sqlite_mem_cfg(), tmp_path, "acme", None, None, 10)
    assert len(rows) == 2
    assert {r["content"] for r in rows} == {"first", "second"}


def test_query_filters_require_all_given_tags(tmp_path):
    save_episode(_sqlite_mem_cfg(), tmp_path, "acme", "s1", "note", "a", ["urgent", "flight"], None)
    save_episode(_sqlite_mem_cfg(), tmp_path, "acme", "s1", "note", "b", ["flight"], None)

    both = query_episodes(_sqlite_mem_cfg(), tmp_path, "acme", None, ["urgent", "flight"], 10)
    assert [r["content"] for r in both] == ["a"]

    just_flight = query_episodes(_sqlite_mem_cfg(), tmp_path, "acme", None, ["flight"], 10)
    assert {r["content"] for r in just_flight} == {"a", "b"}


def test_query_returns_empty_list_for_unknown_scope(tmp_path):
    assert query_episodes(_sqlite_mem_cfg(), tmp_path, "nobody", None, None, 10) == []


def test_noop_for_non_sqlite_postgres_memory_types(tmp_path):
    cfg = MagicMock()
    cfg.type = "buffer"
    save_episode(cfg, tmp_path, "acme", "s1", "note", "should not persist", None, None)
    assert query_episodes(cfg, tmp_path, "acme", None, None, 10) == []
    assert not (tmp_path / ".ai").exists()


def test_save_never_raises_on_write_failure(tmp_path):
    with patch(
        "intagrin.runtime.episodic_memory.sqlite3.connect", side_effect=OSError("disk full")
    ), patch("intagrin.runtime.episodic_memory.Tracer.log_error") as mock_log_error:
        save_episode(_sqlite_mem_cfg(), tmp_path, "acme", "s1", "note", "content", None, None)
    mock_log_error.assert_called_once()


def test_query_never_raises_on_read_failure(tmp_path):
    with patch(
        "intagrin.runtime.episodic_memory.sqlite3.connect", side_effect=OSError("disk full")
    ), patch("intagrin.runtime.episodic_memory.Tracer.log_error") as mock_log_error:
        result = query_episodes(_sqlite_mem_cfg(), tmp_path, "acme", None, None, 10)
    assert result == []
    mock_log_error.assert_called_once()


def test_semantic_search_ranks_by_cosine_similarity(tmp_path):
    async def _run():
        save_episode(_sqlite_mem_cfg(), tmp_path, "acme", "s1", "note", "close match", None, [1.0, 0.0])
        save_episode(_sqlite_mem_cfg(), tmp_path, "acme", "s1", "note", "far match", None, [0.0, 1.0])

        with patch("litellm.aembedding", new_callable=AsyncMock) as mock_aembedding:
            mock_aembedding.return_value = _embed_response([1.0, 0.0])
            results = await semantic_search_episodes(
                _sqlite_mem_cfg(), tmp_path, "acme", "mock-embed", "query", None, None, 5
            )

        assert [r["content"] for r in results] == ["close match", "far match"]

    asyncio.run(_run())


def test_semantic_search_falls_back_to_recency_when_embedding_api_fails(tmp_path):
    async def _run():
        save_episode(_sqlite_mem_cfg(), tmp_path, "acme", "s1", "note", "only entry", None, None)

        with patch("litellm.aembedding", new_callable=AsyncMock) as mock_aembedding:
            mock_aembedding.side_effect = RuntimeError("embedding provider down")
            results = await semantic_search_episodes(
                _sqlite_mem_cfg(), tmp_path, "acme", "mock-embed", "query", None, None, 5
            )

        assert [r["content"] for r in results] == ["only entry"]

    asyncio.run(_run())


# --- Engine integration -----------------------------------------------------------------------


def _mock_graph(episodic_memory=None, memory_type="sqlite"):
    config = AppConfig(
        version="1.0",
        name="test-swarm",
        default_agent="triage",
        model=ModelConfig(primary="mock/model"),
        memory=MemoryConfig(type=memory_type),
        agents={"triage": AgentConfig(description="Triage agent")},
        episodic_memory=episodic_memory,
    )
    return ExecutionGraph(config, {})


def test_episodic_memory_none_by_default_no_tools_registered(tmp_path):
    async def _run():
        engine = RuntimeEngine(graph=_mock_graph(None), project_dir=tmp_path, session_id="s1")
        await engine.initialize()

        assert "remember_episode" not in engine.local_tools
        assert "recall_episodes" not in engine.local_tools
        assert engine._is_tool_allowed_for_active_agent("remember_episode") is False

    asyncio.run(_run())


def test_remember_and_recall_episode_round_trip_through_engine_tools(tmp_path):
    async def _run():
        engine = RuntimeEngine(
            graph=_mock_graph(EpisodicMemoryConfig(scope="session")),
            project_dir=tmp_path,
            session_id="s1",
        )
        await engine.initialize()

        assert engine._is_tool_allowed_for_active_agent("remember_episode") is True
        assert engine._is_tool_allowed_for_active_agent("recall_episodes") is True

        with patch("litellm.aembedding", new_callable=AsyncMock) as mock_aembedding:
            mock_aembedding.side_effect = RuntimeError("no embedding provider in this test")
            result = await engine.remember_episode("preference", "user prefers window seats")
        assert "Recorded episode" in result

        recall = await engine.recall_episodes(event_type="preference")
        assert "user prefers window seats" in recall

    asyncio.run(_run())


def test_episodic_scope_tenant_shares_across_sessions_scope_session_does_not(tmp_path):
    async def _run():
        async def _remember(engine):
            with patch("litellm.aembedding", new_callable=AsyncMock) as mock_aembedding:
                mock_aembedding.side_effect = RuntimeError("no embedding provider in this test")
                await engine.remember_episode("note", "tenant A secret")

        # scope="tenant": two different session_ids under the same tenant prefix see each other's episodes.
        engine_a = RuntimeEngine(
            graph=_mock_graph(EpisodicMemoryConfig(scope="tenant")),
            project_dir=tmp_path,
            session_id="tenantA:session1",
        )
        await engine_a.initialize()
        await _remember(engine_a)

        engine_b = RuntimeEngine(
            graph=_mock_graph(EpisodicMemoryConfig(scope="tenant")),
            project_dir=tmp_path,
            session_id="tenantA:session2",
        )
        await engine_b.initialize()
        assert "tenant A secret" in await engine_b.recall_episodes()

        engine_c = RuntimeEngine(
            graph=_mock_graph(EpisodicMemoryConfig(scope="tenant")),
            project_dir=tmp_path,
            session_id="tenantB:session1",
        )
        await engine_c.initialize()
        assert "tenant A secret" not in await engine_c.recall_episodes()

        # scope="session" (default): even the SAME tenant prefix, a different session_id doesn't see it.
        engine_d = RuntimeEngine(
            graph=_mock_graph(EpisodicMemoryConfig(scope="session")),
            project_dir=tmp_path,
            session_id="tenantA:session3",
        )
        await engine_d.initialize()
        assert "tenant A secret" not in await engine_d.recall_episodes()

    asyncio.run(_run())


def test_engine_reusing_shared_resources_rebinds_episodic_tools_per_session(tmp_path):
    """Regression test for the pooled-SharedResources fast path: remember_episode/recall_episodes
    are bound RuntimeEngine methods (need this engine's own session_id/state), exactly like
    read_state/write_state — not like search_knowledge_base, whose closure captures no `self`.
    Without an explicit rebind after the shallow dict-copy of a pool-builder's local_tools, a
    pooled session's remember_episode would stay bound to the pool-builder engine, silently
    writing under the wrong session/scope."""

    async def _run():
        graph = _mock_graph(EpisodicMemoryConfig(scope="session"))

        builder = RuntimeEngine(graph=graph, project_dir=tmp_path, session_id="builder-session")
        await builder.initialize()
        shared = builder._as_shared_resources()

        engine = RuntimeEngine(
            graph=graph, project_dir=tmp_path, session_id="real-session", shared_resources=shared
        )
        await engine.initialize()

        assert engine.local_tools["remember_episode"] == engine.remember_episode
        assert engine.local_tools["remember_episode"] != builder.remember_episode
        assert engine.local_tools["recall_episodes"] == engine.recall_episodes

        with patch("litellm.aembedding", new_callable=AsyncMock) as mock_aembedding:
            mock_aembedding.side_effect = RuntimeError("no embedding provider in this test")
            await engine.remember_episode("note", "real session's own episode")

        # The pool-builder's own (different session_id, scope="session") recall must NOT see it —
        # proves the tool actually wrote under `engine`'s scope, not `builder`'s.
        builder_recall = await builder.recall_episodes()
        assert "real session's own episode" not in builder_recall

        engine_recall = await engine.recall_episodes()
        assert "real session's own episode" in engine_recall

    asyncio.run(_run())


def test_retention_days_prunes_old_episodes_but_keeps_recent_ones(tmp_path):
    """episodes is append-only with no automatic pruning by default — retention_days (from
    EpisodicMemoryConfig, threaded through save_episode as a plain value) opportunistically
    deletes rows older than the cutoff. Forces the random trigger deterministically via
    monkeypatch instead of relying on its real ~0.5% probability."""
    import sqlite3
    from datetime import UTC, datetime, timedelta
    from unittest.mock import patch

    mem_cfg = _sqlite_mem_cfg()
    save_episode(mem_cfg, tmp_path, "acme", "s1", "note", "old episode", None, None)

    db_path = tmp_path / (mem_cfg.db_path or ".ai/memory.db")
    old_created_at = (datetime.now(UTC) - timedelta(days=100)).strftime("%Y-%m-%d %H:%M:%S")
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute(
            "UPDATE episodes SET created_at = ? WHERE content = 'old episode'", (old_created_at,)
        )
        conn.commit()

    with patch("intagrin.runtime.episodic_memory.random.random", return_value=0.0):
        save_episode(mem_cfg, tmp_path, "acme", "s1", "note", "new episode", None, None, retention_days=30)

    remaining = query_episodes(mem_cfg, tmp_path, "acme", None, None, limit=10)
    contents = {r["content"] for r in remaining}
    assert "new episode" in contents
    assert "old episode" not in contents


def test_retention_days_none_never_prunes(tmp_path):
    """The default (retention_days=None) must not prune anything, even on the lucky random roll —
    pruning is strictly opt-in."""
    import sqlite3
    from datetime import UTC, datetime, timedelta
    from unittest.mock import patch

    mem_cfg = _sqlite_mem_cfg()
    save_episode(mem_cfg, tmp_path, "acme", "s1", "note", "ancient episode", None, None)

    db_path = tmp_path / (mem_cfg.db_path or ".ai/memory.db")
    old_created_at = (datetime.now(UTC) - timedelta(days=9999)).strftime("%Y-%m-%d %H:%M:%S")
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute(
            "UPDATE episodes SET created_at = ? WHERE content = 'ancient episode'", (old_created_at,)
        )
        conn.commit()

    with patch("intagrin.runtime.episodic_memory.random.random", return_value=0.0):
        save_episode(mem_cfg, tmp_path, "acme", "s1", "note", "another episode", None, None)

    remaining = query_episodes(mem_cfg, tmp_path, "acme", None, None, limit=10)
    contents = {r["content"] for r in remaining}
    assert "ancient episode" in contents
