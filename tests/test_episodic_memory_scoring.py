"""Tests for the Generative-Agents-style memory scoring added to episodic_memory.py:
recency-decayed + importance-weighted + relevance ranking in semantic_search_episodes, instead
of pure cosine similarity — see _recency_score and semantic_search_episodes' combined_score."""

import asyncio
import sqlite3
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from intagrin.runtime.episodic_memory import (
    _recency_score,
    ensure_schema,
    query_episodes,
    save_episode,
    semantic_search_episodes,
)
from intagrin.runtime.engine import RuntimeEngine
from intagrin.runtime.tools_loader import get_tool_schema


def _sqlite_mem_cfg():
    cfg = MagicMock()
    cfg.type = "sqlite"
    cfg.db_path = None
    return cfg


def _embed_response(vector):
    return MagicMock(data=[{"embedding": vector}])


def _set_created_at(tmp_path, content: str, when: datetime) -> None:
    db_path = tmp_path / ".ai" / "memory.db"
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute(
            "UPDATE episodes SET created_at = ? WHERE content = ?",
            (when.strftime("%Y-%m-%d %H:%M:%S"), content),
        )
        conn.commit()


# --- importance: storage, clamping, defaults -------------------------------------------------


def test_importance_round_trips_through_save_and_query(tmp_path):
    save_episode(_sqlite_mem_cfg(), tmp_path, "acme", "s1", "note", "rated", None, None, importance=8)
    rows = query_episodes(_sqlite_mem_cfg(), tmp_path, "acme", None, None, 10)
    assert rows[0]["importance"] == 8


def test_unrated_importance_stored_as_none(tmp_path):
    save_episode(_sqlite_mem_cfg(), tmp_path, "acme", "s1", "note", "unrated", None, None)
    rows = query_episodes(_sqlite_mem_cfg(), tmp_path, "acme", None, None, 10)
    assert rows[0]["importance"] is None


def test_importance_is_clamped_to_the_1_to_10_range(tmp_path):
    save_episode(_sqlite_mem_cfg(), tmp_path, "acme", "s1", "note", "too high", None, None, importance=15)
    save_episode(_sqlite_mem_cfg(), tmp_path, "acme", "s1", "note", "too low", None, None, importance=-5)
    rows = {r["content"]: r["importance"] for r in query_episodes(_sqlite_mem_cfg(), tmp_path, "acme", None, None, 10)}
    assert rows["too high"] == 10
    assert rows["too low"] == 1


# --- _recency_score -----------------------------------------------------------------------------


def test_recency_score_is_near_one_for_a_fresh_timestamp():
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
    assert _recency_score(now) > 0.99


def test_recency_score_decays_with_age():
    thirty_days_ago = (datetime.now(UTC) - timedelta(days=30)).strftime("%Y-%m-%d %H:%M:%S")
    a_year_ago = (datetime.now(UTC) - timedelta(days=365)).strftime("%Y-%m-%d %H:%M:%S")
    thirty_day_score = _recency_score(thirty_days_ago)
    year_score = _recency_score(a_year_ago)
    assert 0.80 < thirty_day_score < 0.90  # 0.995**30 ≈ 0.860
    assert 0.10 < year_score < 0.20  # 0.995**365 ≈ 0.161
    assert year_score < thirty_day_score


def test_recency_score_never_raises_on_a_malformed_timestamp():
    assert _recency_score("not a real timestamp") == 0.0
    assert _recency_score(None) == 0.0


# --- semantic_search_episodes: the actual blended ranking ---------------------------------------


def test_recent_important_episode_outranks_a_stale_slightly_more_similar_one(tmp_path):
    """The exact scenario this feature exists for: 'stale' has marginally higher raw cosine
    similarity to the query but is a year old and unrated; 'fresh' is a hair less similar but
    recent and rated highly important. Pure-similarity ranking (the old behavior) would put
    'stale' first; the blended score must put 'fresh' first instead."""

    async def _run():
        mem_cfg = _sqlite_mem_cfg()
        save_episode(mem_cfg, tmp_path, "acme", "s1", "note", "stale", None, [1.0, 0.0], importance=None)
        save_episode(mem_cfg, tmp_path, "acme", "s1", "note", "fresh", None, [0.95, 0.05], importance=9)
        _set_created_at(tmp_path, "stale", datetime.now(UTC) - timedelta(days=365))
        # "fresh" keeps its just-now created_at from save_episode.

        with patch("litellm.aembedding", new_callable=AsyncMock) as mock_aembedding:
            mock_aembedding.return_value = _embed_response([1.0, 0.0])
            results = await semantic_search_episodes(
                mem_cfg, tmp_path, "acme", "mock-embed", "query", None, None, 5
            )

        # Sanity check the premise: "stale" really is the closer raw cosine match.
        from intagrin.runtime.rag import cosine_similarity

        assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) > cosine_similarity([1.0, 0.0], [0.95, 0.05])

        assert [r["content"] for r in results] == ["fresh", "stale"]

    asyncio.run(_run())


def test_two_equally_relevant_recent_episodes_rank_by_importance(tmp_path):
    async def _run():
        mem_cfg = _sqlite_mem_cfg()
        save_episode(mem_cfg, tmp_path, "acme", "s1", "note", "low importance", None, [1.0, 0.0], importance=1)
        save_episode(mem_cfg, tmp_path, "acme", "s1", "note", "high importance", None, [1.0, 0.0], importance=10)

        with patch("litellm.aembedding", new_callable=AsyncMock) as mock_aembedding:
            mock_aembedding.return_value = _embed_response([1.0, 0.0])
            results = await semantic_search_episodes(
                mem_cfg, tmp_path, "acme", "mock-embed", "query", None, None, 5
            )

        assert [r["content"] for r in results] == ["high importance", "low importance"]

    asyncio.run(_run())


# --- migration: a pre-existing database without the `importance` column -------------------------


def test_ensure_schema_migrates_a_pre_existing_database_missing_importance_column(tmp_path):
    """Simulates a memory.db created before `importance` existed — the exact shape a real
    upgrade encounters. ensure_schema's ALTER TABLE migration must add the column without
    erroring, and save/query must work normally afterward."""
    db_dir = tmp_path / ".ai"
    db_dir.mkdir(parents=True)
    db_path = db_dir / "memory.db"
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute(
            """
            CREATE TABLE episodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scope_key TEXT,
                session_id TEXT,
                event_type TEXT,
                content TEXT,
                tags TEXT,
                embedding TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            "INSERT INTO episodes (scope_key, session_id, event_type, content, tags, embedding) "
            "VALUES ('acme', 's0', 'note', 'pre-upgrade row', '[]', NULL)"
        )
        conn.commit()

    mem_cfg = _sqlite_mem_cfg()
    ensure_schema(mem_cfg, tmp_path)  # must not raise on the pre-existing narrower table

    save_episode(mem_cfg, tmp_path, "acme", "s1", "note", "post-upgrade row", None, None, importance=7)
    rows = query_episodes(mem_cfg, tmp_path, "acme", None, None, 10)
    by_content = {r["content"]: r for r in rows}
    assert by_content["pre-upgrade row"]["importance"] is None
    assert by_content["post-upgrade row"]["importance"] == 7


# --- LLM-facing schema: importance must actually carry guidance, not a generic placeholder ------


def test_remember_episode_tool_schema_documents_importance_for_the_model():
    """Regression guard: get_tool_schema's docstring parser only picks up a param's description
    from a single "name: description" line — a prose paragraph (the original style this
    docstring used to be written in) silently falls back to "Parameter importance" with none of
    the calibration guidance the model needs to rate usefully."""
    schema = get_tool_schema(RuntimeEngine.remember_episode)
    importance_desc = schema["function"]["parameters"]["properties"]["importance"]["description"]
    assert "1-10" in importance_desc
    assert importance_desc != "Parameter importance"
