"""Episodic memory (see EpisodicMemoryConfig / ai.yaml's `episodic_memory:` block) — discrete,
structured, individually queryable event records ("user prefers window seats", "booking BK-4471
failed: card declined"), distinct from two other things this framework already has: rag.py's
document/knowledge-base retrieval (unrelated to agent experience), and the single blended
long_term_memory prose summary _compress_memory produces (loses structure/filterability — you
can't ask it "what happened with booking BK-4471 specifically").

Mirrors runtime/shared_memory.py's and runtime/run_logger.py's self-managing-schema style — raw
sqlite3/psycopg, no ORM — and lives in the same database as `checkpoints`/`run_logs`/
`shared_memory` (same sqlite file, or the same Postgres connection). One structural difference
from shared_memory.py: this table is APPEND-ONLY (a plain INSERT per remember_episode call, many
rows per scope_key) rather than a single upserted row per scope_key.

Scoped to memory.type in ("sqlite", "postgres") only — same scope shared_memory.py/run_logger.py
already use. Entirely best-effort: every function catches all exceptions and returns a safe
default rather than raising, so a broken episodic-memory backend never breaks the tool call (or
the turn) that triggered it.
"""

import json
import random
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import litellm

from ..errors import IntaGrinError
from ..tracing.console import Tracer
from .memory import pooled_postgres_connection, postgres_dict_cursor
from .rag import cosine_similarity
from .run_logger import _PRUNE_PROBABILITY, _resolve_postgres_url

# See run_logger.py's identical _NO_DRIVER_ERRORS for why both are caught: pooled_postgres_connection
# raises IntaGrinError, not ImportError, when neither psycopg driver is installed.
_NO_DRIVER_ERRORS = (ImportError, IntaGrinError)

_SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS episodes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scope_key TEXT,
    session_id TEXT,
    event_type TEXT,
    content TEXT,
    tags TEXT,
    embedding TEXT,
    importance INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
"""

# ALTER TABLE ... ADD COLUMN has no "IF NOT EXISTS" in SQLite before 3.35 — guarded against a
# pre-existing memory.db from before `importance` existed, same pattern as worker.py's
# job_queue migration.
_SQLITE_MIGRATIONS = ("ALTER TABLE episodes ADD COLUMN importance INTEGER",)

_SQLITE_INDEX = (
    "CREATE INDEX IF NOT EXISTS ix_episodes_scope_key_created_at "
    "ON episodes(scope_key, created_at)"
)

_POSTGRES_SCHEMA = """
CREATE TABLE IF NOT EXISTS episodes (
    id SERIAL PRIMARY KEY,
    scope_key VARCHAR(255),
    session_id VARCHAR(255),
    event_type VARCHAR(255),
    content TEXT,
    tags JSONB,
    embedding JSONB,
    importance INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
)
"""

# Postgres supports IF NOT EXISTS on ADD COLUMN directly, unlike SQLite — no try/except dance
# needed for the equivalent pre-existing-table case.
_POSTGRES_MIGRATIONS = ("ALTER TABLE episodes ADD COLUMN IF NOT EXISTS importance INTEGER",)

_POSTGRES_INDEX = (
    "CREATE INDEX IF NOT EXISTS ix_episodes_scope_key_created_at "
    "ON episodes(scope_key, created_at)"
)

# Rows fetched (with their stored embeddings) before cosine-reranking down to `limit` — bounds the
# cost of a semantic recall without needing a real vector index for what's meant to be a small,
# per-agent memory store, not a bulk document corpus (that's rag.py's job).
_SEMANTIC_CANDIDATE_CAP = 200


def ensure_schema(mem_cfg, project_dir: Path) -> None:
    """Creates the episodes table (+ index) if it doesn't already exist. Called before every
    read/write, same reasoning as shared_memory.py/run_logger.py — a fresh project must not 500."""
    if mem_cfg.type == "sqlite":
        db_path = project_dir / (mem_cfg.db_path or ".ai/memory.db")
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(db_path), timeout=15.0) as conn:
            conn.execute(_SQLITE_SCHEMA)
            conn.execute(_SQLITE_INDEX)
            for ddl in _SQLITE_MIGRATIONS:
                try:
                    conn.execute(ddl)
                except sqlite3.OperationalError:
                    pass  # column already exists
            conn.commit()
    elif mem_cfg.type == "postgres":
        conn_url = _resolve_postgres_url(mem_cfg)
        if not conn_url:
            return
        try:
            with pooled_postgres_connection(conn_url) as conn:
                with conn.cursor() as cur:
                    cur.execute(_POSTGRES_SCHEMA)
                    cur.execute(_POSTGRES_INDEX)
                    for ddl in _POSTGRES_MIGRATIONS:
                        cur.execute(ddl)
                conn.commit()
        except _NO_DRIVER_ERRORS:
            return


def save_episode(
    mem_cfg,
    project_dir: Path,
    scope_key: str,
    session_id: str,
    event_type: str,
    content: str,
    tags: list[str] | None,
    embedding: list[float] | None,
    importance: int | None = None,
    retention_days: int | None = None,
) -> None:
    """Best-effort append-only INSERT — never UPDATE/UPSERT, unlike shared_memory.py's single row
    per scope_key. Never raises. No-ops for memory.type outside sqlite/postgres. `embedding` may
    be None (the embedding call failed or was skipped) — the row is still stored so structured
    recall (event_type/tags/recency) still works for it; it just won't surface in semantic
    (query=...) ranking. `importance` (1-10, caller's/LLM's own judgment of how significant this
    episode is — see remember_episode) is None when unrated; semantic_search_episodes' scoring
    treats an unrated episode as neutral (5/10) rather than penalizing it to the bottom of the
    ranking. `retention_days` (from EpisodicMemoryConfig.retention_days — a separate config
    object from `mem_cfg`, threaded through as a plain value here) opportunistically prunes rows
    older than that on a small random fraction of calls; None (default) means no pruning."""
    try:
        if mem_cfg.type not in ("sqlite", "postgres"):
            return
        ensure_schema(mem_cfg, project_dir)
        tags_json = json.dumps(tags or [])
        embedding_json = json.dumps(embedding) if embedding is not None else None
        # Clamp rather than trust the caller (an LLM-supplied value) blindly — a rogue 9999
        # would otherwise dominate every future ranking indefinitely.
        clamped_importance = max(1, min(10, importance)) if importance is not None else None

        if mem_cfg.type == "sqlite":
            db_path = project_dir / (mem_cfg.db_path or ".ai/memory.db")
            with sqlite3.connect(str(db_path), timeout=15.0) as conn:
                conn.execute(
                    "INSERT INTO episodes "
                    "(scope_key, session_id, event_type, content, tags, embedding, importance) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (scope_key, session_id, event_type, content, tags_json, embedding_json, clamped_importance),
                )
                conn.commit()
        else:
            conn_url = _resolve_postgres_url(mem_cfg)
            if not conn_url:
                return
            with pooled_postgres_connection(conn_url) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "INSERT INTO episodes "
                        "(scope_key, session_id, event_type, content, tags, embedding, importance) "
                        "VALUES (%s, %s, %s, %s, %s::jsonb, %s::jsonb, %s)",
                        (scope_key, session_id, event_type, content, tags_json, embedding_json, clamped_importance),
                    )
                conn.commit()

        _maybe_prune_old_episodes(mem_cfg, project_dir, retention_days)
    except Exception as e:
        Tracer.log_error(f"Episodic Memory Write Error: {e}")


def _maybe_prune_old_episodes(mem_cfg, project_dir: Path, retention_days: int | None) -> None:
    """Opportunistically deletes episodes rows older than retention_days, on a small random
    fraction of writes (see run_logger._PRUNE_PROBABILITY, same rate) — keeps the append-only
    table bounded over a long production lifetime without an external cron job. Table-wide, not
    scoped to one scope_key: retention is a single project-wide policy, not per-scope."""
    if not retention_days or random.random() >= _PRUNE_PROBABILITY:
        return
    cutoff = datetime.now(UTC) - timedelta(days=retention_days)

    if mem_cfg.type == "sqlite":
        db_path = project_dir / (mem_cfg.db_path or ".ai/memory.db")
        with sqlite3.connect(str(db_path), timeout=15.0) as conn:
            conn.execute(
                "DELETE FROM episodes WHERE created_at < ?",
                (cutoff.strftime("%Y-%m-%d %H:%M:%S"),),
            )
            conn.commit()
    else:
        conn_url = _resolve_postgres_url(mem_cfg)
        if not conn_url:
            return
        try:
            with pooled_postgres_connection(conn_url) as conn:
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM episodes WHERE created_at < %s", (cutoff,))
                conn.commit()
        except _NO_DRIVER_ERRORS:
            return


def _fetch_rows(
    mem_cfg,
    project_dir: Path,
    scope_key: str,
    event_type: str | None,
    cap: int,
    include_embedding: bool,
) -> list[dict[str, Any]]:
    """Shared SELECT helper for query_episodes/semantic_search_episodes. include_embedding=False
    skips selecting/parsing the embedding column entirely, since cheap structured lookups
    shouldn't pay for it. Ordered by created_at DESC (most recent first). Raises on failure —
    callers (query_episodes/semantic_search_episodes) catch and return a safe default; this
    helper itself is not part of the public best-effort contract."""
    if mem_cfg.type not in ("sqlite", "postgres"):
        return []
    ensure_schema(mem_cfg, project_dir)
    columns = "id, session_id, event_type, content, tags, importance, created_at" + (
        ", embedding" if include_embedding else ""
    )

    if mem_cfg.type == "sqlite":
        db_path = project_dir / (mem_cfg.db_path or ".ai/memory.db")
        with sqlite3.connect(str(db_path), timeout=15.0) as conn:
            conn.row_factory = sqlite3.Row
            if event_type:
                cursor = conn.execute(
                    f"SELECT {columns} FROM episodes WHERE scope_key = ? AND event_type = ? "
                    "ORDER BY created_at DESC LIMIT ?",
                    (scope_key, event_type, cap),
                )
            else:
                cursor = conn.execute(
                    f"SELECT {columns} FROM episodes WHERE scope_key = ? "
                    "ORDER BY created_at DESC LIMIT ?",
                    (scope_key, cap),
                )
            rows = [dict(row) for row in cursor.fetchall()]
    else:
        conn_url = _resolve_postgres_url(mem_cfg)
        if not conn_url:
            return []
        with pooled_postgres_connection(conn_url) as conn, postgres_dict_cursor(conn) as cur:
            if event_type:
                cur.execute(
                    f"SELECT {columns} FROM episodes WHERE scope_key = %s AND event_type = %s "
                    "ORDER BY created_at DESC LIMIT %s",
                    (scope_key, event_type, cap),
                )
            else:
                cur.execute(
                    f"SELECT {columns} FROM episodes WHERE scope_key = %s "
                    "ORDER BY created_at DESC LIMIT %s",
                    (scope_key, cap),
                )
            rows = [dict(row) for row in cur.fetchall()]

    for row in rows:
        if row.get("tags"):
            row["tags"] = row["tags"] if isinstance(row["tags"], list) else json.loads(row["tags"])
        else:
            row["tags"] = []
        if include_embedding and row.get("embedding"):
            row["embedding"] = (
                row["embedding"] if isinstance(row["embedding"], list) else json.loads(row["embedding"])
            )
        row["created_at"] = str(row.get("created_at"))
    return rows


def query_episodes(
    mem_cfg,
    project_dir: Path,
    scope_key: str,
    event_type: str | None,
    tags: list[str] | None,
    limit: int,
) -> list[dict[str, Any]]:
    """Non-semantic filter/recency lookup — no embedding call, works standalone. `tags` (if
    given) is an AND filter: a row must contain every requested tag, not just one — applied in
    Python after fetch, since tags is stored as JSON text/JSONB, not portably queryable with a
    single SQL clause across both sqlite and postgres. Best-effort: returns [] (never raises) on
    any failure or if memory.type isn't sqlite/postgres."""
    try:
        rows = _fetch_rows(
            mem_cfg, project_dir, scope_key, event_type,
            cap=max(limit * 5, limit), include_embedding=False,
        )
        if tags:
            rows = [r for r in rows if set(tags).issubset(set(r["tags"]))]
        return rows[:limit]
    except Exception as e:
        Tracer.log_error(f"Episodic Memory Read Error: {e}")
        return []


async def embed_text(embedding_model: str, text: str) -> list[float] | None:
    """Single-string wrapper around litellm.aembedding — same call shape rag.py's own query
    embedding uses. Returns None (never raises) on any embedding-API failure, so a broken/
    misconfigured embedding provider degrades remember_episode to "store without a vector" and
    recall_episodes to "structured-only" rather than failing the tool call."""
    try:
        resp = await litellm.aembedding(model=embedding_model, input=[text])
        return resp.data[0]["embedding"]
    except Exception as e:
        Tracer.log_error(f"Episodic Memory Embedding Error: {e}")
        return None


# Per-day exponential decay applied to an episode's age for the recency component of
# semantic_search_episodes' ranking (see _recency_score) — gentle on purpose: this framework's
# episodes are meant to matter for a production assistant's whole retention window (days to
# months via EpisodicMemoryConfig.retention_days), not a simulated single day the way Park et
# al.'s original Generative Agents decay (0.99 per *hour*) assumes. At this rate a month-old
# episode still retains ~86% of its recency score; a year-old one ~16%.
_RECENCY_DECAY_PER_DAY = 0.995

# importance/10 substitute for an episode that was never rated (remember_episode's importance
# argument is optional) — a neutral midpoint so an unrated episode is neither penalized to the
# bottom of the ranking nor artificially boosted to the top by its absence.
_DEFAULT_IMPORTANCE_SCORE = 0.5


def _recency_score(created_at: str) -> float:
    """Exponential recency decay from an episode's stored created_at down to [0, 1] (1.0 = just
    now). Never raises — an unparseable timestamp (should not happen from this module's own
    writes, but a hand-edited or foreign row is possible) scores as maximally old (0.0) rather
    than crashing the whole ranking."""
    try:
        parsed = datetime.fromisoformat(str(created_at))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        age_days = max((datetime.now(UTC) - parsed).total_seconds() / 86400, 0.0)
        return _RECENCY_DECAY_PER_DAY**age_days
    except (ValueError, TypeError):
        return 0.0


async def semantic_search_episodes(
    mem_cfg,
    project_dir: Path,
    scope_key: str,
    embedding_model: str,
    query: str,
    event_type: str | None,
    tags: list[str] | None,
    limit: int,
) -> list[dict[str, Any]]:
    """Embeds `query`, fetches up to _SEMANTIC_CANDIDATE_CAP scope/type/tag-filtered candidate
    rows WITH their stored embeddings, and ranks the ones that have a non-null embedding by a
    blend of three equally-weighted [0, 1] signals — relevance (cosine similarity to `query`,
    clamped at 0 so an unrelated episode can't drag the blend negative), recency (_recency_score,
    exponential decay from created_at), and importance (the episode's own remember_episode rating,
    or _DEFAULT_IMPORTANCE_SCORE when unrated) — the same three-factor design Park et al.'s
    "Generative Agents" memory stream uses, so a technically-closer semantic match from months ago
    doesn't automatically outrank something more recent and just as relevant. Falls back to
    query_episodes' plain recency order for `limit` rows if the query-embedding call fails,
    rather than returning nothing. Never raises."""
    try:
        query_embedding = await embed_text(embedding_model, query)
        if query_embedding is None:
            return query_episodes(mem_cfg, project_dir, scope_key, event_type, tags, limit)

        rows = _fetch_rows(
            mem_cfg, project_dir, scope_key, event_type,
            cap=_SEMANTIC_CANDIDATE_CAP, include_embedding=True,
        )
        if tags:
            rows = [r for r in rows if set(tags).issubset(set(r["tags"]))]

        def combined_score(r: dict) -> float:
            relevance = max(cosine_similarity(query_embedding, r["embedding"]), 0.0)
            recency = _recency_score(r.get("created_at"))
            importance = r.get("importance")
            importance_score = importance / 10.0 if importance is not None else _DEFAULT_IMPORTANCE_SCORE
            return (relevance + recency + importance_score) / 3.0

        scored = [(combined_score(r), r) for r in rows if r.get("embedding")]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [r for _score, r in scored[:limit]]
    except Exception as e:
        Tracer.log_error(f"Episodic Memory Semantic Search Error: {e}")
        return []
