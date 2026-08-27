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
import sqlite3
from pathlib import Path
from typing import Any

import litellm

from ..tracing.console import Tracer
from .memory import postgres_connect, postgres_dict_cursor
from .rag import cosine_similarity
from .run_logger import _resolve_postgres_url

_SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS episodes (
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
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
)
"""

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
            conn.commit()
    elif mem_cfg.type == "postgres":
        conn_url = _resolve_postgres_url(mem_cfg)
        if not conn_url:
            return
        try:
            conn = postgres_connect(conn_url)
        except ImportError:
            return
        with conn:
            with conn.cursor() as cur:
                cur.execute(_POSTGRES_SCHEMA)
                cur.execute(_POSTGRES_INDEX)
            conn.commit()


def save_episode(
    mem_cfg,
    project_dir: Path,
    scope_key: str,
    session_id: str,
    event_type: str,
    content: str,
    tags: list[str] | None,
    embedding: list[float] | None,
) -> None:
    """Best-effort append-only INSERT — never UPDATE/UPSERT, unlike shared_memory.py's single row
    per scope_key. Never raises. No-ops for memory.type outside sqlite/postgres. `embedding` may
    be None (the embedding call failed or was skipped) — the row is still stored so structured
    recall (event_type/tags/recency) still works for it; it just won't surface in semantic
    (query=...) ranking."""
    try:
        if mem_cfg.type not in ("sqlite", "postgres"):
            return
        ensure_schema(mem_cfg, project_dir)
        tags_json = json.dumps(tags or [])
        embedding_json = json.dumps(embedding) if embedding is not None else None

        if mem_cfg.type == "sqlite":
            db_path = project_dir / (mem_cfg.db_path or ".ai/memory.db")
            with sqlite3.connect(str(db_path), timeout=15.0) as conn:
                conn.execute(
                    "INSERT INTO episodes "
                    "(scope_key, session_id, event_type, content, tags, embedding) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (scope_key, session_id, event_type, content, tags_json, embedding_json),
                )
                conn.commit()
            return

        conn_url = _resolve_postgres_url(mem_cfg)
        if not conn_url:
            return
        conn = postgres_connect(conn_url)
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO episodes "
                    "(scope_key, session_id, event_type, content, tags, embedding) "
                    "VALUES (%s, %s, %s, %s, %s::jsonb, %s::jsonb)",
                    (scope_key, session_id, event_type, content, tags_json, embedding_json),
                )
            conn.commit()
    except Exception as e:
        Tracer.log_error(f"Episodic Memory Write Error: {e}")


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
    columns = "id, session_id, event_type, content, tags, created_at" + (
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
        conn = postgres_connect(conn_url)
        with conn, postgres_dict_cursor(conn) as cur:
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
    rows WITH their stored embeddings, cosine-ranks the ones that have a non-null embedding, and
    returns the top `limit`. Falls back to query_episodes' plain recency order for `limit` rows
    if the query-embedding call fails, rather than returning nothing. Never raises."""
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

        scored = [
            (cosine_similarity(query_embedding, r["embedding"]), r)
            for r in rows
            if r.get("embedding")
        ]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [r for _score, r in scored[:limit]]
    except Exception as e:
        Tracer.log_error(f"Episodic Memory Semantic Search Error: {e}")
        return []
