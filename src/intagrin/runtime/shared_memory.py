"""Cross-session / org-level shared memory (see MemoryConfig.shared_scope). Deliberately does not
introduce a new memory subsystem: it persists the exact same long_term_memory summary
RuntimeEngine._compress_memory already produces, just to a scope broader than one session_id, and
merges it back into a session's own long_term_memory on initialize(). Mirrors runtime/run_logger.py's
style — raw sqlite3/psycopg, no ORM, self-managing schema — and lives in the same database as
`checkpoints`/`run_logs` (same sqlite file, or the same Postgres connection).

Scoped to memory.type in ("sqlite", "postgres") only — the same scope run_logger.py itself has.
Last-write-wins: concurrent sessions saving the same scope_key overwrite each other, no
merge/versioning — an explicit, documented limitation (see docs/04_Shared_State_Redux.md), not an
oversight.
"""

import sqlite3
from pathlib import Path

from ..errors import IntaGrinError
from ..tracing.console import Tracer
from .memory import pooled_postgres_connection
from .run_logger import _resolve_postgres_url

# See run_logger.py's identical _NO_DRIVER_ERRORS for why both are caught: pooled_postgres_connection
# raises IntaGrinError, not ImportError, when neither psycopg driver is installed.
_NO_DRIVER_ERRORS = (ImportError, IntaGrinError)

_SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS shared_memory (
    scope_key TEXT PRIMARY KEY,
    content TEXT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
"""

_POSTGRES_SCHEMA = """
CREATE TABLE IF NOT EXISTS shared_memory (
    scope_key VARCHAR(255) PRIMARY KEY,
    content TEXT,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
)
"""


def ensure_schema(mem_cfg, project_dir: Path) -> None:
    """Creates the shared_memory table if it doesn't already exist. Called before every
    load/save, same reasoning as run_logger.ensure_schema — a fresh project must not 500."""
    if mem_cfg.type == "sqlite":
        db_path = project_dir / (mem_cfg.db_path or ".ai/memory.db")
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(db_path), timeout=15.0) as conn:
            conn.execute(_SQLITE_SCHEMA)
            conn.commit()
    elif mem_cfg.type == "postgres":
        conn_url = _resolve_postgres_url(mem_cfg)
        if not conn_url:
            return
        try:
            with pooled_postgres_connection(conn_url) as conn:
                with conn.cursor() as cur:
                    cur.execute(_POSTGRES_SCHEMA)
                conn.commit()
        except _NO_DRIVER_ERRORS:
            return


def load_shared_memory(mem_cfg, project_dir: Path, scope_key: str) -> str | None:
    """Best-effort: returns None (never raises) on any failure or if memory.type isn't
    sqlite/postgres, so a broken shared-memory read never blocks a session from starting."""
    try:
        if mem_cfg.type not in ("sqlite", "postgres"):
            return None
        ensure_schema(mem_cfg, project_dir)

        if mem_cfg.type == "sqlite":
            db_path = project_dir / (mem_cfg.db_path or ".ai/memory.db")
            with sqlite3.connect(str(db_path), timeout=15.0) as conn:
                row = conn.execute(
                    "SELECT content FROM shared_memory WHERE scope_key = ?", (scope_key,)
                ).fetchone()
            return row[0] if row else None

        conn_url = _resolve_postgres_url(mem_cfg)
        if not conn_url:
            return None
        with pooled_postgres_connection(conn_url) as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT content FROM shared_memory WHERE scope_key = %s", (scope_key,)
            )
            row = cur.fetchone()
        return row[0] if row else None
    except Exception as e:
        Tracer.log_error(f"Shared Memory Read Error: {e}")
        return None


def save_shared_memory(mem_cfg, project_dir: Path, scope_key: str, content: str) -> None:
    """Best-effort: never raises, since a shared-memory write failure must not break the
    (already-succeeded) compression turn that produced this content."""
    try:
        if mem_cfg.type not in ("sqlite", "postgres"):
            return
        ensure_schema(mem_cfg, project_dir)

        if mem_cfg.type == "sqlite":
            db_path = project_dir / (mem_cfg.db_path or ".ai/memory.db")
            with sqlite3.connect(str(db_path), timeout=15.0) as conn:
                conn.execute(
                    "INSERT INTO shared_memory (scope_key, content, updated_at) "
                    "VALUES (?, ?, CURRENT_TIMESTAMP) "
                    "ON CONFLICT(scope_key) DO UPDATE SET content = excluded.content, "
                    "updated_at = CURRENT_TIMESTAMP",
                    (scope_key, content),
                )
                conn.commit()
            return

        conn_url = _resolve_postgres_url(mem_cfg)
        if not conn_url:
            return
        with pooled_postgres_connection(conn_url) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO shared_memory (scope_key, content, updated_at) "
                    "VALUES (%s, %s, CURRENT_TIMESTAMP) "
                    "ON CONFLICT (scope_key) DO UPDATE SET content = EXCLUDED.content, "
                    "updated_at = CURRENT_TIMESTAMP",
                    (scope_key, content),
                )
            conn.commit()
    except Exception as e:
        Tracer.log_error(f"Shared Memory Write Error: {e}")
