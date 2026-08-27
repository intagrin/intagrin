"""Best-effort per-call audit log for API-triggered agent runs (/chat, /chat/stream, /resume,
/stream), used to power the Monitor dashboard's "Logs" page. Mirrors runtime/memory.py's style —
raw sqlite3/psycopg2, no ORM, self-managing schema — and lives in the exact same database as the
`checkpoints` table (same sqlite file, or the same Postgres connection) so the existing Alembic
auto-migration (db_migrations/auto_migrate.py, already run at both server processes' startup)
picks up `run_logs` with no new wiring.

Scoped to `memory.type` in ("sqlite", "postgres") only — the same scope auto_migrate.py already
uses, for the same reason: no natural place to persist this otherwise.
"""
import os
import sqlite3
from pathlib import Path
from typing import Any

from ..tracing.console import Tracer
from .memory import postgres_connect

_SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS run_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,
    endpoint TEXT,
    agent TEXT,
    status TEXT,
    error TEXT,
    tokens_delta INTEGER,
    cost_delta REAL,
    total_tokens INTEGER,
    total_cost REAL,
    message_count INTEGER,
    latency_ms INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
"""

_SQLITE_INDEX = (
    "CREATE INDEX IF NOT EXISTS ix_run_logs_session_id_created_at "
    "ON run_logs(session_id, created_at)"
)

_POSTGRES_SCHEMA = """
CREATE TABLE IF NOT EXISTS run_logs (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(255),
    endpoint TEXT,
    agent TEXT,
    status TEXT,
    error TEXT,
    tokens_delta INTEGER,
    cost_delta REAL,
    total_tokens INTEGER,
    total_cost REAL,
    message_count INTEGER,
    latency_ms INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
)
"""

_POSTGRES_INDEX = (
    "CREATE INDEX IF NOT EXISTS ix_run_logs_session_id_created_at "
    "ON run_logs(session_id, created_at)"
)

_COLUMNS = (
    "session_id",
    "endpoint",
    "agent",
    "status",
    "error",
    "tokens_delta",
    "cost_delta",
    "total_tokens",
    "total_cost",
    "message_count",
    "latency_ms",
)


def _resolve_postgres_url(mem_cfg) -> str | None:
    conn_url = getattr(mem_cfg, "connection_url", None)
    if not conn_url and getattr(mem_cfg, "env_var", None):
        conn_url = os.environ.get(mem_cfg.env_var)
    if not conn_url:
        conn_url = os.environ.get("DATABASE_URL") or os.environ.get("POSTGRES_URL")
    return conn_url


def ensure_schema(mem_cfg, project_dir: Path) -> None:
    """Creates the run_logs table if it doesn't already exist. Called both by record_run_log
    (before every write) and by the /api/logs endpoint (before every read) — a fresh project with
    zero logged runs yet should return an empty list, not a 500 from a missing table."""
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


def record_run_log(mem_cfg, project_dir: Path, **fields: Any) -> None:
    """Writes one row per API-triggered run. Entirely best-effort — never raises, since a logging
    failure must not break the actual API response the caller is waiting on. No-ops silently for
    any memory.type other than sqlite/postgres."""
    try:
        if mem_cfg.type not in ("sqlite", "postgres"):
            return

        ensure_schema(mem_cfg, project_dir)
        values = tuple(fields.get(col) for col in _COLUMNS)

        if mem_cfg.type == "sqlite":
            db_path = project_dir / (mem_cfg.db_path or ".ai/memory.db")
            with sqlite3.connect(str(db_path), timeout=15.0) as conn:
                conn.execute(
                    f"INSERT INTO run_logs ({', '.join(_COLUMNS)}) "
                    f"VALUES ({', '.join('?' for _ in _COLUMNS)})",
                    values,
                )
                conn.commit()
        else:
            conn_url = _resolve_postgres_url(mem_cfg)
            if not conn_url:
                return
            try:
                conn = postgres_connect(conn_url)
            except ImportError:
                return
            with conn:
                with conn.cursor() as cur:
                    cur.execute(
                        f"INSERT INTO run_logs ({', '.join(_COLUMNS)}) "
                        f"VALUES ({', '.join('%s' for _ in _COLUMNS)})",
                        values,
                    )
                conn.commit()
    except Exception as e:
        Tracer.log_error(f"Run Log Write Error: {e}")
