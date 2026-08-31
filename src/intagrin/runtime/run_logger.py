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
import random
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from ..errors import IntaGrinError
from ..tracing.console import Tracer
from .memory import pooled_postgres_connection

# What fraction of record_run_log calls also run a pruning pass when memory.run_log_retention_days
# is set — opportunistic instead of an external cron job, and rare enough that the extra indexed
# DELETE's cost is negligible when amortized over every write on this hot path.
_PRUNE_PROBABILITY = 1 / 200

# _get_or_create_pg_pool (behind pooled_postgres_connection) raises IntaGrinError("IG-RT-004",
# ...) instead of a plain ImportError when neither psycopg driver is installed — caught
# alongside ImportError everywhere this module used to only expect the latter, so a missing
# driver still no-ops (or, in record_run_log, is quietly absorbed by its own outer catch-all)
# exactly as before, rather than surfacing as a differently-shaped error.
_NO_DRIVER_ERRORS = (ImportError, IntaGrinError)

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
            with pooled_postgres_connection(conn_url) as conn:
                with conn.cursor() as cur:
                    cur.execute(_POSTGRES_SCHEMA)
                    cur.execute(_POSTGRES_INDEX)
                conn.commit()
        except _NO_DRIVER_ERRORS:
            return


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
                with pooled_postgres_connection(conn_url) as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            f"INSERT INTO run_logs ({', '.join(_COLUMNS)}) "
                            f"VALUES ({', '.join('%s' for _ in _COLUMNS)})",
                            values,
                        )
                    conn.commit()
            except _NO_DRIVER_ERRORS:
                return

        _maybe_prune_old_rows(mem_cfg, project_dir)
    except Exception as e:
        Tracer.log_error(f"Run Log Write Error: {e}")


def _maybe_prune_old_rows(mem_cfg, project_dir: Path) -> None:
    """Opportunistically deletes run_logs rows older than memory.run_log_retention_days, on a
    small random fraction of writes (see _PRUNE_PROBABILITY) — keeps the table bounded over a
    long production lifetime without needing an external cron job. No-ops when retention isn't
    configured (the default, via getattr since some callers pass a bare mem_cfg without the
    field) or on this call's random miss. Failures propagate to record_run_log's own best-effort
    outer catch — a failed prune must never break the write that just succeeded."""
    retention_days = getattr(mem_cfg, "run_log_retention_days", None)
    if not retention_days or random.random() >= _PRUNE_PROBABILITY:
        return
    cutoff = datetime.now(UTC) - timedelta(days=retention_days)

    if mem_cfg.type == "sqlite":
        db_path = project_dir / (mem_cfg.db_path or ".ai/memory.db")
        with sqlite3.connect(str(db_path), timeout=15.0) as conn:
            conn.execute(
                "DELETE FROM run_logs WHERE created_at < ?",
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
                    cur.execute("DELETE FROM run_logs WHERE created_at < %s", (cutoff,))
                conn.commit()
        except _NO_DRIVER_ERRORS:
            return
