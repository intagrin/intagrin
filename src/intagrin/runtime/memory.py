import json
import sqlite3
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from ..errors import IntaGrinError
from ..tracing.console import Tracer


class SQLiteCheckpointer:
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path, timeout=15.0) as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            # Check if old sessions table exists
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='sessions'"
            )
            has_old_table = cursor.fetchone() is not None

            conn.execute("""
                CREATE TABLE IF NOT EXISTS checkpoints (
                    session_id TEXT PRIMARY KEY,
                    messages TEXT,
                    state TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            if has_old_table:
                # Migrate old data if checkpoints is empty
                cursor = conn.execute("SELECT COUNT(*) FROM checkpoints")
                if cursor.fetchone()[0] == 0:
                    conn.execute("""
                        INSERT INTO checkpoints (session_id, messages, state)
                        SELECT session_id, messages, state FROM sessions
                    """)

            conn.commit()

    def save_checkpoint(
        self, session_id: str, messages: list[dict[str, Any]], state: dict[str, Any]
    ):
        with sqlite3.connect(self.db_path, timeout=15.0) as conn:
            conn.execute(
                """
                INSERT INTO checkpoints (session_id, messages, state, updated_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(session_id) DO UPDATE SET
                    messages=excluded.messages,
                    state=excluded.state,
                    updated_at=CURRENT_TIMESTAMP
            """,
                (session_id, json.dumps(messages), json.dumps(state)),
            )
            conn.commit()

    def load_checkpoint(
        self, session_id: str
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        with sqlite3.connect(self.db_path, timeout=15.0) as conn:
            # First check if the old sessions table exists and migrate it if needed
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='checkpoints'"
            )
            if not cursor.fetchone():
                self._init_db()

            cursor = conn.execute(
                "SELECT messages, state FROM checkpoints WHERE session_id = ?",
                (session_id,),
            )
            row = cursor.fetchone()
            if row:
                return json.loads(row[0]), json.loads(row[1])
            return [], {}

    def list_sessions(
        self, prefix: str | None = None, since: datetime | None = None, limit: int = 200
    ) -> list[str]:
        """Lists session_ids for this project, most recently updated first. Used by anything that
        needs to enumerate historical sessions rather than load one by id (e.g. `inta simulate`)."""
        query = "SELECT session_id FROM checkpoints WHERE 1=1"
        params: list = []
        if prefix:
            query += " AND session_id LIKE ?"
            params.append(f"{prefix}%")
        if since:
            query += " AND updated_at >= ?"
            params.append(since.strftime("%Y-%m-%d %H:%M:%S"))
        query += " ORDER BY updated_at DESC LIMIT ?"
        params.append(limit)
        with sqlite3.connect(self.db_path, timeout=15.0) as conn:
            cursor = conn.execute(query, params)
            return [row[0] for row in cursor.fetchall()]


def resolve_postgres_sqlalchemy_url(connection_url: str) -> str:
    """SQLAlchemy's default `postgresql://` scheme maps to the psycopg2 driver. This project's
    'postgres' extra installs psycopg (v3) instead, so a bare `postgresql://` URL fails with
    "No module named 'psycopg2'" for any SQLAlchemy consumer (currently: Alembic's auto-migrate)
    even though PostgresCheckpointer itself works fine, since it tries psycopg3 directly. Rewrite
    to the explicit `postgresql+psycopg://` dialect when only psycopg3 is installed, mirroring the
    same psycopg3-then-psycopg2 preference order as PostgresCheckpointer._init_pool."""
    if not connection_url.startswith(("postgresql://", "postgres://")):
        return connection_url
    try:
        import psycopg  # noqa: F401

        return "postgresql+psycopg://" + connection_url.split("://", 1)[1]
    except ImportError:
        return connection_url


def postgres_connect(connection_url: str):
    """Raw (non-pooled) Postgres connection for the small one-off queries outside
    PostgresCheckpointer's pool (run_logger.py's per-request audit-log writes, monitor.py's Logs
    page reads). Tries psycopg (v3) first, then psycopg2 — same preference order as
    PostgresCheckpointer._init_pool — so callers work under either driver instead of hard-depending
    on psycopg2 specifically, which this project's 'postgres' extra doesn't install. Returns a
    connection usable as a context manager (`with postgres_connect(url) as conn:`) under both
    drivers. Raises ImportError if neither is installed — callers already catch this."""
    try:
        import psycopg

        return psycopg.connect(connection_url)
    except ImportError:
        import psycopg2

        return psycopg2.connect(connection_url)


def postgres_dict_cursor(conn):
    """A dict-row cursor for a connection from postgres_connect(), regardless of which driver
    produced it — psycopg (v3) uses a row_factory, psycopg2 uses cursor_factory=RealDictCursor.
    Uses the same "does psycopg import" test postgres_connect used to build conn, so the dispatch
    always matches which driver actually opened the connection."""
    try:
        from psycopg.rows import dict_row

        return conn.cursor(row_factory=dict_row)
    except ImportError:
        from psycopg2.extras import RealDictCursor

        return conn.cursor(cursor_factory=RealDictCursor)


_pg_pools = {}
_pg_pools_lock = threading.Lock()


class PostgresCheckpointer:
    """Enterprise PostgreSQL session checkpointer using connection pooling."""

    def __init__(self, connection_url: str):
        self.connection_url = connection_url
        self.use_psycopg3 = False
        self.pool = None
        self._init_pool()
        self._init_db()

    def _init_pool(self):
        # Locked so two RuntimeEngine instances constructed concurrently (e.g. two
        # simultaneous /chat requests for the same project) can't both miss the cache and
        # each create their own pool for the same connection_url.
        with _pg_pools_lock:
            if self.connection_url in _pg_pools:
                self.pool, self.use_psycopg3 = _pg_pools[self.connection_url]
                return

            try:
                import psycopg  # noqa: F401 -- probes for the psycopg3 extra being installed
                from psycopg_pool import ConnectionPool

                self.pool = ConnectionPool(self.connection_url, min_size=2, max_size=20)
                self.use_psycopg3 = True
            except ImportError:
                try:
                    import psycopg2  # noqa: F401 -- probes for the psycopg2 fallback being installed
                    from psycopg2.pool import ThreadedConnectionPool

                    self.pool = ThreadedConnectionPool(2, 20, self.connection_url)
                    self.use_psycopg3 = False
                except ImportError:
                    raise IntaGrinError(
                        "IG-RT-004", "PostgreSQL requires 'psycopg[pool]' or 'psycopg2'."
                    )

            _pg_pools[self.connection_url] = (self.pool, self.use_psycopg3)

    def _execute(
        self,
        query: str,
        params: tuple = None,
        fetch_one: bool = False,
        fetch_all: bool = False,
        commit: bool = False,
    ):
        if self.use_psycopg3:
            with self.pool.connection() as conn, conn.cursor() as cur:
                cur.execute(query, params)
                if commit:
                    conn.commit()
                if fetch_all:
                    return cur.fetchall()
                if fetch_one:
                    return cur.fetchone()
        else:
            conn = self.pool.getconn()
            try:
                with conn.cursor() as cur:
                    cur.execute(query, params)
                    if commit:
                        conn.commit()
                    else:
                        conn.rollback()  # Release transaction snapshot for read queries
                    if fetch_all:
                        res = cur.fetchall()
                    elif fetch_one:
                        res = cur.fetchone()
                    else:
                        res = None
                return res
            finally:
                self.pool.putconn(conn)

    def _init_db(self):
        self._execute(
            """
            CREATE TABLE IF NOT EXISTS checkpoints (
                session_id VARCHAR(255) PRIMARY KEY,
                messages JSONB,
                state JSONB,
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );
        """,
            commit=True,
        )

    def save_checkpoint(
        self, session_id: str, messages: list[dict[str, Any]], state: dict[str, Any]
    ):
        self._execute(
            """
            INSERT INTO checkpoints (session_id, messages, state, updated_at)
            VALUES (%s, %s::jsonb, %s::jsonb, CURRENT_TIMESTAMP)
            ON CONFLICT (session_id) DO UPDATE SET
                messages = EXCLUDED.messages,
                state = EXCLUDED.state,
                updated_at = CURRENT_TIMESTAMP;
        """,
            (session_id, json.dumps(messages), json.dumps(state)),
            commit=True,
        )

    def load_checkpoint(
        self, session_id: str
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        row = self._execute(
            "SELECT messages, state FROM checkpoints WHERE session_id = %s",
            (session_id,),
            fetch_one=True,
        )
        if row:
            msgs = row[0] if isinstance(row[0], list) else json.loads(row[0])
            st = row[1] if isinstance(row[1], dict) else json.loads(row[1])
            return msgs, st
        return [], {}

    def list_sessions(
        self, prefix: str | None = None, since: datetime | None = None, limit: int = 200
    ) -> list[str]:
        """Lists session_ids for this project, most recently updated first."""
        query = "SELECT session_id FROM checkpoints WHERE 1=1"
        params: list = []
        if prefix:
            query += " AND session_id LIKE %s"
            params.append(f"{prefix}%")
        if since:
            query += " AND updated_at >= %s"
            params.append(since)
        query += " ORDER BY updated_at DESC LIMIT %s"
        params.append(limit)
        rows = self._execute(query, tuple(params), fetch_all=True) or []
        return [row[0] for row in rows]


class RedisCheckpointer:
    """High-throughput Redis session checkpointer."""

    def __init__(
        self, connection_url: str, ttl_seconds: int = 604800
    ):  # 7 days default TTL
        self.connection_url = connection_url
        self.ttl = ttl_seconds
        try:
            import redis

            self.client = redis.from_url(self.connection_url, decode_responses=True)
        except ImportError:
            raise IntaGrinError(
                "IG-RT-005", "Redis memory requires 'redis'. Install via: pip install redis"
            )

    def save_checkpoint(
        self, session_id: str, messages: list[dict[str, Any]], state: dict[str, Any]
    ):
        payload = json.dumps(
            {"messages": messages, "state": state, "updated_at": time.time()}
        )
        self.client.set(f"intagrin:session:{session_id}", payload, ex=self.ttl)

    def load_checkpoint(
        self, session_id: str
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        raw = self.client.get(f"intagrin:session:{session_id}")
        if raw:
            data = json.loads(raw)
            return data.get("messages", []), data.get("state", {})
        return [], {}

    def list_sessions(
        self, prefix: str | None = None, since: datetime | None = None, limit: int = 200
    ) -> list[str]:
        """Best-effort: Redis has no queryable updated_at index, so this SCANs matching keys and
        sorts using the `updated_at` timestamp folded into each session's payload by
        save_checkpoint. Also implicitly bounded by this checkpointer's TTL (7 days by default) —
        anything older has already expired and won't appear here regardless of `since`."""
        pattern = f"intagrin:session:{prefix or ''}*"
        since_ts = since.timestamp() if since else None
        candidates = []
        for key in self.client.scan_iter(match=pattern):
            raw = self.client.get(key)
            if not raw:
                continue
            try:
                data = json.loads(raw)
            except Exception:
                continue
            updated_at = data.get("updated_at", 0)
            if since_ts and updated_at < since_ts:
                continue
            session_id = key.split("intagrin:session:", 1)[-1]
            candidates.append((updated_at, session_id))
        candidates.sort(key=lambda c: c[0], reverse=True)
        return [session_id for _, session_id in candidates[:limit]]


class CheckpointerConfigError(IntaGrinError):
    """Raised by build_checkpointer(strict=True) when a project's memory config can't be resolved
    to a real backend — used by CLI tooling (inta replay, inta simulate) where silently falling
    back to a default/local backend could show the wrong history instead of a clear error."""


def build_checkpointer(mem_cfg, project_dir: Path, strict: bool = False):
    """Constructs the checkpointer instance for a project's `memory:` config. Single source of
    truth for the mem_cfg.type branch — used by RuntimeEngine, `inta replay`, and `inta simulate`
    so all three agree on which backend/connection a project's checkpoints live in.

    Non-strict (default, used by RuntimeEngine): missing Postgres credentials fall back to local
    SQLite with a logged warning, missing Redis credentials fall back to `redis://localhost:6379/0`,
    and an unrecognized/unconfigured memory type returns None — an engine should still boot even
    with an imperfect memory config, since chat history is best-effort.

    Strict (used by CLI tooling that reads *existing* history): any of the above instead raises
    CheckpointerConfigError, since guessing a different backend would silently show incorrect or
    empty history rather than a clear error. 'custom' memory types also aren't supported in strict
    mode — CLI tooling has no way to know how to enumerate a user-supplied backend.
    """
    if mem_cfg.type == "sqlite":
        db_path = project_dir / (mem_cfg.db_path or ".ai/memory.db")
        return SQLiteCheckpointer(str(db_path))

    if mem_cfg.type == "postgres":
        import os

        conn_url = mem_cfg.connection_url
        if not conn_url and mem_cfg.env_var:
            conn_url = os.environ.get(mem_cfg.env_var)
        if not conn_url:
            conn_url = os.environ.get("DATABASE_URL") or os.environ.get("POSTGRES_URL")
        if not conn_url:
            if strict:
                raise CheckpointerConfigError(
                    "IG-RT-001",
                    "No connection URL for PostgreSQL memory (set connection_url, env_var, or DATABASE_URL).",
                )
            Tracer.log_error(
                "PostgreSQL memory selected but no connection_url or DATABASE_URL provided. Falling back to SQLite."
            )
            return SQLiteCheckpointer(str(project_dir / ".ai/memory.db"))
        return PostgresCheckpointer(conn_url)

    if mem_cfg.type == "redis":
        import os

        conn_url = mem_cfg.connection_url
        if not conn_url and mem_cfg.env_var:
            conn_url = os.environ.get(mem_cfg.env_var)
        if not conn_url:
            conn_url = os.environ.get("REDIS_URL")
        if not conn_url:
            if strict:
                raise CheckpointerConfigError(
                    "IG-RT-002",
                    "No connection URL for Redis memory (set connection_url, env_var, or REDIS_URL).",
                )
            conn_url = "redis://localhost:6379/0"
        return RedisCheckpointer(conn_url)

    if mem_cfg.type == "custom" and mem_cfg.custom_module and not strict:
        import importlib
        import sys

        if str(project_dir) not in sys.path:
            sys.path.insert(0, str(project_dir))
        mod = importlib.import_module(mem_cfg.custom_module)
        return mod.CustomCheckpointer()

    if strict:
        raise CheckpointerConfigError(
            "IG-RT-003", f"Memory type '{mem_cfg.type}' is not supported for replay/simulate yet."
        )
    return None
