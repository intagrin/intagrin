"""DB-backed reviewer credentials for approving `requires_approval` tool calls via /resume.

Before this, the only way to configure an X-Approver-Key secret was `server.auth.approver_env_var`
/ `approvers` in ai.yaml — each one a plaintext value in an environment variable (typically a
checked-in-adjacent .env file). That's fine for local development, but wrong for a real deployment:
rotating or revoking a reviewer's credential means editing .env and restarting the process, there's
no record of *when* a credential was issued or revoked, and the secret sits in plaintext wherever
the environment is inspectable (process env, container spec, CI logs).

This module adds an alternative: approver credentials stored (hashed, salted, never plaintext) in
the same database the project already uses for `checkpoints`/`run_logs` (mirrors run_logger.py's
style exactly — raw sqlite3/psycopg, self-managing schema, no ORM). Managed via `inta approvers
add/rotate/revoke/list` (cli.py) instead of an ai.yaml/`.env` edit, so credentials can be issued and
revoked without a redeploy. `identify_approver` (server/api.py) checks this table first, then falls
back to the existing env-var candidates — so a project using only `approver_env_var` keeps working
unchanged, and both mechanisms can be used side by side (e.g. a shared env-var default plus
individually-issued DB-backed reviewer credentials).

Scoped to `memory.type` in ("sqlite", "postgres") only, same as run_logger.py/rate_limiter.py — no
natural place to persist this otherwise.
"""
import hashlib
import os
import secrets
import sqlite3
from pathlib import Path

from ..errors import IntaGrinError
from ..tracing.console import Tracer
from .memory import pooled_postgres_connection, postgres_dict_cursor
from .run_logger import _resolve_postgres_url

# See run_logger.py's identical _NO_DRIVER_ERRORS for why both are caught: pooled_postgres_connection
# raises IntaGrinError (IG-RT-004) when neither psycopg driver is installed, while a bare psycopg2
# import failure surfaces as ImportError before that wrapping ever happens.
_NO_DRIVER_ERRORS = (ImportError, IntaGrinError)

_SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS approvers (
    approver_id TEXT PRIMARY KEY,
    salt TEXT NOT NULL,
    secret_hash TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    revoked_at TIMESTAMP
)
"""

_POSTGRES_SCHEMA = """
CREATE TABLE IF NOT EXISTS approvers (
    approver_id VARCHAR(255) PRIMARY KEY,
    salt TEXT NOT NULL,
    secret_hash TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    revoked_at TIMESTAMP WITH TIME ZONE
)
"""

# scrypt (stdlib hashlib, no new dependency) with parameters conservative enough to run per
# /resume call without adding noticeable latency, but not so cheap it's a plain fast hash.
_SCRYPT_N, _SCRYPT_R, _SCRYPT_P, _SCRYPT_DKLEN = 2**14, 8, 1, 32


def _hash_secret(secret: str, salt: bytes) -> str:
    return hashlib.scrypt(
        secret.encode("utf-8"), salt=salt, n=_SCRYPT_N, r=_SCRYPT_R, p=_SCRYPT_P, dklen=_SCRYPT_DKLEN
    ).hex()


def _sqlite_path(mem_cfg, project_dir: Path) -> Path:
    return project_dir / (mem_cfg.db_path or ".ai/memory.db")


def ensure_schema(mem_cfg, project_dir: Path) -> None:
    if mem_cfg.type == "sqlite":
        db_path = _sqlite_path(mem_cfg, project_dir)
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


def add_approver(mem_cfg, project_dir: Path, approver_id: str, secret: str) -> None:
    """Issues (or rotates, if approver_id already exists) a DB-backed reviewer credential.
    Raises on any DB error — unlike the best-effort logging modules, a CLI operator issuing a
    credential needs to know immediately if it didn't actually get stored."""
    ensure_schema(mem_cfg, project_dir)
    salt = os.urandom(16)
    secret_hash = _hash_secret(secret, salt)

    if mem_cfg.type == "sqlite":
        db_path = _sqlite_path(mem_cfg, project_dir)
        with sqlite3.connect(str(db_path), timeout=15.0) as conn:
            conn.execute(
                "INSERT INTO approvers (approver_id, salt, secret_hash, revoked_at) "
                "VALUES (?, ?, ?, NULL) "
                "ON CONFLICT(approver_id) DO UPDATE SET "
                "salt=excluded.salt, secret_hash=excluded.secret_hash, revoked_at=NULL",
                (approver_id, salt.hex(), secret_hash),
            )
            conn.commit()
    elif mem_cfg.type == "postgres":
        conn_url = _resolve_postgres_url(mem_cfg)
        if not conn_url:
            raise RuntimeError("No Postgres connection URL found in config or environment.")
        with pooled_postgres_connection(conn_url) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO approvers (approver_id, salt, secret_hash, revoked_at) "
                    "VALUES (%s, %s, %s, NULL) "
                    "ON CONFLICT (approver_id) DO UPDATE SET "
                    "salt=EXCLUDED.salt, secret_hash=EXCLUDED.secret_hash, revoked_at=NULL",
                    (approver_id, salt.hex(), secret_hash),
                )
            conn.commit()
    else:
        raise ValueError(
            f"DB-backed approvers are only supported for sqlite/postgres, not '{mem_cfg.type}'."
        )


def revoke_approver(mem_cfg, project_dir: Path, approver_id: str) -> bool:
    """Soft-revokes a credential (kept for audit history, excluded from verify_secret). Returns
    whether a row was actually found and revoked."""
    ensure_schema(mem_cfg, project_dir)
    if mem_cfg.type == "sqlite":
        db_path = _sqlite_path(mem_cfg, project_dir)
        with sqlite3.connect(str(db_path), timeout=15.0) as conn:
            cur = conn.execute(
                "UPDATE approvers SET revoked_at = CURRENT_TIMESTAMP "
                "WHERE approver_id = ? AND revoked_at IS NULL",
                (approver_id,),
            )
            conn.commit()
            return cur.rowcount > 0
    elif mem_cfg.type == "postgres":
        conn_url = _resolve_postgres_url(mem_cfg)
        if not conn_url:
            return False
        with pooled_postgres_connection(conn_url) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE approvers SET revoked_at = CURRENT_TIMESTAMP "
                    "WHERE approver_id = %s AND revoked_at IS NULL",
                    (approver_id,),
                )
                affected = cur.rowcount
            conn.commit()
            return affected > 0
    return False


def _active_credentials(mem_cfg, project_dir: Path, approver_id: str | None = None) -> list[dict]:
    """Internal: approver_id + salt + secret_hash for every non-revoked row, or (when
    `approver_id` is given) just that one row — a targeted lookup verify_secret uses to avoid
    scrypt-hashing against every row when the caller already knows which approver is signing in.
    Callers outside this module should use list_approvers (public, secret-free) or verify_secret
    instead."""
    ensure_schema(mem_cfg, project_dir)
    if mem_cfg.type == "sqlite":
        db_path = _sqlite_path(mem_cfg, project_dir)
        with sqlite3.connect(str(db_path), timeout=15.0) as conn:
            conn.row_factory = sqlite3.Row
            query = "SELECT approver_id, salt, secret_hash, created_at FROM approvers WHERE revoked_at IS NULL"
            params = ()
            if approver_id is not None:
                query += " AND approver_id = ?"
                params = (approver_id,)
            return [dict(row) for row in conn.execute(query, params)]
    conn_url = _resolve_postgres_url(mem_cfg)
    if not conn_url:
        return []
    try:
        with pooled_postgres_connection(conn_url) as conn, postgres_dict_cursor(conn) as cur:
            query = "SELECT approver_id, salt, secret_hash, created_at FROM approvers WHERE revoked_at IS NULL"
            params = ()
            if approver_id is not None:
                query += " AND approver_id = %s"
                params = (approver_id,)
            cur.execute(query, params)
            return [dict(r) for r in cur.fetchall()]
    except _NO_DRIVER_ERRORS:
        return []


def list_approvers(mem_cfg, project_dir: Path) -> list[dict]:
    """Every approver, active or revoked — id + issuance/revocation status only, never secrets."""
    if mem_cfg.type not in ("sqlite", "postgres"):
        return []
    ensure_schema(mem_cfg, project_dir)
    if mem_cfg.type == "sqlite":
        db_path = _sqlite_path(mem_cfg, project_dir)
        with sqlite3.connect(str(db_path), timeout=15.0) as conn:
            conn.row_factory = sqlite3.Row
            return [
                dict(row)
                for row in conn.execute(
                    "SELECT approver_id, created_at, revoked_at FROM approvers ORDER BY created_at DESC"
                )
            ]
    conn_url = _resolve_postgres_url(mem_cfg)
    if not conn_url:
        return []
    try:
        with pooled_postgres_connection(conn_url) as conn, postgres_dict_cursor(conn) as cur:
            cur.execute(
                "SELECT approver_id, created_at, revoked_at FROM approvers ORDER BY created_at DESC"
            )
            return [dict(r) for r in cur.fetchall()]
    except _NO_DRIVER_ERRORS:
        return []


def verify_secret(
    mem_cfg, project_dir: Path, provided_secret: str, approver_id_hint: str | None = None
) -> str | None:
    """Returns the matching approver_id for a provided X-Approver-Key, or None if it matches no
    active (non-revoked) DB-backed credential. Best-effort: any DB error is logged and treated as
    "no match" rather than raised, so a project with no `approvers` table yet (or a DB hiccup)
    falls through to identify_approver's existing env-var check instead of 500ing the whole
    /resume call.

    `approver_id_hint` (identify_approver's optional X-Approver-Id header) narrows the scrypt
    check to that one row instead of every active approver — without it, each call runs the
    deliberately expensive KDF once per active approver, which scales linearly (and expensively)
    with roster size. A wrong or absent hint just falls back to a full scan; it's a performance
    hint, not a trust boundary — the scrypt+salt comparison is still what actually authenticates."""
    if not provided_secret or mem_cfg.type not in ("sqlite", "postgres"):
        return None
    try:
        for row in _active_credentials(mem_cfg, project_dir, approver_id=approver_id_hint):
            candidate_hash = _hash_secret(provided_secret, bytes.fromhex(row["salt"]))
            if secrets.compare_digest(candidate_hash, row["secret_hash"]):
                return row["approver_id"]
    except Exception as e:
        Tracer.log_error(f"Approver Verify Error: {e}")
    return None
