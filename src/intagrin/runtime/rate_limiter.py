"""Per-caller rate limiting / usage quotas for the API server, enforced by querying the
run_logs audit table (see runtime/run_logger.py) — no new schema. run_logs.session_id already
carries the f"{user_context}:{session_id}" tenant prefix every API endpoint writes (see
server/api.py), so one caller's own usage is just an aggregate query filtered by that prefix.

Scoped to memory.type in ("sqlite", "postgres") only — the same scope run_logger.py itself has,
since there's no other place these rows live. Mirrors run_logger.py's connection-opening style.
"""

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

from ..errors import IntaGrinError
from ..tracing.console import Tracer
from .memory import pooled_postgres_connection
from .run_logger import _resolve_postgres_url, ensure_schema


def _count_since(mem_cfg, project_dir: Path, prefix: str, since: datetime) -> int:
    if mem_cfg.type == "sqlite":
        db_path = project_dir / (mem_cfg.db_path or ".ai/memory.db")
        with sqlite3.connect(str(db_path), timeout=15.0) as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM run_logs WHERE session_id LIKE ? AND created_at >= ?",
                (prefix, since.strftime("%Y-%m-%d %H:%M:%S")),
            ).fetchone()
        return row[0] if row else 0

    conn_url = _resolve_postgres_url(mem_cfg)
    if not conn_url:
        return 0
    with pooled_postgres_connection(conn_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM run_logs WHERE session_id LIKE %s AND created_at >= %s",
                (prefix, since),
            )
            row = cur.fetchone()
    return row[0] if row else 0


def _cost_and_tokens_since(
    mem_cfg, project_dir: Path, prefix: str, since: datetime
) -> tuple[float, int]:
    if mem_cfg.type == "sqlite":
        db_path = project_dir / (mem_cfg.db_path or ".ai/memory.db")
        with sqlite3.connect(str(db_path), timeout=15.0) as conn:
            row = conn.execute(
                "SELECT COALESCE(SUM(cost_delta), 0), COALESCE(SUM(tokens_delta), 0) "
                "FROM run_logs WHERE session_id LIKE ? AND created_at >= ?",
                (prefix, since.strftime("%Y-%m-%d %H:%M:%S")),
            ).fetchone()
        return (row[0] or 0.0, row[1] or 0) if row else (0.0, 0)

    conn_url = _resolve_postgres_url(mem_cfg)
    if not conn_url:
        return (0.0, 0)
    with pooled_postgres_connection(conn_url) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT COALESCE(SUM(cost_delta), 0), COALESCE(SUM(tokens_delta), 0) "
            "FROM run_logs WHERE session_id LIKE %s AND created_at >= %s",
            (prefix, since),
        )
        row = cur.fetchone()
    return (row[0] or 0.0, row[1] or 0) if row else (0.0, 0)


def check_rate_limit(mem_cfg, project_dir: Path, user_context: str, rate_cfg) -> None:
    """Raises IntaGrinError("IG-RT-008", ..., http_status=429) if user_context has exceeded any
    configured server.rate_limit threshold. No-ops entirely when every threshold is unconfigured
    (the default) or memory.type isn't sqlite/postgres. Fails open (allows the request, logs the
    error) only on infra errors — an unreachable audit DB must not itself become an outage; that
    posture mirrors run_logger.record_run_log's own best-effort philosophy."""
    if mem_cfg.type not in ("sqlite", "postgres"):
        return
    if (
        rate_cfg.max_requests_per_window is None
        and rate_cfg.max_cost_per_caller_per_day is None
        and rate_cfg.max_tokens_per_caller_per_day is None
    ):
        return

    try:
        ensure_schema(mem_cfg, project_dir)
        prefix = f"{user_context}:%"

        if rate_cfg.max_requests_per_window is not None:
            window_start = datetime.now(UTC) - timedelta(seconds=rate_cfg.window_seconds)
            count = _count_since(mem_cfg, project_dir, prefix, window_start)
            if count >= rate_cfg.max_requests_per_window:
                raise IntaGrinError(
                    "IG-RT-008",
                    f"Rate limit exceeded: {count}/{rate_cfg.max_requests_per_window} requests "
                    f"in the last {rate_cfg.window_seconds}s.",
                )

        if (
            rate_cfg.max_cost_per_caller_per_day is not None
            or rate_cfg.max_tokens_per_caller_per_day is not None
        ):
            day_start = datetime.now(UTC) - timedelta(days=1)
            cost, tokens = _cost_and_tokens_since(mem_cfg, project_dir, prefix, day_start)
            if (
                rate_cfg.max_cost_per_caller_per_day is not None
                and cost >= rate_cfg.max_cost_per_caller_per_day
            ):
                raise IntaGrinError(
                    "IG-RT-008",
                    f"Rate limit exceeded: ${cost:.4f}/${rate_cfg.max_cost_per_caller_per_day:.2f} "
                    "spent in the last 24h.",
                )
            if (
                rate_cfg.max_tokens_per_caller_per_day is not None
                and tokens >= rate_cfg.max_tokens_per_caller_per_day
            ):
                raise IntaGrinError(
                    "IG-RT-008",
                    f"Rate limit exceeded: {tokens}/{rate_cfg.max_tokens_per_caller_per_day} "
                    "tokens used in the last 24h.",
                )
    except IntaGrinError as e:
        # Only an actual rate-limit breach (IG-RT-008, raised above) should reach the caller as
        # a 429. Any other IntaGrinError here — e.g. IG-RT-004 from pooled_postgres_connection
        # when neither psycopg driver is installed — is an infra problem, not a quota breach, and
        # must fail open like any other Exception below, not get misreported as "rate limited."
        if e.code == "IG-RT-008":
            raise
        Tracer.log_error(f"Rate Limit Check Error (failing open): {e}")
    except Exception as e:
        Tracer.log_error(f"Rate Limit Check Error (failing open): {e}")
