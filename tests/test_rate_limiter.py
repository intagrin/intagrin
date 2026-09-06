import sqlite3
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from intagrin.config.schema import RateLimitConfig
from intagrin.errors import IntaGrinError
from intagrin.runtime.rate_limiter import check_rate_limit
from intagrin.runtime.run_logger import record_run_log


def _sqlite_mem_cfg():
    cfg = MagicMock()
    cfg.type = "sqlite"
    cfg.db_path = None
    # See test_run_logs.py's identical comment: a bare MagicMock attribute is truthy, which
    # would make record_run_log's opportunistic pruning think retention is configured.
    cfg.run_log_retention_days = None
    return cfg


def _seed(tmp_path, session_id, *, tokens=10, cost=0.01, created_at=None):
    record_run_log(
        _sqlite_mem_cfg(),
        tmp_path,
        session_id=session_id,
        endpoint="/chat",
        agent="triage",
        status="completed",
        error=None,
        tokens_delta=tokens,
        cost_delta=cost,
        total_tokens=tokens,
        total_cost=cost,
        message_count=1,
        latency_ms=10,
    )
    if created_at is not None:
        db_path = tmp_path / ".ai" / "memory.db"
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute(
                "UPDATE run_logs SET created_at = ? WHERE session_id = ? "
                "AND id = (SELECT MAX(id) FROM run_logs WHERE session_id = ?)",
                (created_at.strftime("%Y-%m-%d %H:%M:%S"), session_id, session_id),
            )
            conn.commit()


def test_no_thresholds_configured_is_a_noop(tmp_path):
    for _ in range(5):
        _seed(tmp_path, "tenant:s1")
    check_rate_limit(_sqlite_mem_cfg(), tmp_path, "tenant", RateLimitConfig())


def test_noop_for_non_sqlite_postgres_memory_types(tmp_path):
    cfg = MagicMock()
    cfg.type = "sliding_window"
    rate_cfg = RateLimitConfig(max_requests_per_window=1)
    check_rate_limit(cfg, tmp_path, "tenant", rate_cfg)


def test_max_requests_per_window_allows_under_the_limit(tmp_path):
    for i in range(3):
        _seed(tmp_path, f"tenant:s{i}")
    rate_cfg = RateLimitConfig(max_requests_per_window=5)
    check_rate_limit(_sqlite_mem_cfg(), tmp_path, "tenant", rate_cfg)


def test_max_requests_per_window_blocks_once_exceeded(tmp_path):
    for i in range(5):
        _seed(tmp_path, f"tenant:s{i}")
    rate_cfg = RateLimitConfig(max_requests_per_window=5)

    with pytest.raises(IntaGrinError) as exc_info:
        check_rate_limit(_sqlite_mem_cfg(), tmp_path, "tenant", rate_cfg)

    assert exc_info.value.code == "IG-RT-008"
    assert exc_info.value.http_status == 429


def test_requests_outside_the_window_dont_count(tmp_path):
    old = datetime.now(UTC) - timedelta(seconds=3600)
    for i in range(5):
        _seed(tmp_path, f"tenant:old{i}", created_at=old)
    _seed(tmp_path, "tenant:recent")

    rate_cfg = RateLimitConfig(max_requests_per_window=5, window_seconds=60)
    check_rate_limit(_sqlite_mem_cfg(), tmp_path, "tenant", rate_cfg)


def test_max_cost_per_caller_per_day_blocks_once_exceeded(tmp_path):
    _seed(tmp_path, "tenant:s1", cost=3.0)
    _seed(tmp_path, "tenant:s2", cost=2.5)
    rate_cfg = RateLimitConfig(max_cost_per_caller_per_day=5.0)

    with pytest.raises(IntaGrinError) as exc_info:
        check_rate_limit(_sqlite_mem_cfg(), tmp_path, "tenant", rate_cfg)

    assert exc_info.value.code == "IG-RT-008"
    assert "$5.50" in exc_info.value.message or "5.5" in exc_info.value.message


def test_max_tokens_per_caller_per_day_blocks_once_exceeded(tmp_path):
    _seed(tmp_path, "tenant:s1", tokens=6000)
    _seed(tmp_path, "tenant:s2", tokens=5000)
    rate_cfg = RateLimitConfig(max_tokens_per_caller_per_day=10000)

    with pytest.raises(IntaGrinError) as exc_info:
        check_rate_limit(_sqlite_mem_cfg(), tmp_path, "tenant", rate_cfg)

    assert exc_info.value.code == "IG-RT-008"


def test_different_callers_are_isolated(tmp_path):
    for i in range(5):
        _seed(tmp_path, f"tenant_a:s{i}")
    rate_cfg = RateLimitConfig(max_requests_per_window=5)

    with pytest.raises(IntaGrinError):
        check_rate_limit(_sqlite_mem_cfg(), tmp_path, "tenant_a", rate_cfg)

    # tenant_b has made zero requests — must not be blocked by tenant_a's usage.
    check_rate_limit(_sqlite_mem_cfg(), tmp_path, "tenant_b", rate_cfg)


def test_fails_open_when_the_audit_db_is_unreachable(tmp_path):
    rate_cfg = RateLimitConfig(max_requests_per_window=1)

    with patch(
        "intagrin.runtime.rate_limiter.ensure_schema", side_effect=OSError("disk full")
    ), patch("intagrin.runtime.rate_limiter.Tracer.log_error") as mock_log_error:
        check_rate_limit(_sqlite_mem_cfg(), tmp_path, "tenant", rate_cfg)

    mock_log_error.assert_called_once()


def test_fails_open_on_a_non_breach_intagrin_error_not_misreported_as_a_rate_limit(tmp_path):
    """Regression test: pooled_postgres_connection (runtime/memory.py) raises IntaGrinError
    ("IG-RT-004", missing psycopg driver) instead of a plain ImportError. check_rate_limit's own
    breach signal is also an IntaGrinError ("IG-RT-008") — catching IntaGrinError by class alone
    would re-raise a missing-driver infra error as if it were a 429 rate-limit rejection. Only
    IG-RT-008 may propagate; anything else, including this one, must fail open exactly like an
    OSError does above."""
    rate_cfg = RateLimitConfig(max_requests_per_window=1)

    with patch(
        "intagrin.runtime.rate_limiter.ensure_schema",
        side_effect=IntaGrinError("IG-RT-004", "PostgreSQL driver not installed"),
    ), patch("intagrin.runtime.rate_limiter.Tracer.log_error") as mock_log_error:
        check_rate_limit(_sqlite_mem_cfg(), tmp_path, "tenant", rate_cfg)  # must not raise

    mock_log_error.assert_called_once()
