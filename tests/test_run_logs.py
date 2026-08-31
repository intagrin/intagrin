import sqlite3
from unittest.mock import MagicMock, patch

from intagrin.runtime.run_logger import record_run_log


def _sqlite_mem_cfg(db_path=None):
    cfg = MagicMock()
    cfg.type = "sqlite"
    cfg.db_path = db_path
    # A bare MagicMock attribute is truthy, which would make record_run_log's opportunistic
    # pruning think retention is configured and (on its ~0.5% random roll) try
    # timedelta(days=<MagicMock>) — explicit None matches a real MemoryConfig's own default.
    cfg.run_log_retention_days = None
    return cfg


def test_record_run_log_creates_table_and_inserts_row_sqlite(tmp_path):
    record_run_log(
        _sqlite_mem_cfg(),
        tmp_path,
        session_id="tenant:s1",
        endpoint="/chat",
        agent="triage",
        status="completed",
        error=None,
        tokens_delta=42,
        cost_delta=0.001,
        total_tokens=100,
        total_cost=0.01,
        message_count=4,
        latency_ms=250,
    )

    db_path = tmp_path / ".ai" / "memory.db"
    assert db_path.exists()
    with sqlite3.connect(str(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM run_logs").fetchall()

    assert len(rows) == 1
    row = dict(rows[0])
    assert row["session_id"] == "tenant:s1"
    assert row["endpoint"] == "/chat"
    assert row["agent"] == "triage"
    assert row["status"] == "completed"
    assert row["error"] is None
    assert row["tokens_delta"] == 42
    assert row["cost_delta"] == 0.001
    assert row["total_tokens"] == 100
    assert row["message_count"] == 4
    assert row["latency_ms"] == 250
    assert row["created_at"] is not None


def test_record_run_log_noop_for_non_persistent_memory_types(tmp_path):
    for mem_type in ("sliding_window", "buffer", "redis", "custom"):
        cfg = MagicMock()
        cfg.type = mem_type
        record_run_log(cfg, tmp_path, session_id="s", endpoint="/chat", status="completed")

    assert not (tmp_path / ".ai").exists()


def test_record_run_log_never_raises_on_write_failure(tmp_path):
    with patch(
        "intagrin.runtime.run_logger.sqlite3.connect", side_effect=OSError("disk full")
    ), patch("intagrin.runtime.run_logger.Tracer.log_error") as mock_log_error:
        record_run_log(
            _sqlite_mem_cfg(), tmp_path, session_id="s", endpoint="/chat", status="completed"
        )

    mock_log_error.assert_called_once()
    assert "disk full" in mock_log_error.call_args[0][0]


def test_run_log_retention_days_prunes_old_rows_but_keeps_recent_ones(tmp_path):
    """run_logs is otherwise append-only forever — memory.run_log_retention_days opportunistically
    deletes rows older than the cutoff. Forces the random trigger deterministically instead of
    relying on its real ~0.5% probability (_PRUNE_PROBABILITY)."""
    from datetime import UTC, datetime, timedelta

    cfg = _sqlite_mem_cfg()
    record_run_log(
        cfg, tmp_path, session_id="tenant:old", endpoint="/chat", agent="a",
        status="completed", error=None, tokens_delta=1, cost_delta=0.0,
        total_tokens=1, total_cost=0.0, message_count=1, latency_ms=1,
    )

    db_path = tmp_path / ".ai" / "memory.db"
    old_created_at = (datetime.now(UTC) - timedelta(days=100)).strftime("%Y-%m-%d %H:%M:%S")
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute(
            "UPDATE run_logs SET created_at = ? WHERE session_id = 'tenant:old'",
            (old_created_at,),
        )
        conn.commit()

    cfg.run_log_retention_days = 30
    with patch("intagrin.runtime.run_logger.random.random", return_value=0.0):
        record_run_log(
            cfg, tmp_path, session_id="tenant:new", endpoint="/chat", agent="a",
            status="completed", error=None, tokens_delta=1, cost_delta=0.0,
            total_tokens=1, total_cost=0.0, message_count=1, latency_ms=1,
        )

    with sqlite3.connect(str(db_path)) as conn:
        session_ids = {row[0] for row in conn.execute("SELECT session_id FROM run_logs")}
    assert session_ids == {"tenant:new"}


def test_run_log_retention_days_none_never_prunes(tmp_path):
    """The default (run_log_retention_days=None) must not prune anything, even on the lucky
    random roll — pruning is strictly opt-in."""
    from datetime import UTC, datetime, timedelta

    cfg = _sqlite_mem_cfg()
    record_run_log(
        cfg, tmp_path, session_id="tenant:ancient", endpoint="/chat", agent="a",
        status="completed", error=None, tokens_delta=1, cost_delta=0.0,
        total_tokens=1, total_cost=0.0, message_count=1, latency_ms=1,
    )

    db_path = tmp_path / ".ai" / "memory.db"
    old_created_at = (datetime.now(UTC) - timedelta(days=9999)).strftime("%Y-%m-%d %H:%M:%S")
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute(
            "UPDATE run_logs SET created_at = ? WHERE session_id = 'tenant:ancient'",
            (old_created_at,),
        )
        conn.commit()

    with patch("intagrin.runtime.run_logger.random.random", return_value=0.0):
        record_run_log(
            cfg, tmp_path, session_id="tenant:new2", endpoint="/chat", agent="a",
            status="completed", error=None, tokens_delta=1, cost_delta=0.0,
            total_tokens=1, total_cost=0.0, message_count=1, latency_ms=1,
        )

    with sqlite3.connect(str(db_path)) as conn:
        session_ids = {row[0] for row in conn.execute("SELECT session_id FROM run_logs")}
    assert "tenant:ancient" in session_ids
