import sqlite3
from unittest.mock import MagicMock, patch

from intagrin.runtime.run_logger import record_run_log


def _sqlite_mem_cfg(db_path=None):
    cfg = MagicMock()
    cfg.type = "sqlite"
    cfg.db_path = db_path
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
