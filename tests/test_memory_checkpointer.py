import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from intagrin.config.schema import MemoryConfig
from intagrin.runtime.memory import (
    CheckpointerConfigError,
    SQLiteCheckpointer,
    build_checkpointer,
)


def _backdate(db_path: Path, session_id: str, when: datetime):
    with sqlite3.connect(db_path, timeout=15.0) as conn:
        conn.execute(
            "UPDATE checkpoints SET updated_at = ? WHERE session_id = ?",
            (when.strftime("%Y-%m-%d %H:%M:%S"), session_id),
        )
        conn.commit()


def test_sqlite_list_sessions_filters_by_prefix_since_and_limit(tmp_path):
    db_path = tmp_path / "memory.db"
    cp = SQLiteCheckpointer(str(db_path))

    now = datetime.now(UTC)
    cp.save_checkpoint("tenant_a:s1", [{"role": "user", "content": "hi"}], {})
    cp.save_checkpoint("tenant_a:s2", [{"role": "user", "content": "hi"}], {})
    cp.save_checkpoint("tenant_b:s1", [{"role": "user", "content": "hi"}], {})
    _backdate(db_path, "tenant_a:s1", now - timedelta(days=40))
    _backdate(db_path, "tenant_a:s2", now - timedelta(days=1))
    _backdate(db_path, "tenant_b:s1", now - timedelta(days=1))

    # prefix filter
    tenant_a_sessions = cp.list_sessions(prefix="tenant_a:")
    assert set(tenant_a_sessions) == {"tenant_a:s1", "tenant_a:s2"}

    # since filter excludes the 40-day-old session
    recent = cp.list_sessions(prefix="tenant_a:", since=now - timedelta(days=7))
    assert recent == ["tenant_a:s2"]

    # most-recently-updated first, across prefixes
    all_recent = cp.list_sessions(since=now - timedelta(days=7))
    assert set(all_recent) == {"tenant_a:s2", "tenant_b:s1"}

    # limit
    assert len(cp.list_sessions(limit=1)) == 1


def test_build_checkpointer_sqlite_non_strict_and_strict_agree(tmp_path):
    mem_cfg = MemoryConfig(type="sqlite", db_path=".ai/memory.db")
    cp1 = build_checkpointer(mem_cfg, tmp_path)
    cp2 = build_checkpointer(mem_cfg, tmp_path, strict=True)
    assert isinstance(cp1, SQLiteCheckpointer)
    assert isinstance(cp2, SQLiteCheckpointer)


def test_build_checkpointer_non_strict_falls_back_and_returns_none_gracefully(tmp_path, monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("POSTGRES_URL", raising=False)

    # Postgres with no connection info falls back to local SQLite rather than failing to boot.
    mem_cfg = MemoryConfig(type="postgres")
    cp = build_checkpointer(mem_cfg, tmp_path)
    assert isinstance(cp, SQLiteCheckpointer)

    # A memory type with no checkpointer (in-process only) returns None, not an error.
    mem_cfg2 = MemoryConfig(type="buffer")
    assert build_checkpointer(mem_cfg2, tmp_path) is None


def test_build_checkpointer_strict_raises_instead_of_guessing(tmp_path, monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("POSTGRES_URL", raising=False)
    monkeypatch.delenv("REDIS_URL", raising=False)

    with pytest.raises(CheckpointerConfigError):
        build_checkpointer(MemoryConfig(type="postgres"), tmp_path, strict=True)

    with pytest.raises(CheckpointerConfigError):
        build_checkpointer(MemoryConfig(type="redis"), tmp_path, strict=True)

    with pytest.raises(CheckpointerConfigError):
        build_checkpointer(MemoryConfig(type="buffer"), tmp_path, strict=True)

    with pytest.raises(CheckpointerConfigError):
        build_checkpointer(
            MemoryConfig(type="custom", custom_module="tools.my_checkpointer"),
            tmp_path,
            strict=True,
        )
