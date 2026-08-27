import sqlite3
from unittest.mock import MagicMock, patch

from intagrin.db_migrations.auto_migrate import run_auto_migrations


def test_auto_migrations_run_on_a_fresh_project_with_no_ai_directory_yet(tmp_path, monkeypatch):
    """A freshly `inta new`-scaffolded project has no .ai/ directory — nothing creates it until
    the first session is ever saved. Auto-migrations run at server startup, before any request
    has happened, so this must succeed (creating .ai/ itself) rather than fail with "unable to
    open database file" on a project's very first inta serve/inta monitor run."""
    monkeypatch.chdir(tmp_path)
    assert not (tmp_path / ".ai").exists()

    graph = MagicMock()
    graph.config.memory.type = "sqlite"
    graph.config.memory.db_path = None

    with patch("intagrin.db_migrations.auto_migrate.parse_project", return_value=graph):
        run_auto_migrations()

    db_path = tmp_path / ".ai" / "memory.db"
    assert db_path.exists()
    with sqlite3.connect(str(db_path)) as conn:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
    assert "checkpoints" in tables
    assert "run_logs" in tables
