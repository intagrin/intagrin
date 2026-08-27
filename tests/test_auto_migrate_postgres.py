"""The Postgres branch of run_auto_migrations() had zero test coverage of any kind (mocked or
real) before this file — tests/test_auto_migrate.py only exercises the SQLite branch. Requires
the local docker-compose.test.yml Postgres container (or TEST_POSTGRES_URL pointed elsewhere);
skips cleanly if unreachable, same posture as tests/test_memory_integration.py.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

psycopg = pytest.importorskip("psycopg")

from intagrin.db_migrations.auto_migrate import run_auto_migrations

TEST_POSTGRES_URL = os.environ.get(
    "TEST_POSTGRES_URL", "postgresql://intagrin:intagrin_test@localhost:5433/intagrin_test"
)


def _postgres_reachable() -> bool:
    try:
        with psycopg.connect(TEST_POSTGRES_URL, connect_timeout=2):
            return True
    except Exception:
        return False


@pytest.mark.skipif(
    not _postgres_reachable(), reason=f"No reachable Postgres test instance at {TEST_POSTGRES_URL}"
)
def test_auto_migrations_run_against_a_real_postgres_database():
    with psycopg.connect(TEST_POSTGRES_URL) as conn:
        with conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS checkpoints CASCADE")
            cur.execute("DROP TABLE IF EXISTS run_logs CASCADE")
            cur.execute("DROP TABLE IF EXISTS alembic_version CASCADE")
        conn.commit()

    graph = MagicMock()
    graph.config.memory.type = "postgres"
    graph.config.memory.connection_url = TEST_POSTGRES_URL
    graph.config.memory.env_var = None

    with patch("intagrin.db_migrations.auto_migrate.parse_project", return_value=graph):
        run_auto_migrations()

    with psycopg.connect(TEST_POSTGRES_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
            )
            tables = {row[0] for row in cur.fetchall()}

    assert "checkpoints" in tables
    assert "run_logs" in tables


@pytest.mark.skipif(
    not _postgres_reachable(), reason=f"No reachable Postgres test instance at {TEST_POSTGRES_URL}"
)
def test_auto_migrations_are_idempotent_against_a_real_postgres_database():
    """Server restarts re-run auto-migrations on an already-migrated database — must not error."""
    graph = MagicMock()
    graph.config.memory.type = "postgres"
    graph.config.memory.connection_url = TEST_POSTGRES_URL
    graph.config.memory.env_var = None

    with patch("intagrin.db_migrations.auto_migrate.parse_project", return_value=graph):
        run_auto_migrations()
        run_auto_migrations()

    with psycopg.connect(TEST_POSTGRES_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
            )
            tables = {row[0] for row in cur.fetchall()}

    assert "checkpoints" in tables
