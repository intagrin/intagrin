"""Real Postgres/Redis checkpointer round trips against local Docker containers.

Not run by default `pytest tests/` in CI (no services there). Start the containers first:

    docker compose -f docker-compose.test.yml up -d

Then run this file directly (defaults below point at the docker-compose.test.yml ports, override
with TEST_POSTGRES_URL / TEST_REDIS_URL for a different instance):

    uv run pytest tests/test_memory_integration.py -q
"""

import os
import threading
import time

import pytest

TEST_POSTGRES_URL = os.environ.get(
    "TEST_POSTGRES_URL", "postgresql://intagrin:intagrin_test@localhost:5433/intagrin_test"
)
TEST_REDIS_URL = os.environ.get("TEST_REDIS_URL", "redis://localhost:6380/0")


# --- Postgres -----------------------------------------------------------------------------

psycopg = pytest.importorskip("psycopg")

from intagrin.runtime.memory import (
    PostgresCheckpointer,
    RedisCheckpointer,
    _pg_pools,
    _pg_pools_lock,
)


def _postgres_reachable() -> bool:
    """A fast (2s) reachability probe, separate from PostgresCheckpointer's own connection-pool
    construction — that pool's default connect timeout is ~30s, which would make every test in
    this file that skips (no Docker container running) take 30s instead of failing fast."""
    try:
        with psycopg.connect(TEST_POSTGRES_URL, connect_timeout=2):
            return True
    except Exception:
        return False


@pytest.fixture
def pg_checkpointer():
    if not _postgres_reachable():
        pytest.skip(f"No reachable Postgres test instance at {TEST_POSTGRES_URL}")
    cp = PostgresCheckpointer(TEST_POSTGRES_URL)
    cp._execute("DELETE FROM checkpoints WHERE session_id LIKE 'test_%'", commit=True)
    return cp


def test_postgres_save_and_load_round_trip(pg_checkpointer):
    session_id = f"test_roundtrip_{time.time()}"
    messages = [{"role": "user", "content": "hello"}]
    state = {"foo": "bar", "count": 3}

    pg_checkpointer.save_checkpoint(session_id, messages, state)
    loaded_messages, loaded_state = pg_checkpointer.load_checkpoint(session_id)

    assert loaded_messages == messages
    assert loaded_state == state


def test_postgres_save_overwrites_on_conflict(pg_checkpointer):
    session_id = f"test_overwrite_{time.time()}"
    pg_checkpointer.save_checkpoint(session_id, [{"role": "user", "content": "v1"}], {"v": 1})
    pg_checkpointer.save_checkpoint(session_id, [{"role": "user", "content": "v2"}], {"v": 2})

    messages, state = pg_checkpointer.load_checkpoint(session_id)
    assert state == {"v": 2}
    assert messages == [{"role": "user", "content": "v2"}]


def test_postgres_list_sessions_returns_real_rows(pg_checkpointer):
    prefix = f"test_list_{int(time.time())}_"
    for i in range(3):
        pg_checkpointer.save_checkpoint(f"{prefix}{i}", [], {"i": i})

    sessions = pg_checkpointer.list_sessions(prefix=prefix)

    assert set(sessions) == {f"{prefix}{i}" for i in range(3)}


def test_postgres_pool_is_shared_across_instances_for_same_url(pg_checkpointer):
    second = PostgresCheckpointer(TEST_POSTGRES_URL)
    assert second.pool is pg_checkpointer.pool


def test_postgres_pool_creation_is_race_free_under_concurrent_construction(pg_checkpointer):
    """Proves the _pg_pools_lock fix in memory.py: constructing many PostgresCheckpointers for a
    brand-new connection_url from concurrent threads must all end up sharing exactly one pool
    object, not each racing to create their own (which would exhaust Postgres connection limits
    under real concurrent load — the original bug this test exists to catch)."""
    fresh_url = TEST_POSTGRES_URL + "?application_name=intagrin_race_test"
    with _pg_pools_lock:
        _pg_pools.pop(fresh_url, None)

    results: list[PostgresCheckpointer] = []
    errors: list[Exception] = []

    def make_one():
        try:
            results.append(PostgresCheckpointer(fresh_url))
        except Exception as e:  # pragma: no cover - failure path surfaced via assertion below
            errors.append(e)

    threads = [threading.Thread(target=make_one) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert len(results) == 20
    assert len({id(r.pool) for r in results}) == 1


def test_record_run_log_actually_writes_a_row_against_real_postgres(pg_checkpointer):
    """Regression test: run_logger.py used to hard-import psycopg2, which this project's
    'postgres' extra doesn't install (it installs psycopg v3) — record_run_log's ImportError
    guard made it silently no-op for every Postgres-backed project, so the Logs page (and the
    rate limiter that queries the same run_logs table) would have looked empty. Caught only by
    actually writing to a real Postgres instance and checking the row landed, not by mocking."""
    from types import SimpleNamespace

    from intagrin.runtime.run_logger import ensure_schema, record_run_log

    mem_cfg = SimpleNamespace(type="postgres", connection_url=TEST_POSTGRES_URL, env_var=None)
    session_id = f"test_run_log_{time.time()}"
    ensure_schema(mem_cfg, None)
    record_run_log(
        mem_cfg,
        None,
        session_id=session_id,
        endpoint="/chat",
        agent="triage",
        status="completed",
        error=None,
        tokens_delta=5,
        cost_delta=0.001,
        total_tokens=5,
        total_cost=0.001,
        message_count=1,
        latency_ms=10,
    )

    row = pg_checkpointer._execute(
        "SELECT status FROM run_logs WHERE session_id = %s", (session_id,), fetch_one=True
    )
    assert row is not None, "record_run_log did not write a row against real Postgres"
    assert row[0] == "completed"


def test_shared_memory_save_and_load_round_trip_against_real_postgres(pg_checkpointer):
    from types import SimpleNamespace

    from intagrin.runtime.shared_memory import load_shared_memory, save_shared_memory

    mem_cfg = SimpleNamespace(type="postgres", connection_url=TEST_POSTGRES_URL, env_var=None)
    scope_key = f"test_shared_mem_{time.time()}"

    save_shared_memory(mem_cfg, None, scope_key, "The org prefers concise replies.")
    content = load_shared_memory(mem_cfg, None, scope_key)
    assert content == "The org prefers concise replies."

    save_shared_memory(mem_cfg, None, scope_key, "Updated preference.")
    content = load_shared_memory(mem_cfg, None, scope_key)
    assert content == "Updated preference."


def test_approvers_add_verify_revoke_round_trip_against_real_postgres(pg_checkpointer):
    """runtime/approvers.py switched from a raw postgres_connect() per call to the shared
    connection pool (pooled_postgres_connection) — proves that swap didn't break anything by
    actually issuing, verifying, and revoking a credential against a real Postgres instance."""
    from types import SimpleNamespace

    from intagrin.runtime.approvers import add_approver, list_approvers, revoke_approver, verify_secret

    mem_cfg = SimpleNamespace(type="postgres", connection_url=TEST_POSTGRES_URL, env_var=None)
    approver_id = f"test_approver_{time.time()}"

    add_approver(mem_cfg, None, approver_id, "s3cr3t-value")
    assert verify_secret(mem_cfg, None, "s3cr3t-value") == approver_id
    assert verify_secret(mem_cfg, None, "wrong-value") is None

    rows = list_approvers(mem_cfg, None)
    assert any(r["approver_id"] == approver_id and r["revoked_at"] is None for r in rows)

    assert revoke_approver(mem_cfg, None, approver_id) is True
    assert verify_secret(mem_cfg, None, "s3cr3t-value") is None
    assert revoke_approver(mem_cfg, None, approver_id) is False  # already revoked


def test_rate_limiter_counts_real_postgres_run_logs(pg_checkpointer):
    """rate_limiter.py's _count_since/_cost_and_tokens_since switched to pooled_postgres_connection
    too — proves check_rate_limit actually trips against rows written to a real Postgres
    run_logs table, not just a mocked connection."""
    from types import SimpleNamespace

    from intagrin.errors import IntaGrinError
    from intagrin.runtime.rate_limiter import check_rate_limit
    from intagrin.runtime.run_logger import ensure_schema, record_run_log

    mem_cfg = SimpleNamespace(type="postgres", connection_url=TEST_POSTGRES_URL, env_var=None)
    user_context = f"test_tenant_{time.time()}"
    ensure_schema(mem_cfg, None)
    for _ in range(3):
        record_run_log(
            mem_cfg, None, session_id=f"{user_context}:s1", endpoint="/chat", agent="triage",
            status="completed", error=None, tokens_delta=1, cost_delta=0.0,
            total_tokens=1, total_cost=0.0, message_count=1, latency_ms=1,
        )

    rate_cfg = SimpleNamespace(
        max_requests_per_window=3, window_seconds=3600,
        max_cost_per_caller_per_day=None, max_tokens_per_caller_per_day=None,
    )
    with pytest.raises(IntaGrinError) as exc_info:
        check_rate_limit(mem_cfg, None, user_context, rate_cfg)
    assert exc_info.value.code == "IG-RT-008"


def test_episodic_memory_save_and_recall_round_trip_against_real_postgres(pg_checkpointer):
    """runtime/episodic_memory.py switched to pooled_postgres_connection too — proves
    save_episode/query_episodes still round-trip against a real Postgres instance."""
    from types import SimpleNamespace

    from intagrin.runtime.episodic_memory import query_episodes, save_episode

    mem_cfg = SimpleNamespace(type="postgres", connection_url=TEST_POSTGRES_URL, env_var=None)
    scope_key = f"test_episodes_{time.time()}"

    save_episode(mem_cfg, None, scope_key, "sess1", "preference", "User prefers window seats.", None, None)
    rows = query_episodes(mem_cfg, None, scope_key, None, None, limit=10)
    assert len(rows) == 1
    assert rows[0]["content"] == "User prefers window seats."


# --- Redis ----------------------------------------------------------------------------------

redis = pytest.importorskip("redis")


@pytest.fixture
def redis_checkpointer():
    try:
        cp = RedisCheckpointer(TEST_REDIS_URL)
        cp.client.ping()
    except Exception as e:
        pytest.skip(f"No reachable Redis test instance at {TEST_REDIS_URL}: {e}")
    yield cp
    for key in cp.client.scan_iter(match="intagrin:session:test_*"):
        cp.client.delete(key)


def test_redis_save_and_load_round_trip(redis_checkpointer):
    session_id = f"test_roundtrip_{time.time()}"
    messages = [{"role": "user", "content": "hi"}]
    state = {"x": 1}

    redis_checkpointer.save_checkpoint(session_id, messages, state)
    loaded_messages, loaded_state = redis_checkpointer.load_checkpoint(session_id)

    assert loaded_messages == messages
    assert loaded_state == state


def test_redis_list_sessions_returns_real_keys():
    try:
        cp = RedisCheckpointer(TEST_REDIS_URL)
        cp.client.ping()
    except Exception as e:
        pytest.skip(f"No reachable Redis test instance at {TEST_REDIS_URL}: {e}")

    prefix = f"test_list_{int(time.time())}_"
    for i in range(3):
        cp.save_checkpoint(f"{prefix}{i}", [], {"i": i})

    sessions = cp.list_sessions(prefix=prefix)

    assert set(sessions) == {f"{prefix}{i}" for i in range(3)}
    for key in cp.client.scan_iter(match=f"intagrin:session:{prefix}*"):
        cp.client.delete(key)


def test_redis_ttl_actually_expires_keys():
    try:
        short_ttl_cp = RedisCheckpointer(TEST_REDIS_URL, ttl_seconds=1)
        short_ttl_cp.client.ping()
    except Exception as e:
        pytest.skip(f"No reachable Redis test instance at {TEST_REDIS_URL}: {e}")

    session_id = f"test_ttl_{time.time()}"
    short_ttl_cp.save_checkpoint(session_id, [{"role": "user", "content": "x"}], {"y": 1})

    messages, state = short_ttl_cp.load_checkpoint(session_id)
    assert state == {"y": 1}, "checkpoint should still be present immediately after save"

    time.sleep(2.5)

    messages, state = short_ttl_cp.load_checkpoint(session_id)
    assert messages == []
    assert state == {}, "checkpoint should have expired past its 1s TTL"
