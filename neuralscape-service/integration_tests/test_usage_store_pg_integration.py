"""PostgreSQL usage store conformance (``NS_TEST_PG_DSN``, the gate's PostgreSQL).

The role must be able to create schemas.  Each store is an isolated schema, dropped afterwards.  A missing
PostgreSQL fails the test on the gate and skips it elsewhere (``integration_tests/README.md``).
"""

from __future__ import annotations

import uuid

import pytest

psycopg = pytest.importorskip("psycopg")  # a declared condition: the optional PostgreSQL driver
from psycopg.conninfo import make_conninfo  # noqa: E402

from tests.test_usage_store import SHARED_CASES, TENANT, event  # noqa: E402
from usage_store import PostgresUsageStore, UsageStoreError  # noqa: E402

_created: list[str] = []


@pytest.fixture()
def make_pg(pg_dsn):
    opened = []

    def new_dsn() -> str:
        schema = f"usage_nsit_{uuid.uuid4().hex[:12]}"
        with psycopg.connect(pg_dsn, autocommit=True) as admin:
            admin.execute(f'CREATE SCHEMA "{schema}"')
        _created.append(schema)
        return make_conninfo(pg_dsn, options=f"-c search_path={schema}")

    def make():
        store = PostgresUsageStore(new_dsn())
        opened.append(store)
        return store

    make.dsn_for = lambda: make_conninfo(pg_dsn, options=f"-c search_path={_created[-1]}")
    yield make
    for store in opened:
        store.close()
    with psycopg.connect(pg_dsn, autocommit=True) as admin:
        while _created:
            admin.execute(f'DROP SCHEMA IF EXISTS "{_created.pop()}" CASCADE')


@pytest.mark.parametrize("case", SHARED_CASES, ids=lambda c: c.__name__)
def test_shared_case_on_postgres(case, make_pg):
    case(make_pg)


def test_rows_content_free_and_schema_version(make_pg):
    store = make_pg()
    store.append(event("e1", intent="i1"), ledger_attempt=1)
    assert store.schema_version() == 2
    with store._txn() as c:
        columns = [
            r[0]
            for r in c.execute(
                "SELECT column_name FROM information_schema.columns"
                " WHERE table_schema = current_schema() AND table_name='usage_events'"
                " ORDER BY ordinal_position"
            ).fetchall()
        ]
    assert "body" in columns and not {"prompt", "text", "content", "response"} & set(columns)


def test_newer_schema_refused_and_reopen_idempotent(make_pg):
    make_pg()
    reopened = PostgresUsageStore(make_pg.dsn_for())
    assert reopened.schema_version() == 2
    with reopened._txn() as c:
        c.execute("INSERT INTO usage_schema_version VALUES (99, 'future', 'x', 0)")
    with pytest.raises(UsageStoreError, match="newer"):
        PostgresUsageStore(make_pg.dsn_for())
    reopened.close()


def test_digest_tamper_detected(make_pg):
    store = make_pg()
    store.append(event("e1"))
    with store._txn() as c:
        c.execute("UPDATE usage_events SET body = replace(body, '\"final\"', '\"pending\"')")
    with pytest.raises(UsageStoreError):
        store.events(TENANT)


def test_the_next_append_succeeds_after_the_backend_is_terminated(make_pg, pg_dsn):
    """A PostgreSQL restart or failover kills the store's one connection: it must reconnect, not stay broken."""
    store = make_pg()
    assert store.append(event("e1")) is True
    backend_pid = store._conn.info.backend_pid
    with psycopg.connect(pg_dsn, autocommit=True) as other:
        other.execute("SELECT pg_terminate_backend(%s)", (backend_pid,))
    assert store.append(event("e2")) is True  # a fresh connection, one bounded retry
    assert store.reconnects == 1
    assert store._conn.info.backend_pid != backend_pid
    assert {stored.event_id for stored in store.events(TENANT)} == {"e1", "e2"}


def test_the_durable_recorder_keeps_recording_after_the_backend_is_terminated(make_pg, pg_dsn):
    import usage_capture as uc

    store = make_pg()
    recorder = uc.DurableUsageRecorder(store, retries=0)  # no recorder-level retry: the store alone recovers
    with uc.attempt_context(uc.AttemptContext(tenant_id=TENANT, task_id="task-1", attempt_id="a")):
        assert recorder.record(event("e1")) is True
        with psycopg.connect(pg_dsn, autocommit=True) as other:
            other.execute("SELECT pg_terminate_backend(%s)", (store._conn.info.backend_pid,))
        assert recorder.record(event("e2")) is True
    assert recorder.failures == 0 and recorder.unrecorded == {}


def test_an_unreachable_database_costs_an_append_only_its_budget(make_pg, pg_dsn):
    import time

    store = make_pg()
    store.append(event("e1"))
    with psycopg.connect(pg_dsn, autocommit=True) as other:
        other.execute("SELECT pg_terminate_backend(%s)", (store._conn.info.backend_pid,))
    store._dsn = "host=127.0.0.1 port=1 dbname=none connect_timeout=1"  # nothing listens there
    started = time.monotonic()
    with pytest.raises(Exception):
        store.append(event("e2"))
    with pytest.raises(UsageStoreError, match="backing off"):
        store.append(event("e3"))  # fails at once during the cooldown
    assert time.monotonic() - started < 6.0
