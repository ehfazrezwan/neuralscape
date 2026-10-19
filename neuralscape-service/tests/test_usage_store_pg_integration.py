"""PostgreSQL usage store conformance (integration; needs USAGE_TEST_PG_DSN).

The DSN must point at a disposable PostgreSQL 16 database whose role may create
schemas.  Each store is an isolated schema, dropped afterwards.  Skipped when
the variable is unset; that skip is the declared one.
"""

from __future__ import annotations

import os
import uuid

import pytest

psycopg = pytest.importorskip("psycopg")
from psycopg.conninfo import make_conninfo  # noqa: E402

from tests.test_usage_store import SHARED_CASES, TENANT, event  # noqa: E402
from usage_store import PostgresUsageStore, UsageStoreError  # noqa: E402

DSN = os.environ.get("USAGE_TEST_PG_DSN")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not DSN, reason="USAGE_TEST_PG_DSN not set (postgres usage store integration)"),
]

_created: list[str] = []


def _new_dsn() -> str:
    schema = f"usage_{uuid.uuid4().hex[:12]}"
    with psycopg.connect(DSN, autocommit=True) as admin:
        admin.execute(f'CREATE SCHEMA "{schema}"')
    _created.append(schema)
    return make_conninfo(DSN, options=f"-c search_path={schema}")


@pytest.fixture()
def make_pg():
    opened = []

    def make():
        store = PostgresUsageStore(_new_dsn())
        opened.append(store)
        return store

    yield make
    for store in opened:
        store.close()


@pytest.fixture(autouse=True, scope="module")
def _drop_schemas():
    yield
    if DSN:
        with psycopg.connect(DSN, autocommit=True) as admin:
            for schema in _created:
                admin.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')


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
    store = make_pg()
    dsn = _created[-1]
    reopened = PostgresUsageStore(make_conninfo(DSN, options=f"-c search_path={dsn}"))
    assert reopened.schema_version() == 2
    with reopened._txn() as c:
        c.execute("INSERT INTO usage_schema_version VALUES (99, 'future', 'x', 0)")
    with pytest.raises(UsageStoreError, match="newer"):
        PostgresUsageStore(make_conninfo(DSN, options=f"-c search_path={dsn}"))
    reopened.close()


def test_digest_tamper_detected(make_pg):
    store = make_pg()
    store.append(event("e1"))
    with store._txn() as c:
        c.execute("UPDATE usage_events SET body = replace(body, '\"final\"', '\"pending\"')")
    with pytest.raises(UsageStoreError):
        store.events(TENANT)
