"""PostgreSQL authority store conformance (integration; needs AUTHORITY_TEST_PG_DSN).

The DSN must point at a disposable PostgreSQL 16 database whose role may create
schemas.  Every store is an isolated schema, dropped afterwards.  Skipped when
the variable is unset; that skip is the declared one.  Runs the same cases as
the SQLite suite.
"""

from __future__ import annotations

import os
import uuid

import pytest

psycopg = pytest.importorskip("psycopg")
from psycopg.conninfo import make_conninfo  # noqa: E402

from authority_store_sql import PostgresAuthorityStore  # noqa: E402
from tests.test_authority_store_sql import CASES  # noqa: E402

from tests.authority_pg_support import DSN, require_pg  # noqa: E402,F401  (autouse)

pytestmark = pytest.mark.integration


class PostgresBackend:
    def __init__(self) -> None:
        self.schema = f"authority_{uuid.uuid4().hex[:12]}"
        with psycopg.connect(DSN, autocommit=True) as admin:
            admin.execute(f'CREATE SCHEMA "{self.schema}"')
        self.dsn = make_conninfo(DSN, options=f"-c search_path={self.schema}")
        self.opened: list[PostgresAuthorityStore] = []

    def open(self, clock=None) -> PostgresAuthorityStore:
        store = PostgresAuthorityStore(self.dsn, **({"clock": clock} if clock else {}))
        self.opened.append(store)
        return store

    def close(self) -> None:
        for store in self.opened:
            store.close()
        with psycopg.connect(DSN, autocommit=True) as admin:
            admin.execute(f'DROP SCHEMA IF EXISTS "{self.schema}" CASCADE')


@pytest.fixture
def pg_backend():
    backend = PostgresBackend()
    yield backend
    backend.close()


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.__name__)
def test_postgres_conformance(case, pg_backend) -> None:
    case(pg_backend)


def test_postgres_state_survives_reconnect(pg_backend) -> None:
    from authority_store_sql import PolicyConflict  # noqa: F401
    from tests.test_authority_store_sql import handle, stmt

    store = pg_backend.open()
    store.add_subject("sub_a", "tenant-a", "human", "cred-a")
    store.publish_policy("tenant-a", [stmt("s1", "sub_a", "read", handle())])
    store.close()
    again = pg_backend.open()
    assert again.current_policy("tenant-a").policy_epoch == 1
    assert again.subject_record("sub_a").membership_version == 1


def test_postgres_store_reconnects_after_a_lost_connection(pg_backend) -> None:
    from authority_store import AuthorityUnavailable

    store = pg_backend.open()
    store.add_subject("sub_a", "tenant-a", "human", "cred-a")
    for name in ("_conn", "_rconn"):
        pid = getattr(store, name).info.backend_pid
        with psycopg.connect(DSN, autocommit=True) as admin:
            admin.execute("SELECT pg_terminate_backend(%s)", (pid,))
    with pytest.raises(AuthorityUnavailable):          # fails closed first ...
        store.subject_record("sub_a")
    for _ in range(3):                                  # ... and recovers without a restart
        try:
            assert store.subject_record("sub_a") is not None
            break
        except AuthorityUnavailable:
            continue
    else:
        pytest.fail("store did not reconnect")
    for _ in range(3):                                  # the write connection recovers too
        try:
            assert store.add_subject("sub_b", "tenant-a", "human", "cred-b") == 1
            break
        except AuthorityUnavailable:
            continue
    else:
        pytest.fail("write connection did not reconnect")
    assert store.subject_record("sub_b") is not None
