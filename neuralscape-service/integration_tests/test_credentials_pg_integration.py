"""PostgreSQL credential conformance (integration; needs AUTHORITY_TEST_PG_DSN).

Runs the same cases as the SQLite suite against an isolated schema per case.
Skipped when the variable is unset; that skip is the declared one.
"""

from __future__ import annotations

import os
import uuid

import pytest

psycopg = pytest.importorskip("psycopg")
from psycopg.conninfo import make_conninfo  # noqa: E402

from authority_store_sql import PostgresAuthorityStore  # noqa: E402
from tests.test_credentials import CASES  # noqa: E402

from tests.authority_pg_support import DSN, require_pg  # noqa: E402,F401  (autouse)

pytestmark = pytest.mark.integration


class PostgresBackend:
    def __init__(self) -> None:
        self.schema = f"credentials_{uuid.uuid4().hex[:12]}"
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
def test_postgres_credential_conformance(case, pg_backend) -> None:
    case(pg_backend)
