"""Shared PostgreSQL fixtures for the authority integration tests.

``AUTHORITY_TEST_PG_DSN`` must point at a disposable PostgreSQL 16 server and a role that may create
schemas and roles (used only to set up and tear down).  Each backend gets its own schema, a
non-superuser OWNER role (applies the migrations, grants the runtime role) and a non-superuser RUNTIME
role (what the service connects as: ``migrate=False``, granted by ``grant_authority_runtime_role``).
"""

from __future__ import annotations

import os
import uuid

import pytest

psycopg = pytest.importorskip("psycopg")
from psycopg.conninfo import make_conninfo  # noqa: E402

from authority_store_sql import PostgresAuthorityStore, grant_authority_runtime_role  # noqa: E402

# AUTHORITY_TEST_PG_DSN, else the gate's NS_TEST_PG_DSN: a server whose role may create schemas and roles.
DSN = os.environ.get("AUTHORITY_TEST_PG_DSN") or os.environ.get("NS_TEST_PG_DSN")
REQUIRED = os.environ.get("NS_INTEGRATION_REQUIRED") == "1"
PASSWORD = "authority-test-pw"  # a throwaway role in a disposable database
REASON = "AUTHORITY_TEST_PG_DSN not set (postgres authority runtime-role integration)"


class RuntimeRoleBackend:
    """``open()`` is the least-privilege runtime role; ``open_owner()`` applies the migrations."""

    def __init__(self) -> None:
        tag = uuid.uuid4().hex[:10]
        self.schema, self.owner, self.runtime = f"auth_{tag}", f"ao_{tag}", f"ar_{tag}"
        with psycopg.connect(DSN, autocommit=True) as admin:
            for role in (self.owner, self.runtime):
                admin.execute(f'CREATE ROLE "{role}" LOGIN NOSUPERUSER NOBYPASSRLS PASSWORD \'{PASSWORD}\'')
            admin.execute(f'CREATE SCHEMA "{self.schema}" AUTHORIZATION "{self.owner}"')
        options = f"-c search_path={self.schema}"
        self.owner_dsn = make_conninfo(DSN, user=self.owner, password=PASSWORD, options=options)
        self.runtime_dsn = make_conninfo(DSN, user=self.runtime, password=PASSWORD, options=options)
        self.opened: list[PostgresAuthorityStore] = []
        owner = self.open_owner()
        grant_authority_runtime_role(owner._conn, self.runtime)

    def open_owner(self, clock=None) -> PostgresAuthorityStore:
        store = PostgresAuthorityStore(self.owner_dsn, **({"clock": clock} if clock else {}))
        self.opened.append(store)
        return store

    def open(self, clock=None) -> PostgresAuthorityStore:
        store = PostgresAuthorityStore(self.runtime_dsn, migrate=False, **({"clock": clock} if clock else {}))
        self.opened.append(store)
        return store

    def prepare_scratch(self) -> None:
        """A scratch table the runtime role may only read and insert into (it may not create tables)."""

        with self.raw(as_owner=True) as owner:
            owner.execute("CREATE TABLE scratch(v TEXT)")
            owner.execute(f'GRANT SELECT, INSERT ON scratch TO "{self.runtime}"')

    def raw(self, as_owner: bool = False):
        return psycopg.connect(self.owner_dsn if as_owner else self.runtime_dsn, autocommit=True)

    def close(self) -> None:
        for store in self.opened:
            store.close()
        self.opened.clear()
        with psycopg.connect(DSN, autocommit=True) as admin:
            admin.execute(f'DROP SCHEMA IF EXISTS "{self.schema}" CASCADE')
            for role in (self.runtime, self.owner):
                admin.execute(f'DROP OWNED BY "{role}"')
                admin.execute(f'DROP ROLE IF EXISTS "{role}"')


@pytest.fixture(autouse=True)
def require_pg():
    """Import into an integration module: no PostgreSQL is a skip, or a failure when it is required."""

    if not DSN:
        if REQUIRED:
            pytest.fail("NS_INTEGRATION_REQUIRED=1 but AUTHORITY_TEST_PG_DSN is not set")
        pytest.skip(REASON)
