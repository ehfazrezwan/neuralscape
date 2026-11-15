"""Dispatcher scenarios against PostgreSQL as the non-superuser runtime role.

Needs ``LEDGER_TEST_PG_DSN`` (same contract as ``test_ledger_postgres_integration``);
skipped, with the declared reason, when it is unset.  The external system under
test is the same file-backed idempotent store used by the SQLite scenarios.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytest.importorskip("psycopg")

from tests.test_ledger_dispatch import SCENARIOS, External  # noqa: E402
from tests.test_ledger_postgres_integration import (  # noqa: E402
    PostgresStore,
    _STORES,
)

DSN = os.environ.get("LEDGER_TEST_PG_DSN")
REQUIRED = os.environ.get("NS_INTEGRATION_REQUIRED") == "1"


@pytest.fixture(autouse=True)
def _fail_when_required_store_is_missing():
    """With NS_INTEGRATION_REQUIRED=1 a missing store is a failure, not a skip."""
    if REQUIRED and not DSN:
        pytest.fail("NS_INTEGRATION_REQUIRED=1 but LEDGER_TEST_PG_DSN is not set")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not DSN and not REQUIRED, reason="LEDGER_TEST_PG_DSN not set (postgres ledger integration)"),
]


class DispatchPostgresStore(PostgresStore):
    def __init__(self, root: Path) -> None:
        super().__init__()
        self.root = root

    def external(self) -> External:
        return External(self.root / "external.db")


@pytest.fixture(autouse=True)
def _close_connections_after_each_test():
    yield
    for store in _STORES:
        store.close()


@pytest.fixture(autouse=True, scope="module")
def _drop_everything_at_end():
    yield
    if not DSN:
        return
    import psycopg

    with psycopg.connect(DSN, autocommit=True) as admin:
        for store in _STORES:
            store.close()
            admin.execute(f'DROP SCHEMA IF EXISTS "{store.schema}" CASCADE')
            for role in (store.runtime, store.owner):
                if not admin.execute("SELECT 1 FROM pg_roles WHERE rolname=%s", (role,)).fetchone():
                    continue
                admin.execute(f'DROP OWNED BY "{role}"')
                admin.execute(f'DROP ROLE IF EXISTS "{role}"')
    _STORES.clear()


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_dispatch_scenarios_on_postgres_as_runtime_role(name: str, tmp_path: Path) -> None:
    SCENARIOS[name](DispatchPostgresStore(tmp_path))
