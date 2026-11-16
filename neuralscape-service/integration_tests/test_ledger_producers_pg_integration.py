"""The write producers on PostgreSQL as the non-superuser runtime role (P-08 contract scenarios).

Needs ``LEDGER_TEST_PG_DSN`` (a disposable PostgreSQL); a failure under ``NS_INTEGRATION_REQUIRED``, a skip
elsewhere.  The memory service is the unit suite's labelled stand-in; the ledger, the idempotency records, the
payload store and the receipts are real.
"""

from __future__ import annotations

import os

import pytest

from ledger_env import require_env
from test_ledger_postgres_integration import (  # noqa: F401  (module fixtures: cleanup of roles and schemas)
    PostgresStore,
    _close_connections_after_each_test,
    _drop_everything,
)
from tests.test_producers_ledger import identical_writes_without_a_key_scenario
from write_producer import PostgresPayloadStore

DSN = os.environ.get("LEDGER_TEST_PG_DSN")


@pytest.fixture(autouse=True)
def _fail_when_required_store_is_missing():
    if not DSN:
        require_env(**{"PostgreSQL (LEDGER_TEST_PG_DSN)": DSN})


def test_identical_writes_without_a_key_are_distinct_intents_and_a_resave_after_a_delete_runs() -> None:
    store = PostgresStore()
    payloads = PostgresPayloadStore(store.owner_dsn)
    try:
        identical_writes_without_a_key_scenario(store.open(), payloads)
    finally:
        payloads.close()
