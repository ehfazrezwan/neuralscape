"""Retention scenarios against PostgreSQL as the non-superuser runtime role.

Needs ``LEDGER_TEST_PG_DSN`` (same contract as the other ledger integration modules); skipped, with the
declared reason, when it is unset.  Tombstone pruning runs under a separate maintenance role, because the
runtime role can only insert tombstones.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import pytest

psycopg = pytest.importorskip("psycopg")
from psycopg import errors  # noqa: E402
from psycopg.conninfo import make_conninfo  # noqa: E402

from ledger.postgres import PostgresLedger, grant_retention_role  # noqa: E402
from ledger.retention import DAY, RetentionPolicy, RetentionSweeper  # noqa: E402
from tests.test_ledger_dispatch import External  # noqa: E402
from tests.test_ledger_postgres_integration import PASSWORD, PostgresStore, _STORES  # noqa: E402
from tests.test_ledger_retention import SCENARIOS, T, make, run_to_terminal  # noqa: E402
from tests.test_ledger_retention_fixes import FIX_SCENARIOS  # noqa: E402

DSN = os.environ.get("LEDGER_TEST_PG_DSN")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not DSN, reason="LEDGER_TEST_PG_DSN not set (postgres ledger integration)"),
]


class RetentionPostgresStore(PostgresStore):
    def __init__(self, root: Path) -> None:
        super().__init__()
        self.root = root
        tag = uuid.uuid4().hex[:8]
        self.maintenance = f"lm_{tag}"
        with psycopg.connect(DSN, autocommit=True) as admin:
            admin.execute(f'CREATE ROLE "{self.maintenance}" LOGIN NOSUPERUSER NOBYPASSRLS PASSWORD \'{PASSWORD}\'')
        self.maintenance_dsn = make_conninfo(
            DSN, user=self.maintenance, password=PASSWORD, options=f"-c search_path={self.schema}"
        )
        grant_retention_role(self.open_admin()._conn, self.maintenance)

    def external(self) -> External:
        return External(self.root / "external.db")

    def age(self, seconds: float) -> None:
        with psycopg.connect(self.owner_dsn, autocommit=True) as conn:
            for table, column in (
                ("intents", "updated_at"), ("intents", "accepted_at"), ("stage_receipts", "recorded_at"),
                ("tombstones", "recorded_at"), ("effects", "recorded_at"), ("object_refs", "recorded_at"),
            ):
                conn.execute(f"UPDATE {table} SET {column} = {column} - make_interval(secs => %s)", (seconds,))

    def payload_store(self):
        from write_producer import PostgresPayloadStore

        store = PostgresPayloadStore(self.owner_dsn)  # the table is created by whoever opens the store first
        self.opened_payloads = getattr(self, "opened_payloads", []) + [store]
        return store

    def open_pruner(self) -> PostgresLedger:
        """The runtime role for everything except pruning tombstones, which the maintenance role may do."""

        ledger = PostgresLedger(self.maintenance_dsn, migrate=False)
        self.opened.append(ledger)
        return ledger

    def fresh_target(self):
        target = PostgresStore()

        class Admin:
            def open(self_inner):
                return target.open_admin()

        return Admin()


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
    with psycopg.connect(DSN, autocommit=True) as admin:
        for store in _STORES:
            store.close()
            admin.execute(f'DROP SCHEMA IF EXISTS "{store.schema}" CASCADE')
            for role in (store.runtime, store.owner, getattr(store, "maintenance", None)):
                if role and admin.execute("SELECT 1 FROM pg_roles WHERE rolname=%s", (role,)).fetchone():
                    admin.execute(f'DROP OWNED BY "{role}"')
                    admin.execute(f'DROP ROLE IF EXISTS "{role}"')
    _STORES.clear()


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_retention_scenarios_on_postgres_as_runtime_role(name: str, tmp_path: Path) -> None:
    SCENARIOS[name](RetentionPostgresStore(tmp_path))


@pytest.mark.parametrize("name", sorted(FIX_SCENARIOS))
def test_retention_fixes_on_postgres_as_runtime_role(name: str, tmp_path: Path) -> None:
    FIX_SCENARIOS[name](RetentionPostgresStore(tmp_path))


def test_the_runtime_role_cannot_delete_tombstones_but_the_maintenance_role_can(tmp_path: Path) -> None:
    store = RetentionPostgresStore(tmp_path)
    runtime = store.open()
    runtime.tombstone(T, "mem-gone", 3, "user_delete")
    store.age(400 * DAY)
    policy = RetentionPolicy(tombstone_seconds=120 * DAY, backup_expiry_seconds=90 * DAY)
    with pytest.raises(errors.InsufficientPrivilege):
        RetentionSweeper(runtime, policy, fences_only=True).sweep(T)  # fences are not deletable by the ordinary role
    assert runtime.get_tombstone(T, "mem-gone") is not None
    assert RetentionSweeper(store.open_pruner(), policy, fences_only=True).sweep(T).tombstones_pruned == 1
    assert runtime.get_tombstone(T, "mem-gone") is None


def test_the_installation_checks_and_snapshot_digest_are_unaffected_by_pruning(tmp_path: Path) -> None:
    store = RetentionPostgresStore(tmp_path)
    ledger = store.open()
    run_to_terminal(store, ledger, "v1", "v2")
    admin = store.open_admin()
    assert admin.verify_installation(store.runtime) == []
    store.age(60 * DAY)
    report = RetentionSweeper(ledger, RetentionPolicy()).sweep(T)
    assert report.idempotency_released == 2 and report.content_compacted == 2
    assert admin.verify_installation(store.runtime) == []  # no role privilege or function drifted
    snapshot = ledger.snapshot()
    target = PostgresStore().open_admin()
    target.restore(snapshot, expected_digest=snapshot["digest"])
    assert target.snapshot()["digest"] == snapshot["digest"]


def test_the_postgres_payload_store_deletes_an_intent_and_its_blobs_only(tmp_path: Path) -> None:
    from write_producer import PostgresPayloadStore

    store = RetentionPostgresStore(tmp_path)
    payloads = PostgresPayloadStore(store.owner_dsn)  # the table is created by whoever opens the store first
    try:
        payloads.put("intent-abc", {"x": 1})
        payloads.put("intent-abc:vector-write-started:1", {})
        payloads.put("intent-abcdef", {"x": 2})
        assert payloads.delete_intent("intent-abc") == 2
        assert payloads.get("intent-abcdef") == {"x": 2} and payloads.get("intent-abc") is None
    finally:
        payloads.close()
