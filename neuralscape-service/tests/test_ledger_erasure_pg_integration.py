"""Erasure scenarios against PostgreSQL as the non-superuser runtime role.

Needs ``LEDGER_TEST_PG_DSN``; skipped with the declared reason when unset.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytest.importorskip("psycopg")
import psycopg  # noqa: E402

from ledger.conformance import make_intent, ref  # noqa: E402
from tests.test_ledger_dispatch_pg_integration import DispatchPostgresStore  # noqa: E402
from tests.test_ledger_dispatch_pg_integration import (  # noqa: E402,F401
    _close_connections_after_each_test,
    _drop_everything_at_end,
)
from tests.test_ledger_erasure import SCENARIOS, good_owner, erase  # noqa: E402

DSN = os.environ.get("LEDGER_TEST_PG_DSN")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not DSN, reason="LEDGER_TEST_PG_DSN not set (postgres ledger integration)"),
]


class ErasurePostgresStore(DispatchPostgresStore):
    def sql_error(self, sql: str) -> BaseException | None:
        with self.raw(as_owner=True) as conn:  # the table owner, the strongest non-superuser path
            try:
                conn.execute(sql)
                return None
            except psycopg.Error as exc:
                return exc

    def fresh(self) -> "ErasurePostgresStore":
        return ErasurePostgresStore(self.root)


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_erasure_scenarios_on_postgres_as_runtime_role(name: str, tmp_path: Path) -> None:
    SCENARIOS[name](ErasurePostgresStore(tmp_path))


def test_runtime_role_cannot_rewrite_audit_holds_or_tombstones(tmp_path: Path) -> None:
    store = ErasurePostgresStore(tmp_path)
    ledger = store.open()
    ledger.accept(make_intent("a"), [ref("rec-1")])
    ledger.place_hold("tenant-1", "hold-1", "rec-1", "legal", "counsel", "cred-c")
    from tests.test_ledger_erasure import RECORD

    erase(ledger, good_owner(), record=RECORD)
    with store.raw() as conn:
        for statement in (
            "UPDATE erasure_audit SET actor_id='x'",
            "DELETE FROM erasure_audit",
            "DELETE FROM retention_holds",
            "DELETE FROM tombstones",
            "UPDATE tombstones SET reason_code='x'",
        ):
            with pytest.raises(psycopg.errors.InsufficientPrivilege):
                conn.execute(statement)
        assert conn.execute("SELECT count(*) FROM erasure_audit").fetchone()[0] == 1


def test_even_the_schema_owner_cannot_truncate_the_audit(tmp_path: Path) -> None:
    """H-B2-4: row triggers do not fire on TRUNCATE."""

    store = ErasurePostgresStore(tmp_path)
    ledger = store.open()
    ledger.accept(make_intent("a"), [ref("rec-1")])
    from tests.test_ledger_erasure import RECORD

    erase(ledger, good_owner(), record=RECORD)
    error = store.sql_error("TRUNCATE erasure_audit")
    assert error is not None and "append-only" in str(error)
    assert len(ledger.audit_records("tenant-1")) == 1
