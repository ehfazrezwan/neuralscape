"""PostgreSQL ledger conformance and role checks (integration; needs LEDGER_TEST_PG_DSN).

The DSN must point at a disposable PostgreSQL 16 server and a superuser-capable
role (used only to create schemas and roles).  Every store gets its own schema,
a non-superuser owner role and a non-superuser runtime role; the conformance
suite runs as the runtime role.  Skipped when the variable is unset; that skip
is the declared one.
"""

from __future__ import annotations

import os
import threading
import uuid

import pytest

psycopg = pytest.importorskip("psycopg")
from psycopg.conninfo import make_conninfo  # noqa: E402

from ledger.conformance import CASES, make_intent  # noqa: E402
from ledger.port import DurabilityError, PrivilegeError, SchemaError, load_migrations  # noqa: E402
from ledger.postgres import FUNCTIONS, PostgresLedger, grant_runtime_role  # noqa: E402
from ledger.sqlite import SqliteLedger  # noqa: E402

DSN = os.environ.get("LEDGER_TEST_PG_DSN")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not DSN, reason="LEDGER_TEST_PG_DSN not set (postgres ledger integration)"),
]

PASSWORD = "ledger-test-pw"
_STORES: list["PostgresStore"] = []


class PostgresStore:
    def __init__(self) -> None:
        tag = uuid.uuid4().hex[:10]
        self.schema, self.owner, self.runtime = f"ledger_{tag}", f"lo_{tag}", f"lr_{tag}"
        with psycopg.connect(DSN, autocommit=True) as admin:
            for role in (self.owner, self.runtime):
                admin.execute(f'CREATE ROLE "{role}" LOGIN NOSUPERUSER NOBYPASSRLS PASSWORD \'{PASSWORD}\'')
            admin.execute(f'CREATE SCHEMA "{self.schema}" AUTHORIZATION "{self.owner}"')
        options = f"-c search_path={self.schema}"
        self.owner_dsn = make_conninfo(DSN, user=self.owner, password=PASSWORD, options=options)
        self.runtime_dsn = make_conninfo(DSN, user=self.runtime, password=PASSWORD, options=options)
        self.opened: list[PostgresLedger] = []
        owner = self.open_admin()
        grant_runtime_role(owner._conn, self.runtime)
        _STORES.append(self)

    def open_admin(self) -> PostgresLedger:
        ledger = PostgresLedger(self.owner_dsn)
        self.opened.append(ledger)
        return ledger

    def open(self) -> PostgresLedger:
        ledger = PostgresLedger(self.runtime_dsn, migrate=False)
        self.opened.append(ledger)
        return ledger

    def fresh(self) -> "PostgresStore":
        return PostgresStore()

    def shift_leases(self, seconds: float) -> None:
        with psycopg.connect(self.owner_dsn, autocommit=True) as conn:
            conn.execute(
                "UPDATE outbox SET lease_until = lease_until + make_interval(secs => %s) WHERE state='leased'",
                (seconds,),
            )

    def raw(self, as_owner: bool = False):
        return psycopg.connect(self.owner_dsn if as_owner else self.runtime_dsn, autocommit=True)

    def close(self) -> None:
        for ledger in self.opened:
            ledger.close()
        self.opened.clear()


@pytest.fixture(autouse=True, scope="module")
def _drop_everything():
    yield
    if not DSN:
        return
    with psycopg.connect(DSN, autocommit=True) as admin:
        for store in _STORES:
            store.close()
            admin.execute(f'DROP SCHEMA IF EXISTS "{store.schema}" CASCADE')
            for role in (store.runtime, store.owner):
                admin.execute(f'DROP OWNED BY "{role}"')
                admin.execute(f'DROP ROLE IF EXISTS "{role}"')
    _STORES.clear()


@pytest.fixture(autouse=True)
def _close_connections_after_each_test():
    yield
    for store in _STORES:
        store.close()


@pytest.mark.parametrize("name", sorted(CASES))
def test_conformance_as_non_superuser_runtime_role(name: str) -> None:
    store = PostgresStore()
    with store.raw() as conn:  # the suite really runs unprivileged
        assert conn.execute(
            "SELECT rolsuper OR rolbypassrls FROM pg_roles WHERE rolname=current_user"
        ).fetchone()[0] is False
    CASES[name](store)


def test_migrations_apply_once_under_concurrent_owner_open() -> None:
    store = PostgresStore()
    errors: list[BaseException] = []

    def opener() -> None:
        try:
            store.open_admin()
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=opener) for _ in range(6)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors
    assert store.open().schema_version() == len(load_migrations("postgres"))


def test_runtime_role_has_no_direct_outbox_mutation_or_extra_tables() -> None:
    store = PostgresStore()
    store.open().accept(make_intent())
    with store.raw() as conn:
        assert conn.execute("SELECT count(*) FROM outbox").fetchone()[0] == 1
        for statement in (
            "UPDATE outbox SET state='done'",
            "DELETE FROM outbox",
            "DELETE FROM intents",
            "UPDATE effects SET intent_id='x'",
            "TRUNCATE intents CASCADE",
            "CREATE TABLE sneaky(x int)",
        ):
            with pytest.raises(psycopg.errors.InsufficientPrivilege):
                conn.execute(statement)


def test_superuser_or_bypassrls_runtime_connection_is_refused() -> None:
    store = PostgresStore()
    with pytest.raises(PrivilegeError):
        PostgresLedger(DSN, migrate=False)  # the harness DSN is privileged
    with psycopg.connect(DSN, autocommit=True) as admin:
        admin.execute(f'ALTER ROLE "{store.runtime}" BYPASSRLS')
    with pytest.raises(PrivilegeError):
        store.open()


def test_runtime_refuses_missing_or_mismatched_schema() -> None:
    store = PostgresStore()
    with store.raw(as_owner=True) as conn:
        conn.execute("UPDATE ledger_schema_version SET version = version + 100")
    with pytest.raises(SchemaError, match="does not match"):
        store.open()


def test_definer_functions_are_tenant_scoped_and_not_shadowable() -> None:
    store = PostgresStore()
    ledger = store.open()
    ledger.accept(make_intent("a", tenant="tenant-1"))
    claim = ledger.claim("w", 30, tenant_id="tenant-1")
    with store.raw() as conn:
        args = (claim.outbox_id, claim.owner, claim.attempt)
        assert conn.execute("SELECT ledger_complete('tenant-2', %s, %s, %s)", args).fetchone()[0] is False
        assert conn.execute("SELECT ledger_extend('tenant-2', %s, %s, %s, 5)", args).fetchone()[0] is None
        assert conn.execute("SELECT ledger_check_claim('tenant-2', %s, %s, %s)", args).fetchone()[0] is False
        assert conn.execute("SELECT count(*) FROM ledger_claim('tenant-2', 'x', 5)").fetchone()[0] == 0
        with pytest.raises(psycopg.errors.RaiseException):
            conn.execute("SELECT ledger_close_outbox('tenant-1', 'intent-a', 'pending')")
        # a temp table with the same name must not hijack the definer functions
        conn.execute(
            "CREATE TEMP TABLE outbox(id bigint, intent_id text, tenant_id text, state text, queued boolean,"
            " lease_owner text, lease_until timestamptz, attempt int)"
        )
        conn.execute("INSERT INTO pg_temp.outbox VALUES (99,'evil','tenant-1','pending',false,NULL,NULL,0)")
        assert conn.execute("SELECT count(*) FROM ledger_claim('tenant-1', 'evil', 5)").fetchone()[0] == 0
        assert conn.execute("SELECT ledger_complete('tenant-1', %s, %s, %s)", args).fetchone()[0] is True
        assert conn.execute("SELECT count(*) FROM pg_temp.outbox WHERE state <> 'pending'").fetchone()[0] == 0
    with store.raw(as_owner=True) as conn:
        assert conn.execute("SELECT state FROM outbox").fetchone()[0] == "done"


def test_verify_installation_passes_and_detects_regressions() -> None:
    store = PostgresStore()
    owner = store.open_admin()
    assert owner.verify_installation(store.runtime) == []
    with store.raw(as_owner=True) as conn:
        conn.execute(f'GRANT UPDATE ON outbox TO "{store.runtime}"')
        conn.execute("GRANT EXECUTE ON FUNCTION ledger_claimable_tenants() TO PUBLIC")
        conn.execute(f'REVOKE EXECUTE ON FUNCTION ledger_claim(text,text,double precision) FROM "{store.runtime}"')
        conn.execute("ALTER FUNCTION ledger_release(text,bigint,text,integer) SECURITY INVOKER")
        conn.execute("ALTER FUNCTION ledger_extend(text,bigint,text,integer,double precision) RESET search_path")
    problems = "\n".join(owner.verify_installation(store.runtime))
    for expected in (
        "holds UPDATE on outbox",
        "ledger_claimable_tenants() is executable by PUBLIC",
        "cannot execute ledger_claim",
        "ledger_release(text,bigint,text,integer) is not SECURITY DEFINER",
        "ledger_extend(text,bigint,text,integer,double precision) lacks fixed search_path",
    ):
        assert expected in problems, (expected, problems)
    assert len(FUNCTIONS) == 10


def test_durability_settings_are_enforced_for_postgres() -> None:
    store = PostgresStore()
    weak = make_conninfo(store.runtime_dsn, options=f"-c search_path={store.schema} -c synchronous_commit=off")
    with pytest.raises(DurabilityError, match="synchronous_commit=off"):
        PostgresLedger(weak, migrate=False)
    ledger = PostgresLedger(store.runtime_dsn, migrate=False, durability_check_interval=0)
    ledger._conn.execute("SET synchronous_commit = off")
    with pytest.raises(DurabilityError):
        ledger.accept(make_intent())
    ledger._conn.execute("SET synchronous_commit = on")
    assert ledger.accept(make_intent()).created
    ledger.close()


def test_logical_backup_restore_keeps_roles_functions_and_version(tmp_path) -> None:
    source = PostgresStore()
    ledger = source.open()
    for i in range(3):
        ledger.accept(make_intent(str(i)))
    ledger.claim("w", 30)
    ledger.tombstone("tenant-1", "rec", 3, "admin_delete")
    snapshot = ledger.snapshot()

    target = PostgresStore()
    target.open_admin().restore(snapshot)
    assert target.open_admin().verify_installation(target.runtime) == []  # post-restore checks
    runtime = target.open()
    assert runtime.get_tombstone("tenant-1", "rec").reason_code == "admin_delete"
    assert runtime.snapshot()["tables"]["intents"] == snapshot["tables"]["intents"]
    assert runtime.claim("restored", 30) is not None  # leases were reset, definer path still works
    with pytest.raises(PrivilegeError):
        runtime.restore(snapshot)  # the runtime role cannot restore

    lite = SqliteLedger(str(tmp_path / "x.db"))
    lite.restore(snapshot)
    assert lite.snapshot()["tables"]["intents"] == snapshot["tables"]["intents"]
    again = PostgresStore()
    again.open_admin().restore(lite.snapshot())
    assert again.open().get_intent("tenant-1", "intent-2") is not None
    assert again.open_admin().verify_installation(again.runtime) == []
    lite.close()
