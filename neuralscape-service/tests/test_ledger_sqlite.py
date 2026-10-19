"""SQLite ledger: shared conformance suite plus embedded-profile specifics."""

from __future__ import annotations

import itertools
import sqlite3
from pathlib import Path

import pytest

from config import Settings
from contracts_lifecycle import StageStatus
from ledger import open_ledger
from ledger import port
from ledger.conformance import CASES, make_intent
from ledger.conformance import make_receipt
from ledger.port import DurabilityError, SchemaError, StaleLease
from ledger.sqlite import SqliteLedger


class SqliteStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self._ids = itertools.count()
        self.path = root / f"ledger-{next(self._ids)}.db"
        self._open: list[SqliteLedger] = []

    def open(self) -> SqliteLedger:
        ledger = SqliteLedger(str(self.path))
        self._open.append(ledger)
        return ledger

    def open_admin(self) -> SqliteLedger:
        return self.open()

    def shift_leases(self, seconds: float) -> None:
        raw = sqlite3.connect(self.path, isolation_level=None, timeout=30)
        raw.execute("UPDATE outbox SET lease_until = lease_until + ? WHERE state='leased'", (seconds,))
        raw.close()

    def fresh(self) -> "SqliteStore":
        other = SqliteStore(self.root / f"s{next(self._ids)}")
        other.root.mkdir()
        return other

    def close(self) -> None:
        for ledger in self._open:
            ledger.close()


@pytest.mark.parametrize("name", sorted(CASES))
def test_conformance(name: str, tmp_path: Path) -> None:
    store = SqliteStore(tmp_path)
    try:
        CASES[name](store)
    finally:
        store.close()


def test_normal_synchronous_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="synchronous=FULL"):
        SqliteLedger(str(tmp_path / "x.db"), synchronous="NORMAL")


def test_reopen_is_idempotent_and_versioned(tmp_path: Path) -> None:
    path = str(tmp_path / "x.db")
    first = SqliteLedger(path)
    first.accept(make_intent())
    first.close()
    second = SqliteLedger(path)
    assert second.schema_version() == len(port.load_migrations("sqlite"))
    assert second.get_intent("tenant-1", "intent-1") is not None
    second.close()


def test_injected_clock_controls_lease_expiry(tmp_path: Path) -> None:
    now = [1000.0]
    ledger = SqliteLedger(str(tmp_path / "x.db"), clock=lambda: now[0])
    ledger.accept(make_intent())
    claim = ledger.claim("a", 10)
    assert claim is not None and claim.lease_expires_at.timestamp() == 1010.0
    now[0] = 1009.9
    assert ledger.claim("b", 10) is None
    now[0] = 1010.1
    stolen = ledger.claim("b", 10)
    assert stolen is not None and stolen.attempt == 2
    ledger.close()


def test_modified_applied_migration_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "x.db"
    SqliteLedger(str(path)).close()
    raw = sqlite3.connect(path)
    raw.execute("UPDATE ledger_schema_version SET checksum='tampered' WHERE version=1")
    raw.commit()
    raw.close()
    with pytest.raises(SchemaError, match="modified"):
        SqliteLedger(str(path))


def test_database_newer_than_code_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "x.db"
    SqliteLedger(str(path)).close()
    raw = sqlite3.connect(path)
    raw.execute("INSERT INTO ledger_schema_version VALUES(99, 'future', 'x', 0)")
    raw.commit()
    raw.close()
    with pytest.raises(SchemaError, match="newer"):
        SqliteLedger(str(path))


@pytest.mark.parametrize("dialect", ["sqlite", "postgres"])
def test_migrations_are_contiguous_and_expand_only(dialect: str) -> None:
    migrations = port.load_migrations(dialect)
    assert [m.version for m in migrations] == list(range(1, len(migrations) + 1))
    for migration in migrations:
        port.assert_expand_only(migration)


@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE intents;",
        "ALTER TABLE intents DROP COLUMN body;",
        "ALTER TABLE intents RENAME TO old_intents;",
        "ALTER TABLE intents ALTER COLUMN body TYPE bytea;",
        "DELETE FROM intents;",
        "TRUNCATE intents;",
        "DROP/**/TABLE intents;",
        "DROP TRIGGER t;",
        "DROP VIEW v;",
        "DROP FUNCTION f();",
        "DROP TYPE x;",
        "ALTER TABLE \"intents\" DROP COLUMN body;",
        "ALTER TABLE s.intents RENAME COLUMN a TO b;",
    ],
)
def test_expand_only_lint_rejects_destructive_sql(sql: str) -> None:
    migration = port.Migration(2, "bad", sql, "x")
    with pytest.raises(SchemaError, match="expand-only"):
        port.assert_expand_only(migration)
    assert port.assert_expand_only(port.Migration(2, "ok", "-- DROP TABLE x\nALTER TABLE t ADD COLUMN c TEXT;", "x")) is None


def test_dialects_define_the_same_tables_and_columns(tmp_path: Path) -> None:
    import re

    def tables(dialect: str) -> dict[str, list[str]]:
        sql = "\n".join(m.sql for m in port.load_migrations(dialect))
        out = {}
        for name, body in re.findall(r"CREATE TABLE (\w+) \((.*?)\n\);", sql, re.S):
            cols = [ln.split()[0] for ln in body.splitlines() if ln.strip() and ln.split()[0].islower()
                    and ln.split()[0] not in {"unique", "primary", "check", "foreign"}]
            out[name] = cols
        return out

    sqlite_tables, postgres_tables = tables("sqlite"), tables("postgres")
    assert sqlite_tables == postgres_tables
    assert {t: tuple(c) for t, c in sqlite_tables.items()} == {
        t: tuple(c for c in port.SNAPSHOT_COLUMNS[t]) for t in port.SNAPSHOT_COLUMNS
    } or all(set(port.SNAPSHOT_COLUMNS[t]) == set(c) for t, c in sqlite_tables.items())


def test_native_backup_restores_equivalent_state(tmp_path: Path) -> None:
    ledger = SqliteLedger(str(tmp_path / "x.db"))
    ledger.accept(make_intent("1"))
    ledger.tombstone("tenant-1", "rec", 1, "admin_delete")
    ledger.backup_to(str(tmp_path / "copy.db"))
    copy = SqliteLedger(str(tmp_path / "copy.db"))
    assert copy.get_intent("tenant-1", "intent-1") is not None
    assert copy.get_tombstone("tenant-1", "rec") is not None
    ledger.close()
    copy.close()


def test_ledger_is_disabled_by_default_and_factory_validates(tmp_path: Path) -> None:
    settings = Settings()
    assert settings.ledger_enabled is False
    with pytest.raises(RuntimeError, match="disabled"):
        open_ledger(settings)
    enabled = Settings(ledger_enabled=True, ledger_backend="sqlite", ledger_sqlite_path="")
    with pytest.raises(ValueError, match="LEDGER_SQLITE_PATH"):
        open_ledger(enabled)
    with pytest.raises(ValueError, match="LEDGER_POSTGRES_DSN"):
        open_ledger(Settings(ledger_enabled=True, ledger_backend="postgres"))
    with pytest.raises(ValueError, match="LEDGER_BACKEND"):
        open_ledger(Settings(ledger_enabled=True, ledger_backend="redis"))
    ledger = open_ledger(
        Settings(ledger_enabled=True, ledger_backend="sqlite", ledger_sqlite_path=str(tmp_path / "f.db"))
    )
    assert ledger.schema_version() >= 1
    ledger.close()


def test_durability_is_asserted_at_startup_and_before_accepts(tmp_path: Path) -> None:
    ledger = SqliteLedger(str(tmp_path / "d.db"), durability_check_interval=0)
    ledger.check_durability()
    ledger._conn.execute("PRAGMA synchronous=NORMAL")  # something weakened the connection
    with pytest.raises(DurabilityError, match="synchronous=1"):
        ledger.check_durability()
    with pytest.raises(DurabilityError):
        ledger.accept(make_intent())  # refuses to acknowledge
    assert ledger.scan_unqueued() == ()
    ledger._conn.execute("PRAGMA synchronous=FULL")
    assert ledger.accept(make_intent()).created
    ledger.close()


def test_durability_check_is_throttled_but_catches_later_drift(tmp_path: Path) -> None:
    ledger = SqliteLedger(str(tmp_path / "d.db"), durability_check_interval=3600)
    ledger.accept(make_intent("1"))
    ledger._conn.execute("PRAGMA synchronous=OFF")
    ledger.accept(make_intent("2"))  # inside the throttle window the drift is not yet seen
    with pytest.raises(DurabilityError):
        ledger.check_durability()  # an explicit check always re-reads
    ledger.close()


def test_postgres_durability_rules_are_pure_and_strict() -> None:
    from ledger.postgres import durability_violations

    good = {"fsync": "on", "synchronous_commit": "on", "full_page_writes": "on"}
    assert durability_violations(good) == []
    assert durability_violations({**good, "synchronous_commit": "remote_apply"}) == []
    for name, bad in (("fsync", "off"), ("synchronous_commit", "off"),
                      ("synchronous_commit", "local"), ("full_page_writes", "off")):
        problems = durability_violations({**good, name: bad})
        assert len(problems) == 1 and name in problems[0]
    assert len(durability_violations({})) == 3


def test_forward_clock_step_steals_early_but_fencing_protects(tmp_path: Path) -> None:
    now = [5000.0]
    ledger = SqliteLedger(str(tmp_path / "x.db"), clock=lambda: now[0])
    intent = make_intent()
    ledger.accept(intent)
    held = ledger.claim("A", 30)
    now[0] += 3600  # NTP step forward: the live lease reads as expired
    stolen = ledger.claim("B", 30)
    assert stolen is not None and stolen.attempt == 2
    with pytest.raises(StaleLease):
        ledger.complete(held)
    with pytest.raises(StaleLease):
        ledger.record_receipt(
            make_receipt(intent, intent.stage_requirements[0].stage, StageStatus.PROCESSING),
            "tenant-1", held,
        )
    now[0] -= 7200  # step backwards: B's lease now outlives its grant but nobody else can claim
    assert ledger.claim("C", 30) is None
    ledger.complete(stolen)
    ledger.close()


def test_many_connections_can_open_a_fresh_database_concurrently(tmp_path: Path) -> None:
    import threading

    path = str(tmp_path / "fresh.db")
    errors: list[BaseException] = []
    opened: list[SqliteLedger] = []

    def opener() -> None:
        try:
            opened.append(SqliteLedger(path))
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=opener) for _ in range(16)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors, errors
    assert {ledger.schema_version() for ledger in opened} == {len(port.load_migrations("sqlite"))}
    for ledger in opened:
        ledger.close()


def test_both_backends_share_one_schema_version() -> None:
    # A snapshot carries the schema version and restores only into the same version, so a
    # schema change on one backend needs a (possibly empty) migration on the other.
    assert len(port.load_migrations("sqlite")) == len(port.load_migrations("postgres"))


def test_snapshot_digest_covers_counts_version_and_rows() -> None:
    base = {"format": "f", "schema_version": 1, "counts": {"a": 1}, "tables": {"a": [{"x": 1}]}}
    digest = port.snapshot_digest(base)
    assert digest == port.snapshot_digest(dict(base, digest="ignored"))
    for change in (
        {"schema_version": 2},
        {"counts": {"a": 2}},
        {"tables": {"a": [{"x": 2}]}},
        {"tables": {"a": []}},
    ):
        assert port.snapshot_digest(dict(base, **change)) != digest


class _LockedWalConnection:
    """Proxy that fails the first ``failures`` WAL switches like a concurrent first open does."""

    def __init__(self, real, state):
        self._real, self._state = real, state

    def execute(self, sql, *args):
        if sql.strip().upper() == "PRAGMA JOURNAL_MODE=WAL" and self._state["left"] > 0:
            self._state["left"] -= 1
            self._state["seen"] += 1
            raise sqlite3.OperationalError("database is locked")
        return self._real.execute(sql, *args)

    def __getattr__(self, name):
        return getattr(self._real, name)


def _patched_connect(monkeypatch, failures):
    state = {"left": failures, "seen": 0}
    real_connect = sqlite3.connect
    monkeypatch.setattr(sqlite3, "connect", lambda *a, **k: _LockedWalConnection(real_connect(*a, **k), state))
    return state


def test_a_contended_wal_switch_is_retried_deterministically(tmp_path, monkeypatch) -> None:
    # Root cause of the intermittent idempotency_race failure: two connections switching a
    # fresh file to WAL at once make SQLite fail one of them immediately ("database is
    # locked", the busy handler is not consulted).  The open path must wait it out.
    state = _patched_connect(monkeypatch, failures=4)
    ledger = SqliteLedger(str(tmp_path / "x.db"))
    assert state["seen"] == 4  # all four injected failures were absorbed by retries
    assert ledger.accept(make_intent()).created
    ledger.close()


def test_a_wal_switch_that_never_succeeds_fails_loudly_after_the_deadline(tmp_path, monkeypatch) -> None:
    _patched_connect(monkeypatch, failures=10**6)
    with pytest.raises(sqlite3.OperationalError, match="database is locked"):
        SqliteLedger(str(tmp_path / "x.db"), wal_retry_seconds=0.2)


# ---- the ledger, its payloads and its backups are readable only by the service user


def _mode(path) -> int:
    import os
    import stat

    return stat.S_IMODE(os.stat(path).st_mode)


def test_the_ledger_and_its_payload_store_are_private_under_any_umask(tmp_path) -> None:
    import os

    from write_producer import SqlitePayloadStore

    old = os.umask(0)
    try:
        ledger = SqliteLedger(str(tmp_path / "ledger.db"))
        ledger.accept(make_intent("a"))
        payloads = SqlitePayloadStore(str(tmp_path / "ledger.db.payloads"))
        payloads.put("intent-a", {"content": "private text"})
        ledger.backup_to(str(tmp_path / "backup.db"))
        files = sorted(name for name in os.listdir(tmp_path))
        assert {name: oct(_mode(tmp_path / name)) for name in files} == {name: "0o600" for name in files}
        ledger.close()
        payloads.close()
    finally:
        os.umask(old)


def test_files_left_world_readable_by_an_earlier_run_are_tightened(tmp_path) -> None:
    import os

    for suffix in ("", "-wal", "-shm"):
        path = tmp_path / f"ledger.db{suffix}"
        path.write_bytes(b"")
        os.chmod(path, 0o644)
    SqliteLedger(str(tmp_path / "ledger.db")).close()
    assert {_mode(tmp_path / f"ledger.db{s}") for s in ("", "-wal", "-shm") if (tmp_path / f"ledger.db{s}").exists()} == {0o600}


def test_a_symbolic_link_at_the_ledger_path_is_refused_and_its_target_left_alone(tmp_path) -> None:
    import os

    target = tmp_path / "elsewhere.txt"
    target.write_text("not a ledger")
    os.chmod(target, 0o644)
    os.symlink(target, tmp_path / "ledger.db")
    with pytest.raises(OSError):
        SqliteLedger(str(tmp_path / "ledger.db"))
    assert _mode(target) == 0o644 and target.read_text() == "not a ledger"
