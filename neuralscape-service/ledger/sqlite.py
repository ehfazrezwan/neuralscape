"""Embedded SQLite ledger (single owning host, local filesystem, WAL).

Supported profile: one host, local disk with honest fsync, ``synchronous``
FULL or EXTRA.  Network or shared filesystems and multi-host writers are
unsupported.  ``NORMAL``/``OFF`` are rejected at construction and re-checked
at startup and periodically during accepts, because an acknowledged accept
could be lost on power failure.  Process-crash recovery is tested;
power-loss durability rests on SQLite's documented semantics and these
settings, not on a test.  Lease expiry uses this host's clock inside the
write transaction; a clock step is tolerated through fencing tokens and
idempotent effects, never relied on for safety.
There is no engine-level tenant boundary in a shared file; use a file per
tenant when engine separation is required.
"""

from __future__ import annotations

import contextlib
import os
import sqlite3
import stat
import time
from typing import Any, Callable, Iterator

from ledger._sql import SqlLedger
from ledger.port import SchemaError, plan_migrations

_BOOTSTRAP = (
    "CREATE TABLE IF NOT EXISTS ledger_schema_version("
    "version INTEGER PRIMARY KEY, name TEXT NOT NULL, checksum TEXT NOT NULL, applied_at REAL NOT NULL)"
)


def private_database_file(path: str) -> None:
    """Create a SQLite database file readable only by the service user, or tighten an existing one.

    Run before SQLite opens the file: SQLite creates its journal files (``-wal``, ``-shm``) with the
    database file's mode, and journal files left by an earlier run are tightened here too.  The file is
    opened without following a symbolic link or blocking on a pipe and must be a regular file, so a link
    planted at the path is refused rather than its target changed.
    """

    if path == ":memory:" or path.startswith("file:"):
        return
    for name in (path, path + "-wal", path + "-shm"):
        flags = os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK | (os.O_CREAT if name == path else 0)
        try:
            descriptor = os.open(name, flags, 0o600)
        except FileNotFoundError:
            continue  # a journal created later takes the database file's mode
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise OSError(f"{name} is not a regular file")
            os.fchmod(descriptor, 0o600)
        finally:
            os.close(descriptor)


class SqliteLedger(SqlLedger):
    dialect = "sqlite"
    _integrity_errors = (sqlite3.IntegrityError,)

    def __init__(
        self,
        path: str,
        *,
        synchronous: str = "FULL",
        clock: Callable[[], float] = time.time,
        busy_timeout_ms: int = 30_000,
        durability_check_interval: float = 30.0,
        wal_retry_seconds: float = 30.0,
    ) -> None:
        super().__init__(durability_check_interval)
        if synchronous.upper() not in {"FULL", "EXTRA"}:
            raise ValueError("durable acceptance requires synchronous=FULL or EXTRA")
        private_database_file(path)
        self._conn = sqlite3.connect(
            path, isolation_level=None, timeout=busy_timeout_ms / 1000, check_same_thread=False
        )
        self._conn.create_function("ledger_now", 0, clock)
        mode = self._enable_wal(wal_retry_seconds)
        if mode.lower() != "wal":
            raise SchemaError(f"WAL is required (journal_mode={mode})")
        self._conn.execute(f"PRAGMA synchronous={synchronous.upper()}")
        self._conn.execute(f"PRAGMA busy_timeout={int(busy_timeout_ms)}")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._migrate()
        self.check_durability()

    def _enable_wal(self, retry_seconds: float = 30.0) -> str:
        # Switching a fresh file to WAL fails immediately (no busy wait) when
        # another connection is doing the same, so retry until the deadline.
        deadline = time.monotonic() + retry_seconds
        while True:
            try:
                return self._conn.execute("PRAGMA journal_mode=WAL").fetchone()[0]
            except sqlite3.OperationalError:
                if time.monotonic() > deadline:
                    raise
                time.sleep(0.05)

    def _migrate(self) -> None:
        with self._lock, self._txn() as c:
            c.execute(_BOOTSTRAP)
            pending = plan_migrations(self.dialect, self._applied_migrations(c))
            for migration in pending:
                for statement in _statements(migration.sql):
                    c.execute(statement)
                self._record_migration(c, migration)

    @contextlib.contextmanager
    def _txn(self) -> Iterator[Any]:
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            yield self._conn
        except BaseException:
            self._conn.execute("ROLLBACK")
            raise
        else:
            self._conn.execute("COMMIT")

    @contextlib.contextmanager
    def _read_txn(self) -> Iterator[Any]:
        self._conn.execute("BEGIN")  # deferred: a stable WAL read snapshot
        try:
            yield self._conn
        finally:
            self._conn.execute("ROLLBACK")

    def _op_claim(self, c: Any, owner: str, lease_seconds: float, tenant_id: str | None) -> tuple | None:
        where = "(state='pending' OR (state='leased' AND lease_until < ledger_now()))"
        params: tuple = ()
        if tenant_id is not None:
            where += " AND tenant_id=?"
            params = (tenant_id,)
        row = c.execute(f"SELECT id FROM outbox WHERE {where} ORDER BY id LIMIT 1", params).fetchone()
        if row is None:
            return None
        return c.execute(
            "UPDATE outbox SET state='leased', lease_owner=?, attempt=attempt+1,"
            " lease_until=ledger_now() + ? WHERE id=? RETURNING id, intent_id, tenant_id, attempt, lease_until",
            (owner, lease_seconds, row[0]),
        ).fetchone()

    def _op_extend(self, c: Any, claim: Any, lease_seconds: float) -> float | None:
        row = c.execute(
            "UPDATE outbox SET lease_until=ledger_now() + ? WHERE id=? AND tenant_id=? AND lease_owner=?"
            " AND attempt=? AND state='leased' AND lease_until >= ledger_now() RETURNING lease_until",
            (lease_seconds, claim.outbox_id, claim.tenant_id, claim.owner, claim.attempt),
        ).fetchone()
        return None if row is None else row[0]

    def _is_intent_id_violation(self, exc: BaseException) -> bool:
        return str(exc) == "UNIQUE constraint failed: intents.tenant_id, intents.id"

    def _durability_problems(self) -> list[str]:
        problems = []
        sync = self._conn.execute("PRAGMA synchronous").fetchone()[0]
        if sync not in (2, 3):  # FULL, EXTRA
            problems.append(f"sqlite synchronous={sync} (FULL or EXTRA required)")
        mode = self._conn.execute("PRAGMA journal_mode").fetchone()[0]
        if str(mode).lower() != "wal":
            problems.append(f"sqlite journal_mode={mode} (wal required)")
        return problems

    def backup_to(self, path: str) -> None:
        """Native online backup (consistent while other writers are active)."""

        private_database_file(path)
        dest = sqlite3.connect(path, isolation_level=None)
        try:
            with self._lock:
                self._conn.backup(dest)
        finally:
            dest.close()


def _statements(script: str) -> list[str]:
    """Split a migration into statements without executescript's implicit COMMIT."""

    out, current = [], ""
    for line in script.splitlines():
        if line.strip().startswith("--"):
            continue
        current += line + "\n"
        if sqlite3.complete_statement(current):
            out.append(current.strip())
            current = ""
    if current.strip():
        raise SchemaError("migration ends with an incomplete statement")
    return out
