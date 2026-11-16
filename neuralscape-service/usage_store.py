"""Durable, content-free storage for usage events (SQLite and PostgreSQL).

Rows hold exactly the fields of the :class:`UsageEvent` contract (identifiers,
attribution snapshot, token counters, status), serialized as canonical JSON
plus a digest.  No prompt, response or memory content is ever stored.

Semantics:

* ``append`` is idempotent per ``(tenant_id, event_id)``: the same event again
  is a no-op, a different event under the same id raises
  :class:`UsageEventConflict`.
* Corrections, late usage after cancellation and duplicates are plain rows;
  ``reconcile`` runs the stored events through the pure reconciler, which
  alone decides what is counted, so totals never depend on insertion order.
* Unknown stays unknown: absent counters are stored as the contract's explicit
  missing reasons and are never turned into zero.
* The store has its own schema-version table and expand-only migrations.  A
  database from a newer version, or one whose recorded checksum differs, is
  refused.  SQLite requires WAL with ``synchronous`` FULL/EXTRA; PostgreSQL
  requires ``fsync``, ``synchronous_commit`` and ``full_page_writes`` on.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import sqlite3
import threading
import time
from typing import Any, Callable, Iterator

from contracts_usage import UsageEvent
from contracts_usage_reconcile import UsageReconciliation, reconcile_usage_events

_TABLES = """
CREATE TABLE usage_events (
    seq {seq},
    tenant_id TEXT NOT NULL,
    event_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    session_id TEXT,
    intent_id TEXT,
    attempt_id TEXT NOT NULL,
    ledger TEXT NOT NULL,
    operation TEXT NOT NULL,
    predecessor_event_id TEXT,
    status TEXT NOT NULL,
    attempt_outcome TEXT NOT NULL,
    late_after_cancellation INTEGER NOT NULL,
    ledger_attempt INTEGER,
    digest TEXT NOT NULL,
    body TEXT NOT NULL,
    recorded_at DOUBLE PRECISION NOT NULL,
    UNIQUE (tenant_id, event_id)
);
CREATE INDEX usage_events_task ON usage_events(tenant_id, task_id);
CREATE INDEX usage_events_intent ON usage_events(tenant_id, intent_id);
"""

# Usage writes that could not be stored.  The count says "this task has attempts
# the store never saw", so a rollup can report itself incomplete instead of
# looking complete (or missing).
_GAPS = """
CREATE TABLE usage_gaps (
    tenant_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    unrecorded INTEGER NOT NULL,
    updated_at DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (tenant_id, task_id)
)
"""

_MIGRATIONS = {
    "sqlite": (
        (1, "init", [s.strip() for s in _TABLES.format(seq="INTEGER PRIMARY KEY AUTOINCREMENT").split(";") if s.strip()]),
        (2, "gaps", [_GAPS.strip()]),
    ),
    "postgres": (
        (1, "init", [s.strip() for s in _TABLES.format(seq="BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY").split(";") if s.strip()]),
        (2, "gaps", [_GAPS.strip()]),
    ),
}


class UsageStoreError(Exception):
    """The store, its schema or a stored row cannot be trusted."""


class UsageRollupTooLarge(UsageStoreError):
    """A task holds more events than one rollup will read; a partial rollup would undercount."""


#: Most events one task rollup reads (about 2 ms and 30 KiB each); more is refused, never truncated.
MAX_ROLLUP_EVENTS = 2_000


class UsageEventConflict(UsageStoreError):
    """A different event was already stored under the same event id."""


def _canonical(event: UsageEvent) -> tuple[str, str]:
    body = json.dumps(
        event.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return body, hashlib.sha256(body.encode("ascii")).hexdigest()


def _checksum(statements: list[str]) -> str:
    return hashlib.sha256("\n".join(statements).encode("utf-8")).hexdigest()


class _SqlUsageStore:
    dialect = ""
    _insert_time = "?"

    def __init__(self, clock: Callable[[], float]) -> None:
        self._clock = clock
        self._lock = threading.RLock()
        self._conn: Any = None

    # -- dialect hooks -------------------------------------------------
    def _sql(self, sql: str) -> str:
        return sql

    def _txn(self) -> contextlib.AbstractContextManager:
        raise NotImplementedError

    def _bootstrap(self, c: Any) -> None:
        raise NotImplementedError

    # -- schema --------------------------------------------------------
    def _migrate(self) -> None:
        known = _MIGRATIONS[self.dialect]
        with self._lock, self._txn() as c:
            self._bootstrap(c)
            applied = {
                row[0]: row[1]
                for row in c.execute(
                    "SELECT version, checksum FROM usage_schema_version ORDER BY version"
                ).fetchall()
            }
            by_version = {v: (n, stmts) for v, n, stmts in known}
            for version, checksum in applied.items():
                if version not in by_version:
                    raise UsageStoreError(f"usage schema version {version} is newer than this build")
                if checksum != _checksum(by_version[version][1]):
                    raise UsageStoreError(f"usage schema version {version} checksum mismatch")
            for version, name, statements in known:
                if version in applied:
                    continue
                for statement in statements:
                    c.execute(statement)
                c.execute(
                    self._sql(
                        "INSERT INTO usage_schema_version(version, name, checksum, applied_at)"
                        " VALUES (?, ?, ?, ?)"
                    ),
                    (version, name, _checksum(statements), float(self._clock())),
                )

    def schema_version(self) -> int:
        with self._lock, self._txn() as c:
            return c.execute("SELECT COALESCE(MAX(version), 0) FROM usage_schema_version").fetchone()[0]

    # -- writes --------------------------------------------------------
    def append(
        self, event: UsageEvent, *, ledger_attempt: int | None = None, budget_s: float | None = None
    ) -> bool:
        """Store ``event``; return True when new, False for an identical duplicate.

        ``budget_s`` (the caller's remaining deadline) caps the append budget of a bounded store."""
        if not isinstance(event, UsageEvent):
            raise TypeError("append requires a UsageEvent")
        if ledger_attempt is not None and (
            type(ledger_attempt) is not int or ledger_attempt < 0
        ):
            raise ValueError("ledger_attempt must be a non-negative integer")
        # Revalidate so a mutated or unchecked-copy model is never persisted.
        # Through JSON, the same path reads take, so tuple fields (recipients) survive.
        event = UsageEvent.model_validate_json(event.model_dump_json())
        body, digest = _canonical(event)
        return self._insert(event, body, digest, ledger_attempt, budget_s)

    def _run_write(self, work: Callable[[Any], Any], budget_s: float | None = None) -> Any:
        """Run ``work(connection)`` in one transaction (the bounded stores cap it, see there)."""
        with self._lock, self._txn() as c:
            return work(c)

    def _insert(
        self,
        event: UsageEvent,
        body: str,
        digest: str,
        ledger_attempt: int | None,
        budget_s: float | None = None,
    ) -> bool:
        return self._run_write(
            lambda c: self._insert_in(c, event, body, digest, ledger_attempt), budget_s
        )

    def _insert_in(
        self, c: Any, event: UsageEvent, body: str, digest: str, ledger_attempt: int | None
    ) -> bool:
        cursor = c.execute(
            self._sql(
                "INSERT INTO usage_events(tenant_id, event_id, task_id, session_id, intent_id,"
                " attempt_id, ledger, operation, predecessor_event_id, status, attempt_outcome,"
                " late_after_cancellation, ledger_attempt, digest, body, recorded_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
                " ON CONFLICT (tenant_id, event_id) DO NOTHING"
            ),
            (
                event.tenant_id,
                event.event_id,
                event.task_id,
                event.session_id,
                event.intent_id,
                event.attempt_id,
                event.ledger,
                event.operation,
                event.predecessor_event_id,
                event.status,
                event.attempt_outcome,
                int(event.late_after_cancellation),
                ledger_attempt,
                digest,
                body,
                float(self._clock()),
            ),
        )
        if cursor.rowcount == 1:
            return True
        stored = c.execute(
            self._sql("SELECT digest FROM usage_events WHERE tenant_id=? AND event_id=?"),
            (event.tenant_id, event.event_id),
        ).fetchone()
        if stored is None or stored[0] != digest:
            raise UsageEventConflict("a different usage event exists under this event id")
        return False

    # -- reads ---------------------------------------------------------
    def events(
        self, tenant_id: str, task_id: str | None = None, *, limit: int | None = None
    ) -> tuple[UsageEvent, ...]:
        """Stored events for a tenant (optionally one task), in insertion order.

        With ``limit``, more than that many matching events raises ``UsageRollupTooLarge``
        before any is decoded.
        """
        query = "SELECT event_id, digest, body FROM usage_events WHERE tenant_id=?"
        params: list[Any] = [tenant_id]
        if task_id is not None:
            query += " AND task_id=?"
            params.append(task_id)
        with self._lock, self._txn() as c:
            query += " ORDER BY seq"
            if limit is not None:
                query += " LIMIT ?"
                params.append(limit + 1)
            rows = c.execute(self._sql(query), params).fetchall()
        if limit is not None and len(rows) > limit:
            raise UsageRollupTooLarge(f"more than {limit} usage events")
        out = []
        for event_id, digest, body in rows:
            if hashlib.sha256(body.encode("ascii")).hexdigest() != digest:
                raise UsageStoreError(f"stored usage event {event_id} fails its digest")
            event = UsageEvent.model_validate_json(body)
            if event.tenant_id != tenant_id or event.event_id != event_id:
                raise UsageStoreError(f"stored usage event {event_id} does not match its row")
            out.append(event)
        return tuple(out)

    def record_gap(
        self, tenant_id: str, task_id: str, count: int = 1, *, budget_s: float | None = None
    ) -> None:
        """Add ``count`` to the task's unrecorded-write counter (never lowers it)."""
        if type(count) is not int or count < 1:
            raise ValueError("count must be a positive integer")
        self._run_write(
            lambda c: c.execute(
                self._sql(
                    "INSERT INTO usage_gaps(tenant_id, task_id, unrecorded, updated_at)"
                    " VALUES (?, ?, ?, ?)"
                    " ON CONFLICT (tenant_id, task_id) DO UPDATE"
                    " SET unrecorded = usage_gaps.unrecorded + excluded.unrecorded,"
                    " updated_at = excluded.updated_at"
                ),
                (tenant_id, task_id, count, float(self._clock())),
            ),
            budget_s,
        )

    def gaps(self, tenant_id: str, task_id: str) -> int:
        """Usage writes known to be missing for the task (0 when none was lost)."""
        with self._lock, self._txn() as c:
            row = c.execute(
                self._sql("SELECT unrecorded FROM usage_gaps WHERE tenant_id=? AND task_id=?"),
                (tenant_id, task_id),
            ).fetchone()
        return int(row[0]) if row else 0

    def ledger_attempts(self, tenant_id: str, intent_id: str) -> dict[str, int | None]:
        """Map event id -> lifecycle-ledger attempt for events of one intent."""
        with self._lock, self._txn() as c:
            rows = c.execute(
                self._sql(
                    "SELECT event_id, ledger_attempt FROM usage_events"
                    " WHERE tenant_id=? AND intent_id=? ORDER BY seq"
                ),
                (tenant_id, intent_id),
            ).fetchall()
        return {event_id: attempt for event_id, attempt in rows}

    def reconcile(self, tenant_id: str, task_id: str | None = None) -> UsageReconciliation | None:
        """Reconcile stored events; ``None`` when nothing is stored (unreported)."""
        events = self.events(tenant_id, task_id)
        return reconcile_usage_events(events) if events else None

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None


class SqliteUsageStore(_SqlUsageStore):
    dialect = "sqlite"

    def __init__(
        self,
        path: str,
        *,
        synchronous: str = "FULL",
        clock: Callable[[], float] = time.time,
        busy_timeout_ms: int = 30_000,
        append_budget_s: float = 2.0,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__(clock)
        self._busy_timeout_ms = int(busy_timeout_ms)
        self._append_budget_s = float(append_budget_s)
        self._monotonic = monotonic
        if synchronous.upper() not in {"FULL", "EXTRA"}:
            raise ValueError("durable usage storage requires synchronous=FULL or EXTRA")
        owner_only = path not in ("", ":memory:") and not path.startswith("file:")
        if owner_only:
            # The ledger names tenants and tasks: create it owner-only whatever the umask
            # (SQLite gives the -wal/-shm sidecars the main file's mode).
            os.close(os.open(path, os.O_RDWR | os.O_CREAT, 0o600))
            os.chmod(path, 0o600)
        self._conn = sqlite3.connect(
            path, isolation_level=None, timeout=busy_timeout_ms / 1000, check_same_thread=False
        )
        mode = self._conn.execute("PRAGMA journal_mode=WAL").fetchone()[0]
        if mode.lower() != "wal":
            raise UsageStoreError(f"WAL is required (journal_mode={mode})")
        if owner_only:
            for suffix in ("-wal", "-shm"):
                with contextlib.suppress(OSError):
                    os.chmod(path + suffix, 0o600)
        self._conn.execute(f"PRAGMA synchronous={synchronous.upper()}")
        level = self._conn.execute("PRAGMA synchronous").fetchone()[0]
        if level not in (2, 3):
            raise UsageStoreError(f"synchronous FULL/EXTRA is required (level={level})")
        self._conn.execute(f"PRAGMA busy_timeout={int(busy_timeout_ms)}")
        self._migrate()

    def _write_budget(self, budget_s: float | None) -> float:
        """The append budget, capped by the caller's remaining deadline (never negative)."""
        if budget_s is None:
            return self._append_budget_s
        return max(0.0, min(self._append_budget_s, float(budget_s)))

    def _run_write(self, work: Callable[[Any], Any], budget_s: float | None = None) -> Any:
        """A write (an append, a gap count) within the append budget: the wait for the connection
        lock and SQLite's own busy wait both draw on one deadline, so a locked database cannot hold
        the provider-call thread for the connection's full (30 s) busy timeout."""

        deadline = self._monotonic() + self._write_budget(budget_s)
        remaining = deadline - self._monotonic()
        if remaining <= 0 or not self._lock.acquire(timeout=remaining):
            raise UsageStoreError("usage store busy: the append budget was spent waiting for the connection")
        try:
            budget_ms = max(1, int((deadline - self._monotonic()) * 1000))
            self._conn.execute(f"PRAGMA busy_timeout={min(budget_ms, self._busy_timeout_ms)}")
            try:
                with self._txn() as c:
                    return work(c)
            finally:
                self._conn.execute(f"PRAGMA busy_timeout={self._busy_timeout_ms}")
        finally:
            self._lock.release()

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

    def _bootstrap(self, c: Any) -> None:
        c.execute(
            "CREATE TABLE IF NOT EXISTS usage_schema_version("
            "version INTEGER PRIMARY KEY, name TEXT NOT NULL, checksum TEXT NOT NULL,"
            " applied_at REAL NOT NULL)"
        )


class PostgresUsageStore(_SqlUsageStore):
    """PostgreSQL usage store.

    One connection, reopened when it drops (a restart, a failover, a killed backend) and never held
    across an unbounded wait: an append has a short budget (``append_budget_s``) covering the wait for
    the connection lock, the statement and one reconnect-and-retry (the insert is idempotent), so a slow
    or absent database costs the provider-call thread at most that budget.  Reads and the rollup use the
    longer session timeouts.  After a failed reconnect the store fails fast for ``reconnect_cooldown_s``
    instead of paying a connect timeout on every call.
    """

    dialect = "postgres"
    _MIGRATION_LOCK = 7_302_011_017  # application-wide advisory lock key

    def __init__(
        self,
        dsn: str,
        *,
        clock: Callable[[], float] = time.time,
        connect_timeout: int = 10,
        statement_timeout_ms: int = 30_000,
        lock_timeout_ms: int = 10_000,
        append_budget_s: float = 2.0,
        reconnect_cooldown_s: float = 5.0,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__(clock)
        import psycopg

        self._psycopg = psycopg
        self._dsn = dsn
        self._connect_timeout = connect_timeout
        self._statement_timeout_ms = statement_timeout_ms
        self._lock_timeout_ms = lock_timeout_ms
        self._append_budget_s = float(append_budget_s)
        self._reconnect_cooldown_s = float(reconnect_cooldown_s)
        self._monotonic = monotonic
        self._retry_after = 0.0
        self.reconnects = 0
        self._conn = self._open_connection(connect_timeout)
        self._assert_durable_settings(self._conn)
        self._migrate()

    def _open_connection(self, connect_timeout: float) -> Any:
        conn = self._psycopg.connect(self._dsn, autocommit=True, connect_timeout=connect_timeout)
        # A stuck statement or a held lock (the migration advisory lock included) must fail
        # the call, not hang the writer or the rollup route for ever.
        for name, value in (
            ("statement_timeout", self._statement_timeout_ms),
            ("lock_timeout", self._lock_timeout_ms),
        ):
            conn.execute("SELECT set_config(%s, %s, false)", (name, f"{int(value)}ms"))
        return conn

    @staticmethod
    def _assert_durable_settings(conn: Any) -> None:
        for setting in ("fsync", "synchronous_commit", "full_page_writes"):
            value = conn.execute(f"SHOW {setting}").fetchone()[0]
            if value != "on":
                raise UsageStoreError(f"PostgreSQL {setting} must be on (is {value})")

    def _connection_failure(self, error: BaseException) -> bool:
        kinds = tuple(
            kind
            for kind in (
                getattr(self._psycopg, "OperationalError", None),
                getattr(self._psycopg, "InterfaceError", None),
            )
            if isinstance(kind, type)
        )
        return bool(kinds) and isinstance(error, kinds)

    def _drop_connection(self) -> None:
        conn, self._conn = self._conn, None
        if conn is not None:
            with contextlib.suppress(Exception):
                conn.close()

    def _connect_within(self, remaining: float) -> Any:
        """Open and check a connection, giving up once ``remaining`` seconds are spent.

        libpq floors ``connect_timeout`` at two seconds, so the driver's own timeout cannot honour a
        smaller deadline: the attempt runs on a helper thread that is abandoned (and closes whatever
        it opened) when the deadline passes first."""

        outcome: dict[str, Any] = {}
        done = threading.Event()
        abandoned = threading.Event()

        def attempt() -> None:
            try:
                conn = self._open_connection(min(self._connect_timeout, 2))
                try:
                    self._assert_durable_settings(conn)
                except BaseException:
                    with contextlib.suppress(Exception):
                        conn.close()
                    raise
                outcome["conn"] = conn
            except BaseException as error:  # handed to the caller, which re-raises it
                outcome["error"] = error
            finally:
                done.set()
                if abandoned.is_set() and outcome.get("conn") is not None:
                    with contextlib.suppress(Exception):
                        outcome["conn"].close()

        threading.Thread(target=attempt, name="usage-store-reconnect", daemon=True).start()
        if not done.wait(max(0.0, remaining)):
            abandoned.set()
            if done.is_set() and outcome.get("conn") is not None:  # finished between the wait and the flag
                with contextlib.suppress(Exception):
                    outcome["conn"].close()
            raise UsageStoreError("usage store reconnect did not finish within the remaining budget")
        if "error" in outcome:
            raise outcome["error"]
        return outcome["conn"]

    def _ensure_connection(self, deadline: float | None = None) -> None:
        conn = self._conn
        if conn is not None and not getattr(conn, "closed", False) and not getattr(conn, "broken", False):
            return
        now = self._monotonic()
        if now < self._retry_after:
            raise UsageStoreError("usage store unavailable: reconnect is backing off")
        self._drop_connection()
        try:
            if deadline is None:
                conn = self._open_connection(min(self._connect_timeout, 2))
                self._assert_durable_settings(conn)
            else:
                conn = self._connect_within(deadline - now)
        except Exception:
            self._retry_after = now + self._reconnect_cooldown_s
            raise
        self._conn = conn
        self._retry_after = 0.0
        self.reconnects += 1

    def _sql(self, sql: str) -> str:
        return sql.replace("?", "%s")

    @contextlib.contextmanager
    def _txn(self, deadline: float | None = None) -> Iterator[Any]:
        self._ensure_connection(deadline)
        try:
            with self._conn.transaction():
                yield self._conn
        except Exception as error:
            if self._connection_failure(error):
                self._drop_connection()  # the next call reconnects
            raise

    def _write_budget(self, budget_s: float | None) -> float:
        """The append budget, capped by the caller's remaining deadline (never negative)."""
        if budget_s is None:
            return self._append_budget_s
        return max(0.0, min(self._append_budget_s, float(budget_s)))

    def _run_write(self, work: Callable[[Any], Any], budget_s: float | None = None) -> Any:
        """A write (an append, a gap count) within the append budget: the wait for the connection
        lock, the statement and, after a dropped connection, one reconnect and one retry (every
        write here is idempotent or additive-and-recorded, and a failed first try did not commit)."""

        deadline = self._monotonic() + self._write_budget(budget_s)
        last: BaseException | None = None
        for attempt in range(2):  # the second try follows one reconnect
            remaining = deadline - self._monotonic()
            if remaining <= 0:
                break
            if not self._lock.acquire(timeout=remaining):
                raise UsageStoreError("usage store busy: the append budget was spent waiting for the connection")
            try:
                budget_ms = max(100, int((deadline - self._monotonic()) * 1000))
                try:
                    with self._txn(deadline) as c:
                        c.execute(
                            "SELECT set_config('statement_timeout', %s, true),"
                            " set_config('lock_timeout', %s, true)",
                            (f"{budget_ms}ms", f"{budget_ms}ms"),
                        )
                        return work(c)
                except Exception as error:
                    if attempt == 0 and self._connection_failure(error):
                        last = error
                        continue
                    raise
            finally:
                self._lock.release()
        if last is not None:
            raise last
        raise UsageStoreError("usage store append budget spent")

    def _bootstrap(self, c: Any) -> None:
        c.execute("SELECT pg_advisory_xact_lock(%s)", (self._MIGRATION_LOCK,))
        c.execute(
            "CREATE TABLE IF NOT EXISTS usage_schema_version("
            "version INTEGER PRIMARY KEY, name TEXT NOT NULL, checksum TEXT NOT NULL,"
            " applied_at DOUBLE PRECISION NOT NULL)"
        )


def open_usage_store(settings: Any = None) -> _SqlUsageStore | None:
    """Open the configured store, or ``None`` when usage persistence is disabled."""
    if settings is None:
        from config import settings as default_settings

        settings = default_settings
    if not settings.usage_store_enabled:
        return None
    backend = settings.usage_store_backend
    if backend == "sqlite":
        return SqliteUsageStore(settings.usage_store_sqlite_path)
    if backend == "postgres":
        if not settings.usage_store_postgres_dsn:
            raise ValueError("USAGE_STORE_POSTGRES_DSN is required for the postgres backend")
        return PostgresUsageStore(settings.usage_store_postgres_dsn)
    raise ValueError(f"unknown USAGE_STORE_BACKEND {backend!r}")


def install_usage_capture(settings: Any = None) -> _SqlUsageStore | None:
    """Open the configured store and install the durable recorder; ``None`` (nothing installed) when disabled.

    The recorder must exist before the service builds its provider clients so they are wrapped.  The API, every
    worker queue and any other process that makes provider calls call this at start-up.
    """
    store = open_usage_store(settings)
    if store is None:
        return None
    import usage_capture

    usage_capture.set_usage_recorder(usage_capture.DurableUsageRecorder(store))
    return store


def uninstall_usage_capture(store: _SqlUsageStore | None) -> None:
    """Flush pending gaps, remove the recorder and close the store (best effort, never raises)."""
    if store is None:
        return
    import usage_capture

    try:
        recorder = usage_capture.get_usage_recorder()
        if hasattr(recorder, "flush_gaps"):
            recorder.flush_gaps()
    except Exception:
        pass
    usage_capture.set_usage_recorder(None)
    try:
        store.close()
    except Exception:
        pass


_ATTEMPT_CATEGORIES = (
    "input_tokens",
    "output_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
    "reasoning_tokens",
)


def _quantity(quantity: Any) -> dict[str, Any]:
    return {"value": quantity.value, "missing_reason": quantity.missing_reason}


_OTHER_LEDGERS = {
    "consuming_agent": {
        "coverage": "not_collected",
        "reason": "consumer metering is off until the owner consents",
    },
    "evaluation": {
        "coverage": "unreported",
        "note": "kept separate and never added to service totals",
    },
}


def _as_root(event: UsageEvent) -> UsageEvent:
    values = event.model_dump(mode="json")
    values["predecessor_event_id"] = None
    return UsageEvent.model_validate_json(json.dumps(values))


def task_rollup(store: Any, tenant_id: str, task_id: str) -> dict[str, Any] | None:
    """Service-ledger usage of one task, per attempt, from the stored events.

    The attempts and totals are exactly ``reconcile_usage_events`` over the
    stored rows, so a duplicate, a correction or late usage is counted once.
    An attempt with unknown usage (pending, failed, cancelled, unreported) has
    ``total_tokens: null``; an unknown total is never rendered as zero.  Only
    the service ledger is itemised: consumer metering is not collected and the
    evaluation ledger is reported as coverage only.  ``None`` when the task has
    no stored events.
    """

    events = store.events(tenant_id, task_id, limit=MAX_ROLLUP_EVENTS)
    gaps = store.gaps(tenant_id, task_id) if hasattr(store, "gaps") else 0
    if not events and not gaps:
        return None
    # An event whose predecessor was never stored (a failed pending write, or rows
    # from before writes were made safe) would make reconciliation reject the
    # whole task.  Treat it as the root of its stream and say so.
    stored_ids = {event.event_id for event in events}
    orphaned = {
        event.event_id for event in events
        if event.predecessor_event_id is not None and event.predecessor_event_id not in stored_ids
    }
    if orphaned:
        events = tuple(_as_root(event) if event.event_id in orphaned else event for event in events)
    if not events:
        return {
            "task_id": task_id,
            "tenant_id": tenant_id,
            "event_count": 0,
            "attempts": [],
            "recording": {"complete": False, "unrecorded_writes": gaps, "orphaned_events": 0},
            "service": {
                "coverage": "unreported",
                "known_token_subtotal": 0,
                "total_tokens": None,
                "incomplete_attempt_ids": [],
            },
            "other_ledgers": _OTHER_LEDGERS,
        }
    orphaned_attempts = {event.attempt_id for event in events if event.event_id in orphaned}
    result = reconcile_usage_events(events)
    ledgers = {ledger.ledger: ledger for ledger in result.ledgers}
    ledger_attempts = {}
    for event in events:
        if event.intent_id is not None:
            ledger_attempts.update(store.ledger_attempts(tenant_id, event.intent_id))
            break
    attempts = []
    for stream in result.streams:
        if stream.ledger != "service":
            continue
        usage = stream.usage
        attempts.append(
            {
                "attempt_id": stream.attempt_id,
                "operation": stream.operation,
                "ledger_attempt": ledger_attempts.get(stream.head_event_id),
                "status": stream.status,
                "attempt_outcome": stream.attempt_outcome,
                "late_after_cancellation": stream.late_after_cancellation,
                "head_event_id": stream.head_event_id,
                "predecessor_not_stored": stream.attempt_id in orphaned_attempts,
                "provider": stream.attribution.provider,
                "model": stream.attribution.model,
                "recipients": list(stream.attribution.recipients or ()),
                "measurement": usage.measurement if usage is not None else None,
                "tokens": (
                    {name: _quantity(getattr(usage, name)) for name in _ATTEMPT_CATEGORIES}
                    if usage is not None
                    else None
                ),
                "usage_missing_reason": stream.usage_missing_reason,
                "known_token_subtotal": stream.known_token_subtotal,
                "total_tokens": stream.total_tokens,
                "incomplete_categories": list(stream.incomplete_categories),
            }
        )
    service = ledgers["service"]
    return {
        "task_id": task_id,
        "tenant_id": tenant_id,
        "event_count": len(events),
        "attempts": attempts,
        "recording": {
            "complete": not gaps and not orphaned,
            "unrecorded_writes": gaps,
            "orphaned_events": len(orphaned),
        },
        "service": {
            "coverage": service.coverage,
            "known_token_subtotal": service.known_token_subtotal,
            "total_tokens": service.total_tokens,
            "incomplete_attempt_ids": list(service.incomplete_attempt_ids),
        },
        "other_ledgers": {
            "consuming_agent": {
                "coverage": "not_collected",
                "reason": "consumer metering is off until the owner consents",
            },
            "evaluation": {
                "coverage": ledgers["evaluation"].coverage,
                "note": "kept separate and never added to service totals",
            },
        },
    }
