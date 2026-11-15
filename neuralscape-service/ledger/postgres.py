"""PostgreSQL ledger (hosted tenant-cell profile).

Intended unit: one database (or schema) per tenant cell, so the isolation
guarantee of this profile is cell separation.  Row-level security is optional
hardening for a shared-database deployment and is not applied here.

Roles
-----
* The *owner* role runs migrations, owns the tables and the SECURITY DEFINER
  functions, and performs restores.  It should not be a superuser either.
* The *runtime* role (``grant_runtime_role``) must not be superuser or
  BYPASSRLS (refused at startup).  It has no UPDATE/DELETE on ``outbox``;
  claim, check, extend, complete, release, mark-queued and close go through
  tenant-scoped definer functions with a fixed ``search_path`` (schema, then
  ``pg_temp`` last).

Durability: ``fsync``, ``synchronous_commit`` (on or remote_apply) and
``full_page_writes`` must be on, checked at startup and periodically; the
ledger refuses to accept otherwise.  Process-crash recovery is tested;
power-loss durability rests on PostgreSQL's documented behavior and these
settings.  Lease expiry uses the server clock (``clock_timestamp()``) inside
the claiming transaction; a clock step is tolerated through fencing tokens
and idempotent effects.  Backups are logical and consistent-snapshot only
(snapshot/restore or ``pg_dump``); no RPO/RTO is claimed.
"""

from __future__ import annotations

import contextlib
import functools
import hashlib
import inspect
from typing import Any, Callable, Iterator, Mapping

import psycopg
from psycopg import sql as pgsql

from ledger._sql import SqlLedger
from ledger.port import (
    TIME_COLUMNS,
    PrivilegeError,
    SchemaError,
    load_migrations,
    plan_migrations,
)

_BOOTSTRAP = (
    "CREATE TABLE IF NOT EXISTS ledger_schema_version("
    "version INTEGER PRIMARY KEY, name TEXT NOT NULL, checksum TEXT NOT NULL, applied_at TIMESTAMPTZ NOT NULL)"
)
_MIGRATION_LOCK = 7_302_011_002  # arbitrary application-wide advisory lock key

FUNCTIONS = (
    "ledger_claimable_tenants()",
    "ledger_claim(text,text,double precision)",
    "ledger_check_claim(text,bigint,text,integer)",
    "ledger_extend(text,bigint,text,integer,double precision)",
    "ledger_complete(text,bigint,text,integer)",
    "ledger_release(text,bigint,text,integer)",
    "ledger_mark_queued(text,bigint)",
    "ledger_close_outbox(text,text,text)",
    "ledger_take_unqueued(text,bigint,integer)",
    "ledger_unmark_queued(text,bigint,integer)",
)
_DML = {
    "intents": "SELECT, INSERT, UPDATE",
    "stage_receipts": "SELECT, INSERT, UPDATE",
    "effects": "SELECT, INSERT",
    "object_refs": "SELECT, INSERT, DELETE",
    "tombstones": "SELECT, INSERT",
    "outbox": "SELECT, INSERT",  # all other outbox changes use definer functions
    "ledger_schema_version": "SELECT",
    "retention_holds": "SELECT, INSERT, UPDATE",
    "erasure_audit": "SELECT, INSERT",  # append-only: no UPDATE/DELETE, plus a trigger
}


def durability_violations(settings: Mapping[str, str]) -> list[str]:
    """Pure check of the PostgreSQL settings an acknowledged accept depends on."""

    problems = []
    if settings.get("fsync") != "on":
        problems.append(f"postgres fsync={settings.get('fsync')} (on required)")
    if settings.get("synchronous_commit") not in {"on", "remote_apply"}:
        problems.append(
            f"postgres synchronous_commit={settings.get('synchronous_commit')} (on required)"
        )
    if settings.get("full_page_writes") != "on":
        problems.append(f"postgres full_page_writes={settings.get('full_page_writes')} (on required)")
    return problems


def grant_runtime_role(owner_conn: Any, role: str) -> None:
    """Grant the least privileges the runtime needs; run as the owner role.

    ``role`` must already exist (created by the deployment, never here).
    """

    ident = pgsql.Identifier(role)
    schema = owner_conn.execute("SELECT current_schema()").fetchone()[0]
    owner_conn.execute(
        pgsql.SQL("GRANT USAGE ON SCHEMA {} TO {}").format(pgsql.Identifier(schema), ident)
    )
    for table, privileges in _DML.items():
        owner_conn.execute(
            pgsql.SQL("GRANT " + privileges + " ON {} TO {}").format(
                pgsql.Identifier(table), ident
            )
        )
    owner_conn.execute(
        pgsql.SQL("GRANT USAGE ON SEQUENCE {} TO {}").format(pgsql.Identifier("outbox_id_seq"), ident)
    )
    for signature in FUNCTIONS:
        owner_conn.execute(
            pgsql.SQL("GRANT EXECUTE ON FUNCTION " + signature + " TO {}").format(ident)
        )


def grant_retention_role(owner_conn: Any, role: str) -> None:
    """Let a dedicated maintenance role delete aged tombstones; run as the owner role.

    The ordinary runtime role can only insert tombstones (they are anti-resurrection fences).  Pruning
    one is an explicit, separately granted capability, used only when tombstone retention is
    configured and a backup expiry is known.
    """

    schema = owner_conn.execute("SELECT current_schema()").fetchone()[0]
    ident = pgsql.Identifier(role)
    owner_conn.execute(pgsql.SQL("GRANT USAGE ON SCHEMA {} TO {}").format(pgsql.Identifier(schema), ident))
    for table, privileges in (
        ("tombstones", "SELECT, DELETE"),
        ("retention_holds", "SELECT"),
        ("ledger_schema_version", "SELECT"),
    ):
        owner_conn.execute(
            pgsql.SQL("GRANT " + privileges + " ON {} TO {}").format(pgsql.Identifier(table), ident)
        )


def ungranted_schemas(conn) -> list[str]:
    """Schemas on the connection's search path that exist but the current role may not use."""

    path = conn.execute("SHOW search_path").fetchone()[0]
    names = [part.strip().strip('"') for part in path.split(",") if part.strip() and "$user" not in part]
    rows = conn.execute(
        "SELECT nspname FROM pg_namespace WHERE nspname = ANY(%s) AND NOT has_schema_privilege(nspname, 'USAGE')",
        (names,),
    ).fetchall()
    return [r[0] for r in rows]


def owner_privilege_problem(conn) -> str | None:
    """Why the connected role holds the schema owner's privileges, or ``None``.

    A least-privilege runtime connection must not be the owner of its schema or of any table in it, nor a
    member of such an owner (inherited or by ``SET ROLE``): either can create, drop and delete anything,
    which is exactly what the runtime role exists to prevent.
    """

    path = conn.execute("SHOW search_path").fetchone()[0]
    names = [part.strip().strip('"') for part in path.split(",") if part.strip() and "$user" not in part]
    row = conn.execute(
        """
        WITH owners AS (
            SELECT n.nspowner AS owner FROM pg_namespace n WHERE n.nspname = ANY(%s)
            UNION
            SELECT c.relowner FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE n.nspname = ANY(%s)
        )
        SELECT r.rolname, r.oid = (SELECT oid FROM pg_roles WHERE rolname = current_user)
        FROM owners o JOIN pg_roles r ON r.oid = o.owner
        WHERE pg_has_role(current_user, o.owner, 'MEMBER')
        LIMIT 1
        """,
        (names, names),
    ).fetchone()
    if row is None:
        return None
    if row[1]:
        return "the runtime role owns the schema or its tables; connect as a separate least-privilege role"
    return "the runtime role is a member of the schema owner and holds its privileges; revoke that membership"


class PostgresLedger(SqlLedger):
    dialect = "postgres"
    NOW = "clock_timestamp()"
    _integrity_errors = (psycopg.errors.UniqueViolation,)

    def __init__(
        self,
        dsn: str,
        *,
        migrate: bool = True,
        connect_timeout: int = 10,
        durability_check_interval: float = 30.0,
        inspect_only: bool = False,
    ) -> None:
        """``migrate=True`` is the owner connection (migrations, restore).

        ``migrate=False`` is the runtime connection: the role must not be a
        superuser or BYPASSRLS, must not own the schema or be a member of its
        owner, the schema must already be at this code's version, and a role
        granted ledger functions must hold all of them.

        ``inspect_only=True`` (with ``migrate=False``) is the owner's read-only
        inspection for ``ledger.admin check``: no migration and no runtime-role
        check, but the schema version is still verified.
        """

        super().__init__(durability_check_interval)
        self._privileged = migrate
        self._dsn, self._connect_timeout = dsn, connect_timeout
        self._conn = psycopg.connect(dsn, autocommit=True, connect_timeout=connect_timeout)
        try:
            if migrate:
                self._migrate()
            else:
                if not inspect_only:
                    self._verify_runtime_role()
                self._verify_schema_version()
                if not inspect_only:
                    self._verify_function_grants()
            self.check_durability()
        except BaseException:
            self._conn.close()
            raise

    def _migrate(self) -> None:
        with self._lock, self._txn() as c:
            c.execute("SELECT pg_advisory_xact_lock(%s)", (_MIGRATION_LOCK,))
            c.execute(_BOOTSTRAP)
            for migration in plan_migrations(self.dialect, self._applied_migrations(c)):
                c.execute(migration.sql)
                self._record_migration(c, migration)

    def _verify_runtime_role(self) -> None:
        row = self._conn.execute(
            "SELECT rolsuper, rolbypassrls FROM pg_roles WHERE rolname = current_user"
        ).fetchone()
        if row is None or row[0] or row[1]:
            raise PrivilegeError("runtime role must not be superuser or BYPASSRLS")
        problem = owner_privilege_problem(self._conn)
        if problem is not None:
            raise PrivilegeError(problem)

    def _verify_schema_version(self) -> None:
        expected = len(load_migrations(self.dialect))
        try:
            found = self.schema_version()
        except psycopg.errors.UndefinedTable as exc:
            if ungranted_schemas(self._conn):
                raise SchemaError(
                    "the runtime role was not granted the ledger schema; run the owner grant step"
                ) from exc
            raise SchemaError("ledger schema is not installed; run migrations as the owner role") from exc
        except psycopg.errors.InsufficientPrivilege as exc:
            raise SchemaError("the runtime role was not granted the ledger schema; run the owner grant step") from exc
        if found != expected:
            raise SchemaError(f"ledger schema version {found} does not match code version {expected}")

    def _verify_function_grants(self) -> None:
        # A release that adds ledger functions needs the owner's grant step re-run; without it the
        # runtime would start and then fail on the first call that needs the new function.  A role
        # holding none of them is not a runtime role (the tombstone maintenance role) and is left alone.
        missing = [
            signature for signature in FUNCTIONS
            if not self._conn.execute(
                "SELECT has_function_privilege(to_regprocedure(%s), 'EXECUTE')", (signature,)
            ).fetchone()[0]
        ]
        if missing and len(missing) < len(FUNCTIONS):
            raise SchemaError(f"the runtime role cannot execute {missing[0]}; run the owner grant step")

    def _is_intent_id_violation(self, exc: BaseException) -> bool:
        return getattr(getattr(exc, "diag", None), "constraint_name", None) == "intents_pkey"

    def _durability_problems(self) -> list[str]:
        settings = {
            name: self._conn.execute("SELECT current_setting(%s)", (name,)).fetchone()[0]
            for name in ("fsync", "synchronous_commit", "full_page_writes")
        }
        return durability_violations(settings)

    def _sql(self, sql: str) -> str:
        return sql.replace("?", "%s")

    @contextlib.contextmanager
    def _txn(self) -> Iterator[Any]:
        with self._conn.transaction():
            yield self._conn

    @contextlib.contextmanager
    def _read_txn(self) -> Iterator[Any]:
        with self._conn.transaction():
            self._conn.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
            yield self._conn

    def _lock_suffix(self) -> str:
        return " FOR UPDATE"

    def _lock_record(self, c: Any, tenant_id: str, record_id: str) -> None:
        # Holds, tombstones and references of one record decide on each other's rows without a
        # common row to lock, so serialize them with a transaction-scoped advisory lock.
        digest = hashlib.sha256(f"{tenant_id}\x00{record_id}".encode()).digest()
        c.execute("SELECT pg_advisory_xact_lock(%s)", (int.from_bytes(digest[:8], "big", signed=True),))

    def _epoch(self, column: str) -> str:
        return f"EXTRACT(EPOCH FROM {column})::float8"

    def _now_epoch(self) -> str:
        return f"EXTRACT(EPOCH FROM {self.NOW})::float8"

    def _bool(self, value: bool) -> Any:
        return bool(value)

    def _ins_placeholder(self, column: str) -> str:
        if column in TIME_COLUMNS:
            return "to_timestamp(?)"
        if column == "queued":
            return "(?)::int::boolean"
        return "?"

    def _snap_expr(self, column: str) -> str:
        if column == "queued":
            return "queued::int"
        return super()._snap_expr(column)

    def _after_restore(self, c: Any) -> None:
        max_outbox = c.execute("SELECT COALESCE(MAX(id), 0) FROM outbox").fetchone()[0]
        if max_outbox > 0:
            c.execute("SELECT setval(pg_get_serial_sequence('outbox', 'id'), %s, true)", (max_outbox,))

    def restore(self, snapshot: dict[str, Any], *, expected_digest: str | None = None) -> None:
        if not self._privileged:
            raise PrivilegeError("restore requires the owner connection")
        super().restore(snapshot, expected_digest=expected_digest)

    # ---- outbox operations via tenant-scoped SECURITY DEFINER functions ---
    def _op_claim(self, c: Any, owner: str, lease_seconds: float, tenant_id: str | None) -> tuple | None:
        tenants = (
            [tenant_id]
            if tenant_id is not None
            else [r[0] for r in c.execute("SELECT ledger_claimable_tenants()").fetchall()]
        )
        for tenant in tenants:
            row = c.execute(
                "SELECT o_id, o_intent_id, o_tenant_id, o_attempt, o_lease_epoch FROM ledger_claim(%s, %s, %s)",
                (tenant, owner, lease_seconds),
            ).fetchone()
            if row is not None:
                return row
        return None

    def _op_check(self, c: Any, claim: Any) -> bool:
        return bool(c.execute(
            "SELECT ledger_check_claim(%s, %s, %s, %s)",
            (claim.tenant_id, claim.outbox_id, claim.owner, claim.attempt),
        ).fetchone()[0])

    def _op_extend(self, c: Any, claim: Any, lease_seconds: float) -> float | None:
        return c.execute(
            "SELECT ledger_extend(%s, %s, %s, %s, %s)",
            (claim.tenant_id, claim.outbox_id, claim.owner, claim.attempt, lease_seconds),
        ).fetchone()[0]

    def _op_complete(self, c: Any, claim: Any) -> bool:
        return bool(c.execute(
            "SELECT ledger_complete(%s, %s, %s, %s)",
            (claim.tenant_id, claim.outbox_id, claim.owner, claim.attempt),
        ).fetchone()[0])

    def _op_release(self, c: Any, claim: Any) -> bool:
        return bool(c.execute(
            "SELECT ledger_release(%s, %s, %s, %s)",
            (claim.tenant_id, claim.outbox_id, claim.owner, claim.attempt),
        ).fetchone()[0])

    def _op_mark_queued(self, c: Any, tenant_id: str, outbox_id: int) -> None:
        c.execute("SELECT ledger_mark_queued(%s, %s)", (tenant_id, outbox_id))

    def _op_take_unqueued(self, c: Any, tenant_id: str, outbox_id: int, attempt: int) -> bool:
        return bool(c.execute(
            "SELECT ledger_take_unqueued(%s, %s, %s)", (tenant_id, outbox_id, attempt)
        ).fetchone()[0])

    def _op_unmark_queued(self, c: Any, tenant_id: str, outbox_id: int, attempt: int) -> None:
        c.execute("SELECT ledger_unmark_queued(%s, %s, %s)", (tenant_id, outbox_id, attempt))

    def _op_close_outbox(self, c: Any, tenant_id: str, intent_id: str, state: str) -> None:
        c.execute("SELECT ledger_close_outbox(%s, %s, %s)", (tenant_id, intent_id, state))

    # ---- installation checks ---------------------------------------------
    def verify_installation(self, runtime_role: str | None = None) -> list[str]:
        """Return problems with schema version, definer functions and role grants.

        Run after a restore or provisioning; an empty list means the checks pass.
        """

        problems: list[str] = []
        with self._lock, self._read_txn() as c:
            expected = len(load_migrations(self.dialect))
            found = self.schema_version_in(c)
            if found != expected:
                problems.append(f"schema version {found} != code version {expected}")
            schema = c.execute("SELECT current_schema()").fetchone()[0]
            want_path = f"search_path={schema}, pg_temp"
            for signature in FUNCTIONS:
                row = c.execute(
                    "SELECT p.prosecdef, p.proconfig, p.oid FROM pg_proc p WHERE p.oid = to_regprocedure(%s)",
                    (signature,),
                ).fetchone()
                if row is None:
                    problems.append(f"missing function {signature}")
                    continue
                if not row[0]:
                    problems.append(f"{signature} is not SECURITY DEFINER")
                if want_path not in (row[1] or []):
                    problems.append(f"{signature} lacks fixed search_path ({want_path})")
                if c.execute("SELECT has_function_privilege('public', %s, 'EXECUTE')", (row[2],)).fetchone()[0]:
                    problems.append(f"{signature} is executable by PUBLIC")
                if runtime_role and not c.execute(
                    "SELECT has_function_privilege(%s, %s, 'EXECUTE')", (runtime_role, row[2])
                ).fetchone()[0]:
                    problems.append(f"role {runtime_role} cannot execute {signature}")
            if runtime_role:
                role = c.execute(
                    "SELECT rolsuper, rolbypassrls FROM pg_roles WHERE rolname=%s", (runtime_role,)
                ).fetchone()
                if role is None:
                    problems.append(f"role {runtime_role} does not exist")
                else:
                    if role[0] or role[1]:
                        problems.append(f"role {runtime_role} is superuser or BYPASSRLS")
                    if c.execute(
                        "SELECT pg_has_role(%s, n.nspowner, 'MEMBER') FROM pg_namespace n WHERE n.nspname = %s",
                        (runtime_role, schema),
                    ).fetchone()[0]:
                        problems.append(f"role {runtime_role} owns the schema or is a member of its owner")
                    for privilege in ("UPDATE", "DELETE"):
                        if c.execute(
                            "SELECT has_table_privilege(%s, 'outbox', %s)", (runtime_role, privilege)
                        ).fetchone()[0]:
                            problems.append(f"role {runtime_role} holds {privilege} on outbox")
        return problems


# ---- surviving a database restart ---------------------------------------------------------------

def connection_lost(conn: Any) -> bool:
    return conn is None or conn.closed or conn.broken


def retry_on_lost_connection(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Reconnect once and retry when the database dropped the connection (a restart, a failover).

    Only a lost connection is retried, never an ordinary error.  The ledger's operations are idempotent
    (accepts replay by key, receipts replay by body, status writes are compare-and-set), so repeating one
    whose commit outcome was lost with the connection gives the same result.
    """

    @functools.wraps(fn)
    def call(self: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            return fn(self, *args, **kwargs)
        except psycopg.OperationalError:
            if not connection_lost(getattr(self, "_conn", None)):
                raise
            self._reconnect()
            return fn(self, *args, **kwargs)

    return call


def _reconnect(self: Any) -> None:
    with self._lock:
        try:
            self._conn.close()
        except Exception:  # noqa: BLE001 - already broken
            pass
        self._conn = psycopg.connect(self._dsn, autocommit=True, connect_timeout=self._connect_timeout)


PostgresLedger._reconnect = _reconnect  # type: ignore[attr-defined]
for _name in sorted(set(dir(PostgresLedger))):
    if _name.startswith("_") or _name in {"close", "dialect", "NOW"}:
        continue
    _member = getattr(PostgresLedger, _name)
    if inspect.isfunction(_member):
        setattr(PostgresLedger, _name, retry_on_lost_connection(_member))
del _name, _member
