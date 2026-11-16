"""``VectorStorePort`` over PostgreSQL + pgvector (a different store than Qdrant).

Design choices (see the engine-alternatives decision):

* **Separate schema, separate failure domain.**  Everything lives in its own
  schema (default ``ns_engine``) with its own ``schema_version`` table; the
  ledger's database or schema is never touched.  Point it at a separate
  database for real isolation.
* **Explicit filters, no RLS.**  Every statement carries the tenant in its
  ``WHERE``; the authorized-record-id set and labels are applied in SQL
  *before* ``ORDER BY ... LIMIT``.
* **Exact delete with a tombstone.**  A removal keeps ``(tenant, record_id,
  revision, removed)`` and wipes text, labels and the vector, so an older
  revision cannot return.  A removal of a record that was never applied (an
  erasure processed before the write it erases) still writes that content-free
  tombstone before reporting ``missing_resource``, so a delayed older apply is
  refused instead of bringing the erased content back; a repeated removal only
  ever raises the fence.
* **Content-free typed errors.**  Port methods raise ``PortFailure`` only: a
  non-finite embedding, a NUL character or an embedder failure is a typed code,
  and a database error is ``unavailable_dependency``.  No driver or embedder
  message (which can carry hosts, keys or content) is ever propagated.
* **Atomic fencing.**  Apply and remove take a per-(schema, tenant, record)
  advisory transaction lock first (a row lock cannot protect a row that does
  not exist yet), then read, decide and write in one transaction.
* **Every store call is bounded.**  Each port operation sets the transaction's
  ``statement_timeout`` and ``lock_timeout`` from the operation's ``deadline_ms``
  (a held record lock or a slow statement ends as ``unavailable_dependency``
  instead of blocking the caller), and the default connection factory sets a
  connect timeout (libpq's minimum is two seconds).  Dropping a generation waits
  at most ``DESTROY_LOCK_TIMEOUT_MS`` for its locks, so a release refused for a
  busy schema is reported instead of stalling the router.
* **No embedding provider is chosen here.**  ``embed`` is injected and must
  return exactly ``dimensions`` floats.
* **Bounded results.**  A search returns at most ``MAX_SEARCH_LIMIT`` hits.
* **No ANN index is created implicitly.**  Search is an exact scan until
  ``create_ann_index`` is called deliberately; approximate indexes apply
  filters after the index scan and can return fewer than ``limit`` hits, which
  has not been measured for this deployment.

Manifests are honest: supported and configured, never qualified, healthy only
after a live probe (connection, ``vector`` extension and schema version).
"""

from __future__ import annotations

import contextlib
import math
import re
from collections.abc import Iterator
from typing import Any, Callable, Optional

from contracts_engines import CapabilityHealth, CapabilityManifest, EngineAdapterKind
from contracts_errors import InternalErrorCode
from engine_adapters_current import _check_tenant, _manifest, _receipt
from engine_ports import (
    EngineRecord,
    OperationContext,
    PortFailure,
    ProjectionReceipt,
    SearchHit,
    TenantFilter,
)

ADAPTER_VERSION = "pgvector-v1"
ADMIN_CONNECT_TIMEOUT_S = 10  # schema setup, destroy and index creation
MANIFEST_DEADLINE_MS = 5_000  # the health probe
DESTROY_LOCK_TIMEOUT_MS = 30_000
MAX_SEARCH_LIMIT = 1_000
#: PostgreSQL's statement, lock and connect timeouts are 32-bit; a longer deadline is clamped to the largest.
_MAX_TIMEOUT_MS = 2**31 - 1  # hits one search may return, whatever limit the caller asks for
SCHEMA_VERSION = 1
_IDENTIFIER = re.compile(r"^[a-z_][a-z0-9_]{0,62}$")

# Version -> statements; ``{schema}`` and ``{dims}`` are validated before use.
_MIGRATIONS: dict[int, tuple[str, ...]] = {
    1: (
        "CREATE TABLE {schema}.records ("
        " tenant text NOT NULL, record_id text NOT NULL, revision bigint NOT NULL,"
        " removed boolean NOT NULL DEFAULT false, body text NOT NULL DEFAULT '',"
        " labels text[] NOT NULL DEFAULT '{{}}', embedding vector({dims}),"
        " PRIMARY KEY (tenant, record_id))",
        "CREATE INDEX records_live_idx ON {schema}.records (tenant) WHERE NOT removed",
    ),
}


class SchemaVersionError(RuntimeError):
    """The database schema is newer than, or incompatible with, this adapter."""


def _identifier(value: str, what: str) -> str:
    if type(value) is not str or not _IDENTIFIER.match(value):
        raise ValueError(f"{what} must match {_IDENTIFIER.pattern}")
    return value


def _vector_literal(values: list[float]) -> str:
    return "[" + ",".join(repr(float(v)) for v in values) + "]"


def _no_nul(*values: str) -> None:
    # PostgreSQL text cannot hold NUL; refuse it before the driver does.
    if any("\x00" in value for value in values):
        raise PortFailure(InternalErrorCode.INVALID_SCHEMA, "text contains a NUL character")


class PgVectorAdapter:
    """pgvector-backed ``VectorStorePort``.

    ``connect`` returns a new psycopg connection per call (default:
    ``psycopg.connect(dsn, connect_timeout=...)``); inject a pool-backed factory for
    production use.  An injected factory owns its own connect timeout; the
    statement and lock timeouts are set per operation either way.
    """

    def __init__(
        self,
        dsn: Optional[str] = None,
        *,
        embed: Callable[[str], list[float]],
        dimensions: int,
        schema: str = "ns_engine",
        connect: Optional[Callable[[], Any]] = None,
        implementation_id: str = "pgvector",
    ) -> None:
        if type(dimensions) is not int or not 1 <= dimensions <= 16000:
            raise ValueError("dimensions must be an integer between 1 and 16000")
        if connect is None:
            if not dsn:
                raise ValueError("a dsn or a connect factory is required")
            import psycopg

            def connect(timeout_s: float = ADMIN_CONNECT_TIMEOUT_S) -> Any:  # noqa: F811 - default factory
                seconds = min(max(2, math.ceil(timeout_s)), _MAX_TIMEOUT_MS)
                return psycopg.connect(dsn, connect_timeout=seconds)

            self._default_connect: Optional[Callable[..., Any]] = connect
        else:
            self._default_connect = None
        self._connect = connect
        self._embed = embed
        self._dimensions = dimensions
        self._schema = _identifier(schema, "schema")
        self._implementation_id = implementation_id

    # -- schema -----------------------------------------------------------------

    def ensure_schema(self, *, create_extension: bool = False) -> int:
        """Create or migrate the adapter schema; return its version.

        ``create_extension`` runs ``CREATE EXTENSION IF NOT EXISTS vector`` and
        normally needs elevated rights; by default the extension must already
        exist.  A newer schema version is refused, never downgraded.
        """

        with self._connect() as conn:
            if create_extension:
                conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            if conn.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'").fetchone() is None:
                raise SchemaVersionError("the pgvector extension is not installed in this database")
            # Serialize concurrent first-time setup before any DDL.
            conn.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))", (self._schema + "\x1fddl",))
            conn.execute(f"CREATE SCHEMA IF NOT EXISTS {self._schema}")
            conn.execute(
                f"CREATE TABLE IF NOT EXISTS {self._schema}.schema_version "
                "(version integer NOT NULL, dimensions integer NOT NULL, applied_at timestamptz NOT NULL DEFAULT now())"
            )
            row = conn.execute(f"SELECT max(version), max(dimensions) FROM {self._schema}.schema_version").fetchone()
            current = row[0] or 0
            if current > SCHEMA_VERSION:
                raise SchemaVersionError(f"database schema version {current} is newer than {SCHEMA_VERSION}")
            if current and row[1] != self._dimensions:
                raise SchemaVersionError(
                    f"database was created for {row[1]} dimensions, adapter configured for {self._dimensions}"
                )
            for version in range(current + 1, SCHEMA_VERSION + 1):
                for statement in _MIGRATIONS[version]:
                    conn.execute(statement.format(schema=self._schema, dims=self._dimensions))
                conn.execute(
                    f"INSERT INTO {self._schema}.schema_version (version, dimensions) VALUES (%s, %s)",
                    (version, self._dimensions),
                )
            return SCHEMA_VERSION

    def destroy(self) -> None:
        """Drop this generation's schema and every record in it (used when a rollback target is released).

        Waits at most ``DESTROY_LOCK_TIMEOUT_MS`` for the schema's locks; a busy schema raises
        instead of blocking the caller (the router keeps it as the rollback target).
        """

        with self._connect() as conn:
            conn.execute("SELECT set_config('lock_timeout', %s, true)", (f"{DESTROY_LOCK_TIMEOUT_MS}ms",))
            conn.execute(f"DROP SCHEMA IF EXISTS {self._schema} CASCADE")

    def create_ann_index(self, kind: str = "hnsw") -> None:
        """Deliberately create an approximate index; not part of default setup.

        Approximate indexes filter after the index scan, so selective tenant or
        authorization filters can return fewer than ``limit`` hits.
        """

        if kind != "hnsw":
            raise ValueError("only hnsw is supported")
        with self._connect() as conn:
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS records_embedding_hnsw ON {self._schema}.records "
                "USING hnsw (embedding vector_cosine_ops)"
            )

    # -- capability ---------------------------------------------------------------

    def manifest(self) -> CapabilityManifest:
        health = CapabilityHealth.UNHEALTHY
        try:
            with self._open(MANIFEST_DEADLINE_MS / 1000) as conn:
                self._bound(conn, MANIFEST_DEADLINE_MS)
                row = conn.execute(
                    f"SELECT max(version) FROM {self._schema}.schema_version"
                ).fetchone()
                ok = conn.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'").fetchone()
            if ok is not None and row is not None and row[0] == SCHEMA_VERSION:
                health = CapabilityHealth.HEALTHY
        except Exception:
            pass
        return _manifest(self._implementation_id, EngineAdapterKind.VECTOR_STORE, health=health)

    # -- helpers ------------------------------------------------------------------

    def _receipt(self, record_id: str, revision: int, operation: str) -> ProjectionReceipt:
        return _receipt(self._implementation_id, record_id, revision, operation)

    def _open(self, timeout_s: float) -> Any:
        """A connection; the default factory is given the connect timeout, an injected one owns its own."""

        if self._connect is self._default_connect:
            return self._connect(timeout_s)
        return self._connect()

    @staticmethod
    def _bound(conn: Any, deadline_ms: int) -> None:
        """Bound every statement and lock wait of this transaction by the operation's deadline (a deadline
        beyond the server's 32-bit range is clamped to it, so a very long deadline still works)."""

        deadline_ms = min(deadline_ms, _MAX_TIMEOUT_MS)
        conn.execute(
            "SELECT set_config('statement_timeout', %s, true), set_config('lock_timeout', %s, true)",
            (f"{deadline_ms}ms", f"{deadline_ms}ms"),
        )

    @contextlib.contextmanager
    def _db(self, ctx: OperationContext) -> Iterator[Any]:
        """A connection bounded by ``ctx.deadline_ms`` whose driver errors (a timeout included)
        become a content-free ``unavailable_dependency``."""

        import psycopg

        try:
            with self._open(ctx.deadline_ms / 1000) as conn:
                self._bound(conn, ctx.deadline_ms)
                yield conn
        except psycopg.Error:
            raise PortFailure(InternalErrorCode.UNAVAILABLE_DEPENDENCY, "vector store unavailable") from None

    def _lock(self, conn: Any, tenant_id: str, record_id: str) -> None:
        conn.execute(
            "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
            (f"{self._schema}\x1f{tenant_id}\x1f{record_id}",),
        )

    def _vector(self, text: str) -> str:
        try:
            values = list(self._embed(text))
        except Exception:  # noqa: BLE001 - the embedder's message may carry keys or content
            raise PortFailure(InternalErrorCode.UNAVAILABLE_DEPENDENCY, "embedding provider unavailable") from None
        if len(values) != self._dimensions:
            raise PortFailure(InternalErrorCode.INVALID_SCHEMA, "embedding dimension mismatch")
        try:
            finite = all(math.isfinite(float(v)) for v in values)
        except (TypeError, ValueError):
            finite = False
        if not finite:
            raise PortFailure(InternalErrorCode.INVALID_SCHEMA, "embedding is not a finite vector")
        return _vector_literal(values)

    # -- port ---------------------------------------------------------------------

    def apply(self, ctx: OperationContext, record: EngineRecord) -> ProjectionReceipt:
        _check_tenant(ctx, record.tenant_id)
        _no_nul(record.record_id, record.text, *record.labels)
        vector = self._vector(record.text)
        with self._db(ctx) as conn:
            self._lock(conn, record.tenant_id, record.record_id)
            row = conn.execute(
                f"SELECT revision, removed, body, labels FROM {self._schema}.records "
                "WHERE tenant = %s AND record_id = %s",
                (record.tenant_id, record.record_id),
            ).fetchone()
            if row is not None:
                stored_revision, removed, body, labels = row
                if record.revision < stored_revision:
                    raise PortFailure(InternalErrorCode.STALE_REVISION)
                if record.revision == stored_revision:
                    if removed or body != record.text:
                        raise PortFailure(InternalErrorCode.IDEMPOTENCY_CONFLICT)
                    if tuple(labels) != tuple(record.labels):
                        # Same revision and text, other labels: a relabel (the embedding stays as it is).
                        conn.execute(
                            f"UPDATE {self._schema}.records SET labels = %s "
                            "WHERE tenant = %s AND record_id = %s AND revision = %s AND NOT removed",
                            (list(record.labels), record.tenant_id, record.record_id, record.revision),
                        )
                    return self._receipt(record.record_id, record.revision, "applied")
            conn.execute(
                f"INSERT INTO {self._schema}.records (tenant, record_id, revision, removed, body, labels, embedding) "
                "VALUES (%s, %s, %s, false, %s, %s, %s::vector) "
                "ON CONFLICT (tenant, record_id) DO UPDATE SET revision = EXCLUDED.revision, removed = false, "
                "body = EXCLUDED.body, labels = EXCLUDED.labels, embedding = EXCLUDED.embedding",
                (record.tenant_id, record.record_id, record.revision, record.text, list(record.labels), vector),
            )
        return self._receipt(record.record_id, record.revision, "applied")

    def remove(self, ctx: OperationContext, record_id: str, revision: int) -> ProjectionReceipt:
        _no_nul(record_id)
        missing = False
        with self._db(ctx) as conn:
            self._lock(conn, ctx.tenant_id, record_id)
            row = conn.execute(
                f"SELECT revision, removed FROM {self._schema}.records WHERE tenant = %s AND record_id = %s",
                (ctx.tenant_id, record_id),
            ).fetchone()
            if row is None:
                # Never applied: commit a content-free fence at this revision so a delayed older
                # apply cannot bring the erased record back, then report the missing record.
                conn.execute(
                    f"INSERT INTO {self._schema}.records (tenant, record_id, revision, removed) "
                    "VALUES (%s, %s, %s, true)",
                    (ctx.tenant_id, record_id, revision),
                )
                missing = True
            elif row[1]:
                if revision > row[0]:  # a repeated removal only ever raises the fence
                    conn.execute(
                        f"UPDATE {self._schema}.records SET revision = %s WHERE tenant = %s AND record_id = %s",
                        (revision, ctx.tenant_id, record_id),
                    )
                missing = True
            else:
                if revision < row[0]:
                    raise PortFailure(InternalErrorCode.STALE_REVISION)
                conn.execute(
                    f"UPDATE {self._schema}.records SET revision = %s, removed = true, body = '', "
                    "labels = '{}', embedding = NULL WHERE tenant = %s AND record_id = %s",
                    (revision, ctx.tenant_id, record_id),
                )
        if missing:  # raised only after the fence is committed
            raise PortFailure(InternalErrorCode.MISSING_RESOURCE)
        return self._receipt(record_id, revision, "removed")

    def get(self, ctx: OperationContext, record_id: str) -> Optional[EngineRecord]:
        _no_nul(record_id)
        with self._db(ctx) as conn:
            row = conn.execute(
                f"SELECT revision, removed, body, labels FROM {self._schema}.records "
                "WHERE tenant = %s AND record_id = %s",
                (ctx.tenant_id, record_id),
            ).fetchone()
        if row is None or row[1]:
            return None
        return EngineRecord(
            record_id=record_id,
            revision=row[0],
            tenant_id=ctx.tenant_id,
            text=row[2],
            labels=tuple(row[3]),
        )

    def search(
        self, ctx: OperationContext, query: str, flt: TenantFilter, limit: int
    ) -> tuple[SearchHit, ...]:
        _check_tenant(ctx, flt.tenant_id)
        if limit <= 0 or flt.authorized_record_ids == ():
            return ()
        limit = min(limit, MAX_SEARCH_LIMIT)  # one call never reads a whole tenant
        _no_nul(query, *flt.labels_all, *(flt.authorized_record_ids or ()))
        vector = self._vector(query)
        authorized = None if flt.authorized_record_ids is None else list(flt.authorized_record_ids)
        with self._db(ctx) as conn:
            rows = conn.execute(
                f"SELECT record_id, revision, 1 - (embedding <=> %s::vector) FROM {self._schema}.records "
                "WHERE tenant = %s AND NOT removed AND labels @> %s::text[] "
                "AND (%s::text[] IS NULL OR record_id = ANY(%s::text[])) "
                "ORDER BY embedding <=> %s::vector, record_id LIMIT %s",
                (vector, flt.tenant_id, list(flt.labels_all), authorized, authorized, vector, limit),
            ).fetchall()
        return tuple(
            SearchHit(record_id=row[0], revision=row[1], score=float(row[2])) for row in rows
        )


def build_from_settings(settings: Any, embed: Callable[[str], list[float]]) -> Optional[PgVectorAdapter]:
    """Build the adapter from ``ENGINE_PGVECTOR_*`` settings; ``None`` when off.

    Not wired into the service: callers decide when to register it.
    """

    if settings.engine_pgvector_enabled is not True:
        return None
    return PgVectorAdapter(
        settings.engine_pgvector_dsn,
        embed=embed,
        dimensions=settings.engine_pgvector_dimensions,
        schema=settings.engine_pgvector_schema,
    )


__all__ = [
    "ADAPTER_VERSION",
    "PgVectorAdapter",
    "SCHEMA_VERSION",
    "SchemaVersionError",
    "build_from_settings",
]
