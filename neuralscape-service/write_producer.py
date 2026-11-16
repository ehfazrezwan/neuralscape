"""Durable, governed write path over the ledger (behind ``LEDGER_ENABLED``).

An accepted write is a ledger intent plus an outbox obligation committed in one
transaction; the queue only carries a wake-up.  A worker *pulls* obligations
with a ledger lease, runs the required stages (vector, then graph), records a
stage receipt per attempt, and completes with the fencing token.  Nothing here
claims a write is applied because a queue returned: ``applied`` exists only
once the ledger has validated stage receipts for every required stage.

Scope and honesty notes:

* Tenant and credential are fixed placeholders until identity and policy are
  wired (``DEFAULT_TENANT``, ``UNBOUND_CREDENTIAL``); ``actor_id`` is the
  *asserted* user id, not a verified principal.
* The ledger intent carries a digest, not the content.  The payload lives in a
  small durable store keyed by intent id (``PayloadStore``), written before the
  ledger accept so a committed intent always has its payload.
* Retry is at-least-once per stage.  The vector write is idempotent by content
  hash; the graph stage can be repeated after a crash mid-stage.  What is
  exactly-once is the ledger-visible outcome: one applied receipt per stage,
  one effect record per stage, one aggregate status.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import functools
import hashlib
import json
import logging
import sqlite3
import threading
import time
import uuid
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from typing import Any, NamedTuple, Protocol

from config import settings
from contracts_lifecycle import (
    Intent,
    IntentStatus,
    ProcessingStage,
    StageError,
    StageEvidenceRequirement,
    StageReceipt,
    StageStatus,
)
from contracts_references import ReferenceHandle, SourceVersion
from identity_binding import NamespaceKey, resolve_binding
from ledger.dispatch import Applied, Pending, Skipped, StageContext, StageFailure, StageWorker
from legacy_jobs import DELETE_CLASS, JOB_SPECS, JobSpec
from policy_enforcement import _build_principal
from ledger.port import (
    AcceptResult,
    Claim,
    HoldActive,
    IdempotencyConflict,
    LedgerPort,
    ObjectRef,
    RETIRED_KEY_PREFIX,
    DuplicateIntentId,
    ObjectRefConflict,
    OutboxEntry,
    StaleLease,
    StoredIntent,
    TERMINAL_STATUSES,
)
from ledger.stages import build_intent as build_registry_intent

logger = logging.getLogger(__name__)

DEFAULT_TENANT = "tenant-default"
UNBOUND_CREDENTIAL = "credential-unbound"
RAW_WRITE_OPERATION = "memory.write_raw"
EDIT_OPERATION = "memory.edit"
INGEST_TEXT_OPERATION = "ingest.text"
CONNECTOR_SYNC_OPERATION = "connector.sync"
DELETE_OPERATION = "memory.delete"
RAW_BATCH_OPERATION = "memory.raw_batch"
MARKDOWN_WRITE_OPERATION = "markdown.write"
STACK_PROFILE = "current_stack"
LOCAL_ISSUER = "local"
REQUIRED_STAGES = (ProcessingStage.VECTOR, ProcessingStage.GRAPH)
WAKE_FUNCTION = "process_ledger_work"


def wake_job_id(entry: OutboxEntry) -> str:
    """Queue job id of one hand-off of an obligation: the same for every sender of it."""

    return f"ledger-wake:{entry.intent_id}:{entry.outbox_id}.{entry.attempt}"


def ledger_queue_name() -> str:
    """Dedicated queue: a graph stage can run for minutes and must not occupy
    the fast write workers' slots."""

    return f"{settings.arq_queue_name}:ledger"


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def request_digest(payload: dict[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def intent_id_for(tenant_id: str, actor_id: str, operation: str, key: str) -> str:
    """Deterministic id so a replayed accept maps onto the same intent."""

    h = hashlib.sha256(canonical_json([tenant_id, actor_id, operation, key]).encode()).hexdigest()
    return f"intent-{h[:32]}"


GENERATED_KEY_PREFIX = "auto:"  # text of a generated key; whether a key was generated is never read from it


def is_intent_id(task_id: str) -> bool:
    return task_id.startswith("intent-")


# --------------------------------------------------------------- payload store


class PayloadStore(Protocol):
    def put(self, intent_id: str, payload: dict[str, Any]) -> None:
        """Idempotent: the first payload for an intent id wins."""

    def get(self, intent_id: str) -> dict[str, Any] | None: ...

    def delete_intent(self, intent_id: str) -> int:
        """Remove the intent's payload and its per-attempt blobs (keys ``<id>:...``); returns rows removed."""

    def delete_key(self, key: str) -> int:
        """Remove exactly one key; returns rows removed."""

    def keys_with_suffix(
        self, suffix: str, limit: int, after: str = ""
    ) -> tuple[tuple[str, dict[str, Any]], ...]:
        """Up to ``limit`` ``(key, body)`` pairs whose key ends with ``suffix`` and sorts after ``after``, in key order."""

    def close(self) -> None: ...


class SqlitePayloadStore:
    """Durable payloads in their own WAL/FULL SQLite file next to the ledger."""

    def __init__(self, path: str) -> None:
        from ledger.sqlite import private_database_file

        self._lock = threading.RLock()
        private_database_file(path)  # payloads hold the content of accepted writes
        self._conn = sqlite3.connect(
            path, isolation_level=None, check_same_thread=False, timeout=30
        )
        if self._conn.execute("PRAGMA journal_mode=WAL").fetchone()[0].lower() != "wal":
            raise RuntimeError("payload store requires WAL")
        self._conn.execute("PRAGMA synchronous=FULL")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS ledger_payloads("
            "intent_id TEXT PRIMARY KEY, body TEXT NOT NULL)"
        )

    def put(self, intent_id: str, payload: dict[str, Any]) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR IGNORE INTO ledger_payloads(intent_id, body) VALUES(?, ?)",
                (intent_id, canonical_json(payload)),
            )

    def get(self, intent_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT body FROM ledger_payloads WHERE intent_id=?", (intent_id,)
            ).fetchone()
        return None if row is None else json.loads(row[0])

    def delete_intent(self, intent_id: str) -> int:
        prefix = intent_id + ":"
        with self._lock:
            return self._conn.execute(
                "DELETE FROM ledger_payloads WHERE intent_id=? OR substr(intent_id, 1, ?)=?",
                (intent_id, len(prefix), prefix),
            ).rowcount

    def delete_key(self, key: str) -> int:
        with self._lock:
            return self._conn.execute("DELETE FROM ledger_payloads WHERE intent_id=?", (key,)).rowcount

    def keys_with_suffix(
        self, suffix: str, limit: int, after: str = ""
    ) -> tuple[tuple[str, dict[str, Any]], ...]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT intent_id, body FROM ledger_payloads WHERE intent_id > ?"
                " AND substr(intent_id, length(intent_id) - ? + 1) = ? ORDER BY intent_id LIMIT ?",
                (after, len(suffix), suffix, limit),
            ).fetchall()
        return tuple((k, json.loads(b)) for k, b in rows)

    def close(self) -> None:
        with self._lock:
            self._conn.close()


class PostgresPayloadStore:
    def __init__(self, dsn: str) -> None:
        import psycopg

        self._lock = threading.RLock()
        self._dsn, self._connect_timeout = dsn, 10
        self._conn = psycopg.connect(dsn, autocommit=True, connect_timeout=10)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS ledger_payloads("
            "intent_id TEXT PRIMARY KEY, body TEXT NOT NULL)"
        )

    def put(self, intent_id: str, payload: dict[str, Any]) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO ledger_payloads(intent_id, body) VALUES(%s, %s)"
                " ON CONFLICT(intent_id) DO NOTHING",
                (intent_id, canonical_json(payload)),
            )

    def get(self, intent_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT body FROM ledger_payloads WHERE intent_id=%s", (intent_id,)
            ).fetchone()
        return None if row is None else json.loads(row[0])

    def delete_intent(self, intent_id: str) -> int:
        prefix = intent_id + ":"
        with self._lock:
            return self._conn.execute(
                "DELETE FROM ledger_payloads WHERE intent_id=%s OR substr(intent_id, 1, %s)=%s",
                (intent_id, len(prefix), prefix),
            ).rowcount

    def delete_key(self, key: str) -> int:
        with self._lock:
            return self._conn.execute("DELETE FROM ledger_payloads WHERE intent_id=%s", (key,)).rowcount

    def keys_with_suffix(
        self, suffix: str, limit: int, after: str = ""
    ) -> tuple[tuple[str, dict[str, Any]], ...]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT intent_id, body FROM ledger_payloads WHERE intent_id > %s"
                " AND substr(intent_id, length(intent_id) - %s + 1) = %s ORDER BY intent_id LIMIT %s",
                (after, len(suffix), suffix, limit),
            ).fetchall()
        return tuple((k, json.loads(b)) for k, b in rows)

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def _wrap_payload_store_for_restarts() -> None:
    """A database restart drops the payload store's one connection: reconnect once and retry (all idempotent)."""

    from ledger.postgres import _reconnect, retry_on_lost_connection

    PostgresPayloadStore._reconnect = _reconnect  # type: ignore[attr-defined]
    for name in ("put", "get", "delete_intent", "delete_key", "keys_with_suffix"):
        member = PostgresPayloadStore.__dict__.get(name)
        if member is not None:
            setattr(PostgresPayloadStore, name, retry_on_lost_connection(member))


try:  # psycopg is optional for SQLite-only deployments
    _wrap_payload_store_for_restarts()
except ImportError:
    pass


def open_payload_store(cfg: Any) -> PayloadStore:
    if cfg.ledger_backend == "sqlite":
        return SqlitePayloadStore(cfg.ledger_sqlite_path + ".payloads")
    if cfg.ledger_backend == "postgres":
        return PostgresPayloadStore(cfg.ledger_postgres_dsn)
    raise ValueError("LEDGER_BACKEND must be 'sqlite' or 'postgres'")


# ------------------------------------------------------- principal and revisions


class StaleRevision(Exception):
    """The edit named a revision that is not the current one; nothing was written."""

    def __init__(self, current: int) -> None:
        super().__init__("stale_revision")
        self.current = current


class UnknownRevision(Exception):
    """The memory has no registered revision and cannot be baselined."""


class UnboundPrincipal(Exception):
    """The caller does not resolve to an active, bound principal."""


class ResolvedPrincipal(NamedTuple):
    tenant_id: str
    subject_id: str
    credential_id: str
    membership_version: int
    policy_epoch: int


class PrincipalResolver:
    """Resolve the verified user id to the bound principal and current epoch.

    Read-only against the authority store (which is also the binding store).
    An unbound, inactive or policy-less caller raises ``UnboundPrincipal``.
    """

    def __init__(self, authority: Any) -> None:
        self._authority = authority

    def subject_for_user(self, user_id: str | None) -> str | None:
        """The bound subject of a user id (immutable binding), or ``None``."""

        verified = self._scoped_credential_for(user_id)
        if verified is not None:
            return verified.subject_id
        return resolve_binding(self._authority, LOCAL_ISSUER, user_id) if user_id else None

    def _scoped_credential_for(self, user_id: str | None) -> Any:
        """The request's verified scoped credential when it acts for ``user_id``, else ``None``.

        ``user_id`` is either the credential's subject id (as the authentication layer sets
        it) or a local user id bound to that same subject.  Only the authentication layer sets
        the credential, after verifying it; request data never does.  A credential acting for
        another subject is ignored.
        """

        verified = _verified_scoped_credential()
        if verified is None or not user_id:
            return None
        if verified.subject_id == user_id:
            return verified
        if resolve_binding(self._authority, LOCAL_ISSUER, user_id) == verified.subject_id:
            return verified
        return None

    def acting_user(self, user_id: str) -> str:
        """The user id work is done as: a scoped credential's subject acts as its local alias.

        Rows are owned by local user ids, so a write made with a scoped credential is stored
        exactly as the subject's own writes are (same owner, same pool).  A subject without a
        local alias acts as its subject id.
        """

        verified = self._scoped_credential_for(user_id)
        if verified is None or verified.subject_id != user_id:
            return user_id
        alias = getattr(self._authority, "local_alias", None)
        return (alias(verified.subject_id) if alias is not None else None) or user_id

    def resolve(self, user_id: str) -> ResolvedPrincipal:
        # A verified scoped credential names its subject directly (the request's user id IS
        # that subject): there is no alias to resolve, and the principal acts under the
        # presented credential, so its retirement and its scope apply to the queued work.
        verified = self._scoped_credential_for(user_id)
        subject_id = (
            verified.subject_id if verified is not None
            else resolve_binding(self._authority, LOCAL_ISSUER, user_id)
        )
        if subject_id is None:
            raise UnboundPrincipal("unbound subject")
        record = self._authority.subject_record(subject_id)
        if record is None:
            raise UnboundPrincipal("not an active member")
        if verified is not None and record.tenant_id != verified.tenant_id:
            raise UnboundPrincipal("credential tenant mismatch")
        policy = self._authority.current_policy(record.tenant_id)
        if policy is None:
            raise UnboundPrincipal("tenant has no policy")
        return ResolvedPrincipal(
            record.tenant_id,
            subject_id,
            verified.credential_id if verified is not None else record.credential_id,
            record.membership_version,
            policy.policy_epoch,
        )


def _verified_scoped_credential() -> Any:
    """The verified scoped credential of the current request (set only by the auth layer)."""

    import credentials

    return credentials.current_credential.get()


class ServicePrincipalResolver:
    """Explicit worker/service-account principals for cron-started jobs.

    ``mapping`` is {tenant_id: subject_id}.  A cron job is accepted once per listed
    tenant under that subject; a tenant with no entry gets no job.  Nothing here ever
    falls back to a default user.
    """

    KINDS = frozenset({"worker", "service_account"})

    def __init__(self, authority: Any, mapping: dict[str, str]) -> None:
        self._authority = authority
        self._mapping = dict(mapping)

    @classmethod
    def from_setting(cls, authority: Any, text: str) -> "ServicePrincipalResolver":
        mapping = json.loads(text) if text else {}
        if not isinstance(mapping, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in mapping.items()):
            raise ValueError("LEDGER_CRON_SUBJECTS must be a JSON object of tenant id to subject id")
        return cls(authority, mapping)

    def tenants(self) -> list[str]:
        return sorted(self._mapping)

    def resolve(self, tenant_id: str) -> ResolvedPrincipal:
        subject_id = self._mapping.get(tenant_id)
        record = None if subject_id is None else self._authority.subject_record(subject_id)
        if record is None or record.tenant_id != tenant_id or record.subject_kind not in self.KINDS:
            raise UnboundPrincipal("no active worker or service-account subject for this tenant")
        try:
            _build_principal(subject_id, record)  # the publication fence builds the same principal
        except ValueError as exc:  # e.g. a worker subject without delegation constraints
            raise UnboundPrincipal("subject cannot form a principal for this tenant") from exc
        policy = self._authority.current_policy(tenant_id)
        if policy is None:
            raise UnboundPrincipal("tenant has no policy")
        return ResolvedPrincipal(
            tenant_id, subject_id, record.credential_id, record.membership_version, policy.policy_epoch
        )


_RAW_BASE_KEYS = ("content", "user_id", "category", "scope", "project_id", "tags", "agent_id", "run_id")


def _raw_payload(item: dict[str, Any]) -> dict[str, Any]:
    """A raw-write payload from a batch item (same shape ``enqueue_raw`` builds)."""

    extras = {k: v for k, v in item.items() if k not in _RAW_BASE_KEYS and v is not None}
    return {
        "content": item["content"], "user_id": item["user_id"], "category": item["category"],
        "scope": item.get("scope", "global"), "project_id": item.get("project_id"),
        "tags": item.get("tags"), "agent_id": item.get("agent_id"), "run_id": item.get("run_id"),
        "v2_extras": extras,
    }


def _workspace(value: str | None) -> str | None:
    return None if value in (None, "", "memory") else value


def namespace_for(
    principal: ResolvedPrincipal,
    visibility: str | None,
    project: str | None,
    workspace: str | None,
    owner_subject: str | None = None,
) -> NamespaceKey:
    """The pool a request or row lives in.  A private pool belongs to
    ``owner_subject`` (the row's owner when editing someone else's row),
    defaulting to the caller."""

    pool = visibility or "private"
    return NamespaceKey(
        principal.tenant_id,
        pool,
        (owner_subject or principal.subject_id) if pool == "private" else None,
        project or None,
        _workspace(workspace),
    )


class MemoryRevisions:
    """Memory revisions, kept as the ledger's immutable object references.

    Revision ``n`` of memory ``m`` is the reference ``(tenant, m, n)``.  An edit
    registers revision ``n + 1`` in the same transaction that accepts the edit
    intent, so two edits that name the same expected revision cannot both
    commit.
    """

    def __init__(self, ledger: LedgerPort) -> None:
        self.ledger = ledger

    def latest(self, tenant_id: str, memory_id: str) -> int:
        revision = 0
        while self.ledger.get_object_ref(tenant_id, memory_id, revision + 1) is not None:
            revision += 1
        return revision

    def register_baseline(
        self, tenant_id: str, memory_id: str, checksum: str, intent_id: str, policy_epoch: int = 0
    ) -> bool:
        return self.ledger.put_object_ref(
            ObjectRef(
                tenant_id,
                memory_id,
                1,
                policy_epoch,
                (
                    ReferenceHandle(kind="intent", id=intent_id, tenant_id=tenant_id, resolver="ns:intent"),
                ),
                checksum,
            )
        )


# ------------------------------------------------------------------- producer


class DurableWriteProducer:
    """API-side: commit the intent and outbox obligation, then wake a worker."""

    def __init__(
        self,
        ledger: LedgerPort,
        payloads: PayloadStore,
        *,
        tenant_id: str = DEFAULT_TENANT,
        principals: PrincipalResolver | None = None,
        service_principals: ServicePrincipalResolver | None = None,
        now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self.ledger = ledger
        self.payloads = payloads
        self.tenant_id = tenant_id
        self.principals = principals
        self.service_principals = service_principals
        self.revisions = MemoryRevisions(ledger)
        self._now = now

    # ---- identity ------------------------------------------------------
    def principal_for(self, user_id: str) -> ResolvedPrincipal | None:
        """The bound principal when authority is configured, else ``None``."""

        return None if self.principals is None else self.principals.resolve(user_id)

    def _acting(self, user_id: str) -> str:
        """The user id the work is done as (see ``PrincipalResolver.acting_user``)."""

        return user_id if self.principals is None or not user_id else self.principals.acting_user(user_id)

    def tenant_for(self, user_id: str | None) -> str | None:
        """Tenant to read receipts under; ``None`` means the caller is unbound."""

        if self.principals is None:
            return self.tenant_id
        if not user_id:
            return None
        try:
            return self.principals.resolve(user_id).tenant_id
        except UnboundPrincipal:
            return None

    def _identity(self, user_id: str) -> tuple[str, str, str, ResolvedPrincipal | None]:
        principal = self.principal_for(user_id)
        if principal is None:
            return self.tenant_id, user_id, UNBOUND_CREDENTIAL, None
        return principal.tenant_id, principal.subject_id, principal.credential_id, principal

    @staticmethod
    def _admission(principal: ResolvedPrincipal | None, namespace: NamespaceKey | None) -> dict | None:
        if principal is None or namespace is None:
            return None
        return {
            "tenant_id": principal.tenant_id,
            "subject_id": principal.subject_id,
            "credential_id": principal.credential_id,
            "membership_version": principal.membership_version,
            "policy_epoch": principal.policy_epoch,
            "namespace": list(namespace),
        }

    # ---- intents -------------------------------------------------------
    def _intent_id(self, tenant: str, actor: str, operation: str, key: str) -> str:
        """The id for a command key: deterministic, unless the key's earlier intent was retired.

        Inside the idempotency window a repeated key maps onto the same intent (a replay).  After
        retention frees the key, the old intent keeps its id, so the same key is new work and gets a
        fresh deterministic id (key plus a generation).
        """

        generation = 0
        while True:
            seed = key if generation == 0 else f"{key}\x00retired-{generation}"
            candidate = intent_id_for(tenant, actor, operation, seed)
            stored = self.ledger.get_intent(tenant, candidate)
            if stored is None or not stored.intent.idempotency_key.startswith(RETIRED_KEY_PREFIX):
                return candidate
            generation += 1

    def _intent(
        self,
        *,
        operation: str,
        user_id: str,
        digest: str,
        key: str,
        source_record: str | None,
        source_revision: int,
        identity: tuple[str, str, str, ResolvedPrincipal | None] | None = None,
    ) -> Intent:
        tenant, actor, credential, principal = identity or self._identity(user_id)
        intent_id = self._intent_id(tenant, actor, operation, key)
        record_id = source_record or f"write-{intent_id}"
        epoch = principal.policy_epoch if principal else 0
        return build_registry_intent(
            operation,
            intent_id=intent_id,
            tenant_id=tenant,
            actor_id=actor,
            credential_id=credential,
            idempotency_key=key,
            request_digest=digest,
            correlation_id=intent_id,
            preconditions=(
                SourceVersion(record_id=record_id, content_revision=source_revision, policy_epoch=epoch),
            ),
            accepted_at=self._now(),
            profile=STACK_PROFILE,
        )

    def build_intent(
        self, *, user_id: str, payload: dict[str, Any], idempotency_key: str | None
    ) -> Intent:
        # No client key: every request is its own intent.  A content-derived key
        # would coalesce a re-save of deleted content onto the old, applied
        # receipt and silently skip the write; store_raw's content-hash dedup
        # already absorbs true duplicates.
        key = idempotency_key or f"{GENERATED_KEY_PREFIX}{uuid.uuid4().hex}"
        return self._intent(
            operation=RAW_WRITE_OPERATION,
            user_id=user_id,
            digest=request_digest(payload),
            key=key,
            source_record=None,
            source_revision=1,
        )

    def _accept_stored(
        self, intent: Intent, stored: dict[str, Any], objects: Sequence[ObjectRef] = (), *, generated: bool
    ) -> AcceptResult:
        """Store the payload, then accept the intent; a generated key's payload does not outlive a failure.

        The payload goes first (idempotent, keyed by the deterministic intent id), so a committed
        intent always has its payload; a conflicting replay keeps the first payload and the ledger
        rejects it.  When the accept fails, a client's key may be retried or shared by a concurrent
        request, so its payload stays.  A key this producer generated belongs to this request alone
        and can never be retried, so its payload is removed, once the ledger confirms no intent took
        it (an accept that reports a failure may still have committed).  ``generated`` is what the
        caller knows (it made the key), never inferred from the key's text: a client may send a key
        that looks generated.
        """

        self.payloads.put(intent.id, stored)
        try:
            return self.ledger.accept(intent, objects) if objects else self.ledger.accept(intent)
        except BaseException:
            if generated:
                self._discard_unaccepted_payload(intent)
            raise

    def _discard_unaccepted_payload(self, intent: Intent) -> None:
        try:
            if self.ledger.get_intent(intent.tenant_id, intent.id) is None:
                self.payloads.delete_intent(intent.id)
        except Exception:  # noqa: BLE001 - unknown whether the accept committed: keep the payload
            logger.warning("could not confirm intent %s was not accepted; its payload is kept", intent.id)

    def accept_raw(
        self, *, user_id: str, payload: dict[str, Any], idempotency_key: str | None = None,
        _generated_key: bool = False,
    ) -> AcceptResult:
        """Durably accept one raw write.  Raises the ledger's
        ``IdempotencyConflict`` when the key was used with a different payload."""

        acting = self._acting(user_id)
        if acting != user_id:
            user_id, payload = acting, {**payload, "user_id": acting}
        intent = self.build_intent(user_id=user_id, payload=payload, idempotency_key=idempotency_key)
        principal = self.principal_for(user_id)
        namespace = None
        if principal is not None:
            extras = payload.get("v2_extras") or {}
            namespace = namespace_for(
                principal, extras.get("visibility"), payload.get("project_id"), extras.get("workspace")
            )
        stored = {"op": "write", **payload, "admission": self._admission(principal, namespace)}
        return self._accept_stored(intent, stored, generated=_generated_key or idempotency_key is None)

    def accept_edit(
        self,
        *,
        user_id: str,
        memory_id: str,
        changes: dict[str, Any],
        expected_revision: int,
        memory_row: dict[str, Any],
        idempotency_key: str | None = None,
    ) -> tuple[AcceptResult, int]:
        """Durably accept a correction of one memory at an expected revision.

        Raises ``StaleRevision`` (nothing written) when the expected revision is
        not the current one, including when a concurrent edit already took
        ``expected_revision + 1``.  Returns ``(accept_result, new_revision)``.
        """

        user_id = self._acting(user_id)

        tenant, actor, _credential, principal = self._identity(user_id)
        if idempotency_key:
            replay = self._edit_replay(tenant, actor, user_id, memory_id, changes, expected_revision, idempotency_key)
            if replay is not None:
                return replay
        latest = self.revisions.latest(tenant, memory_id)
        if latest == 0:
            # A memory written before the ledger existed: baseline it only for
            # an editor who names revision 1 and only if the row exists.
            if expected_revision != 1 or not memory_row:
                raise UnknownRevision(memory_id)
            checksum = request_digest({"content": memory_row.get("content")})
            self.revisions.register_baseline(
                tenant, memory_id, checksum, "baseline", principal.policy_epoch if principal else 0
            )
            latest = 1
        if expected_revision != latest:
            raise StaleRevision(latest)
        new_revision = expected_revision + 1
        request = {
            "memory_id": memory_id,
            "changes": changes,
            "expected_revision": expected_revision,
            "user_id": user_id,
        }
        digest = request_digest(request)
        key = idempotency_key or f"{GENERATED_KEY_PREFIX}{uuid.uuid4().hex}"
        intent = self._intent(
            operation=EDIT_OPERATION,
            user_id=user_id,
            digest=digest,
            key=key,
            source_record=memory_id,
            source_revision=new_revision,
        )
        before = after = None
        if principal is not None:
            owner = self.principals.subject_for_user(memory_row.get("owner_user_id"))
            before = namespace_for(
                principal, memory_row.get("visibility"), memory_row.get("project_id"),
                memory_row.get("workspace"), owner,
            )
            after = namespace_for(
                principal,
                changes.get("visibility", memory_row.get("visibility")),
                changes.get("project_id", memory_row.get("project_id")),
                changes.get("workspace", memory_row.get("workspace")),
                owner,
            )
        stored = {
            "op": "edit",
            "user_id": user_id,
            "memory_id": memory_id,
            "changes": changes,
            "expected_revision": expected_revision,
            "new_revision": new_revision,
            "admission": self._admission(principal, before),
            "admission_after": self._admission(principal, after),
        }
        reference = ObjectRef(
            tenant,
            memory_id,
            new_revision,
            principal.policy_epoch if principal else 0,
            (
                ReferenceHandle(kind="intent", id=intent.id, tenant_id=tenant, resolver="ns:intent"),
                ReferenceHandle(kind="memory", id=memory_id, tenant_id=tenant, resolver="ns:memory"),
            ),
            digest,
        )
        try:
            result = self._accept_stored(intent, stored, [reference], generated=idempotency_key is None)
        except ObjectRefConflict as exc:
            # A concurrent edit committed ``expected_revision + 1`` first.
            raise StaleRevision(self.revisions.latest(tenant, memory_id)) from exc
        return result, new_revision

    def accept_ingest(
        self, *, user_id: str, doc: dict[str, Any], idempotency_key: str | None = None
    ) -> AcceptResult:
        """Durably accept a text/document ingest (``ingest.text``).

        No client key: every request is its own intent (the pipeline's
        content-hash dedup absorbs true duplicates).  A key replays the
        original receipt; the same key with a different document conflicts.
        """

        user_id = self._acting(user_id)

        key = idempotency_key or f"{GENERATED_KEY_PREFIX}{uuid.uuid4().hex}"
        # The synthetic source reference stamps the request time; it is not part
        # of what the client asked for, so it must not make a replay conflict.
        stable = {**doc, "source": {k: v for k, v in (doc.get("source") or {}).items() if k != "last_synced_at"}}
        intent = self._intent(
            operation=INGEST_TEXT_OPERATION, user_id=user_id, digest=request_digest(stable), key=key,
            source_record=f"ingest-{self._intent_id(*self._identity(user_id)[:2], INGEST_TEXT_OPERATION, key)}",
            source_revision=1,
        )
        principal = self.principal_for(user_id)
        namespace = None
        if principal is not None:
            namespace = namespace_for(principal, doc.get("visibility"), doc.get("project_id"), None)
        return self._accept_stored(intent, {"op": "ingest", "doc": doc, "user_id": user_id,
                                            "admission": self._admission(principal, namespace)}, generated=idempotency_key is None)

    def accept_connector_sync(
        self, *, user_id: str, connector_id: str, record: dict[str, Any] | None = None,
        idempotency_key: str | None = None,
    ) -> AcceptResult:
        """Durably accept a connector sync (``connector.sync``).

        ``user_id`` is the connector's owner; the sync runs with the owner's
        authority and the connector's default pool.  Fetch failures surface as
        a failed stage, not as a silent skip inside a completed parent.
        """

        record = record or {}
        key = idempotency_key or f"{GENERATED_KEY_PREFIX}{uuid.uuid4().hex}"
        digest = request_digest({"connector_id": connector_id, "user_id": user_id})
        tenant, actor, _credential, principal = self._identity(user_id)
        intent_id = self._intent_id(tenant, actor, CONNECTOR_SYNC_OPERATION, key)
        intent = self._intent(
            operation=CONNECTOR_SYNC_OPERATION, user_id=user_id, digest=digest, key=key,
            source_record=f"sync-{intent_id}", source_revision=1,
        )
        namespace = None
        if principal is not None:
            namespace = namespace_for(
                principal, record.get("default_visibility"), record.get("default_project_id"), None
            )
        return self._accept_stored(intent, {"op": "connector_sync", "connector_id": connector_id,
                                            "user_id": user_id, "admission": self._admission(principal, namespace)}, generated=idempotency_key is None)

    def accept_delete(
        self, *, user_id: str, memory_id: str, memory_row: dict[str, Any] | None = None,
        idempotency_key: str | None = None,
    ) -> AcceptResult:
        """Durably accept an ordinary delete (``memory.delete``).

        Refuses early, with ``HoldActive``, while a retention hold exists; the
        tombstone stage repeats the check inside the ledger transaction.  The
        owner-override path is ``ledger.erasure`` and is never reached from here.
        """

        user_id = self._acting(user_id)

        from ledger.erasure import guard_ordinary_delete

        if self.principals is not None and memory_row is None:
            raise ValueError("memory_row is required for a delete under governance (the fence needs the row's pool)")
        tenant, actor, _credential, principal = self._identity(user_id)
        guard_ordinary_delete(self.ledger, tenant, memory_id)
        revision = self.revisions.latest(tenant, memory_id)  # 0: never registered (written before the ledger)
        key = idempotency_key or f"{GENERATED_KEY_PREFIX}{uuid.uuid4().hex}"
        digest = request_digest({"memory_id": memory_id, "user_id": user_id})
        intent = self._intent(
            operation=DELETE_OPERATION, user_id=user_id, digest=digest, key=key,
            source_record=memory_id, source_revision=revision,
        )
        namespace = None
        if principal is not None and memory_row is not None:
            owner = self.principals.subject_for_user(memory_row.get("owner_user_id"))
            namespace = namespace_for(
                principal, memory_row.get("visibility"), memory_row.get("project_id"),
                memory_row.get("workspace"), owner,
            )
        return self._accept_stored(intent, {"op": "delete", "memory_id": memory_id, "user_id": user_id,
                                            "revision": revision, "admission": self._admission(principal, namespace)}, generated=idempotency_key is None)

    # ---- generic job intents (writers that still run as one legacy function) --
    def accept_job(
        self,
        *,
        operation: str,
        payload: dict[str, Any],
        user_id: str | None = None,
        identity: tuple[str, str, str, ResolvedPrincipal | None] | None = None,
        idempotency_key: str | None = None,
        visibility: str | None = None,
        project_id: str | None = None,
        workspace: str | None = None,
        stable: Any = None,
        _generated_key: bool = False,
    ) -> AcceptResult:
        """Durably accept one job (the operation's single stage runs the legacy function).

        ``identity`` is an explicit (tenant, actor, credential, principal) for cron
        jobs; otherwise the caller's bound principal.  A delete-class job is refused
        with ``HoldActive`` while the tenant has any active retention hold, because a
        filter cannot be checked against record-scoped holds.
        """

        tenant, actor, credential, principal = identity or self._identity(user_id or "")
        if operation in DELETE_CLASS and self.ledger.has_active_holds(tenant):
            raise HoldActive("*", ())
        key = idempotency_key or f"{GENERATED_KEY_PREFIX}{uuid.uuid4().hex}"
        digest = request_digest(payload if stable is None else stable)
        intent = self._intent(
            operation=operation, user_id=user_id or actor, digest=digest, key=key,
            source_record=f"job-{self._intent_id(tenant, actor, operation, key)}", source_revision=1,
            identity=(tenant, actor, credential, principal),
        )
        namespace = None
        if principal is not None:
            namespace = namespace_for(principal, visibility, project_id, workspace)
        return self._accept_stored(intent, {"op": "job", "operation": operation, "job": payload,
                                            "admission": self._admission(principal, namespace)},
                                   generated=_generated_key or idempotency_key is None)

    def accept_system_jobs(
        self, *, operation: str, payload: dict[str, Any] | None = None, idempotency_key: str | None = None
    ) -> list[AcceptResult]:
        """Accept a cron-started job under explicit service principals, never a default user.

        Governed (authority store on): one intent per tenant that has a configured
        worker/service-account subject, none for any other tenant.  Ungoverned: a single
        intent under the explicit label ``system:<operation>``.
        """

        payload = payload or {}
        if self.principals is None:
            identity = (self.tenant_id, f"system:{operation}", UNBOUND_CREDENTIAL, None)
            return [self.accept_job(operation=operation, payload=payload, identity=identity,
                                    idempotency_key=idempotency_key)]
        results: list[AcceptResult] = []
        if self.service_principals is None:
            return results
        for tenant in self.service_principals.tenants():
            try:
                principal = self.service_principals.resolve(tenant)
            except UnboundPrincipal:
                logger.warning("no active service principal for tenant %s; %s not accepted", tenant, operation)
                continue
            identity = (tenant, principal.subject_id, principal.credential_id, principal)
            results.append(self.accept_job(operation=operation, payload=payload, identity=identity,
                                           idempotency_key=idempotency_key, visibility="shared"))
        return results

    def accept_batch(self, *, items: list[dict[str, Any]], idempotency_key: str | None = None) -> AcceptResult:
        """A batch (or checkpoint): one ``memory.write_raw`` intent per item plus a parent.

        Children are accepted first (keys derived from the batch key, so a retried
        request replays them), then the parent that links them.  The parent applies
        only when every child has settled and applied.
        """

        key = idempotency_key or f"{GENERATED_KEY_PREFIX}{uuid.uuid4().hex}"
        generated = idempotency_key is None  # the children's and the parent's keys derive from it
        children: list[list[str]] = []
        for index, item in enumerate(items):
            payload = _raw_payload(item)
            child = self.accept_raw(
                user_id=payload["user_id"], payload=payload, idempotency_key=f"batch:{key}:{index}",
                _generated_key=generated,
            )
            children.append([self._identity(payload["user_id"])[0], child.intent_id])
        first_user = items[0]["user_id"] if items else "batch"
        return self.accept_job(
            operation=RAW_BATCH_OPERATION, payload={"children": children}, user_id=first_user,
            idempotency_key=f"batch:{key}", stable={"items": items}, _generated_key=generated,
        )

    def accept_markdown_write(
        self, *, rel_path: str, content: str, user_id: str | None = None,
        identity: tuple[str, str, str, ResolvedPrincipal | None] | None = None,
    ) -> AcceptResult:
        """One vault file with an exact file effect: the key is the hash of path + content."""

        if user_id is None and identity is None:
            if self.principals is not None:
                raise UnboundPrincipal("a vault write needs a principal when authority is configured")
            identity = (self.tenant_id, f"system:{MARKDOWN_WRITE_OPERATION}", UNBOUND_CREDENTIAL, None)
        digest = hashlib.sha256(f"{rel_path}\0{content}".encode()).hexdigest()
        return self.accept_job(
            operation=MARKDOWN_WRITE_OPERATION, payload={"rel_path": rel_path, "content": content},
            user_id=user_id, identity=identity, idempotency_key=f"md:{digest}", visibility="shared",
        )

    def _edit_replay(
        self,
        tenant: str,
        actor: str,
        user_id: str,
        memory_id: str,
        changes: dict[str, Any],
        expected_revision: int,
        key: str,
    ) -> tuple[AcceptResult, int] | None:
        """A repeated key returns the original receipt even after the revision
        advanced; the same key with a different request is a conflict."""

        stored = self.ledger.get_intent(tenant, self._intent_id(tenant, actor, EDIT_OPERATION, key))
        if stored is None or stored.intent.idempotency_key.startswith(RETIRED_KEY_PREFIX):
            return None
        digest = request_digest(
            {"memory_id": memory_id, "changes": changes, "expected_revision": expected_revision, "user_id": user_id}
        )
        if stored.intent.request_digest != digest:
            raise IdempotencyConflict("idempotency key reused with a different request")
        payload = self.payloads.get(stored.intent.id) or {}
        return (
            AcceptResult(stored.intent.id, False, stored.status, stored.version),
            int(payload.get("new_revision", expected_revision + 1)),
        )

    async def dispatch_unqueued(
        self, enqueue: Callable[[str, str], Any], limit: int = 100
    ) -> int:
        """Wake a worker for every committed obligation not yet handed to the
        queue.  Safe to call repeatedly and concurrently, and after a crash
        between commit and enqueue.

        Each row is taken (marked queued) before its wake-up is sent, and only
        the dispatcher whose take succeeded sends it, so concurrent accepts that
        scan the same rows send one wake-up per obligation.  The wake-up's job id
        names the obligation and its attempt, so a repeated send is also refused
        by the queue.  A failed or cancelled enqueue returns the row to the scan;
        a crash between the take and the enqueue is recovered by the worker's poll.
        """

        entries = await asyncio.to_thread(self.ledger.scan_unqueued, limit)
        sent = 0
        for entry in entries:
            if not await asyncio.to_thread(self.ledger.take_unqueued, entry.tenant_id, entry.outbox_id, entry.attempt):
                continue  # another dispatcher sent it, or a worker already claimed it
            try:
                await enqueue(entry.intent_id, wake_job_id(entry))
            except BaseException as exc:
                # Give the row back on any failure, a cancelled enqueue (request cancelled, shutdown) too.
                # Synchronously: a second cancellation cannot interrupt a give-back that is not awaited.
                self.ledger.unmark_queued(entry.tenant_id, entry.outbox_id, entry.attempt)
                if not isinstance(exc, Exception):
                    raise
                logger.warning("ledger wake enqueue failed; row returned to the scan", exc_info=True)
                break
            sent += 1
        return sent


def _retry_once_if_key_was_retired(method: Callable[..., Any]) -> Callable[..., Any]:
    """Re-run an accept once when retention freed its key between choosing the intent id and accepting.

    The id is chosen from the ledger's current state; if the key's earlier intent was retired in that
    instant, the first attempt names a retired intent's id.  The second attempt derives a fresh one.
    A genuine id collision fails again and is raised.
    """

    @functools.wraps(method)
    def accept(self: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            return method(self, *args, **kwargs)
        except DuplicateIntentId:
            return method(self, *args, **kwargs)

    return accept


for _name in [n for n in dir(DurableWriteProducer) if n.startswith("accept_")]:
    setattr(DurableWriteProducer, _name, _retry_once_if_key_was_retired(getattr(DurableWriteProducer, _name)))
del _name


# --------------------------------------------- source rows and enforcement rows


class MemorySourceResolver:
    """``SourceResolver`` over the memory service and the revision registry.

    For a memory row it returns the row as it is now: its revision (from the
    ledger registry), visibility and namespace.  For the not-yet-existing source
    of a new write (``write-<intent>``) the only source is the admitted request
    itself, so it returns the admitted target namespace at revision 1.
    """

    def __init__(self, service: Any, ledger: LedgerPort, payloads: PayloadStore, bindings: Any) -> None:
        self._service = service
        self._revisions = MemoryRevisions(ledger)
        self._payloads = payloads
        self._bindings = bindings
        # Bindings are immutable once created, so a resolved owner is cached.
        # The fence calls ``current`` again *inside* the authority store's held
        # transaction, where that store's connection cannot be queried; the
        # unlocked evaluation just before it has already filled the cache.
        self._owners: dict[str, str] = {}

    def _owner_subject(self, owner_user: str) -> str | None:
        subject = self._owners.get(owner_user)
        if subject is None:
            subject = resolve_binding(self._bindings, LOCAL_ISSUER, owner_user)
            if subject is not None:
                self._owners[owner_user] = subject
        return subject

    def current(self, tenant_id: str, record_id: str):
        from publication_fence import CurrentSource

        prefix = next((p for p in ("write-", "ingest-", "sync-", "job-") if record_id.startswith(p)), None)
        if prefix is not None:
            payload = self._payloads.get(record_id[len(prefix):])
            admission = (payload or {}).get("admission")
            if not admission:
                return None
            key = NamespaceKey(*admission["namespace"])
            return CurrentSource(1, key, key.pool_kind)
        row = self._service.get_memory(record_id)
        if row is None:
            return None
        visibility = getattr(row, "visibility", None) or "private"
        owner = None
        if visibility == "private":
            owner_user = getattr(row, "owner_user_id", None)
            owner = self._owner_subject(owner_user) if owner_user else None
            if owner is None:
                return None  # a private row with no resolvable owner is not publishable
        key = NamespaceKey(
            tenant_id,
            visibility,
            owner,
            getattr(row, "project_id", None) or None,
            _workspace(getattr(row, "workspace", None)),
        )
        return CurrentSource(self._revisions.latest(tenant_id, record_id), key, visibility)


class MemoryRowLookup:
    """``policy_enforcement`` row lookup for memory-addressed operations."""

    def __init__(self, service: Any, bindings: Any) -> None:
        self._service = service
        self._bindings = bindings

    def __call__(self, kind: str, identifier: str):
        from policy_enforcement import RowInfo

        if kind != "memory":
            return None
        row = self._service.get_memory(identifier)
        if row is None:
            return None
        visibility = getattr(row, "visibility", None) or "private"
        owner = None
        if visibility == "private":
            owner_user = getattr(row, "owner_user_id", None)
            owner = resolve_binding(self._bindings, LOCAL_ISSUER, owner_user) if owner_user else None
        return RowInfo(
            visibility,
            owner,
            getattr(row, "project_id", None) or None,
            _workspace(getattr(row, "workspace", None)),
            visibility,
        )


# --------------------------------------------------------------------- worker


class _PublicationDenied(Exception):
    """The fence refused to publish an applied stage; remaining work is closed out."""


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _memory_ref(memory_id: str, tenant_id: str) -> ReferenceHandle:
    return ReferenceHandle(kind="memory", id=memory_id, tenant_id=tenant_id, resolver="ns:memory")


_PERMANENT = (PermissionError, ValueError, LookupError)


class _Run:
    """Per-claim state shared by the stage handlers of one intent."""

    def __init__(self, payload: dict[str, Any] | None, loop: asyncio.AbstractEventLoop | None) -> None:
        self.payload = payload
        self.loop = loop
        self.memory_id: str | None = None
        self.created: bool | None = None  # known only when the vector stage ran in this attempt
        self.delete_result: dict[str, Any] | None = None
        self.created_ids: list[str] = []  # rows a stage created, for retraction on a publication denial


class _StageHandlers(dict):
    """``dispatch`` handler table: (operation, stage) -> handler for this worker."""


def _first(value: Any) -> Any:
    return value


class _Base:
    def __init__(self, worker: "LedgerWriteWorker") -> None:
        self.w = worker

    def run(self, ctx: StageContext) -> _Run:
        return self.w._runs[ctx.intent.id]

    def payload(self, ctx: StageContext) -> dict[str, Any]:
        payload = self.run(ctx).payload
        if payload is None:
            raise StageFailure("payload_missing", "payload missing for the intent", retryable=False)
        return payload


def _project_row(tenant_id: str, memory_id: str) -> None:
    """Project the memory into the routed engine, only when ENGINE_ROUTING_ENABLED (default off).

    The row and its registered revision come from canonical state, so a retried
    stage re-projects the same thing.  A projection failure fails the stage
    (retryable): the engine copy must not silently fall behind the ledger.
    """

    from config import settings
    from engine_adapters_switch import projection_switch

    switch = projection_switch(settings)
    if switch is not None:
        switch.project_row(tenant_id, memory_id)


def _project_remove(tenant_id: str, memory_id: str) -> None:
    from config import settings
    from engine_adapters_switch import projection_switch

    switch = projection_switch(settings)
    if switch is not None:
        switch.project_remove(tenant_id, memory_id)


class _WriteVector(_Base):
    def apply(self, ctx: StageContext) -> Applied:
        payload, run = self.payload(ctx), self.run(ctx)
        if payload.get("op") == "edit":
            self.w.service.patch_memory(payload["memory_id"], payload["user_id"], dict(payload["changes"]))
            _project_row(ctx.tenant_id, payload["memory_id"])
            run.memory_id, run.created = payload["memory_id"], True
            return Applied((_memory_ref(payload["memory_id"], ctx.tenant_id),))
        extras = dict(payload.get("v2_extras") or {})
        expires_at = extras.get("expires_at")
        if isinstance(expires_at, str):
            try:
                expires_at = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
            except ValueError:
                expires_at = None
        # Ownership evidence for a later attempt: a marker written before the store call, and the
        # call's outcome written after it.  Only a death between the two leaves the row's origin
        # unknown; a recorded dedup hit or a raised call proves this attempt did not create it.
        self._mark(ctx, "started", {})
        try:
            memories, created = self.w.service.store_raw(
                content=payload["content"],
                user_id=payload["user_id"],
                category=payload["category"],
                scope=payload.get("scope", "global"),
                project_id=payload.get("project_id"),
                tags=payload.get("tags"),
                agent_id=payload.get("agent_id"),
                run_id=payload.get("run_id"),
                domain=extras.get("domain"),
                observation_type=extras.get("observation_type"),
                concepts=extras.get("concepts"),
                source_type=extras.get("source_type"),
                related_memory_ids=extras.get("related_memory_ids"),
                confidence=extras.get("confidence"),
                expires_at=expires_at,
                occurred_at=extras.get("occurred_at"),
                derived_from=extras.get("derived_from"),
                epistemic_level=extras.get("epistemic_level"),
                visibility=extras.get("visibility"),
                sensitivity_override=extras.get("sensitivity_override", False),
                memory_kind=extras.get("memory_kind"),
                source_ref=extras.get("source_ref"),
                add_to_graph=False,
                return_created=True,
            )
        except Exception:
            self._mark(ctx, "raised", {})
            raise
        if not memories:
            raise RuntimeError("store_raw returned no memory")
        memory_id = memories[0].id
        # Revision 1 of the new memory.  A dedup hit keeps the record the row already has: a second
        # intent saving the same content must not try to rewrite revision 1 under its own source.
        if created or self.w.revisions.latest(ctx.tenant_id, memory_id) < 1:
            try:
                self.w.revisions.register_baseline(
                    ctx.tenant_id, memory_id, request_digest({"content": payload["content"]}), ctx.intent.id,
                    int((payload.get("admission") or {}).get("policy_epoch", 0)),
                )
            except ObjectRefConflict:
                # a legacy backfill pass may have registered this row between the store write and here
                from ledger.backfill import is_backfill_baseline

                if not is_backfill_baseline(self.w.ledger.get_object_ref(ctx.tenant_id, memory_id, 1)):
                    raise
        _project_row(ctx.tenant_id, memory_id)
        # A row that already exists is a true duplicate only when no earlier attempt of this
        # stage could have written it: an attempt that died between the store write and its
        # receipt left a PROCESSING receipt, and its row must still reach the graph.
        self._mark(ctx, "result", {"created": bool(created), "memory_id": memory_id})
        run.memory_id, run.created = memory_id, bool(created) or self._own_earlier_write(ctx, memory_id)
        payload["_memory_id"] = memory_id
        return Applied((_memory_ref(memory_id, ctx.tenant_id),))

    def _mark(self, ctx: StageContext, kind: str, blob: dict[str, Any]) -> None:
        self.w.payloads.put(_attempt_key(ctx.intent.id, f"vector-write-{kind}", ctx.attempt), blob)

    def _own_earlier_write(self, ctx: StageContext, memory_id: str) -> bool:
        """Whether an earlier attempt of this intent wrote ``memory_id`` (a store hit is then not a dedup).

        A retry flag is not evidence: an earlier attempt may have failed before the store was
        touched, or deduplicated onto another intent's row.  Per attempt, the recorded outcome
        decides; only an attempt with a start marker and no outcome (it died in flight) counts as
        possibly having written the row, and enriching that row is the safe side of the doubt.
        """

        payloads, intent_id = self.w.payloads, ctx.intent.id
        for attempt in range(1, ctx.attempt):
            result = payloads.get(_attempt_key(intent_id, "vector-write-result", attempt))
            if result is not None:
                if result.get("created") and result.get("memory_id") == memory_id:
                    return True
                continue
            if payloads.get(_attempt_key(intent_id, "vector-write-raised", attempt)) is not None:
                continue
            if payloads.get(_attempt_key(intent_id, "vector-write-started", attempt)) is not None:
                return True
        return False


class _WriteGraph(_Base):
    def apply(self, ctx: StageContext) -> Applied:
        payload, run = self.payload(ctx), self.run(ctx)
        op = payload.get("op", "write")
        memory_id = run.memory_id or (payload["memory_id"] if op == "edit" else None)
        if memory_id is None:  # resumed attempt: the vector stage's output names the memory
            memory_id = next((r.id for refs in ctx.prior_outputs.values() for r in refs if r.kind == "memory"), None)
        if memory_id is None:
            raise StageFailure("no_vector_output", "graph stage has no vector output to enrich", retryable=False)
        if run.created is False and op == "write":
            return Applied()  # dedup hit: already in the graph
        if op == "edit" and not (set(payload["changes"]) & {"content", "project_id", "visibility"}):
            return Applied()  # metadata-only edit: nothing in the graph depends on it
        mem = self.w.service.get_memory(memory_id)
        if mem is None:
            raise ValueError("memory no longer in the store")
        extras = payload.get("v2_extras") or {}
        enriched = self.w.service.enrich_graph(
            content=getattr(mem, "memory", None) or payload.get("content"),
            user_id=payload["user_id"],
            project_id=getattr(mem, "project_id", None) or payload.get("project_id"),
            visibility=getattr(mem, "visibility", None) or "private",
            memory_id=memory_id,
            source_ref=extras.get("source_ref"),
            graph_ontology=None,
            occurred_at=getattr(mem, "occurred_at", None),
        )
        if not enriched:
            raise RuntimeError("graph enrichment did not complete")
        return Applied()


def _ingest_doc(doc: dict[str, Any]):
    from ingest.pipeline import IngestDoc

    return IngestDoc(
        content=doc["content"],
        source=doc["source"],
        user_id=doc["user_id"],
        category=doc.get("category", "domain_knowledge"),
        scope=doc.get("scope", "global"),
        project_id=doc.get("project_id"),
        visibility=doc.get("visibility"),
        sensitivity_override=doc.get("sensitivity_override", False),
        tags=doc.get("tags"),
        agent_id=doc.get("agent_id"),
        run_id=doc.get("run_id"),
        extract_facts=doc.get("extract_facts", True),
        index_passages=doc.get("index_passages", True),
        max_chars=doc.get("max_chars"),
        overlap=doc.get("overlap"),
        adapter=doc.get("adapter", "default"),
        occurred_at=doc.get("occurred_at"),
    )


def _attempt_key(intent_id: str, kind: str, attempt: int) -> str:
    return f"{intent_id}:{kind}:{attempt}"


def _latest_stage_blob(payloads: PayloadStore, intent_id: str, kind: str, attempt: int) -> dict[str, Any] | None:
    for candidate in range(attempt, 0, -1):
        blob = payloads.get(_attempt_key(intent_id, kind, candidate))
        if blob is not None:
            return blob
    return None


class _CapturingService:
    """Delegates to the memory service and remembers which rows ``store_raw`` actually created.

    The ingest pipeline does not report created-versus-deduplicated rows for passages, so the stage
    asks the service for the flag itself; only created rows may be retracted if the fence denies.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.created: list[str] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def store_raw(self, **kwargs: Any):
        wants_flag = bool(kwargs.get("return_created"))
        kwargs["return_created"] = True
        stored, created = self._inner.store_raw(**kwargs)
        if created:
            self.created.extend(m.id for m in stored if m.id not in self.created)
        return (stored, created) if wants_flag else stored


def _store_documents(w: "LedgerWriteWorker", ctx: StageContext, documents: list[dict[str, Any]],
                     service: Any = None):
    """Ingest documents (idempotent by content hash); returns (memory ids, graph jobs)."""

    from ingest.pipeline import ingest_document

    memory_ids: list[str] = []
    graph_jobs: list[dict[str, Any]] = []
    for doc in documents:
        result = ingest_document(service or w.service, _ingest_doc(doc))
        graph_jobs.extend(
            dict(job, adapter=result.get("adapter") or doc.get("adapter")) for job in result.pop("graph_jobs", [])
        )
        for memory_id in result.get("memory_ids", []):
            if memory_id not in memory_ids:
                memory_ids.append(memory_id)
    return memory_ids, graph_jobs


def _enrich_jobs(w: "LedgerWriteWorker", jobs: list[dict[str, Any]]) -> None:
    """Run deferred fact-graph enrichment inline; a memory deleted meanwhile is skipped."""

    failed = 0
    for job in jobs:
        if w.service.get_memory(job["memory_id"]) is None:
            continue
        ontology = None
        if job.get("adapter"):
            from adapters import require_adapter

            ontology = require_adapter(job["adapter"]).graph_ontology_kwargs()
        ok = w.service.enrich_graph(
            content=job["content"],
            user_id=job["user_id"],
            project_id=job.get("project_id"),
            visibility=job.get("visibility") or "private",
            memory_id=job["memory_id"],
            source_ref=job.get("source_ref"),
            graph_ontology=ontology,
        )
        failed += 0 if ok else 1
    if failed:
        raise RuntimeError(f"graph enrichment did not complete for {failed} fact(s)")


#: The most output references one stage receipt carries.  A stage that produces more (an ingest of a
#: long document, a large batch) records ONE artifact reference instead, and the full list is kept in the
#: payload store under that reference's id.  Receipts therefore stay bounded however large the work is, so
#: they can always be validated and settled.
MAX_STAGE_OUTPUT_REFS = 256
from receipts import STAGE_OUTPUTS_RESOLVER  # noqa: E402  (one name for writer and reader)


class StageOutputsUnavailable(LookupError):
    """The full output list behind an aggregated reference is gone (pruned or never stored).

    Distinct from "the stage produced nothing": a caller that must account for every output (an
    erasure, a support-aware removal) has to treat this as unknown and fail closed, never as empty.
    """


class StageOutputsInvalid(ValueError):
    """The stored output list does not belong to the handle that names it (tenant or intent mismatch)."""


def _bounded_outputs(
    w: "LedgerWriteWorker", ctx: StageContext, refs: Sequence[ReferenceHandle], label: str
) -> tuple[ReferenceHandle, ...]:
    refs = tuple(refs)
    if len(refs) <= MAX_STAGE_OUTPUT_REFS:
        return refs
    # The artifact id is also the payload key (``<intent>:...``), so retention removes the list with the
    # intent's other blobs and a reader can resolve the reference without any other index.  The list records
    # the tenant and intent it belongs to, and every reference in it is the stage's own tenant, so a reader
    # can check the binding the ledger itself checks for output references.
    for ref in refs:
        if ref.tenant_id != ctx.tenant_id:
            raise StageFailure("output_tenant_mismatch", "a stage output names another tenant", retryable=False)
    artifact_id = _attempt_key(ctx.intent.id, f"outputs-{label}", ctx.attempt)
    w.payloads.put(artifact_id, {
        "tenant_id": ctx.tenant_id,
        "intent_id": ctx.intent.id,
        "count": len(refs),
        "refs": [{"kind": r.kind, "id": r.id, "tenant_id": r.tenant_id, "resolver": r.resolver} for r in refs],
    })
    return (ReferenceHandle(kind="artifact", id=artifact_id, tenant_id=ctx.tenant_id, resolver=STAGE_OUTPUTS_RESOLVER),)


def stage_output_refs(
    payloads: "PayloadStore", handle: ReferenceHandle, *, expected_intent_id: str | None = None
) -> tuple[ReferenceHandle, ...]:
    """The full reference list behind an aggregated stage output (an ordinary reference is itself).

    Raises :class:`StageOutputsUnavailable` when the list is gone (never an empty result, which would
    read as "no outputs"), and :class:`StageOutputsInvalid` when the stored list is not bound to the
    handle's tenant, to the intent its key names, or to ``expected_intent_id`` when the caller gives it.
    """

    if handle.resolver != STAGE_OUTPUTS_RESOLVER:
        return (handle,)
    blob = payloads.get(handle.id)
    if blob is None:
        raise StageOutputsUnavailable(f"the output list {handle.id} is no longer available")
    intent_id = blob.get("intent_id") if isinstance(blob, dict) else None
    if (
        not isinstance(intent_id, str)
        or blob.get("tenant_id") != handle.tenant_id
        or not handle.id.startswith(intent_id + ":")
        or (expected_intent_id is not None and intent_id != expected_intent_id)
    ):
        raise StageOutputsInvalid("the output list is not bound to this tenant and intent")
    refs = tuple(
        ReferenceHandle(kind=r["kind"], id=r["id"], tenant_id=r["tenant_id"], resolver=r["resolver"])
        for r in blob["refs"]
    )
    if blob.get("count") != len(refs) or any(ref.tenant_id != handle.tenant_id for ref in refs):
        raise StageOutputsInvalid("the output list is not bound to this tenant and intent")
    return refs


class _IngestStore(_Base):
    def apply(self, ctx: StageContext) -> Applied:
        payload = self.payload(ctx)
        proxy = _CapturingService(self.w.service)
        try:
            memory_ids, jobs = _store_documents(self.w, ctx, [payload["doc"]], proxy)
        except Exception:
            self.w.retract_ids(proxy.created)  # a half-finished stage leaves nothing unreferenced
            raise
        finally:
            self.run(ctx).created_ids.extend(proxy.created)
        self.w.payloads.put(_attempt_key(ctx.intent.id, "graph", ctx.attempt), {"jobs": jobs})
        return Applied(_bounded_outputs(self.w, ctx, [_memory_ref(m, ctx.tenant_id) for m in memory_ids], "ingest"))


class _IngestGraph(_Base):
    def apply(self, ctx: StageContext) -> Applied:
        blob = _latest_stage_blob(self.w.payloads, ctx.intent.id, "graph", ctx.attempt) or {"jobs": []}
        _enrich_jobs(self.w, blob["jobs"])
        return Applied()


class _ConnectorFetch(_Base):
    """Fetch changed resources.  A failed fetch fails this stage (retryable) instead
    of disappearing inside a successful parent, as the legacy loop allowed."""

    def apply(self, ctx: StageContext) -> Applied:
        from connectors.registry import build_adapter

        payload, run = self.payload(ctx), self.run(ctx)
        vault = self.w.vault
        if vault is None:
            raise StageFailure("connectors_disabled", "connectors are disabled in this worker", retryable=False)
        record = self.w.await_(vault.get(payload["connector_id"]))
        if not record:
            raise StageFailure("connector_not_found", "connector not found", retryable=False)
        if not record.get("enabled", True):
            raise StageFailure("connector_disabled", "connector disabled", retryable=False)

        async def collect() -> dict[str, Any]:
            adapter = build_adapter(record)
            revisions = dict(record.get("last_revision_by_id") or {})
            documents: list[dict[str, Any]] = []
            skipped = failures = 0
            try:
                resources, next_cursor = await adapter.list_resources(record.get("cursor"))
                for r in resources:
                    if r.revision and revisions.get(r.external_id) == r.revision:
                        skipped += 1
                        continue
                    try:
                        content = await adapter.fetch(r)
                    except Exception:  # noqa: BLE001
                        failures += 1
                        continue
                    if not content or not content.strip():
                        continue
                    documents.append({
                        "content": content,
                        "source": adapter.source_descriptor(r),
                        "user_id": record.get("owner_user_id") or settings.default_user_id,
                        "category": record.get("default_category", "domain_knowledge"),
                        "scope": record.get("default_scope", "global"),
                        "project_id": record.get("default_project_id"),
                        "visibility": record.get("default_visibility"),
                        "_external_id": r.external_id,
                        "_revision": r.revision,
                    })
            finally:
                await adapter.aclose()
            return {"documents": documents, "next_cursor": next_cursor, "skipped": skipped,
                    "failures": failures, "revisions": revisions}

        collected = self.w.await_(collect())
        # Persisted per attempt: the store stage may run in a later attempt/worker.
        self.w.payloads.put(_attempt_key(ctx.intent.id, "docs", ctx.attempt), collected)
        if collected["failures"]:
            raise StageFailure(
                "connector_fetch_failed", "one or more resources could not be fetched", retryable=True
            )
        return Applied()


class _ConnectorStore(_Base):
    def apply(self, ctx: StageContext) -> Applied:
        payload = self.payload(ctx)
        collected = _latest_stage_blob(self.w.payloads, ctx.intent.id, "docs", ctx.attempt)
        if collected is None:
            raise StageFailure("docs_missing", "fetched documents are missing", retryable=False)
        documents = [{k: v for k, v in d.items() if not k.startswith("_")} for d in collected["documents"]]
        proxy = _CapturingService(self.w.service)
        try:
            memory_ids, jobs = _store_documents(self.w, ctx, documents, proxy)
        except Exception:
            self.w.retract_ids(proxy.created)
            raise
        finally:
            self.run(ctx).created_ids.extend(proxy.created)
        self.w.payloads.put(_attempt_key(ctx.intent.id, "graph", ctx.attempt), {"jobs": jobs})
        revisions = dict(collected["revisions"])
        for d in collected["documents"]:
            if d.get("_revision"):
                revisions[d["_external_id"]] = d["_revision"]
        # Sync state only advances once everything fetched was stored.
        self.w.await_(self.w.vault.update_sync_state(
            payload["connector_id"], cursor=collected["next_cursor"],
            last_synced_at=_utcnow().isoformat(), revisions=revisions,
        ))
        return Applied(_bounded_outputs(self.w, ctx, [_memory_ref(m, ctx.tenant_id) for m in memory_ids], "connector"))


class _DeleteCanonical(_Base):
    """The ledger tombstone is the canonical effect; a hold makes it a hard refusal."""

    def apply(self, ctx: StageContext) -> Applied:
        payload = self.payload(ctx)
        if self.w.ledger.active_holds(ctx.tenant_id, payload["memory_id"]):
            raise StageFailure("hold_active", "an active retention hold blocks this delete", retryable=False)
        # The tombstone itself is written when the stage is published (``_record``), inside the
        # publication fence's hold, so it can never commit under revoked authority.
        return Applied()


class _DeleteVector(_Base):
    def apply(self, ctx: StageContext) -> Applied:
        payload, run = self.payload(ctx), self.run(ctx)
        result = self.w.service.delete_memory(payload["memory_id"], payload["user_id"])
        _project_remove(ctx.tenant_id, payload["memory_id"])
        run.delete_result = result if isinstance(result, dict) else {}
        self.w.payloads.put(_attempt_key(ctx.intent.id, "delete", ctx.attempt), {"result": run.delete_result})
        return Applied()


class _DeleteGraph(_Base):
    def apply(self, ctx: StageContext):
        run = self.run(ctx)
        result = run.delete_result
        if result is None:  # the vector stage ran in an earlier attempt
            blob = _latest_stage_blob(self.w.payloads, ctx.intent.id, "delete", ctx.attempt)
            result = (blob or {}).get("result") or {}
        cascade = result.get("graph_cascade")
        if cascade == "resolved":
            return Applied()
        if cascade is None:
            return Skipped("graph is not configured; nothing to remove")
        raise StageFailure(
            "graph_cascade_unresolved", "graph cleanup could not be verified", retryable=False
        )


class _LegacyJob(_Base):
    """Runs the operation's legacy function (or service method) as the single stage."""

    def __init__(self, worker: "LedgerWriteWorker", spec: JobSpec) -> None:
        super().__init__(worker)
        self.spec = spec

    def apply(self, ctx: StageContext):
        payload = self.payload(ctx)
        job = payload.get("job") or {}
        spec = self.spec
        if spec.operation in DELETE_CLASS and self.w.ledger.has_active_holds(ctx.tenant_id):
            raise StageFailure("hold_active", "an active retention hold blocks this delete", retryable=False)
        if spec.kind == "worker_fn":
            import worker as worker_module

            if self.w.job_ctx is None:
                raise StageFailure("job_context_missing", "no worker context for this job", retryable=False)
            target = getattr(worker_module, spec.target)
            target = getattr(target, "__wrapped__", target)  # the cutover fence guards legacy queue jobs, not ledger stages
            result = self.w.await_(target(self.w.job_ctx, *[job.get(k) for k in spec.args]))
        else:  # service_method
            result = getattr(self.w.service, spec.target)(**{k: job.get(k) for k in spec.args})
        if isinstance(result, list):  # extract_and_store returns the stored rows
            result = {"memories": [{"id": getattr(m, "id", None)} for m in result]}
        return self._outcome(result, ctx)

    def _outcome(self, result: Any, ctx: StageContext):
        if not isinstance(result, dict):
            return Applied()
        if result.get("skipped"):
            return Skipped(str(result.get("reason") or "skipped by the job"))
        per_user = result.get("per_user")
        if result.get("errors") or (isinstance(per_user, list) and any(isinstance(r, dict) and r.get("error") for r in per_user)):
            raise StageFailure("job_reported_errors", "the job reported errors", retryable=False)
        if result.get("partial_extraction"):
            raise StageFailure("partial_extraction", "extraction was partial; retrying", retryable=True)
        if result.get("graph_cascade_unresolved_ids"):
            raise StageFailure("graph_cascade_unresolved", "graph cleanup could not be verified", retryable=False)
        if result.get("failed_ids") or result.get("truncated"):
            # A delete that left rows behind is not applied; repeating it is safe (see ``reentrancy``).
            raise StageFailure("delete_incomplete", "the delete did not reach every row", retryable=True)
        ids = [m.get("id") for m in result.get("memories", []) if isinstance(m, dict) and m.get("id")]
        ids += [i for i in result.get("memory_ids", []) if isinstance(i, str)]
        unique = list(dict.fromkeys(ids))
        return Applied(_bounded_outputs(self.w, ctx, [_memory_ref(i, ctx.tenant_id) for i in unique], "job"))


class _BatchChildren(_Base):
    """Parent of a batch: applied only when every per-item intent settled and applied."""

    def apply(self, ctx: StageContext):
        children = self.payload(ctx).get("job", {}).get("children", [])
        stored = [self.w.ledger.get_intent(tenant, child) for tenant, child in children]
        if any(item is None for item in stored):
            raise StageFailure("child_missing", "a batch item intent is missing", retryable=False)
        if any(item.status not in TERMINAL_STATUSES for item in stored):
            return Pending(self.w.batch_poll_seconds)
        bad = sum(1 for item in stored if item.status is not IntentStatus.APPLIED)
        self.w.payloads.put(_attempt_key(ctx.intent.id, "children", ctx.attempt),
                            {"total": len(stored), "not_applied": bad})
        if bad:
            raise StageFailure("children_not_applied", f"{bad} of {len(stored)} items did not apply", retryable=False)
        return Applied(_bounded_outputs(self.w, ctx, [
            ReferenceHandle(kind="intent", id=child, tenant_id=tenant, resolver="ns:intent")
            for tenant, child in children
        ], "children"))


class _MarkdownFile(_Base):
    """One vault file, written atomically; an existing identical file is the recovered effect."""

    def _target(self, ctx: StageContext):
        job = self.payload(ctx)["job"]
        writer = self.w.vault_writer
        if writer is None:
            from extensions.conversation_compiler.obsidian_writer import ObsidianWriter

            writer = ObsidianWriter()
        try:
            path = writer._safe_resolve(job["rel_path"])
        except ValueError as exc:
            raise StageFailure("path_rejected", "the vault path is outside the vault", retryable=False) from exc
        return path, job["content"], hashlib.sha256(f"{job['rel_path']}\0{job['content']}".encode()).hexdigest()

    @staticmethod
    def _ref(ctx: StageContext, digest: str) -> Applied:
        return Applied((ReferenceHandle(kind="artifact", id=f"vault-{digest[:48]}", tenant_id=ctx.tenant_id,
                                        resolver="ns:vault"),))

    def inspect(self, ctx: StageContext):
        path, content, digest = self._target(ctx)
        if path.is_file() and path.read_text(encoding="utf-8") == content:
            return self._ref(ctx, digest)
        return None

    def apply(self, ctx: StageContext):
        from extensions.conversation_compiler.obsidian_writer import _atomic_write

        path, content, digest = self._target(ctx)
        _atomic_write(path, content)
        if path.read_text(encoding="utf-8") != content:
            raise StageFailure("vault_verify_failed", "the written vault file does not match", retryable=True)
        return self._ref(ctx, digest)


class LedgerWriteWorker(StageWorker):
    """Worker-side adapter over ``ledger.dispatch.StageWorker``.

    The generic loop (claim, ordered stages, receipts that follow the claim
    attempt, lease heartbeat, retries as new attempts, settle) is the
    dispatcher's.  This class adds what V1 needs on top: the stage handlers for
    each current-stack operation, the per-stage authority pre-check and the
    publication fence around the applied receipt, retraction of a just-created
    row when the fence denies, retry back-off, and an async facade for ARQ.
    """

    def __init__(
        self,
        ledger: LedgerPort,
        payloads: PayloadStore,
        service: Any,
        *,
        owner: str,
        lease_seconds: float = 120.0,
        max_attempts: int | None = None,
        retry_delay: Callable[[int], float] = lambda attempt: min(2.0 ** attempt, 30.0),
        emit_stored: Callable[[dict[str, Any]], Any] | None = None,
        fence: Any | None = None,
        vault: Any | None = None,
        job_ctx: dict[str, Any] | None = None,
        vault_writer: Any | None = None,
        batch_poll_seconds: float = 1.0,
    ) -> None:
        handlers = _StageHandlers()
        V, G, E, C = (ProcessingStage.VECTOR, ProcessingStage.GRAPH, ProcessingStage.EXTRACT,
                      ProcessingStage.CANONICAL)
        table = {
            (RAW_WRITE_OPERATION, V): _WriteVector, (RAW_WRITE_OPERATION, G): _WriteGraph,
            (EDIT_OPERATION, V): _WriteVector, (EDIT_OPERATION, G): _WriteGraph,
            (INGEST_TEXT_OPERATION, V): _IngestStore, (INGEST_TEXT_OPERATION, G): _IngestGraph,
            (CONNECTOR_SYNC_OPERATION, E): _ConnectorFetch, (CONNECTOR_SYNC_OPERATION, V): _ConnectorStore,
            (CONNECTOR_SYNC_OPERATION, G): _IngestGraph,
            (DELETE_OPERATION, C): _DeleteCanonical, (DELETE_OPERATION, V): _DeleteVector,
            (DELETE_OPERATION, G): _DeleteGraph,
        }
        for key, handler_class in table.items():
            handlers[key] = handler_class(self)
        for spec in JOB_SPECS.values():
            handlers[(spec.operation, spec.stage)] = _LegacyJob(self, spec)
        handlers[(RAW_BATCH_OPERATION, V)] = _BatchChildren(self)
        handlers[(MARKDOWN_WRITE_OPERATION, ProcessingStage.MARKDOWN)] = _MarkdownFile(self)
        attempts = max_attempts if max_attempts is not None else max(1, settings.arq_max_retries)
        super().__init__(
            ledger, handlers, owner=owner, lease_seconds=lease_seconds,
            heartbeat_seconds=max(lease_seconds / 3.0, 0.05), max_attempts=attempts,
        )
        self.ledger, self.payloads, self.service, self.owner = ledger, payloads, service, owner
        self.lease_seconds, self.max_attempts = lease_seconds, attempts
        self.retry_delay, self.emit_stored, self.fence, self.vault = retry_delay, emit_stored, fence, vault
        self.job_ctx, self.vault_writer, self.batch_poll_seconds = job_ctx, vault_writer, batch_poll_seconds
        self.revisions = MemoryRevisions(ledger)
        self._runs: dict[str, _Run] = {}
        self._loop: asyncio.AbstractEventLoop | None = None

    # ---- async facade (ARQ and tests) -----------------------------------
    async def drain(self, limit: int = 1000) -> int:
        """Claim and process obligations until none is available."""

        self._loop = asyncio.get_running_loop()
        done = 0
        while done < limit:
            claim = await asyncio.to_thread(self.ledger.claim, self.owner, self.lease_seconds)
            if claim is None:
                return done
            result = await asyncio.to_thread(self._process_claim, claim)
            if result.lost_lease:
                logger.info("lost lease for %s; another worker owns it", claim.intent_id)
            done += 1
        return done

    async def process(self, claim: Claim) -> None:
        self._loop = asyncio.get_running_loop()
        await asyncio.to_thread(self._process_claim, claim)

    def await_(self, coroutine: Any) -> Any:
        """Run a coroutine on the worker's event loop from a stage thread."""

        if self._loop is None:
            return asyncio.run(coroutine)
        return asyncio.run_coroutine_threadsafe(coroutine, self._loop).result()

    # ---- per-claim state ------------------------------------------------
    def _process_claim(self, claim: Claim):
        payload = self.payloads.get(claim.intent_id)
        self._runs[claim.intent_id] = _Run(payload, self._loop)
        try:
            return self._run_claim(claim)
        finally:
            self._runs.pop(claim.intent_id, None)

    def _run_stages(self, intent: Intent, holder: Any, result: Any) -> None:
        try:
            super()._run_stages(intent, holder, result)
        except _PublicationDenied:
            self._close_out(intent, holder, result)

    def _receipt(
        self, stored: StoredIntent, stage: ProcessingStage, attempt: int, status: StageStatus, *,
        started: datetime | None, outputs: tuple[ReferenceHandle, ...] = (),
        error: StageError | None = None,
    ) -> StageReceipt:
        """Build a stage receipt for the intent's declared requirement (kept for callers/tests)."""

        requirement = next(r for r in stored.intent.stage_requirements if r.stage is stage)
        terminal = status not in (StageStatus.PENDING, StageStatus.PROCESSING)
        return StageReceipt(
            schema_version="candidate-v1", intent_id=stored.intent.id, attempt=attempt, stage=stage,
            effect_id=requirement.effect_id, status=status,
            applied_sources=stored.intent.source_preconditions if started else (),
            committed_sources=(), output_refs=outputs, error=error, started_at=started,
            finished_at=_utcnow() if terminal else None,
        )

    # ---- governance -------------------------------------------------------
    def _admission(self, payload: dict[str, Any] | None, stage: ProcessingStage, intent_id: str):
        """The admission claims for one stage, or ``None`` when ungoverned."""

        from publication_fence import Admission

        if not payload or not payload.get("admission") or self.fence is None:
            return None
        raw = payload["admission"]
        op = payload.get("op")
        if op == "edit":
            use = payload.get("admission_after") if stage is ProcessingStage.GRAPH else raw
            record, revision = payload["memory_id"], int(payload["new_revision"])
        elif op == "delete":
            if stage is not ProcessingStage.CANONICAL:
                return None  # past the tombstone the delete must complete, not be re-gated
            use, record, revision = raw, payload["memory_id"], int(payload["revision"])
        elif op == "job":
            use, record, revision = raw, f"job-{intent_id}", 1
        elif op == "ingest":
            use, record, revision = raw, f"ingest-{intent_id}", 1
        elif op == "connector_sync":
            use, record, revision = raw, f"sync-{intent_id}", 1
        elif stage is ProcessingStage.VECTOR:
            use, record, revision = raw, f"write-{intent_id}", 1
        else:
            use, record, revision = raw, payload.get("_memory_id") or f"write-{intent_id}", 1
        return Admission(
            use["tenant_id"], use["subject_id"], use["credential_id"],
            int(use["membership_version"]), int(use["policy_epoch"]),
            record, revision, NamespaceKey(*use["namespace"]),
        )

    def _execute(self, handler: Any, ctx: StageContext):
        run = self._runs[ctx.intent.id]
        if (
            self.fence is not None and (run.payload or {}).get("op") == "delete"
            and ctx.stage is ProcessingStage.CANONICAL and not (run.payload or {}).get("admission")
        ):
            # A governed worker never deletes on an intent that carries no admission claims.
            raise StageFailure("admission_missing", "the delete carries no admission claims", retryable=False)
        admission = self._admission(run.payload, ctx.stage, ctx.intent.id)
        if admission is not None:
            decision = self.fence.evaluate(admission)  # unlocked authority + source check before the provider call
            if not decision.allowed:
                raise StageFailure(f"policy_{decision.reason}", "stage denied by current authority", retryable=False)
        try:
            return self._bound_execute(handler, ctx)
        except StageFailure:
            raise
        except Exception as exc:  # noqa: BLE001 - recorded as a stage failure, never a raw message
            logger.warning("stage %s failed for %s: %s", ctx.stage.value, ctx.intent.id, exc)
            raise StageFailure(
                _error_code(exc), "stage failed; see service logs", retryable=not isinstance(exc, _PERMANENT)
            ) from exc

    def _tombstone(self, intent: Intent, run: _Run | None) -> None:
        payload = (run.payload if run else None) or {}
        revision = intent.source_preconditions[0].content_revision
        self.ledger.tombstone(intent.tenant_id, payload["memory_id"], revision, "user_delete")

    def _bound_execute(self, handler: Any, ctx: StageContext):
        """Run the stage with the ledger claim bound as the usage attempt context.

        Only when a usage recorder is installed (``USAGE_STORE_ENABLED``); otherwise
        this is a plain call.  Every provider call made by the stage is then
        attributed to this tenant, intent, stage and ledger attempt.
        """

        import usage_capture

        if usage_capture.get_usage_recorder() is None:
            return super()._execute(handler, ctx)
        context = usage_capture.AttemptContext(
            tenant_id=ctx.tenant_id,
            task_id=ctx.intent.id,
            attempt_id=f"{ctx.intent.id}:{ctx.stage.value}:{ctx.attempt}",
            intent_id=ctx.intent.id,
            ledger_attempt=ctx.attempt,
        )
        try:
            with usage_capture.attempt_context(context):
                return super()._execute(handler, ctx)
        finally:
            flush = getattr(usage_capture.get_usage_recorder(), "flush_gaps", None)
            if flush is not None:
                try:
                    flush()  # persist any events this stage could not store
                except Exception:
                    pass

    def _record(self, intent, requirement, claim, status, result, *, output_refs=(), error=None):
        stage = requirement.stage
        run = self._runs.get(intent.id)
        if status is StageStatus.SKIPPED and self._retry_pending(intent, requirement, claim):
            return  # an upstream failure that will be retried must not close this stage early
        admission = None
        if status is StageStatus.APPLIED and run is not None:
            admission = self._admission(run.payload, stage, intent.id)
        # The delete's irreversible effect (the tombstone) is the canonical stage's publication.
        fence_tombstone = (
            status is StageStatus.APPLIED and run is not None and (run.payload or {}).get("op") == "delete"
            and stage is ProcessingStage.CANONICAL
        )
        held: list[HoldActive] = []

        def write() -> None:
            if fence_tombstone:
                try:
                    self._tombstone(intent, run)
                except HoldActive as exc:  # a hold placed after the stage pre-check
                    held.append(exc)
                    return
            super(LedgerWriteWorker, self)._record(
                intent, requirement, claim, status, result, output_refs=output_refs, error=error
            )

        if admission is None:
            write()
        else:
            decision, _ = self.fence.publish(admission, lambda _connection=None: write())
            if not decision.allowed:
                failures = self._retract(run, intent, stage)
                code = "retraction_failed" if failures else f"policy_{decision.reason}"
                super()._record(
                    intent, requirement, claim, StageStatus.FAILED, result,
                    error=StageError(code=code, retryable=False,
                                     safe_message="stage denied by current authority"),
                )
                result.outcomes[stage] = f"failed ({code})"
                raise _PublicationDenied()
        if held:
            super()._record(
                intent, requirement, claim, StageStatus.FAILED, result,
                error=StageError(code="hold_active", retryable=False,
                                 safe_message="an active retention hold blocks this delete"),
            )
            result.outcomes[stage] = "failed (hold_active)"
            raise _PublicationDenied()
        if (
            status is StageStatus.APPLIED and stage is ProcessingStage.VECTOR and run is not None
            and (run.payload or {}).get("op", "write") == "write" and self.emit_stored is not None and run.created
        ):
            self._emit(run)

    def _retry_pending(self, intent: Intent, requirement: Any, claim: Claim) -> bool:
        if claim.attempt >= self.max_attempts or not requirement.upstream_stages:
            return False
        latest: dict[ProcessingStage, StageReceipt] = {}
        for r in self.ledger.receipts(intent.tenant_id, intent.id):
            held = latest.get(r.stage)
            if held is None or r.attempt >= held.attempt:
                latest[r.stage] = r
        return any(
            (r := latest.get(up)) is not None and r.status is StageStatus.FAILED
            and r.error is not None and r.error.retryable
            for up in requirement.upstream_stages
        )

    def retract_ids(self, memory_ids: list[str]) -> int:
        """Delete rows a stage created; returns how many deletions failed (never raises)."""

        failures = 0
        for memory_id in memory_ids:
            try:
                self.service.delete_memory(memory_id)
            except Exception:  # noqa: BLE001
                failures += 1
                logger.warning("retraction of %s failed", memory_id, exc_info=True)
        return failures

    def _retract(self, run: _Run | None, intent: Intent, stage: ProcessingStage) -> int:
        """Compensate when authority changed after the provider call: remove what the stage created.

        Returns the number of failed retractions so the receipt can say so (``retraction_failed``).
        """

        if run is None or stage is not ProcessingStage.VECTOR:
            return 0
        payload = run.payload or {}
        op = payload.get("op", "write")
        if op == "write":
            ids = [run.memory_id] if run.created and run.memory_id else []
        elif op in ("ingest", "connector_sync"):
            ids = list(run.created_ids)
        else:
            ids = []
        return self.retract_ids(ids)

    def _emit(self, run: _Run) -> None:
        if self.emit_stored is None:
            return
        with contextlib.suppress(Exception):
            outcome = self.emit_stored({"memory_id": run.memory_id, "payload": run.payload})
            if asyncio.iscoroutine(outcome):
                self.await_(outcome)

    # ---- retries and close-out ---------------------------------------------
    def _finish(self, intent: Intent, holder: Any, result: Any, failed_retryable: bool) -> None:
        claim = holder.get()
        if failed_retryable and claim.attempt < self.max_attempts:
            time.sleep(self.retry_delay(claim.attempt))
        super()._finish(intent, holder, result, failed_retryable)

    def _close_out(self, intent: Intent, holder: Any, result: Any) -> None:
        """After a publication denial: close every unsettled stage, then settle."""

        holder.freeze()
        claim, tenant = holder.get(), intent.tenant_id
        latest: dict[ProcessingStage, StageReceipt] = {}
        for r in self.ledger.receipts(tenant, intent.id):
            held = latest.get(r.stage)
            if held is None or held.status is not StageStatus.APPLIED and (
                r.status is StageStatus.APPLIED or r.attempt >= held.attempt
            ):
                latest[r.stage] = r
        try:
            for requirement in intent.stage_requirements:
                held = latest.get(requirement.stage)
                if held is None:
                    StageWorker._record(self, intent, requirement, claim, StageStatus.SKIPPED, result)
                elif held.status in (StageStatus.PENDING, StageStatus.PROCESSING):
                    StageWorker._record(
                        self, intent, requirement, claim, StageStatus.FAILED, result,
                        error=StageError(code="stage_abandoned", retryable=False,
                                         safe_message="stage abandoned"),
                    )
            result.status = self.ledger.settle(tenant, intent.id, claim)
            self.ledger.complete(claim)
        except StaleLease:
            result.lost_lease = True


def _error_code(exc: BaseException) -> str:
    if isinstance(exc, PermissionError):
        return "denied"
    if isinstance(exc, LookupError):
        return "not_found"
    if isinstance(exc, ValueError):
        return "invalid_input"
    return "stage_failed"


async def _maybe_await(value: Any) -> Any:
    if asyncio.iscoroutine(value):
        return await value
    return value


__all__ = [
    "DEFAULT_TENANT",
    "DurableWriteProducer",
    "LedgerWriteWorker",
    "MemoryRevisions",
    "MemoryRowLookup",
    "MemorySourceResolver",
    "PayloadStore",
    "PostgresPayloadStore",
    "PrincipalResolver",
    "ResolvedPrincipal",
    "SqlitePayloadStore",
    "StaleRevision",
    "UnboundPrincipal",
    "UnknownRevision",
    "intent_id_for",
    "is_intent_id",
    "ledger_queue_name",
    "namespace_for",
    "open_payload_store",
    "request_digest",
]
