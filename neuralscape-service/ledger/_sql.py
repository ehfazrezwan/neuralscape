"""Shared SQL implementation of ``LedgerPort`` over two thin dialect hooks.

Dialect subclasses supply the connection, transaction scope, ledger clock
expressions, sequence allocation and the claim query; everything else (idempotency,
fencing, validated status writes, snapshots) is identical so both backends
are held to one behavior.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
import time
from datetime import datetime, timezone
from typing import Any, Iterator, Sequence

from contracts_lifecycle import (
    Intent,
    IntentStatus,
    StageError,
    validate_required_stage_claim,
)
from contracts_references import ReferenceHandle
from contracts_lifecycle import ProcessingStage, StageReceipt, StageStatus
from ledger import port as _port
from ledger.port import (
    BOOL_COLUMNS,
    FENCED_REFUSED_RECEIPT_STATUSES,
    FENCING_STATUSES,
    SNAPSHOT_COLUMNS,
    SNAPSHOT_FORMAT,
    SNAPSHOT_ORDER,
    SNAPSHOT_TABLES,
    TERMINAL_STATUSES,
    TIME_COLUMNS,
    UNLEASED_RECEIPT_STATUSES,
    AcceptResult,
    Claim,
    DuplicateIntentId,
    AuthorityDenied,
    DurabilityError,
    ErasureAudit,
    ErasureResult,
    ExpiryRefused,
    Hold,
    RETIRED_KEY_PREFIX,
    HoldActive,
    StaleIntent,
    VerifiedErasure,
    is_reserved_reason,
    verify_erasure_token,
    IdempotencyConflict,
    LedgerError,
    LedgerPort,
    ObjectRef,
    ObjectRefConflict,
    OutboxEntry,
    PublicationFenced,
    ReceiptMismatch,
    ResurrectionDenied,
    SchemaError,
    StaleLease,
    StoredIntent,
    Tombstone,
    UnknownIntent,
    checked_intent,
    checked_object_refs,
    applied_evidence_conflict,
    checked_receipt,
    claimability_problem,
    checked_status_write,
    declared_effect_ids,
    derive_aggregate,
    load_migrations,
    plan_migrations,
    snapshot_digest,
    validate_snapshot,
)

# Statuses a dispatcher may only write while holding a live claim.
_LEASED_TARGETS = frozenset(
    {
        IntentStatus.PROCESSING,
        IntentStatus.APPLIED,
        IntentStatus.PARTIAL,
        IntentStatus.FAILED,
    }
)


_RANK = {StageStatus.PENDING: 0, StageStatus.PROCESSING: 1}
_RANK = {s: _RANK.get(s, 2) for s in StageStatus}


def _utc(epoch: float) -> datetime:
    return datetime.fromtimestamp(epoch, tz=timezone.utc)


def _refs_json(refs: Sequence[ReferenceHandle]) -> str:
    return json.dumps([json.loads(ref.model_dump_json()) for ref in refs], sort_keys=True)


def _refs_from_json(text: str) -> tuple[ReferenceHandle, ...]:
    return tuple(ReferenceHandle.model_validate(item) for item in json.loads(text))


class SqlLedger(LedgerPort):
    dialect: str
    _integrity_errors: tuple[type[BaseException], ...]

    def __init__(self, durability_check_interval: float = 30.0) -> None:
        self._lock = threading.RLock()
        self._conn: Any = None
        self._durability_interval = durability_check_interval
        self._durability_checked = float("-inf")

    # ---- dialect hooks -------------------------------------------------
    def _txn(self) -> Any:  # context manager yielding the connection
        raise NotImplementedError

    def _read_txn(self) -> Any:
        raise NotImplementedError

    def _sql(self, sql: str) -> str:
        return sql

    NOW = "ledger_now()"

    def _epoch(self, column: str) -> str:
        return column

    def _lock_suffix(self) -> str:
        return ""

    def _ins_placeholder(self, column: str) -> str:
        return "?"

    def _snap_expr(self, column: str) -> str:
        if column in TIME_COLUMNS:
            return self._epoch(column)
        if column in BOOL_COLUMNS:
            return f"CASE WHEN {column} THEN 1 ELSE 0 END"
        return column

    def _after_restore(self, c: Any) -> None:
        return None

    def _lock_record(self, c: Any, tenant_id: str, record_id: str) -> None:
        """Serialize writers that decide on one record (holds, tombstones, references).

        SQLite already serializes all writers; PostgreSQL takes a transaction-scoped advisory lock.
        """
        return None

    def _is_intent_id_violation(self, exc: BaseException) -> bool:
        """True only for a violation of the intent (tenant, id) primary key."""

        raise NotImplementedError

    def _durability_problems(self) -> list[str]:
        raise NotImplementedError

    # Outbox operations.  The defaults issue direct SQL; PostgreSQL overrides
    # them with SECURITY DEFINER functions scoped by tenant.
    def _op_claim(self, c: Any, owner: str, lease_seconds: float, tenant_id: str | None) -> tuple | None:
        raise NotImplementedError

    def _op_check(self, c: Any, claim: Claim) -> bool:
        """Lock the obligation row; True when the claim holds a live lease."""

        row = self._one(
            c,
            f"SELECT state='leased' AND lease_until >= {self.NOW} FROM outbox"
            " WHERE id=? AND tenant_id=? AND intent_id=? AND lease_owner=? AND attempt=?"
            + self._lock_suffix(),
            (claim.outbox_id, claim.tenant_id, claim.intent_id, claim.owner, claim.attempt),
        )
        return bool(row and row[0])

    def _op_extend(self, c: Any, claim: Claim, lease_seconds: float) -> float | None:
        raise NotImplementedError

    def _op_complete(self, c: Any, claim: Claim) -> bool:
        cur = self._x(
            c,
            f"UPDATE outbox SET state='done' WHERE id=? AND tenant_id=? AND lease_owner=? AND attempt=?"
            f" AND state='leased' AND lease_until >= {self.NOW}",
            (claim.outbox_id, claim.tenant_id, claim.owner, claim.attempt),
        )
        if cur.rowcount == 1:
            return True
        return self._one(
            c,
            "SELECT 1 FROM outbox WHERE id=? AND tenant_id=? AND lease_owner=? AND attempt=? AND state='done'",
            (claim.outbox_id, claim.tenant_id, claim.owner, claim.attempt),
        ) is not None

    def _op_release(self, c: Any, claim: Claim) -> bool:
        cur = self._x(
            c,
            "UPDATE outbox SET state='pending', queued=?, lease_owner=NULL, lease_until=NULL"
            f" WHERE id=? AND tenant_id=? AND lease_owner=? AND attempt=? AND state='leased'"
            f" AND lease_until >= {self.NOW}",
            (self._bool(False), claim.outbox_id, claim.tenant_id, claim.owner, claim.attempt),
        )
        return cur.rowcount == 1

    def _op_mark_queued(self, c: Any, tenant_id: str, outbox_id: int) -> None:
        self._x(c, "UPDATE outbox SET queued=? WHERE id=? AND tenant_id=? AND state='pending'",
                (self._bool(True), outbox_id, tenant_id))

    def _op_take_unqueued(self, c: Any, tenant_id: str, outbox_id: int, attempt: int) -> bool:
        cur = self._x(
            c, "UPDATE outbox SET queued=? WHERE id=? AND tenant_id=? AND attempt=? AND state='pending' AND NOT queued",
            (self._bool(True), outbox_id, tenant_id, attempt),
        )
        return cur.rowcount == 1

    def _op_unmark_queued(self, c: Any, tenant_id: str, outbox_id: int, attempt: int) -> None:
        self._x(c, "UPDATE outbox SET queued=? WHERE id=? AND tenant_id=? AND attempt=? AND state='pending'",
                (self._bool(False), outbox_id, tenant_id, attempt))

    def _op_close_outbox(self, c: Any, tenant_id: str, intent_id: str, state: str) -> None:
        self._x(c, "UPDATE outbox SET state=? WHERE intent_id=? AND tenant_id=? AND state IN ('pending','leased')",
                (state, intent_id, tenant_id))

    def _assert_durable(self) -> None:
        now = time.monotonic()
        if now - self._durability_checked < self._durability_interval:
            return
        problems = self._durability_problems()
        if problems:
            raise DurabilityError("; ".join(problems))
        self._durability_checked = now

    def check_durability(self) -> None:
        with self._lock:
            self._durability_checked = float("-inf")
            self._assert_durable()

    # ---- helpers -------------------------------------------------------
    def _x(self, c: Any, sql: str, params: Sequence[Any] = ()) -> Any:
        return c.execute(self._sql(sql), tuple(params))

    def _one(self, c: Any, sql: str, params: Sequence[Any] = ()) -> tuple | None:
        return self._x(c, sql, params).fetchone()

    def _load_intent(self, c: Any, tenant_id: str, intent_id: str, lock: bool = False) -> StoredIntent:
        row = self._one(
            c,
            "SELECT body, status, version, " + self._epoch("accepted_at")
            + " FROM intents WHERE tenant_id=? AND id=?" + (self._lock_suffix() if lock else ""),
            (tenant_id, intent_id),
        )
        if row is None:
            raise UnknownIntent(intent_id)
        return StoredIntent(
            Intent.model_validate_json(row[0]), IntentStatus(row[1]), row[2], _utc(row[3])
        )

    def _load_receipts(self, c: Any, tenant_id: str, intent_id: str) -> tuple[StageReceipt, ...]:
        rows = self._x(
            c,
            "SELECT body FROM stage_receipts WHERE tenant_id=? AND intent_id=? ORDER BY version",
            (tenant_id, intent_id),
        ).fetchall()
        return tuple(StageReceipt.model_validate_json(r[0]) for r in rows)

    def _require_live(self, c: Any, claim: Claim) -> None:
        if not self._op_check(c, claim):
            raise StaleLease("claim is not a live lease (expired, stolen, completed or fenced)")

    def _claim_from_row(self, c: Any, row: tuple, owner: str) -> Claim:
        return Claim(row[0], row[1], row[2], owner, row[3], _utc(row[4]))

    @staticmethod
    def _positive(seconds: float) -> float:
        if isinstance(seconds, bool) or not isinstance(seconds, (int, float)):
            raise TypeError("lease_seconds must be a number")
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("lease_seconds must be positive and finite")
        return float(seconds)

    # ---- port ----------------------------------------------------------
    def accept(self, intent: Intent, objects: Sequence[ObjectRef] = ()) -> AcceptResult:
        intent = checked_intent(intent)
        refs = checked_object_refs(objects, intent.tenant_id)
        with self._lock:
            self._assert_durable()
            try:
                with self._txn() as c:
                    return self._accept_in(c, intent, refs)
            except self._integrity_errors as exc:
                if self._is_intent_id_violation(exc):
                    raise DuplicateIntentId(intent.id) from exc
                raise

    def _accept_in(self, c: Any, intent: Intent, refs: Sequence[ObjectRef]) -> AcceptResult:
        scope = (intent.tenant_id, intent.actor_id, intent.operation, intent.idempotency_key)
        find = (
            "SELECT id, request_digest, status, version FROM intents"
            " WHERE tenant_id=? AND actor_id=? AND operation=? AND idempotency_key=?"
        )

        def replay(row: tuple) -> AcceptResult:
            if row[1] != intent.request_digest:
                raise IdempotencyConflict("idempotency key reused with a different request digest")
            return AcceptResult(row[0], False, IntentStatus(row[2]), row[3])

        row = self._one(c, find, scope)
        if row is not None:
            return replay(row)
        if self._one(c, "SELECT 1 FROM intents WHERE tenant_id=? AND id=?", (intent.tenant_id, intent.id)):
            raise DuplicateIntentId(intent.id)
        cur = self._x(
            c,
            "INSERT INTO intents(id,tenant_id,actor_id,credential_id,operation,"
            "idempotency_key,request_digest,status,body,accepted_at,version,receipt_version,updated_at)"
            f" VALUES(?,?,?,?,?,?,?,?,?,{self.NOW},1,0,{self.NOW})"
            " ON CONFLICT(tenant_id,actor_id,operation,idempotency_key) DO NOTHING",
            (intent.id, intent.tenant_id, intent.actor_id, intent.credential_id,
             intent.operation, intent.idempotency_key, intent.request_digest,
             IntentStatus.ACCEPTED.value, intent.model_dump_json()),
        )
        if cur.rowcount == 0:  # lost a race with a concurrent identical accept
            row = self._one(c, find, scope)
            if row is None:
                raise LedgerError("intent insert conflicted without a visible row")
            return replay(row)
        self._x(
            c,
            "INSERT INTO outbox(intent_id,tenant_id,state) VALUES(?,?,'pending')",
            (intent.id, intent.tenant_id),
        )
        for ref in sorted(refs, key=lambda r: (r.record_id, r.content_revision)):  # stable lock order
            self._put_object(c, ref)
        return AcceptResult(intent.id, True, IntentStatus.ACCEPTED, 1)

    def get_intent(self, tenant_id: str, intent_id: str) -> StoredIntent | None:
        with self._lock, self._read_txn() as c:
            try:
                return self._load_intent(c, tenant_id, intent_id)
            except UnknownIntent:
                return None

    def receipts(self, tenant_id: str, intent_id: str) -> tuple[StageReceipt, ...]:
        with self._lock, self._read_txn() as c:
            self._load_intent(c, tenant_id, intent_id)
            return self._load_receipts(c, tenant_id, intent_id)

    def claim(self, owner: str, lease_seconds: float, tenant_id: str | None = None) -> Claim | None:
        if not isinstance(owner, str) or not owner:
            raise ValueError("owner must be a non-empty string")
        lease_seconds = self._positive(lease_seconds)
        with self._lock, self._txn() as c:
            row = self._op_claim(c, owner, lease_seconds, tenant_id)
            return None if row is None else self._claim_from_row(c, row, owner)

    def extend_lease(self, claim: Claim, lease_seconds: float) -> Claim:
        lease_seconds = self._positive(lease_seconds)
        with self._lock, self._txn() as c:
            expires = self._op_extend(c, claim, lease_seconds)
            if expires is None:
                raise StaleLease("claim is not a live lease (expired, stolen, completed or fenced)")
            return Claim(claim.outbox_id, claim.intent_id, claim.tenant_id, claim.owner,
                         claim.attempt, _utc(expires))

    def complete(self, claim: Claim) -> None:
        with self._lock, self._txn() as c:
            if not self._op_complete(c, claim):
                raise StaleLease("claim is not a live lease (expired, stolen, completed or fenced)")

    def release(self, claim: Claim) -> None:
        with self._lock, self._txn() as c:
            if not self._op_release(c, claim):
                raise StaleLease("claim is not a live lease (expired, stolen, completed or fenced)")

    def _bool(self, value: bool) -> Any:
        return int(value)

    def _bump_receipt_version(self, c: Any, tenant_id: str, intent_id: str) -> int:
        # The intent row is already locked by the caller, so versions are
        # contiguous per intent.
        return self._one(
            c,
            "UPDATE intents SET receipt_version=receipt_version+1 WHERE tenant_id=? AND id=?"
            " RETURNING receipt_version",
            (tenant_id, intent_id),
        )[0]

    def record_receipt(self, receipt: StageReceipt, tenant_id: str, claim: Claim | None = None) -> int:
        # The claimability walk can take seconds for a very large receipt set, so it runs before the write
        # transaction (unless this exact receipt was just validated against the current history); the
        # transaction only confirms that the history it saw is still the current one.
        if not self._claimable_cached(receipt, tenant_id):
            self.check_receipt_claimable(receipt, tenant_id)
        with self._lock, self._txn() as c:
            stored = self._load_intent(c, tenant_id, receipt.intent_id, lock=True)
            receipt = checked_receipt(receipt, stored.intent)
            if stored.status in FENCING_STATUSES and receipt.status in FENCED_REFUSED_RECEIPT_STATUSES:
                raise PublicationFenced(f"intent is {stored.status.value}")
            if receipt.status not in UNLEASED_RECEIPT_STATUSES:
                if claim is None:
                    raise StaleLease("a live claim is required for this receipt status")
                if (claim.intent_id, claim.tenant_id) != (receipt.intent_id, tenant_id):
                    raise StaleLease("claim belongs to a different intent")
                if receipt.attempt != claim.attempt:
                    raise ReceiptMismatch("receipt attempt must equal the claim's attempt")
                self._require_live(c, claim)
            body = receipt.model_dump_json()
            row = self._one(
                c,
                "SELECT body, version, status FROM stage_receipts"
                " WHERE tenant_id=? AND intent_id=? AND stage=? AND attempt=?",
                (tenant_id, receipt.intent_id, receipt.stage.value, receipt.attempt),
            )
            existing = None if row is None else StageReceipt.model_validate_json(row[0])
            if existing == receipt:
                return row[1]
            if stored.status in TERMINAL_STATUSES:
                raise PublicationFenced(f"intent is {stored.status.value}; its history is closed")
            self._require_claimable(c, stored.intent, tenant_id, receipt)
            if existing is not None:
                # One record per (stage, attempt) that only moves forward:
                # pending -> processing -> a single terminal outcome.
                if _RANK[receipt.status] <= _RANK[existing.status]:
                    raise ReceiptMismatch("conflicting receipt for the same stage attempt")
                version = self._bump_receipt_version(c, tenant_id, receipt.intent_id)
                self._x(
                    c,
                    f"UPDATE stage_receipts SET version=?, status=?, effect_id=?, body=?, recorded_at={self.NOW}"
                    " WHERE tenant_id=? AND intent_id=? AND stage=? AND attempt=?",
                    (version, receipt.status.value, receipt.effect_id, body,
                     tenant_id, receipt.intent_id, receipt.stage.value, receipt.attempt),
                )
                return version
            version = self._bump_receipt_version(c, tenant_id, receipt.intent_id)
            self._x(
                c,
                "INSERT INTO stage_receipts(version,intent_id,tenant_id,stage,attempt,status,effect_id,body,recorded_at)"
                f" VALUES(?,?,?,?,?,?,?,?,{self.NOW})",
                (version, receipt.intent_id, tenant_id, receipt.stage.value, receipt.attempt,
                 receipt.status.value, receipt.effect_id, body),
            )
            return version

    _CLAIMABLE_CACHE_SIZE = 256

    @staticmethod
    def _receipt_digest(receipt: StageReceipt) -> str:
        return hashlib.sha256(receipt.model_dump_json().encode()).hexdigest()

    def _claimable_cached(self, receipt: StageReceipt, tenant_id: str) -> bool:
        cached = self.__dict__.get("_claimable_cache", {}).get((tenant_id, receipt.intent_id))
        if cached is None:
            return False
        with self._lock, self._read_txn() as c:
            row = self._one(
                c, "SELECT receipt_version FROM intents WHERE tenant_id=? AND id=?", (tenant_id, receipt.intent_id)
            )
        return row is not None and cached == (row[0], self._receipt_digest(receipt))

    def check_receipt_claimable(self, receipt: StageReceipt, tenant_id: str) -> str | None:
        """Validate, outside any write transaction, the receipt set the intent would hold with ``receipt``.

        Returns ``None`` or the reason the claim could not walk it.  A success is remembered against the
        intent's receipt version, so the write that follows does not walk it again unless the history moved.
        """

        with self._lock, self._read_txn() as c:
            try:
                stored = self._load_intent(c, tenant_id, receipt.intent_id)
            except UnknownIntent:
                return None  # the write path reports the unknown intent itself
            version = self._one(
                c, "SELECT receipt_version FROM intents WHERE tenant_id=? AND id=?", (tenant_id, receipt.intent_id)
            )[0]
            held = tuple(
                r for r in self._load_receipts(c, tenant_id, receipt.intent_id)
                if (r.stage, r.attempt) != (receipt.stage, receipt.attempt)
            )
        if applied_evidence_conflict(held, receipt):
            return "applied receipts of one stage must report the same results"
        problem = claimability_problem(stored.intent, held + (receipt,))
        if problem is None:
            cache = self.__dict__.setdefault("_claimable_cache", {})
            cache[(tenant_id, receipt.intent_id)] = (version, self._receipt_digest(receipt))
            while len(cache) > self._CLAIMABLE_CACHE_SIZE:
                cache.pop(next(iter(cache)))
        return problem

    def _require_claimable(self, c: Any, intent: Intent, tenant_id: str, receipt: StageReceipt) -> None:
        """Refuse a receipt that would make the intent's receipt set one the claim validator cannot walk.

        The walk normally ran just before the transaction (``check_receipt_claimable``); here the
        transaction confirms the receipt history is the one that was validated, and walks it again only
        when another writer changed it in between.
        """

        version = self._one(
            c, "SELECT receipt_version FROM intents WHERE tenant_id=? AND id=?", (tenant_id, intent.id)
        )[0]
        cached = self.__dict__.get("_claimable_cache", {}).get((tenant_id, intent.id))
        if cached == (version, self._receipt_digest(receipt)):
            return
        held = tuple(
            r for r in self._load_receipts(c, tenant_id, intent.id)
            if (r.stage, r.attempt) != (receipt.stage, receipt.attempt)
        )
        if applied_evidence_conflict(held, receipt):
            raise ReceiptMismatch("applied receipts of one stage must report the same results")
        problem = claimability_problem(intent, held + (receipt,))
        if problem is not None:
            raise ReceiptMismatch(f"receipt set cannot be validated by the claim: {problem}")

    def set_status(self, tenant_id: str, intent_id: str, target: IntentStatus,
                   claim: Claim | None = None) -> IntentStatus:
        if type(target) is not IntentStatus:
            raise TypeError("target must be an IntentStatus")
        with self._lock, self._txn() as c:
            return self._write_status(c, tenant_id, intent_id, target, claim)

    def _write_status(self, c: Any, tenant_id: str, intent_id: str, target: IntentStatus,
                      claim: Claim | None, *, trusted: bool = False) -> IntentStatus:
        stored = self._load_intent(c, tenant_id, intent_id, lock=True)
        receipts = self._load_receipts(c, tenant_id, intent_id)
        checked_status_write(stored.status, target, stored.intent, receipts)
        if target in _LEASED_TARGETS and stored.status is not target and not trusted:
            if claim is None:
                raise StaleLease("a live claim is required for this status")
            if (claim.intent_id, claim.tenant_id) != (intent_id, tenant_id):
                raise StaleLease("claim belongs to a different intent")
            self._require_live(c, claim)
        if stored.status is target:
            return target
        self._x(
            c,
            f"UPDATE intents SET status=?, version=version+1, updated_at={self.NOW}"
            " WHERE tenant_id=? AND id=?",
            (target.value, tenant_id, intent_id),
        )
        if target in FENCING_STATUSES:
            self._op_close_outbox(c, tenant_id, intent_id, "fenced")
        elif target in TERMINAL_STATUSES:
            self._op_close_outbox(c, tenant_id, intent_id, "done")
        return target

    def derive_status(self, tenant_id: str, intent_id: str) -> IntentStatus:
        with self._lock, self._read_txn() as c:
            stored = self._load_intent(c, tenant_id, intent_id)
            return derive_aggregate(stored.intent, self._load_receipts(c, tenant_id, intent_id))

    def settle(self, tenant_id: str, intent_id: str, claim: Claim | None = None) -> IntentStatus:
        with self._lock, self._txn() as c:
            stored = self._load_intent(c, tenant_id, intent_id, lock=True)
            derived = derive_aggregate(stored.intent, self._load_receipts(c, tenant_id, intent_id))
            return self._write_status(c, tenant_id, intent_id, derived, claim)

    def record_effect(self, tenant_id: str, effect_id: str, intent_id: str) -> bool:
        with self._lock, self._txn() as c:
            stored = self._load_intent(c, tenant_id, intent_id, lock=True)
            if effect_id not in declared_effect_ids(stored.intent):
                raise ReceiptMismatch("effect_id is not declared by the intent")
            row = self._one(
                c, "SELECT intent_id FROM effects WHERE tenant_id=? AND effect_id=?", (tenant_id, effect_id)
            )
            if row is not None:
                if row[0] != intent_id:
                    raise ReceiptMismatch("effect_id already belongs to another intent")
                return False
            self._x(
                c,
                f"INSERT INTO effects(tenant_id,effect_id,intent_id,recorded_at) VALUES(?,?,?,{self.NOW})",
                (tenant_id, effect_id, intent_id),
            )
            return True

    def effect_recorded(self, tenant_id: str, effect_id: str) -> bool:
        with self._lock, self._read_txn() as c:
            return self._one(
                c, "SELECT 1 FROM effects WHERE tenant_id=? AND effect_id=?", (tenant_id, effect_id)
            ) is not None

    def scan_unqueued(self, limit: int = 100) -> tuple[OutboxEntry, ...]:
        if type(limit) is not int or limit < 1:
            raise ValueError("limit must be a positive integer")
        with self._lock, self._read_txn() as c:
            rows = self._x(
                c,
                "SELECT id, intent_id, tenant_id, attempt FROM outbox WHERE state='pending' AND NOT queued"
                " ORDER BY id LIMIT ?",
                (limit,),
            ).fetchall()
        return tuple(OutboxEntry(*row) for row in rows)

    def mark_queued(self, tenant_id: str, outbox_id: int) -> None:
        with self._lock, self._txn() as c:
            self._op_mark_queued(c, tenant_id, outbox_id)

    def take_unqueued(self, tenant_id: str, outbox_id: int, attempt: int) -> bool:
        with self._lock, self._txn() as c:
            return self._op_take_unqueued(c, tenant_id, outbox_id, attempt)

    def unmark_queued(self, tenant_id: str, outbox_id: int, attempt: int) -> None:
        with self._lock, self._txn() as c:
            self._op_unmark_queued(c, tenant_id, outbox_id, attempt)

    def _put_object(self, c: Any, ref: ObjectRef) -> bool:
        self._lock_record(c, ref.tenant_id, ref.record_id)
        if self._one(c, "SELECT 1 FROM tombstones WHERE tenant_id=? AND record_id=?",
                     (ref.tenant_id, ref.record_id)):
            raise ResurrectionDenied(ref.record_id)
        refs_json = _refs_json(ref.source_refs)
        cur = self._x(
            c,
            "INSERT INTO object_refs(tenant_id,record_id,content_revision,policy_epoch,checksum,"
            f"source_refs,recorded_at) VALUES(?,?,?,?,?,?,{self.NOW})"
            " ON CONFLICT(tenant_id,record_id,content_revision) DO NOTHING",
            (ref.tenant_id, ref.record_id, ref.content_revision,
             ref.policy_epoch, ref.checksum, refs_json),
        )
        if cur.rowcount == 1:
            return True
        # Someone else (possibly a concurrent transaction) holds this revision:
        # identical content is a replay, anything else is a conflict.
        row = self._one(
            c,
            "SELECT policy_epoch, checksum, source_refs FROM object_refs"
            " WHERE tenant_id=? AND record_id=? AND content_revision=?",
            (ref.tenant_id, ref.record_id, ref.content_revision),
        )
        if row is None:
            raise LedgerError("object reference conflicted without a visible row")
        if (row[0], row[1], row[2]) != (ref.policy_epoch, ref.checksum, refs_json):
            raise ObjectRefConflict("object reference is immutable")
        return False

    def put_object_ref(self, ref: ObjectRef) -> bool:
        (ref,) = checked_object_refs((ref,), ref.tenant_id)
        with self._lock, self._txn() as c:
            return self._put_object(c, ref)

    def get_object_ref(self, tenant_id: str, record_id: str, content_revision: int) -> ObjectRef | None:
        with self._lock, self._read_txn() as c:
            row = self._one(
                c,
                "SELECT policy_epoch, checksum, source_refs FROM object_refs"
                " WHERE tenant_id=? AND record_id=? AND content_revision=?",
                (tenant_id, record_id, content_revision),
            )
        if row is None:
            return None
        return ObjectRef(tenant_id, record_id, content_revision, row[0], _refs_from_json(row[2]), row[1])

    def _active_hold_ids(self, c: Any, tenant_id: str, record_id: str) -> tuple[str, ...]:
        return tuple(r[0] for r in self._x(
            c,
            "SELECT hold_id FROM retention_holds WHERE tenant_id=? AND record_id=? AND released_at IS NULL"
            " ORDER BY hold_id",
            (tenant_id, record_id),
        ).fetchall())

    def _tombstone_in(self, c: Any, tenant_id: str, record_id: str, content_revision: int,
                      reason_code: str, *, override: bool = False) -> bool:
        self._lock_record(c, tenant_id, record_id)
        if not override:
            holds = self._active_hold_ids(c, tenant_id, record_id)
            if holds:
                raise HoldActive(record_id, holds)
        if self._one(c, "SELECT 1 FROM tombstones WHERE tenant_id=? AND record_id=?",
                     (tenant_id, record_id)):
            return False
        # Lineage for an erased record is removed; only the fence remains.
        self._x(c, "DELETE FROM object_refs WHERE tenant_id=? AND record_id=?", (tenant_id, record_id))
        cur = self._x(
            c,
            "INSERT INTO tombstones(tenant_id,record_id,content_revision,reason_code,recorded_at)"
            f" VALUES(?,?,?,?,{self.NOW}) ON CONFLICT(tenant_id,record_id) DO NOTHING",
            (tenant_id, record_id, content_revision, reason_code),
        )
        return cur.rowcount == 1

    def tombstone(self, tenant_id: str, record_id: str, content_revision: int, reason_code: str) -> bool:
        if not reason_code:
            raise ValueError("reason_code is required")
        if is_reserved_reason(reason_code):
            raise ValueError("owner_erasure reasons are reserved for erase_record")
        with self._lock, self._txn() as c:
            return self._tombstone_in(c, tenant_id, record_id, content_revision, reason_code)

    def get_tombstone(self, tenant_id: str, record_id: str) -> Tombstone | None:
        with self._lock, self._read_txn() as c:
            row = self._one(
                c,
                "SELECT content_revision, reason_code FROM tombstones WHERE tenant_id=? AND record_id=?",
                (tenant_id, record_id),
            )
        return None if row is None else Tombstone(tenant_id, record_id, row[0], row[1])

    # ---- holds, audit, owner-override erasure ---------------------------
    def has_object_refs(self, tenant_id: str, record_id: str) -> bool:
        with self._lock, self._read_txn() as c:
            return self._one(
                c, "SELECT 1 FROM object_refs WHERE tenant_id=? AND record_id=? LIMIT 1",
                (tenant_id, record_id),
            ) is not None

    def object_heads(
        self, after: tuple[str, str] | None = None, limit: int = 500
    ) -> list[tuple[str, str, int]]:
        """One page of ``(tenant, record, latest content revision)`` over the object references,
        in ``(tenant, record)`` order after ``after``, so a reader can walk them page by page."""

        return self._heads(
            "SELECT tenant_id, record_id, MAX(content_revision) FROM object_refs", "GROUP BY tenant_id, record_id",
            after, limit,
        )

    def tombstone_heads(
        self, after: tuple[str, str] | None = None, limit: int = 500
    ) -> list[tuple[str, str, int]]:
        """One page of ``(tenant, record, fenced revision)`` over the tombstones (see ``object_heads``)."""

        return self._heads("SELECT tenant_id, record_id, content_revision FROM tombstones", "", after, limit)

    def _heads(self, select: str, group: str, after: tuple[str, str] | None, limit: int) -> list[tuple[str, str, int]]:
        limit = max(1, min(int(limit), 10_000))
        tenant, record = after if after is not None else ("", "")
        with self._lock, self._read_txn() as c:
            rows = self._x(
                c,
                f"{select} WHERE tenant_id > ? OR (tenant_id = ? AND record_id > ?) {group}"
                " ORDER BY tenant_id, record_id LIMIT ?",
                (tenant, tenant, record, limit),
            ).fetchall()
        return [(row[0], row[1], int(row[2])) for row in rows]

    _HOLD_COLUMNS = (
        "tenant_id, hold_id, record_id, reason_code, holder_actor_id, holder_credential_id"
    )

    def _hold_from(self, row: tuple) -> Hold:
        return Hold(
            row[0], row[1], row[2], row[3], row[4], row[5], _utc(row[6]),
            None if row[7] is None else _utc(row[7]), row[8], row[9],
        )

    def _hold_select(self) -> str:
        return (
            f"SELECT {self._HOLD_COLUMNS}, {self._epoch('placed_at')}, {self._epoch('released_at')},"
            " released_by, release_reason FROM retention_holds"
        )

    def place_hold(self, tenant_id: str, hold_id: str, record_id: str, reason_code: str,
                   holder_actor_id: str, holder_credential_id: str) -> bool:
        for value in (tenant_id, hold_id, record_id, reason_code, holder_actor_id, holder_credential_id):
            if not isinstance(value, str) or not value:
                raise ValueError("hold fields must be non-empty strings")
        with self._lock, self._txn() as c:
            self._lock_record(c, tenant_id, record_id)
            if self._one(c, "SELECT 1 FROM tombstones WHERE tenant_id=? AND record_id=?", (tenant_id, record_id)):
                raise ResurrectionDenied(record_id)
            row = self._one(c, self._hold_select() + " WHERE tenant_id=? AND hold_id=?", (tenant_id, hold_id))
            if row is not None:
                held = self._hold_from(row)
                if (held.record_id, held.reason_code, held.holder_actor_id, held.holder_credential_id) != (
                    record_id, reason_code, holder_actor_id, holder_credential_id,
                ):
                    raise LedgerError("hold id already used with different content")
                return False
            self._x(
                c,
                "INSERT INTO retention_holds(tenant_id,hold_id,record_id,reason_code,holder_actor_id,"
                f"holder_credential_id,placed_at) VALUES(?,?,?,?,?,?,{self.NOW})",
                (tenant_id, hold_id, record_id, reason_code, holder_actor_id, holder_credential_id),
            )
            return True

    def release_hold(self, tenant_id: str, hold_id: str, released_by: str, release_reason: str) -> bool:
        if not released_by or not release_reason:
            raise ValueError("released_by and release_reason are required")
        with self._lock, self._txn() as c:
            row = self._one(c, self._hold_select() + " WHERE tenant_id=? AND hold_id=?" + self._lock_suffix(),
                            (tenant_id, hold_id))
            if row is None:
                raise LedgerError("unknown hold")
            if row[7] is not None:
                return False
            self._x(
                c,
                f"UPDATE retention_holds SET released_at={self.NOW}, released_by=?, release_reason=?"
                " WHERE tenant_id=? AND hold_id=? AND released_at IS NULL",
                (released_by, release_reason, tenant_id, hold_id),
            )
            return True

    def has_active_holds(self, tenant_id: str) -> bool:
        with self._lock, self._read_txn() as c:
            return self._one(
                c, "SELECT 1 FROM retention_holds WHERE tenant_id=? AND released_at IS NULL LIMIT 1",
                (tenant_id,),
            ) is not None

    def get_hold(self, tenant_id: str, hold_id: str) -> Hold | None:
        with self._lock, self._read_txn() as c:
            row = self._one(c, self._hold_select() + " WHERE tenant_id=? AND hold_id=?", (tenant_id, hold_id))
        return None if row is None else self._hold_from(row)

    def active_holds(self, tenant_id: str, record_id: str) -> tuple[Hold, ...]:
        with self._lock, self._read_txn() as c:
            rows = self._x(
                c,
                self._hold_select() + " WHERE tenant_id=? AND record_id=? AND released_at IS NULL ORDER BY hold_id",
                (tenant_id, record_id),
            ).fetchall()
        return tuple(self._hold_from(r) for r in rows)

    def erase_record(self, token: VerifiedErasure, intent: Intent) -> ErasureResult:
        verify_erasure_token(token)
        intent = checked_intent(intent)
        record = intent.source_preconditions[0] if intent.source_preconditions else None
        if (
            intent.operation != "memory.erase"
            or record is None
            or (token.tenant_id, token.record_id) != (intent.tenant_id, record.record_id)
            or token.actor_id != intent.actor_id
            or token.credential_id != intent.credential_id
            or token.intent_id != intent.id
            or token.content_revision != record.content_revision
        ):
            raise LedgerError("erasure token and memory.erase intent do not describe the same erasure")
        if not (_port.erasure_now() <= token.expires_at):  # no expiry, or a NaN one, counts as expired
            raise AuthorityDenied("the erasure token has expired")
        audit_id = f"erase:{intent.id}"
        with self._lock:
            self._assert_durable()
            try:
                with self._txn() as c:
                    accepted = self._accept_in(c, intent, ())
                    if not accepted.created:  # identical replay: nothing is written twice
                        row = self._one(
                            c,
                            "SELECT overridden_hold_ids FROM erasure_audit WHERE tenant_id=? AND intent_id=?",
                            (intent.tenant_id, accepted.intent_id),
                        )
                        held = tuple(json.loads(row[0])) if row else ()
                        return ErasureResult(accepted.intent_id, False, f"erase:{accepted.intent_id}", held)
                    holds = self._active_hold_ids(c, intent.tenant_id, token.record_id)
                    reason = "owner_erasure_hold_override" if holds else "owner_erasure"
                    self._tombstone_in(c, intent.tenant_id, token.record_id, token.content_revision,
                                       reason, override=True)
                    if holds:
                        self._x(
                            c,
                            f"UPDATE retention_holds SET released_at={self.NOW}, released_by=?, release_reason='erased'"
                            " WHERE tenant_id=? AND record_id=? AND released_at IS NULL",
                            (token.actor_id, intent.tenant_id, token.record_id),
                        )
                    self._x(
                        c,
                        "INSERT INTO erasure_audit(tenant_id,audit_id,record_id,content_revision,actor_id,"
                        "credential_id,authority_witness_id,overridden_hold_ids,reason_code,intent_id,recorded_at)"
                        f" VALUES(?,?,?,?,?,?,?,?,?,?,{self.NOW})",
                        (intent.tenant_id, audit_id, token.record_id, token.content_revision, token.actor_id,
                         token.credential_id, token.authority_witness_id, json.dumps(list(holds)),
                         reason, intent.id),
                    )
                    # The transaction itself is the canonical effect: record it as applied
                    # and move the intent to PROCESSING so projections follow.
                    canonical = next(r for r in intent.stage_requirements if r.stage is ProcessingStage.CANONICAL)
                    now = datetime.now(timezone.utc)
                    receipt = StageReceipt(
                        schema_version="candidate-v1", intent_id=intent.id, attempt=1,
                        stage=ProcessingStage.CANONICAL, effect_id=canonical.effect_id,
                        status=StageStatus.APPLIED,
                        applied_sources=tuple(p for p in intent.source_preconditions
                                              if p.record_id in set(canonical.precondition_source_ids)),
                        committed_sources=canonical.committed_sources, output_refs=(), error=None,
                        started_at=now, finished_at=now,
                    )
                    version = self._bump_receipt_version(c, intent.tenant_id, intent.id)
                    self._x(
                        c,
                        "INSERT INTO stage_receipts(version,intent_id,tenant_id,stage,attempt,status,effect_id,body,recorded_at)"
                        f" VALUES(?,?,?,?,?,?,?,?,{self.NOW})",
                        (version, intent.id, intent.tenant_id, receipt.stage.value, 1,
                         receipt.status.value, receipt.effect_id, receipt.model_dump_json()),
                    )
                    self._x(c, "INSERT INTO effects(tenant_id,effect_id,intent_id,recorded_at)"
                               f" VALUES(?,?,?,{self.NOW})", (intent.tenant_id, canonical.effect_id, intent.id))
                    self._write_status(c, intent.tenant_id, intent.id, IntentStatus.PROCESSING, None, trusted=True)
                    return ErasureResult(intent.id, True, audit_id, holds)
            except self._integrity_errors as exc:
                if self._is_intent_id_violation(exc):
                    raise DuplicateIntentId(intent.id) from exc
                raise

    def audit_records(self, tenant_id: str, record_id: str | None = None) -> tuple[ErasureAudit, ...]:
        sql = (
            "SELECT tenant_id, audit_id, record_id, content_revision, actor_id, credential_id,"
            f" authority_witness_id, overridden_hold_ids, reason_code, intent_id, {self._epoch('recorded_at')}"
            " FROM erasure_audit WHERE tenant_id=?"
        )
        params: list[Any] = [tenant_id]
        if record_id is not None:
            sql += " AND record_id=?"
            params.append(record_id)
        with self._lock, self._read_txn() as c:
            rows = self._x(c, sql + " ORDER BY audit_id", params).fetchall()
        return tuple(
            ErasureAudit(r[0], r[1], r[2], r[3], r[4], r[5], r[6], tuple(json.loads(r[7])), r[8], r[9], _utc(r[10]))
            for r in rows
        )

    # ---- retention ------------------------------------------------------
    RETIRED_PREFIX = RETIRED_KEY_PREFIX

    def _now_epoch(self) -> str:
        return self.NOW

    def _retired(self, intent_id: str) -> str:
        return self.RETIRED_PREFIX + hashlib.sha256(intent_id.encode()).hexdigest()[:48]

    def _held_records(self, c: Any, tenant_id: str) -> set[str]:
        rows = self._x(
            c, "SELECT record_id FROM retention_holds WHERE tenant_id=? AND released_at IS NULL", (tenant_id,)
        ).fetchall()
        return {r[0] for r in rows}

    @staticmethod
    def _intent_records(intent: Intent) -> set[str]:
        records = {item.record_id for item in intent.source_preconditions}
        for requirement in intent.stage_requirements:
            records.update(item.record_id for item in requirement.committed_sources)
        records.update(ref.id for ref in intent.target_refs)
        return records

    # The resolver of an aggregated stage output (``receipts.STAGE_OUTPUTS_RESOLVER``): one handle standing for a list
    # kept outside the ledger.  Named here because the ledger package cannot import the write path.
    AGGREGATED_OUTPUTS_RESOLVER = "ns:stage-outputs"

    def _touches_held(self, c: Any, tenant_id: str, stored_intent: Intent, intent_id: str, held: set[str]) -> bool:
        """Whether the intent touched a held record: named by the intent, or created or changed by one of its stages.

        A write that creates a memory does not know its id when it is accepted (its source is a placeholder
        record); the id exists only as a stage output.  A hold on that memory must keep the intent that created
        it, so the receipts' output references count.  An output list too long for a receipt is kept behind one
        aggregated handle the ledger cannot read: while the tenant has any active hold such an intent is left
        whole (kept, never guessed at); releasing the holds lets it be pruned.
        """

        if not held:
            return False
        if held & self._intent_records(stored_intent):
            return True
        for receipt in self._load_receipts(c, tenant_id, intent_id):
            for ref in receipt.output_refs:
                if ref.id in held or ref.resolver == self.AGGREGATED_OUTPUTS_RESOLVER:
                    return True
        return False

    def _is_retired(self, stage: str, row_key: str, row_digest: str) -> bool:
        if stage == "idempotency":
            return row_key.startswith(self.RETIRED_PREFIX)
        return row_digest.startswith(self.RETIRED_PREFIX)

    def retention_candidates(
        self, tenant_id: str, *, stage: str, older_than_seconds: float, limit: int
    ) -> tuple[str, ...]:
        if stage not in ("idempotency", "content"):
            raise ValueError("stage must be 'idempotency' or 'content'")
        self._positive(older_than_seconds)
        found: list[str] = []
        last = ""
        terminal = ", ".join(f"'{status.value}'" for status in TERMINAL_STATUSES)
        # Retired rows are filtered in SQL (their key or digest carries the prefix), so they can never
        # crowd later candidates out of a bounded scan.  Only held intents are skipped in Python.
        column = "idempotency_key" if stage == "idempotency" else "request_digest"
        width = len(self.RETIRED_PREFIX)
        with self._lock, self._read_txn() as c:
            held = self._held_records(c, tenant_id)
            while len(found) < limit:
                rows = self._x(
                    c,
                    "SELECT id, body FROM intents"
                    f" WHERE tenant_id=? AND id > ? AND status IN ({terminal})"
                    f" AND substr({column}, 1, {width}) <> '{self.RETIRED_PREFIX}'"
                    f" AND {self._epoch('updated_at')} < {self._now_epoch()} - ? ORDER BY id LIMIT ?",
                    (tenant_id, last, older_than_seconds, limit),
                ).fetchall()
                if not rows:
                    break
                for intent_id, body in rows:
                    last = intent_id
                    if held and self._touches_held(c, tenant_id, Intent.model_validate_json(body), intent_id, held):
                        continue  # a held record blocks pruning of the work that touched it
                    found.append(intent_id)
                    if len(found) >= limit:
                        break
        return tuple(found)

    def apply_retention(
        self, tenant_id: str, *, stage: str, intent_ids: Sequence[str], older_than_seconds: float
    ) -> tuple[str, ...]:
        if stage not in ("idempotency", "content"):
            raise ValueError("stage must be 'idempotency' or 'content'")
        self._positive(older_than_seconds)
        changed: list[str] = []
        for intent_id in intent_ids:
            with self._lock, self._txn() as c:
                if self._retire_in(c, tenant_id, intent_id, stage, older_than_seconds):
                    changed.append(intent_id)
        return tuple(changed)

    def _retire_in(self, c: Any, tenant_id: str, intent_id: str, stage: str, older_than_seconds: float) -> bool:
        try:
            stored = self._load_intent(c, tenant_id, intent_id, lock=True)
        except UnknownIntent:
            return False
        if stored.status not in TERMINAL_STATUSES:
            return False
        row = self._one(
            c,
            "SELECT idempotency_key, request_digest FROM intents WHERE tenant_id=? AND id=?"
            f" AND {self._epoch('updated_at')} < {self._now_epoch()} - ?",
            (tenant_id, intent_id, older_than_seconds),
        )
        if row is None or self._is_retired(stage, row[0], row[1]):
            return False
        if self._touches_held(c, tenant_id, stored.intent, intent_id, self._held_records(c, tenant_id)):
            return False
        retired = self._retired(intent_id)
        values = json.loads(stored.intent.model_dump_json())
        values["idempotency_key"] = retired
        receipts = self._load_receipts(c, tenant_id, intent_id)
        new_receipts = list(receipts)
        if stage == "content":
            values.update(request_digest=retired, correlation_id=intent_id, target_refs=[])
        try:
            new_intent = Intent.model_validate_json(json.dumps(values))
            if stage == "content":
                new_receipts = [self._compact_receipt(r, new_intent) for r in receipts]
            validate_required_stage_claim(
                claimed_status=stored.status, intent=new_intent, receipts=tuple(new_receipts)
            )
        except ValueError:
            return False  # a rewrite that would leave the evidence invalid is not made
        self._x(
            c,
            "UPDATE intents SET idempotency_key=?, request_digest=?, body=? WHERE tenant_id=? AND id=?",
            (new_intent.idempotency_key, new_intent.request_digest, new_intent.model_dump_json(),
             tenant_id, intent_id),
        )
        if stage == "content":
            for old, new in zip(receipts, new_receipts):
                if old != new:
                    self._x(
                        c,
                        "UPDATE stage_receipts SET body=? WHERE tenant_id=? AND intent_id=? AND stage=? AND attempt=?",
                        (new.model_dump_json(), tenant_id, intent_id, new.stage.value, new.attempt),
                    )
        return True

    @staticmethod
    def _compact_receipt(receipt: Any, intent: Intent) -> Any:
        """A minimal terminal record: keeps what the evidence rules need, drops the rest."""

        requirement = next(item for item in intent.stage_requirements if item.stage is receipt.stage)
        values = json.loads(receipt.model_dump_json())
        keep = requirement.minimum_output_refs if receipt.status is StageStatus.APPLIED else 0
        values["output_refs"] = values["output_refs"][:keep]
        if values.get("error"):
            values["error"]["safe_message"] = "retained as a minimal terminal record"
        return StageReceipt.model_validate_json(json.dumps(values))

    def retired_intent_ids(self, tenant_id: str, *, after_intent_id: str, limit: int) -> tuple[str, ...]:
        width = len(self.RETIRED_PREFIX)
        with self._lock, self._read_txn() as c:
            rows = self._x(
                c,
                "SELECT id FROM intents WHERE tenant_id=? AND id > ?"
                f" AND substr(request_digest, 1, {width}) = '{self.RETIRED_PREFIX}' ORDER BY id LIMIT ?",
                (tenant_id, after_intent_id, limit),
            ).fetchall()
        return tuple(r[0] for r in rows)

    def stale_processing(
        self, tenant_id: str, *, older_than_seconds: float, limit: int, after_intent_id: str = ""
    ) -> tuple[StaleIntent, ...]:
        self._positive(older_than_seconds)
        out: list[StaleIntent] = []
        with self._lock, self._read_txn() as c:
            rows = self._x(
                c,
                "SELECT i.id, i.receipt_version FROM intents i WHERE i.tenant_id=? AND i.status=? AND i.id > ?"
                f" AND {self._epoch('i.updated_at')} < {self._now_epoch()} - ?"
                " AND NOT EXISTS (SELECT 1 FROM outbox o WHERE o.tenant_id=i.tenant_id AND o.intent_id=i.id"
                f"   AND o.state='leased' AND o.lease_until >= {self.NOW})"
                " AND COALESCE((SELECT MAX(" + self._epoch("r.recorded_at") + ") FROM stage_receipts r"
                f"   WHERE r.tenant_id=i.tenant_id AND r.intent_id=i.id), 0) < {self._now_epoch()} - ?"
                " ORDER BY i.id LIMIT ?",
                (tenant_id, IntentStatus.PROCESSING.value, after_intent_id, older_than_seconds, older_than_seconds, limit),
            ).fetchall()
            for intent_id, version in rows:
                stored = self._load_intent(c, tenant_id, intent_id)
                unfinished = self._unfinished_stages(stored.intent, self._load_receipts(c, tenant_id, intent_id))
                out.append(StaleIntent(tenant_id, intent_id, version,
                                       tuple((r.stage.value, r.effect_id) for r, _ in unfinished)))
        return tuple(out)

    @staticmethod
    def _unfinished_stages(intent: Intent, receipts: tuple[Any, ...]) -> list[tuple[Any, Any]]:
        """(requirement, latest receipt or None) for stages with no applied receipt and no settled outcome."""

        latest: dict[Any, Any] = {}
        applied: set[Any] = set()
        for receipt in receipts:
            if receipt.status is StageStatus.APPLIED:
                applied.add(receipt.stage)
            held = latest.get(receipt.stage)
            if held is None or receipt.attempt >= held.attempt:
                latest[receipt.stage] = receipt
        out = []
        for requirement in intent.stage_requirements:
            if requirement.stage in applied:
                continue
            last = latest.get(requirement.stage)
            settled = last is not None and (
                last.status in (StageStatus.SKIPPED, StageStatus.CANCELLED, StageStatus.SUPERSEDED)
                or (last.status is StageStatus.FAILED and not (last.error and last.error.retryable))
            )
            if not settled:
                out.append((requirement, last))
        return out

    def expire_stale(
        self, tenant_id: str, intent_id: str, *, older_than_seconds: float, expected_receipt_version: int
    ) -> IntentStatus:
        self._positive(older_than_seconds)
        with self._lock, self._txn() as c:
            stored = self._load_intent(c, tenant_id, intent_id, lock=True)
            if stored.status is not IntentStatus.PROCESSING:
                raise ExpiryRefused("not_processing", "only a PROCESSING intent can be expired")
            row = self._one(
                c,
                "SELECT receipt_version FROM intents i WHERE i.tenant_id=? AND i.id=?"
                f" AND {self._epoch('i.updated_at')} < {self._now_epoch()} - ?"
                " AND NOT EXISTS (SELECT 1 FROM outbox o WHERE o.tenant_id=i.tenant_id AND o.intent_id=i.id"
                f"   AND o.state='leased' AND o.lease_until >= {self.NOW})"
                " AND COALESCE((SELECT MAX(" + self._epoch("r.recorded_at") + ") FROM stage_receipts r"
                f"   WHERE r.tenant_id=i.tenant_id AND r.intent_id=i.id), 0) < {self._now_epoch()} - ?",
                (tenant_id, intent_id, older_than_seconds, older_than_seconds),
            )
            if row is None:
                raise ExpiryRefused("moved", "the intent is no longer stale (a lease or receipt appeared)")
            if row[0] != expected_receipt_version:
                raise ExpiryRefused("moved", "the intent's receipt history changed since it was read")
            receipts = self._load_receipts(c, tenant_id, intent_id)
            unfinished = self._unfinished_stages(stored.intent, receipts)
            for requirement, _last in unfinished:
                # An effect is recorded separately from its receipt, so one can land after the review
                # (a worker whose lease lapsed): failing that stage would claim "not stored" over a store.
                if self._one(c, "SELECT 1 FROM effects WHERE tenant_id=? AND effect_id=?",
                             (tenant_id, requirement.effect_id)):
                    raise ExpiryRefused("effect_recorded", "an effect was recorded since the review")
            now = datetime.now(timezone.utc)
            for requirement, last in unfinished:
                reuse = last is not None and last.status in (StageStatus.PENDING, StageStatus.PROCESSING)
                attempt = last.attempt if reuse else (last.attempt + 1 if last is not None else 1)
                started = (last.started_at if reuse and last.started_at else now)
                failed = StageReceipt(
                    schema_version="candidate-v1", intent_id=intent_id, attempt=attempt,
                    stage=requirement.stage, effect_id=requirement.effect_id, status=StageStatus.FAILED,
                    applied_sources=(), committed_sources=(), output_refs=(),
                    error=StageError(code="stage_expired", retryable=False,
                                     safe_message="the stage made no progress inside the retention horizon"),
                    started_at=started, finished_at=now,
                )
                failed = checked_receipt(failed, stored.intent)
                version = self._bump_receipt_version(c, tenant_id, intent_id)
                if reuse:
                    self._x(
                        c,
                        f"UPDATE stage_receipts SET version=?, status=?, body=?, recorded_at={self.NOW}"
                        " WHERE tenant_id=? AND intent_id=? AND stage=? AND attempt=?",
                        (version, failed.status.value, failed.model_dump_json(),
                         tenant_id, intent_id, failed.stage.value, failed.attempt),
                    )
                else:
                    self._x(
                        c,
                        "INSERT INTO stage_receipts(version,intent_id,tenant_id,stage,attempt,status,effect_id,body,recorded_at)"
                        f" VALUES(?,?,?,?,?,?,?,?,{self.NOW})",
                        (version, intent_id, tenant_id, failed.stage.value, failed.attempt,
                         failed.status.value, failed.effect_id, failed.model_dump_json()),
                    )
            derived = derive_aggregate(stored.intent, self._load_receipts(c, tenant_id, intent_id))
            # no live lease exists (checked above), so the settle is the maintenance path's own decision
            return self._write_status(c, tenant_id, intent_id, derived, None, trusted=True)

    def prune_tombstones(self, tenant_id: str, *, older_than_seconds: float, limit: int) -> int:
        self._positive(older_than_seconds)
        with self._lock, self._txn() as c:
            rows = self._x(
                c,
                "SELECT t.record_id FROM tombstones t WHERE t.tenant_id=?"
                f" AND {self._epoch('t.recorded_at')} < {self._now_epoch()} - ?"
                " AND substr(t.reason_code, 1, 13) <> 'owner_erasure'"
                " AND NOT EXISTS (SELECT 1 FROM retention_holds h WHERE h.tenant_id=t.tenant_id"
                "   AND h.record_id=t.record_id AND h.released_at IS NULL)"
                " ORDER BY t.record_id LIMIT ?",
                (tenant_id, older_than_seconds, limit),
            ).fetchall()
            for (record_id,) in rows:
                self._lock_record(c, tenant_id, record_id)
                self._x(c, "DELETE FROM tombstones WHERE tenant_id=? AND record_id=?", (tenant_id, record_id))
            return len(rows)

    # ---- snapshot / restore -------------------------------------------
    def snapshot(self) -> dict[str, Any]:
        out: dict[str, Any] = {"format": SNAPSHOT_FORMAT, "tables": {}}
        with self._lock, self._read_txn() as c:
            out["schema_version"] = self.schema_version_in(c)
            for table in SNAPSHOT_TABLES:
                cols = SNAPSHOT_COLUMNS[table]
                rows = self._x(
                    c,
                    f"SELECT {', '.join(self._snap_expr(col) for col in cols)} FROM {table}"
                    f" ORDER BY {SNAPSHOT_ORDER[table]}",
                ).fetchall()
                out["tables"][table] = [dict(zip(cols, row)) for row in rows]
        out["counts"] = {name: len(rows) for name, rows in out["tables"].items()}
        out["digest"] = snapshot_digest(out)
        return out

    def restore(self, snapshot: dict[str, Any], *, expected_digest: str | None = None) -> None:
        validate_snapshot(
            snapshot,
            code_version=len(load_migrations(self.dialect)),
            expected_digest=expected_digest,
        )
        tables = snapshot["tables"]
        with self._lock, self._txn() as c:
            for table in SNAPSHOT_TABLES:
                if self._one(c, f"SELECT 1 FROM {table} LIMIT 1"):
                    raise LedgerError("restore requires an empty ledger")
            for table in SNAPSHOT_TABLES:
                cols = SNAPSHOT_COLUMNS[table]
                marks = ", ".join(self._ins_placeholder(col) for col in cols)
                for row in tables[table]:
                    row = dict(row)
                    if table == "outbox" and row["state"] == "leased":
                        row.update(state="pending", lease_owner=None, lease_until=None)
                    if table == "outbox" and row["state"] == "pending":
                        row["queued"] = 0  # leases and queue hand-off do not survive a restore
                    self._x(c, f"INSERT INTO {table}({', '.join(cols)}) VALUES({marks})",
                            [row[col] for col in cols])
            self._after_restore(c)

    def schema_version_in(self, c: Any) -> int:
        row = self._one(c, "SELECT COALESCE(MAX(version), 0) FROM ledger_schema_version")
        return row[0]

    def schema_version(self) -> int:
        with self._lock, self._read_txn() as c:
            return self.schema_version_in(c)

    # ---- migrations -----------------------------------------------------
    def _applied_migrations(self, c: Any) -> dict[int, str]:
        return {r[0]: r[1] for r in self._x(
            c, "SELECT version, checksum FROM ledger_schema_version").fetchall()}

    def _record_migration(self, c: Any, migration: Any) -> None:
        self._x(c, f"INSERT INTO ledger_schema_version(version,name,checksum,applied_at) VALUES(?,?,?,{self.NOW})",
                (migration.version, migration.name, migration.checksum))

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None


__all__ = ["SqlLedger", "plan_migrations", "SchemaError"]
