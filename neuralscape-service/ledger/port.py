"""Ledger port: durable intent/outbox acceptance over the lifecycle contracts.

The port is a small transactional surface.  Backends (SQLite, PostgreSQL)
implement the same operations and must pass ``ledger.conformance``.

Authority notes:

* Ordering basis is a per-aggregate version, never a global sequence: the intent
  status version (starts at 1), the per-intent receipt version (contiguous,
  assigned under the intent's row lock) and a record's ``content_revision``.
  Outbox draining is status-based (unqueued/claimed/completed), never gap-based,
  because PostgreSQL commit order is not allocation order.
* Lease expiry is evaluated by the ledger inside the claiming/completing
  transaction: the PostgreSQL server clock, or the owning host's clock for
  SQLite.  Clock steps are tolerated, not prevented: safety rests on fencing
  tokens (owner, attempt) plus idempotent effect ids, never on lease accuracy.
  ``Intent.accepted_at`` and receipt timestamps are caller witnesses inside
  the validated body.
* Durability: process-crash recovery is tested; power-loss durability rests on
  engine semantics plus the settings asserted at startup (SQLite ``synchronous``
  FULL/EXTRA and WAL; PostgreSQL ``fsync``, ``synchronous_commit`` and
  ``full_page_writes`` on) and on local storage with honest fsync.  The ledger
  refuses to accept intents when those settings are not in force.
* Every status write re-runs ``is_legal_intent_transition`` and
  ``validate_required_stage_claim`` against the stored intent and receipts.
* The ledger is not authorization.  Tenant arguments scope every lookup, but
  callers (policy/identity) decide whether an action is permitted.
* Tombstones hold only non-content resurrection metadata.
"""

from __future__ import annotations

import abc
import hashlib
import time
import hmac
import json
import re
import secrets
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from contracts_lifecycle import (
    Intent,
    IntentStatus,
    StageEvidenceRequirement,
    StageReceipt,
    StageStatus,
    UpstreamApplication,
    _reference_sets_match,
    _source_sets_match,
    is_legal_intent_transition,
    validate_required_stage_claim,
)
from contracts_limits import DISCOVERY_BUDGET_MESSAGE
from contracts_references import ReferenceHandle, SourceVersion

SNAPSHOT_FORMAT = "ledger-snapshot-v1"
TERMINAL_STATUSES = frozenset(
    {
        IntentStatus.APPLIED,
        IntentStatus.PARTIAL,
        IntentStatus.FAILED,
        IntentStatus.CANCELLED,
        IntentStatus.SUPERSEDED,
    }
)
FENCING_STATUSES = frozenset({IntentStatus.CANCELLED, IntentStatus.SUPERSEDED})
# Receipts that need a live lease; cancellation/supersession is a client fence.
UNLEASED_RECEIPT_STATUSES = frozenset({StageStatus.CANCELLED, StageStatus.SUPERSEDED})
# Publication evidence is refused once the intent has been fenced.
FENCED_REFUSED_RECEIPT_STATUSES = frozenset({StageStatus.APPLIED, StageStatus.PROCESSING})
# Backend-neutral snapshot shape: times are ledger-clock epoch seconds and
# booleans are 0/1 so a snapshot restores into either backend.
SNAPSHOT_COLUMNS: dict[str, tuple[str, ...]] = {
    "intents": (
        "id", "tenant_id", "actor_id", "credential_id", "operation",
        "idempotency_key", "request_digest", "status", "body", "accepted_at",
        "version", "receipt_version", "updated_at",
    ),
    "outbox": (
        "id", "intent_id", "tenant_id", "state", "queued", "lease_owner",
        "lease_until", "attempt",
    ),
    "stage_receipts": (
        "version", "intent_id", "tenant_id", "stage", "attempt", "status",
        "effect_id", "body", "recorded_at",
    ),
    "effects": ("tenant_id", "effect_id", "intent_id", "recorded_at"),
    "object_refs": (
        "tenant_id", "record_id", "content_revision", "policy_epoch",
        "checksum", "source_refs", "recorded_at",
    ),
    "tombstones": (
        "tenant_id", "record_id", "content_revision", "reason_code",
        "recorded_at",
    ),
    "retention_holds": (
        "tenant_id", "hold_id", "record_id", "reason_code", "holder_actor_id",
        "holder_credential_id", "placed_at", "released_at", "released_by",
        "release_reason",
    ),
    "erasure_audit": (
        "tenant_id", "audit_id", "record_id", "content_revision", "actor_id",
        "credential_id", "authority_witness_id", "overridden_hold_ids",
        "reason_code", "intent_id", "recorded_at",
    ),
}
SNAPSHOT_TABLES = tuple(SNAPSHOT_COLUMNS)
SNAPSHOT_ORDER = {
    "intents": "id",
    "outbox": "id",
    "stage_receipts": "tenant_id, intent_id, version",
    "effects": "tenant_id, effect_id",
    "object_refs": "tenant_id, record_id, content_revision",
    "tombstones": "tenant_id, record_id",
    "retention_holds": "tenant_id, hold_id",
    "erasure_audit": "tenant_id, audit_id",
}
TIME_COLUMNS = frozenset(
    {"accepted_at", "updated_at", "lease_until", "recorded_at", "placed_at", "released_at"}
)
BOOL_COLUMNS = frozenset({"queued"})


#: Idempotency keys of intents whose retention window passed are rewritten to this prefix, which
#: frees the client key (a later use is a new key); producers derive a fresh intent id for such a key.
RETIRED_KEY_PREFIX = "retired:"


class LedgerError(Exception):
    """Base class for ledger failures."""


class ExpiryRefused(LedgerError):
    """Expiry of a stale intent was refused; ``reason`` is ``not_processing``, ``moved`` or ``effect_recorded``."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


class IdempotencyConflict(LedgerError):
    """Same tenant/actor/operation/key accepted earlier with a different digest."""


class DuplicateIntentId(LedgerError):
    """The intent id already exists under a different idempotency scope."""


class IllegalTransition(LedgerError):
    """The status write is not a legal transition or lacks stage evidence."""


class StaleLease(LedgerError):
    """The claim is expired, stolen, completed or fenced."""


class PublicationFenced(LedgerError):
    """The intent was cancelled or superseded; publication evidence is refused."""


class UnknownIntent(LedgerError):
    """No intent with that id exists for the tenant."""


class ReceiptMismatch(LedgerError):
    """The receipt does not match the intent's declared stages or effects."""


class ObjectRefConflict(LedgerError):
    """An immutable object reference already exists with different content."""


class ResurrectionDenied(LedgerError):
    """The record was tombstoned; new revisions are refused."""


class HoldActive(LedgerError):
    """An active retention hold blocks an ordinary delete or tombstone."""

    def __init__(self, record_id: str, hold_ids: tuple[str, ...]) -> None:
        super().__init__(f"record {record_id} has active retention holds")
        self.record_id, self.hold_ids = record_id, hold_ids


class AuthorityDenied(LedgerError):
    """Owner authority for an override erasure was not established."""


class DurabilityError(LedgerError):
    """A durability setting required for acknowledged accepts is not in force."""


class PrivilegeError(LedgerError):
    """The runtime role holds more privilege than the ledger permits."""


class IntentTooLarge(LedgerError):
    """The intent is too large for the claim to settle: once its stages applied it could never be claimed."""


class SchemaError(LedgerError):
    """Schema version drift or a modified applied migration."""


@dataclass(frozen=True)
class AcceptResult:
    intent_id: str
    created: bool
    status: IntentStatus
    version: int  # per-intent status version (1 at acceptance)


@dataclass(frozen=True)
class StoredIntent:
    intent: Intent
    status: IntentStatus
    version: int
    accepted_at: datetime  # ledger clock


@dataclass(frozen=True)
class Claim:
    """Fencing token: (owner, attempt) valid until the ledger-clock expiry."""

    outbox_id: int
    intent_id: str
    tenant_id: str
    owner: str
    attempt: int
    lease_expires_at: datetime


@dataclass(frozen=True)
class OutboxEntry:
    outbox_id: int
    intent_id: str
    tenant_id: str
    attempt: int = 0  # claims so far; a released obligation is handed out again under a new value


@dataclass(frozen=True)
class ObjectRef:
    """Immutable (record, revision, policy epoch) reference with source refs."""

    tenant_id: str
    record_id: str
    content_revision: int
    policy_epoch: int
    source_refs: tuple[ReferenceHandle, ...] = ()
    checksum: str | None = None


@dataclass(frozen=True)
class Tombstone:
    tenant_id: str
    record_id: str
    content_revision: int
    reason_code: str


@dataclass(frozen=True)
class Hold:
    tenant_id: str
    hold_id: str
    record_id: str
    reason_code: str
    holder_actor_id: str
    holder_credential_id: str
    placed_at: datetime
    released_at: datetime | None = None
    released_by: str | None = None
    release_reason: str | None = None

    @property
    def active(self) -> bool:
        return self.released_at is None


@dataclass(frozen=True)
class ErasureAudit:
    """Content-free, append-only record of an owner-override erasure."""

    tenant_id: str
    audit_id: str
    record_id: str
    content_revision: int
    actor_id: str
    credential_id: str
    authority_witness_id: str
    overridden_hold_ids: tuple[str, ...]
    reason_code: str
    intent_id: str
    recorded_at: datetime


_SECRET = secrets.token_bytes(32)  # per process; never serialized


def _mac(tenant_id: str, record_id: str, content_revision: int, actor_id: str, credential_id: str,
         witness_id: str, reason_code: str, intent_id: str = "", expires_at: float = 0.0) -> str:
    body = json.dumps(
        [tenant_id, record_id, content_revision, actor_id, credential_id, witness_id, reason_code,
         intent_id, expires_at],
        separators=(",", ":"),
    )
    return hmac.new(_SECRET, body.encode(), hashlib.sha256).hexdigest()


@dataclass(frozen=True)
class VerifiedErasure:
    """Owner authority, verified by ``ledger.erasure``, bound to exactly one erasure.

    The token carries an HMAC over every field under a per-process secret, so changing any field
    (``dataclasses.replace`` to retarget it at another record or actor, a different witness, a
    different reason) invalidates it, and ``erase_record`` refuses an invalid token.  This stops
    accidental or field-level misuse; code that can read this module's private state in the same
    process can still mint one, so the policy layer must remain the only caller of the ledger's
    erasure path and ``AuthorityStoreVerifier`` the only verifier it trusts.
    """

    tenant_id: str
    record_id: str
    content_revision: int
    actor_id: str
    credential_id: str
    authority_witness_id: str
    reason_code: str
    intent_id: str = ""  # the one erasure intent this token may commit
    expires_at: float = 0.0  # epoch seconds; the token is a same-request artifact, not a standing grant
    mac: str = field(default="", repr=False, compare=False)


def erasure_now() -> float:
    """The one clock for erasure tokens: minting and the ledger's expiry check both read it."""

    return time.time()


def mint_erasure_token(
    tenant_id: str, record_id: str, content_revision: int, actor_id: str, credential_id: str,
    authority_witness_id: str, reason_code: str = "owner_erasure", intent_id: str = "", expires_at: float = 0.0,
) -> VerifiedErasure:
    return VerifiedErasure(
        tenant_id, record_id, content_revision, actor_id, credential_id, authority_witness_id, reason_code,
        intent_id, expires_at,
        _mac(tenant_id, record_id, content_revision, actor_id, credential_id, authority_witness_id, reason_code,
             intent_id, expires_at),
    )


def verify_erasure_token(token: Any) -> None:
    if not isinstance(token, VerifiedErasure) or not token.mac or not hmac.compare_digest(
        token.mac,
        _mac(token.tenant_id, token.record_id, token.content_revision, token.actor_id, token.credential_id,
             token.authority_witness_id, token.reason_code, token.intent_id, token.expires_at),
    ):
        raise AuthorityDenied("the erasure token is not valid for this erasure")


def is_reserved_reason(reason_code: str) -> bool:
    """Reserved ``owner_erasure*`` reasons in any spelling (case, spacing, dashes, width forms)."""

    folded = unicodedata.normalize("NFKC", reason_code).strip().casefold().replace("-", "_")
    return folded.startswith("owner_erasure")


@dataclass(frozen=True)
class ErasureResult:
    intent_id: str
    created: bool  # False: an identical erasure had already been accepted
    audit_id: str
    overridden_hold_ids: tuple[str, ...]


@dataclass(frozen=True)
class StaleIntent:
    """A PROCESSING intent with no live lease and no receipt progress inside the horizon."""

    tenant_id: str
    intent_id: str
    receipt_version: int  # the history this candidate was read at; expiry refuses if it moved
    unfinished: tuple[tuple[str, str], ...]  # (stage, effect_id) of stages with no applied/skipped/failed outcome


class LedgerPort(abc.ABC):
    """Operations every ledger backend implements."""

    @abc.abstractmethod
    def accept(
        self, intent: Intent, objects: Sequence[ObjectRef] = ()
    ) -> AcceptResult:
        """Atomically commit intent, outbox obligation and object references.

        Same (tenant, actor, operation, key) with the same digest returns the
        original receipt (``created=False``); a different digest raises
        ``IdempotencyConflict``.
        """

    @abc.abstractmethod
    def get_intent(self, tenant_id: str, intent_id: str) -> StoredIntent | None: ...

    @abc.abstractmethod
    def receipts(self, tenant_id: str, intent_id: str) -> tuple[StageReceipt, ...]: ...

    @abc.abstractmethod
    def claim(
        self, owner: str, lease_seconds: float, tenant_id: str | None = None
    ) -> Claim | None:
        """Lease a pending or lease-expired obligation (steal).

        ``tenant_id`` scopes the claim; ``None`` visits tenants one at a time.
        Oldest-first is best effort only, not a correctness property.
        """

    @abc.abstractmethod
    def extend_lease(self, claim: Claim, lease_seconds: float) -> Claim: ...

    @abc.abstractmethod
    def complete(self, claim: Claim) -> None:
        """Mark the obligation done.

        ``StaleLease`` unless the claim is the current lease.  Completing a
        claim that already completed (or whose intent reached a terminal
        status and closed the obligation) is idempotent for the same
        (owner, attempt) and returns normally.
        """

    @abc.abstractmethod
    def release(self, claim: Claim) -> None:
        """Return the obligation to pending for a later attempt."""

    @abc.abstractmethod
    def record_receipt(
        self, receipt: StageReceipt, tenant_id: str, claim: Claim | None = None
    ) -> int:
        """Store stage evidence (idempotent for an identical receipt).

        Returns the per-intent receipt version (contiguous from 1).
        """

    def check_receipt_claimable(self, receipt: StageReceipt, tenant_id: str) -> str | None:
        """``None`` when the intent's receipts plus ``receipt`` are a set the claim can validate, else why not.

        Backends may run this outside their write transaction and remember the result against the receipt
        history; ``record_receipt`` refuses the same sets regardless.
        """

        stored = self.get_intent(tenant_id, receipt.intent_id)
        if stored is None:
            return None
        held = tuple(
            r for r in self.receipts(tenant_id, receipt.intent_id)
            if (r.stage, r.attempt) != (receipt.stage, receipt.attempt)
        )
        if applied_evidence_conflict(held, receipt):
            return "applied receipts of one stage must report the same results"
        return claimability_problem(stored.intent, held + (receipt,))

    @abc.abstractmethod
    def set_status(
        self,
        tenant_id: str,
        intent_id: str,
        target: IntentStatus,
        claim: Claim | None = None,
    ) -> IntentStatus:
        """Validated status write; same-status writes only revalidate."""

    @abc.abstractmethod
    def derive_status(self, tenant_id: str, intent_id: str) -> IntentStatus:
        """The single aggregate the stored receipts support (never writes)."""

    @abc.abstractmethod
    def settle(
        self, tenant_id: str, intent_id: str, claim: Claim | None = None
    ) -> IntentStatus:
        """Write the derived aggregate through the validated ``set_status`` path."""

    @abc.abstractmethod
    def record_effect(self, tenant_id: str, effect_id: str, intent_id: str) -> bool:
        """Idempotently register a declared effect; True only the first time."""

    @abc.abstractmethod
    def effect_recorded(self, tenant_id: str, effect_id: str) -> bool: ...

    @abc.abstractmethod
    def scan_unqueued(self, limit: int = 100) -> tuple[OutboxEntry, ...]:
        """Pending obligations never handed to the queue (crash-recovery scan).

        Rows already marked queued are not returned: if a queue message is lost
        after ``mark_queued``, recovery is by dispatchers polling ``claim``.
        """

    @abc.abstractmethod
    def mark_queued(self, tenant_id: str, outbox_id: int) -> None: ...

    @abc.abstractmethod
    def take_unqueued(self, tenant_id: str, outbox_id: int, attempt: int) -> bool:
        """Mark a pending obligation queued; True only for the one caller that changed it.

        Concurrent dispatchers that scanned the same row race here, and only the
        winner enqueues a wake-up, so each hand-off is sent once.  ``attempt`` is
        the one the scan returned: a row retried since then is a new hand-off.
        """

    @abc.abstractmethod
    def unmark_queued(self, tenant_id: str, outbox_id: int, attempt: int) -> None:
        """Return a taken obligation to the scan after its enqueue failed.

        Only the hand-off of ``attempt`` is returned: a late give-back after the
        row was claimed, released and handed off again leaves the newer one alone.
        """

    @abc.abstractmethod
    def put_object_ref(self, ref: ObjectRef) -> bool:
        """Immutable reference; identical re-put is a no-op (False)."""

    @abc.abstractmethod
    def get_object_ref(
        self, tenant_id: str, record_id: str, content_revision: int
    ) -> ObjectRef | None: ...

    @abc.abstractmethod
    def tombstone(
        self, tenant_id: str, record_id: str, content_revision: int, reason_code: str
    ) -> bool:
        """Record a non-content resurrection fence; idempotent (False if present).

        Raises ``HoldActive`` while an active retention hold exists; only
        ``erase_record`` (owner override) may fence a held record.
        """

    @abc.abstractmethod
    def get_tombstone(self, tenant_id: str, record_id: str) -> Tombstone | None: ...

    @abc.abstractmethod
    def has_object_refs(self, tenant_id: str, record_id: str) -> bool:
        """Whether any revision of the record has a stored reference (never content)."""

    @abc.abstractmethod
    def place_hold(
        self, tenant_id: str, hold_id: str, record_id: str, reason_code: str,
        holder_actor_id: str, holder_credential_id: str,
    ) -> bool:
        """Create a retention hold (idempotent for identical arguments)."""

    @abc.abstractmethod
    def release_hold(
        self, tenant_id: str, hold_id: str, released_by: str, release_reason: str
    ) -> bool:
        """Release an active hold; False if it was already released.

        The ledger records who released it but does not authorize the release: a hold is exactly as
        strong as the policy check in front of this call (the policy layer decides who may place or lift one).
        """

    @abc.abstractmethod
    def has_active_holds(self, tenant_id: str) -> bool:
        """Whether the tenant has any active retention hold (for filter-based deletes)."""

    @abc.abstractmethod
    def get_hold(self, tenant_id: str, hold_id: str) -> Hold | None: ...

    @abc.abstractmethod
    def active_holds(self, tenant_id: str, record_id: str) -> tuple[Hold, ...]: ...

    @abc.abstractmethod
    def erase_record(self, token: VerifiedErasure, intent: Intent) -> ErasureResult:
        """Owner-override erasure in one transaction.

        Accepts the ``memory.erase`` intent, appends the content-free audit
        record (naming every overridden hold), writes the tombstone and removes
        the record's references, releases its holds and records the canonical
        stage as applied.  Projection stages remain for the dispatcher.
        """

    @abc.abstractmethod
    def audit_records(self, tenant_id: str, record_id: str | None = None) -> tuple[ErasureAudit, ...]: ...

    # ---- retention primitives (see ``ledger.retention`` for policy) ----------------------------
    @abc.abstractmethod
    def retention_candidates(
        self, tenant_id: str, *, stage: str, older_than_seconds: float, limit: int
    ) -> tuple[str, ...]:
        """Terminal intents past the window for ``stage`` (``"idempotency"`` or ``"content"``).

        Windows count from the intent's terminal time on the ledger clock.  Intents whose
        records carry an active hold, and intents already retired for the stage, are never listed.
        """

    @abc.abstractmethod
    def apply_retention(
        self, tenant_id: str, *, stage: str, intent_ids: Sequence[str], older_than_seconds: float
    ) -> tuple[str, ...]:
        """Retire the listed intents for ``stage``; returns the ids that actually changed.

        Eligibility is re-checked inside the transaction, and a rewrite that would leave the
        intent's evidence invalid is skipped, so the ledger always still passes snapshot validation.
        ``idempotency`` frees the client key (a later use is a new key) and neutralizes the request
        digest; ``content`` also drops reference lists and free text from the intent and its receipts.
        """

    @abc.abstractmethod
    def retired_intent_ids(self, tenant_id: str, *, after_intent_id: str, limit: int) -> tuple[str, ...]:
        """Ids of intents whose content retention compacted, in id order after ``after_intent_id``."""

    @abc.abstractmethod
    def stale_processing(
        self, tenant_id: str, *, older_than_seconds: float, limit: int, after_intent_id: str = ""
    ) -> tuple["StaleIntent", ...]:
        """PROCESSING intents with no live lease and no receipt progress inside the horizon (read only).

        Ordered by intent id; ``after_intent_id`` continues past the last one returned, so a caller can
        page through every stale intent instead of seeing the same first ``limit`` each time.
        """

    @abc.abstractmethod
    def expire_stale(
        self, tenant_id: str, intent_id: str, *, older_than_seconds: float, expected_receipt_version: int
    ) -> IntentStatus:
        """Settle a stale intent by recording ``stage_expired`` failure receipts for its unfinished stages.

        Refuses (``LedgerError``) when the intent is no longer stale or its receipt history moved
        since it was read.  Never writes an applied receipt; the aggregate comes from the same
        validator every settle uses.
        """

    @abc.abstractmethod
    def prune_tombstones(self, tenant_id: str, *, older_than_seconds: float, limit: int) -> int:
        """Delete aged tombstones, except audited owner-erasure fences and records under an active hold."""

    @abc.abstractmethod
    def snapshot(self) -> dict[str, Any]:
        """Transactionally consistent, backend-neutral logical snapshot."""

    @abc.abstractmethod
    def restore(self, snapshot: dict[str, Any], *, expected_digest: str | None = None) -> None:
        """Load a snapshot into an empty ledger; leases reset, tombstones kept.

        Verifies counts, the content digest (and ``expected_digest`` when given,
        for authenticity from a trusted channel), row/body agreement, terminal
        evidence and obligation coherence before writing anything.
        """

    @abc.abstractmethod
    def schema_version(self) -> int: ...

    @abc.abstractmethod
    def check_durability(self) -> None:
        """Raise ``DurabilityError`` unless the required settings are in force."""

    @abc.abstractmethod
    def close(self) -> None: ...


# ---------------------------------------------------------------- shared logic


def checked_intent(intent: Intent) -> Intent:
    """Revalidate a caller-supplied intent before it is stored; refuse one too large to ever be claimed."""

    if not isinstance(intent, Intent):
        raise TypeError("intent must be an Intent")
    checked = Intent.model_validate_json(intent.model_dump_json())
    problem = intent_size_problem(checked)
    if problem is not None:
        raise IntentTooLarge(problem)
    return checked


def checked_object_refs(objects: Sequence[ObjectRef], tenant_id: str) -> tuple[ObjectRef, ...]:
    out = []
    for ref in objects:
        if not isinstance(ref, ObjectRef):
            raise TypeError("objects must be ObjectRef values")
        if ref.tenant_id != tenant_id:
            raise ReceiptMismatch("object reference tenant must match the intent tenant")
        if ref.content_revision < 0 or ref.policy_epoch < 0:
            raise ValueError("revision and epoch must be non-negative")
        for handle in ref.source_refs:
            ReferenceHandle.model_validate(handle)
        out.append(ref)
    return tuple(out)


def checked_receipt(receipt: StageReceipt, intent: Intent) -> StageReceipt:
    receipt = StageReceipt.model_validate_json(receipt.model_dump_json())
    if receipt.intent_id != intent.id:
        raise ReceiptMismatch("receipt belongs to a different intent")
    requirements = {item.stage: item for item in intent.stage_requirements}
    requirement = requirements.get(receipt.stage)
    if requirement is None:
        raise ReceiptMismatch("receipt stage is not a declared required stage")
    if receipt.effect_id != requirement.effect_id:
        raise ReceiptMismatch("receipt effect_id must match the declared effect")
    if any(ref.tenant_id != intent.tenant_id for ref in receipt.output_refs):
        raise ReceiptMismatch("receipt output_refs must match the intent tenant")
    return receipt


def declared_effect_ids(intent: Intent) -> frozenset[str]:
    return frozenset(item.effect_id for item in intent.stage_requirements)


def checked_status_write(
    current: IntentStatus,
    target: IntentStatus,
    intent: Intent,
    receipts: tuple[StageReceipt, ...],
) -> None:
    """Raise ``IllegalTransition`` unless the write is legal and evidenced."""

    if current is not target and not is_legal_intent_transition(current, target):
        raise IllegalTransition(f"illegal transition {current.value} -> {target.value}")
    try:
        validate_required_stage_claim(
            claimed_status=target, intent=intent, receipts=receipts
        )
    except ValueError as exc:
        raise IllegalTransition(str(exc)) from exc


def request_digest_matches(stored: str, supplied: str) -> bool:
    return stored == supplied


# Settled outcomes first: the claim validator admits exactly one aggregate for any receipt set (a contract
# property the lifecycle suite pins), so the first legal candidate is the answer and the common outcomes are
# found with one validation instead of one per status.
_AGGREGATE_ORDER = (
    IntentStatus.APPLIED,
    IntentStatus.PARTIAL,
    IntentStatus.FAILED,
    IntentStatus.PROCESSING,
    IntentStatus.ACCEPTED,
    IntentStatus.CANCELLED,
    IntentStatus.SUPERSEDED,
)


def derive_aggregate(intent: Intent, receipts: tuple[StageReceipt, ...]) -> IntentStatus:
    """The one aggregate the receipts support, via the public claim validator."""

    for candidate in _AGGREGATE_ORDER:
        try:
            validate_required_stage_claim(
                claimed_status=candidate, intent=intent, receipts=receipts
            )
        except ValueError:
            continue
        return candidate
    raise LedgerError("receipts support 0 aggregates: []")


def _claim_fast_path_refs() -> int:
    """Total output references below which a receipt set cannot reach the claim's budget.

    Derived from the lifecycle tier with a fourfold margin over the measured cost of a reference handle
    (about 12 edges and under 2 nodes each, capture plus reconstruction), so it follows the tier and leaves
    room for the receipts' own fields.
    """

    from contracts_lifecycle import LIFECYCLE_TIER

    return min(LIFECYCLE_TIER.max_nodes // 8, LIFECYCLE_TIER.max_edges // 48)


def _claim_items(intent: Intent, receipts: tuple[StageReceipt, ...]) -> int:
    """Reference handles and source versions the claim walks: the intent's own and every receipt's."""

    own = len(intent.source_preconditions) + len(intent.target_refs) + sum(
        len(r.committed_sources) + len(r.precondition_source_ids) for r in intent.stage_requirements
    )
    return own + sum(
        len(r.output_refs) + len(r.applied_sources) + len(r.committed_sources) + len(r.upstream_applications)
        for r in receipts
    )


def claimability_problem(intent: Intent, receipts: tuple[StageReceipt, ...]) -> str | None:
    """``None`` when the claim validator can walk this receipt set, else the reason.

    Sets far below the budget skip the walk: they cannot reach it, and their identity checks are made by
    ``checked_receipt`` on the way in.  Everything the claim walks counts towards that, sources as well as
    output references, so a large intent is walked even when its outputs are few.  Larger sets run the
    claim's own budgeted validation.
    """

    if _claim_items(intent, receipts) <= _claim_fast_path_refs():
        return None
    from contracts_lifecycle import validate_stage_receipts

    try:
        validate_stage_receipts(intent, receipts)
    except (ValueError, TypeError) as exc:
        return str(exc)
    return None


def expected_inputs(intent: Intent, requirement: StageEvidenceRequirement) -> tuple[SourceVersion, ...]:
    """The sources an applied receipt of ``requirement`` must report: its preconditions and upstream commits."""

    by_stage = {item.stage: item for item in intent.stage_requirements}
    wanted = set(requirement.precondition_source_ids)
    selected = [item for item in intent.source_preconditions if item.record_id in wanted]
    for upstream in requirement.upstream_stages:
        selected.extend(by_stage[upstream].committed_sources)
    return tuple(selected)


def full_run_receipts(intent: Intent) -> tuple[StageReceipt, ...]:
    """The receipts of a run in which every stage applied once, with its minimum outputs.

    No other run of the intent carries more sources, so if the claim can walk these, the intent's own size
    never stops it from settling (outputs beyond the minimum are checked as each receipt is recorded).
    """

    by_stage = {item.stage: item for item in intent.stage_requirements}
    now = intent.accepted_at
    return tuple(
        StageReceipt(
            schema_version="candidate-v1", intent_id=intent.id, attempt=1, stage=requirement.stage,
            effect_id=requirement.effect_id, status=StageStatus.APPLIED,
            applied_sources=expected_inputs(intent, requirement),
            committed_sources=requirement.committed_sources,
            upstream_applications=tuple(
                UpstreamApplication(intent_id=intent.id, stage=stage, attempt=1, effect_id=by_stage[stage].effect_id)
                for stage in requirement.upstream_stages
            ),
            output_refs=tuple(
                ReferenceHandle(kind="artifact", id=f"size-{requirement.stage.value}-{n}",
                                tenant_id=intent.tenant_id, resolver="ledger")
                for n in range(requirement.minimum_output_refs)
            ),
            error=None, started_at=now, finished_at=now,
        )
        for requirement in intent.stage_requirements
    )


def intent_size_problem(intent: Intent) -> str | None:
    """``None`` unless the intent is too large for the claim to settle once its stages applied.

    Ordinary intents are far below the budget and are not walked.  A large one is walked as a full run;
    only the budget refusal counts, so a stricter future rule elsewhere in the claim never blocks an accept.
    """

    try:
        receipts = full_run_receipts(intent)
    except (ValueError, TypeError):
        return None  # not a shape this check can model; the record-time check still guards every receipt
    if _claim_items(intent, receipts) <= _claim_fast_path_refs():
        return None
    try:
        validate_required_stage_claim(claimed_status=IntentStatus.APPLIED, intent=intent, receipts=receipts)
    except (ValueError, TypeError) as exc:
        if str(exc) == DISCOVERY_BUDGET_MESSAGE:
            return "the intent is too large for its receipts to be claimed"
    return None


def applied_evidence_conflict(held: tuple[StageReceipt, ...], receipt: StageReceipt) -> bool:
    """Whether ``receipt`` is a second applied receipt for its stage with different evidence.

    The claim requires every applied receipt of one stage to report the same results; a conflicting pair
    would be stored and then refused when the intent is claimed.  "The same" is the claim's own comparison:
    the sources and the output references as sets, so a retry reporting them in another order agrees.
    """

    if receipt.status is not StageStatus.APPLIED:
        return False
    return any(
        r.stage is receipt.stage and r.status is StageStatus.APPLIED and r.attempt != receipt.attempt
        and not (
            _source_sets_match(r.applied_sources, receipt.applied_sources)
            and _source_sets_match(r.committed_sources, receipt.committed_sources)
            and _reference_sets_match(r.output_refs, receipt.output_refs)
        )
        for r in held
    )


def validate_snapshot(snapshot: Any, *, code_version: int, expected_digest: str | None) -> None:
    """Backend-neutral integrity and consistency checks; raises ``LedgerError``."""

    if not isinstance(snapshot, dict) or snapshot.get("format") != SNAPSHOT_FORMAT:
        raise LedgerError("unsupported snapshot format")
    tables = snapshot.get("tables")
    if not isinstance(tables, dict) or set(tables) != set(SNAPSHOT_TABLES):
        raise LedgerError("snapshot tables do not match the ledger schema")
    if any(not isinstance(rows, list) for rows in tables.values()):
        raise LedgerError("snapshot tables must be lists of rows")
    counts = snapshot.get("counts")
    if counts != {name: len(rows) for name, rows in tables.items()}:
        raise LedgerError("snapshot row counts do not match its tables")
    digest = snapshot.get("digest")
    if not isinstance(digest, str) or not hmac.compare_digest(digest, snapshot_digest(snapshot)):
        raise LedgerError("snapshot integrity check failed (digest mismatch)")
    if expected_digest is not None and not hmac.compare_digest(digest, expected_digest):
        raise LedgerError("snapshot digest does not match the expected digest")
    if snapshot.get("schema_version") != code_version:
        raise LedgerError("snapshot schema_version does not match this ledger")
    for name, rows in tables.items():
        for row in rows:
            if not isinstance(row, dict) or set(row) != set(SNAPSHOT_COLUMNS[name]):
                raise LedgerError("snapshot row columns do not match the ledger schema")

    intents: dict[tuple[str, str], tuple[Intent, IntentStatus]] = {}
    for row in tables["intents"]:
        intent = Intent.model_validate_json(row["body"])
        status = IntentStatus(row["status"])
        for column, value in (
            ("tenant_id", intent.tenant_id), ("id", intent.id), ("actor_id", intent.actor_id),
            ("credential_id", intent.credential_id), ("operation", intent.operation),
            ("idempotency_key", intent.idempotency_key), ("request_digest", intent.request_digest),
        ):
            if row[column] != value:
                raise LedgerError(f"intent row {column} does not match its body")
        key = (row["tenant_id"], row["id"])
        if key in intents:
            raise LedgerError("duplicate intent in snapshot")
        intents[key] = (intent, status)

    receipts: dict[tuple[str, str], list[tuple[int, StageReceipt]]] = {}
    for row in tables["stage_receipts"]:
        key = (row["tenant_id"], row["intent_id"])
        if key not in intents:
            raise LedgerError("receipt references an unknown intent")
        receipt = checked_receipt(StageReceipt.model_validate_json(row["body"]), intents[key][0])
        for column, value in (
            ("stage", receipt.stage.value), ("attempt", receipt.attempt),
            ("status", receipt.status.value), ("effect_id", receipt.effect_id),
        ):
            if row[column] != value:
                raise LedgerError(f"receipt row {column} does not match its body")
        receipts.setdefault(key, []).append((row["version"], receipt))

    for key, (intent, status) in intents.items():
        if status in TERMINAL_STATUSES:  # terminal intents are immutable, so their evidence must hold
            ordered = tuple(r for _, r in sorted(receipts.get(key, []), key=lambda item: item[0]))
            try:
                validate_required_stage_claim(claimed_status=status, intent=intent, receipts=ordered)
            except ValueError as exc:
                raise LedgerError(f"restored {status.value} intent lacks supporting evidence: {exc}") from exc

    seen: set[tuple[str, str]] = set()
    for row in tables["outbox"]:
        key = (row["tenant_id"], row["intent_id"])
        if key not in intents or key in seen:
            raise LedgerError("outbox row must reference exactly one intent of the same tenant")
        seen.add(key)
        status = intents[key][1]
        if row["state"] not in {"pending", "leased", "done", "fenced"}:
            raise LedgerError("unknown outbox state")
        if status in TERMINAL_STATUSES and row["state"] not in {"done", "fenced"}:
            raise LedgerError("terminal intent holds an open obligation")
        if row["state"] == "fenced" and status not in FENCING_STATUSES:
            raise LedgerError("fenced obligation on a non-fenced intent")
    if seen != set(intents):
        raise LedgerError("every intent needs exactly one outbox obligation")

    for row in tables["effects"]:
        key = (row["tenant_id"], row["intent_id"])
        if key not in intents or row["effect_id"] not in declared_effect_ids(intents[key][0]):
            raise LedgerError("effect is not declared by its intent")

    fenced = {(r["tenant_id"], r["record_id"]) for r in tables["tombstones"]}
    if any((r["tenant_id"], r["record_id"]) in fenced for r in tables["object_refs"]):
        raise LedgerError("snapshot holds object references for tombstoned records")

    # Erasure coherence: an audited or erase-intent record must still be fenced,
    # and an owner-erasure fence must still be audited.
    audited: set[tuple[str, str]] = set()
    for row in tables["erasure_audit"]:
        key = (row["tenant_id"], row["intent_id"])
        if key not in intents or intents[key][0].operation != "memory.erase":
            raise LedgerError("erasure audit must reference a memory.erase intent")
        if (row["tenant_id"], row["record_id"]) not in fenced:
            raise LedgerError("erased record lost its tombstone")
        if not isinstance(json.loads(row["overridden_hold_ids"]), list):
            raise LedgerError("erasure audit hold list is malformed")
        audited.add((row["tenant_id"], row["record_id"]))
    for (tenant, _), (intent, _status) in intents.items():
        if intent.operation == "memory.erase":
            record = intent.source_preconditions[0].record_id if intent.source_preconditions else None
            if (tenant, record) not in fenced:
                raise LedgerError("memory.erase intent lost its tombstone")
    for row in tables["tombstones"]:
        if row["reason_code"].startswith("owner_erasure") and (row["tenant_id"], row["record_id"]) not in audited:
            raise LedgerError("owner-erasure tombstone has no audit record")
    for row in tables["retention_holds"]:
        if row["released_at"] is None and (row["tenant_id"], row["record_id"]) in fenced:
            raise LedgerError("active hold on an erased record")


def snapshot_digest(snapshot: dict[str, Any]) -> str:
    """SHA-256 over the canonical snapshot content (format, version, counts, rows).

    Detects truncation and accidental or unsynchronized edits.  It is an
    integrity check, not authentication: anyone able to rewrite a snapshot can
    recompute it, so pass ``expected_digest`` from a trusted channel on restore
    when authenticity matters.
    """

    content = {
        "format": snapshot.get("format"),
        "schema_version": snapshot.get("schema_version"),
        "counts": snapshot.get("counts"),
        "tables": snapshot.get("tables"),
    }
    canonical = json.dumps(content, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


# ------------------------------------------------------------------ migrations

MIGRATIONS_DIR = Path(__file__).parent / "migrations"
_FILE_RE = re.compile(r"^(\d{4})_([a-z0-9_]+)\.sql$")
# Expand-only: later migrations may add objects, never remove or rewrite them.
_DESTRUCTIVE_RE = re.compile(
    r"\b(drop\s+\w+|rename\s+(to|column)|"
    r"alter\s+\w+\s+[\w\.\"]+\s+(drop|rename|alter\s+column)|delete\s+from)\b",
    re.IGNORECASE,
)
# ``TRUNCATE`` as a statement (not the ``BEFORE TRUNCATE`` trigger event)
_TRUNCATE_STATEMENT_RE = re.compile(r"(?:^|;)\s*truncate\b", re.IGNORECASE | re.MULTILINE)


@dataclass(frozen=True)
class Migration:
    version: int
    name: str
    sql: str
    checksum: str


def load_migrations(dialect: str) -> tuple[Migration, ...]:
    if dialect not in {"sqlite", "postgres"}:
        raise ValueError("unknown dialect")
    found: list[Migration] = []
    for path in sorted((MIGRATIONS_DIR / dialect).glob("*.sql")):
        match = _FILE_RE.match(path.name)
        if match is None:
            raise SchemaError(f"bad migration file name: {path.name}")
        sql = path.read_text(encoding="utf-8")
        found.append(
            Migration(
                int(match.group(1)),
                match.group(2),
                sql,
                hashlib.sha256(sql.encode("utf-8")).hexdigest(),
            )
        )
    versions = [item.version for item in found]
    if versions != list(range(1, len(versions) + 1)):
        raise SchemaError("migration versions must be contiguous from 0001")
    return tuple(found)


def assert_expand_only(migration: Migration) -> None:
    # Comments become a space so ``DROP/**/TABLE`` cannot hide a statement.
    stripped = re.sub(r"/\*.*?\*/", " ", migration.sql, flags=re.DOTALL)
    stripped = re.sub(r"--[^\n]*", " ", stripped)
    # Lint only, not a control: it cannot see the live schema or dynamic SQL.
    hit = _DESTRUCTIVE_RE.search(stripped) or _TRUNCATE_STATEMENT_RE.search(stripped)
    if hit:
        raise SchemaError(f"migration {migration.version} is not expand-only: {hit.group(0).strip(' ;')}")


def plan_migrations(
    dialect: str, applied: dict[int, str]
) -> tuple[Migration, ...]:
    """Return pending migrations; reject drift and schemas newer than this code."""

    available = load_migrations(dialect)
    known = {item.version: item for item in available}
    for version, checksum in applied.items():
        if version not in known:
            raise SchemaError(f"database schema version {version} is newer than this code")
        if known[version].checksum != checksum:
            raise SchemaError(f"applied migration {version} was modified")
    pending = tuple(item for item in available if item.version not in applied)
    for item in pending:
        assert_expand_only(item)
    return pending
