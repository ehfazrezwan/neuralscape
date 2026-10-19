"""Holds, owner-override erasure and erased-reference resolution.

The ledger stores holds, tombstones and the audit trail but does not
authenticate anyone.  Authority comes from an ``AuthorityVerifier`` supplied
by the policy layer (the durable authority store); ``ledger.erasure`` accepts
only a protocol, so it needs no import of the policy modules.

Flow for an owner-override erasure:

1. the verifier confirms, from the verified credential, that the principal is
   the owner of the tenant's record (fail closed on any error or ``False``);
2. ``ledger.erase_record`` commits, in one transaction, the ``memory.erase``
   intent, the content-free audit record naming every overridden hold, the
   tombstone (references removed, fence kept), released holds, and the applied
   canonical stage;
3. a dispatcher then runs the remaining projection stages (vector, graph,
   markdown) from the outbox obligation, so the intent reaches ``APPLIED``
   only after every projection receipt exists.

Backup limit (stated, not enforced): bodies inside backups stay recoverable
until those backups expire; the retention window ``N`` is a deployment choice.
"""

from __future__ import annotations

import hmac
import math
import weakref
from dataclasses import dataclass
from typing import Any, Callable, Protocol, TypeVar

from contracts_lifecycle import Intent
from contracts_policy import PrincipalContext
from contracts_references import ReferenceHandle, SourceVersion
from ledger.port import (
    AuthorityDenied,
    ErasureResult,
    HoldActive,
    LedgerPort,
    VerifiedErasure,
    mint_erasure_token,
)
from ledger import port as _port
from ledger.stages import build_intent

T = TypeVar("T")
DEFAULT_TOKEN_TTL_SECONDS = 30.0
MAX_TOKEN_TTL_SECONDS = 300.0


_PROOF = object()  # held only by this module


_ISSUED: dict[int, "weakref.ref[PrincipalContext]"] = {}


def mark_verified_principal(principal: PrincipalContext) -> PrincipalContext:
    """Record that the request layer verified ``principal`` for this request; returns it.

    Only the enforcement layer, at the point where it has authenticated the request, should call
    this.  Membership is by object identity (an equal copy is not issued) and lapses with the
    object.  It is a deliberate, greppable act that a body-built principal never performs; it is not
    a defense against code that chooses to call it.
    """

    if not isinstance(principal, PrincipalContext):
        raise TypeError("only a PrincipalContext can be marked as verified")
    key = id(principal)
    _ISSUED[key] = weakref.ref(principal, lambda _ref, key=key: _ISSUED.pop(key, None))
    return principal


def _is_issued(principal: object) -> bool:
    ref = _ISSUED.get(id(principal))
    return ref is not None and ref() is principal


class ErasureAuthority:
    """Who is asking to erase: built from a principal the request layer issued, read-only, not copyable.

    The ids are all derivable from public state, so a value assembled from a body, header or query
    proves nothing.  :meth:`from_principal` therefore accepts only a ``PrincipalContext`` that the
    enforcement layer marked with :func:`mark_verified_principal`, and the result cannot be
    re-aimed: it is not a dataclass (no ``replace``), its fields are read-only, and ``copy``,
    ``deepcopy`` and pickling raise.

    What this is and is not: it makes the intended path the easy one and turns the obvious misuses
    (strings, a hand-built principal, ``replace``, copying a genuine value) into errors.  It is not a
    security boundary against code that can call :func:`mark_verified_principal` or read this
    module's private state.  The security rests on the request layer being the only caller that
    issues principals and on the commit being re-verified under a held authority (see ``hold``).
    ``witness_id`` (``membership_version:policy_epoch``) proves currency, not identity.
    """

    __slots__ = ("_tenant_id", "_principal_id", "_credential_id", "_witness_id")

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("ErasureAuthority cannot be subclassed")

    def __init__(
        self, tenant_id: str, principal_id: str, credential_id: str, witness_id: str, _proof: object = None
    ) -> None:
        if _proof is not _PROOF:
            raise TypeError("an ErasureAuthority is built from an issued principal: use ErasureAuthority.from_principal")
        for name, value in (("_tenant_id", tenant_id), ("_principal_id", principal_id),
                            ("_credential_id", credential_id), ("_witness_id", witness_id)):
            object.__setattr__(self, name, value)

    tenant_id = property(lambda self: self._tenant_id)
    principal_id = property(lambda self: self._principal_id)
    credential_id = property(lambda self: self._credential_id)
    witness_id = property(lambda self: self._witness_id)

    @classmethod
    def from_principal(cls, principal: PrincipalContext, *, policy_epoch: int) -> "ErasureAuthority":
        if not isinstance(principal, PrincipalContext):
            raise TypeError("from_principal needs the verified PrincipalContext, not identifiers")
        if not _is_issued(principal):
            raise TypeError("from_principal needs a principal the enforcement layer issued for this request")
        if isinstance(policy_epoch, bool) or not isinstance(policy_epoch, int) or policy_epoch < 0:
            raise TypeError("policy_epoch must be the non-negative epoch the request was admitted under")
        return cls(
            principal.tenant_id, principal.subject_id, principal.credential_id,
            f"{principal.membership_version}:{policy_epoch}", _PROOF,
        )

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("an ErasureAuthority is read-only")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("an ErasureAuthority is read-only")

    def __copy__(self) -> "ErasureAuthority":
        raise TypeError("an ErasureAuthority cannot be copied")

    def __deepcopy__(self, memo: object) -> "ErasureAuthority":
        raise TypeError("an ErasureAuthority cannot be copied")

    def __reduce__(self) -> object:
        raise TypeError("an ErasureAuthority cannot be pickled")

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ErasureAuthority) and (
            self._tenant_id, self._principal_id, self._credential_id, self._witness_id
        ) == (other._tenant_id, other._principal_id, other._credential_id, other._witness_id)

    def __hash__(self) -> int:
        return hash((self._tenant_id, self._principal_id, self._credential_id))

    def __repr__(self) -> str:
        return f"<ErasureAuthority tenant={self._tenant_id!r} principal={self._principal_id!r}>"


class AuthorityVerifier(Protocol):
    def verify_owner_erasure(self, authority: ErasureAuthority, *, record_id: str) -> bool:
        """True only if the witness currently proves the principal owns the record."""

    def hold(self, authority: ErasureAuthority, body: Callable[[], T]) -> T:
        """Run ``body`` while the authority is verified unchanged and kept so until it returns.

        Required: a verifier that cannot hold the authority through the commit cannot be used, and
        ``erase_owner_override`` refuses it rather than erasing on a pre-check alone.
        """


class AuthorityStoreVerifier:
    """Verifier backed by the durable authority store (the policy layer's ``AuthorityStore``).

    Every check fails closed.  The principal must be a bound **human** subject of the tenant whose
    current credential is the one presented and active; the witness must equal the principal's
    *current* ``membership_version:policy_epoch`` (so a witness proving an older membership or
    policy is stale and refused); and ``owner_of`` — supplied by the caller from the source of
    truth for record ownership — must name that same subject.  Workers, service accounts,
    delegated agents and asserted subjects never pass.

    **Contract: erasure is subject to policy by default.**  The owner's ``delete`` on the record's
    pool is evaluated against the tenant's current policy snapshot and a deny wins, exactly as for
    an ordinary delete; the caller supplies ``resource_of`` (record -> pool handle).  A deployment
    that holds erasure to be an owner right independent of administrator policy (for example an
    erasure right that survives a suspended tenant) must say so explicitly with
    ``policy_independent=True``; neither is a silent default.  Holds are overridden either way.

    The checks read the store in several transactions, so they are a pre-check only: the commit runs
    through :meth:`hold`, which re-verifies the witnessed authority and keeps it locked while the
    ledger commit happens (the same pattern as the publication fence).
    """

    def __init__(
        self,
        authority_store: Any,
        owner_of: Callable[[str, str], str | None] | None,
        resource_of: Callable[[str, str], ReferenceHandle | None] | None = None,
        *,
        policy_independent: bool = False,
    ) -> None:
        if not policy_independent and resource_of is None:
            raise ValueError("erasure is subject to policy: supply resource_of, or pass policy_independent=True")
        self._store = authority_store
        self._owner_of = owner_of
        self._resource_of = resource_of
        self._independent = policy_independent

    def verify_owner_erasure(self, authority: ErasureAuthority, *, record_id: str) -> bool:
        if self._owner_of is None:
            return False
        record = self._store.subject_record(authority.principal_id)
        if (
            record is None
            or record.tenant_id != authority.tenant_id
            or record.subject_kind != "human"
            or record.credential_id != authority.credential_id
            or self._store.credential_state(authority.credential_id) != "active"
        ):
            return False
        policy = self._store.current_policy(authority.tenant_id)
        expected = f"{record.membership_version}:{policy.policy_epoch if policy else 0}"
        if not hmac.compare_digest(expected, authority.witness_id):
            return False
        owner = self._owner_of(authority.tenant_id, record_id)
        if owner is None or owner != authority.principal_id:
            return False
        return self._independent or self._policy_allows_delete(record, authority, policy, record_id)

    def _policy_allows_delete(self, record: Any, authority: ErasureAuthority, policy: Any, record_id: str) -> bool:
        if policy is None or self._resource_of is None:
            return False
        resource = self._resource_of(authority.tenant_id, record_id)
        if resource is None:
            return False
        from contracts_policy import PolicyEvaluationInput
        from contracts_policy_reference import evaluate_policy
        from policy_enforcement import _build_principal

        # Only this principal's statements can grant or deny its delete; handing the evaluator the
        # whole tenant snapshot would make the owner's erasure right fail once a tenant is large.
        mine = tuple(item for item in policy.statements if item.subject_id == authority.principal_id)
        sliced = policy.model_copy(update={
            "statements": mine,
            "membership_versions": tuple(
                m for m in policy.membership_versions if m.subject_id == authority.principal_id
            ),
        })
        result = evaluate_policy(
            principal=_build_principal(authority.principal_id, record),
            evaluation=PolicyEvaluationInput(schema_version="candidate-v1", action="delete", resource=resource),
            policy=sliced,
        )
        return result.outcome == "allow"

    def hold(self, authority: ErasureAuthority, body: Callable[[], T]) -> T:
        """Run ``body`` while the witnessed authority is verified unchanged and locked.

        The witness names the membership version and policy epoch; the store re-checks them, the
        subject and the credential inside one transaction and keeps those rows locked until
        ``body`` returns, so a revocation either lands before (the erasure is refused) or waits.
        """

        from authority_store_sql import WitnessStale

        try:
            membership, epoch = (int(part) for part in authority.witness_id.split(":", 1))
        except ValueError as exc:
            raise AuthorityDenied("the authority witness is malformed") from exc
        try:
            # Only the authority checks are translated; whatever ``body`` raises passes through unchanged.
            return self._store.hold_current(
                tenant_id=authority.tenant_id, subject_id=authority.principal_id,
                credential_id=authority.credential_id, membership_version=membership, policy_epoch=epoch,
                body=lambda _connection=None: body(),
            )
        except WitnessStale as exc:
            raise AuthorityDenied("owner authority changed before the erasure committed") from exc


def issue_erasure_token(
    verifier: AuthorityVerifier,
    authority: ErasureAuthority,
    *,
    record_id: str,
    content_revision: int,
    intent_id: str,
    ttl_seconds: float = DEFAULT_TOKEN_TTL_SECONDS,
    now: Callable[[], float] | None = None,
) -> VerifiedErasure:
    """A token for exactly one erasure intent, valid for a short time (at most ``MAX_TOKEN_TTL_SECONDS``).

    Minting and the ledger read the same clock (``ledger.port.erasure_now``, looked up at call time) unless ``now`` is given.
    """

    if (
        isinstance(ttl_seconds, bool) or not isinstance(ttl_seconds, (int, float))
        or not math.isfinite(ttl_seconds) or not 0 < ttl_seconds <= MAX_TOKEN_TTL_SECONDS
    ):
        raise ValueError(f"token lifetime must be positive and at most {MAX_TOKEN_TTL_SECONDS:g} seconds")
    issued_at = (now or _port.erasure_now)()
    if not isinstance(issued_at, (int, float)) or not math.isfinite(issued_at):
        raise ValueError("the clock must return a finite time")

    if type(authority) is not ErasureAuthority:  # exact type: no subclass or look-alike
        raise AuthorityDenied("authority must come from a verified principal")
    for value in (authority.tenant_id, authority.principal_id, authority.credential_id, authority.witness_id,
                  record_id, intent_id):
        if not isinstance(value, str) or not value:
            raise AuthorityDenied("authority and record identifiers are required")
    try:
        allowed = verifier.verify_owner_erasure(authority, record_id=record_id)
    except Exception as exc:  # noqa: BLE001  # fail closed, never leak verifier internals
        raise AuthorityDenied("owner authority could not be verified") from exc
    if allowed is not True:
        raise AuthorityDenied("owner authority was not established")
    return mint_erasure_token(
        authority.tenant_id, record_id, content_revision, authority.principal_id, authority.credential_id,
        authority.witness_id, intent_id=intent_id, expires_at=issued_at + ttl_seconds,
    )


def build_erase_intent(
    authority: ErasureAuthority,
    *,
    record: SourceVersion,
    intent_id: str,
    idempotency_key: str,
    request_digest: str,
    correlation_id: str,
) -> Intent:
    """The ``memory.erase`` intent (frozen stage set: tombstone + every projection)."""

    return build_intent(
        "memory.erase", intent_id=intent_id, tenant_id=authority.tenant_id, actor_id=authority.principal_id,
        credential_id=authority.credential_id, idempotency_key=idempotency_key,
        request_digest=request_digest, correlation_id=correlation_id,
        preconditions=(record,), committed=(),
    )


def erase_owner_override(
    ledger: LedgerPort,
    verifier: AuthorityVerifier,
    authority: ErasureAuthority,
    *,
    record: SourceVersion,
    intent_id: str,
    idempotency_key: str,
    request_digest: str,
    correlation_id: str,
) -> ErasureResult:
    """Erase a record even under retention holds; requires verified owner authority."""

    token = issue_erasure_token(
        verifier, authority, record_id=record.record_id, content_revision=record.content_revision,
        intent_id=intent_id,
    )
    intent = build_erase_intent(
        authority, record=record, intent_id=intent_id, idempotency_key=idempotency_key,
        request_digest=request_digest, correlation_id=correlation_id,
    )
    hold = getattr(verifier, "hold", None)
    if hold is None:  # a pre-check alone is not enough: fail closed
        raise AuthorityDenied("the verifier cannot hold the authority through the commit")
    return hold(authority, lambda: ledger.erase_record(token, intent))


def guard_ordinary_delete(ledger: LedgerPort, tenant_id: str, record_id: str) -> None:
    """Early, explicit refusal of an ordinary delete under a hold.

    ``LedgerPort.tombstone`` repeats the check inside its transaction, so a hold
    placed after this call still blocks the fence.
    """

    holds = ledger.active_holds(tenant_id, record_id)
    if holds:
        raise HoldActive(record_id, tuple(h.hold_id for h in holds))


@dataclass(frozen=True)
class Resolution:
    """Outcome of resolving a (possibly cross-tenant) reference; never carries content."""

    state: str  # "denied" | "erased" | "present" | "unknown"
    handle: ReferenceHandle


def resolve_reference(
    ledger: LedgerPort,
    handle: ReferenceHandle,
    authorize: Callable[[ReferenceHandle], bool],
) -> Resolution:
    """Resolve a memory reference without returning content.

    ``authorize`` is mandatory and evaluated on **every** resolution, same-tenant
    or cross-tenant (confirmed ``lineage=authorized_cross``); an unauthorized
    caller learns nothing, not even whether the record was erased.  An erased
    record resolves to ``erased``, never to its content.
    """

    try:
        allowed = authorize(handle)
    except Exception:  # noqa: BLE001
        allowed = False
    if allowed is not True:
        return Resolution("denied", handle)
    if handle.kind != "memory":
        return Resolution("unknown", handle)
    if ledger.get_tombstone(handle.tenant_id, handle.id) is not None:
        return Resolution("erased", handle)
    if ledger.has_object_refs(handle.tenant_id, handle.id):
        return Resolution("present", handle)
    return Resolution("unknown", handle)
