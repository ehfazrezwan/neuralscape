"""Publication fence: re-resolve current authority and source at publication.

LIBRARY ONLY IN THIS SLICE: nothing in the worker or the write paths calls this module yet (slices 05 and
07 wire it to the durable write path), so revoking a credential or offboarding a member does not yet stop
work that is already queued.

Admission only records *claims* about the world when a job was accepted (who
asked, under which credential, membership version and policy epoch, and which
source revision and namespace the work derived from).  At publication nothing
is taken from those claims:

1. the subject, credential, membership version and policy are read from the
   authority store now;
2. the source row's current revision, namespace and visibility are read from
   the ``SourceResolver`` now;
3. ``background_derive`` is evaluated for the current principal against the
   pool the source row lives in now, never the pool named by the payload.

``PublicationFence.evaluate`` is an unlocked pre-check, suitable before the job
starts and before every provider call or fallback.  ``PublicationFence.publish``
adds the atomic part.

Atomic check-and-publish contract
---------------------------------
``publish`` calls ``store.hold_current(...)``, which verifies in one database
transaction, holding locks on the rows that authorize the work, that the
witnessed policy epoch, membership version, subject and credential are still
current, and then runs the caller's ``publish_fn(connection)`` before commit.
The caller must therefore put its publication write, and the ledger's
completion of the claim, inside ``publish_fn`` on that connection.  That is
only atomic when the published data and the ledger live in the same database as
the authority store (the SQLite file, or the same PostgreSQL cell database).
When the target is external (a vector or graph store) the authority check is
still atomic with the ledger completion, but the external write is not, so a
reconciliation sweep must retract derivatives published under authority that
was revoked after the last check.  The source row is external in every
deployment, so the source revision check is check-then-act, narrowed by one
re-resolution immediately before the held transaction (never inside it, so a
slow resolver cannot stall every authority read).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple, Protocol, TypeVar

from authority_store import AuthorityStore
from authority_store_sql import WitnessStale
from contracts_policy import PolicyEvaluationInput, PrincipalContext
from contracts_policy_reference import evaluate_policy
from identity_binding import NamespaceKey, encode_namespace
from policy_enforcement import _build_principal, pool_handle

T = TypeVar("T")
ACTION = "background_derive"


class Admission(NamedTuple):
    """Claims recorded at admission.  Never trusted as authority."""

    tenant_id: str
    subject_id: str
    credential_id: str
    membership_version: int
    policy_epoch: int
    source_record_id: str
    source_revision: int
    target_namespace: NamespaceKey


class CurrentSource(NamedTuple):
    """The source row as it is now."""

    revision: int
    namespace: NamespaceKey
    visibility: str


class SourceResolver(Protocol):
    def current(self, tenant_id: str, record_id: str) -> CurrentSource | None:
        """The source row now, or ``None`` when it is gone or tombstoned."""
        ...


class WitnessedAuthority(NamedTuple):
    tenant_id: str
    subject_id: str
    credential_id: str
    membership_version: int
    policy_epoch: int
    source_record_id: str
    source_revision: int
    namespace: str


class FenceDecision(NamedTuple):
    allowed: bool
    reason: str
    witness: WitnessedAuthority | None = None
    epoch_advanced: bool = False


class CredentialStates(Protocol):
    def credential_state(self, credential_id: str) -> str: ...

    def credential_owner(self, credential_id: str) -> str | None: ...

    def credential_scope(self, credential_id: str) -> tuple[str, ...] | None: ...


class FenceStore(AuthorityStore, CredentialStates, Protocol):
    def hold_current(self, **kwargs: Any) -> Any: ...


class PublicationFence:
    def __init__(
        self,
        store: FenceStore,
        sources: SourceResolver,
        *,
        strict_epoch: bool = False,
    ) -> None:
        """``strict_epoch`` additionally denies when the policy epoch advanced
        since admission.  By default the current policy is evaluated instead,
        which is the authoritative decision; strict mode only adds retries."""

        self._store = store
        self._sources = sources
        self._strict_epoch = strict_epoch

    def evaluate(self, admission: Admission) -> FenceDecision:
        """Unlocked pre-check against current authority and current source."""

        # The admitted credential must be one of THIS subject's credentials; a
        # queued payload that names another subject's active credential must not
        # borrow its state.
        owner = self._store.credential_owner(admission.credential_id)
        if owner is not None and owner != admission.subject_id:
            return FenceDecision(False, "credential_subject_mismatch")
        record = self._store.subject_record(admission.subject_id)
        if record is None:
            state = self._store.credential_state(admission.credential_id)
            reason = "credential_retired" if state in {"retired", "expired"} else "subject_not_active"
            return FenceDecision(False, reason)
        if record.tenant_id != admission.tenant_id:
            return FenceDecision(False, "tenant_mismatch")
        state = self._store.credential_state(admission.credential_id)
        if state != "active":
            return FenceDecision(False, "credential_retired")
        # Work admitted under a scoped credential is derived only within its scope; a scope
        # only narrows what the subject itself may do.
        scope_of = getattr(self._store, "credential_scope", None)
        if scope_of is not None:
            scope = scope_of(admission.credential_id)
        else:
            # A store that cannot report scopes: only the subject's own sign-in credential is
            # known to be unscoped; any other credential is treated as scoped to nothing.
            scope = None if admission.credential_id == record.credential_id else ()
        if scope is not None and ACTION not in scope:
            return FenceDecision(False, "outside_credential_scope")
        if record.membership_version != admission.membership_version:
            return FenceDecision(False, "membership_changed")
        policy = self._store.current_policy(admission.tenant_id)
        if policy is None:
            return FenceDecision(False, "no_policy")
        if policy.policy_epoch < admission.policy_epoch:
            return FenceDecision(False, "policy_epoch_regressed")
        advanced = policy.policy_epoch > admission.policy_epoch
        if advanced and self._strict_epoch:
            return FenceDecision(False, "policy_epoch_advanced", epoch_advanced=True)

        source = self._sources.current(admission.tenant_id, admission.source_record_id)
        if source is None:
            return FenceDecision(False, "source_gone", epoch_advanced=advanced)
        if source.revision != admission.source_revision:
            return FenceDecision(False, "source_revision_changed", epoch_advanced=advanced)
        if source.visibility != source.namespace.pool_kind:
            return FenceDecision(False, "visibility_mismatch", epoch_advanced=advanced)
        if source.namespace != admission.target_namespace:
            return FenceDecision(False, "namespace_changed", epoch_advanced=advanced)
        if source.namespace.tenant != admission.tenant_id:
            return FenceDecision(False, "tenant_mismatch", epoch_advanced=advanced)

        principal = _principal(admission, record)
        result = evaluate_policy(
            principal=principal,
            evaluation=PolicyEvaluationInput(
                schema_version="candidate-v1",
                action=ACTION,
                resource=pool_handle(*source.namespace),
            ),
            policy=policy,
        )
        if result.outcome != "allow":
            return FenceDecision(False, result.reason_code, epoch_advanced=advanced)
        witness = WitnessedAuthority(
            admission.tenant_id,
            admission.subject_id,
            admission.credential_id,
            record.membership_version,
            policy.policy_epoch,
            admission.source_record_id,
            source.revision,
            encode_namespace(source.namespace),
        )
        return FenceDecision(True, "explicit_grant", witness, advanced)

    def publish(
        self, admission: Admission, publish_fn: Callable[[Any], T]
    ) -> tuple[FenceDecision, T | None]:
        """Evaluate, then verify-and-publish atomically (see the module docs).

        Returns ``(decision, result)``; ``result`` is ``None`` unless allowed.
        ``publish_fn`` receives the open transaction's connection and must do
        its writes there.  Raising from it rolls everything back.
        """

        decision = self.evaluate(admission)
        if not decision.allowed:
            return decision, None
        witness = decision.witness
        assert witness is not None

        # Re-resolve the source immediately BEFORE the held transaction, so a
        # slow resolver never runs while authority rows are locked.  The window
        # between this read and the commit is the documented check-then-act one.
        again = self._sources.current(witness.tenant_id, witness.source_record_id)
        if (
            again is None
            or again.revision != witness.source_revision
            or encode_namespace(again.namespace) != witness.namespace
        ):
            return FenceDecision(False, "source_changed", witness, decision.epoch_advanced), None

        def guarded(connection: Any) -> T:
            return publish_fn(connection)

        try:
            result = self._store.hold_current(
                tenant_id=witness.tenant_id,
                subject_id=witness.subject_id,
                credential_id=witness.credential_id,
                membership_version=witness.membership_version,
                policy_epoch=witness.policy_epoch,
                body=guarded,
            )
        except WitnessStale as stale:
            return FenceDecision(False, stale.reason, witness, decision.epoch_advanced), None
        return decision, result


def _principal(admission: Admission, record: Any) -> PrincipalContext:
    # The work acts under the credential it was admitted with (a scoped credential, or the
    # subject's own), so credential-specific statements apply to that credential.
    return _build_principal(admission.subject_id, record._replace(credential_id=admission.credential_id))


__all__ = [
    "ACTION",
    "Admission",
    "CurrentSource",
    "FenceDecision",
    "PublicationFence",
    "SourceResolver",
    "WitnessedAuthority",
]
