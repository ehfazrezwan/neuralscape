"""Candidate policy contracts with no implicit identity or role authority.

These models describe already-authenticated evaluation inputs and decisions.
They do not authenticate callers, establish resource existence, or assign any
permissions from a role name.  Policy statements are exact grants or denials;
broader policy languages require a new contract version.
"""

from __future__ import annotations

from typing import Literal

from pydantic import model_validator

from contracts_common import ContractModel, OpaqueId, SafeCounter, VersionedContract
from contracts_references import ReferenceHandle


PolicyOutcome = Literal["allow", "deny"]
PolicyEffect = Literal["allow", "deny"]
SubjectKind = Literal["human", "delegated_agent", "worker"]
PolicyReasonCode = Literal[
    "explicit_grant",
    "not_authorized",
    "unsupported_action",
    "unsupported_policy_semantics",
]

# This is an evaluator vocabulary, not a role matrix.  Every allowed result
# still requires a matching statement, including administrative operations.
SUPPORTED_POLICY_ACTIONS = frozenset(
    {
        "read",
        "search",
        "list",
        "count",
        "create",
        "update",
        "share",
        "publish_standard",
        "manage_membership",
        "manage_credentials",
        "delete",
        "restore",
        "export",
        "background_derive",
        "background_replay",
    }
)


class DelegationConstraints(ContractModel):
    """Explicit ceilings applied after ordinary policy evaluation.

    An empty action or resource collection grants nothing.  Presence in these
    collections is never itself a grant: a matching allow statement is still
    required from the current policy snapshot.
    """

    delegated_by_subject_id: OpaqueId
    delegated_by_credential_id: OpaqueId
    allowed_actions: tuple[OpaqueId, ...]
    allowed_resources: tuple[ReferenceHandle, ...]


class PrincipalContext(VersionedContract):
    """Verified actor context constructed at a trusted service boundary.

    The subject and credential are intentionally separate identities.  The
    membership version is a per-subject authority witness, not a policy epoch.
    Callers must not treat serialized instances as authentication evidence.
    """

    tenant_id: OpaqueId
    subject_id: OpaqueId
    subject_kind: SubjectKind
    credential_id: OpaqueId
    membership_version: SafeCounter
    device_id: OpaqueId | None
    delegation: DelegationConstraints | None

    @model_validator(mode="after")
    def validate_delegation(self) -> "PrincipalContext":
        if self.subject_kind in {"delegated_agent", "worker"} and self.delegation is None:
            raise ValueError("delegated_agent and worker contexts require delegation constraints")
        if self.subject_kind == "human" and self.delegation is not None:
            raise ValueError("human contexts cannot carry delegation constraints")
        if self.delegation is not None:
            for resource in self.delegation.allowed_resources:
                if resource.tenant_id != self.tenant_id:
                    raise ValueError("delegated resources must belong to the principal tenant")
        return self


class MembershipVersion(ContractModel):
    """Current membership witness for one subject in a policy snapshot."""

    subject_id: OpaqueId
    version: SafeCounter


class PolicyStatement(ContractModel):
    """One exact action/resource grant or denial.

    A statement without ``credential_id`` applies to every credential acting
    for the named subject.  A credential-specific statement applies only to
    that exact credential.  Neither form assigns authority to another subject.
    """

    id: OpaqueId
    effect: PolicyEffect
    subject_id: OpaqueId
    credential_id: OpaqueId | None
    action: OpaqueId
    resource: ReferenceHandle


class AccessPolicy(VersionedContract):
    """A complete, versioned input snapshot for the reference evaluator."""

    tenant_id: OpaqueId
    policy_epoch: SafeCounter
    membership_versions: tuple[MembershipVersion, ...]
    statements: tuple[PolicyStatement, ...]

    @model_validator(mode="after")
    def validate_snapshot(self) -> "AccessPolicy":
        subjects = [membership.subject_id for membership in self.membership_versions]
        if len(subjects) != len(set(subjects)):
            raise ValueError("membership_versions must contain at most one entry per subject")

        statement_ids = [statement.id for statement in self.statements]
        if len(statement_ids) != len(set(statement_ids)):
            raise ValueError("policy statement ids must be unique")

        for statement in self.statements:
            if statement.resource.tenant_id != self.tenant_id:
                raise ValueError("policy resources must belong to the policy tenant")
        return self


class PolicyEvaluationInput(VersionedContract):
    """Validated action and resource presented to the pure evaluator.

    Raw caller values must cross this boundary before evaluation.  Malformed
    values produce ordinary contract validation errors; this model does not
    prescribe transport-specific error mapping.
    """

    action: OpaqueId
    resource: ReferenceHandle


class PolicyDecision(VersionedContract):
    """Safe result of evaluating one action against one resource value.

    A denied decision deliberately does not say whether a resource exists or
    whether denial came from an explicit statement, absent grant, stale
    membership, tenant mismatch, or delegation ceiling.
    """

    action: OpaqueId
    resource: ReferenceHandle
    outcome: PolicyOutcome
    reason_code: PolicyReasonCode
    policy_epoch: SafeCounter
    evaluated_tenant_id: OpaqueId
    evaluated_subject_id: OpaqueId
    evaluated_credential_id: OpaqueId
    evaluated_membership_version: SafeCounter

    @model_validator(mode="after")
    def validate_outcome_reason(self) -> "PolicyDecision":
        if self.outcome == "allow" and self.reason_code != "explicit_grant":
            raise ValueError("allow decisions require the explicit_grant reason")
        if self.outcome == "deny" and self.reason_code == "explicit_grant":
            raise ValueError("deny decisions cannot use the explicit_grant reason")
        return self


__all__ = [
    "AccessPolicy",
    "DelegationConstraints",
    "MembershipVersion",
    "PolicyDecision",
    "PolicyEvaluationInput",
    "PolicyEffect",
    "PolicyOutcome",
    "PolicyReasonCode",
    "PolicyStatement",
    "PrincipalContext",
    "SUPPORTED_POLICY_ACTIONS",
    "SubjectKind",
]
