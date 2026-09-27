"""Deterministic, side-effect-free reference policy evaluator."""

from __future__ import annotations

from contracts_policy import (
    SUPPORTED_POLICY_ACTIONS,
    AccessPolicy,
    PolicyDecision,
    PolicyOutcome,
    PolicyReasonCode,
    PrincipalContext,
)
from contracts_references import ReferenceHandle


def _decision(
    *,
    principal: PrincipalContext,
    action: str,
    resource: ReferenceHandle,
    policy: AccessPolicy,
    outcome: PolicyOutcome,
    reason_code: PolicyReasonCode,
) -> PolicyDecision:
    return PolicyDecision(
        schema_version="candidate-v1",
        action=action,
        resource=resource,
        outcome=outcome,
        reason_code=reason_code,
        policy_epoch=policy.policy_epoch,
        evaluated_tenant_id=principal.tenant_id,
        evaluated_subject_id=principal.subject_id,
        evaluated_credential_id=principal.credential_id,
        evaluated_membership_version=principal.membership_version,
    )


def evaluate_policy(
    *,
    principal: PrincipalContext,
    action: str,
    resource: ReferenceHandle,
    policy: AccessPolicy,
) -> PolicyDecision:
    """Evaluate exact policy statements, returning a safe allow/deny decision.

    Evaluation is deliberately limited to the action vocabulary and exact
    reference matching implemented here.  Unknown actions or policy terms fail
    closed.  The function neither resolves the resource nor checks existence.
    """

    def deny(reason_code: PolicyReasonCode = "not_authorized") -> PolicyDecision:
        return _decision(
            principal=principal,
            action=action,
            resource=resource,
            policy=policy,
            outcome="deny",
            reason_code=reason_code,
        )

    if type(action) is not str or action not in SUPPORTED_POLICY_ACTIONS:
        return deny("unsupported_action")

    # A snapshot containing terms this evaluator does not understand cannot be
    # partially interpreted as permissive, even when the unknown term is on an
    # otherwise unrelated statement.
    if any(
        statement.action not in SUPPORTED_POLICY_ACTIONS
        for statement in policy.statements
    ):
        return deny("unsupported_policy_semantics")

    if not (
        principal.tenant_id == policy.tenant_id
        and resource.tenant_id == policy.tenant_id
    ):
        return deny()

    current_membership = next(
        (
            membership
            for membership in policy.membership_versions
            if membership.subject_id == principal.subject_id
        ),
        None,
    )
    if (
        current_membership is None
        or current_membership.version != principal.membership_version
    ):
        return deny()

    if principal.delegation is not None:
        if action not in principal.delegation.allowed_actions:
            return deny()
        if resource not in principal.delegation.allowed_resources:
            return deny()

    matching_statements = tuple(
        statement
        for statement in policy.statements
        if statement.subject_id == principal.subject_id
        and statement.action == action
        and statement.resource == resource
        and (
            statement.credential_id is None
            or statement.credential_id == principal.credential_id
        )
    )

    # Denial precedence is independent of statement ordering.
    if any(statement.effect == "deny" for statement in matching_statements):
        return deny()
    if any(statement.effect == "allow" for statement in matching_statements):
        return _decision(
            principal=principal,
            action=action,
            resource=resource,
            policy=policy,
            outcome="allow",
            reason_code="explicit_grant",
        )
    return deny()


__all__ = ["evaluate_policy"]
