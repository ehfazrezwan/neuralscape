"""Deterministic, side-effect-free reference policy evaluator."""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import BaseModel

from contracts_policy import (
    SUPPORTED_POLICY_ACTIONS,
    AccessPolicy,
    PolicyDecision,
    PolicyEvaluationInput,
    PolicyOutcome,
    PolicyReasonCode,
    PrincipalContext,
)
from contracts_references import ReferenceHandle


def _is_native_dict(value: object) -> bool:
    """Recognize dict storage without consulting an instance-level class view."""

    return issubclass(type(value), dict)


def _mapping_items(
    value: Mapping[object, object],
) -> tuple[tuple[object, object], ...]:
    """Capture one mapping inventory without trusting native-dict overrides."""

    if _is_native_dict(value):
        return tuple(dict.items(value))
    return tuple(value.items())


def _complete_contract_input(value: object, active_ids: set[int]) -> object:
    """Copy a native input graph without normalizing away invalid data."""

    if isinstance(value, BaseModel):
        identity = id(value)
        if identity in active_ids:
            raise ValueError("cyclic contract input is not supported")
        active_ids.add(identity)
        try:
            stored_values = dict(value.__dict__)
            declared_fields = type(value).model_fields
            extra_values = value.__pydantic_extra__
            if extra_values is not None:
                if not (
                    _is_native_dict(extra_values)
                    or isinstance(extra_values, Mapping)
                ):
                    raise ValueError(
                        "contract input extra storage must be a mapping"
                    )
                extra_items = _mapping_items(extra_values)
                for key, item in extra_items:
                    if key in stored_values or key in declared_fields:
                        raise ValueError(
                            "contract input contains duplicate stored fields"
                        )
                    stored_values[key] = item

            # An unchecked copy normally stores its update in ``__dict__``.
            # Retain even an anomalous set-only field so closed-model validation
            # cannot silently erase evidence of unknown input semantics.
            for field_name in value.__pydantic_fields_set__:
                if (
                    field_name not in declared_fields
                    and field_name not in stored_values
                ):
                    stored_values[field_name] = None

            return {
                key: _complete_contract_input(item, active_ids)
                for key, item in stored_values.items()
            }
        finally:
            active_ids.remove(identity)

    if isinstance(value, Mapping):
        identity = id(value)
        if identity in active_ids:
            raise ValueError("cyclic contract input is not supported")
        active_ids.add(identity)
        try:
            # Keys are deliberately not stringified or otherwise normalized.
            return {
                key: _complete_contract_input(item, active_ids)
                for key, item in _mapping_items(value)
            }
        finally:
            active_ids.remove(identity)

    if isinstance(value, list):
        identity = id(value)
        if identity in active_ids:
            raise ValueError("cyclic contract input is not supported")
        active_ids.add(identity)
        try:
            return [_complete_contract_input(item, active_ids) for item in value]
        finally:
            active_ids.remove(identity)

    if isinstance(value, tuple):
        identity = id(value)
        if identity in active_ids:
            raise ValueError("cyclic contract input is not supported")
        active_ids.add(identity)
        try:
            return tuple(_complete_contract_input(item, active_ids) for item in value)
        finally:
            active_ids.remove(identity)

    return value


def _validated_principal_snapshot(principal: PrincipalContext) -> PrincipalContext:
    return PrincipalContext.model_validate(_complete_contract_input(principal, set()))


def _validated_evaluation_snapshot(
    evaluation: PolicyEvaluationInput,
) -> PolicyEvaluationInput:
    return PolicyEvaluationInput.model_validate(
        _complete_contract_input(evaluation, set())
    )


def _validated_policy_snapshot(policy: AccessPolicy) -> AccessPolicy:
    return AccessPolicy.model_validate(_complete_contract_input(policy, set()))


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
    evaluation: PolicyEvaluationInput,
    policy: AccessPolicy,
) -> PolicyDecision:
    """Evaluate exact policy statements, returning a safe allow/deny decision.

    All three input graphs are reconstructed and validated before authorization
    logic.  Evaluation is deliberately limited to the action vocabulary and
    exact reference matching implemented here.  Unknown actions or policy terms
    fail closed.  The function neither resolves the resource nor checks
    existence.
    """

    principal = _validated_principal_snapshot(principal)
    evaluation = _validated_evaluation_snapshot(evaluation)
    policy = _validated_policy_snapshot(policy)

    action = evaluation.action
    resource = evaluation.resource

    def deny(reason_code: PolicyReasonCode = "not_authorized") -> PolicyDecision:
        return _decision(
            principal=principal,
            action=action,
            resource=resource,
            policy=policy,
            outcome="deny",
            reason_code=reason_code,
        )

    if action not in SUPPORTED_POLICY_ACTIONS:
        return deny("unsupported_action")

    # A snapshot containing terms this evaluator does not understand cannot be
    # partially interpreted as permissive, even when the unknown term is on an
    # otherwise unrelated statement.
    if any(
        statement.action not in SUPPORTED_POLICY_ACTIONS
        for statement in policy.statements
    ):
        return deny("unsupported_policy_semantics")
    if principal.delegation is not None and any(
        delegated_action not in SUPPORTED_POLICY_ACTIONS
        for delegated_action in principal.delegation.allowed_actions
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


def validate_policy_decision(decision: PolicyDecision) -> PolicyDecision:
    """Return an independent, deeply validated receiving snapshot.

    Contract instances and unchecked copies are not trusted as provenance.  A
    consumer should use the returned snapshot before serialization or use.
    Validation does not prove that the decision came from an authority.
    """

    return PolicyDecision.model_validate(_complete_contract_input(decision, set()))


__all__ = ["evaluate_policy", "validate_policy_decision"]
