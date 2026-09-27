"""Contract and adversarial tests for the pure reference policy evaluator."""

from __future__ import annotations

from copy import deepcopy

import pytest
from pydantic import ValidationError

from contracts_policy import (
    AccessPolicy,
    DelegationConstraints,
    MembershipVersion,
    PolicyDecision,
    PolicyStatement,
    PrincipalContext,
)
from contracts_policy_reference import evaluate_policy
from contracts_references import ReferenceHandle


VERSION = "candidate-v1"


def reference(
    resource_id: str,
    *,
    tenant_id: str = "tenant-a",
    resolver: str = "memory_by_id",
) -> ReferenceHandle:
    return ReferenceHandle(
        kind="memory",
        id=resource_id,
        tenant_id=tenant_id,
        resolver=resolver,
    )


def principal(
    *,
    tenant_id: str = "tenant-a",
    subject_id: str = "subject-a",
    credential_id: str = "credential-a",
    membership_version: int = 7,
    subject_kind: str = "human",
    delegation: DelegationConstraints | None = None,
) -> PrincipalContext:
    return PrincipalContext(
        schema_version=VERSION,
        tenant_id=tenant_id,
        subject_id=subject_id,
        subject_kind=subject_kind,
        credential_id=credential_id,
        membership_version=membership_version,
        device_id=None,
        delegation=delegation,
    )


def statement(
    statement_id: str,
    effect: str,
    action: str,
    resource_value: ReferenceHandle,
    *,
    subject_id: str = "subject-a",
    credential_id: str | None = None,
) -> PolicyStatement:
    return PolicyStatement(
        id=statement_id,
        effect=effect,
        subject_id=subject_id,
        credential_id=credential_id,
        action=action,
        resource=resource_value,
    )


def policy(
    *statements: PolicyStatement,
    tenant_id: str = "tenant-a",
    policy_epoch: int = 19,
    subject_id: str = "subject-a",
    membership_version: int = 7,
) -> AccessPolicy:
    return AccessPolicy(
        schema_version=VERSION,
        tenant_id=tenant_id,
        policy_epoch=policy_epoch,
        membership_versions=(
            MembershipVersion(subject_id=subject_id, version=membership_version),
        ),
        statements=statements,
    )


def evaluate(
    principal_value: PrincipalContext,
    action: str,
    resource_value: ReferenceHandle,
    policy_value: AccessPolicy,
) -> PolicyDecision:
    return evaluate_policy(
        principal=principal_value,
        action=action,
        resource=resource_value,
        policy=policy_value,
    )


def test_exact_explicit_grant_records_distinct_authority_witnesses() -> None:
    resource_value = reference("memory-1")
    principal_value = principal(membership_version=7)
    policy_value = policy(
        statement("grant-1", "allow", "read", resource_value),
        policy_epoch=19,
        membership_version=7,
    )

    decision = evaluate(principal_value, "read", resource_value, policy_value)

    assert decision.outcome == "allow"
    assert decision.reason_code == "explicit_grant"
    assert decision.policy_epoch == 19
    assert decision.evaluated_membership_version == 7
    assert decision.evaluated_subject_id == "subject-a"
    assert decision.evaluated_credential_id == "credential-a"
    assert decision.resource == resource_value


@pytest.mark.parametrize("reverse", [False, True])
def test_explicit_deny_wins_independently_of_statement_order(reverse: bool) -> None:
    resource_value = reference("memory-1")
    rules = [
        statement("allow-1", "allow", "read", resource_value),
        statement("deny-1", "deny", "read", resource_value),
    ]
    if reverse:
        rules.reverse()

    decision = evaluate(principal(), "read", resource_value, policy(*rules))

    assert decision.outcome == "deny"
    assert decision.reason_code == "not_authorized"


def test_hidden_resource_and_explicit_deny_have_same_safe_result() -> None:
    visible = reference("visible-memory")
    hidden = reference("hidden-memory")
    principal_value = principal()
    policy_value = policy(statement("deny-hidden", "deny", "read", hidden))

    absent_grant = evaluate(principal_value, "read", visible, policy_value)
    explicit_deny = evaluate(principal_value, "read", hidden, policy_value)

    assert (absent_grant.outcome, absent_grant.reason_code) == (
        explicit_deny.outcome,
        explicit_deny.reason_code,
    ) == ("deny", "not_authorized")


def test_credential_specific_grant_does_not_transfer_between_credentials() -> None:
    resource_value = reference("memory-1")
    policy_value = policy(
        statement(
            "credential-grant",
            "allow",
            "read",
            resource_value,
            credential_id="credential-a",
        )
    )

    allowed = evaluate(
        principal(credential_id="credential-a"),
        "read",
        resource_value,
        policy_value,
    )
    denied = evaluate(
        principal(credential_id="credential-b"),
        "read",
        resource_value,
        policy_value,
    )

    assert allowed.outcome == "allow"
    assert denied.outcome == "deny"


def test_delegation_can_narrow_but_never_create_a_grant() -> None:
    allowed_resource = reference("memory-allowed")
    other_resource = reference("memory-other")
    constraints = DelegationConstraints(
        delegated_by_subject_id="delegator-a",
        delegated_by_credential_id="delegator-credential-a",
        allowed_actions=("read",),
        allowed_resources=(allowed_resource,),
    )
    delegated = principal(subject_kind="delegated_agent", delegation=constraints)
    policy_value = policy(
        statement("read-allowed", "allow", "read", allowed_resource),
        statement("update-allowed", "allow", "update", allowed_resource),
        statement("read-other", "allow", "read", other_resource),
    )

    assert evaluate(delegated, "read", allowed_resource, policy_value).outcome == "allow"
    assert evaluate(delegated, "update", allowed_resource, policy_value).outcome == "deny"
    assert evaluate(delegated, "read", other_resource, policy_value).outcome == "deny"

    no_policy_grant = policy()
    assert evaluate(delegated, "read", allowed_resource, no_policy_grant).outcome == "deny"


def test_membership_and_policy_versions_are_independent() -> None:
    resource_value = reference("memory-1")
    grant = statement("grant-1", "allow", "read", resource_value)
    principal_value = principal(membership_version=7)

    stale_membership = policy(grant, membership_version=8, policy_epoch=7)
    current_membership = policy(grant, membership_version=7, policy_epoch=8)

    stale_decision = evaluate(principal_value, "read", resource_value, stale_membership)
    current_decision = evaluate(principal_value, "read", resource_value, current_membership)

    assert stale_decision.outcome == "deny"
    assert stale_decision.policy_epoch == 7
    assert stale_decision.evaluated_membership_version == 7
    assert current_decision.outcome == "allow"
    assert current_decision.policy_epoch == 8


@pytest.mark.parametrize(
    ("principal_tenant", "resource_tenant", "policy_tenant"),
    [
        ("tenant-b", "tenant-a", "tenant-a"),
        ("tenant-a", "tenant-b", "tenant-a"),
        ("tenant-a", "tenant-a", "tenant-b"),
    ],
)
def test_cross_tenant_inputs_fail_closed(
    principal_tenant: str,
    resource_tenant: str,
    policy_tenant: str,
) -> None:
    resource_value = reference("memory-1", tenant_id=resource_tenant)
    policy_resource = reference("memory-1", tenant_id=policy_tenant)
    policy_value = policy(
        statement("grant-1", "allow", "read", policy_resource),
        tenant_id=policy_tenant,
    )

    decision = evaluate(
        principal(tenant_id=principal_tenant),
        "read",
        resource_value,
        policy_value,
    )

    assert decision.outcome == "deny"
    assert decision.reason_code == "not_authorized"


def test_unknown_request_and_policy_semantics_fail_closed() -> None:
    resource_value = reference("memory-1")
    principal_value = principal()
    ordinary_grant = statement("grant-1", "allow", "read", resource_value)

    unknown_request = evaluate(
        principal_value,
        "future_action",
        resource_value,
        policy(ordinary_grant),
    )
    assert unknown_request.outcome == "deny"
    assert unknown_request.reason_code == "unsupported_action"

    unknown_policy = policy(
        ordinary_grant,
        statement("future-grant", "allow", "future_action", resource_value),
    )
    known_request = evaluate(principal_value, "read", resource_value, unknown_policy)
    assert known_request.outcome == "deny"
    assert known_request.reason_code == "unsupported_policy_semantics"


def test_reference_value_must_match_exactly_and_is_not_authority() -> None:
    granted_reference = reference("memory-1", resolver="memory_by_id")
    alternate_reference = reference("memory-1", resolver="historical_memory")
    policy_value = policy(statement("grant-1", "allow", "read", granted_reference))

    assert evaluate(principal(), "read", alternate_reference, policy_value).outcome == "deny"


def test_administration_content_and_standard_authority_are_independent() -> None:
    team = reference("team-settings")
    private_memory = reference("private-memory")
    standard = reference("candidate-standard")

    admin_policy = policy(
        statement("admin", "allow", "manage_membership", team),
        statement("no-content", "deny", "read", private_memory),
    )
    reader_policy = policy(
        statement("content", "allow", "read", private_memory),
    )
    publisher_policy = policy(
        statement("publisher", "allow", "publish_standard", standard),
    )

    assert (
        evaluate(principal(), "manage_membership", team, admin_policy).outcome
        == "allow"
    )
    assert evaluate(principal(), "read", private_memory, admin_policy).outcome == "deny"
    assert evaluate(principal(), "read", private_memory, reader_policy).outcome == "allow"
    assert (
        evaluate(principal(), "manage_membership", team, reader_policy).outcome
        == "deny"
    )
    assert (
        evaluate(principal(), "publish_standard", standard, publisher_policy).outcome
        == "allow"
    )
    assert evaluate(principal(), "read", private_memory, publisher_policy).outcome == "deny"


def test_evaluation_is_deterministic_and_does_not_mutate_inputs() -> None:
    resource_value = reference("memory-1")
    principal_value = principal()
    policy_value = policy(statement("grant-1", "allow", "read", resource_value))
    before_principal = deepcopy(principal_value)
    before_policy = deepcopy(policy_value)

    first = evaluate(principal_value, "read", resource_value, policy_value)
    second = evaluate(principal_value, "read", resource_value, policy_value)

    assert first == second
    assert principal_value == before_principal
    assert policy_value == before_policy


def test_contracts_reject_missing_versions_unknown_fields_and_invalid_delegation() -> None:
    base = {
        "schema_version": VERSION,
        "tenant_id": "tenant-a",
        "subject_id": "subject-a",
        "subject_kind": "human",
        "credential_id": "credential-a",
        "membership_version": 1,
        "device_id": None,
        "delegation": None,
    }

    without_version = dict(base)
    without_version.pop("schema_version")
    with_extra = {**base, "role": "owner"}

    with pytest.raises(ValidationError):
        PrincipalContext.model_validate(without_version)
    with pytest.raises(ValidationError):
        PrincipalContext.model_validate(with_extra)
    with pytest.raises(ValidationError):
        PrincipalContext.model_validate({**base, "membership_version": True})
    with pytest.raises(ValidationError):
        PrincipalContext.model_validate({**base, "subject_kind": "delegated_agent"})


def test_policy_rejects_ambiguous_memberships_duplicate_rules_and_foreign_resources() -> None:
    resource_value = reference("memory-1")
    rule = statement("rule-1", "allow", "read", resource_value)

    with pytest.raises(ValidationError):
        AccessPolicy(
            schema_version=VERSION,
            tenant_id="tenant-a",
            policy_epoch=1,
            membership_versions=(
                MembershipVersion(subject_id="subject-a", version=1),
                MembershipVersion(subject_id="subject-a", version=2),
            ),
            statements=(),
        )
    with pytest.raises(ValidationError):
        policy(rule, rule)
    with pytest.raises(ValidationError):
        policy(
            statement(
                "foreign-rule",
                "allow",
                "read",
                reference("memory-1", tenant_id="tenant-b"),
            )
        )


def test_policy_decision_schema_requires_every_evaluation_identity() -> None:
    schema = PolicyDecision.model_json_schema()

    assert schema["additionalProperties"] is False
    assert {
        "schema_version",
        "action",
        "resource",
        "outcome",
        "reason_code",
        "policy_epoch",
        "evaluated_tenant_id",
        "evaluated_subject_id",
        "evaluated_credential_id",
        "evaluated_membership_version",
    } <= set(schema["required"])
