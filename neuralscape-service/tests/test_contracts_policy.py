"""Contract and adversarial tests for the pure reference policy evaluator."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from copy import deepcopy
from types import MappingProxyType

import pytest
from pydantic import ValidationError

from contracts_policy import (
    AccessPolicy,
    DelegationConstraints,
    MembershipVersion,
    PolicyDecision,
    PolicyEvaluationInput,
    PolicyStatement,
    PrincipalContext,
)
from contracts_policy_reference import evaluate_policy, validate_policy_decision
from contracts_references import ReferenceHandle


VERSION = "candidate-v1"


class _HiddenBackingExtras(dict[str, object]):
    """A native dict whose overridable public views conceal its backing."""

    def __init__(self, entries: dict[str, object]) -> None:
        super().__init__(entries)
        self.view_calls = 0

    def __bool__(self) -> bool:
        self.view_calls += 1
        return False

    def __len__(self) -> int:
        self.view_calls += 1
        return 0

    def __iter__(self) -> Iterator[str]:
        self.view_calls += 1
        return iter(())

    def keys(self):
        self.view_calls += 1
        return {}.keys()

    def items(self):
        self.view_calls += 1
        return {}.items()


class _RaisingClassExtras(_HiddenBackingExtras):
    """A native dict whose instance-level class view must not be consulted."""

    @property
    def __class__(self):
        raise RuntimeError("instance __class__ consulted")


class _InverseViewExtras(Mapping[str, object]):
    """Expose names through iteration while keeping the items inventory empty."""

    def __init__(self, entries: dict[str, object]) -> None:
        self._entries = entries
        self.iteration_calls = 0
        self.items_calls = 0
        self.getitem_calls = 0

    def __getitem__(self, key: str) -> object:
        self.getitem_calls += 1
        return self._entries[key]

    def __iter__(self) -> Iterator[str]:
        self.iteration_calls += 1
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def items(self):
        self.items_calls += 1
        return {}.items()


class _DictSpoofingExtras(_InverseViewExtras):
    """A non-dict mapping whose instance-level class view claims dict."""

    @property
    def __class__(self):
        return dict


class _ChangingItemsExtras(Mapping[str, object]):
    """Reveal an unknown item only if a consumer requests a second inventory."""

    def __init__(self) -> None:
        self.items_calls = 0

    def __getitem__(self, key: str) -> object:
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return iter(())

    def __len__(self) -> int:
        return 0

    def items(self):
        self.items_calls += 1
        if self.items_calls == 1:
            return {}.items()
        return {"future_constraint": "deny"}.items()


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
        evaluation=PolicyEvaluationInput(
            schema_version=VERSION,
            action=action,
            resource=resource_value,
        ),
        policy=policy_value,
    )


def decision_payload(
    *,
    action: str = "read",
    resource_tenant: str = "tenant-a",
    evaluated_tenant_id: str = "tenant-a",
    outcome: str = "allow",
    reason_code: str = "explicit_grant",
) -> dict[str, object]:
    return {
        "schema_version": VERSION,
        "action": action,
        "resource": reference("memory-1", tenant_id=resource_tenant).model_dump(
            mode="python"
        ),
        "outcome": outcome,
        "reason_code": reason_code,
        "policy_epoch": 1,
        "evaluated_tenant_id": evaluated_tenant_id,
        "evaluated_subject_id": "subject-a",
        "evaluated_credential_id": "credential-a",
        "evaluated_membership_version": 1,
    }


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


def test_unknown_delegation_action_semantics_fail_closed() -> None:
    resource_value = reference("memory-1")
    delegated = principal(
        subject_kind="delegated_agent",
        delegation=DelegationConstraints(
            delegated_by_subject_id="delegator-a",
            delegated_by_credential_id="delegator-credential-a",
            allowed_actions=("read", "future_action"),
            allowed_resources=(resource_value,),
        ),
    )
    policy_value = policy(
        statement("read-grant", "allow", "read", resource_value),
    )

    decision = evaluate(delegated, "read", resource_value, policy_value)

    assert decision.outcome == "deny"
    assert decision.reason_code == "unsupported_policy_semantics"


@pytest.mark.parametrize("malformed_action", [None, True, "", "x" * 257])
def test_evaluation_input_rejects_malformed_raw_actions(malformed_action: object) -> None:
    with pytest.raises(ValidationError):
        PolicyEvaluationInput(
            schema_version=VERSION,
            action=malformed_action,
            resource=reference("memory-1"),
        )


@pytest.mark.parametrize(
    ("outcome", "reason_code"),
    [
        ("allow", "not_authorized"),
        ("deny", "explicit_grant"),
    ],
)
def test_policy_decision_rejects_contradictory_outcome_reason_pairs(
    outcome: str,
    reason_code: str,
) -> None:
    payload = decision_payload(outcome=outcome, reason_code=reason_code)

    with pytest.raises(ValidationError):
        PolicyDecision.model_validate(payload)
    with pytest.raises(ValidationError):
        PolicyDecision.model_validate_json(json.dumps(payload))


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (
            decision_payload(action="future-action"),
            "allow decisions require a supported action",
        ),
        (
            decision_payload(resource_tenant="tenant-b"),
            "resource tenant must match the evaluated tenant",
        ),
    ],
)
def test_policy_decision_rejects_structurally_impossible_allow_states(
    payload: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        PolicyDecision(**payload)
    with pytest.raises(ValidationError, match=message):
        PolicyDecision.model_validate(payload)
    with pytest.raises(ValidationError, match=message):
        PolicyDecision.model_validate_json(json.dumps(payload))


def test_policy_decision_accepts_valid_allow_in_all_representations() -> None:
    payload = decision_payload()

    constructed = PolicyDecision(**payload)
    native = PolicyDecision.model_validate(payload)
    from_json = PolicyDecision.model_validate_json(json.dumps(payload))

    assert constructed == native == from_json
    assert (constructed.outcome, constructed.reason_code) == (
        "allow",
        "explicit_grant",
    )


@pytest.mark.parametrize(
    "payload",
    [
        decision_payload(
            action="future-action",
            outcome="deny",
            reason_code="unsupported_action",
        ),
        decision_payload(
            resource_tenant="tenant-b",
            outcome="deny",
            reason_code="not_authorized",
        ),
    ],
)
def test_policy_decision_preserves_relevant_deny_states(
    payload: dict[str, object],
) -> None:
    constructed = PolicyDecision(**payload)
    native = PolicyDecision.model_validate(payload)
    from_json = PolicyDecision.model_validate_json(json.dumps(payload))

    assert constructed == native == from_json
    assert constructed.outcome == "deny"
    assert validate_policy_decision(constructed) == constructed


def test_evaluator_revalidates_unchecked_principal_copy_before_grant_logic() -> None:
    resource_value = reference("memory-1")
    invalid_principal = principal().model_copy(
        update={"subject_kind": "delegated_agent"}
    )
    policy_value = policy(
        statement("read-grant", "allow", "read", resource_value),
    )

    with pytest.raises(ValidationError, match="require delegation constraints"):
        evaluate_policy(
            principal=invalid_principal,
            evaluation=PolicyEvaluationInput(
                schema_version=VERSION,
                action="read",
                resource=resource_value,
            ),
            policy=policy_value,
        )


def test_evaluator_revalidates_complete_unchecked_policy_copy() -> None:
    resource_value = reference("memory-1")
    stale_policy = policy(
        statement("read-grant", "allow", "read", resource_value),
        membership_version=8,
    )
    invalid_policy = stale_policy.model_copy(
        update={
            "membership_versions": (
                MembershipVersion(subject_id="subject-a", version=7),
                MembershipVersion(subject_id="subject-a", version=8),
            )
        }
    )

    with pytest.raises(ValidationError, match="at most one entry per subject"):
        evaluate_policy(
            principal=principal(membership_version=7),
            evaluation=PolicyEvaluationInput(
                schema_version=VERSION,
                action="read",
                resource=resource_value,
            ),
            policy=invalid_policy,
        )


def test_evaluator_revalidates_unchecked_evaluation_copy() -> None:
    resource_value = reference("memory-1")
    invalid_evaluation = PolicyEvaluationInput(
        schema_version=VERSION,
        action="read",
        resource=resource_value,
    ).model_copy(update={"action": True})

    with pytest.raises(ValidationError):
        evaluate_policy(
            principal=principal(),
            evaluation=invalid_evaluation,
            policy=policy(
                statement("read-grant", "allow", "read", resource_value),
            ),
        )


@pytest.mark.parametrize(
    "unknown_position",
    [
        "principal",
        "evaluation",
        "statement",
        "policy",
        "membership",
        "delegation",
        "reference",
    ],
)
def test_evaluator_rejects_undeclared_fields_in_complete_input_graph(
    unknown_position: str,
) -> None:
    resource_value = reference("memory-1")
    principal_value = principal()
    evaluation_value = PolicyEvaluationInput(
        schema_version=VERSION,
        action="read",
        resource=resource_value,
    )
    policy_value = policy(
        statement("read-grant", "allow", "read", resource_value),
    )

    if unknown_position == "principal":
        principal_value = principal_value.model_copy(
            update={"future_constraint": "deny"}
        )
    elif unknown_position == "evaluation":
        evaluation_value = evaluation_value.model_copy(
            update={"future_constraint": "deny"}
        )
    elif unknown_position == "statement":
        invalid_statement = policy_value.statements[0].model_copy(
            update={"future_constraint": "deny"}
        )
        policy_value = policy_value.model_copy(
            update={"statements": (invalid_statement,)}
        )
    elif unknown_position == "policy":
        policy_value = policy_value.model_copy(
            update={"future_constraint": "deny"}
        )
    elif unknown_position == "membership":
        invalid_membership = policy_value.membership_versions[0].model_copy(
            update={"future_constraint": "deny"}
        )
        policy_value = policy_value.model_copy(
            update={"membership_versions": (invalid_membership,)}
        )
    elif unknown_position == "delegation":
        valid_constraints = DelegationConstraints(
            delegated_by_subject_id="delegator-a",
            delegated_by_credential_id="delegator-credential-a",
            allowed_actions=("read",),
            allowed_resources=(resource_value,),
        )
        constraints = valid_constraints.model_copy(
            update={"future_constraint": "deny"}
        )
        with pytest.raises(ValidationError, match="future_constraint"):
            principal(
                subject_kind="delegated_agent",
                delegation=constraints,
            )
        principal_value = principal(
            subject_kind="delegated_agent",
            delegation=valid_constraints,
        ).model_copy(
            update={"delegation": constraints}
        )
    else:
        invalid_resource = resource_value.model_copy(
            update={"future_constraint": "deny"}
        )
        invalid_statement = policy_value.statements[0].model_copy(
            update={"resource": invalid_resource}
        )
        policy_value = policy_value.model_copy(
            update={"statements": (invalid_statement,)}
        )

    with pytest.raises(ValidationError):
        evaluate_policy(
            principal=principal_value,
            evaluation=evaluation_value,
            policy=policy_value,
        )


def test_evaluator_preserves_valid_declared_copy_updates() -> None:
    resource_value = reference("memory-1")
    evaluation_value = PolicyEvaluationInput(
        schema_version=VERSION,
        action="read",
        resource=resource_value,
    ).model_copy(update={"action": "update"})
    read_grant = statement("grant", "allow", "read", resource_value)
    policy_value = policy(read_grant).model_copy(
        update={"statements": (read_grant.model_copy(update={"action": "update"}),)}
    )

    decision = evaluate_policy(
        principal=principal(),
        evaluation=evaluation_value,
        policy=policy_value,
    )

    assert (decision.outcome, decision.reason_code) == ("allow", "explicit_grant")


def test_evaluator_preserves_list_type_for_strict_revalidation() -> None:
    resource_value = reference("memory-1")
    grant = statement("read-grant", "allow", "read", resource_value)
    invalid_policy = policy(grant).model_copy(update={"statements": [grant]})

    with pytest.raises(ValidationError):
        evaluate_policy(
            principal=principal(),
            evaluation=PolicyEvaluationInput(
                schema_version=VERSION,
                action="read",
                resource=resource_value,
            ),
            policy=invalid_policy,
        )


def test_evaluator_preserves_invalid_mapping_key_for_revalidation() -> None:
    resource_value = reference("memory-1")
    principal_payload: dict[object, object] = principal().model_dump(mode="python")
    principal_payload[1] = "unknown-semantics"

    with pytest.raises(ValidationError) as exc_info:
        evaluate_policy(
            principal=principal_payload,  # type: ignore[arg-type]
            evaluation=PolicyEvaluationInput(
                schema_version=VERSION,
                action="read",
                resource=resource_value,
            ),
            policy=policy(
                statement("read-grant", "allow", "read", resource_value),
            ),
        )

    assert exc_info.value.errors()[0]["type"] == "invalid_key"
    assert exc_info.value.errors()[0]["loc"] == (1,)


@pytest.mark.parametrize("malformed_argument", ["principal", "evaluation", "policy"])
def test_evaluator_rejects_malformed_scalar_helper_arguments(
    malformed_argument: str,
) -> None:
    resource_value = reference("memory-1")
    arguments = {
        "principal": principal(),
        "evaluation": PolicyEvaluationInput(
            schema_version=VERSION,
            action="read",
            resource=resource_value,
        ),
        "policy": policy(statement("read-grant", "allow", "read", resource_value)),
    }
    arguments[malformed_argument] = "not-a-contract"

    with pytest.raises(ValidationError):
        evaluate_policy(**arguments)  # type: ignore[arg-type]


def test_evaluator_rejects_cyclic_input_predictably() -> None:
    cyclic_delegation: list[object] = []
    cyclic_delegation.append(cyclic_delegation)
    invalid_principal = principal().model_copy(
        update={"delegation": cyclic_delegation}
    )

    with pytest.raises(ValueError, match="cyclic contract input is not supported"):
        evaluate_policy(
            principal=invalid_principal,
            evaluation=PolicyEvaluationInput(
                schema_version=VERSION,
                action="read",
                resource=reference("memory-1"),
            ),
            policy=policy(),
        )


@pytest.mark.parametrize(
    "malformed_extra",
    [[], (), "", 0, False, ["retained_unknown"]],
)
@pytest.mark.parametrize("location", ["principal", "nested_reference"])
def test_evaluator_rejects_malformed_extra_storage(
    malformed_extra: object,
    location: str,
) -> None:
    resource_value = reference("memory-1")
    principal_value = principal()
    evaluation_value = PolicyEvaluationInput(
        schema_version=VERSION,
        action="read",
        resource=resource_value,
    )
    if location == "principal":
        object.__setattr__(
            principal_value,
            "__pydantic_extra__",
            malformed_extra,
        )
    else:
        object.__setattr__(
            evaluation_value.resource,
            "__pydantic_extra__",
            malformed_extra,
        )

    with pytest.raises(
        ValueError,
        match="contract input extra storage must be a mapping",
    ):
        evaluate_policy(
            principal=principal_value,
            evaluation=evaluation_value,
            policy=policy(
                statement("read-grant", "allow", "read", resource_value),
            ),
        )


@pytest.mark.parametrize("extra_storage", [None, {}, MappingProxyType({})])
def test_evaluator_accepts_valid_empty_extra_storage(
    extra_storage: object,
) -> None:
    resource_value = reference("memory-1")
    principal_value = principal()
    object.__setattr__(
        principal_value,
        "__pydantic_extra__",
        extra_storage,
    )

    decision = evaluate_policy(
        principal=principal_value,
        evaluation=PolicyEvaluationInput(
            schema_version=VERSION,
            action="read",
            resource=resource_value,
        ),
        policy=policy(
            statement("read-grant", "allow", "read", resource_value),
        ),
    )

    assert (decision.outcome, decision.reason_code) == ("allow", "explicit_grant")


@pytest.mark.parametrize(
    ("boundary", "location"),
    [
        ("evaluator", "direct"),
        ("evaluator", "nested"),
        ("receiving", "direct"),
        ("receiving", "nested"),
    ],
)
def test_policy_boundaries_reject_hidden_native_dict_unknown_extra(
    boundary: str,
    location: str,
) -> None:
    extras = _HiddenBackingExtras({"future_constraint": "deny"})

    if boundary == "evaluator":
        resource_value = reference("memory-1")
        principal_value = principal()
        evaluation_value = PolicyEvaluationInput(
            schema_version=VERSION,
            action="read",
            resource=resource_value,
        )
        target = principal_value if location == "direct" else evaluation_value.resource
        object.__setattr__(target, "__pydantic_extra__", extras)

        with pytest.raises(ValidationError, match="future_constraint"):
            evaluate_policy(
                principal=principal_value,
                evaluation=evaluation_value,
                policy=policy(
                    statement(
                        "read-grant",
                        "allow",
                        "read",
                        reference("memory-1"),
                    ),
                ),
            )
    else:
        decision = PolicyDecision(**decision_payload())
        target = decision if location == "direct" else decision.resource
        object.__setattr__(target, "__pydantic_extra__", extras)

        with pytest.raises(ValidationError, match="future_constraint"):
            validate_policy_decision(decision)

    assert extras.view_calls == 0


@pytest.mark.parametrize(
    ("boundary", "location"),
    [
        ("evaluator", "direct"),
        ("evaluator", "nested"),
        ("receiving", "direct"),
        ("receiving", "nested"),
    ],
)
def test_policy_boundaries_use_concrete_type_for_native_dict_extra(
    boundary: str,
    location: str,
) -> None:
    extras = _RaisingClassExtras({"future_constraint": "deny"})

    if boundary == "evaluator":
        resource_value = reference("memory-1")
        principal_value = principal()
        evaluation_value = PolicyEvaluationInput(
            schema_version=VERSION,
            action="read",
            resource=resource_value,
        )
        target = principal_value if location == "direct" else evaluation_value.resource
        object.__setattr__(target, "__pydantic_extra__", extras)

        with pytest.raises(ValidationError, match="future_constraint"):
            evaluate_policy(
                principal=principal_value,
                evaluation=evaluation_value,
                policy=policy(
                    statement(
                        "read-grant",
                        "allow",
                        "read",
                        reference("memory-1"),
                    ),
                ),
            )
    else:
        decision = PolicyDecision(**decision_payload())
        target = decision if location == "direct" else decision.resource
        object.__setattr__(target, "__pydantic_extra__", extras)

        with pytest.raises(ValidationError, match="future_constraint"):
            validate_policy_decision(decision)

    assert extras.view_calls == 0


@pytest.mark.parametrize(
    ("boundary", "location"),
    [
        ("evaluator", "direct"),
        ("evaluator", "nested"),
        ("receiving", "direct"),
        ("receiving", "nested"),
    ],
)
def test_policy_boundaries_do_not_trust_spoofed_dict_class(
    boundary: str,
    location: str,
) -> None:
    extras = _DictSpoofingExtras({"future_constraint": "deny"})

    if boundary == "evaluator":
        resource_value = reference("memory-1")
        principal_value = principal()
        evaluation_value = PolicyEvaluationInput(
            schema_version=VERSION,
            action="read",
            resource=resource_value,
        )
        target = principal_value if location == "direct" else evaluation_value.resource
        object.__setattr__(target, "__pydantic_extra__", extras)
        received = evaluate_policy(
            principal=principal_value,
            evaluation=evaluation_value,
            policy=policy(
                statement(
                    "read-grant",
                    "allow",
                    "read",
                    reference("memory-1"),
                ),
            ),
        )
        assert (received.outcome, received.reason_code) == (
            "allow",
            "explicit_grant",
        )
    else:
        decision = PolicyDecision(**decision_payload())
        target = decision if location == "direct" else decision.resource
        object.__setattr__(target, "__pydantic_extra__", extras)
        received = validate_policy_decision(decision)
        assert received.model_dump(mode="python") == decision_payload()

    assert extras.items_calls == 1
    assert extras.iteration_calls == 0
    assert extras.getitem_calls == 0


@pytest.mark.parametrize(
    ("boundary", "location", "field_name", "extra_value"),
    [
        ("evaluator", "direct", "subject_id", "shadow-subject"),
        ("evaluator", "nested", "id", "shadow-memory"),
        ("receiving", "direct", "action", "delete"),
        ("receiving", "nested", "id", "shadow-memory"),
    ],
)
def test_policy_boundaries_reject_hidden_native_dict_declared_overlap(
    boundary: str,
    location: str,
    field_name: str,
    extra_value: str,
) -> None:
    extras = _HiddenBackingExtras({field_name: extra_value})

    if boundary == "evaluator":
        resource_value = reference("memory-1")
        principal_value = principal()
        evaluation_value = PolicyEvaluationInput(
            schema_version=VERSION,
            action="read",
            resource=resource_value,
        )
        target = principal_value if location == "direct" else evaluation_value.resource
        object.__setattr__(target, "__pydantic_extra__", extras)

        with pytest.raises(
            ValueError,
            match="contract input contains duplicate stored fields",
        ):
            evaluate_policy(
                principal=principal_value,
                evaluation=evaluation_value,
                policy=policy(
                    statement(
                        "read-grant",
                        "allow",
                        "read",
                        reference("memory-1"),
                    ),
                ),
            )
    else:
        decision = PolicyDecision(**decision_payload())
        target = decision if location == "direct" else decision.resource
        object.__setattr__(target, "__pydantic_extra__", extras)

        with pytest.raises(
            ValueError,
            match="contract input contains duplicate stored fields",
        ):
            validate_policy_decision(decision)

    assert extras.view_calls == 0


@pytest.mark.parametrize(
    ("boundary", "location"),
    [
        ("evaluator", "direct"),
        ("evaluator", "nested"),
        ("receiving", "direct"),
        ("receiving", "nested"),
    ],
)
def test_policy_boundaries_preserve_non_dict_inverse_items_authority(
    boundary: str,
    location: str,
) -> None:
    extras = _InverseViewExtras({"future_constraint": "deny"})

    if boundary == "evaluator":
        resource_value = reference("memory-1")
        principal_value = principal()
        evaluation_value = PolicyEvaluationInput(
            schema_version=VERSION,
            action="read",
            resource=resource_value,
        )
        target = principal_value if location == "direct" else evaluation_value.resource
        object.__setattr__(target, "__pydantic_extra__", extras)
        received = evaluate_policy(
            principal=principal_value,
            evaluation=evaluation_value,
            policy=policy(
                statement(
                    "read-grant",
                    "allow",
                    "read",
                    reference("memory-1"),
                ),
            ),
        )
        assert (received.outcome, received.reason_code) == (
            "allow",
            "explicit_grant",
        )
    else:
        decision = PolicyDecision(**decision_payload())
        target = decision if location == "direct" else decision.resource
        object.__setattr__(target, "__pydantic_extra__", extras)
        received = validate_policy_decision(decision)
        assert received.model_dump(mode="python") == decision_payload()

    assert extras.items_calls == 1
    assert extras.iteration_calls == 0
    assert extras.getitem_calls == 0


@pytest.mark.parametrize("boundary", ["evaluator", "receiving"])
def test_policy_boundaries_capture_non_dict_items_inventory_once(
    boundary: str,
) -> None:
    extras = _ChangingItemsExtras()

    if boundary == "evaluator":
        resource_value = reference("memory-1")
        principal_value = principal()
        object.__setattr__(principal_value, "__pydantic_extra__", extras)
        received = evaluate(
            principal_value,
            "read",
            resource_value,
            policy(statement("read-grant", "allow", "read", resource_value)),
        )
        assert received.outcome == "allow"
    else:
        decision = PolicyDecision(**decision_payload())
        object.__setattr__(decision, "__pydantic_extra__", extras)
        received = validate_policy_decision(decision)
        assert received.model_dump(mode="python") == decision_payload()

    assert extras.items_calls == 1


@pytest.mark.parametrize("boundary", ["evaluator", "receiving"])
def test_policy_boundaries_accept_genuinely_empty_native_dict_subclass(
    boundary: str,
) -> None:
    extras = _HiddenBackingExtras({})

    if boundary == "evaluator":
        resource_value = reference("memory-1")
        principal_value = principal()
        object.__setattr__(principal_value, "__pydantic_extra__", extras)
        received = evaluate(
            principal_value,
            "read",
            resource_value,
            policy(statement("read-grant", "allow", "read", resource_value)),
        )
        assert received.outcome == "allow"
    else:
        decision = PolicyDecision(**decision_payload())
        object.__setattr__(decision, "__pydantic_extra__", extras)
        received = validate_policy_decision(decision)
        assert received.model_dump(mode="python") == decision_payload()

    assert extras.view_calls == 0


@pytest.mark.parametrize(
    ("location", "field_name"),
    [
        ("principal", "subject_id"),
        ("nested_reference", "id"),
    ],
)
def test_evaluator_rejects_required_field_moved_to_extra_storage(
    location: str,
    field_name: str,
) -> None:
    resource_value = reference("memory-1")
    principal_value = principal()
    evaluation_value = PolicyEvaluationInput(
        schema_version=VERSION,
        action="read",
        resource=resource_value,
    )
    target = principal_value if location == "principal" else evaluation_value.resource
    moved_value = target.__dict__.pop(field_name)
    object.__setattr__(
        target,
        "__pydantic_extra__",
        {field_name: moved_value},
    )

    with pytest.raises(
        ValueError,
        match="contract input contains duplicate stored fields",
    ):
        evaluate_policy(
            principal=principal_value,
            evaluation=evaluation_value,
            policy=policy(
                statement("read-grant", "allow", "read", resource_value),
            ),
        )


def test_evaluator_still_rejects_required_field_absent_from_all_storage() -> None:
    resource_value = reference("memory-1")
    principal_value = principal()
    principal_value.__dict__.pop("subject_id")

    with pytest.raises(ValidationError):
        evaluate_policy(
            principal=principal_value,
            evaluation=PolicyEvaluationInput(
                schema_version=VERSION,
                action="read",
                resource=resource_value,
            ),
            policy=policy(
                statement("read-grant", "allow", "read", resource_value),
            ),
        )


def test_evaluator_retains_nonconflicting_unknown_extra_for_rejection() -> None:
    resource_value = reference("memory-1")
    principal_value = principal()
    object.__setattr__(
        principal_value,
        "__pydantic_extra__",
        {"future_constraint": "deny"},
    )

    with pytest.raises(ValidationError, match="future_constraint"):
        evaluate_policy(
            principal=principal_value,
            evaluation=PolicyEvaluationInput(
                schema_version=VERSION,
                action="read",
                resource=resource_value,
            ),
            policy=policy(
                statement("read-grant", "allow", "read", resource_value),
            ),
        )


@pytest.mark.parametrize("extra_subject_id", ["subject-a", "subject-b"])
def test_evaluator_rejects_equal_or_conflicting_declared_extra_storage(
    extra_subject_id: str,
) -> None:
    resource_value = reference("memory-1")
    principal_value = principal()
    object.__setattr__(
        principal_value,
        "__pydantic_extra__",
        {"subject_id": extra_subject_id},
    )

    with pytest.raises(
        ValueError,
        match="contract input contains duplicate stored fields",
    ):
        evaluate_policy(
            principal=principal_value,
            evaluation=PolicyEvaluationInput(
                schema_version=VERSION,
                action="read",
                resource=resource_value,
            ),
            policy=policy(
                statement("read-grant", "allow", "read", resource_value),
            ),
        )


def test_evaluator_preserves_missing_subclass_default_without_extra_override() -> None:
    class ExtendedPrincipalContext(PrincipalContext):
        audit_marker: str = "default-marker"

    principal_value = ExtendedPrincipalContext.model_validate(
        principal().model_dump(mode="python")
    )
    principal_value.__dict__.pop("audit_marker")
    resource_value = reference("memory-1")

    decision = evaluate_policy(
        principal=principal_value,
        evaluation=PolicyEvaluationInput(
            schema_version=VERSION,
            action="read",
            resource=resource_value,
        ),
        policy=policy(
            statement("read-grant", "allow", "read", resource_value),
        ),
    )

    assert (decision.outcome, decision.reason_code) == ("allow", "explicit_grant")


def test_evaluator_rejects_subclass_declared_default_moved_to_extra_storage() -> None:
    class ExtendedPrincipalContext(PrincipalContext):
        audit_marker: str = "default-marker"

    principal_value = ExtendedPrincipalContext.model_validate(
        principal().model_dump(mode="python")
    )
    principal_value.__dict__.pop("audit_marker")
    object.__setattr__(
        principal_value,
        "__pydantic_extra__",
        {"audit_marker": "override-marker"},
    )
    resource_value = reference("memory-1")

    with pytest.raises(
        ValueError,
        match="contract input contains duplicate stored fields",
    ):
        evaluate_policy(
            principal=principal_value,
            evaluation=PolicyEvaluationInput(
                schema_version=VERSION,
                action="read",
                resource=resource_value,
            ),
            policy=policy(
                statement("read-grant", "allow", "read", resource_value),
            ),
        )


@pytest.mark.parametrize("input_graph", ["principal", "evaluation", "policy"])
def test_evaluator_revalidates_nested_references_in_every_input_graph(
    input_graph: str,
) -> None:
    resource_value = reference("memory-1")
    invalid_resource = resource_value.model_copy(update={"tenant_id": ""})
    principal_value = principal()
    evaluation_value = PolicyEvaluationInput(
        schema_version=VERSION,
        action="read",
        resource=resource_value,
    )
    policy_value = policy(
        statement("read-grant", "allow", "read", resource_value),
    )

    if input_graph == "principal":
        valid_constraints = DelegationConstraints(
            delegated_by_subject_id="delegator-a",
            delegated_by_credential_id="delegator-credential-a",
            allowed_actions=("read",),
            allowed_resources=(resource_value,),
        )
        constraints = valid_constraints.model_copy(
            update={"allowed_resources": (invalid_resource,)}
        )
        principal_value = principal(
            subject_kind="delegated_agent",
            delegation=valid_constraints,
        ).model_copy(update={"delegation": constraints})
    elif input_graph == "evaluation":
        evaluation_value = evaluation_value.model_copy(
            update={"resource": invalid_resource}
        )
    else:
        invalid_statement = policy_value.statements[0].model_copy(
            update={"resource": invalid_resource}
        )
        policy_value = policy_value.model_copy(
            update={"statements": (invalid_statement,)}
        )

    with pytest.raises(ValidationError):
        evaluate_policy(
            principal=principal_value,
            evaluation=evaluation_value,
            policy=policy_value,
        )


def test_policy_decision_is_frozen_after_validation() -> None:
    resource_value = reference("memory-1")
    decision = evaluate(
        principal(),
        "read",
        resource_value,
        policy(statement("read-grant", "allow", "read", resource_value)),
    )

    with pytest.raises(ValidationError, match="frozen"):
        decision.outcome = "deny"


def test_receiving_boundary_revalidates_unchecked_decision_copy() -> None:
    resource_value = reference("memory-1")
    decision = evaluate(
        principal(),
        "read",
        resource_value,
        policy(statement("read-grant", "allow", "read", resource_value)),
    )
    contradictory_copy = decision.model_copy(update={"outcome": "deny"})

    with pytest.raises(ValidationError, match="explicit_grant"):
        validate_policy_decision(contradictory_copy)


@pytest.mark.parametrize("invalid_field", ["action", "resource_tenant"])
def test_receiving_boundary_rejects_impossible_unchecked_allow_copy(
    invalid_field: str,
) -> None:
    decision = PolicyDecision(**decision_payload())
    if invalid_field == "action":
        invalid_decision = decision.model_copy(update={"action": "future-action"})
        message = "allow decisions require a supported action"
    else:
        invalid_resource = decision.resource.model_copy(
            update={"tenant_id": "tenant-b"}
        )
        invalid_decision = decision.model_copy(update={"resource": invalid_resource})
        message = "resource tenant must match the evaluated tenant"

    with pytest.raises(ValidationError, match=message):
        validate_policy_decision(invalid_decision)


def test_receiving_boundary_rejects_undeclared_decision_copy_field() -> None:
    resource_value = reference("memory-1")
    decision = evaluate(
        principal(),
        "read",
        resource_value,
        policy(statement("read-grant", "allow", "read", resource_value)),
    ).model_copy(update={"future_constraint": "deny"})

    with pytest.raises(ValidationError):
        validate_policy_decision(decision)


def test_receiving_boundary_rejects_malformed_scalar_argument() -> None:
    with pytest.raises(ValidationError):
        validate_policy_decision("not-a-decision")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "malformed_extra",
    [[], (), "", 0, False, ["retained_unknown"]],
)
def test_receiving_boundary_rejects_malformed_extra_storage(
    malformed_extra: object,
) -> None:
    decision = PolicyDecision(**decision_payload())
    object.__setattr__(decision, "__pydantic_extra__", malformed_extra)

    with pytest.raises(
        ValueError,
        match="contract input extra storage must be a mapping",
    ):
        validate_policy_decision(decision)


@pytest.mark.parametrize("extra_storage", [None, {}, MappingProxyType({})])
def test_receiving_boundary_accepts_valid_empty_extra_storage(
    extra_storage: object,
) -> None:
    decision = PolicyDecision(**decision_payload())
    object.__setattr__(decision, "__pydantic_extra__", extra_storage)

    received = validate_policy_decision(decision)

    assert received.model_dump(mode="python") == decision.model_dump(mode="python")
    assert received is not decision


def test_receiving_boundary_rejects_required_field_moved_to_extra_storage() -> None:
    decision = PolicyDecision(**decision_payload())
    moved_value = decision.__dict__.pop("evaluated_subject_id")
    object.__setattr__(
        decision,
        "__pydantic_extra__",
        {"evaluated_subject_id": moved_value},
    )

    with pytest.raises(
        ValueError,
        match="contract input contains duplicate stored fields",
    ):
        validate_policy_decision(decision)


def test_receiving_boundary_still_rejects_required_field_absent_from_all_storage() -> None:
    decision = PolicyDecision(**decision_payload())
    decision.__dict__.pop("evaluated_subject_id")

    with pytest.raises(ValidationError):
        validate_policy_decision(decision)


def test_receiving_boundary_retains_nonconflicting_unknown_extra_for_rejection() -> None:
    decision = PolicyDecision(**decision_payload())
    object.__setattr__(
        decision,
        "__pydantic_extra__",
        {"future_constraint": "deny"},
    )

    with pytest.raises(ValidationError, match="future_constraint"):
        validate_policy_decision(decision)


@pytest.mark.parametrize("extra_action", ["read", "delete"])
def test_receiving_boundary_rejects_equal_or_conflicting_declared_extra_storage(
    extra_action: str,
) -> None:
    decision = PolicyDecision(**decision_payload())
    object.__setattr__(decision, "__pydantic_extra__", {"action": extra_action})

    with pytest.raises(
        ValueError,
        match="contract input contains duplicate stored fields",
    ):
        validate_policy_decision(decision)


def test_receiving_boundary_revalidates_nested_reference_copy() -> None:
    resource_value = reference("memory-1")
    decision = evaluate(
        principal(),
        "read",
        resource_value,
        policy(statement("read-grant", "allow", "read", resource_value)),
    )
    invalid_resource = decision.resource.model_copy(update={"tenant_id": ""})
    invalid_decision = decision.model_copy(update={"resource": invalid_resource})

    with pytest.raises(ValidationError):
        validate_policy_decision(invalid_decision)


def test_receiving_boundary_returns_independent_valid_snapshot() -> None:
    resource_value = reference("memory-1")
    decision = evaluate(
        principal(),
        "read",
        resource_value,
        policy(statement("read-grant", "allow", "read", resource_value)),
    )

    received = validate_policy_decision(decision)

    assert received == decision
    assert received is not decision
    assert received.resource is not decision.resource


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
