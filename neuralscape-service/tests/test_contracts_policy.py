"""Contract and adversarial tests for the pure reference policy evaluator."""

from __future__ import annotations

import gc
import json
import weakref
from collections.abc import Iterator, Mapping
from copy import deepcopy
from types import MappingProxyType

import pytest
from pydantic import BaseModel, ValidationError

from contracts_policy import (
    AccessPolicy,
    DelegationConstraints,
    MembershipVersion,
    PolicyDecision,
    PolicyEvaluationInput,
    PolicyStatement,
    PrincipalContext,
)
from contracts_policy_reference import (
    _CapturedFailure,
    _capture_model_inventory,
    _complete_contract_input,
    _freeze_mapping_entries,
    _freeze_mapping_inventories,
    _freeze_reachable_model_inventories,
    evaluate_policy,
    validate_policy_decision,
)
from contracts_references import ReferenceHandle


VERSION = "candidate-v1"
_BASE_MODEL_DICT_DESCRIPTOR = vars(BaseModel)["__dict__"]
_BASE_MODEL_EXTRA_DESCRIPTOR = vars(BaseModel)["__pydantic_extra__"]
_BASE_MODEL_FIELDS_SET_DESCRIPTOR = vars(BaseModel)["__pydantic_fields_set__"]


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


class _HiddenStoredViewDecision(PolicyDecision):
    """Hide real stored state from instance-level ``__dict__`` access."""

    def __getattribute__(self, name: str) -> object:
        if name == "__dict__":
            native = object.__getattribute__(self, "__dict__")
            visible = dict(dict.items(native))
            visible.pop("future_constraint", None)
            if visible.get("action") == "future-action":
                visible["action"] = "read"
            return visible
        return object.__getattribute__(self, name)


class _HiddenExtraViewDecision(PolicyDecision):
    """Hide real Pydantic extra storage from instance-level access."""

    def __getattribute__(self, name: str) -> object:
        if name == "__pydantic_extra__":
            return {}
        return object.__getattribute__(self, name)


class _HiddenFieldsSetViewDecision(PolicyDecision):
    """Hide real set-only state from instance-level fields-set access."""

    def __getattribute__(self, name: str) -> object:
        if name == "__pydantic_fields_set__":
            return set(type(self).model_fields)
        return object.__getattribute__(self, name)


class _HiddenFieldsSetViewReference(ReferenceHandle):
    """Nested contract hiding its real set-only state from instance access."""

    def __getattribute__(self, name: str) -> object:
        if name == "__pydantic_fields_set__":
            return set(type(self).model_fields)
        return object.__getattribute__(self, name)


class _HiddenIterationFieldsSet(set[str]):
    """A real set whose overridable views conceal its native backing."""

    def __init__(self, values: set[str]) -> None:
        set.__init__(self, values)
        self.view_calls = 0

    def __bool__(self) -> bool:
        self.view_calls += 1
        return False

    def __len__(self) -> int:
        self.view_calls += 1
        return 0

    def __iter__(self):  # type: ignore[no-untyped-def]
        self.view_calls += 1
        return iter(())


class _NestedRepairingFieldName(str):
    """Repair a nested model if model-owned name hashing is dispatched."""

    def __new__(
        cls,
        value: str,
        target: BaseModel | None,
    ) -> "_NestedRepairingFieldName":
        instance = super().__new__(cls, value)
        instance.target = target
        instance.armed = False
        instance.hash_calls = 0
        instance.equality_calls = 0
        return instance

    def _repair_or_reject_callback(self) -> None:
        if self.target is None:
            raise AssertionError("benign model-owned name callback was invoked")
        native = _BASE_MODEL_DICT_DESCRIPTOR.__get__(self.target, BaseModel)
        dict.__setitem__(native, "id", "memory-1")

    def __hash__(self) -> int:
        if self.armed:
            self.hash_calls += 1
            self._repair_or_reject_callback()
        return str.__hash__(self)

    def __eq__(self, other: object) -> bool:
        if self.armed:
            self.equality_calls += 1
            self._repair_or_reject_callback()
        return str.__eq__(self, other)

    def __str__(self) -> str:
        if self.armed:
            raise AssertionError("model-owned name used subclass string hook")
        return str.__str__(self)


class _NestedRepairingClassView:
    """Repair a nested model if instance-sensitive classification is used."""

    def __init__(self, target: BaseModel) -> None:
        self.target = target
        self.class_calls = 0

    @property
    def __class__(self):  # type: ignore[override]
        self.class_calls += 1
        native = _BASE_MODEL_DICT_DESCRIPTOR.__get__(self.target, BaseModel)
        dict.__setitem__(native, "id", "memory-1")
        return object


class _RepairingFieldsIterable:
    """Legacy fields-set iterable that can repair a referenced model."""

    def __init__(
        self,
        names: tuple[str, ...],
        target: BaseModel | None,
    ) -> None:
        self.names = names
        self.target = target
        self.iteration_calls = 0

    def __iter__(self):  # type: ignore[no-untyped-def]
        self.iteration_calls += 1
        if self.target is not None:
            native = _BASE_MODEL_DICT_DESCRIPTOR.__get__(self.target, BaseModel)
            dict.__setitem__(native, "id", "memory-1")
        return iter(self.names)


class _LengthAwareFieldsIterable:
    """Legacy fields-set iterable whose length view is not authoritative."""

    def __init__(
        self,
        names: tuple[str, ...],
        *,
        raise_on_length: bool,
    ) -> None:
        self.names = names
        self.raise_on_length = raise_on_length
        self.iteration_calls = 0
        self.length_calls = 0

    def __iter__(self):  # type: ignore[no-untyped-def]
        self.iteration_calls += 1
        return iter(self.names)

    def __len__(self) -> int:
        self.length_calls += 1
        if self.raise_on_length:
            raise RuntimeError("legacy fields-set length callback ran")
        return len(self.names)


class _RepairingEmptyExtras(Mapping[str, object]):
    """Supported empty Mapping whose items authority can repair a model."""

    def __init__(self, target: BaseModel | None) -> None:
        self.target = target
        self.items_calls = 0
        self.iteration_calls = 0
        self.getitem_calls = 0

    def __getitem__(self, key: str) -> object:
        self.getitem_calls += 1
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        self.iteration_calls += 1
        return iter(())

    def __len__(self) -> int:
        return 0

    def items(self):  # type: ignore[no-untyped-def]
        self.items_calls += 1
        if self.target is not None:
            native = _BASE_MODEL_DICT_DESCRIPTOR.__get__(self.target, BaseModel)
            dict.__setitem__(native, "id", "memory-1")
        return {}.items()


class _StoredDescriptorDecision(PolicyDecision):
    @property
    def __dict__(self):  # type: ignore[override]
        native = _BASE_MODEL_DICT_DESCRIPTOR.__get__(self, BaseModel)
        visible = dict(dict.items(native))
        visible.pop("future_constraint", None)
        return visible

    @__dict__.setter
    def __dict__(self, value):  # type: ignore[no-untyped-def]
        _BASE_MODEL_DICT_DESCRIPTOR.__set__(self, value)


class _ExtraDescriptorDecision(PolicyDecision):
    @property
    def __pydantic_extra__(self):  # type: ignore[override]
        return {}

    @__pydantic_extra__.setter
    def __pydantic_extra__(self, value):  # type: ignore[no-untyped-def]
        _BASE_MODEL_EXTRA_DESCRIPTOR.__set__(self, value)


class _FieldsSetDescriptorDecision(PolicyDecision):
    @property
    def __pydantic_fields_set__(self):  # type: ignore[override]
        return set(type(self).model_fields)

    @__pydantic_fields_set__.setter
    def __pydantic_fields_set__(self, value):  # type: ignore[no-untyped-def]
        _BASE_MODEL_FIELDS_SET_DESCRIPTOR.__set__(self, value)


class _StoredDescriptorReference(ReferenceHandle):
    @property
    def __dict__(self):  # type: ignore[override]
        native = _BASE_MODEL_DICT_DESCRIPTOR.__get__(self, BaseModel)
        visible = dict(dict.items(native))
        visible.pop("future_constraint", None)
        return visible

    @__dict__.setter
    def __dict__(self, value):  # type: ignore[no-untyped-def]
        _BASE_MODEL_DICT_DESCRIPTOR.__set__(self, value)


class _ExtraDescriptorReference(ReferenceHandle):
    @property
    def __pydantic_extra__(self):  # type: ignore[override]
        return {}

    @__pydantic_extra__.setter
    def __pydantic_extra__(self, value):  # type: ignore[no-untyped-def]
        _BASE_MODEL_EXTRA_DESCRIPTOR.__set__(self, value)


class _FieldsSetDescriptorReference(ReferenceHandle):
    @property
    def __pydantic_fields_set__(self):  # type: ignore[override]
        return set(type(self).model_fields)

    @__pydantic_fields_set__.setter
    def __pydantic_fields_set__(self, value):  # type: ignore[no-untyped-def]
        _BASE_MODEL_FIELDS_SET_DESCRIPTOR.__set__(self, value)


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


@pytest.mark.parametrize(
    "decision_type",
    [PolicyDecision, _HiddenStoredViewDecision],
    ids=["ordinary", "overridden-dict-view"],
)
def test_receiving_boundary_reads_native_stored_unknown_state(
    decision_type: type[PolicyDecision],
) -> None:
    decision = decision_type(**decision_payload())
    native_storage = object.__getattribute__(decision, "__dict__")
    dict.__setitem__(native_storage, "future_constraint", "deny")

    assert ("future_constraint", "deny") in tuple(dict.items(native_storage))
    if decision_type is _HiddenStoredViewDecision:
        assert "future_constraint" not in decision.__dict__

    with pytest.raises(ValidationError, match="future_constraint"):
        validate_policy_decision(decision)


@pytest.mark.parametrize(
    "decision_type",
    [PolicyDecision, _HiddenStoredViewDecision],
    ids=["ordinary", "overridden-dict-view"],
)
def test_receiving_boundary_reads_native_invalid_declared_state(
    decision_type: type[PolicyDecision],
) -> None:
    decision = decision_type(**decision_payload())
    native_storage = object.__getattribute__(decision, "__dict__")
    dict.__setitem__(native_storage, "action", "future-action")

    assert dict.__getitem__(native_storage, "action") == "future-action"
    if decision_type is _HiddenStoredViewDecision:
        assert decision.__dict__["action"] == "read"

    with pytest.raises(
        ValidationError,
        match="allow decisions require a supported action",
    ):
        validate_policy_decision(decision)


@pytest.mark.parametrize(
    "decision_type",
    [PolicyDecision, _HiddenExtraViewDecision],
    ids=["ordinary", "overridden-extra-view"],
)
def test_receiving_boundary_reads_native_unknown_extra_storage(
    decision_type: type[PolicyDecision],
) -> None:
    decision = decision_type(**decision_payload())
    object.__setattr__(
        decision,
        "__pydantic_extra__",
        {"future_constraint": "deny"},
    )
    native_extra = object.__getattribute__(decision, "__pydantic_extra__")

    assert tuple(dict.items(native_extra)) == (("future_constraint", "deny"),)
    if decision_type is _HiddenExtraViewDecision:
        assert decision.__pydantic_extra__ == {}

    with pytest.raises(ValidationError, match="future_constraint"):
        validate_policy_decision(decision)


@pytest.mark.parametrize("location", ["decision", "nested-reference"])
@pytest.mark.parametrize(
    "storage_kind",
    ["ordinary", "model-override", "native-set-subclass"],
)
def test_receiving_boundary_reads_native_unknown_fields_set_storage(
    location: str,
    storage_kind: str,
) -> None:
    if location == "decision":
        decision_type = (
            _HiddenFieldsSetViewDecision
            if storage_kind == "model-override"
            else PolicyDecision
        )
        decision = decision_type(**decision_payload())
        target: BaseModel = decision
        expected_location = ("future_constraint",)
    else:
        decision = PolicyDecision(**decision_payload())
        reference_type = (
            _HiddenFieldsSetViewReference
            if storage_kind == "model-override"
            else ReferenceHandle
        )
        target = reference_type(**decision.resource.model_dump(mode="python"))
        decision.__dict__["resource"] = target
        expected_location = ("resource", "future_constraint")

    native_fields_set = object.__getattribute__(
        target,
        "__pydantic_fields_set__",
    )
    if storage_kind == "native-set-subclass":
        native_fields_set = _HiddenIterationFieldsSet(set(native_fields_set))
        object.__setattr__(
            target,
            "__pydantic_fields_set__",
            native_fields_set,
        )
    set.add(native_fields_set, "future_constraint")

    assert "future_constraint" in tuple(set.__iter__(native_fields_set))
    assert "future_constraint" not in object.__getattribute__(target, "__dict__")
    assert object.__getattribute__(target, "__pydantic_extra__") is None
    if storage_kind == "model-override":
        assert "future_constraint" not in target.__pydantic_fields_set__

    with pytest.raises(ValidationError) as raised:
        validate_policy_decision(decision)

    assert raised.value.errors()[0]["type"] == "extra_forbidden"
    assert raised.value.errors()[0]["loc"] == expected_location
    assert raised.value.errors()[0]["input"] is None
    if storage_kind == "native-set-subclass":
        assert native_fields_set.view_calls == 0


def _nested_model_name_case(
    boundary: str,
    *,
    invalid: bool,
):  # type: ignore[no-untyped-def]
    resource_value = reference("memory-1")
    nested = (
        resource_value.model_copy(update={"id": ""})
        if invalid
        else resource_value
    )
    if boundary == "evaluator":
        candidate = PolicyEvaluationInput(
            schema_version=VERSION,
            action="read",
            resource=resource_value,
        ).model_copy(update={"resource": nested})
        policy_value = policy(
            statement("read-grant", "allow", "read", resource_value)
        )

        def invoke(value):  # type: ignore[no-untyped-def]
            return evaluate_policy(
                principal=principal(),
                evaluation=value,
                policy=policy_value,
            )

    else:
        candidate = PolicyDecision(**decision_payload()).model_copy(
            update={"resource": nested}
        )

        def invoke(value):  # type: ignore[no-untyped-def]
            return validate_policy_decision(value)

    return candidate, nested, invoke


def _install_model_name_subclass(
    model: BaseModel,
    nested: BaseModel | None,
    storage_name: str,
) -> _NestedRepairingFieldName:
    name = _NestedRepairingFieldName("resource", nested)
    _replace_model_owned_name(model, name, storage_name)
    name.armed = True
    name.hash_calls = 0
    name.equality_calls = 0
    return name


def _replace_model_owned_name(
    model: BaseModel,
    name: object,
    storage_name: str,
) -> None:
    if storage_name == "stored":
        storage = _BASE_MODEL_DICT_DESCRIPTOR.__get__(model, BaseModel)
        item = dict.__getitem__(storage, "resource")
        dict.__delitem__(storage, "resource")
        dict.__setitem__(storage, name, item)
    else:
        fields_set = _BASE_MODEL_FIELDS_SET_DESCRIPTOR.__get__(model, BaseModel)
        set.discard(fields_set, "resource")
        set.add(fields_set, name)


@pytest.mark.parametrize("boundary", ["evaluator", "receiving"])
@pytest.mark.parametrize("storage_name", ["stored", "fields-set"])
def test_policy_boundaries_snapshot_nested_models_before_name_hashing(
    boundary: str,
    storage_name: str,
) -> None:
    ordinary, ordinary_nested, ordinary_invoke = _nested_model_name_case(
        boundary,
        invalid=True,
    )
    with pytest.raises(ValidationError) as ordinary_error:
        ordinary_invoke(ordinary)

    attacked, attacked_nested, attacked_invoke = _nested_model_name_case(
        boundary,
        invalid=True,
    )
    hostile_name = _install_model_name_subclass(
        attacked,
        attacked_nested,
        storage_name,
    )
    with pytest.raises(ValidationError) as attacked_error:
        attacked_invoke(attacked)

    ordinary_diagnostics = [
        (error["loc"], error["type"])
        for error in ordinary_error.value.errors()
    ]
    attacked_diagnostics = [
        (error["loc"], error["type"])
        for error in attacked_error.value.errors()
    ]
    assert attacked_diagnostics == ordinary_diagnostics
    assert ordinary_nested.id == ""
    assert attacked_nested.id == ""
    assert hostile_name.hash_calls == 0
    assert hostile_name.equality_calls == 0


@pytest.mark.parametrize("boundary", ["evaluator", "receiving"])
@pytest.mark.parametrize("storage_name", ["stored", "fields-set"])
def test_policy_boundaries_accept_benign_string_subclass_model_names(
    boundary: str,
    storage_name: str,
) -> None:
    candidate, _, invoke = _nested_model_name_case(boundary, invalid=False)
    benign_name = _install_model_name_subclass(candidate, None, storage_name)

    result = invoke(candidate)

    assert result.outcome == "allow"
    assert benign_name.hash_calls == 0
    assert benign_name.equality_calls == 0


@pytest.mark.parametrize("boundary", ["evaluator", "receiving"])
@pytest.mark.parametrize("storage_name", ["stored", "fields-set"])
def test_policy_boundaries_classify_malformed_model_names_concretely(
    boundary: str,
    storage_name: str,
) -> None:
    ordinary, ordinary_nested, ordinary_invoke = _nested_model_name_case(
        boundary,
        invalid=True,
    )
    _replace_model_owned_name(ordinary, object(), storage_name)
    with pytest.raises(ValueError) as ordinary_error:
        ordinary_invoke(ordinary)

    attacked, attacked_nested, attacked_invoke = _nested_model_name_case(
        boundary,
        invalid=True,
    )
    hostile_name = _NestedRepairingClassView(attacked_nested)
    _replace_model_owned_name(attacked, hostile_name, storage_name)
    with pytest.raises(ValueError) as attacked_error:
        attacked_invoke(attacked)

    assert str(attacked_error.value) == str(ordinary_error.value)
    assert str(attacked_error.value) == (
        "contract input model field names must be strings"
    )
    assert ordinary_nested.id == ""
    assert attacked_nested.id == ""
    assert hostile_name.class_calls == 0


@pytest.mark.parametrize("boundary", ["evaluator", "receiving"])
def test_policy_boundaries_classify_graph_values_concretely(
    boundary: str,
) -> None:
    ordinary, ordinary_nested, ordinary_invoke = _nested_model_name_case(
        boundary,
        invalid=True,
    )
    ordinary_storage = _BASE_MODEL_DICT_DESCRIPTOR.__get__(ordinary, BaseModel)
    dict.__setitem__(ordinary_storage, "action", object())
    with pytest.raises(ValidationError) as ordinary_error:
        ordinary_invoke(ordinary)

    attacked, attacked_nested, attacked_invoke = _nested_model_name_case(
        boundary,
        invalid=True,
    )
    hostile_value = _NestedRepairingClassView(attacked_nested)
    attacked_storage = _BASE_MODEL_DICT_DESCRIPTOR.__get__(attacked, BaseModel)
    dict.__setitem__(attacked_storage, "action", hostile_value)
    with pytest.raises(ValidationError) as attacked_error:
        attacked_invoke(attacked)

    ordinary_diagnostics = [
        (error["loc"], error["type"])
        for error in ordinary_error.value.errors()
    ]
    attacked_diagnostics = [
        (error["loc"], error["type"])
        for error in attacked_error.value.errors()
    ]
    assert attacked_diagnostics == ordinary_diagnostics
    assert ordinary_diagnostics == [
        (("action",), "string_type"),
        (("resource", "id"), "string_too_short"),
    ]
    assert ordinary_nested.id == ""
    assert attacked_nested.id == ""
    assert hostile_value.class_calls == 0


def _install_repairing_protocol(
    model: BaseModel,
    target: BaseModel | None,
    protocol: str,
) -> _RepairingFieldsIterable | _RepairingEmptyExtras:
    if protocol == "fields-iterable":
        hook = _RepairingFieldsIterable(tuple(type(model).model_fields), target)
        _BASE_MODEL_FIELDS_SET_DESCRIPTOR.__set__(model, hook)
        assert _BASE_MODEL_FIELDS_SET_DESCRIPTOR.__get__(model, BaseModel) is hook
        return hook
    hook = _RepairingEmptyExtras(target)
    _BASE_MODEL_EXTRA_DESCRIPTOR.__set__(model, hook)
    assert _BASE_MODEL_EXTRA_DESCRIPTOR.__get__(model, BaseModel) is hook
    return hook


def _assert_protocol_called_once(
    hook: _RepairingFieldsIterable | _RepairingEmptyExtras,
) -> None:
    if type(hook) is _RepairingFieldsIterable:
        assert hook.iteration_calls == 1
        return
    assert hook.items_calls == 1
    assert hook.iteration_calls == 0
    assert hook.getitem_calls == 0


def test_model_inventory_retains_owner_only_for_cache_lifetime() -> None:
    candidate = reference("memory-1")
    candidate_reference = weakref.ref(candidate)
    candidate_id = id(candidate)
    inventories = {
        candidate_id: _capture_model_inventory(candidate),
    }

    del candidate
    gc.collect()
    assert candidate_reference() is inventories[candidate_id].owner

    inventories.clear()
    gc.collect()
    assert candidate_reference() is None


@pytest.mark.parametrize("consumer", ["freeze", "completion"])
@pytest.mark.parametrize("owner_matches", [False, True])
def test_model_inventory_consumers_require_exact_owner_identity(
    consumer: str,
    owner_matches: bool,
) -> None:
    candidate = reference("memory-1")
    owner = candidate if owner_matches else reference("memory-2")
    inventories = {
        id(candidate): _capture_model_inventory(owner),
    }

    if not owner_matches:
        with pytest.raises(
            ValueError,
            match="contract input model inventory owner mismatch",
        ):
            if consumer == "freeze":
                _freeze_reachable_model_inventories(
                    candidate,
                    inventories,
                    {},
                )
            else:
                _complete_contract_input(candidate, set(), inventories)
        return

    if consumer == "freeze":
        assert (
            _freeze_reachable_model_inventories(
                candidate,
                inventories,
                {},
            )
            is None
        )
    else:
        completed = _complete_contract_input(candidate, set(), inventories)
        assert isinstance(completed, dict)
        assert completed["id"] == "memory-1"


def test_mapping_inventory_retains_owner_entries_and_exposed_models() -> None:
    original = reference("memory-1")
    backing = {"resource": original}
    candidate = MappingProxyType(backing)
    model_inventories = {}
    mapping_owners = {}
    mapping_inventories = {}
    visited_containers: dict[int, int] = {}

    _freeze_reachable_model_inventories(
        candidate,
        model_inventories,
        visited_containers,
        mapping_owners,
    )
    assert mapping_owners[id(candidate)] is candidate

    _freeze_mapping_inventories(
        mapping_owners,
        mapping_inventories,
        model_inventories,
        visited_containers,
    )
    retained = mapping_inventories[id(candidate)]
    assert retained.owner is candidate
    assert id(original) in model_inventories

    backing["resource"] = reference("memory-2")
    completed = _complete_contract_input(
        candidate,
        set(),
        model_inventories,
        mapping_inventories,
    )

    assert completed == {
        "resource": original.model_dump(mode="python"),
    }


def test_model_extra_mapping_reuses_inventory_after_direct_edit() -> None:
    backing: dict[str, object] = {}
    extras = MappingProxyType(backing)
    candidate = PolicyDecision(**decision_payload())
    object.__setattr__(candidate, "__pydantic_extra__", extras)
    model_inventories = {}
    mapping_owners = {}
    mapping_inventories = {}
    visited_containers: dict[int, int] = {}

    _freeze_reachable_model_inventories(
        candidate,
        model_inventories,
        visited_containers,
        mapping_owners,
    )
    _freeze_mapping_inventories(
        mapping_owners,
        mapping_inventories,
        model_inventories,
        visited_containers,
    )
    assert mapping_inventories[id(extras)].owner is extras

    backing["future_constraint"] = "deny"
    completed = _complete_contract_input(
        candidate,
        set(),
        model_inventories,
        mapping_inventories,
    )

    assert PolicyDecision.model_validate(completed).model_dump(mode="python") == (
        decision_payload()
    )


def test_mapping_inventory_consumer_requires_exact_owner_identity() -> None:
    candidate = MappingProxyType({"resource": reference("memory-1")})
    other = MappingProxyType({"resource": reference("memory-2")})
    other_owners = {id(other): other}
    mapping_inventories = {}
    _freeze_mapping_inventories(other_owners, mapping_inventories, {}, {})
    mismatched_inventories = {
        id(candidate): mapping_inventories[id(other)],
    }

    with pytest.raises(
        ValueError,
        match="contract input mapping inventory owner mismatch",
    ):
        _complete_contract_input(
            candidate,
            set(),
            {},
            mismatched_inventories,
        )


def test_mapping_entries_stop_at_first_terminal_pair_failure() -> None:
    retained = reference("memory-1")
    unreachable = reference("memory-2")
    model_inventories = {}

    entries = _freeze_mapping_entries(
        (
            ("resource", retained),
            ("terminal",),
            ("unreachable", unreachable),
        ),
        {},
        model_inventories,
        {},
    )

    assert entries[0] == ("resource", retained)
    assert type(entries[1]) is _CapturedFailure
    assert isinstance(entries[1].error, ValueError)
    assert len(entries) == 2
    assert id(retained) in model_inventories
    assert id(unreachable) not in model_inventories


@pytest.mark.parametrize("boundary", ["evaluator", "receiving"])
@pytest.mark.parametrize("raise_on_length", [False, True])
def test_legacy_fields_set_fallback_uses_iteration_without_length_hint(
    boundary: str,
    raise_on_length: bool,
) -> None:
    candidate, _, invoke = _nested_model_name_case(boundary, invalid=False)
    fields = _LengthAwareFieldsIterable(
        tuple(type(candidate).model_fields),
        raise_on_length=raise_on_length,
    )
    _BASE_MODEL_FIELDS_SET_DESCRIPTOR.__set__(candidate, fields)

    result = invoke(candidate)

    assert result.outcome == "allow"
    assert fields.iteration_calls == 1
    assert fields.length_calls == 0


@pytest.mark.parametrize("boundary", ["evaluator", "receiving"])
@pytest.mark.parametrize("protocol", ["fields-iterable", "extra-mapping"])
@pytest.mark.parametrize("location", ["root", "nested"])
def test_policy_boundaries_freeze_models_before_supported_callbacks(
    boundary: str,
    protocol: str,
    location: str,
) -> None:
    ordinary, ordinary_nested, ordinary_invoke = _nested_model_name_case(
        boundary,
        invalid=True,
    )
    with pytest.raises(ValidationError) as ordinary_error:
        ordinary_invoke(ordinary)

    attacked, attacked_nested, attacked_invoke = _nested_model_name_case(
        boundary,
        invalid=True,
    )
    callback_owner = attacked if location == "root" else attacked_nested
    hook = _install_repairing_protocol(
        callback_owner,
        attacked_nested,
        protocol,
    )
    native = _BASE_MODEL_DICT_DESCRIPTOR.__get__(attacked_nested, BaseModel)
    assert dict.__getitem__(native, "id") == ""

    with pytest.raises(ValidationError) as attacked_error:
        attacked_invoke(attacked)

    ordinary_diagnostics = [
        (error["loc"], error["type"])
        for error in ordinary_error.value.errors()
    ]
    attacked_diagnostics = [
        (error["loc"], error["type"])
        for error in attacked_error.value.errors()
    ]
    assert attacked_diagnostics == ordinary_diagnostics
    assert ordinary_nested.id == ""
    assert dict.__getitem__(native, "id") == "memory-1"
    _assert_protocol_called_once(hook)


@pytest.mark.parametrize("boundary", ["evaluator", "receiving"])
@pytest.mark.parametrize("protocol", ["fields-iterable", "extra-mapping"])
@pytest.mark.parametrize("location", ["root", "nested"])
def test_policy_boundaries_preserve_valid_supported_callback_controls(
    boundary: str,
    protocol: str,
    location: str,
) -> None:
    candidate, nested, invoke = _nested_model_name_case(boundary, invalid=False)
    callback_owner = candidate if location == "root" else nested
    hook = _install_repairing_protocol(callback_owner, nested, protocol)

    result = invoke(candidate)

    assert result.outcome == "allow"
    assert result.resource.id == "memory-1"
    assert nested.id == "memory-1"
    _assert_protocol_called_once(hook)


@pytest.mark.parametrize("protocol", ["fields-iterable", "extra-mapping"])
def test_evaluator_freezes_all_input_graphs_before_supported_callbacks(
    protocol: str,
) -> None:
    evaluation, nested, _ = _nested_model_name_case("evaluator", invalid=True)
    principal_value = principal()
    hook = _install_repairing_protocol(principal_value, nested, protocol)
    native = _BASE_MODEL_DICT_DESCRIPTOR.__get__(nested, BaseModel)
    assert dict.__getitem__(native, "id") == ""

    with pytest.raises(ValidationError) as raised:
        evaluate_policy(
            principal=principal_value,
            evaluation=evaluation,
            policy=policy(
                statement("read-grant", "allow", "read", reference("memory-1"))
            ),
        )

    assert raised.value.errors()[0]["loc"] == ("resource", "id")
    assert raised.value.errors()[0]["type"] == "string_too_short"
    assert dict.__getitem__(native, "id") == "memory-1"
    _assert_protocol_called_once(hook)


def test_prefreeze_preserves_overlap_before_nested_storage_failure() -> None:
    nested = reference("memory-1")
    _BASE_MODEL_FIELDS_SET_DESCRIPTOR.__delete__(nested)
    decision = PolicyDecision(**decision_payload()).model_copy(
        update={"resource": nested}
    )
    _BASE_MODEL_EXTRA_DESCRIPTOR.__set__(decision, {"action": "delete"})

    with pytest.raises(
        ValueError,
        match="contract input contains duplicate stored fields",
    ):
        validate_policy_decision(decision)


@pytest.mark.parametrize("location", ["decision", "nested-reference"])
@pytest.mark.parametrize("storage_name", ["stored", "extra", "fields-set"])
def test_receiving_boundary_bypasses_subclass_storage_descriptors(
    location: str,
    storage_name: str,
) -> None:
    decision_types = {
        "stored": _StoredDescriptorDecision,
        "extra": _ExtraDescriptorDecision,
        "fields-set": _FieldsSetDescriptorDecision,
    }
    reference_types = {
        "stored": _StoredDescriptorReference,
        "extra": _ExtraDescriptorReference,
        "fields-set": _FieldsSetDescriptorReference,
    }
    if location == "decision":
        decision = decision_types[storage_name](**decision_payload())
        target: BaseModel = decision
        expected_location = ("future_constraint",)
    else:
        decision = PolicyDecision(**decision_payload())
        target = reference_types[storage_name](
            **decision.resource.model_dump(mode="python")
        )
        decision.__dict__["resource"] = target
        expected_location = ("resource", "future_constraint")

    normal = validate_policy_decision(decision)
    assert normal.action == "read"

    if storage_name == "stored":
        native_storage = _BASE_MODEL_DICT_DESCRIPTOR.__get__(target, BaseModel)
        dict.__setitem__(native_storage, "future_constraint", "deny")
        assert ("future_constraint", "deny") in tuple(
            dict.items(native_storage)
        )
        assert "future_constraint" not in target.__dict__
        expected_input: object = "deny"
    elif storage_name == "extra":
        _BASE_MODEL_EXTRA_DESCRIPTOR.__set__(
            target,
            {"future_constraint": "deny"},
        )
        native_storage = _BASE_MODEL_EXTRA_DESCRIPTOR.__get__(target, BaseModel)
        assert tuple(dict.items(native_storage)) == (
            ("future_constraint", "deny"),
        )
        assert target.__pydantic_extra__ == {}
        expected_input = "deny"
    else:
        native_storage = _BASE_MODEL_FIELDS_SET_DESCRIPTOR.__get__(
            target,
            BaseModel,
        )
        set.add(native_storage, "future_constraint")
        assert "future_constraint" in tuple(set.__iter__(native_storage))
        assert "future_constraint" not in target.__pydantic_fields_set__
        expected_input = None

    with pytest.raises(ValidationError) as raised:
        validate_policy_decision(decision)

    assert raised.value.errors()[0]["type"] == "extra_forbidden"
    assert raised.value.errors()[0]["loc"] == expected_location
    assert raised.value.errors()[0]["input"] == expected_input


@pytest.mark.parametrize("location", ["decision", "nested-reference"])
def test_receiving_boundary_descriptor_snapshot_preserves_missing_subclass_default(
    location: str,
) -> None:
    if location == "decision":
        class DefaultedDecision(_StoredDescriptorDecision):
            audit_marker: str = "default-marker"

        decision = DefaultedDecision(**decision_payload())
        native_storage = _BASE_MODEL_DICT_DESCRIPTOR.__get__(decision, BaseModel)
    else:
        class DefaultedReference(_StoredDescriptorReference):
            audit_marker: str = "default-marker"

        decision = PolicyDecision(**decision_payload())
        target = DefaultedReference(**decision.resource.model_dump(mode="python"))
        decision.__dict__["resource"] = target
        native_storage = _BASE_MODEL_DICT_DESCRIPTOR.__get__(target, BaseModel)
    dict.pop(native_storage, "audit_marker", None)

    received = validate_policy_decision(decision)

    assert received.action == "read"


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


class _DivergentStatements(tuple):
    """A tuple whose public iterator disagrees with its native backing.

    These public-first tests pin the evaluator's own snapshot traversal: the
    tuple subclass is injected after validation, and the evaluator converts its
    public iterator into a plain tuple before model validation runs.  They
    therefore do not exercise pydantic-core's tuple-subclass handling and
    provide no coverage against a dependency upgrade changing it.
    """

    def __new__(cls, native, public_views):
        instance = super().__new__(cls, native)
        instance.public_views = [tuple(view) for view in public_views]
        instance.iteration_calls = 0
        return instance

    def __iter__(self):
        self.iteration_calls += 1
        index = min(self.iteration_calls, len(self.public_views)) - 1
        return iter(self.public_views[index])


def _evaluate_with_statements(statements: tuple[PolicyStatement, ...]) -> PolicyDecision:
    resource_value = reference("memory-1")
    policy_value = policy(statement("seed", "allow", "read", resource_value))
    object.__setattr__(policy_value, "statements", statements)
    return evaluate(principal(), "read", resource_value, policy_value)


def test_public_first_hidden_native_deny_is_not_part_of_the_snapshot() -> None:
    resource_value = reference("memory-1")
    allow = statement("allow-read", "allow", "read", resource_value)
    deny = statement("deny-read", "deny", "read", resource_value)
    statements = _DivergentStatements([allow, deny], [[allow]])

    received = _evaluate_with_statements(statements)

    assert (received.outcome, received.reason_code) == ("allow", "explicit_grant")


def test_public_first_public_only_deny_is_authoritative() -> None:
    resource_value = reference("memory-1")
    allow = statement("allow-read", "allow", "read", resource_value)
    deny = statement("deny-read", "deny", "read", resource_value)
    statements = _DivergentStatements([allow], [[allow, deny]])

    received = _evaluate_with_statements(statements)

    assert (received.outcome, received.reason_code) == ("deny", "not_authorized")


@pytest.mark.parametrize("first_view_denies", [False, True])
def test_public_first_view_is_captured_once_and_later_views_are_not_seen(
    first_view_denies: bool,
) -> None:
    resource_value = reference("memory-1")
    allow = statement("allow-read", "allow", "read", resource_value)
    deny = statement("deny-read", "deny", "read", resource_value)
    first, later = ((allow, deny), (allow,)) if first_view_denies else ((allow,), (allow, deny))
    statements = _DivergentStatements([allow, deny], [first, later])

    received = _evaluate_with_statements(statements)

    # The call count is asserted only to prove "captured once at first
    # observation"; it is a pin on that claim, not a general callback contract.
    assert statements.iteration_calls == 1
    assert received.outcome == ("deny" if first_view_denies else "allow")


def _policy_with_membership_view(native, public) -> AccessPolicy:
    policy_value = policy(statement("seed", "allow", "read", reference("memory-1")))
    object.__setattr__(
        policy_value,
        "membership_versions",
        _DivergentStatements(native, [public]),
    )
    return policy_value


@pytest.mark.parametrize(
    "public_version,expected",
    [(7, "allow"), (1, "deny")],
)
def test_public_first_membership_versions_follow_the_public_view(
    public_version: int,
    expected: str,
) -> None:
    current = MembershipVersion(subject_id="subject-a", version=7)
    stale = MembershipVersion(subject_id="subject-a", version=1)
    public = current if public_version == 7 else stale
    native = stale if public_version == 7 else current
    policy_value = _policy_with_membership_view((native,), (public,))

    received = evaluate(principal(), "read", reference("memory-1"), policy_value)

    assert received.outcome == expected


@pytest.mark.parametrize("field", ["allowed_actions", "allowed_resources"])
def test_public_first_delegation_ceiling_follows_the_public_view(field: str) -> None:
    resource_value = reference("memory-1")
    granted = {"allowed_actions": ("read",), "allowed_resources": (resource_value,)}
    open_ceiling = DelegationConstraints(
        delegated_by_subject_id="subject-human",
        delegated_by_credential_id="credential-human",
        **granted,
    )
    policy_value = policy(statement("grant", "allow", "read", resource_value))

    # Native backing grants, public view is empty: the empty public view wins.
    narrowed = principal(subject_kind="delegated_agent", delegation=open_ceiling)
    object.__setattr__(
        narrowed.delegation, field, _DivergentStatements(granted[field], [()])
    )
    assert evaluate(narrowed, "read", resource_value, policy_value).outcome == "deny"

    # Native backing is empty, public view grants: the public view wins, and it
    # still needs the policy statement above to produce an allow.
    empty = {"allowed_actions": (), "allowed_resources": ()}
    closed_ceiling = DelegationConstraints(
        delegated_by_subject_id="subject-human",
        delegated_by_credential_id="credential-human",
        **{**granted, field: ()},
    )
    widened = principal(subject_kind="delegated_agent", delegation=closed_ceiling)
    object.__setattr__(
        widened.delegation, field, _DivergentStatements(empty[field], [granted[field]])
    )
    assert evaluate(widened, "read", resource_value, policy_value).outcome == "allow"
    assert evaluate(widened, "read", resource_value, policy()).outcome == "deny"


# ── first-observation retention of native dict entries and of iterator children ──


class _CallbackItemsExtras(Mapping[str, object]):
    """A public mapping whose ``items()`` runs a callback and exposes nothing."""

    def __init__(self, callback) -> None:
        self._callback = callback

    def __getitem__(self, key: str) -> object:
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return iter(())

    def __len__(self) -> int:
        return 0

    def items(self):  # type: ignore[override]
        self._callback()
        return {}.items()


def _raw_statement(identifier: str, effect: str, resource_value: ReferenceHandle) -> dict[str, object]:
    return {
        "id": identifier,
        "effect": effect,
        "subject_id": "subject-a",
        "credential_id": None,
        "action": "read",
        "resource": resource_value.model_dump(mode="python"),
    }


def test_native_dict_entries_are_replayed_not_reread_after_callbacks() -> None:
    """A deny captured at the callback-free entry pass cannot be replaced by an allow."""

    resource_value = reference("memory-1")
    allow = statement("allow-read", "allow", "read", resource_value)
    deny = _raw_statement("deny-read", "deny", resource_value)
    policy_value = policy(allow)
    object.__setattr__(policy_value, "statements", (allow, deny))
    principal_value = principal()
    object.__setattr__(
        principal_value,
        "__pydantic_extra__",
        _CallbackItemsExtras(lambda: deny.__setitem__("effect", "allow")),
    )

    received = evaluate(principal_value, "read", resource_value, policy_value)

    assert (received.outcome, received.reason_code) == ("deny", "not_authorized")
    assert deny["effect"] == "allow"  # the callback ran; its effect was simply not observed


def test_native_dict_entries_captured_first_stay_invalid_when_repaired_later() -> None:
    resource_value = reference("memory-1")
    broken = {"kind": "memory", "id": "memory-1", "tenant_id": 5, "resolver": "memory_by_id"}
    evaluation = PolicyEvaluationInput(schema_version=VERSION, action="read", resource=resource_value)
    object.__setattr__(evaluation, "resource", broken)
    principal_value = principal()
    object.__setattr__(
        principal_value,
        "__pydantic_extra__",
        _CallbackItemsExtras(lambda: broken.__setitem__("tenant_id", "tenant-a")),
    )

    with pytest.raises(ValidationError):
        evaluate_policy(
            principal=principal_value,
            evaluation=evaluation,
            policy=policy(statement("allow-read", "allow", "read", resource_value)),
        )


def test_native_dict_without_callbacks_is_unchanged_by_retention() -> None:
    resource_value = reference("memory-1")
    allow = _raw_statement("allow-read", "allow", resource_value)
    policy_value = policy(statement("seed", "allow", "read", reference("other")))
    object.__setattr__(policy_value, "statements", (allow,))

    assert evaluate(principal(), "read", resource_value, policy_value).outcome == "allow"


def _delegated_principal(delegation_mapping) -> PrincipalContext:
    ceiling = DelegationConstraints(
        delegated_by_subject_id="subject-human",
        delegated_by_credential_id="credential-human",
        allowed_actions=("read",),
        allowed_resources=(reference("memory-1"),),
    )
    value = principal(subject_kind="delegated_agent", delegation=ceiling)
    object.__setattr__(value, "delegation", delegation_mapping)
    return value


class _GeneratorMapping(Mapping[str, object]):
    """A public mapping whose pair iterator is a caller-supplied generator."""

    def __init__(self, generator_factory) -> None:
        self._factory = generator_factory

    def __getitem__(self, key: str) -> object:
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return iter(())

    def __len__(self) -> int:
        return 0

    def items(self):  # type: ignore[override]
        return self._factory()


def _delegation_pairs(handle, *, repair: bool):
    yield ("delegated_by_subject_id", "subject-human")
    yield ("delegated_by_credential_id", "credential-human")
    yield ("allowed_actions", ("read",))
    yield ("allowed_resources", (handle,))
    if repair:
        # Resumed only when the consumer advances: repair the child already exposed.
        handle.__dict__["tenant_id"] = "tenant-a"


def _tampered_handle() -> ReferenceHandle:
    handle = reference("memory-1")
    handle.__dict__["tenant_id"] = 5  # invalid; bypasses the frozen model
    return handle


def test_iterator_children_are_frozen_before_the_iterator_advances() -> None:
    resource_value = reference("memory-1")
    handle = _tampered_handle()
    mapping = _GeneratorMapping(lambda: _delegation_pairs(handle, repair=True))
    principal_value = _delegated_principal(mapping)
    grant = policy(statement("allow-read", "allow", "read", resource_value))

    with pytest.raises(ValidationError):
        evaluate(principal_value, "read", resource_value, grant)


def test_the_same_generator_without_repair_is_rejected_and_a_valid_one_is_allowed() -> None:
    resource_value = reference("memory-1")
    grant = policy(statement("allow-read", "allow", "read", resource_value))
    broken = _tampered_handle()
    with pytest.raises(ValidationError):
        evaluate(_delegated_principal(_GeneratorMapping(lambda: _delegation_pairs(broken, repair=False))),
                 "read", resource_value, grant)
    valid = reference("memory-1")
    received = evaluate(_delegated_principal(_GeneratorMapping(lambda: _delegation_pairs(valid, repair=False))),
                        "read", resource_value, grant)
    assert received.outcome == "allow"


def test_iterator_failures_stay_deferred_and_keep_their_class() -> None:
    resource_value = reference("memory-1")
    grant = policy(statement("allow-read", "allow", "read", resource_value))

    def failing():
        yield ("delegated_by_subject_id", "subject-human")
        raise RuntimeError("iteration failed")

    with pytest.raises(RuntimeError, match="iteration failed"):
        evaluate(_delegated_principal(_GeneratorMapping(failing)), "read", resource_value, grant)

    def malformed_then_failing():
        yield ("delegated_by_subject_id", "subject-human")
        yield 5  # not a pair: terminal
        raise RuntimeError("late failure")

    # The drain after the terminal pair keeps a later iteration error's precedence.
    with pytest.raises(RuntimeError, match="late failure"):
        evaluate(_delegated_principal(_GeneratorMapping(malformed_then_failing)), "read", resource_value, grant)


def test_the_first_malformed_pair_is_terminal_and_later_pairs_are_ignored() -> None:
    resource_value = reference("memory-1")
    grant = policy(statement("allow-read", "allow", "read", resource_value))
    seen: list[str] = []

    def malformed_then_valid():
        yield ("delegated_by_subject_id", "subject-human")
        yield 5
        seen.append("drained")
        yield ("delegated_by_credential_id", "credential-human")  # never retained

    with pytest.raises(TypeError):
        evaluate(_delegated_principal(_GeneratorMapping(malformed_then_valid)), "read", resource_value, grant)
    assert seen == ["drained"]


# ── lazily discovered children: native dict snapshots are retained too ──────


class _PublicOnlyTuple(tuple):
    """A tuple whose native backing is empty: its items exist only through ``__iter__``."""

    def __new__(cls, items):
        instance = super().__new__(cls, ())
        instance.public_items = tuple(items)
        return instance

    def __iter__(self):
        return iter(self.public_items)


def _invalid_resource_dict() -> dict[str, object]:
    return {"kind": "memory", "id": "memory-1", "tenant_id": 5, "resolver": "memory_by_id"}


def _repair_after_yield(resource: dict[str, object]) -> None:
    resource["tenant_id"] = "tenant-a"


def _policy_with_public_only_statements(*statements) -> AccessPolicy:
    policy_value = policy(statement("seed", "allow", "read", reference("other")))
    object.__setattr__(policy_value, "statements", _PublicOnlyTuple(statements))
    return policy_value


class _RepairingTuple(tuple):
    """A public tuple whose iteration runs a callback (a public-view callback)."""

    def __new__(cls, items, callback):
        instance = super().__new__(cls, items)
        instance.callback = callback
        return instance

    def __iter__(self):
        self.callback()
        return iter(tuple.__iter__(self))


def _tampered_delegation(broken: dict[str, object]) -> DelegationConstraints:
    """A model whose earlier field runs a callback when completed, before the dict child is copied."""

    value = DelegationConstraints(
        delegated_by_subject_id="subject-human",
        delegated_by_credential_id="credential-human",
        allowed_actions=("read",),
        allowed_resources=(),
    )
    value.__dict__["allowed_actions"] = _RepairingTuple(("read",), lambda: _repair_after_yield(broken))
    value.__dict__["allowed_resources"] = (broken,)
    return value


def test_a_lazily_discovered_model_keeps_the_first_observation_of_its_native_dicts() -> None:
    """Completion discovers the model itself (it was not reached at entry): its native
    dict child must be snapshotted at that discovery, not re-read after a callback."""

    broken = _invalid_resource_dict()
    completed = _complete_contract_input(_tampered_delegation(broken), set(), {}, {})

    assert broken["tenant_id"] == "tenant-a"                       # the callback ran
    assert completed["allowed_resources"][0]["tenant_id"] == 5     # the first observation was kept


def test_a_lazily_discovered_mapping_keeps_the_first_observation_of_dicts_in_its_pairs() -> None:
    broken = _invalid_resource_dict()

    class _Pairs(Mapping[str, object]):
        def __getitem__(self, key: str) -> object:
            raise KeyError(key)

        def __iter__(self) -> Iterator[str]:
            return iter(())

        def __len__(self) -> int:
            return 0

        def items(self):  # type: ignore[override]
            yield ("allowed_actions", _RepairingTuple(("read",), lambda: _repair_after_yield(broken)))
            yield ("allowed_resources", (broken,))

    completed = _complete_contract_input(_Pairs(), set(), {}, {})

    assert broken["tenant_id"] == "tenant-a"
    assert completed["allowed_resources"][0]["tenant_id"] == 5


def test_lazy_discovery_reuses_an_existing_snapshot_instead_of_taking_a_new_one() -> None:
    """A dict already retained keeps its entries when a lazy path meets it again."""

    broken = _invalid_resource_dict()
    mapping_inventories: dict = {}
    _complete_contract_input(
        _tampered_delegation(broken), set(), {}, mapping_inventories
    )  # first discovery retains the dict's entries
    snapshot = mapping_inventories[id(broken)]
    _repair_after_yield(broken)
    again = _complete_contract_input(
        _tampered_delegation(broken), set(), {}, mapping_inventories
    )
    assert mapping_inventories[id(broken)] is snapshot
    assert again["allowed_resources"][0]["tenant_id"] == 5


def test_a_lazily_discovered_mapping_keeps_dict_snapshots_taken_as_its_iterator_yields() -> None:
    """Public-only child mapping; a generator repairs the dict it already yielded."""

    resource_value = reference("memory-1")
    broken = _invalid_resource_dict()

    class _StatementMapping(Mapping[str, object]):
        def __getitem__(self, key: str) -> object:
            raise KeyError(key)

        def __iter__(self) -> Iterator[str]:
            return iter(())

        def __len__(self) -> int:
            return 0

        def items(self):  # type: ignore[override]
            return self._pairs()

        def _pairs(self):
            yield ("id", "lazy-allow")
            yield ("effect", "allow")
            yield ("subject_id", "subject-a")
            yield ("credential_id", None)
            yield ("action", "read")
            yield ("resource", broken)
            _repair_after_yield(broken)  # resumed only when the consumer advances

    policy_value = _policy_with_public_only_statements(_StatementMapping())

    with pytest.raises(ValidationError):
        evaluate(principal(), "read", resource_value, policy_value)


def test_a_lazily_discovered_mapping_with_a_valid_dict_is_accepted() -> None:
    resource_value = reference("memory-1")
    good = resource_value.model_dump(mode="python")

    class _StatementMapping(Mapping[str, object]):
        def __getitem__(self, key: str) -> object:
            raise KeyError(key)

        def __iter__(self) -> Iterator[str]:
            return iter(())

        def __len__(self) -> int:
            return 0

        def items(self):  # type: ignore[override]
            yield ("id", "lazy-allow")
            yield ("effect", "allow")
            yield ("subject_id", "subject-a")
            yield ("credential_id", None)
            yield ("action", "read")
            yield ("resource", good)

    policy_value = _policy_with_public_only_statements(_StatementMapping())

    assert evaluate(principal(), "read", resource_value, policy_value).outcome == "allow"


# ── non-dict mapping children are captured before the parent iterator advances ──


def test_a_mapping_proxy_child_is_captured_before_the_parent_iterator_advances() -> None:
    broken = _invalid_resource_dict()

    class _Pairs(Mapping[str, object]):
        def __getitem__(self, key: str) -> object:
            raise KeyError(key)

        def __iter__(self) -> Iterator[str]:
            return iter(())

        def __len__(self) -> int:
            return 0

        def items(self):  # type: ignore[override]
            yield ("allowed_resources", (MappingProxyType(broken),))
            _repair_after_yield(broken)  # resumed only when the consumer advances

    completed = _complete_contract_input(_Pairs(), set(), {}, {})

    assert broken["tenant_id"] == "tenant-a"                                # the generator repaired it
    assert dict(completed["allowed_resources"][0])["tenant_id"] == 5        # the first observation holds


def test_a_mapping_proxy_resource_repaired_after_the_yield_is_still_refused() -> None:
    resource_value = reference("memory-1")
    broken = _invalid_resource_dict()

    class _StatementMapping(Mapping[str, object]):
        def __getitem__(self, key: str) -> object:
            raise KeyError(key)

        def __iter__(self) -> Iterator[str]:
            return iter(())

        def __len__(self) -> int:
            return 0

        def items(self):  # type: ignore[override]
            yield ("id", "lazy-allow")
            yield ("effect", "allow")
            yield ("subject_id", "subject-a")
            yield ("credential_id", None)
            yield ("action", "read")
            yield ("resource", MappingProxyType(broken))
            _repair_after_yield(broken)

    policy_value = _policy_with_public_only_statements(_StatementMapping())

    with pytest.raises(ValidationError):
        evaluate(principal(), "read", resource_value, policy_value)


def test_a_custom_mapping_child_is_captured_before_the_parent_iterator_advances() -> None:
    """A non-proxy public mapping child, nested inside a list, is captured at the same point."""

    resource_value = reference("memory-1")
    broken = _invalid_resource_dict()

    class _Child(Mapping[str, object]):
        def __getitem__(self, key: str) -> object:
            return broken[key]

        def __iter__(self) -> Iterator[str]:
            return iter(tuple(broken))

        def __len__(self) -> int:
            return len(broken)

    class _StatementMapping(Mapping[str, object]):
        def __getitem__(self, key: str) -> object:
            raise KeyError(key)

        def __iter__(self) -> Iterator[str]:
            return iter(())

        def __len__(self) -> int:
            return 0

        def items(self):  # type: ignore[override]
            yield ("id", "lazy-allow")
            yield ("effect", "allow")
            yield ("subject_id", "subject-a")
            yield ("credential_id", None)
            yield ("action", "read")
            yield ("resource", _Child())
            _repair_after_yield(broken)

    policy_value = _policy_with_public_only_statements(_StatementMapping())

    with pytest.raises(ValidationError):
        evaluate(principal(), "read", resource_value, policy_value)


def test_a_mapping_cycle_through_proxy_children_terminates_and_keeps_the_first_observation() -> None:
    holder: dict[str, object] = {}

    class _Loop(Mapping[str, object]):
        def __getitem__(self, key: str) -> object:
            raise KeyError(key)

        def __iter__(self) -> Iterator[str]:
            return iter(())

        def __len__(self) -> int:
            return 0

        def items(self):  # type: ignore[override]
            yield ("child", MappingProxyType(holder))
            yield ("again", loop)

    loop = _Loop()
    holder["back"] = loop
    inventories: dict = {}
    with pytest.raises(ValueError, match="cyclic"):  # terminates with the existing cycle refusal
        _complete_contract_input(loop, set(), {}, inventories)
    assert id(loop) in inventories


def test_a_valid_mapping_proxy_child_is_accepted() -> None:
    resource_value = reference("memory-1")
    good = resource_value.model_dump(mode="python")

    class _StatementMapping(Mapping[str, object]):
        def __getitem__(self, key: str) -> object:
            raise KeyError(key)

        def __iter__(self) -> Iterator[str]:
            return iter(())

        def __len__(self) -> int:
            return 0

        def items(self):  # type: ignore[override]
            yield ("id", "lazy-allow")
            yield ("effect", "allow")
            yield ("subject_id", "subject-a")
            yield ("credential_id", None)
            yield ("action", "read")
            yield ("resource", MappingProxyType(good))

    policy_value = _policy_with_public_only_statements(_StatementMapping())

    assert evaluate(principal(), "read", resource_value, policy_value).outcome == "allow"


# ── bounded discovery: one shared budget per validation call ────────────────

from contracts_limits import (  # noqa: E402
    DISCOVERY_BUDGET_MESSAGE,
    STANDARD_MAX_DEPTH,
    STANDARD_MAX_EDGES,
    DiscoveryBudget,
)


def _nested(kind: str, depth: int, leaf: object = None) -> object:
    value: object = leaf
    for _ in range(depth):
        value = {"k": value} if kind == "dict" else ([value] if kind == "list" else (value,))
    return value


def _evaluate_budget_statements(*statements) -> object:
    return evaluate(principal(), "read", reference("memory-1"), _policy_with_public_only_statements(*statements))


@pytest.mark.parametrize("kind", ["tuple", "list", "dict"])
def test_deep_nesting_is_refused_with_the_budget_error_not_a_crash(kind) -> None:
    deep = _nested(kind, 5000)  # far past the recursion limit
    with pytest.raises(ValueError, match=DISCOVERY_BUDGET_MESSAGE):
        _complete_contract_input(deep, set(), {}, {})
    with pytest.raises(ValueError, match=DISCOVERY_BUDGET_MESSAGE):
        _evaluate_budget_statements(deep)


@pytest.mark.parametrize("kind", ["tuple", "list", "dict"])
def test_nesting_up_to_the_depth_limit_is_not_a_budget_error_and_cannot_overflow_the_stack(kind) -> None:
    ok_depth = _nested(kind, STANDARD_MAX_DEPTH - 8)
    completed = _complete_contract_input(ok_depth, set(), {}, {})  # accepted by the budget
    assert completed is not None
    with pytest.raises(ValidationError):  # and then judged by its shape, not by the budget
        _evaluate_budget_statements(ok_depth)
    with pytest.raises(ValueError, match=DISCOVERY_BUDGET_MESSAGE):
        _complete_contract_input(_nested(kind, STANDARD_MAX_DEPTH + 2), set(), {}, {})


def test_a_wide_container_is_refused_before_it_is_walked() -> None:
    wide = tuple(range(STANDARD_MAX_EDGES + 10))
    with pytest.raises(ValueError, match=DISCOVERY_BUDGET_MESSAGE):
        _evaluate_budget_statements(*wide)
    with pytest.raises(ValueError, match=DISCOVERY_BUDGET_MESSAGE):
        _complete_contract_input(list(wide), set(), {}, {})
    with pytest.raises(ValueError, match=DISCOVERY_BUDGET_MESSAGE):
        _complete_contract_input({i: i for i in range(STANDARD_MAX_EDGES + 10)}, set(), {}, {})


def test_native_shared_substructure_is_copied_once_per_object() -> None:
    """A diamond chain is 2 nodes per level to discover and 2**n paths to reach: native
    subtrees are pure, so each is copied once and shared by every path."""

    import time

    shared: object = ("leaf",)
    for _ in range(40):  # 2**40 paths
        shared = (shared, shared)
    started = time.monotonic()
    completed = _complete_contract_input(shared, set(), {}, {})
    assert time.monotonic() - started < 5
    assert completed[0] is completed[1]  # one copy, shared by both paths
    with pytest.raises(ValidationError):  # and a contract still judges it by its shape, quickly
        _evaluate_budget_statements(shared)


def test_a_diamond_of_public_views_is_still_charged_per_path() -> None:
    """A container subclass is observed again on every edge that reaches it, so reuse is
    not allowed and the paths are charged: the budget refuses the explosion."""

    class _View(tuple):
        def __iter__(self):
            return iter(tuple.__iter__(self))

    shared: object = _View(("leaf",))
    for _ in range(20):
        shared = _View((shared, shared))
    with pytest.raises(ValueError, match=DISCOVERY_BUDGET_MESSAGE):
        _complete_contract_input(shared, set(), {}, {})


def test_the_depth_limit_holds_on_a_path_that_reaches_a_reused_subtree() -> None:
    deep = _nested("tuple", STANDARD_MAX_DEPTH - 1)
    _complete_contract_input((deep,), set(), {}, {})                  # fits at depth 1
    with pytest.raises(ValueError, match=DISCOVERY_BUDGET_MESSAGE):   # the same subtree, one level lower
        _complete_contract_input((deep, ((deep,),)), set(), {}, {})


def test_a_long_public_pair_iterator_is_cut_off_at_the_edge_limit() -> None:
    produced = []

    class _Many(Mapping[str, object]):
        def __getitem__(self, key: str) -> object:
            raise KeyError(key)

        def __iter__(self) -> Iterator[str]:
            return iter(())

        def __len__(self) -> int:
            return 0

        def items(self):  # type: ignore[override]
            for index in range(STANDARD_MAX_EDGES * 3):
                produced.append(index)
                yield (f"k{index}", index)

    with pytest.raises(ValueError, match=DISCOVERY_BUDGET_MESSAGE):
        _complete_contract_input(_Many(), set(), {}, {})
    assert len(produced) <= STANDARD_MAX_EDGES + 2          # the iterator was not drained


def test_a_long_public_pair_iterator_is_cut_off_even_after_a_terminal_pair() -> None:
    produced = []

    class _Drain(Mapping[str, object]):
        def __getitem__(self, key: str) -> object:
            raise KeyError(key)

        def __iter__(self) -> Iterator[str]:
            return iter(())

        def __len__(self) -> int:
            return 0

        def items(self):  # type: ignore[override]
            yield "not-a-pair"                                # terminal malformed pair
            for index in range(STANDARD_MAX_EDGES * 3):       # then a long tail to drain
                produced.append(index)
                yield (f"k{index}", index)

    with pytest.raises(ValueError, match=DISCOVERY_BUDGET_MESSAGE):
        _complete_contract_input(_Drain(), set(), {}, {})
    assert len(produced) <= STANDARD_MAX_EDGES + 2


def test_a_long_public_sequence_view_is_cut_off_during_completion() -> None:
    produced = []

    class _Long(tuple):
        def __iter__(self):
            for index in range(STANDARD_MAX_EDGES * 3):
                produced.append(index)
                yield index

    with pytest.raises(ValueError, match=DISCOVERY_BUDGET_MESSAGE):
        _complete_contract_input(_Long(), set(), {}, {})
    assert len(produced) <= STANDARD_MAX_EDGES + 2


def test_accounting_runs_no_hook_of_the_caller_objects() -> None:
    calls: list[str] = []

    class _Hostile(tuple):
        def __len__(self):
            calls.append("len")
            raise AssertionError("len")

        def __eq__(self, other):
            calls.append("eq")
            raise AssertionError("eq")

        def __hash__(self):
            calls.append("hash")
            raise AssertionError("hash")

        def __bool__(self):
            calls.append("bool")
            raise AssertionError("bool")

    resource_value = reference("memory-1")
    grant = policy(statement("allow-1", "allow", "read", resource_value))
    object.__setattr__(grant, "statements", _Hostile(grant.statements))
    assert evaluate(principal(), "read", resource_value, grant).outcome == "allow"
    assert calls == []


def test_the_budget_is_shared_by_every_root_of_one_call() -> None:
    budget = DiscoveryBudget()
    half = list(range(STANDARD_MAX_EDGES // 2 + 1000))
    _freeze_reachable_model_inventories(half, {}, {}, None, None, budget)
    with pytest.raises(ValueError, match=DISCOVERY_BUDGET_MESSAGE):
        _freeze_reachable_model_inventories(list(half), {}, {}, None, None, budget)
    # the same two walks under separate budgets are fine: it is the sharing that bounds the call
    _freeze_reachable_model_inventories(half, {}, {}, None, None, DiscoveryBudget())
    _freeze_reachable_model_inventories(list(half), {}, {}, None, None, DiscoveryBudget())


def test_depth_outcome_does_not_depend_on_the_order_of_aliased_references() -> None:
    shared = _nested("tuple", STANDARD_MAX_DEPTH - 20)
    for order in ((shared, shared), (shared, (shared,)), ((shared,), shared)):
        _complete_contract_input(order, set(), {}, {})  # same outcome in every order
    too_deep = _nested("tuple", STANDARD_MAX_DEPTH)  # fits when reached directly, not when wrapped
    _complete_contract_input((too_deep, too_deep), set(), {}, {})
    for order in (((too_deep,), too_deep), (too_deep, (too_deep,)), ((too_deep,), (too_deep,))):
        with pytest.raises(ValueError, match=DISCOVERY_BUDGET_MESSAGE):
            _complete_contract_input(order, set(), {}, {})


def test_ordinary_policies_are_unaffected_by_the_budget() -> None:
    resource_value = reference("memory-1")
    grants = tuple(statement(f"allow-{i}", "allow", "read", resource_value) for i in range(200))
    assert evaluate(principal(), "read", resource_value, policy(*grants)).outcome == "allow"
    assert validate_policy_decision(
        evaluate(principal(), "read", resource_value, policy(*grants))
    ).outcome == "allow"


def _proxy_chain(depth: int) -> object:
    value: object = None
    for _ in range(depth):
        value = MappingProxyType({"k": value})
    return value


def test_a_deep_chain_of_public_mappings_is_refused_with_the_budget_error() -> None:
    with pytest.raises(ValueError, match=DISCOVERY_BUDGET_MESSAGE):
        _complete_contract_input(_proxy_chain(5000), set(), {}, {})
    with pytest.raises(ValueError, match=DISCOVERY_BUDGET_MESSAGE):
        _evaluate_budget_statements(_proxy_chain(5000))


def test_a_public_mapping_chain_within_the_limit_cannot_overflow_the_stack() -> None:
    completed = _complete_contract_input(_proxy_chain(STANDARD_MAX_DEPTH - 8), set(), {}, {})
    assert completed is not None


# ── a mapping is captured exactly once, at its first exposure ───────────────


def _entry_flow(root: object) -> object:
    """The full evaluator flow for one root: entry freeze, pending captures, completion."""

    inventories: dict = {}
    owners: dict = {}
    mapping_inventories: dict = {}
    native_dicts: dict = {}
    visited: dict = {}
    budget = DiscoveryBudget()
    reach: dict = {}
    _freeze_reachable_model_inventories(root, inventories, visited, owners, native_dicts, budget, 0, True, reach)
    _freeze_mapping_inventories(owners, mapping_inventories, inventories, visited, native_dicts, None, budget, reach)
    mapping_inventories.update(native_dicts)
    return _complete_contract_input(root, set(), inventories, mapping_inventories, budget)


class _Items(Mapping[str, object]):
    """A public mapping whose pairs come from a generator function."""

    def __init__(self, pairs) -> None:
        self._pairs = pairs

    def __getitem__(self, key: str) -> object:
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return iter(())

    def __len__(self) -> int:
        return 0

    def items(self):  # type: ignore[override]
        return self._pairs()


def test_a_queued_mapping_finished_by_recursion_is_not_captured_a_second_time() -> None:
    """A and B are queued; A exposes a new C; C exposes B.  B is captured when C exposes
    it, and the outer loop must not read it again after A repaired it."""

    broken = _invalid_resource_dict()
    reads: list[str] = []
    b = _Items(lambda: _b_pairs())

    def _b_pairs():
        reads.append("b")
        yield ("tenant_id", broken["tenant_id"])

    c = _Items(lambda: iter((("b", b),)))

    def _a_pairs():
        yield ("c", c)
        _repair_after_yield(broken)  # runs when the walk advances A

    a = _Items(_a_pairs)
    completed = _entry_flow((a, b))

    assert broken["tenant_id"] == "tenant-a"                       # A repaired the backing
    assert completed[1]["tenant_id"] == 5                          # B's first observation was kept
    assert completed[0]["c"]["b"]["tenant_id"] == 5
    assert reads == ["b"]                                          # and B was read exactly once


def test_a_child_mapping_the_entry_walk_already_queued_is_captured_when_it_is_exposed() -> None:
    """The same proxy is queued by the entry walk (second root) and exposed by a generator
    before that generator is advanced; its first observation is the one at exposure."""

    broken = _invalid_resource_dict()
    shared = MappingProxyType(broken)

    def _generator_pairs():
        yield ("allowed_resources", (shared,))
        _repair_after_yield(broken)

    completed = _entry_flow((_Items(_generator_pairs), shared))

    assert broken["tenant_id"] == "tenant-a"
    assert dict(completed[0]["allowed_resources"][0])["tenant_id"] == 5
    assert dict(completed[1])["tenant_id"] == 5


def test_each_mapping_is_read_once_however_the_walk_reaches_it() -> None:
    reads: list[str] = []

    def counted(name):
        def pairs():
            reads.append(name)
            yield ("leaf", name)
        return _Items(pairs)

    leaf = counted("leaf")
    middle = _Items(lambda: iter((("leaf", leaf), ("leaf-again", leaf))))
    top = _Items(lambda: iter((("middle", middle), ("leaf", leaf))))
    _entry_flow((top, middle, leaf, [leaf, middle]))
    assert sorted(reads) == ["leaf"]


# ── pre-emptive sweep: classes seen across reviews ──────────────────────────


def _statement_dict(**overrides) -> dict:
    value = {
        "id": "dict-allow", "effect": "allow", "subject_id": "subject-a", "credential_id": None,
        "action": "read", "resource": reference("memory-1").model_dump(mode="python"),
    }
    value.update(overrides)
    return value


def _with_statements(*statements) -> AccessPolicy:
    return _policy_with_public_only_statements(*statements)


# class 1: a registered-only ABC is not trusted as the real protocol


class _NotReallyAMapping:
    def __init__(self, entries):
        self._entries = entries

    def items(self):
        return iter(self._entries.items())


def test_a_registered_only_mapping_is_not_treated_as_a_mapping() -> None:
    Mapping.register(_NotReallyAMapping)
    resource_value = reference("memory-1")
    grant = _NotReallyAMapping(_statement_dict())
    with pytest.raises(ValidationError):
        evaluate(principal(), "read", resource_value, _with_statements(grant))
    # the completed copy is the object itself (not a rebuilt dict), so strict validation refuses it
    assert isinstance(_complete_contract_input(grant, set(), {}, {}), _NotReallyAMapping)


@pytest.mark.filterwarnings("ignore:For performance reasons")
def test_a_registered_only_model_is_not_treated_as_a_model() -> None:
    class Impostor:
        pass

    BaseModel.register(Impostor)
    resource_value = reference("memory-1")
    with pytest.raises((ValidationError, TypeError, ValueError)):
        evaluate(principal(), "read", resource_value, _with_statements(Impostor()))


# class 6: a str-subclass key is one key by its text; two keys with one text are refused


class _DuplicateEffectKey(str):
    def __hash__(self) -> int:
        return hash("effect")

    def __eq__(self, other: object) -> bool:
        return False

    def __ne__(self, other: object) -> bool:
        return True


class _TextKey(str):
    """An ordinary str subclass (no hook overridden)."""


def test_a_second_key_with_the_same_text_cannot_flip_an_allow_into_a_silent_pass() -> None:
    resource_value = reference("memory-1")
    mirrored = _statement_dict(effect="allow")
    dict.__setitem__(mirrored, _DuplicateEffectKey("effect"), "deny")
    with pytest.raises(ValueError):
        evaluate(principal(), "read", resource_value, _with_statements(mirrored))
    first = _statement_dict(effect="deny")
    dict.__setitem__(first, _DuplicateEffectKey("effect"), "allow")
    with pytest.raises(ValueError):
        evaluate(principal(), "read", resource_value, _with_statements(first))


def test_an_ordinary_str_subclass_key_is_read_as_its_text() -> None:
    resource_value = reference("memory-1")
    keyed = {_TextKey(name): value for name, value in _statement_dict().items()}
    assert evaluate(principal(), "read", resource_value, _with_statements(keyed)).outcome == "allow"


# class 3: a native dict first met lazily keeps its first observation for every later edge


def test_a_lazily_met_dict_shared_by_two_edges_keeps_its_first_observation() -> None:
    broken = _invalid_resource_dict()
    mapping_inventories: dict = {}
    first = _complete_contract_input((broken,), set(), {}, mapping_inventories)
    _repair_after_yield(broken)
    second = _complete_contract_input((broken,), set(), {}, mapping_inventories)
    assert first[0]["tenant_id"] == 5
    assert second[0]["tenant_id"] == 5  # the same snapshot, not a re-read of the repaired dict


# class 5: each pair of a custom mapping is frozen as a plain tuple, read once


def test_a_pair_that_is_a_tuple_subclass_with_a_changing_iterator_is_read_once() -> None:
    reads = []

    class _Pair(tuple):
        def __iter__(self):
            reads.append(1)
            return iter(("k", len(reads)))

    class _Pairs(Mapping[str, object]):
        def __getitem__(self, key):
            raise KeyError(key)

        def __iter__(self):
            return iter(())

        def __len__(self):
            return 0

        def items(self):  # type: ignore[override]
            return iter((_Pair(("ignored", "ignored")),))

    completed = _complete_contract_input(_Pairs(), set(), {}, {})
    assert completed == {"k": 1} and len(reads) == 1


# class 8: the objects an inventory is keyed by are kept alive by the inventory


def test_objects_keyed_by_identity_stay_alive_for_the_whole_call() -> None:
    class _Weak(dict):
        pass

    child = _Weak({"k": 1})
    reference_to_child = weakref.ref(child)
    holder = [child]
    native_dicts: dict = {}
    _freeze_reachable_model_inventories(holder, {}, {}, None, native_dicts)
    del child, holder
    gc.collect()
    assert reference_to_child() is not None  # an id() could not be recycled while the inventory is live


# class 9: the evaluator takes no time from the wall clock


def test_the_policy_modules_never_read_the_clock() -> None:
    import ast
    from pathlib import Path

    import contracts_policy
    import contracts_policy_reference

    for module in (contracts_policy, contracts_policy_reference):
        tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
        imported = {
            alias.name.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in (node.names if isinstance(node, ast.Import) else [ast.alias(name=node.module or "")])
        }
        assert not imported & {"time", "datetime", "calendar", "zoneinfo"}, module.__name__


# ── discovery findings: depth through model-owned mappings, shared containers, fields-set charges ──


def _extra_chain(length: int) -> object:
    """Models linked through ``__pydantic_extra__`` = a read-only mapping holding the next model."""

    tail: object = None
    for index in range(length):
        model = SourceVersionLike(index)
        if tail is not None:
            object.__setattr__(model, "__pydantic_extra__", MappingProxyType({"child": tail}))
        tail = model
    return tail


def SourceVersionLike(index: int):
    from contracts_references import SourceVersion

    return SourceVersion(record_id=f"r{index}", content_revision=1, policy_epoch=1)


def test_a_chain_of_models_linked_through_extra_mappings_keeps_its_depth() -> None:
    chain = _extra_chain(200)  # two levels per link (the model, then its mapping): past the limit
    with pytest.raises(ValueError, match=DISCOVERY_BUDGET_MESSAGE):
        _entry_flow(chain)
    assert _entry_flow(_extra_chain(20)) is not None  # a short chain is within the limit


def test_a_mapping_reachable_through_an_already_visited_container_is_captured_at_exposure() -> None:
    broken = _invalid_resource_dict()
    shared = (MappingProxyType(broken),)

    def _parent_pairs():
        yield ("allowed_resources", shared)
        _repair_after_yield(broken)  # resumed only when the walk advances the parent

    completed = _entry_flow((_Items(_parent_pairs), shared))

    assert broken["tenant_id"] == "tenant-a"
    assert dict(completed[0]["allowed_resources"][0])["tenant_id"] == 5   # exposed first: first observation
    assert dict(completed[1][0])["tenant_id"] == 5


def test_a_merge_of_reaches_is_charged_so_the_bookkeeping_cannot_hold_memory_for_free() -> None:
    proxies = _public_mappings(2000)
    nested: object = (proxies,)
    for _ in range(200):
        nested = (nested, MappingProxyType({"extra": 1}))  # every level merges a growing reach with one more mapping
    with pytest.raises(ValueError, match=DISCOVERY_BUDGET_MESSAGE):
        _freeze_reachable_model_inventories(nested, {}, {}, None, None, DiscoveryBudget())


def test_native_fields_set_names_are_charged_per_model() -> None:
    from contracts_references import SourceVersion

    names = {f"name-{i}" for i in range(2000)}
    models = []
    for index in range(100):
        model = SourceVersion(record_id=f"r{index}", content_revision=1, policy_epoch=1)
        _BASE_MODEL_FIELDS_SET_DESCRIPTOR.__set__(model, set(names))  # a large set per model
        models.append(model)
    budget = DiscoveryBudget()
    with pytest.raises(ValueError, match=DISCOVERY_BUDGET_MESSAGE):  # 100 x 2000 names > the edge limit
        _freeze_reachable_model_inventories(tuple(models), {}, {}, None, None, budget)


def test_a_models_ordinary_fields_set_costs_nothing_extra() -> None:
    from contracts_references import SourceVersion

    model = SourceVersion(record_id="r", content_revision=1, policy_epoch=1)
    budget = DiscoveryBudget()
    _freeze_reachable_model_inventories(model, {}, {}, None, None, budget)
    assert budget.edges == 3  # the three stored fields; the names of a legitimate set are those fields


def test_only_names_beyond_the_stored_fields_are_charged() -> None:
    from contracts_references import SourceVersion

    model = SourceVersion(record_id="r", content_revision=1, policy_epoch=1)
    _BASE_MODEL_FIELDS_SET_DESCRIPTOR.__set__(model, {f"n{i}" for i in range(1003)})  # 1,000 beyond the 3 stored
    budget = DiscoveryBudget()
    _freeze_reachable_model_inventories(model, {}, {}, None, None, budget)
    assert budget.edges == 3 + 1000


def test_a_large_legitimate_policy_fits_the_standard_budget() -> None:
    """230 members with 25 distinct statements each (5,750 statements): the shape a hosted
    tenant produces.  Charging every fields-set name used to refuse this above 162 members."""

    statements = []
    members = []
    for member in range(230):
        subject = f"subject-{member:04d}"
        members.append(MembershipVersion(subject_id=subject, version=1))
        for index in range(25):
            statements.append(
                PolicyStatement(
                    id=f"s-{member}-{index}", effect="allow", subject_id=subject, credential_id=None,
                    action="read", resource=reference(f"memory-{index}"),
                )
            )
    big = AccessPolicy(
        schema_version=VERSION, tenant_id="tenant-a", policy_epoch=1,
        membership_versions=tuple(members), statements=tuple(statements),
    )
    decision = evaluate(
        principal(subject_id="subject-0000", membership_version=1), "read", reference("memory-0"), big
    )
    assert decision.outcome == "allow"



# ── the error for a shared invalid graph is bounded in size and in the time to render ──


def _tuple_diamond(depth: int) -> object:
    shared: object = ("leaf",)
    for _ in range(depth):
        shared = (shared, shared)
    return shared


def _dict_diamond(depth: int) -> object:
    shared: object = {"leaf": 1}
    for _ in range(depth):
        shared = {"a": shared, "b": shared}
    return shared


@pytest.mark.parametrize("build,depth", [(_tuple_diamond, 22), (_dict_diamond, 19)])
def test_the_error_for_a_shared_invalid_graph_renders_in_bounded_time_and_size(build, depth) -> None:
    import time

    resource_value = reference("memory-1")
    with pytest.raises(ValidationError) as caught:
        _evaluate_budget_statements(build(depth))
    started = time.monotonic()
    text = str(caught.value)
    assert time.monotonic() - started < 1.0  # the base takes seconds at depth 22 and doubles per level
    assert len(text) < 4096
    assert "statements" in text  # the diagnostic still says where
    assert caught.value.error_count() >= 1
    with pytest.raises(ValidationError):  # policy and principal roots are bounded the same way
        evaluate(principal(), "read", resource_value, _policy_with_public_only_statements(build(depth)))


def test_diagnostics_for_ordinary_invalid_inputs_keep_their_input_echo() -> None:
    with pytest.raises(ValidationError) as caught:
        _evaluate_budget_statements(("not-a-statement",))
    assert "input_value" in str(caught.value) and "not-a-statement" in str(caught.value)


# ── the reach of a shared container is held by reference and charged once ──


def _public_mappings(count: int) -> tuple:
    return tuple(MappingProxyType({"k": index}) for index in range(count))


def test_a_container_shared_by_many_parents_is_not_charged_once_per_parent() -> None:
    shared = _public_mappings(20)
    parents = tuple((shared,) for _ in range(2000))  # 2,000 parents, each holding the one container
    budget = DiscoveryBudget()
    _freeze_reachable_model_inventories(parents, {}, {}, None, None, budget)
    # the walk itself costs ~6,000 entries; the reach bookkeeping must not add one copy per parent
    assert budget.edges < 10_000, budget.edges


def test_a_mapping_reached_through_two_children_is_recorded_once() -> None:
    mapping = MappingProxyType({"k": 1})
    holder = ((mapping,), [mapping])
    reach: dict = {}
    _freeze_reachable_model_inventories(holder, {}, {}, None, None, DiscoveryBudget(), 0, True, reach)
    assert reach[id(holder)] == (mapping,)  # once, not once per path to it


def test_the_unrebuildable_validation_error_falls_back_to_a_value_free_error(monkeypatch) -> None:
    from pydantic_core import ValidationError as CoreValidationError

    def refuse(*args, **kwargs):
        raise TypeError("cannot rebuild")

    monkeypatch.setattr(CoreValidationError, "from_exception_data", staticmethod(refuse))
    with pytest.raises(ValueError) as caught:
        _evaluate_budget_statements(_tuple_diamond(22))
    assert not isinstance(caught.value, ValidationError)  # the bounded fallback, not the echoing error
    text = str(caught.value)
    assert len(text) < 256 and "validation error" in text and "AccessPolicy" in text


# ── aliased inputs stay inside the memory and capture-depth bounds ──────────


def test_a_repeated_wide_subgraph_does_not_multiply_the_reach_work() -> None:
    """50,000 references to one tuple of 10,000 distinct proxies: discovery charges
    ~80,000 edges, but merging each reference's reach used to touch 500 million entries."""

    import time

    wide = _public_mappings(10_000)
    repeated = (wide,) * 50_000
    budget = DiscoveryBudget()
    reach: dict = {}
    started = time.monotonic()
    _freeze_reachable_model_inventories(repeated, {}, {}, None, None, budget, 0, True, reach)
    assert time.monotonic() - started < 5.0
    assert reach[id(repeated)] is reach[id(wide)]  # one distinct source: held by reference, no merge


def test_merging_reaches_is_charged_before_it_is_built() -> None:
    """Distinct reach tuples whose merge is far larger than the walk: refused quickly by the
    charge taken before the merge, not after building it."""

    import time

    shared = _public_mappings(2_000)
    # 400 distinct wrappers of the same 2,000 proxies: each wrapper has its own reach tuple
    # (the shared proxies plus one of its own), so the root merges 400 x 2,001 entries.
    wrappers = tuple((shared, MappingProxyType({"own": index})) for index in range(400))
    started = time.monotonic()
    with pytest.raises(ValueError, match=DISCOVERY_BUDGET_MESSAGE):
        _freeze_reachable_model_inventories(wrappers, {}, {}, None, None, DiscoveryBudget())
    assert time.monotonic() - started < 5.0


def _proxy_chain_nodes(length: int) -> list:
    nodes: list = []
    tail: object = None
    for _ in range(length):
        tail = MappingProxyType({"next": tail})
        nodes.append(tail)
    return nodes


def test_shallow_aliases_of_a_long_mapping_chain_end_in_the_budget_error() -> None:
    nodes = _proxy_chain_nodes(5_000)
    root = tuple(reversed(nodes))  # head first: capture follows the chain; every node is also at depth 1
    with pytest.raises(ValueError, match=DISCOVERY_BUDGET_MESSAGE):  # previously a RecursionError
        _entry_flow(root)
    with pytest.raises(ValueError, match=DISCOVERY_BUDGET_MESSAGE):
        _evaluate_budget_statements(root)


def test_a_short_aliased_mapping_chain_is_still_captured() -> None:
    nodes = _proxy_chain_nodes(40)
    completed = _entry_flow(tuple(reversed(nodes)))
    assert len(completed) == 40 and dict(completed[0])["next"] is not None


# ── a pair is read incrementally: its value is frozen before the exhaustion check ──


class _RepairingPair(tuple):
    """A key/value pair whose public iterator repairs the value during the exhaustion check."""

    def __new__(cls, key, value, repair):
        instance = super().__new__(cls, (key, value))
        instance.key, instance.value, instance.repair = key, value, repair
        return instance

    def __iter__(self):
        yield self.key
        yield self.value
        self.repair()  # runs when the unpacking asks for a third item


def _invalid_handle():
    handle = reference("memory-1")
    handle.__dict__["tenant_id"] = 5
    return handle


def test_a_pair_cannot_repair_its_value_during_the_exhaustion_check() -> None:
    handle = _invalid_handle()

    def repair():
        handle.__dict__["tenant_id"] = "tenant-a"

    pairs = _Items(lambda: iter((_RepairingPair("allowed_resources", (handle,), repair),)))
    completed = _entry_flow(pairs)
    assert handle.tenant_id == "tenant-a"                          # the repair ran
    assert completed["allowed_resources"][0]["tenant_id"] == 5      # the first observation was kept


def test_a_statement_value_repaired_during_the_exhaustion_check_is_still_refused() -> None:
    resource_value = reference("memory-1")
    handle = _invalid_handle()

    def repair():
        handle.__dict__["tenant_id"] = "tenant-a"

    def pairs():
        yield ("id", "lazy-allow")
        yield ("effect", "allow")
        yield ("subject_id", "subject-a")
        yield ("credential_id", None)
        yield ("action", "read")
        yield _RepairingPair("resource", handle, repair)

    with pytest.raises(ValidationError):
        evaluate(principal(), "read", resource_value, _with_statements(_Items(pairs)))


def test_pair_shape_errors_keep_their_types_and_terminal_ordering() -> None:
    class Three(tuple):
        def __iter__(self):
            return iter(("a", 1, "extra"))

    class One(tuple):
        def __iter__(self):
            return iter(("a",))

    for bad, kind, text in ((Three(), ValueError, "too many values to unpack"),
                            (One(), ValueError, "not enough values to unpack"),
                            (5, TypeError, "cannot unpack non-iterable int object")):
        later = []

        def pairs(bad=bad):
            yield ("first", 1)
            yield bad
            later.append("drained")
            yield ("after", 2)  # drained, never retained

        with pytest.raises(kind, match=text):
            _entry_flow(_Items(pairs))
        assert later == ["drained"]


def test_freeze_mapping_entries_freezes_each_value_before_the_exhaustion_check() -> None:
    handle = _invalid_handle()

    def repair():
        handle.__dict__["tenant_id"] = "tenant-a"

    inventories: dict = {}
    entries = _freeze_mapping_entries(
        (_RepairingPair("resource", handle, repair),), {}, inventories, {}, {}, DiscoveryBudget()
    )
    assert entries[0].key == "resource"
    frozen = inventories[id(handle)]
    assert dict(frozen.stored_items)["tenant_id"] == 5  # captured before the repair ran


@pytest.mark.parametrize("container", [list, tuple])
def test_a_sequence_view_cannot_repair_an_item_it_already_yielded(container) -> None:
    broken = _invalid_resource_dict()
    handle = _invalid_handle()

    class View(container):  # type: ignore[misc, valid-type]
        def __iter__(self):
            yield broken
            yield handle
            broken["tenant_id"] = "tenant-a"  # both repairs run on the advance after the last yield
            handle.__dict__["tenant_id"] = "tenant-a"

    completed = _complete_contract_input(View(), set(), {}, {})
    assert completed[0]["tenant_id"] == 5 and completed[1]["tenant_id"] == 5


def test_the_module_no_longer_has_a_drain_then_freeze_mapping_reader() -> None:
    import contracts_policy_reference

    assert not hasattr(contracts_policy_reference, "_mapping_items")


def _unpack_error(value) -> tuple[type, str]:
    try:
        key, item = value  # noqa: F841 - the reference behaviour
    except Exception as error:
        return type(error), str(error)
    raise AssertionError("unpacking succeeded")


class _IterReturnsList:
    def __iter__(self):
        return [1, 2]


class _IterRaisesTypeError:
    def __iter__(self):
        raise TypeError("the object's own message")


class _NoIteration:
    __iter__ = None


@pytest.mark.parametrize("value", [_IterReturnsList(), _IterRaisesTypeError(), _NoIteration(), 5, None, object()])
def test_pair_reading_raises_exactly_what_unpacking_raises(value) -> None:
    from contracts_policy_reference import _open_pair

    expected = _unpack_error(value)
    with pytest.raises(Exception) as caught:
        _open_pair(value)
    assert (type(caught.value), str(caught.value)) == expected


# ── completion does not charge again what discovery already charged ─────────


def _large_policy(members: int, per: int = 25) -> AccessPolicy:
    statements, versions = [], []
    for member in range(members):
        subject = f"subject-{member:04d}"
        versions.append(MembershipVersion(subject_id=subject, version=1))
        statements.extend(
            PolicyStatement(id=f"s-{member}-{i}", effect="allow", subject_id=subject, credential_id=None,
                            action="read", resource=reference(f"memory-{i}"))
            for i in range(per)
        )
    return AccessPolicy(schema_version=VERSION, tenant_id="tenant-a", policy_epoch=1,
                        membership_versions=tuple(versions), statements=tuple(statements))


def _charged_edges(policy_value) -> tuple[int, int]:
    """(edges charged by the whole call, edges charged by discovery alone)."""

    import contracts_policy_reference as module

    budgets = []
    original = module.DiscoveryBudget.__init__

    def recording(self, *args, **kwargs):
        original(self, *args, **kwargs)
        budgets.append(self)

    module.DiscoveryBudget.__init__ = recording
    try:
        evaluate(principal(subject_id="subject-0000", membership_version=1), "read", reference("memory-0"), policy_value)
        whole = budgets[0].edges
        discovery = DiscoveryBudget()
        _freeze_reachable_model_inventories(policy_value, {}, {}, None, {}, discovery)
    finally:
        module.DiscoveryBudget.__init__ = original
    return whole, discovery.edges


def test_a_native_policy_is_charged_about_once_not_once_per_phase() -> None:
    whole, discovery = _charged_edges(_large_policy(40))
    # every native edge is paid once, at discovery; completion's first copy adds only what
    # discovery could not see (the other two roots, a handful of edges)
    assert whole <= discovery * 1.1, (whole, discovery)


def test_a_policy_of_7500_statements_fits_the_standard_budget() -> None:
    """Charging each edge once leaves room for richer statements: 300 members with 25
    statements each (7,500 statements) fits.  The node limit (one per object) now binds first,
    at about 321 members of this shape."""

    decision = evaluate(principal(subject_id="subject-0000", membership_version=1), "read",
                        reference("memory-0"), _large_policy(300))
    assert decision.outcome == "allow"


def test_a_re_copied_impure_subtree_is_still_charged_per_copy() -> None:
    """A native tuple holding an empty public view is impure, so it is rebuilt on every path:
    each re-copy is charged, so a diamond of such tuples is refused."""

    class EmptyView(tuple):
        def __iter__(self):
            return iter(())

    node: object = (EmptyView(),)
    for _ in range(40):
        node = (node, node, EmptyView())
    with pytest.raises(ValueError, match=DISCOVERY_BUDGET_MESSAGE):
        _complete_contract_input(node, set(), {}, {})


def test_a_container_first_seen_through_a_public_view_is_charged_when_copied() -> None:
    inner = list(range(STANDARD_MAX_EDGES // 2 + 1000))

    class View(tuple):
        def __iter__(self):
            return iter((inner, list(inner)))  # two lists only the public view exposes

    with pytest.raises(ValueError, match=DISCOVERY_BUDGET_MESSAGE):
        _complete_contract_input(View(), set(), {}, {})


# ── a non-string key cannot overwrite a real one; a cached reach is scanned once ──


class _ImpersonatingKey:
    """Hashes and compares equal to "effect": inserting it would overwrite the real key."""

    calls: list = []

    def __hash__(self):
        _ImpersonatingKey.calls.append("hash")
        return hash("effect")

    def __eq__(self, other):
        _ImpersonatingKey.calls.append("eq")
        return other == "effect" or other is self


def test_a_non_string_key_cannot_overwrite_a_denial() -> None:
    resource_value = reference("memory-1")

    def pairs():
        yield ("id", "s-1")
        yield ("effect", "deny")
        yield ("subject_id", "subject-a")
        yield ("credential_id", None)
        yield ("action", "read")
        yield ("resource", resource_value.model_dump(mode="python"))
        yield (_ImpersonatingKey(), "allow")

    _ImpersonatingKey.calls.clear()
    with pytest.raises(ValidationError) as caught:
        evaluate(principal(), "read", resource_value, _with_statements(_Items(pairs)))
    assert any(error["type"] == "invalid_key" for error in caught.value.errors())
    assert _ImpersonatingKey.calls == []  # its hash and equality never ran


def test_plain_non_string_keys_keep_their_exact_diagnostics() -> None:
    resource_value = reference("memory-1")
    for key, expected_loc in ((1, 1), (b"k", "b'k'")):
        bad = {**resource_value.model_dump(mode="python"), key: "x"}

        def pairs(bad=bad):
            yield ("id", "s-1")
            yield ("effect", "allow")
            yield ("subject_id", "subject-a")
            yield ("credential_id", None)
            yield ("action", "read")
            yield ("resource", bad)

        with pytest.raises(ValidationError) as caught:
            evaluate(principal(), "read", resource_value, _with_statements(_Items(pairs)))
        locations = [error["loc"] for error in caught.value.errors() if error["type"] == "invalid_key"]
        assert locations and locations[0][-1] == expected_loc


def test_mapping_capture_scans_a_shared_cached_reach_once() -> None:
    import time

    proxies = tuple(MappingProxyType({"k": i}) for i in range(10_000))

    def pairs():
        for index in range(50_000):
            yield (f"k{index}", proxies)

    started = time.monotonic()
    try:
        _entry_flow(_Items(pairs))
    except ValueError as error:
        assert DISCOVERY_BUDGET_MESSAGE in str(error)
    assert time.monotonic() - started < 10.0  # was 500 million reach iterations


class _HookedMeta(type):
    """A metaclass whose attribute reads run (and fail) while armed."""

    armed = False
    calls: list = []

    def __getattribute__(cls, name):
        if type.__getattribute__(_HookedMeta, "armed"):
            _HookedMeta.calls.append(name)
            raise RuntimeError("metaclass hook ran")
        return super().__getattribute__(name)


class _Hooked(metaclass=_HookedMeta):
    pass


def _armed(call):
    _HookedMeta.calls.clear()
    _HookedMeta.armed = True
    try:
        return call()
    except Exception as error:  # noqa: BLE001 - the test compares the outcome after disarming
        return error
    finally:
        _HookedMeta.armed = False


def test_a_non_iterable_pair_names_its_type_without_running_a_metaclass_hook() -> None:
    outcome = _armed(lambda: _entry_flow(_Items(lambda: iter((_Hooked(),)))))
    assert type(outcome) is TypeError
    assert str(outcome) == "cannot unpack non-iterable _Hooked object"
    assert _HookedMeta.calls == []


def test_a_boxed_key_names_its_type_without_running_a_metaclass_hook() -> None:
    from contracts_policy_reference import _OpaqueKey

    key = _OpaqueKey(_Hooked())
    assert _armed(lambda: repr(key)) == "<_Hooked key>"
    assert _HookedMeta.calls == []


def _evaluate_with_extra_values(**extra: object) -> None:
    """Run the full evaluator on a native statement holding ``extra`` values.

    Every container from the policy down is native (exact tuple, dict, list), so discovery
    charges their entries and completion's first copy is free; completion runs before
    validation refuses the extra keys.
    """

    resource_value = reference("memory-1")
    native_statement = {
        "id": "s-1", "effect": "allow", "subject_id": "subject-a", "credential_id": None,
        "action": "read", "resource": resource_value.model_dump(mode="python"), **extra,
    }
    policy_value = policy(statement("seed", "allow", "read", reference("other")))
    object.__setattr__(policy_value, "statements", (native_statement,))
    evaluate(principal(), "read", resource_value, policy_value)


def test_a_list_that_grows_during_its_own_completion_is_charged_for_the_growth() -> None:
    import time

    from contracts_limits import DISCOVERY_BUDGET_MESSAGE

    grown = [0]
    host: list[object] = []

    class SelfAppending(tuple):
        def __iter__(self):  # yields nothing, appends itself to the list being copied
            if grown[0] < 400_000:  # bounded, so the pre-fix tree terminates (and accepts the growth)
                grown[0] += 1
                host.append(self)
            return iter(())

    host.append(SelfAppending())
    started = time.monotonic()
    with pytest.raises(ValueError, match=DISCOVERY_BUDGET_MESSAGE):
        _evaluate_with_extra_values(extra=host)
    assert time.monotonic() - started < 30
    assert grown[0] < 200_000  # stopped by the edge budget, not by the probe's own bound


def test_a_list_grown_by_an_earlier_iterator_is_charged_for_the_growth() -> None:
    from contracts_limits import DISCOVERY_BUDGET_MESSAGE

    target = [1, 2, 3]

    class Grower(tuple):
        def __iter__(self):  # runs during completion, after discovery charged ``target``'s 3 entries
            target.extend(range(200_000))
            return iter(())

    with pytest.raises(ValueError, match=DISCOVERY_BUDGET_MESSAGE):
        _evaluate_with_extra_values(grower=Grower(), target=target)  # completed in this order


def _model_with_disjoint_fields_set(stored: int, fields_set: int, *, overlap: bool = False):
    handle = reference("memory-1")
    storage = object.__getattribute__(handle, "__dict__")
    for index in range(stored):
        storage[f"stored_{index}"] = index
    names = {f"stored_{index}" if overlap else f"set_{index}" for index in range(fields_set)}
    object.__setattr__(handle, "__pydantic_fields_set__", names)
    return handle


def _evaluate_with_statement_resource(resource_model) -> None:
    """A native statement (exact tuple, dict) whose resource is ``resource_model``: discovery
    charges it, so completion's first copy of the model is free."""

    native_statement = {
        "id": "s-1", "effect": "allow", "subject_id": "subject-a", "credential_id": None,
        "action": "read", "resource": resource_model,
    }
    policy_value = policy(statement("seed", "allow", "read", reference("other")))
    object.__setattr__(policy_value, "statements", (native_statement,))
    evaluate(principal(), "read", reference("memory-1"), policy_value)


def test_fields_set_names_disjoint_from_stored_fields_are_charged_when_synthesized() -> None:
    """70,000 stored fields and 70,000 other fields-set names make 140,000 output fields:
    completion must charge the synthesized ones discovery did not pay for."""

    with pytest.raises(ValueError, match=DISCOVERY_BUDGET_MESSAGE):
        _evaluate_with_statement_resource(_model_with_disjoint_fields_set(70_000, 70_000))


def test_fields_set_names_that_name_stored_fields_are_not_charged_twice() -> None:
    """70,000 stored fields all named by the fields-set make 70,000 output fields: charged once that
    fits the 131,072-edge budget, charged twice (140,000) it would not."""

    with pytest.raises(ValidationError):  # refused for its unknown fields, not by the budget
        _evaluate_with_statement_resource(_model_with_disjoint_fields_set(70_000, 70_000, overlap=True))


def test_a_validation_error_stays_bounded_for_a_statement_with_many_invalid_fields() -> None:
    """100,000 unknown fields (inside the edge allowance) used to give about 20 MB of error text."""

    resource_value = reference("memory-1")
    native_statement = {
        "id": "s-1", "effect": "allow", "subject_id": "subject-a", "credential_id": None,
        "action": "read", "resource": resource_value.model_dump(mode="python"),
        **{f"unknown_{index}": "x" * 20 for index in range(100_000)},
    }
    policy_value = policy(statement("seed", "allow", "read", reference("other")))
    object.__setattr__(policy_value, "statements", (native_statement,))
    with pytest.raises(ValidationError) as caught:
        evaluate(principal(), "read", resource_value, policy_value)
    text = str(caught.value)
    assert len(text.encode()) < 2_000_000, len(text)
    errors = caught.value.errors()
    assert errors[0]["type"] == "extra_forbidden" and errors[0]["loc"][-1] == "unknown_0"
    assert errors[-1]["type"] == "too_many_errors" and "further validation errors omitted" in errors[-1]["msg"]


def test_a_small_validation_error_keeps_its_full_diagnostic() -> None:
    resource_value = reference("memory-1")
    native_statement = {
        "id": "s-1", "effect": "allow", "subject_id": "subject-a", "credential_id": None,
        "action": "read", "resource": resource_value.model_dump(mode="python"), "unknown": "visible",
    }
    policy_value = policy(statement("seed", "allow", "read", reference("other")))
    object.__setattr__(policy_value, "statements", (native_statement,))
    with pytest.raises(ValidationError) as caught:
        evaluate(principal(), "read", resource_value, policy_value)
    assert [(e["type"], e["input"]) for e in caught.value.errors()] == [("extra_forbidden", "visible")]


def test_a_reach_tuple_that_reuses_a_scanned_id_is_still_scanned(monkeypatch) -> None:
    """Scanned reach tuples are held by identity, not just by id.

    An id is reused once its object is released (a shallower visit can replace a reach tuple).
    The collision is forced here by having the module's ``id`` give the second reach tuple the id
    of the first: that tuple must still be scanned, so the mapping behind an already-visited list
    is captured at its exposure, before the parent iterator changes it.
    """

    import builtins

    import contracts_policy_reference as reference_module

    backing_a, backing_b = {"v": "a-first"}, {"v": "b-first"}
    proxy_a, proxy_b = MappingProxyType(backing_a), MappingProxyType(backing_b)
    list_a, list_b = [proxy_a], [proxy_b]

    def pairs():
        yield ("a", list_a)
        yield ("b", list_b)
        backing_b["v"] = "b-changed-after-exposure"  # runs only when the iterator is advanced again
        yield ("c", 1)

    owner = _Items(pairs)
    root = (owner, list_a, list_b)  # the walk visits both lists before the owner's pairs expose them

    real_id = builtins.id
    first = {}

    def colliding_id(obj):
        if type(obj) is tuple and len(obj) == 1 and obj[0] is proxy_a:
            first["id"] = real_id(obj)
        if type(obj) is tuple and len(obj) == 1 and obj[0] is proxy_b and "id" in first:
            return first["id"]
        return real_id(obj)

    monkeypatch.setattr(reference_module, "id", colliding_id, raising=False)
    completed = _entry_flow(root)
    assert completed[2][0] == {"v": "b-first"}  # captured at exposure, not after the change


# ── a statement exposed only by a tuple's public iterator keeps the mappings it exposes ──


def _statement_with_proxy_resource_and_extra(resource_id: str, flip_to: str | None):
    """An allow statement whose resource is a mapping proxy and whose (empty) extra storage is a Mapping whose
    ``items()`` rewrites the proxy's backing dict when it is read."""

    backing = {"kind": "memory", "id": resource_id, "tenant_id": "tenant-a", "resolver": "memory_by_id"}

    class _MutatingExtra(Mapping[str, object]):
        def __getitem__(self, key: str) -> object:
            raise KeyError(key)

        def __iter__(self) -> Iterator[str]:
            return iter(())

        def __len__(self) -> int:
            return 0

        def items(self):  # type: ignore[override]
            if flip_to is not None:
                backing["id"] = flip_to
            return iter(())

    value = statement("allow-it", "allow", "read", reference(resource_id))
    object.__setattr__(value, "resource", MappingProxyType(backing))
    _BASE_MODEL_EXTRA_DESCRIPTOR.__set__(value, _MutatingExtra())
    return value, backing


def _policy_with_native_statements(*statements) -> AccessPolicy:
    """The statements sit in the policy's own tuple: built first, then tampered, so entry discovery sees them."""

    policy_value = policy(statement("seed", "allow", "read", reference("other")))
    object.__setattr__(policy_value, "statements", tuple(statements))
    return policy_value


def _decide(policy_value) -> str:
    return evaluate(principal(), "read", reference("memory-1"), policy_value).outcome


def test_a_public_iterator_only_statement_cannot_have_its_resource_rewritten_by_its_extra_mapping() -> None:
    entry_visible, _ = _statement_with_proxy_resource_and_extra("other", "memory-1")
    public_only, backing = _statement_with_proxy_resource_and_extra("other", "memory-1")

    expected = _decide(_policy_with_native_statements(entry_visible))
    assert expected == "deny"  # the entry-visible statement keeps its original resource: no match
    assert _decide(_policy_with_public_only_statements(public_only)) == expected  # was "allow": the id flipped first
    assert backing["id"] == "memory-1"  # the callback did run; the captured resource was already frozen


def test_a_public_iterator_only_statement_with_an_honest_extra_mapping_is_still_granted() -> None:
    honest_entry, _ = _statement_with_proxy_resource_and_extra("memory-1", None)
    honest_public, _ = _statement_with_proxy_resource_and_extra("memory-1", None)
    assert _decide(_policy_with_native_statements(honest_entry)) == "allow"
    assert _decide(_policy_with_public_only_statements(honest_public)) == "allow"  # valid grants survive lazy discovery
    other, _ = _statement_with_proxy_resource_and_extra("other", None)
    assert _decide(_policy_with_public_only_statements(other)) == "deny"
