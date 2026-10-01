"""Deterministic, side-effect-free reference policy evaluator."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import NamedTuple, cast

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


_BASE_MODEL_DICT_DESCRIPTOR = vars(BaseModel)["__dict__"]
_BASE_MODEL_EXTRA_DESCRIPTOR = vars(BaseModel)["__pydantic_extra__"]
_BASE_MODEL_FIELDS_SET_DESCRIPTOR = vars(BaseModel)["__pydantic_fields_set__"]


class _CapturedFailure(NamedTuple):
    error: Exception


class _ModelInventory(NamedTuple):
    owner: BaseModel
    stored_storage: object
    stored_items: tuple[tuple[object, object], ...] | None
    extra_storage: object
    extra_items: tuple[tuple[object, object], ...] | None
    fields_set_storage: object
    fields_set_names: tuple[object, ...] | None


class _MappingEntry(NamedTuple):
    key: object
    value: object


class _MappingInventory(NamedTuple):
    owner: Mapping[object, object]
    entries: tuple[_MappingEntry | _CapturedFailure, ...] | _CapturedFailure


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


def _fields_set_names(value: object) -> tuple[object, ...]:
    """Capture native set backing without changing other iterable fallbacks."""

    if issubclass(type(value), set):
        return tuple(set.__iter__(value))
    return tuple(item for item in cast(Iterable[object], value))


def _normalize_model_names(names: tuple[object, ...]) -> tuple[str, ...]:
    """Normalize model-owned string names without invoking subclass hooks."""

    normalized: list[str] = []
    seen: set[str] = set()
    for candidate in names:
        if not issubclass(type(candidate), str):
            raise ValueError("contract input model field names must be strings")
        name = str.__str__(candidate)
        if name in seen:
            raise ValueError("contract input contains colliding model field names")
        seen.add(name)
        normalized.append(name)
    return tuple(normalized)


def _normalize_model_items(
    items: tuple[tuple[object, object], ...],
) -> tuple[tuple[str, object], ...]:
    """Pair native model values with callback-free normalized names."""

    names = _normalize_model_names(tuple(name for name, _ in items))
    return tuple((name, item) for name, (_, item) in zip(names, items))


def _capture_model_store(descriptor: object, value: BaseModel) -> object:
    """Capture a model-owned slot while deferring its existing error order."""

    try:
        return descriptor.__get__(value, BaseModel)  # type: ignore[attr-defined]
    except Exception as error:
        return _CapturedFailure(error)


def _capture_model_inventory(value: BaseModel) -> _ModelInventory:
    """Capture native inventories without invoking supported fallback protocols."""

    stored_storage = _capture_model_store(_BASE_MODEL_DICT_DESCRIPTOR, value)
    stored_items = (
        tuple(dict.items(stored_storage))
        if _is_native_dict(stored_storage)
        else None
    )
    extra_storage = _capture_model_store(_BASE_MODEL_EXTRA_DESCRIPTOR, value)
    extra_items = (
        tuple(dict.items(extra_storage))
        if _is_native_dict(extra_storage)
        else (() if extra_storage is None else None)
    )
    fields_set_storage = _capture_model_store(
        _BASE_MODEL_FIELDS_SET_DESCRIPTOR,
        value,
    )
    fields_set_names = (
        tuple(set.__iter__(fields_set_storage))
        if issubclass(type(fields_set_storage), set)
        else None
    )
    return _ModelInventory(
        value,
        stored_storage,
        stored_items,
        extra_storage,
        extra_items,
        fields_set_storage,
        fields_set_names,
    )


def _validated_model_inventory(
    value: BaseModel,
    inventories: dict[int, _ModelInventory],
) -> _ModelInventory | None:
    """Return only an inventory owned by this exact model instance."""

    inventory = inventories.get(id(value))
    if inventory is not None and inventory.owner is not value:
        raise ValueError("contract input model inventory owner mismatch")
    return inventory


def _retain_mapping_owner(
    value: Mapping[object, object],
    owners: dict[int, Mapping[object, object]],
) -> None:
    """Retain and authenticate a mapping reached without public callbacks."""

    identity = id(value)
    owner = owners.get(identity)
    if owner is not None and owner is not value:
        raise ValueError("contract input mapping owner mismatch")
    owners[identity] = value


def _validated_mapping_inventory(
    value: Mapping[object, object],
    inventories: dict[int, _MappingInventory],
) -> _MappingInventory | None:
    """Return only an inventory owned by this exact mapping instance."""

    inventory = inventories.get(id(value))
    if inventory is not None and inventory.owner is not value:
        raise ValueError("contract input mapping inventory owner mismatch")
    return inventory


def _freeze_reachable_model_inventories(
    value: object,
    inventories: dict[int, _ModelInventory],
    visited_containers: set[int],
    mapping_owners: dict[int, Mapping[object, object]] | None = None,
) -> None:
    """Freeze natively reachable models before callback-bearing traversal."""

    if mapping_owners is None:
        mapping_owners = {}

    if issubclass(type(value), BaseModel):
        identity = id(value)
        if _validated_model_inventory(value, inventories) is not None:
            return
        inventory = _capture_model_inventory(value)
        inventories[identity] = inventory
        if inventory.stored_items is not None:
            for _, item in inventory.stored_items:
                _freeze_reachable_model_inventories(
                    item,
                    inventories,
                    visited_containers,
                    mapping_owners,
                )
        elif issubclass(type(inventory.stored_storage), Mapping):
            _retain_mapping_owner(inventory.stored_storage, mapping_owners)
        if inventory.extra_items is not None:
            for _, item in inventory.extra_items:
                _freeze_reachable_model_inventories(
                    item,
                    inventories,
                    visited_containers,
                    mapping_owners,
                )
        elif issubclass(type(inventory.extra_storage), Mapping):
            _retain_mapping_owner(inventory.extra_storage, mapping_owners)
        return

    if _is_native_dict(value):
        identity = id(value)
        if identity in visited_containers:
            return
        visited_containers.add(identity)
        for _, item in dict.items(value):
            _freeze_reachable_model_inventories(
                item,
                inventories,
                visited_containers,
                mapping_owners,
            )
        return

    if issubclass(type(value), Mapping):
        _retain_mapping_owner(value, mapping_owners)
        return

    if issubclass(type(value), list):
        identity = id(value)
        if identity in visited_containers:
            return
        visited_containers.add(identity)
        for item in list.__iter__(value):
            _freeze_reachable_model_inventories(
                item,
                inventories,
                visited_containers,
                mapping_owners,
            )
        return

    if issubclass(type(value), tuple):
        identity = id(value)
        if identity in visited_containers:
            return
        visited_containers.add(identity)
        for item in tuple.__iter__(value):
            _freeze_reachable_model_inventories(
                item,
                inventories,
                visited_containers,
                mapping_owners,
            )
        return


def _freeze_mapping_inventories(
    owners: dict[int, Mapping[object, object]],
    mapping_inventories: dict[int, _MappingInventory],
    model_inventories: dict[int, _ModelInventory],
    visited_containers: set[int],
) -> None:
    """Observe each retained public mapping inventory exactly once."""

    while True:
        pending = tuple(
            owner
            for owner in owners.values()
            if _validated_mapping_inventory(owner, mapping_inventories) is None
        )
        if not pending:
            return

        for owner in pending:
            try:
                raw_entries: tuple[object, ...] = _mapping_items(owner)
            except Exception as error:
                mapping_inventories[id(owner)] = _MappingInventory(
                    owner,
                    _CapturedFailure(error),
                )
                continue

            entries: list[_MappingEntry | _CapturedFailure] = []
            for raw_entry in raw_entries:
                try:
                    key, item = raw_entry  # type: ignore[misc]
                except Exception as error:
                    entries.append(_CapturedFailure(error))
                    continue
                entry = _MappingEntry(key, item)
                entries.append(entry)
                _freeze_reachable_model_inventories(
                    item,
                    model_inventories,
                    visited_containers,
                    owners,
                )
            mapping_inventories[id(owner)] = _MappingInventory(
                owner,
                tuple(entries),
            )


def _raise_captured_failure(value: object) -> object:
    if type(value) is _CapturedFailure:
        raise value.error
    return value


def _retained_mapping_entries(
    value: Mapping[object, object],
    mapping_inventories: dict[int, _MappingInventory],
    model_inventories: dict[int, _ModelInventory],
) -> tuple[_MappingEntry | _CapturedFailure, ...]:
    """Return one authenticated public inventory, capturing it if newly seen."""

    inventory = _validated_mapping_inventory(value, mapping_inventories)
    if inventory is None:
        owners: dict[int, Mapping[object, object]] = {}
        _retain_mapping_owner(value, owners)
        _freeze_mapping_inventories(
            owners,
            mapping_inventories,
            model_inventories,
            set(),
        )
        inventory = _validated_mapping_inventory(value, mapping_inventories)
        if inventory is None:
            raise ValueError("contract input mapping inventory is missing")
    return cast(
        tuple[_MappingEntry | _CapturedFailure, ...],
        _raise_captured_failure(inventory.entries),
    )


def _retained_mapping_items(
    value: Mapping[object, object],
    mapping_inventories: dict[int, _MappingInventory],
    model_inventories: dict[int, _ModelInventory],
) -> tuple[tuple[object, object], ...]:
    """Rebuild model-owned items while retaining per-entry error priority."""

    items: list[tuple[object, object]] = []
    for entry in _retained_mapping_entries(
        value,
        mapping_inventories,
        model_inventories,
    ):
        retained = cast(_MappingEntry, _raise_captured_failure(entry))
        items.append((retained.key, retained.value))
    return tuple(items)


def _complete_contract_input(
    value: object,
    active_ids: set[int],
    inventories: dict[int, _ModelInventory],
    mapping_inventories: dict[int, _MappingInventory] | None = None,
) -> object:
    """Copy a native input graph without normalizing away invalid data."""

    if mapping_inventories is None:
        mapping_inventories = {}

    if issubclass(type(value), BaseModel):
        identity = id(value)
        if identity in active_ids:
            raise ValueError("cyclic contract input is not supported")
        active_ids.add(identity)
        try:
            inventory = _validated_model_inventory(value, inventories)
            if inventory is None:
                _freeze_reachable_model_inventories(value, inventories, set())
                inventory = _validated_model_inventory(value, inventories)
                if inventory is None:
                    raise ValueError("contract input model inventory is missing")
            stored_storage = _raise_captured_failure(inventory.stored_storage)
            native_stored_items = (
                inventory.stored_items
                if inventory.stored_items is not None
                else _retained_mapping_items(
                    stored_storage,  # type: ignore[arg-type]
                    mapping_inventories,
                    inventories,
                )
            )
            stored_items = _normalize_model_items(native_stored_items)
            stored_values = dict(stored_items)
            declared_fields = type(value).model_fields
            extra_values = _raise_captured_failure(inventory.extra_storage)
            if extra_values is not None:
                if not (
                    _is_native_dict(extra_values)
                    or issubclass(type(extra_values), Mapping)
                ):
                    raise ValueError(
                        "contract input extra storage must be a mapping"
                    )
                native_extra_items = (
                    inventory.extra_items
                    if inventory.extra_items is not None
                    else _retained_mapping_items(
                        extra_values,  # type: ignore[arg-type]
                        mapping_inventories,
                        inventories,
                    )
                )
                extra_items = _normalize_model_items(native_extra_items)
                for key, item in extra_items:
                    if key in stored_values or key in declared_fields:
                        raise ValueError(
                            "contract input contains duplicate stored fields"
                        )
                    stored_values[key] = item

            # An unchecked copy normally stores its update in ``__dict__``.
            # Retain even an anomalous set-only field so closed-model validation
            # cannot silently erase evidence of unknown input semantics.
            fields_set_storage = _raise_captured_failure(
                inventory.fields_set_storage
            )
            fields_set_names = _normalize_model_names(
                inventory.fields_set_names
                if inventory.fields_set_names is not None
                else _fields_set_names(fields_set_storage)
            )
            for field_name in fields_set_names:
                if (
                    field_name not in declared_fields
                    and field_name not in stored_values
                ):
                    stored_values[field_name] = None

            return {
                key: _complete_contract_input(
                    item,
                    active_ids,
                    inventories,
                    mapping_inventories,
                )
                for key, item in stored_values.items()
            }
        finally:
            active_ids.remove(identity)

    if issubclass(type(value), Mapping):
        identity = id(value)
        if identity in active_ids:
            raise ValueError("cyclic contract input is not supported")
        active_ids.add(identity)
        try:
            # Keys are deliberately not stringified or otherwise normalized.
            if _is_native_dict(value):
                entries = tuple(
                    _MappingEntry(key, item)
                    for key, item in _mapping_items(value)
                )
            else:
                entries = _retained_mapping_entries(
                    value,
                    mapping_inventories,
                    inventories,
                )
            completed: dict[object, object] = {}
            for entry in entries:
                retained = cast(
                    _MappingEntry,
                    _raise_captured_failure(entry),
                )
                completed[retained.key] = _complete_contract_input(
                    retained.value,
                    active_ids,
                    inventories,
                    mapping_inventories,
                )
            return completed
        finally:
            active_ids.remove(identity)

    if issubclass(type(value), list):
        identity = id(value)
        if identity in active_ids:
            raise ValueError("cyclic contract input is not supported")
        active_ids.add(identity)
        try:
            return [
                _complete_contract_input(
                    item,
                    active_ids,
                    inventories,
                    mapping_inventories,
                )
                for item in value
            ]
        finally:
            active_ids.remove(identity)

    if issubclass(type(value), tuple):
        identity = id(value)
        if identity in active_ids:
            raise ValueError("cyclic contract input is not supported")
        active_ids.add(identity)
        try:
            return tuple(
                _complete_contract_input(
                    item,
                    active_ids,
                    inventories,
                    mapping_inventories,
                )
                for item in value
            )
        finally:
            active_ids.remove(identity)

    return value


def _validated_principal_snapshot(
    principal: PrincipalContext,
    inventories: dict[int, _ModelInventory],
    mapping_inventories: dict[int, _MappingInventory],
) -> PrincipalContext:
    return PrincipalContext.model_validate(
        _complete_contract_input(
            principal,
            set(),
            inventories,
            mapping_inventories,
        )
    )


def _validated_evaluation_snapshot(
    evaluation: PolicyEvaluationInput,
    inventories: dict[int, _ModelInventory],
    mapping_inventories: dict[int, _MappingInventory],
) -> PolicyEvaluationInput:
    return PolicyEvaluationInput.model_validate(
        _complete_contract_input(
            evaluation,
            set(),
            inventories,
            mapping_inventories,
        )
    )


def _validated_policy_snapshot(
    policy: AccessPolicy,
    inventories: dict[int, _ModelInventory],
    mapping_inventories: dict[int, _MappingInventory],
) -> AccessPolicy:
    return AccessPolicy.model_validate(
        _complete_contract_input(
            policy,
            set(),
            inventories,
            mapping_inventories,
        )
    )


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

    inventories: dict[int, _ModelInventory] = {}
    mapping_owners: dict[int, Mapping[object, object]] = {}
    mapping_inventories: dict[int, _MappingInventory] = {}
    visited_containers: set[int] = set()
    for value in (principal, evaluation, policy):
        _freeze_reachable_model_inventories(
            value,
            inventories,
            visited_containers,
            mapping_owners,
        )
    _freeze_mapping_inventories(
        mapping_owners,
        mapping_inventories,
        inventories,
        visited_containers,
    )
    principal = _validated_principal_snapshot(
        principal,
        inventories,
        mapping_inventories,
    )
    evaluation = _validated_evaluation_snapshot(
        evaluation,
        inventories,
        mapping_inventories,
    )
    policy = _validated_policy_snapshot(
        policy,
        inventories,
        mapping_inventories,
    )

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

    inventories: dict[int, _ModelInventory] = {}
    mapping_owners: dict[int, Mapping[object, object]] = {}
    mapping_inventories: dict[int, _MappingInventory] = {}
    visited_containers: set[int] = set()
    _freeze_reachable_model_inventories(
        decision,
        inventories,
        visited_containers,
        mapping_owners,
    )
    _freeze_mapping_inventories(
        mapping_owners,
        mapping_inventories,
        inventories,
        visited_containers,
    )
    return PolicyDecision.model_validate(
        _complete_contract_input(
            decision,
            set(),
            inventories,
            mapping_inventories,
        )
    )


__all__ = ["evaluate_policy", "validate_policy_decision"]
