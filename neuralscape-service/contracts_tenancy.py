"""Candidate tenant lifecycle contracts with topology-neutral validation.

These value types describe control-plane intent and observations.  They do not
allocate resources, grant authority, or select a shared-plane or cell topology.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Literal

from pydantic import BaseModel, TypeAdapter, model_validator

from contracts_common import OpaqueId, SafeCounter, VersionedContract


TenantOperationKind = Literal[
    "provision", "suspend", "resume", "export", "restore", "upgrade", "delete"
]
"""Lifecycle operations whose policy and execution live outside this module."""

TenantDesiredState = Literal["active", "suspended", "deleted", "unchanged"]
"""The lifecycle result requested by an operation."""

TenantObservedState = Literal[
    "pending", "running", "succeeded", "failed", "cancelled", "unknown"
]
"""Observed progress, kept distinct from the requested lifecycle result."""


_DESIRED_STATE_BY_OPERATION: dict[TenantOperationKind, TenantDesiredState] = {
    "provision": "active",
    "suspend": "suspended",
    "resume": "active",
    "export": "unchanged",
    "restore": "active",
    "upgrade": "active",
    "delete": "deleted",
}

_OPAQUE_ID_ADAPTER = TypeAdapter(OpaqueId)
_SAFE_COUNTER_ADAPTER = TypeAdapter(SafeCounter)
_PYDANTIC_DICT_DESCRIPTOR = BaseModel.__dict__["__dict__"]
_PYDANTIC_EXTRA_DESCRIPTOR = BaseModel.__dict__["__pydantic_extra__"]


def _model_extra_storage(value: BaseModel) -> object:
    """Read Pydantic-owned extra storage without model instance dispatch."""

    try:
        return _PYDANTIC_EXTRA_DESCRIPTOR.__get__(value, type(value))
    except AttributeError:
        return None


class ResourceManifestReference(VersionedContract):
    """Tenant-bound reference to a resource manifest for one placement generation.

    A reference is neither proof that the manifest exists nor permission to read
    it.  The explicit tenant identity prevents a reference from being silently
    re-used in another tenant's placement.
    """

    tenant_id: OpaqueId
    manifest_id: OpaqueId
    manifest_revision: SafeCounter
    placement_generation: SafeCounter


def _validate_manifest_references(
    *,
    tenant_id: str,
    generation: int,
    references: tuple[ResourceManifestReference, ...],
) -> None:
    seen: set[tuple[str, int]] = set()
    for reference in references:
        if reference.tenant_id != tenant_id:
            raise ValueError("resource manifest tenant_id must match owning tenant_id")
        if reference.placement_generation != generation:
            raise ValueError(
                "resource manifest placement_generation must match owning generation"
            )
        key = (reference.manifest_id, reference.manifest_revision)
        if key in seen:
            raise ValueError("resource manifest references must be unique")
        seen.add(key)


class TenantPlacement(VersionedContract):
    """A topology-neutral placement generation and its resource manifests."""

    tenant_id: OpaqueId
    generation: SafeCounter
    resource_manifests: tuple[ResourceManifestReference, ...]

    @model_validator(mode="after")
    def validate_manifest_identity(self) -> TenantPlacement:
        _validate_manifest_references(
            tenant_id=self.tenant_id,
            generation=self.generation,
            references=self.resource_manifests,
        )
        return self


class TenantOperationState(VersionedContract):
    """Desired and observed state for one idempotently identified operation."""

    tenant_id: OpaqueId
    operation_id: OpaqueId
    operation: TenantOperationKind
    desired_state: TenantDesiredState
    observed_state: TenantObservedState
    placement_generation: SafeCounter
    resource_manifests: tuple[ResourceManifestReference, ...] = ()

    @model_validator(mode="after")
    def validate_operation_identity(self) -> TenantOperationState:
        expected = _DESIRED_STATE_BY_OPERATION[self.operation]
        if self.desired_state != expected:
            raise ValueError(f"{self.operation} requires desired_state={expected}")
        _validate_manifest_references(
            tenant_id=self.tenant_id,
            generation=self.placement_generation,
            references=self.resource_manifests,
        )
        return self


_ALLOWED_OBSERVED_TRANSITIONS: dict[
    TenantObservedState, frozenset[TenantObservedState]
] = {
    "unknown": frozenset({"pending", "running", "succeeded", "failed", "cancelled"}),
    "pending": frozenset({"running", "failed", "cancelled"}),
    "running": frozenset({"succeeded", "failed", "cancelled"}),
    "succeeded": frozenset(),
    "failed": frozenset(),
    "cancelled": frozenset(),
}


def _closed_graph_snapshot_session() -> tuple[
    Callable[[object], None],
    Callable[[object], object],
]:
    """Return native capture and public rebuild phases sharing one inventory."""

    frozen_models: dict[int, tuple[object, bool, object]] = {}
    frozen_containers: dict[int, object] = {}
    discovered: set[int] = set()

    def freeze_native_graph(item: object) -> None:
        """Discover model state and exact-container edges without callbacks."""

        item_type = type(item)
        if not (
            issubclass(item_type, BaseModel)
            or issubclass(item_type, dict)
            or issubclass(item_type, list)
            or issubclass(item_type, tuple)
        ):
            return

        identity = id(item)
        if identity in discovered:
            return
        discovered.add(identity)

        if issubclass(item_type, BaseModel):
            stored_entries = tuple(
                dict.items(_PYDANTIC_DICT_DESCRIPTOR.__get__(item, BaseModel))
            )
            extras = _model_extra_storage(item)
            if extras is not None and issubclass(type(extras), dict):
                frozen_extras: object = tuple(dict.items(extras))
                extras_are_native = True
            else:
                frozen_extras = extras
                extras_are_native = False
            frozen_models[identity] = (
                stored_entries,
                extras_are_native,
                frozen_extras,
            )
            for _name, field_value in stored_entries:
                freeze_native_graph(field_value)
            return

        if issubclass(item_type, dict):
            entries = tuple(dict.items(item))
            if item_type is dict:
                frozen_containers[identity] = entries
            for _key, field_value in entries:
                freeze_native_graph(field_value)
            return

        if issubclass(item_type, list):
            items = tuple(list.__iter__(item))
            if item_type is list:
                frozen_containers[identity] = items
        else:
            items = tuple(tuple.__iter__(item))
            if item_type is tuple:
                frozen_containers[identity] = items
        for child in items:
            freeze_native_graph(child)

    active: set[int] = set()

    def rebuild(item: object) -> object:
        item_type = type(item)
        if not (
            issubclass(item_type, BaseModel)
            or issubclass(item_type, dict)
            or issubclass(item_type, list)
            or issubclass(item_type, tuple)
        ):
            return item

        identity = id(item)
        if identity not in discovered:
            freeze_native_graph(item)
        if identity in active:
            raise ValueError("contract input graph must be acyclic")
        active.add(identity)
        try:
            if isinstance(item, BaseModel):
                stored_entries, extras_are_native, frozen_extras = frozen_models[
                    identity
                ]
                declared = item_type.model_fields
                if extras_are_native:
                    extra_entries = frozen_extras
                    observed_extra_length = len(extra_entries)
                    iterated_extra_names = tuple(
                        name for name, _field_value in extra_entries
                    )
                elif frozen_extras is None:
                    observed_extra_length = 0
                    iterated_extra_names = ()
                    extra_entries = ()
                elif not isinstance(frozen_extras, Mapping):
                    raise ValueError("contract extra storage must be a mapping")
                else:
                    observed_extra_length = len(frozen_extras)
                    iterated_extra_names = tuple(name for name in frozen_extras)
                    extra_entries = tuple(
                        entry for entry in frozen_extras.items()
                    )
                stored_names = {name for name, _field_value in stored_entries}
                undeclared = stored_names - declared.keys()
                if (
                    undeclared
                    or observed_extra_length
                    or iterated_extra_names
                    or extra_entries
                ):
                    raise ValueError("contract input contains undeclared fields")
                return {
                    name: rebuild(field_value)
                    for name, field_value in stored_entries
                }
            if isinstance(item, dict):
                entries = (
                    frozen_containers[identity]
                    if item_type is dict
                    else item.items()
                )
                return {
                    key: rebuild(field_value)
                    for key, field_value in entries
                }
            if isinstance(item, list):
                items = (
                    frozen_containers[identity]
                    if item_type is list
                    else item
                )
                return [rebuild(child) for child in items]
            items = (
                frozen_containers[identity]
                if item_type is tuple
                else item
            )
            return tuple(rebuild(child) for child in items)
        finally:
            active.remove(identity)

    return freeze_native_graph, rebuild


def _snapshot_closed_graph(value: object) -> object:
    """Copy a model/native graph without dropping stored undeclared fields."""

    freeze_native_graph, rebuild = _closed_graph_snapshot_session()
    freeze_native_graph(value)
    return rebuild(value)


def _validated_operation_snapshot(
    operation: TenantOperationState,
    *,
    rebuild: Callable[[object], object] | None = None,
) -> TenantOperationState:
    if not isinstance(operation, TenantOperationState):
        raise TypeError("operation must be a TenantOperationState")
    snapshot = (
        _snapshot_closed_graph(operation)
        if rebuild is None
        else rebuild(operation)
    )
    return TenantOperationState.model_validate(snapshot)


def _validated_placement_snapshot(placement: TenantPlacement) -> TenantPlacement:
    if not isinstance(placement, TenantPlacement):
        raise TypeError("placement must be a TenantPlacement")
    return TenantPlacement.model_validate(_snapshot_closed_graph(placement))


def validate_operation_transition(
    previous: TenantOperationState, current: TenantOperationState
) -> TenantOperationState:
    """Reject identity changes, generation rollback, and invalid progress changes."""

    if not isinstance(previous, TenantOperationState):
        raise TypeError("operation must be a TenantOperationState")

    freeze_native_graph, rebuild = _closed_graph_snapshot_session()
    previous_capture_error: Exception | None = None
    try:
        freeze_native_graph(previous)
    except Exception as error:
        previous_capture_error = error

    current_capture_error: Exception | None = None
    if isinstance(current, TenantOperationState):
        try:
            freeze_native_graph(current)
        except Exception as error:
            current_capture_error = error
    else:
        current_capture_error = TypeError(
            "operation must be a TenantOperationState"
        )

    if previous_capture_error is not None:
        raise previous_capture_error
    previous = _validated_operation_snapshot(previous, rebuild=rebuild)
    if current_capture_error is not None:
        raise current_capture_error
    current = _validated_operation_snapshot(current, rebuild=rebuild)

    immutable_fields = ("tenant_id", "operation_id", "operation", "desired_state")
    for field_name in immutable_fields:
        if getattr(previous, field_name) != getattr(current, field_name):
            raise ValueError(f"operation transition changed immutable {field_name}")

    if current.placement_generation < previous.placement_generation:
        raise ValueError("operation transition cannot decrease placement_generation")

    if current.observed_state == previous.observed_state:
        if current.observed_state in {"succeeded", "failed", "cancelled"}:
            if current != previous:
                raise ValueError("terminal operation republication must be identical")
        return current
    allowed = _ALLOWED_OBSERVED_TRANSITIONS[previous.observed_state]
    if current.observed_state not in allowed:
        raise ValueError(
            "invalid observed_state transition: "
            f"{previous.observed_state} -> {current.observed_state}"
        )
    return current


def validate_placement_publication(
    placement: TenantPlacement,
    *,
    expected_tenant_id: OpaqueId,
    current_generation: SafeCounter,
) -> TenantPlacement:
    """Accept only a publication for the exact tenant and current generation."""

    placement = _validated_placement_snapshot(placement)
    validated_tenant_id = _OPAQUE_ID_ADAPTER.validate_python(
        expected_tenant_id, strict=True
    )
    validated_generation = _SAFE_COUNTER_ADAPTER.validate_python(
        current_generation, strict=True
    )
    if placement.tenant_id != validated_tenant_id:
        raise ValueError("placement publication tenant_id does not match")
    if placement.generation != validated_generation:
        raise ValueError("placement publication generation is stale or unexpected")
    return placement


__all__ = [
    "ResourceManifestReference",
    "TenantDesiredState",
    "TenantObservedState",
    "TenantOperationKind",
    "TenantOperationState",
    "TenantPlacement",
    "validate_operation_transition",
    "validate_placement_publication",
]
