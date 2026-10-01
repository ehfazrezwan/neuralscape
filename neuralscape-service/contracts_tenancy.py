"""Candidate tenant lifecycle contracts with topology-neutral validation.

These value types describe control-plane intent and observations.  They do not
allocate resources, grant authority, or select a shared-plane or cell topology.
"""

from __future__ import annotations

from collections.abc import Mapping
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


def _snapshot_closed_graph(
    value: object, active: set[int] | None = None
) -> object:
    """Copy a model/native graph without dropping stored undeclared fields."""

    if active is None:
        active = set()
    if not isinstance(value, (BaseModel, dict, list, tuple)):
        return value

    identity = id(value)
    if identity in active:
        raise ValueError("contract input graph must be acyclic")
    active.add(identity)
    try:
        if isinstance(value, BaseModel):
            stored_entries = tuple(
                dict.items(object.__getattribute__(value, "__dict__"))
            )
            declared = type(value).model_fields
            extras = _model_extra_storage(value)
            if extras is None:
                observed_extra_length = 0
                iterated_extra_names = ()
                extra_entries = ()
            elif issubclass(type(extras), dict):
                extra_entries = tuple(entry for entry in dict.items(extras))
                observed_extra_length = len(extra_entries)
                iterated_extra_names = tuple(
                    name for name, _field_value in extra_entries
                )
            elif not isinstance(extras, Mapping):
                raise ValueError("contract extra storage must be a mapping")
            else:
                observed_extra_length = len(extras)
                iterated_extra_names = tuple(name for name in extras)
                extra_entries = tuple(entry for entry in extras.items())
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
                name: _snapshot_closed_graph(field_value, active)
                for name, field_value in stored_entries
            }
        if isinstance(value, dict):
            return {
                key: _snapshot_closed_graph(field_value, active)
                for key, field_value in value.items()
            }
        if isinstance(value, list):
            return [_snapshot_closed_graph(item, active) for item in value]
        return tuple(_snapshot_closed_graph(item, active) for item in value)
    finally:
        active.remove(identity)


def _validated_operation_snapshot(
    operation: TenantOperationState,
) -> TenantOperationState:
    if not isinstance(operation, TenantOperationState):
        raise TypeError("operation must be a TenantOperationState")
    return TenantOperationState.model_validate(_snapshot_closed_graph(operation))


def _validated_placement_snapshot(placement: TenantPlacement) -> TenantPlacement:
    if not isinstance(placement, TenantPlacement):
        raise TypeError("placement must be a TenantPlacement")
    return TenantPlacement.model_validate(_snapshot_closed_graph(placement))


def validate_operation_transition(
    previous: TenantOperationState, current: TenantOperationState
) -> TenantOperationState:
    """Reject identity changes, generation rollback, and invalid progress changes."""

    previous = _validated_operation_snapshot(previous)
    current = _validated_operation_snapshot(current)

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
