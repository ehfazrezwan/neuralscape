"""Candidate tenant lifecycle contracts with topology-neutral validation.

These value types describe control-plane intent and observations.  They do not
allocate resources, grant authority, or select a shared-plane or cell topology.
"""

from __future__ import annotations

from typing import Literal

from pydantic import TypeAdapter, model_validator

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


def validate_operation_transition(
    previous: TenantOperationState, current: TenantOperationState
) -> TenantOperationState:
    """Reject identity changes, generation rollback, and invalid progress changes."""

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
