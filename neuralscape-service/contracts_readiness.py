"""Capability-specific readiness observations and publication validation.

Readiness is evidence about a named capability.  It is deliberately not
derived from process liveness, object construction, or another capability.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Literal

from pydantic import AwareDatetime, BaseModel, TypeAdapter, model_validator

from contracts_common import OpaqueId, SafeCounter, VersionedContract


ReadinessCapability = Literal[
    "api_acceptance", "exact_reads", "graph_reads", "worker_processing"
]
"""Independently probed service capabilities."""

CapabilityStatus = Literal[
    "disabled", "unconfigured", "healthy", "degraded", "unavailable", "unknown"
]
"""Operational status without coercing unknown or unavailable to healthy."""

_AWARE_DATETIME_ADAPTER = TypeAdapter(AwareDatetime)
_OPAQUE_ID_ADAPTER = TypeAdapter(OpaqueId)
_SAFE_COUNTER_ADAPTER = TypeAdapter(SafeCounter)


class CapabilityReadiness(VersionedContract):
    """One capability observation timestamped by its bounded real probe."""

    capability: ReadinessCapability
    status: CapabilityStatus
    observed_at: AwareDatetime
    detail_code: OpaqueId | None = None


class TenantReadinessPublication(VersionedContract):
    """Readiness observations bound to one tenant placement generation."""

    tenant_id: OpaqueId
    placement_generation: SafeCounter
    capabilities: tuple[CapabilityReadiness, ...]

    @model_validator(mode="after")
    def validate_capability_set(self) -> TenantReadinessPublication:
        if not self.capabilities:
            raise ValueError(
                "readiness publication must contain at least one capability"
            )
        names = [observation.capability for observation in self.capabilities]
        if len(names) != len(set(names)):
            raise ValueError("readiness publication capabilities must be unique")
        return self


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
            stored = vars(value)
            declared = type(value).model_fields
            undeclared = stored.keys() - declared.keys()
            extras = getattr(value, "__pydantic_extra__", None)
            if extras is not None and not isinstance(extras, Mapping):
                raise ValueError("contract extra storage must be a mapping")
            extra_entries = () if extras is None else tuple(extras.items())
            if undeclared or extra_entries:
                raise ValueError("contract input contains undeclared fields")
            return {
                name: _snapshot_closed_graph(field_value, active)
                for name, field_value in stored.items()
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


def _validated_observation_snapshot(
    observation: CapabilityReadiness,
) -> CapabilityReadiness:
    if not isinstance(observation, CapabilityReadiness):
        raise TypeError("observation must be a CapabilityReadiness")
    return CapabilityReadiness.model_validate(_snapshot_closed_graph(observation))


def _validated_publication_snapshot(
    publication: TenantReadinessPublication,
) -> TenantReadinessPublication:
    if not isinstance(publication, TenantReadinessPublication):
        raise TypeError("publication must be a TenantReadinessPublication")
    return TenantReadinessPublication.model_validate(
        _snapshot_closed_graph(publication)
    )


def validate_readiness_publication(
    publication: TenantReadinessPublication,
    *,
    expected_tenant_id: OpaqueId,
    current_placement_generation: SafeCounter,
) -> TenantReadinessPublication:
    """Reject cross-tenant and stale-generation readiness publication."""

    publication = _validated_publication_snapshot(publication)
    validated_tenant_id = _OPAQUE_ID_ADAPTER.validate_python(
        expected_tenant_id, strict=True
    )
    validated_generation = _SAFE_COUNTER_ADAPTER.validate_python(
        current_placement_generation, strict=True
    )
    if publication.tenant_id != validated_tenant_id:
        raise ValueError("readiness publication tenant_id does not match")
    if publication.placement_generation != validated_generation:
        raise ValueError(
            "readiness publication placement_generation is stale or unexpected"
        )
    return publication


def capability_is_ready(
    observation: CapabilityReadiness,
    *,
    evaluated_at: datetime,
    max_probe_age_seconds: SafeCounter,
) -> bool:
    """Evaluate explicit capability evidence at a supplied time and age limit.

    Structural validation cannot prove that the represented probe occurred.
    """

    observation = _validated_observation_snapshot(observation)
    validated_evaluation_time = _AWARE_DATETIME_ADAPTER.validate_python(
        evaluated_at, strict=True
    )
    validated_max_age = _SAFE_COUNTER_ADAPTER.validate_python(
        max_probe_age_seconds, strict=True
    )
    if validated_evaluation_time < observation.observed_at:
        raise ValueError("evaluated_at cannot precede observed_at")
    age_seconds = (
        validated_evaluation_time - observation.observed_at
    ).total_seconds()
    return observation.status == "healthy" and age_seconds <= validated_max_age


__all__ = [
    "CapabilityReadiness",
    "CapabilityStatus",
    "ReadinessCapability",
    "TenantReadinessPublication",
    "capability_is_ready",
    "validate_readiness_publication",
]
