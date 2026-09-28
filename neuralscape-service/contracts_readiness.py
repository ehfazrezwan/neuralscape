"""Capability-specific readiness observations and publication validation.

Readiness is evidence about a named capability.  It is deliberately not
derived from process liveness, object construction, or another capability.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import AwareDatetime, TypeAdapter, model_validator

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


def _validated_observation_snapshot(
    observation: CapabilityReadiness,
) -> CapabilityReadiness:
    return CapabilityReadiness.model_validate(
        observation.model_dump(mode="python", round_trip=True, warnings=False)
    )


def _validated_publication_snapshot(
    publication: TenantReadinessPublication,
) -> TenantReadinessPublication:
    return TenantReadinessPublication.model_validate(
        publication.model_dump(mode="python", round_trip=True, warnings=False)
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
