"""Capability-specific readiness observations and publication validation.

Readiness is evidence about a named capability.  It is deliberately not
derived from process liveness, object construction, or another capability.
"""

from __future__ import annotations

from typing import Literal

from pydantic import model_validator

from contracts_common import OpaqueId, SafeCounter, VersionedContract


ReadinessCapability = Literal[
    "api_acceptance", "exact_reads", "graph_reads", "worker_processing"
]
"""Independently probed service capabilities."""

CapabilityStatus = Literal[
    "disabled", "unconfigured", "healthy", "degraded", "unavailable", "unknown"
]
"""Operational status without coercing unknown or unavailable to healthy."""


class CapabilityReadiness(VersionedContract):
    """One capability observation and the age of its bounded real probe."""

    capability: ReadinessCapability
    status: CapabilityStatus
    probe_age_seconds: SafeCounter | None
    detail_code: OpaqueId | None = None

    @model_validator(mode="after")
    def validate_probe_age(self) -> CapabilityReadiness:
        measured_statuses = {"healthy", "degraded", "unavailable"}
        if self.status in measured_statuses and self.probe_age_seconds is None:
            raise ValueError(f"{self.status} requires probe_age_seconds")
        if self.status in {"disabled", "unconfigured"}:
            if self.probe_age_seconds is not None:
                raise ValueError(f"{self.status} cannot claim a probe age")
        return self


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


def validate_readiness_publication(
    publication: TenantReadinessPublication,
    *,
    expected_tenant_id: OpaqueId,
    current_placement_generation: SafeCounter,
) -> TenantReadinessPublication:
    """Reject cross-tenant and stale-generation readiness publication."""

    if publication.tenant_id != expected_tenant_id:
        raise ValueError("readiness publication tenant_id does not match")
    if publication.placement_generation != current_placement_generation:
        raise ValueError(
            "readiness publication placement_generation is stale or unexpected"
        )
    return publication


def capability_is_ready(
    observation: CapabilityReadiness, *, max_probe_age_seconds: SafeCounter
) -> bool:
    """Return readiness for one capability under the caller's explicit age limit."""

    return (
        observation.status == "healthy"
        and observation.probe_age_seconds is not None
        and observation.probe_age_seconds <= max_probe_age_seconds
    )


__all__ = [
    "CapabilityReadiness",
    "CapabilityStatus",
    "ReadinessCapability",
    "TenantReadinessPublication",
    "capability_is_ready",
    "validate_readiness_publication",
]
