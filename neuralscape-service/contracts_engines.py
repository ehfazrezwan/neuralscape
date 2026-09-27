"""Engine capability declarations that keep distinct operational facts distinct.

A manifest describes an implementation; it does not select a provider, grant
access, or prove that a backend is currently usable.  Callers must evaluate all
of support, configuration, health, and qualification for each operation.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import model_validator

from contracts_common import ContractModel, OpaqueId, VersionedContract


class EngineAdapterKind(StrEnum):
    """The architectural seam implemented by an engine adapter."""

    VECTOR_STORE = "vector_store"
    GRAPH_STORE = "graph_store"
    MEMORY_ENGINE = "memory_engine"
    INFERENCE = "inference"


class CapabilityHealth(StrEnum):
    """Observed health of one configured operation."""

    UNKNOWN = "unknown"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class CapabilityOperationState(ContractModel):
    """Independent facts about one operation on one implementation.

    Qualification records evidence about semantics and can remain true while a
    deployment is unconfigured or unhealthy.  Conversely, a healthy operation
    is not necessarily qualified for a product profile.
    """

    operation: OpaqueId
    supported: bool
    configured: bool
    health: CapabilityHealth
    qualified: bool

    @model_validator(mode="after")
    def validate_state(self) -> CapabilityOperationState:
        if self.configured and not self.supported:
            raise ValueError("an unsupported operation cannot be configured")
        if self.qualified and not self.supported:
            raise ValueError("an unsupported operation cannot be qualified")
        if self.health is not CapabilityHealth.UNKNOWN and not self.configured:
            raise ValueError("an unconfigured operation must have unknown health")
        return self

    def is_runtime_eligible(self) -> bool:
        """Return whether every fact required for qualified dispatch is true."""

        return (
            self.supported
            and self.configured
            and self.health is CapabilityHealth.HEALTHY
            and self.qualified
        )


class CapabilityManifest(VersionedContract):
    """A versioned, operation-specific capability declaration."""

    implementation_id: OpaqueId
    implementation_version: OpaqueId
    adapter_kind: EngineAdapterKind
    contract_schema_versions: tuple[OpaqueId, ...]
    operations: tuple[CapabilityOperationState, ...]

    @model_validator(mode="after")
    def validate_unique_values(self) -> CapabilityManifest:
        if not self.contract_schema_versions:
            raise ValueError("at least one contract schema version is required")
        if len(set(self.contract_schema_versions)) != len(self.contract_schema_versions):
            raise ValueError("contract schema versions must be unique")
        operation_names = [operation.operation for operation in self.operations]
        if len(set(operation_names)) != len(operation_names):
            raise ValueError("operation declarations must be unique")
        return self

    def operation_state(self, operation: str) -> CapabilityOperationState | None:
        """Return the exact declared operation, without aliases or approximation."""

        return next(
            (state for state in self.operations if state.operation == operation),
            None,
        )


class CapabilityRequirement(ContractModel):
    """Facts a consumer requires for a named operation."""

    operation: OpaqueId
    require_configured: bool
    require_healthy: bool
    require_qualified: bool


class CapabilityViolation(ContractModel):
    """A stable explanation of one unmet capability requirement."""

    operation: OpaqueId
    fact: OpaqueId


def validate_capability_requirements(
    manifest: CapabilityManifest,
    requirements: tuple[CapabilityRequirement, ...],
) -> tuple[CapabilityViolation, ...]:
    """Return every unmet fact without silently substituting another operation."""

    violations: list[CapabilityViolation] = []
    for requirement in requirements:
        state = manifest.operation_state(requirement.operation)
        if state is None or not state.supported:
            violations.append(
                CapabilityViolation(operation=requirement.operation, fact="supported")
            )
            continue
        if requirement.require_configured and not state.configured:
            violations.append(
                CapabilityViolation(operation=requirement.operation, fact="configured")
            )
        if requirement.require_healthy and state.health is not CapabilityHealth.HEALTHY:
            violations.append(
                CapabilityViolation(operation=requirement.operation, fact="healthy")
            )
        if requirement.require_qualified and not state.qualified:
            violations.append(
                CapabilityViolation(operation=requirement.operation, fact="qualified")
            )
    return tuple(violations)
