"""Engine capability declarations that keep operational facts distinct.

A manifest describes an implementation; it does not select a provider, grant
access, or prove current usability. Qualification references make a claim
reviewable but are not themselves proof that the referenced evidence is valid.
"""

from __future__ import annotations

from enum import Enum

from pydantic import model_validator

from contracts_common import ContractModel, OpaqueId, VersionedContract
from contracts_references import ReferenceHandle


class EngineAdapterKind(str, Enum):
    """The architectural seam implemented by an engine adapter."""

    VECTOR_STORE = "vector_store"
    GRAPH_STORE = "graph_store"
    MEMORY_ENGINE = "memory_engine"
    INFERENCE = "inference"


class CapabilityHealth(str, Enum):
    """Observed health of one configured operation."""

    UNKNOWN = "unknown"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class QualificationClaim(ContractModel):
    """Versioned references supporting a qualification claim for one operation.

    Validation preserves the claim's identity. It does not resolve the
    references or prove that the evidence qualifies the operation.
    """

    operation: OpaqueId
    profile_reference: ReferenceHandle
    profile_version: OpaqueId
    evidence_reference: ReferenceHandle
    evidence_version: OpaqueId


class CapabilityOperationState(ContractModel):
    """Independent deployment and qualification facts for one operation."""

    operation: OpaqueId
    supported: bool
    configured: bool
    health: CapabilityHealth
    qualification: QualificationClaim | None

    @model_validator(mode="after")
    def validate_state(self) -> "CapabilityOperationState":
        if self.configured and not self.supported:
            raise ValueError("an unsupported operation cannot be configured")
        if self.qualification is not None and not self.supported:
            raise ValueError("an unsupported operation cannot be qualified")
        if (
            self.qualification is not None
            and self.qualification.operation != self.operation
        ):
            raise ValueError("qualification operation must match the capability operation")
        if self.health is not CapabilityHealth.UNKNOWN and not self.configured:
            raise ValueError("an unconfigured operation must have unknown health")
        return self

    def has_declared_prerequisites(self) -> bool:
        """Return whether structural dispatch prerequisites are declared.

        This does not resolve the qualification evidence and therefore is not a
        runtime authorization or proof of qualification.
        """

        return (
            self.supported
            and self.configured
            and self.health is CapabilityHealth.HEALTHY
            and self.qualification is not None
        )


class CapabilityManifest(VersionedContract):
    """A versioned, operation-specific capability declaration."""

    implementation_id: OpaqueId
    implementation_version: OpaqueId
    adapter_kind: EngineAdapterKind
    contract_schema_versions: tuple[OpaqueId, ...]
    operations: tuple[CapabilityOperationState, ...]

    @model_validator(mode="after")
    def validate_unique_values(self) -> "CapabilityManifest":
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
    """Exact contract and qualification profile required for an operation."""

    operation: OpaqueId
    contract_schema_version: OpaqueId
    require_configured: bool
    require_healthy: bool
    qualification_profile_reference: ReferenceHandle | None
    qualification_profile_version: OpaqueId | None

    @model_validator(mode="after")
    def validate_qualification_requirement(self) -> "CapabilityRequirement":
        supplied = (
            self.qualification_profile_reference is not None,
            self.qualification_profile_version is not None,
        )
        if supplied[0] != supplied[1]:
            raise ValueError("qualification profile reference and version are paired")
        return self


class CapabilityViolation(ContractModel):
    """A stable explanation of one unmet capability requirement."""

    operation: OpaqueId
    fact: OpaqueId


def validate_capability_requirements(
    manifest: CapabilityManifest,
    requirements: tuple[CapabilityRequirement, ...],
) -> tuple[CapabilityViolation, ...]:
    """Return structurally unmet facts without treating references as proof.

    An empty result means the declarations agree. Runtime dispatch must still
    resolve and evaluate the referenced qualification evidence.
    """

    violations: list[CapabilityViolation] = []
    for requirement in requirements:
        state = manifest.operation_state(requirement.operation)
        if state is None or not state.supported:
            violations.append(
                CapabilityViolation(operation=requirement.operation, fact="supported")
            )
            continue
        if requirement.contract_schema_version not in manifest.contract_schema_versions:
            violations.append(
                CapabilityViolation(
                    operation=requirement.operation,
                    fact="contract_schema_version",
                )
            )
        if requirement.require_configured and not state.configured:
            violations.append(
                CapabilityViolation(operation=requirement.operation, fact="configured")
            )
        if requirement.require_healthy and state.health is not CapabilityHealth.HEALTHY:
            violations.append(
                CapabilityViolation(operation=requirement.operation, fact="healthy")
            )
        if requirement.qualification_profile_reference is not None:
            claim = state.qualification
            if claim is None:
                violations.append(
                    CapabilityViolation(operation=requirement.operation, fact="qualified")
                )
            elif (
                claim.profile_reference != requirement.qualification_profile_reference
                or claim.profile_version != requirement.qualification_profile_version
            ):
                violations.append(
                    CapabilityViolation(
                        operation=requirement.operation,
                        fact="qualification_profile",
                    )
                )
    return tuple(violations)


__all__ = [
    "CapabilityHealth",
    "CapabilityManifest",
    "CapabilityOperationState",
    "CapabilityRequirement",
    "CapabilityViolation",
    "EngineAdapterKind",
    "QualificationClaim",
    "validate_capability_requirements",
]
