"""Substantive invariants for engine capability declarations."""

from enum import Enum

import pytest
from pydantic import ValidationError

from contracts_engines import (
    CapabilityHealth,
    CapabilityManifest,
    CapabilityOperationState,
    CapabilityRequirement,
    EngineAdapterKind,
    QualificationClaim,
    validate_capability_requirements,
)
from contracts_references import ReferenceHandle


VERSION = "candidate-v1"


def reference(identifier: str) -> ReferenceHandle:
    return ReferenceHandle(
        kind="artifact",
        id=identifier,
        tenant_id="tenant-1",
        resolver="evidence",
    )


def qualification(
    operation: str = "retrieve",
    profile: str = "runtime-profile",
    profile_version: str = "profile-v1",
) -> QualificationClaim:
    return QualificationClaim(
        operation=operation,
        profile_reference=reference(profile),
        profile_version=profile_version,
        evidence_reference=reference("evidence-1"),
        evidence_version="evidence-v3",
    )


def state(**overrides) -> CapabilityOperationState:
    values = {
        "operation": "retrieve",
        "supported": True,
        "configured": True,
        "health": CapabilityHealth.HEALTHY,
        "qualification": qualification(),
    }
    values.update(overrides)
    return CapabilityOperationState(**values)


def manifest(*operations, versions=(VERSION,)) -> CapabilityManifest:
    return CapabilityManifest(
        schema_version=VERSION,
        implementation_id="engine-a",
        implementation_version="1.2.3",
        adapter_kind=EngineAdapterKind.MEMORY_ENGINE,
        contract_schema_versions=versions,
        operations=operations,
    )


def requirement(**overrides) -> CapabilityRequirement:
    values = {
        "operation": "retrieve",
        "contract_schema_version": VERSION,
        "require_configured": True,
        "require_healthy": True,
        "qualification_profile_reference": reference("runtime-profile"),
        "qualification_profile_version": "profile-v1",
    }
    values.update(overrides)
    return CapabilityRequirement(**values)


def test_public_enums_use_python_310_compatible_str_enum_pattern():
    assert issubclass(EngineAdapterKind, str)
    assert issubclass(EngineAdapterKind, Enum)
    assert issubclass(CapabilityHealth, str)
    assert EngineAdapterKind.VECTOR_STORE.value == "vector_store"


def test_declared_prerequisites_require_all_independent_facts():
    assert state().has_declared_prerequisites()
    assert not state(
        configured=False, health=CapabilityHealth.UNKNOWN
    ).has_declared_prerequisites()
    assert not state(health=CapabilityHealth.DEGRADED).has_declared_prerequisites()
    assert not state(qualification=None).has_declared_prerequisites()


def test_qualification_claim_can_outlive_configuration_but_is_not_runtime_eligibility():
    operation = state(
        configured=False,
        health=CapabilityHealth.UNKNOWN,
    )
    assert operation.qualification is not None
    assert not operation.has_declared_prerequisites()


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"supported": False, "configured": True}, "cannot be configured"),
        (
            {
                "supported": False,
                "configured": False,
                "qualification": qualification(),
            },
            "cannot be qualified",
        ),
        (
            {"configured": False, "health": CapabilityHealth.HEALTHY},
            "must have unknown health",
        ),
        (
            {"qualification": qualification(operation="export")},
            "operation must match",
        ),
    ],
)
def test_impossible_operational_states_are_rejected(overrides, message):
    with pytest.raises(ValidationError, match=message):
        state(**overrides)


def test_manifest_rejects_duplicate_operations_and_contract_versions():
    with pytest.raises(ValidationError, match="operation declarations must be unique"):
        manifest(state(), state(qualification=None))
    with pytest.raises(ValidationError, match="contract schema versions must be unique"):
        manifest(state(), versions=(VERSION, VERSION))


def test_requirement_profile_reference_and_version_are_paired():
    with pytest.raises(ValidationError, match="are paired"):
        requirement(qualification_profile_version=None)


def test_validator_binds_schema_and_qualification_profile_versions():
    declared = manifest(state(), versions=(VERSION,))
    assert validate_capability_requirements(declared, (requirement(),)) == ()

    violations = validate_capability_requirements(
        declared,
        (
            requirement(contract_schema_version="candidate-v2"),
            requirement(qualification_profile_version="profile-v2"),
        ),
    )
    assert [(item.operation, item.fact) for item in violations] == [
        ("retrieve", "contract_schema_version"),
        ("retrieve", "qualification_profile"),
    ]


def test_validator_reports_unconfigured_unhealthy_unqualified_and_unknown_operations():
    declared = manifest(
        state(
            operation="retrieve",
            configured=False,
            health=CapabilityHealth.UNKNOWN,
            qualification=None,
        )
    )
    violations = validate_capability_requirements(
        declared,
        (
            requirement(),
            requirement(
                operation="delete",
                require_configured=False,
                require_healthy=False,
                qualification_profile_reference=None,
                qualification_profile_version=None,
            ),
        ),
    )
    assert [(item.operation, item.fact) for item in violations] == [
        ("retrieve", "configured"),
        ("retrieve", "healthy"),
        ("retrieve", "qualified"),
        ("delete", "supported"),
    ]


def test_unknown_fields_are_rejected_by_common_contract_base():
    with pytest.raises(ValidationError, match="extra"):
        state(provider_name="not-a-capability-fact")
