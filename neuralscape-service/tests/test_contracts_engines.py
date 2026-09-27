"""Substantive invariants for engine capability declarations."""

import pytest
from pydantic import ValidationError

from contracts_engines import (
    CapabilityHealth,
    CapabilityManifest,
    CapabilityOperationState,
    CapabilityRequirement,
    EngineAdapterKind,
    validate_capability_requirements,
)


VERSION = "candidate-v1"


def state(**overrides):
    values = {
        "operation": "retrieve",
        "supported": True,
        "configured": True,
        "health": CapabilityHealth.HEALTHY,
        "qualified": True,
    }
    values.update(overrides)
    return CapabilityOperationState(**values)


def manifest(*operations):
    return CapabilityManifest(
        schema_version=VERSION,
        implementation_id="engine-a",
        implementation_version="1.2.3",
        adapter_kind=EngineAdapterKind.MEMORY_ENGINE,
        contract_schema_versions=(VERSION,),
        operations=operations,
    )


def test_runtime_eligibility_requires_all_four_independent_facts():
    assert state().is_runtime_eligible()
    assert not state(configured=False, health=CapabilityHealth.UNKNOWN).is_runtime_eligible()
    assert not state(health=CapabilityHealth.DEGRADED).is_runtime_eligible()
    assert not state(qualified=False).is_runtime_eligible()


def test_qualification_can_outlive_configuration_and_health_observation():
    operation = state(
        configured=False,
        health=CapabilityHealth.UNKNOWN,
        qualified=True,
    )

    assert operation.supported
    assert operation.qualified
    assert not operation.is_runtime_eligible()


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"supported": False, "configured": True}, "cannot be configured"),
        (
            {"supported": False, "configured": False, "qualified": True},
            "cannot be qualified",
        ),
        (
            {"configured": False, "health": CapabilityHealth.HEALTHY},
            "must have unknown health",
        ),
    ],
)
def test_impossible_operational_states_are_rejected(overrides, message):
    with pytest.raises(ValidationError, match=message):
        state(**overrides)


def test_manifest_rejects_duplicate_operations_instead_of_last_write_wins():
    with pytest.raises(ValidationError, match="operation declarations must be unique"):
        manifest(state(), state(qualified=False))


def test_manifest_rejects_duplicate_or_absent_contract_versions():
    base = {
        "schema_version": VERSION,
        "implementation_id": "engine-a",
        "implementation_version": "1",
        "adapter_kind": EngineAdapterKind.VECTOR_STORE,
        "operations": (),
    }
    with pytest.raises(ValidationError, match="at least one"):
        CapabilityManifest(**base, contract_schema_versions=())
    with pytest.raises(ValidationError, match="must be unique"):
        CapabilityManifest(**base, contract_schema_versions=(VERSION, VERSION))


def test_validator_reports_each_unmet_fact_without_approximating_operations():
    declared = manifest(
        state(operation="retrieve", health=CapabilityHealth.DEGRADED, qualified=False),
        state(
            operation="export",
            configured=False,
            health=CapabilityHealth.UNKNOWN,
            qualified=True,
        ),
    )
    requirements = (
        CapabilityRequirement(
            operation="retrieve",
            require_configured=True,
            require_healthy=True,
            require_qualified=True,
        ),
        CapabilityRequirement(
            operation="export",
            require_configured=True,
            require_healthy=False,
            require_qualified=True,
        ),
        CapabilityRequirement(
            operation="delete",
            require_configured=False,
            require_healthy=False,
            require_qualified=False,
        ),
    )

    violations = validate_capability_requirements(declared, requirements)

    assert [(item.operation, item.fact) for item in violations] == [
        ("retrieve", "healthy"),
        ("retrieve", "qualified"),
        ("export", "configured"),
        ("delete", "supported"),
    ]


def test_unknown_fields_are_rejected_by_the_common_contract_base():
    with pytest.raises(ValidationError, match="extra"):
        state(provider_name="not-a-capability-fact")
