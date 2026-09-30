"""Substantive invariants for engine capability declarations."""

from enum import Enum

import contracts_engines
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


class FalseyExtraDict(dict):
    def __bool__(self):
        return False


class HiddenExtraKeysDict(dict):
    def __iter__(self):
        return iter(())


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


def test_public_capability_checks_revalidate_mutated_and_unchecked_graphs():
    operation = state()
    operation.qualification.operation = "different-operation"
    with pytest.raises(ValidationError, match="operation must match"):
        operation.has_declared_prerequisites()

    declared = manifest(state())
    copied_claim = declared.operations[0].qualification.model_copy(
        update={"operation": "different-operation"}
    )
    copied_state = declared.operations[0].model_copy(
        update={"qualification": copied_claim}
    )
    copied_manifest = declared.model_copy(update={"operations": (copied_state,)})
    with pytest.raises(ValidationError, match="operation must match"):
        validate_capability_requirements(copied_manifest, (requirement(),))


def test_capability_boundary_rejects_invalid_requirements_duplicates_and_references():
    declared = manifest(state())
    unpaired = requirement().model_copy(
        update={"qualification_profile_version": None}
    )
    with pytest.raises(ValidationError, match="are paired"):
        validate_capability_requirements(declared, (unpaired,))

    duplicate_manifest = declared.model_copy(
        update={"operations": (declared.operations[0], declared.operations[0])}
    )
    with pytest.raises(ValidationError, match="operation declarations must be unique"):
        validate_capability_requirements(duplicate_manifest, (requirement(),))

    invalid_reference = reference("runtime-profile").model_copy(update={"id": ""})
    invalid_requirement = requirement().model_copy(
        update={"qualification_profile_reference": invalid_reference}
    )
    with pytest.raises(ValidationError):
        validate_capability_requirements(declared, (invalid_requirement,))


def test_capability_boundary_rejects_unknown_fields_wrong_containers_and_cycles():
    declared = manifest(state())
    unknown_state = declared.operations[0].model_copy(
        update={"future_semantics": "deny"}
    )
    unknown_manifest = declared.model_copy(update={"operations": (unknown_state,)})
    with pytest.raises(ValueError, match="undeclared contract field"):
        validate_capability_requirements(unknown_manifest, (requirement(),))

    with pytest.raises(TypeError, match="must be a tuple"):
        validate_capability_requirements(declared, [requirement()])

    cycle = []
    cycle.append(cycle)
    cyclic_manifest = declared.model_copy(update={"operations": cycle})
    with pytest.raises(ValueError, match="cyclic contract graph"):
        validate_capability_requirements(cyclic_manifest, (requirement(),))


@pytest.mark.parametrize("extra_operation", ["retrieve", "shadow-operation"])
def test_capability_boundary_rejects_declared_fields_duplicated_in_extras(
    extra_operation,
):
    declared = manifest(state())
    duplicated = requirement()
    object.__setattr__(
        duplicated,
        "__pydantic_extra__",
        {"operation": extra_operation},
    )

    with pytest.raises(
        ValueError,
        match=r"malformed contract extras.*duplicate field\(s\): operation",
    ):
        validate_capability_requirements(declared, (duplicated,))


def test_capability_boundary_rejects_declared_extra_when_field_is_not_stored():
    declared = manifest(state())
    moved = requirement()
    del moved.__dict__["operation"]
    object.__setattr__(moved, "__pydantic_extra__", {"operation": "retrieve"})

    with pytest.raises(
        ValueError,
        match=r"malformed contract extras.*duplicate field\(s\): operation",
    ):
        validate_capability_requirements(declared, (moved,))


def test_capability_boundary_checks_defaulted_subclass_declarations_in_extras():
    class ExtendedRequirement(CapabilityRequirement):
        extension_mode: str = "strict"

    declared = manifest(state())
    values = requirement().model_dump()

    normally_absent = ExtendedRequirement(**values)
    del normally_absent.__dict__["extension_mode"]
    assert validate_capability_requirements(declared, (normally_absent,)) == ()

    moved_default = ExtendedRequirement(**values)
    del moved_default.__dict__["extension_mode"]
    object.__setattr__(
        moved_default,
        "__pydantic_extra__",
        {"extension_mode": "shadow"},
    )
    with pytest.raises(
        ValueError,
        match=r"malformed contract extras.*duplicate field\(s\): extension_mode",
    ):
        validate_capability_requirements(declared, (moved_default,))


@pytest.mark.parametrize("extras", [None, {}])
def test_capability_boundary_preserves_absent_or_empty_extra_storage(extras):
    declared = manifest(state())
    valid_requirement = requirement()
    object.__setattr__(valid_requirement, "__pydantic_extra__", extras)

    assert validate_capability_requirements(declared, (valid_requirement,)) == ()


def test_capability_boundary_still_rejects_nonconflicting_unknown_extras():
    declared = manifest(state())
    unknown = requirement()
    object.__setattr__(unknown, "__pydantic_extra__", {"future_semantics": "deny"})

    with pytest.raises(ValueError, match="undeclared contract field.*future_semantics"):
        validate_capability_requirements(declared, (unknown,))


@pytest.mark.parametrize(
    "extras_type",
    [FalseyExtraDict, HiddenExtraKeysDict],
    ids=["falsey-dict-subclass", "hidden-keys-dict-subclass"],
)
@pytest.mark.parametrize(
    "payload",
    [
        {"operation": "shadow-operation"},
        {"future_semantics": "deny"},
    ],
    ids=["declared-field", "unknown-field"],
)
def test_capability_boundary_rejects_dict_subclass_extra_storage(
    extras_type,
    payload,
):
    declared = manifest(state())
    malformed = requirement()
    object.__setattr__(malformed, "__pydantic_extra__", extras_type(payload))

    with pytest.raises(ValueError, match="malformed contract extras"):
        validate_capability_requirements(declared, (malformed,))


@pytest.mark.parametrize("operation", ["", True, "x" * 257])
def test_operation_lookup_validates_opaque_scalar(operation):
    with pytest.raises(ValidationError):
        manifest(state()).operation_state(operation)


def test_requirement_validation_builds_one_index_from_one_manifest_snapshot(
    monkeypatch,
):
    declared = manifest(
        state(),
        state(operation="export", qualification=qualification("export")),
    )
    requirements = (
        requirement(),
        requirement(operation="export"),
        requirement(
            operation="delete",
            require_configured=False,
            require_healthy=False,
            qualification_profile_reference=None,
            qualification_profile_version=None,
        ),
    )
    counts = {"manifest_snapshots": 0, "operation_indexes": 0}
    original_snapshot = contracts_engines._validated_contract_snapshot
    original_lookup = contracts_engines._operation_state_lookup

    def counted_snapshot(value, expected_type, *, label):
        if expected_type is CapabilityManifest:
            counts["manifest_snapshots"] += 1
        return original_snapshot(value, expected_type, label=label)

    def counted_lookup(value):
        counts["operation_indexes"] += 1
        return original_lookup(value)

    monkeypatch.setattr(
        contracts_engines,
        "_validated_contract_snapshot",
        counted_snapshot,
    )
    monkeypatch.setattr(contracts_engines, "_operation_state_lookup", counted_lookup)

    violations = validate_capability_requirements(declared, requirements)

    assert [(item.operation, item.fact) for item in violations] == [
        ("delete", "supported")
    ]
    assert counts == {"manifest_snapshots": 1, "operation_indexes": 1}
