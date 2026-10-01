"""Substantive invariants for engine capability declarations."""

from enum import Enum

import contracts_engines
import pytest
from pydantic import BaseModel, ValidationError

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

    def keys(self):
        return {}.keys()


class HiddenSet(set):
    def __iter__(self):
        return iter(())


class HiddenFrozenSet(frozenset):
    def __iter__(self):
        return iter(())


_MODEL_DICT_DESCRIPTOR = BaseModel.__dict__["__dict__"]
_MODEL_FIELDS_SET_DESCRIPTOR = BaseModel.__dict__["__pydantic_fields_set__"]
_MODEL_EXTRAS_DESCRIPTOR = BaseModel.__dict__["__pydantic_extra__"]


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


def test_capability_boundary_reads_model_storage_past_getattribute_override():
    class HiddenRequirement(CapabilityRequirement):
        def __getattribute__(self, name):
            if name == "__dict__":
                native = _MODEL_DICT_DESCRIPTOR.__get__(self, BaseModel)
                return {
                    key: item
                    for key, item in dict.items(native)
                    if key != "future_semantics"
                }
            return object.__getattribute__(self, name)

    declared = manifest(state())
    hidden = HiddenRequirement(**requirement().model_dump())
    native = _MODEL_DICT_DESCRIPTOR.__get__(hidden, BaseModel)
    dict.__setitem__(native, "future_semantics", "deny")

    assert "future_semantics" not in hidden.__dict__
    with pytest.raises(ValueError, match="undeclared contract field.*future_semantics"):
        validate_capability_requirements(declared, (hidden,))


@pytest.mark.parametrize("location", ["stored", "fields_set", "extras"])
def test_capability_boundary_reads_model_storage_past_subclass_properties(location):
    class PropertyHiddenRequirement(CapabilityRequirement):
        @property
        def __dict__(self):
            return {}

        @__dict__.setter
        def __dict__(self, value):
            _MODEL_DICT_DESCRIPTOR.__set__(self, value)

        @property
        def __pydantic_fields_set__(self):
            return set()

        @__pydantic_fields_set__.setter
        def __pydantic_fields_set__(self, value):
            _MODEL_FIELDS_SET_DESCRIPTOR.__set__(self, value)

        @property
        def __pydantic_extra__(self):
            return None

        @__pydantic_extra__.setter
        def __pydantic_extra__(self, value):
            _MODEL_EXTRAS_DESCRIPTOR.__set__(self, value)

    declared = manifest(state())
    hidden = PropertyHiddenRequirement(**requirement().model_dump())
    if location == "stored":
        native = _MODEL_DICT_DESCRIPTOR.__get__(hidden, BaseModel)
        dict.__setitem__(native, "future_semantics", "deny")
    elif location == "fields_set":
        native = set(_MODEL_FIELDS_SET_DESCRIPTOR.__get__(hidden, BaseModel))
        native.add("future_semantics")
        _MODEL_FIELDS_SET_DESCRIPTOR.__set__(hidden, native)
    else:
        _MODEL_EXTRAS_DESCRIPTOR.__set__(hidden, {"future_semantics": "deny"})

    with pytest.raises(ValueError, match="undeclared contract field.*future_semantics"):
        validate_capability_requirements(declared, (hidden,))


def test_capability_boundary_reads_nested_model_storage_past_property():
    class PropertyHiddenReference(ReferenceHandle):
        @property
        def __dict__(self):
            native = _MODEL_DICT_DESCRIPTOR.__get__(self, BaseModel)
            return {
                key: item
                for key, item in dict.items(native)
                if key != "future_semantics"
            }

        @__dict__.setter
        def __dict__(self, value):
            _MODEL_DICT_DESCRIPTOR.__set__(self, value)

    hidden_reference = PropertyHiddenReference(
        **reference("runtime-profile").model_dump()
    )
    native = _MODEL_DICT_DESCRIPTOR.__get__(hidden_reference, BaseModel)
    dict.__setitem__(native, "future_semantics", "deny")
    nested = requirement().model_copy(
        update={"qualification_profile_reference": hidden_reference}
    )

    with pytest.raises(
        ValueError,
        match=r"undeclared contract field.*capability requirements\[0\]"
        r"\.qualification_profile_reference.*future_semantics",
    ):
        validate_capability_requirements(manifest(state()), (nested,))


@pytest.mark.parametrize("fields_type", [HiddenSet, HiddenFrozenSet])
def test_capability_boundary_reads_native_field_set_backing(fields_type):
    declared = manifest(state())
    hidden = requirement()
    fields = set(_MODEL_FIELDS_SET_DESCRIPTOR.__get__(hidden, BaseModel))
    fields.add("future_semantics")
    _MODEL_FIELDS_SET_DESCRIPTOR.__set__(hidden, fields_type(fields))

    with pytest.raises(ValueError, match="undeclared contract field.*future_semantics"):
        validate_capability_requirements(declared, (hidden,))


def test_capability_boundary_reads_native_dict_subclass_backing():
    declared = manifest(state())
    hidden = requirement()
    native = _MODEL_DICT_DESCRIPTOR.__get__(hidden, BaseModel)
    replacement = HiddenExtraKeysDict(dict(dict.items(native)))
    dict.__setitem__(replacement, "future_semantics", "deny")
    _MODEL_DICT_DESCRIPTOR.__set__(hidden, replacement)

    with pytest.raises(ValueError, match="undeclared contract field.*future_semantics"):
        validate_capability_requirements(declared, (hidden,))


def test_capability_boundary_preserves_valid_hidden_backing_controls():
    class HiddenValidRequirement(CapabilityRequirement):
        @property
        def __dict__(self):
            return {}

        @__dict__.setter
        def __dict__(self, value):
            _MODEL_DICT_DESCRIPTOR.__set__(self, value)

        @property
        def __pydantic_fields_set__(self):
            return set()

        @__pydantic_fields_set__.setter
        def __pydantic_fields_set__(self, value):
            _MODEL_FIELDS_SET_DESCRIPTOR.__set__(self, value)

        @property
        def __pydantic_extra__(self):
            return None

        @__pydantic_extra__.setter
        def __pydantic_extra__(self, value):
            _MODEL_EXTRAS_DESCRIPTOR.__set__(self, value)

    valid = HiddenValidRequirement(**requirement().model_dump())
    native = _MODEL_DICT_DESCRIPTOR.__get__(valid, BaseModel)
    _MODEL_DICT_DESCRIPTOR.__set__(valid, HiddenExtraKeysDict(dict(dict.items(native))))
    native_fields = _MODEL_FIELDS_SET_DESCRIPTOR.__get__(valid, BaseModel)
    _MODEL_FIELDS_SET_DESCRIPTOR.__set__(valid, HiddenSet(native_fields))

    assert validate_capability_requirements(manifest(state()), (valid,)) == ()


def test_capability_boundary_freezes_parent_values_before_child_recursion():
    class RepairingReference(dict):
        parent = None

        def items(self):
            native = _MODEL_DICT_DESCRIPTOR.__get__(self.parent, BaseModel)
            dict.__setitem__(native, "qualification_profile_version", "profile-v1")
            return dict.items(self)

    invalid = requirement().model_copy(
        update={
            "qualification_profile_reference": RepairingReference(
                reference("runtime-profile").model_dump()
            ),
            "qualification_profile_version": None,
        }
    )
    invalid.qualification_profile_reference.parent = invalid

    with pytest.raises(ValidationError, match="are paired"):
        validate_capability_requirements(manifest(state()), (invalid,))

    assert invalid.qualification_profile_version == "profile-v1"


def test_capability_boundary_preserves_valid_child_mutation_control():
    class MutatingValidReference(dict):
        parent = None

        def items(self):
            native = _MODEL_DICT_DESCRIPTOR.__get__(self.parent, BaseModel)
            dict.__setitem__(native, "qualification_profile_version", "profile-v1")
            return dict.items(self)

    valid = requirement().model_copy(
        update={
            "qualification_profile_reference": MutatingValidReference(
                reference("runtime-profile").model_dump()
            )
        }
    )
    valid.qualification_profile_reference.parent = valid

    assert validate_capability_requirements(manifest(state()), (valid,)) == ()


def test_capability_boundary_preserves_missing_extra_storage():
    valid = requirement()
    _MODEL_EXTRAS_DESCRIPTOR.__delete__(valid)

    assert validate_capability_requirements(manifest(state()), (valid,)) == ()


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
