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


class ObservableTuple(tuple):
    def __new__(cls, values, *, callback=None, view=None):
        instance = super().__new__(cls, values)
        instance.callback = callback
        instance.view = view
        instance.calls = 0
        return instance

    def __iter__(self):
        self.calls += 1
        if self.callback is not None:
            self.callback()
        if self.view is not None:
            return iter(self.view)
        return tuple.__iter__(self)


_MODEL_DICT_DESCRIPTOR = BaseModel.__dict__["__dict__"]
_MODEL_FIELDS_SET_DESCRIPTOR = BaseModel.__dict__["__pydantic_fields_set__"]
_MODEL_EXTRAS_DESCRIPTOR = BaseModel.__dict__["__pydantic_extra__"]


class HookedFieldName(str):
    def __new__(cls, value):
        instance = super().__new__(cls, value)
        instance.events = []
        instance.armed = False
        instance.target = None
        instance.field_name = None
        instance.replacement = None
        return instance

    def arm(self, target=None, field_name=None, replacement=None):
        self.events.clear()
        self.target = target
        self.field_name = field_name
        self.replacement = replacement
        self.armed = True

    def _hook(self, name):
        if self.armed:
            self.events.append(name)
            if self.target is not None:
                native = _MODEL_DICT_DESCRIPTOR.__get__(self.target, BaseModel)
                dict.__setitem__(native, self.field_name, self.replacement)

    def __hash__(self):
        self._hook("hash")
        return str.__hash__(self)

    def __eq__(self, other):
        self._hook("eq")
        return str.__eq__(self, other)

    def __str__(self):
        self._hook("str")
        return str.__str__(self)


class DistinctFieldName(HookedFieldName):
    def __hash__(self):
        self._hook("hash")
        return object.__hash__(self)

    def __eq__(self, other):
        self._hook("eq")
        return self is other


class NonStringFieldName:
    def __init__(self):
        self.events = []
        self.armed = False

    def arm(self):
        self.events.clear()
        self.armed = True

    def _hook(self, name):
        if self.armed:
            self.events.append(name)

    def __hash__(self):
        self._hook("hash")
        return object.__hash__(self)

    def __eq__(self, other):
        self._hook("eq")
        return self is other

    def __str__(self):
        self._hook("str")
        return "non-string-field"

    def __repr__(self):
        self._hook("repr")
        return "NonStringFieldName()"


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


@pytest.mark.parametrize("tuple_field", ["contract_schema_versions", "operations"])
@pytest.mark.parametrize("public_surface", ["requirements", "operation_state"])
def test_capability_boundary_freezes_manifest_children_before_tuple_callbacks(
    tuple_field,
    public_surface,
):
    invalid = state().model_copy(update={"supported": False})

    def repair_child():
        native = _MODEL_DICT_DESCRIPTOR.__get__(invalid, BaseModel)
        dict.__setitem__(native, "supported", True)

    declared = manifest(state()).model_copy(update={"operations": (invalid,)})
    if tuple_field == "contract_schema_versions":
        observed = ObservableTuple((VERSION,), callback=repair_child)
        declared = declared.model_copy(
            update={"contract_schema_versions": observed}
        )
    else:
        observed = ObservableTuple((invalid,), callback=repair_child)
        declared = declared.model_copy(update={"operations": observed})

    with pytest.raises(ValidationError, match="cannot be configured"):
        if public_surface == "requirements":
            validate_capability_requirements(declared, (requirement(),))
        else:
            declared.operation_state("retrieve")

    assert observed.calls == 1
    assert invalid.supported is True


def test_capability_boundary_freezes_requirements_before_tuple_callback():
    invalid = requirement().model_copy(
        update={"qualification_profile_version": None}
    )

    def repair_child():
        native = _MODEL_DICT_DESCRIPTOR.__get__(invalid, BaseModel)
        dict.__setitem__(native, "qualification_profile_version", "profile-v1")

    requirements = ObservableTuple(
        (requirement(operation="export"), invalid),
        callback=repair_child,
    )

    with pytest.raises(ValidationError, match="are paired"):
        validate_capability_requirements(manifest(state()), requirements)

    assert requirements.calls == 1
    assert invalid.qualification_profile_version == "profile-v1"


@pytest.mark.parametrize("tuple_field", ["contract_schema_versions", "operations"])
def test_capability_boundary_freezes_requirements_before_manifest_callbacks(
    tuple_field,
):
    ordinary = requirement().model_copy(
        update={"qualification_profile_version": None}
    )
    with pytest.raises(ValidationError, match="are paired"):
        validate_capability_requirements(manifest(state()), (ordinary,))

    invalid = requirement().model_copy(
        update={"qualification_profile_version": None}
    )

    def repair_requirement():
        native = _MODEL_DICT_DESCRIPTOR.__get__(invalid, BaseModel)
        dict.__setitem__(
            native,
            "qualification_profile_version",
            "profile-v1",
        )

    declared = manifest(state())
    observed = ObservableTuple(
        getattr(declared, tuple_field),
        callback=repair_requirement,
    )
    native = _MODEL_DICT_DESCRIPTOR.__get__(declared, BaseModel)
    dict.__setitem__(native, tuple_field, observed)

    with pytest.raises(ValidationError, match="are paired"):
        validate_capability_requirements(declared, (invalid,))

    assert observed.calls == 1
    assert invalid.qualification_profile_version == "profile-v1"

    callbacks = []
    valid_manifest = manifest(state())
    benign = ObservableTuple(
        getattr(valid_manifest, tuple_field),
        callback=lambda: callbacks.append(tuple_field),
    )
    valid_native = _MODEL_DICT_DESCRIPTOR.__get__(valid_manifest, BaseModel)
    dict.__setitem__(valid_native, tuple_field, benign)

    assert validate_capability_requirements(
        valid_manifest,
        (requirement(),),
    ) == ()
    assert benign.calls == 1
    assert callbacks == [tuple_field]


def test_capability_boundary_preserves_sibling_type_error_priority():
    invalid_state = state().model_copy(update={"supported": False})
    invalid_manifest = manifest(state()).model_copy(
        update={"operations": (invalid_state,)}
    )

    with pytest.raises(ValidationError, match="cannot be configured"):
        validate_capability_requirements(invalid_manifest, [])

    with pytest.raises(TypeError, match="capability requirements must be a tuple"):
        validate_capability_requirements(manifest(state()), [])


def test_capability_boundary_freezes_manifest_edge_before_earlier_callback():
    invalid = state().model_copy(
        update={"qualification": qualification(operation="export")}
    )
    declared = manifest(state()).model_copy(update={"operations": (invalid,)})

    def replace_edge():
        native = _MODEL_DICT_DESCRIPTOR.__get__(invalid, BaseModel)
        dict.__setitem__(native, "qualification", qualification())

    versions = ObservableTuple((VERSION,), callback=replace_edge)
    native = _MODEL_DICT_DESCRIPTOR.__get__(declared, BaseModel)
    dict.__setitem__(native, "contract_schema_versions", versions)

    with pytest.raises(ValidationError, match="operation must match"):
        validate_capability_requirements(declared, (requirement(),))

    assert versions.calls == 1
    assert invalid.qualification.operation == "retrieve"


@pytest.mark.parametrize("tuple_field", ["contract_schema_versions", "operations"])
def test_capability_boundary_freezes_exact_reference_dict_before_manifest_callback(
    tuple_field,
):
    ordinary_reference = reference("runtime-profile").model_dump(mode="python")
    ordinary_reference["id"] = ""
    ordinary = requirement().model_copy(
        update={"qualification_profile_reference": ordinary_reference}
    )
    with pytest.raises(ValidationError) as caught:
        validate_capability_requirements(manifest(state()), (ordinary,))
    assert caught.value.errors(include_url=False)[0]["type"] == "string_too_short"
    assert caught.value.errors(include_url=False)[0]["loc"] == (
        "qualification_profile_reference",
        "id",
    )

    invalid_reference = reference("runtime-profile").model_dump(mode="python")
    invalid_reference["id"] = ""
    invalid = requirement().model_copy(
        update={"qualification_profile_reference": invalid_reference}
    )

    def repair_reference():
        invalid_reference["id"] = "runtime-profile"

    declared = manifest(state())
    observed = ObservableTuple(
        getattr(declared, tuple_field),
        callback=repair_reference,
    )
    native = _MODEL_DICT_DESCRIPTOR.__get__(declared, BaseModel)
    dict.__setitem__(native, tuple_field, observed)

    with pytest.raises(ValidationError) as caught:
        validate_capability_requirements(declared, (invalid,))
    assert caught.value.errors(include_url=False)[0]["type"] == "string_too_short"
    assert caught.value.errors(include_url=False)[0]["loc"] == (
        "qualification_profile_reference",
        "id",
    )
    assert observed.calls == 1
    assert invalid_reference["id"] == "runtime-profile"

    valid_reference = reference("runtime-profile").model_dump(mode="python")
    valid = requirement().model_copy(
        update={"qualification_profile_reference": valid_reference}
    )
    valid_manifest = manifest(state())
    benign = ObservableTuple(
        getattr(valid_manifest, tuple_field),
        callback=lambda: valid_reference.__setitem__("id", "runtime-profile"),
    )
    valid_native = _MODEL_DICT_DESCRIPTOR.__get__(valid_manifest, BaseModel)
    dict.__setitem__(valid_native, tuple_field, benign)

    assert validate_capability_requirements(valid_manifest, (valid,)) == ()
    assert benign.calls == 1


def test_capability_boundary_freezes_exact_reference_dict_before_requirements_callback():
    invalid_reference = reference("runtime-profile").model_dump(mode="python")
    invalid_reference["id"] = ""
    invalid = requirement().model_copy(
        update={"qualification_profile_reference": invalid_reference}
    )

    requirements = ObservableTuple(
        (invalid,),
        callback=lambda: invalid_reference.__setitem__("id", "runtime-profile"),
    )

    with pytest.raises(ValidationError) as caught:
        validate_capability_requirements(manifest(state()), requirements)
    assert caught.value.errors(include_url=False)[0]["type"] == "string_too_short"
    assert caught.value.errors(include_url=False)[0]["loc"] == (
        "qualification_profile_reference",
        "id",
    )
    assert requirements.calls == 1
    assert invalid_reference["id"] == "runtime-profile"

    valid_reference = reference("runtime-profile").model_dump(mode="python")
    valid = requirement().model_copy(
        update={"qualification_profile_reference": valid_reference}
    )
    control = ObservableTuple(
        (valid,),
        callback=lambda: valid_reference.__setitem__("id", "runtime-profile"),
    )

    assert validate_capability_requirements(manifest(state()), control) == ()
    assert control.calls == 1


@pytest.mark.parametrize("public_surface", ["requirements", "operation_state"])
def test_capability_boundary_freezes_exact_qualification_dict_before_manifest_callback(
    public_surface,
):
    ordinary_qualification = qualification(operation="export").model_dump(
        mode="python"
    )
    ordinary_state = state().model_copy(
        update={"qualification": ordinary_qualification}
    )
    ordinary_manifest = manifest(state()).model_copy(
        update={"operations": (ordinary_state,)}
    )
    with pytest.raises(ValidationError) as caught:
        if public_surface == "requirements":
            validate_capability_requirements(
                ordinary_manifest,
                (requirement(),),
            )
        else:
            ordinary_manifest.operation_state("retrieve")
    assert caught.value.errors(include_url=False)[0]["type"] == "value_error"
    assert caught.value.errors(include_url=False)[0]["loc"] == ("operations", 0)

    invalid_qualification = qualification(operation="export").model_dump(
        mode="python"
    )
    invalid_state = state().model_copy(
        update={"qualification": invalid_qualification}
    )
    declared = manifest(state()).model_copy(
        update={"operations": (invalid_state,)}
    )
    versions = ObservableTuple(
        (VERSION,),
        callback=lambda: invalid_qualification.__setitem__(
            "operation",
            "retrieve",
        ),
    )
    native = _MODEL_DICT_DESCRIPTOR.__get__(declared, BaseModel)
    dict.__setitem__(native, "contract_schema_versions", versions)

    with pytest.raises(ValidationError) as caught:
        if public_surface == "requirements":
            validate_capability_requirements(declared, (requirement(),))
        else:
            declared.operation_state("retrieve")
    assert caught.value.errors(include_url=False)[0]["type"] == "value_error"
    assert caught.value.errors(include_url=False)[0]["loc"] == ("operations", 0)
    assert versions.calls == 1
    assert invalid_qualification["operation"] == "retrieve"

    valid_qualification = qualification().model_dump(mode="python")
    valid_state = state().model_copy(update={"qualification": valid_qualification})
    valid_manifest = manifest(state()).model_copy(
        update={"operations": (valid_state,)}
    )
    benign = ObservableTuple(
        (VERSION,),
        callback=lambda: valid_qualification.__setitem__(
            "operation",
            "retrieve",
        ),
    )
    valid_native = _MODEL_DICT_DESCRIPTOR.__get__(valid_manifest, BaseModel)
    dict.__setitem__(valid_native, "contract_schema_versions", benign)

    if public_surface == "requirements":
        assert validate_capability_requirements(
            valid_manifest,
            (requirement(),),
        ) == ()
    else:
        assert valid_manifest.operation_state("retrieve") == state()
    assert benign.calls == 1


def test_capability_boundary_preserves_dict_subclass_public_view_authority():
    class DivergentReference(dict):
        def __init__(self, native, public):
            super().__init__(native)
            self.public = public
            self.calls = 0

        def items(self):
            self.calls += 1
            return self.public.items()

    valid = reference("runtime-profile").model_dump(mode="python")
    invalid = reference("runtime-profile").model_dump(mode="python")
    invalid["id"] = ""

    public_valid = DivergentReference(invalid, valid)
    received = requirement().model_copy(
        update={"qualification_profile_reference": public_valid}
    )
    assert validate_capability_requirements(manifest(state()), (received,)) == ()
    assert public_valid.calls == 1

    public_invalid = DivergentReference(valid, invalid)
    received = requirement().model_copy(
        update={"qualification_profile_reference": public_invalid}
    )
    with pytest.raises(ValidationError) as caught:
        validate_capability_requirements(manifest(state()), (received,))
    assert caught.value.errors(include_url=False)[0]["type"] == "string_too_short"
    assert caught.value.errors(include_url=False)[0]["loc"] == (
        "qualification_profile_reference",
        "id",
    )
    assert public_invalid.calls == 1


def test_capability_boundary_preserves_divergent_tuple_view_authority():
    invalid_backing = requirement().model_copy(
        update={"qualification_profile_version": None}
    )
    requirements = ObservableTuple(
        (invalid_backing,),
        view=(requirement(),),
    )

    assert validate_capability_requirements(manifest(state()), requirements) == ()
    assert requirements.calls == 1

    hidden_cross_tenant = requirement().model_copy(
        update={
            "qualification_profile_reference": reference(
                "runtime-profile"
            ).model_copy(update={"tenant_id": "tenant-2"})
        }
    )
    requirements = ObservableTuple(
        (requirement(),),
        view=(hidden_cross_tenant,),
    )

    violations = validate_capability_requirements(manifest(state()), requirements)

    assert [(item.operation, item.fact) for item in violations] == [
        ("retrieve", "qualification_profile")
    ]
    assert requirements.calls == 1


def test_manifest_preserves_divergent_operations_tuple_view_authority():
    invalid_backing = state().model_copy(update={"supported": False})
    visible = state(operation="export", qualification=qualification("export"))
    operations = ObservableTuple((invalid_backing,), view=(visible,))
    declared = manifest(state()).model_copy(update={"operations": operations})

    assert declared.operation_state("export") == visible
    assert operations.calls == 1


def test_capability_boundary_preserves_missing_extra_storage():
    valid = requirement()
    _MODEL_EXTRAS_DESCRIPTOR.__delete__(valid)

    assert validate_capability_requirements(manifest(state()), (valid,)) == ()


def _replace_stored_name(value, original_name, replacement):
    native = _MODEL_DICT_DESCRIPTOR.__get__(value, BaseModel)
    item = dict.pop(native, original_name)
    dict.__setitem__(native, replacement, item)


def _replace_fields_set_name(value, original_name, replacement):
    native = _MODEL_FIELDS_SET_DESCRIPTOR.__get__(value, BaseModel)
    members = set(set.__iter__(native))
    members.remove(original_name)
    set.add(members, replacement)
    _MODEL_FIELDS_SET_DESCRIPTOR.__set__(value, members)


@pytest.mark.parametrize("inventory", ["stored", "fields_set"])
@pytest.mark.parametrize("target_name", ["direct", "nested"])
def test_capability_boundary_captures_values_before_field_name_hooks(
    inventory,
    target_name,
):
    if target_name == "direct":
        target = requirement().model_copy(
            update={"qualification_profile_version": None}
        )
        received = target
        name = "operation"
        invalid_field = "qualification_profile_version"
        replacement = "profile-v1"
        message = "are paired"
    else:
        target = reference("runtime-profile").model_copy(update={"id": ""})
        received = requirement().model_copy(
            update={"qualification_profile_reference": target}
        )
        name = "kind"
        invalid_field = "id"
        replacement = "runtime-profile"
        message = "at least 1 character"

    hooked_name = HookedFieldName(name)
    if inventory == "stored":
        _replace_stored_name(target, name, hooked_name)
    else:
        _replace_fields_set_name(target, name, hooked_name)
    hooked_name.arm(target, invalid_field, replacement)

    with pytest.raises(ValidationError, match=message):
        validate_capability_requirements(manifest(state()), (received,))

    native = _MODEL_DICT_DESCRIPTOR.__get__(target, BaseModel)
    assert dict.__getitem__(native, invalid_field) in (None, "")
    assert hooked_name.events == []


@pytest.mark.parametrize("inventory", ["stored", "fields_set"])
@pytest.mark.parametrize("target_name", ["direct", "nested"])
def test_capability_boundary_preserves_benign_string_subclass_names(
    inventory,
    target_name,
):
    if target_name == "direct":
        target = requirement()
        received = target
        name = "operation"
    else:
        target = reference("runtime-profile")
        received = requirement().model_copy(
            update={"qualification_profile_reference": target}
        )
        name = "kind"
    hooked_name = HookedFieldName(name)
    if inventory == "stored":
        _replace_stored_name(target, name, hooked_name)
    else:
        _replace_fields_set_name(target, name, hooked_name)
    hooked_name.arm()

    assert validate_capability_requirements(manifest(state()), (received,)) == ()
    assert hooked_name.events == []


@pytest.mark.parametrize("target_name", ["direct", "nested"])
def test_capability_boundary_normalizes_extra_string_subclass_without_hooks(
    target_name,
):
    if target_name == "direct":
        target = requirement()
        received = target
    else:
        target = reference("runtime-profile")
        received = requirement().model_copy(
            update={"qualification_profile_reference": target}
        )
    hooked_name = HookedFieldName("future_semantics")
    extras = {}
    dict.__setitem__(extras, hooked_name, "deny")
    _MODEL_EXTRAS_DESCRIPTOR.__set__(target, extras)
    hooked_name.arm()

    with pytest.raises(ValueError, match="undeclared contract field.*future_semantics"):
        validate_capability_requirements(manifest(state()), (received,))

    assert hooked_name.events == []


@pytest.mark.parametrize("inventory", ["stored", "fields_set", "extras"])
def test_capability_boundary_rejects_normalized_field_name_collisions(inventory):
    invalid = requirement()
    if inventory == "extras":
        exact_name = "future_semantics"
        hooked_name = DistinctFieldName(exact_name)
        extras = {exact_name: "first"}
        dict.__setitem__(extras, hooked_name, "second")
        _MODEL_EXTRAS_DESCRIPTOR.__set__(invalid, extras)
        message = "malformed contract extras.*collide after normalization"
    else:
        exact_name = "operation"
        hooked_name = DistinctFieldName(exact_name)
        if inventory == "stored":
            native = _MODEL_DICT_DESCRIPTOR.__get__(invalid, BaseModel)
            dict.__setitem__(native, hooked_name, "retrieve")
        else:
            native = _MODEL_FIELDS_SET_DESCRIPTOR.__get__(invalid, BaseModel)
            set.add(native, hooked_name)
        message = "malformed contract model.*collide after normalization"
    hooked_name.arm()

    with pytest.raises(ValueError, match=message):
        validate_capability_requirements(manifest(state()), (invalid,))

    assert hooked_name.events == []


@pytest.mark.parametrize("inventory", ["stored", "fields_set", "extras"])
def test_capability_boundary_rejects_nonstring_field_names_without_hooks(inventory):
    invalid = requirement()
    hostile_name = NonStringFieldName()
    if inventory == "stored":
        native = _MODEL_DICT_DESCRIPTOR.__get__(invalid, BaseModel)
        dict.__setitem__(native, hostile_name, "deny")
        message = "malformed contract model.*field names must be strings"
    elif inventory == "fields_set":
        native = _MODEL_FIELDS_SET_DESCRIPTOR.__get__(invalid, BaseModel)
        set.add(native, hostile_name)
        message = "malformed contract model.*field names must be strings"
    else:
        extras = {}
        dict.__setitem__(extras, hostile_name, "deny")
        _MODEL_EXTRAS_DESCRIPTOR.__set__(invalid, extras)
        message = "malformed contract extras.*field names must be strings"
    hostile_name.arm()

    with pytest.raises(ValueError, match=message):
        validate_capability_requirements(manifest(state()), (invalid,))

    assert hostile_name.events == []


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
    original_snapshot = contracts_engines._validated_contract_snapshot_from_frozen
    original_lookup = contracts_engines._operation_state_lookup

    def counted_snapshot(
        value,
        expected_type,
        *,
        label,
        frozen_models,
        frozen_containers,
        discovered,
    ):
        if expected_type is CapabilityManifest:
            counts["manifest_snapshots"] += 1
        return original_snapshot(
            value,
            expected_type,
            label=label,
            frozen_models=frozen_models,
            frozen_containers=frozen_containers,
            discovered=discovered,
        )

    def counted_lookup(value):
        counts["operation_indexes"] += 1
        return original_lookup(value)

    monkeypatch.setattr(
        contracts_engines,
        "_validated_contract_snapshot_from_frozen",
        counted_snapshot,
    )
    monkeypatch.setattr(contracts_engines, "_operation_state_lookup", counted_lookup)

    violations = validate_capability_requirements(declared, requirements)

    assert [(item.operation, item.fact) for item in violations] == [
        ("delete", "supported")
    ]
    assert counts == {"manifest_snapshots": 1, "operation_indexes": 1}
