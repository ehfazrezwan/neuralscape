"""Contract tests for topology-neutral tenant operation state."""

from __future__ import annotations

import json
from collections.abc import Mapping
from copy import deepcopy
from types import MappingProxyType

import pytest
from pydantic import BaseModel, ValidationError

from contracts_tenancy import (
    ResourceManifestReference,
    TenantOperationState,
    TenantPlacement,
    validate_operation_transition,
    validate_placement_publication,
)


VERSION = "candidate-v1"


class InconsistentExtraMapping(Mapping[object, object]):
    """Expose entries through items while hiding them from normal iteration."""

    def __init__(self, entries: dict[object, object]) -> None:
        self._entries = tuple(entries.items())

    def __getitem__(self, key: object) -> object:
        return dict(self._entries)[key]

    def __iter__(self):
        return iter(())

    def __len__(self) -> int:
        return 0

    def items(self):
        return self._entries


class ChangingItemsMapping(Mapping[object, object]):
    """Return one entry snapshot first and a different one thereafter."""

    def __init__(
        self,
        first: tuple[tuple[object, object], ...],
        later: tuple[tuple[object, object], ...],
    ) -> None:
        self._first = first
        self._later = later
        self.calls = 0

    def __getitem__(self, key: object) -> object:
        raise KeyError(key)

    def __iter__(self):
        return iter(())

    def __len__(self) -> int:
        return 0

    def items(self):
        self.calls += 1
        return self._first if self.calls == 1 else self._later


class FalseyEntries(dict[object, object]):
    def __bool__(self) -> bool:
        return False


class HiddenNativeBacking(dict[object, object]):
    """Hide native dict storage from every overridable mapping view."""

    def __init__(self, entries: dict[object, object]) -> None:
        dict.__init__(self, entries)
        self.length_calls = 0
        self.iteration_calls = 0
        self.keys_calls = 0
        self.items_calls = 0
        self.boolean_calls = 0

    def __len__(self) -> int:
        self.length_calls += 1
        return 0

    def __iter__(self):
        self.iteration_calls += 1
        return iter(())

    def keys(self):
        self.keys_calls += 1
        return ()

    def items(self):
        self.items_calls += 1
        return ()

    def __bool__(self) -> bool:
        self.boolean_calls += 1
        return False


class ClassTrapNativeBacking(HiddenNativeBacking):
    @property
    def __class__(self):
        raise AssertionError("native dict dispatch must not inspect __class__")


class DictSpoofMapping(Mapping[object, object]):
    """A non-dict mapping whose dynamic class view claims to be dict."""

    def __init__(self, entries: dict[object, object]) -> None:
        self._entries = tuple(entries.items())
        self.class_calls = 0
        self.length_calls = 0
        self.iteration_calls = 0
        self.items_calls = 0
        self.boolean_calls = 0

    @property
    def __class__(self):
        self.class_calls += 1
        return dict

    def __getitem__(self, key: object) -> object:
        return dict(self._entries)[key]

    def __iter__(self):
        self.iteration_calls += 1
        return (name for name, _field_value in self._entries)

    def __len__(self) -> int:
        self.length_calls += 1
        return len(self._entries)

    def items(self):
        self.items_calls += 1
        return self._entries

    def __bool__(self) -> bool:
        self.boolean_calls += 1
        return False


class HiddenModelStorage(dict[str, object]):
    """Retain native model state while exposing a sanitized public view."""

    def __init__(
        self,
        actual: dict[str, object],
        visible: dict[str, object],
    ) -> None:
        dict.__init__(self, actual)
        self._visible = tuple(visible.items())
        self.keys_calls = 0
        self.items_calls = 0

    def keys(self):
        self.keys_calls += 1
        return dict(self._visible).keys()

    def items(self):
        self.items_calls += 1
        return self._visible


def replace_model_storage(
    model: object,
    *,
    stored_name: str,
    stored_value: object,
) -> HiddenModelStorage:
    visible = dict(
        dict.items(object.__getattribute__(model, "__dict__"))
    )
    actual = dict(visible)
    actual[stored_name] = stored_value
    backing = HiddenModelStorage(actual, visible)
    object.__setattr__(model, "__dict__", backing)
    return backing


PYDANTIC_DICT_DESCRIPTOR = BaseModel.__dict__["__dict__"]
PYDANTIC_EXTRA_DESCRIPTOR = BaseModel.__dict__["__pydantic_extra__"]
PYDANTIC_FIELDS_SET_DESCRIPTOR = BaseModel.__dict__["__pydantic_fields_set__"]


class RepairingName(str):
    def __new__(cls, value: str, repair):
        instance = super().__new__(cls, value)
        instance.repair = repair
        instance.armed = False
        instance.calls = 0
        return instance

    def __hash__(self) -> int:
        if self.armed:
            self.calls += 1
            self.repair()
        return str.__hash__(self)


class RepairingTuple(tuple):
    def __new__(cls, values: tuple[object, ...], repair):
        instance = super().__new__(cls, values)
        instance.repair = repair
        instance.calls = 0
        return instance

    def __iter__(self):
        self.calls += 1
        self.repair()
        return tuple.__iter__(self)


class RepairingList(list):
    def __init__(self, values: tuple[object, ...], repair) -> None:
        super().__init__(values)
        self.repair = repair
        self.calls = 0

    def __iter__(self):
        self.calls += 1
        self.repair()
        return list.__iter__(self)


class DivergentTuple(tuple):
    def __new__(
        cls,
        native_values: tuple[object, ...],
        visible_values: tuple[object, ...],
    ):
        instance = super().__new__(cls, native_values)
        instance.visible_values = visible_values
        instance.calls = 0
        return instance

    def __iter__(self):
        self.calls += 1
        return iter(self.visible_values)


class RepairingEmptyExtraMapping(Mapping[object, object]):
    def __init__(self, repair) -> None:
        self.repair = repair
        self.calls = 0

    def _run(self) -> None:
        self.calls += 1
        self.repair()

    def __getitem__(self, key: object) -> object:
        raise KeyError(key)

    def __iter__(self):
        self._run()
        return iter(())

    def __len__(self) -> int:
        self._run()
        return 0

    def items(self):
        self._run()
        return ()


def model_with_hidden_extra(
    model: BaseModel,
    *,
    reported_view: str,
) -> tuple[BaseModel, list[str]]:
    calls: list[str] = []
    target = model
    if reported_view != "ordinary":
        reported = None if reported_view == "none" else {}
        model_type = type(model)

        def masked_getattribute(self, name: str):
            if name == "__pydantic_extra__":
                calls.append(name)
                return reported
            return super(masked_type, self).__getattribute__(name)

        masked_type = type(
            f"Masked{model_type.__name__}",
            (model_type,),
            {"__getattribute__": masked_getattribute},
        )
        target = masked_type.model_validate(
            dict(dict.items(object.__getattribute__(model, "__dict__")))
        )
    PYDANTIC_EXTRA_DESCRIPTOR.__set__(
        target,
        {"hidden_unknown": "deny"},
    )
    return target, calls


def model_with_descriptor_storage(
    model: BaseModel,
    *,
    representation: str,
    include_unknown: bool,
) -> tuple[BaseModel, list[str]]:
    calls: list[str] = []
    target = model
    if representation == "property":
        model_type = type(model)

        def stored_getter(self):
            calls.append("__dict__")
            native = PYDANTIC_DICT_DESCRIPTOR.__get__(self, BaseModel)
            return {
                name: field_value
                for name, field_value in dict.items(native)
                if name != "hidden_unknown"
            }

        def stored_setter(self, value):
            PYDANTIC_DICT_DESCRIPTOR.__set__(self, value)

        masked_type = type(
            f"Descriptor{model_type.__name__}",
            (model_type,),
            {"__dict__": property(stored_getter, stored_setter)},
        )
        target = masked_type.model_validate(
            dict(
                dict.items(
                    PYDANTIC_DICT_DESCRIPTOR.__get__(model, BaseModel)
                )
            )
        )
    native = PYDANTIC_DICT_DESCRIPTOR.__get__(target, BaseModel)
    if include_unknown:
        dict.__setitem__(native, "hidden_unknown", "deny")
    calls.clear()
    return target, calls


class InverseItemsMapping(dict[object, object]):
    """Expose iterated keys while hiding every entry from items()."""

    def __init__(self, entries: dict[object, object]) -> None:
        super().__init__(entries)
        self.iteration_calls = 0
        self.items_calls = 0

    def __iter__(self):
        self.iteration_calls += 1
        return super().__iter__()

    def items(self):
        self.items_calls += 1
        return ()


class ObservedLengthMapping(Mapping[object, object]):
    """Expose only a configured length, with counters for every observed view."""

    def __init__(self, observed_length: int) -> None:
        self.observed_length = observed_length
        self.length_calls = 0
        self.iteration_calls = 0
        self.items_calls = 0
        self.boolean_calls = 0

    def __getitem__(self, key: object) -> object:
        raise KeyError(key)

    def __iter__(self):
        self.iteration_calls += 1
        return iter(())

    def __len__(self) -> int:
        self.length_calls += 1
        return self.observed_length

    def items(self):
        self.items_calls += 1
        return ()


class FalseBooleanLengthMapping(ObservedLengthMapping):
    def __bool__(self) -> bool:
        self.boolean_calls += 1
        return False


class TrueBooleanEmptyMapping(ObservedLengthMapping):
    def __bool__(self) -> bool:
        self.boolean_calls += 1
        return True


class DefaultItemsLengthProbe(Mapping[object, object]):
    """Instrument the standard Mapping.items() view and its source mapping."""

    def __init__(self) -> None:
        self.length_calls = 0
        self.iteration_calls = 0
        self.items_calls = 0

    def __getitem__(self, key: object) -> object:
        raise KeyError(key)

    def __iter__(self):
        self.iteration_calls += 1
        return iter(())

    def __len__(self) -> int:
        self.length_calls += 1
        return 0

    def items(self):
        self.items_calls += 1
        return super().items()


def manifest(
    *, tenant_id: str = "tenant-a", generation: int = 7, manifest_id: str = "primary"
) -> dict[str, object]:
    return {
        "schema_version": VERSION,
        "tenant_id": tenant_id,
        "manifest_id": manifest_id,
        "manifest_revision": 2,
        "placement_generation": generation,
    }


def operation(**updates: object) -> TenantOperationState:
    document: dict[str, object] = {
        "schema_version": VERSION,
        "tenant_id": "tenant-a",
        "operation_id": "operation-1",
        "operation": "provision",
        "desired_state": "active",
        "observed_state": "pending",
        "placement_generation": 7,
        "resource_manifests": [manifest()],
    }
    document.update(updates)
    return TenantOperationState.model_validate_json(json.dumps(document))


def test_placement_preserves_explicit_tenant_isolation_identity() -> None:
    placement = TenantPlacement.model_validate_json(
        json.dumps(
            {
                "schema_version": VERSION,
                "tenant_id": "tenant-a",
                "generation": 7,
                "resource_manifests": [manifest()],
            }
        )
    )

    assert placement.tenant_id == "tenant-a"
    assert placement.resource_manifests[0].tenant_id == "tenant-a"


def test_placement_rejects_cross_tenant_manifest_reference() -> None:
    with pytest.raises(ValidationError, match="tenant_id must match"):
        TenantPlacement.model_validate_json(
            json.dumps(
                {
                    "schema_version": VERSION,
                    "tenant_id": "tenant-a",
                    "generation": 7,
                    "resource_manifests": [manifest(tenant_id="tenant-b")],
                }
            )
        )


def test_placement_rejects_manifest_from_mismatched_generation_epoch() -> None:
    with pytest.raises(ValidationError, match="placement_generation must match"):
        TenantPlacement.model_validate_json(
            json.dumps(
                {
                    "schema_version": VERSION,
                    "tenant_id": "tenant-a",
                    "generation": 8,
                    "resource_manifests": [manifest(generation=7)],
                }
            )
        )


def test_placement_rejects_duplicate_manifest_identity_and_revision() -> None:
    with pytest.raises(ValidationError, match="must be unique"):
        TenantPlacement.model_validate_json(
            json.dumps(
                {
                    "schema_version": VERSION,
                    "tenant_id": "tenant-a",
                    "generation": 7,
                    "resource_manifests": [manifest(), deepcopy(manifest())],
                }
            )
        )


@pytest.mark.parametrize(
    ("kind", "desired"),
    [
        ("provision", "active"),
        ("suspend", "suspended"),
        ("resume", "active"),
        ("export", "unchanged"),
        ("restore", "active"),
        ("upgrade", "active"),
        ("delete", "deleted"),
    ],
)
def test_each_operation_has_an_explicit_desired_result(kind: str, desired: str) -> None:
    state = operation(
        operation=kind,
        desired_state=desired,
        resource_manifests=[],
    )

    assert state.desired_state == desired


def test_operation_rejects_semantically_wrong_desired_result() -> None:
    with pytest.raises(ValidationError, match="suspend requires desired_state=suspended"):
        operation(operation="suspend", desired_state="active")


def test_operation_rejects_manifest_from_another_tenant() -> None:
    with pytest.raises(ValidationError, match="tenant_id must match"):
        operation(resource_manifests=[manifest(tenant_id="tenant-b")])


def test_operation_rejects_exact_duplicate_manifest_references() -> None:
    with pytest.raises(ValidationError, match="must be unique"):
        operation(resource_manifests=[manifest(), deepcopy(manifest())])


def test_valid_operation_progress_is_accepted() -> None:
    pending = operation(observed_state="pending")
    running = operation(observed_state="running")
    succeeded = operation(observed_state="succeeded")

    running_result = validate_operation_transition(pending, running)
    succeeded_result = validate_operation_transition(running, succeeded)

    assert running_result == running
    assert running_result is not running
    assert succeeded_result == succeeded
    assert succeeded_result is not succeeded


@pytest.mark.parametrize(
    ("callback_kind", "expected_calls"),
    [("stored-name", 2), ("tuple", 1)],
)
def test_transition_freezes_current_before_previous_callbacks(
    callback_kind: str,
    expected_calls: int,
) -> None:
    def invalid_current() -> tuple[TenantOperationState, ResourceManifestReference]:
        current = operation(observed_state="running")
        child = current.resource_manifests[0]
        _set_native_field(child, "manifest_id", "")
        return current, child

    def install_callback(
        previous: TenantOperationState,
        repair,
    ) -> RepairingName | RepairingTuple:
        if callback_kind == "stored-name":
            return _install_repairing_stored_name(previous, repair)
        repairing = RepairingTuple(previous.resource_manifests, repair)
        _set_native_field(previous, "resource_manifests", repairing)
        return repairing

    ordinary_current, _ordinary_child = invalid_current()
    with pytest.raises(ValidationError) as ordinary_error:
        validate_operation_transition(operation(), ordinary_current)
    assert ordinary_error.value.errors()[0]["type"] == "string_too_short"
    assert ordinary_error.value.errors()[0]["loc"] == (
        "resource_manifests",
        0,
        "manifest_id",
    )

    attacked_current, attacked_child = invalid_current()
    previous = operation()
    callback = install_callback(
        previous,
        lambda: _set_native_field(attacked_child, "manifest_id", "primary"),
    )

    with pytest.raises(ValidationError) as callback_error:
        validate_operation_transition(previous, attacked_current)

    assert callback_error.value.errors() == ordinary_error.value.errors()
    assert callback.calls == expected_calls
    assert attacked_child.manifest_id == "primary"

    valid_current = operation(observed_state="running")
    valid_child = valid_current.resource_manifests[0]
    valid_previous = operation()
    valid_callback = install_callback(
        valid_previous,
        lambda: _set_native_field(valid_child, "manifest_id", "primary"),
    )

    result = validate_operation_transition(valid_previous, valid_current)

    assert result == valid_current
    assert valid_callback.calls == expected_calls
    assert valid_child.manifest_id == "primary"


@pytest.mark.parametrize(
    ("previous", "current"),
    [
        ("pending", "succeeded"),
        ("running", "pending"),
        ("succeeded", "running"),
        ("failed", "running"),
        ("cancelled", "running"),
    ],
)
def test_invalid_operation_progress_is_rejected(previous: str, current: str) -> None:
    with pytest.raises(ValueError, match="invalid observed_state transition"):
        validate_operation_transition(
            operation(observed_state=previous),
            operation(observed_state=current),
        )


@pytest.mark.parametrize(
    ("field", "updates"),
    [
        ("tenant_id", {"tenant_id": "tenant-b"}),
        ("operation_id", {"operation_id": "operation-2"}),
        ("operation", {"operation": "resume", "desired_state": "active"}),
    ],
)
def test_transition_rejects_changed_operation_identity(
    field: str, updates: dict[str, str]
) -> None:
    previous = operation(resource_manifests=[])
    current = operation(
        observed_state="running", resource_manifests=[], **updates
    )
    with pytest.raises(ValueError, match=f"immutable {field}"):
        validate_operation_transition(previous, current)


def test_transition_rejects_generation_rollback() -> None:
    with pytest.raises(ValueError, match="cannot decrease"):
        validate_operation_transition(
            operation(placement_generation=8, resource_manifests=[]),
            operation(
                placement_generation=7,
                observed_state="running",
                resource_manifests=[],
            ),
        )


@pytest.mark.parametrize("terminal_state", ["succeeded", "failed", "cancelled"])
def test_terminal_republication_must_be_exactly_idempotent(
    terminal_state: str,
) -> None:
    terminal = operation(observed_state=terminal_state)

    assert validate_operation_transition(terminal, terminal.model_copy()) == terminal

    with pytest.raises(ValueError, match="must be identical"):
        validate_operation_transition(
            terminal,
            operation(
                observed_state=terminal_state,
                placement_generation=8,
                resource_manifests=[manifest(generation=8)],
            ),
        )
    with pytest.raises(ValueError, match="must be identical"):
        validate_operation_transition(
            terminal,
            operation(
                observed_state=terminal_state,
                resource_manifests=[manifest(manifest_id="replacement")],
            ),
        )


def test_publication_requires_exact_current_generation() -> None:
    placement = TenantPlacement.model_validate_json(
        json.dumps(
            {
                "schema_version": VERSION,
                "tenant_id": "tenant-a",
                "generation": 7,
                "resource_manifests": [manifest()],
            }
        )
    )

    result = validate_placement_publication(
        placement, expected_tenant_id="tenant-a", current_generation=7
    )
    assert result == placement
    assert result is not placement
    with pytest.raises(ValueError, match="stale or unexpected"):
        validate_placement_publication(
            placement, expected_tenant_id="tenant-a", current_generation=8
        )


def test_publication_rejects_cross_tenant_context() -> None:
    placement = TenantPlacement.model_validate_json(
        json.dumps(
            {
                "schema_version": VERSION,
                "tenant_id": "tenant-a",
                "generation": 7,
                "resource_manifests": [],
            }
        )
    )

    with pytest.raises(ValueError, match="tenant_id does not match"):
        validate_placement_publication(
            placement, expected_tenant_id="tenant-b", current_generation=7
        )


def test_publication_revalidates_copied_nested_manifest_graph() -> None:
    placement = TenantPlacement.model_validate_json(
        json.dumps(
            {
                "schema_version": VERSION,
                "tenant_id": "tenant-a",
                "generation": 7,
                "resource_manifests": [manifest()],
            }
        )
    )
    cross_tenant = ResourceManifestReference.model_validate_json(
        json.dumps(manifest(tenant_id="tenant-b"))
    )
    corrupted = placement.model_copy(
        update={"resource_manifests": (cross_tenant,)}
    )

    with pytest.raises(ValidationError, match="tenant_id must match"):
        validate_placement_publication(
            corrupted,
            expected_tenant_id="tenant-a",
            current_generation=7,
        )


def test_publication_rejects_retained_copied_extra_fields() -> None:
    placement = TenantPlacement.model_validate_json(
        json.dumps(
            {
                "schema_version": VERSION,
                "tenant_id": "tenant-a",
                "generation": 7,
                "resource_manifests": [manifest()],
            }
        )
    )
    top_level_extra = placement.model_copy(
        update={"unreviewed_topology": "shared-plane"}
    )
    nested_extra = placement.resource_manifests[0].model_copy(
        update={"authority": "admin"}
    )
    nested_corruption = placement.model_copy(
        update={"resource_manifests": (nested_extra,)}
    )

    for corrupted in (top_level_extra, nested_corruption):
        with pytest.raises(ValueError, match="undeclared fields"):
            validate_placement_publication(
                corrupted,
                expected_tenant_id="tenant-a",
                current_generation=7,
            )


@pytest.mark.parametrize("corrupt_previous", [False, True])
def test_transition_revalidates_copied_operation_graph(
    corrupt_previous: bool,
) -> None:
    previous = operation(resource_manifests=[])
    current = operation(observed_state="running", resource_manifests=[])
    cross_tenant = ResourceManifestReference.model_validate_json(
        json.dumps(manifest(tenant_id="tenant-b"))
    )
    corrupted = (previous if corrupt_previous else current).model_copy(
        update={"resource_manifests": (cross_tenant,)}
    )

    with pytest.raises(ValidationError, match="tenant_id must match"):
        validate_operation_transition(
            corrupted if corrupt_previous else previous,
            current if corrupt_previous else corrupted,
        )


@pytest.mark.parametrize("corrupt_previous", [False, True])
def test_transition_rejects_retained_copied_top_level_extra(
    corrupt_previous: bool,
) -> None:
    previous = operation(resource_manifests=[])
    current = operation(observed_state="running", resource_manifests=[])
    corrupted = (previous if corrupt_previous else current).model_copy(
        update={"unreviewed_policy": "allow"}
    )

    with pytest.raises(ValueError, match="undeclared fields"):
        validate_operation_transition(
            corrupted if corrupt_previous else previous,
            current if corrupt_previous else corrupted,
        )


def test_transition_rejects_retained_copied_nested_extra() -> None:
    previous = operation()
    current = operation(observed_state="running")
    nested_extra = current.resource_manifests[0].model_copy(
        update={"authority": "admin"}
    )
    corrupted = current.model_copy(
        update={"resource_manifests": (nested_extra,)}
    )

    with pytest.raises(ValueError, match="undeclared fields"):
        validate_operation_transition(previous, corrupted)


@pytest.mark.parametrize(
    "malformed_extra",
    [[], (), "", 0, False, ["unreviewed"]],
    ids=["list", "tuple", "string", "zero", "false", "nonempty-list"],
)
def test_transition_rejects_each_malformed_extra_storage_representation(
    malformed_extra: object,
) -> None:
    previous = operation(resource_manifests=[])
    current = operation(observed_state="running", resource_manifests=[])
    object.__setattr__(current, "__pydantic_extra__", malformed_extra)

    with pytest.raises(ValueError, match="extra storage must be a mapping"):
        validate_operation_transition(previous, current)


@pytest.mark.parametrize(
    "extra_storage",
    [None, {}, MappingProxyType({})],
    ids=["none", "empty-dict", "empty-custom-mapping"],
)
def test_transition_accepts_absent_or_empty_mapping_extra_storage(
    extra_storage: object,
) -> None:
    previous = operation(resource_manifests=[])
    current = operation(observed_state="running", resource_manifests=[])
    object.__setattr__(current, "__pydantic_extra__", extra_storage)

    result = validate_operation_transition(previous, current)

    assert result.observed_state == "running"


@pytest.mark.parametrize(
    "stored_extra",
    [{"unreviewed_policy": "allow"}, {"tenant_id": "tenant-a"}],
    ids=["unknown", "declared-overlap"],
)
def test_transition_rejects_nonempty_mapping_extra_storage(
    stored_extra: dict[str, object],
) -> None:
    previous = operation(resource_manifests=[])
    current = operation(observed_state="running", resource_manifests=[])
    object.__setattr__(current, "__pydantic_extra__", stored_extra)

    with pytest.raises(ValueError, match="undeclared fields"):
        validate_operation_transition(previous, current)


@pytest.mark.parametrize("location", ["direct", "nested"])
@pytest.mark.parametrize(
    "extra_kind",
    ["unknown", "declared"],
    ids=["unknown", "declared"],
)
def test_transition_rejects_entries_hidden_from_mapping_iteration(
    location: str,
    extra_kind: str,
) -> None:
    previous = operation()
    current = operation(observed_state="running")
    target = current if location == "direct" else current.resource_manifests[0]
    stored_extra = (
        {"future_constraint": "reject"}
        if extra_kind == "unknown"
        else {"tenant_id": "tenant-a"}
    )
    object.__setattr__(
        target,
        "__pydantic_extra__",
        InconsistentExtraMapping(stored_extra),
    )

    with pytest.raises(ValueError, match="undeclared fields"):
        validate_operation_transition(previous, current)


@pytest.mark.parametrize(
    "stored_extra",
    [{"future_constraint": "reject"}, {"tenant_id": "tenant-a"}],
    ids=["unknown", "declared"],
)
def test_transition_rejects_populated_falsey_extra_storage(
    stored_extra: dict[object, object],
) -> None:
    previous = operation(resource_manifests=[])
    current = operation(observed_state="running", resource_manifests=[])
    object.__setattr__(current, "__pydantic_extra__", FalseyEntries(stored_extra))

    with pytest.raises(ValueError, match="undeclared fields"):
        validate_operation_transition(previous, current)


def test_transition_consumes_one_stable_extra_entry_snapshot() -> None:
    previous = operation(resource_manifests=[])
    current = operation(observed_state="running", resource_manifests=[])
    populated_first = ChangingItemsMapping(
        (("future_constraint", "reject"),),
        (),
    )
    object.__setattr__(current, "__pydantic_extra__", populated_first)

    with pytest.raises(ValueError, match="undeclared fields"):
        validate_operation_transition(previous, current)
    assert populated_first.calls == 1

    current = operation(observed_state="running", resource_manifests=[])
    empty_first = ChangingItemsMapping(
        (),
        (("future_constraint", "reject"),),
    )
    object.__setattr__(current, "__pydantic_extra__", empty_first)

    result = validate_operation_transition(previous, current)

    assert result.observed_state == "running"
    assert empty_first.calls == 1


@pytest.mark.parametrize("boundary", ["transition", "placement"])
@pytest.mark.parametrize("location", ["direct", "nested"])
def test_tenancy_boundaries_reject_iteration_visible_items_empty_extras(
    boundary: str,
    location: str,
) -> None:
    extras = InverseItemsMapping({"future_constraint": "reject"})
    if boundary == "transition":
        previous = operation()
        candidate = operation(observed_state="running")
        target = (
            candidate
            if location == "direct"
            else candidate.resource_manifests[0]
        )
        object.__setattr__(target, "__pydantic_extra__", extras)

        with pytest.raises(ValueError, match="undeclared fields"):
            validate_operation_transition(previous, candidate)
    else:
        candidate = TenantPlacement.model_validate_json(
            json.dumps(
                {
                    "schema_version": VERSION,
                    "tenant_id": "tenant-a",
                    "generation": 7,
                    "resource_manifests": [manifest()],
                }
            )
        )
        target = (
            candidate
            if location == "direct"
            else candidate.resource_manifests[0]
        )
        object.__setattr__(target, "__pydantic_extra__", extras)

        with pytest.raises(ValueError, match="undeclared fields"):
            validate_placement_publication(
                candidate,
                expected_tenant_id="tenant-a",
                current_generation=7,
            )

    assert extras.iteration_calls == 0
    assert extras.items_calls == 0


@pytest.mark.parametrize("boundary", ["transition", "placement"])
@pytest.mark.parametrize("location", ["direct", "nested"])
@pytest.mark.parametrize(
    "extra_kind",
    ["unknown", "declared"],
    ids=["unknown", "declared-overlap"],
)
def test_tenancy_boundaries_reject_hidden_native_dict_backing(
    boundary: str,
    location: str,
    extra_kind: str,
) -> None:
    if boundary == "transition":
        previous = operation()
        candidate = operation(observed_state="running")
        target = (
            candidate
            if location == "direct"
            else candidate.resource_manifests[0]
        )
        declared_name = "tenant_id"

        def validate() -> object:
            return validate_operation_transition(previous, candidate)

    else:
        candidate = TenantPlacement.model_validate_json(
            json.dumps(
                {
                    "schema_version": VERSION,
                    "tenant_id": "tenant-a",
                    "generation": 7,
                    "resource_manifests": [manifest()],
                }
            )
        )
        target = (
            candidate
            if location == "direct"
            else candidate.resource_manifests[0]
        )
        declared_name = "tenant_id"

        def validate() -> object:
            return validate_placement_publication(
                candidate,
                expected_tenant_id="tenant-a",
                current_generation=7,
            )

    name = "future_constraint" if extra_kind == "unknown" else declared_name
    extras = HiddenNativeBacking({name: "reject"})
    object.__setattr__(target, "__pydantic_extra__", extras)

    with pytest.raises(ValueError, match="undeclared fields"):
        validate()

    assert tuple(dict.items(extras)) == ((name, "reject"),)
    assert (
        extras.length_calls,
        extras.iteration_calls,
        extras.keys_calls,
        extras.items_calls,
        extras.boolean_calls,
    ) == (0, 0, 0, 0, 0)


def test_transition_accepts_empty_native_dict_subclass_backing() -> None:
    previous = operation(resource_manifests=[])
    current = operation(observed_state="running", resource_manifests=[])
    extras = HiddenNativeBacking({})
    object.__setattr__(current, "__pydantic_extra__", extras)

    result = validate_operation_transition(previous, current)

    assert result.observed_state == "running"
    assert (
        extras.length_calls,
        extras.iteration_calls,
        extras.keys_calls,
        extras.items_calls,
        extras.boolean_calls,
    ) == (0, 0, 0, 0, 0)


@pytest.mark.parametrize("boundary", ["transition", "placement"])
@pytest.mark.parametrize("location", ["direct", "nested"])
@pytest.mark.parametrize(
    ("extra_kind", "should_reject"),
    [("empty", False), ("unknown", True), ("declared", True)],
)
def test_tenancy_boundaries_dispatch_native_dict_before_mapping_abc(
    boundary: str,
    location: str,
    extra_kind: str,
    should_reject: bool,
) -> None:
    if boundary == "transition":
        previous = operation()
        candidate = operation(observed_state="running")
        target = (
            candidate
            if location == "direct"
            else candidate.resource_manifests[0]
        )

        def validate() -> object:
            return validate_operation_transition(previous, candidate)

    else:
        candidate = TenantPlacement.model_validate_json(
            json.dumps(
                {
                    "schema_version": VERSION,
                    "tenant_id": "tenant-a",
                    "generation": 7,
                    "resource_manifests": [manifest()],
                }
            )
        )
        target = (
            candidate
            if location == "direct"
            else candidate.resource_manifests[0]
        )

        def validate() -> object:
            return validate_placement_publication(
                candidate,
                expected_tenant_id="tenant-a",
                current_generation=7,
            )

    stored = {
        "unknown": {"future_constraint": "reject"},
        "declared": {"tenant_id": "tenant-a"},
    }.get(extra_kind, {})
    extras = ClassTrapNativeBacking(stored)
    object.__setattr__(target, "__pydantic_extra__", extras)

    if should_reject:
        with pytest.raises(ValueError, match="undeclared fields"):
            validate()
    else:
        validate()

    assert (
        extras.length_calls,
        extras.iteration_calls,
        extras.keys_calls,
        extras.items_calls,
        extras.boolean_calls,
    ) == (0, 0, 0, 0, 0)


@pytest.mark.parametrize("boundary", ["transition", "placement"])
@pytest.mark.parametrize("location", ["direct", "nested"])
@pytest.mark.parametrize(
    ("extra_kind", "should_reject"),
    [("empty", False), ("unknown", True), ("declared", True)],
)
def test_tenancy_boundaries_keep_spoofing_non_dict_on_mapping_path(
    boundary: str,
    location: str,
    extra_kind: str,
    should_reject: bool,
) -> None:
    if boundary == "transition":
        previous = operation()
        candidate = operation(observed_state="running")
        target = (
            candidate
            if location == "direct"
            else candidate.resource_manifests[0]
        )

        def validate() -> object:
            return validate_operation_transition(previous, candidate)

    else:
        candidate = TenantPlacement.model_validate_json(
            json.dumps(
                {
                    "schema_version": VERSION,
                    "tenant_id": "tenant-a",
                    "generation": 7,
                    "resource_manifests": [manifest()],
                }
            )
        )
        target = (
            candidate
            if location == "direct"
            else candidate.resource_manifests[0]
        )

        def validate() -> object:
            return validate_placement_publication(
                candidate,
                expected_tenant_id="tenant-a",
                current_generation=7,
            )

    stored = {
        "unknown": {"future_constraint": "reject"},
        "declared": {"tenant_id": "tenant-a"},
    }.get(extra_kind, {})
    extras = DictSpoofMapping(stored)
    object.__setattr__(target, "__pydantic_extra__", extras)

    if should_reject:
        with pytest.raises(ValueError, match="undeclared fields"):
            validate()
    else:
        validate()

    assert extras.class_calls == 1
    assert extras.length_calls == 1
    assert extras.iteration_calls == 1
    assert extras.items_calls == 1
    assert extras.boolean_calls == 0


@pytest.mark.parametrize("boundary", ["transition", "placement"])
@pytest.mark.parametrize("location", ["direct", "nested"])
@pytest.mark.parametrize(
    "corruption",
    ["hidden-unknown", "invalid-declared"],
)
def test_tenancy_boundaries_use_frozen_native_model_storage(
    boundary: str,
    location: str,
    corruption: str,
) -> None:
    if boundary == "transition":
        previous = operation()
        candidate = operation(observed_state="running")
        target = (
            candidate
            if location == "direct"
            else candidate.resource_manifests[0]
        )

        def validate() -> object:
            return validate_operation_transition(previous, candidate)

    else:
        candidate = TenantPlacement.model_validate_json(
            json.dumps(
                {
                    "schema_version": VERSION,
                    "tenant_id": "tenant-a",
                    "generation": 7,
                    "resource_manifests": [manifest()],
                }
            )
        )
        target = (
            candidate
            if location == "direct"
            else candidate.resource_manifests[0]
        )

        def validate() -> object:
            return validate_placement_publication(
                candidate,
                expected_tenant_id="tenant-a",
                current_generation=7,
            )

    if corruption == "hidden-unknown":
        stored_name = "hidden_unknown"
        stored_value: object = "deny"
    elif location == "direct":
        stored_name = "tenant_id"
        stored_value = ""
    else:
        stored_name = "manifest_revision"
        stored_value = -1
    backing = replace_model_storage(
        target,
        stored_name=stored_name,
        stored_value=stored_value,
    )

    if corruption == "hidden-unknown":
        with pytest.raises(ValueError, match="undeclared fields"):
            validate()
    else:
        with pytest.raises(ValidationError):
            validate()

    assert dict.__getitem__(backing, stored_name) == stored_value
    assert backing.keys_calls == 0
    assert backing.items_calls == 0


@pytest.mark.parametrize("boundary", ["transition", "placement"])
@pytest.mark.parametrize("location", ["direct", "nested"])
@pytest.mark.parametrize("reported_view", ["ordinary", "none", "empty"])
def test_tenancy_boundaries_read_model_owned_extra_storage(
    boundary: str,
    location: str,
    reported_view: str,
) -> None:
    if boundary == "transition":
        previous = operation()
        candidate = operation(observed_state="running")
        if location == "direct":
            target, calls = model_with_hidden_extra(
                candidate,
                reported_view=reported_view,
            )
            candidate = target
        else:
            target, calls = model_with_hidden_extra(
                candidate.resource_manifests[0],
                reported_view=reported_view,
            )
            candidate = candidate.model_copy(
                update={"resource_manifests": (target,)}
            )

        def validate() -> object:
            return validate_operation_transition(previous, candidate)

    else:
        candidate = TenantPlacement.model_validate_json(
            json.dumps(
                {
                    "schema_version": VERSION,
                    "tenant_id": "tenant-a",
                    "generation": 7,
                    "resource_manifests": [manifest()],
                }
            )
        )
        if location == "direct":
            target, calls = model_with_hidden_extra(
                candidate,
                reported_view=reported_view,
            )
            candidate = target
        else:
            target, calls = model_with_hidden_extra(
                candidate.resource_manifests[0],
                reported_view=reported_view,
            )
            candidate = candidate.model_copy(
                update={"resource_manifests": (target,)}
            )

        def validate() -> object:
            return validate_placement_publication(
                candidate,
                expected_tenant_id="tenant-a",
                current_generation=7,
            )

    actual_extras = PYDANTIC_EXTRA_DESCRIPTOR.__get__(target, type(target))
    assert dict.__getitem__(actual_extras, "hidden_unknown") == "deny"
    with pytest.raises(ValueError, match="undeclared fields"):
        validate()
    assert calls == []


@pytest.mark.parametrize("boundary", ["transition", "placement"])
@pytest.mark.parametrize("location", ["direct", "nested"])
@pytest.mark.parametrize("representation", ["ordinary", "property"])
@pytest.mark.parametrize("include_unknown", [False, True], ids=["valid", "unknown"])
def test_tenancy_boundaries_read_base_model_dict_descriptor(
    boundary: str,
    location: str,
    representation: str,
    include_unknown: bool,
) -> None:
    if boundary == "transition":
        previous = operation()
        candidate = operation(observed_state="running")
        if location == "direct":
            target, calls = model_with_descriptor_storage(
                candidate,
                representation=representation,
                include_unknown=include_unknown,
            )
            candidate = target
        else:
            target, calls = model_with_descriptor_storage(
                candidate.resource_manifests[0],
                representation=representation,
                include_unknown=include_unknown,
            )
            candidate = candidate.model_copy(
                update={"resource_manifests": (target,)}
            )

        def validate() -> object:
            return validate_operation_transition(previous, candidate)

    else:
        candidate = TenantPlacement.model_validate_json(
            json.dumps(
                {
                    "schema_version": VERSION,
                    "tenant_id": "tenant-a",
                    "generation": 7,
                    "resource_manifests": [manifest()],
                }
            )
        )
        if location == "direct":
            target, calls = model_with_descriptor_storage(
                candidate,
                representation=representation,
                include_unknown=include_unknown,
            )
            candidate = target
        else:
            target, calls = model_with_descriptor_storage(
                candidate.resource_manifests[0],
                representation=representation,
                include_unknown=include_unknown,
            )
            candidate = candidate.model_copy(
                update={"resource_manifests": (target,)}
            )

        def validate() -> object:
            return validate_placement_publication(
                candidate,
                expected_tenant_id="tenant-a",
                current_generation=7,
            )

    native = PYDANTIC_DICT_DESCRIPTOR.__get__(target, BaseModel)
    assert dict.__contains__(native, "hidden_unknown") is include_unknown
    if include_unknown:
        with pytest.raises(ValueError, match="undeclared fields"):
            validate()
    else:
        validate()
    assert calls == []


@pytest.mark.parametrize("boundary", ["transition", "placement"])
@pytest.mark.parametrize("location", ["direct", "nested"])
@pytest.mark.parametrize(
    ("case", "should_reject"),
    [
        ("length-one", True),
        ("length-one-false-bool", True),
        ("length-zero-true-bool", False),
    ],
)
def test_tenancy_boundaries_use_length_without_mapping_truthiness(
    boundary: str,
    location: str,
    case: str,
    should_reject: bool,
) -> None:
    if case == "length-one":
        extras = ObservedLengthMapping(1)
    elif case == "length-one-false-bool":
        extras = FalseBooleanLengthMapping(1)
    else:
        extras = TrueBooleanEmptyMapping(0)

    if boundary == "transition":
        previous = operation()
        candidate = operation(observed_state="running")
        target = (
            candidate
            if location == "direct"
            else candidate.resource_manifests[0]
        )
        object.__setattr__(target, "__pydantic_extra__", extras)

        def validate_transition() -> object:
            return validate_operation_transition(previous, candidate)

        validate = validate_transition
    else:
        candidate = TenantPlacement.model_validate_json(
            json.dumps(
                {
                    "schema_version": VERSION,
                    "tenant_id": "tenant-a",
                    "generation": 7,
                    "resource_manifests": [manifest()],
                }
            )
        )
        target = (
            candidate
            if location == "direct"
            else candidate.resource_manifests[0]
        )
        object.__setattr__(target, "__pydantic_extra__", extras)

        def validate_placement() -> object:
            return validate_placement_publication(
                candidate,
                expected_tenant_id="tenant-a",
                current_generation=7,
            )

        validate = validate_placement

    if should_reject:
        with pytest.raises(ValueError, match="undeclared fields"):
            validate()
    else:
        validate()

    assert extras.length_calls == 1
    assert extras.iteration_calls == 1
    assert extras.items_calls == 1
    assert extras.boolean_calls == 0


def test_transition_default_items_view_does_not_repeat_length_hint() -> None:
    previous = operation(resource_manifests=[])
    current = operation(observed_state="running", resource_manifests=[])
    extras = DefaultItemsLengthProbe()
    object.__setattr__(current, "__pydantic_extra__", extras)

    result = validate_operation_transition(previous, current)

    assert result.observed_state == "running"
    assert extras.length_calls == 1
    assert extras.iteration_calls == 2
    assert extras.items_calls == 1


def test_transition_rejects_malformed_nested_extra_storage() -> None:
    previous = operation()
    current = operation(observed_state="running")
    object.__setattr__(
        current.resource_manifests[0],
        "__pydantic_extra__",
        [],
    )

    with pytest.raises(ValueError, match="extra storage must be a mapping"):
        validate_operation_transition(previous, current)


def test_transition_rejects_cyclic_mutated_input_graph() -> None:
    previous = operation(resource_manifests=[])
    current = operation(observed_state="running", resource_manifests=[])
    current.resource_manifests = (current,)  # type: ignore[assignment]

    with pytest.raises(ValueError, match="must be acyclic"):
        validate_operation_transition(previous, current)


def test_transition_revalidates_copied_operation_fields() -> None:
    previous = operation(resource_manifests=[])
    current = operation(observed_state="running", resource_manifests=[])
    invalid = current.model_copy(update={"desired_state": "suspended"})

    with pytest.raises(ValidationError, match="requires desired_state"):
        validate_operation_transition(previous, invalid)


def test_publication_revalidates_post_construction_nested_mutation() -> None:
    placement = TenantPlacement.model_validate_json(
        json.dumps(
            {
                "schema_version": VERSION,
                "tenant_id": "tenant-a",
                "generation": 7,
                "resource_manifests": [manifest()],
            }
        )
    )
    placement.resource_manifests[0].tenant_id = "tenant-b"

    with pytest.raises(ValidationError, match="tenant_id must match"):
        validate_placement_publication(
            placement,
            expected_tenant_id="tenant-a",
            current_generation=7,
        )


@pytest.mark.parametrize(
    "invalid_generation",
    [True, 7.0, -1, 9_007_199_254_740_992],
)
def test_publication_helper_strictly_validates_generation(
    invalid_generation: object,
) -> None:
    placement = TenantPlacement.model_validate_json(
        json.dumps(
            {
                "schema_version": VERSION,
                "tenant_id": "tenant-a",
                "generation": 7,
                "resource_manifests": [],
            }
        )
    )

    with pytest.raises(ValidationError):
        validate_placement_publication(
            placement,
            expected_tenant_id="tenant-a",
            current_generation=invalid_generation,
        )


@pytest.mark.parametrize("invalid_tenant_id", [1, b"tenant-a", ""])
def test_publication_helper_strictly_validates_tenant_id(
    invalid_tenant_id: object,
) -> None:
    placement = TenantPlacement.model_validate_json(
        json.dumps(
            {
                "schema_version": VERSION,
                "tenant_id": "tenant-a",
                "generation": 7,
                "resource_manifests": [],
            }
        )
    )

    with pytest.raises(ValidationError):
        validate_placement_publication(
            placement,
            expected_tenant_id=invalid_tenant_id,
            current_generation=7,
        )


def test_contracts_require_version_and_reject_unknown_fields() -> None:
    document = manifest()
    document.pop("schema_version")
    with pytest.raises(ValidationError):
        ResourceManifestReference.model_validate_json(json.dumps(document))

    with pytest.raises(ValidationError):
        operation(unreviewed_topology="shared-plane")


@pytest.mark.parametrize("invalid_generation", [True, -1, 9_007_199_254_740_992])
def test_generation_uses_safe_counter_bounds(invalid_generation: object) -> None:
    with pytest.raises(ValidationError):
        operation(
            placement_generation=invalid_generation,
            resource_manifests=[],
        )


def _tenancy_nested_boundary_case(boundary: str):
    if boundary == "operation":
        previous = operation(observed_state="pending")
        candidate = operation(observed_state="running")

        def invoke(value):
            return validate_operation_transition(previous, value)

    else:
        candidate = TenantPlacement.model_validate(
            {
                "schema_version": VERSION,
                "tenant_id": "tenant-a",
                "generation": 7,
                "resource_manifests": (manifest(),),
            }
        )

        def invoke(value):
            return validate_placement_publication(
                value,
                expected_tenant_id="tenant-a",
                current_generation=7,
            )

    return candidate, candidate.resource_manifests[0], invoke


def _set_native_field(model: BaseModel, name: str, value: object) -> None:
    native = PYDANTIC_DICT_DESCRIPTOR.__get__(model, BaseModel)
    dict.__setitem__(native, name, value)


def _install_repairing_stored_name(
    model: BaseModel, repair
) -> RepairingName:
    native = PYDANTIC_DICT_DESCRIPTOR.__get__(model, BaseModel)
    stored_value = dict.__getitem__(native, "schema_version")
    dict.__delitem__(native, "schema_version")
    name = RepairingName("schema_version", repair)
    dict.__setitem__(native, name, stored_value)
    name.calls = 0
    name.armed = True
    return name


@pytest.mark.parametrize("boundary", ["operation", "placement"])
def test_tenancy_boundaries_freeze_nested_models_before_stored_name_hooks(
    boundary: str,
) -> None:
    ordinary, ordinary_child, ordinary_invoke = _tenancy_nested_boundary_case(
        boundary
    )
    _set_native_field(ordinary_child, "manifest_id", "")
    with pytest.raises(ValidationError):
        ordinary_invoke(ordinary)

    candidate, child, invoke = _tenancy_nested_boundary_case(boundary)
    _set_native_field(child, "manifest_id", "")
    name = _install_repairing_stored_name(
        candidate,
        lambda: _set_native_field(child, "manifest_id", "primary"),
    )

    with pytest.raises(ValidationError):
        invoke(candidate)

    assert name.calls > 0
    assert child.manifest_id == "primary"


@pytest.mark.parametrize("boundary", ["operation", "placement"])
def test_tenancy_boundaries_freeze_tuple_backing_before_traversal_hooks(
    boundary: str,
) -> None:
    candidate, child, invoke = _tenancy_nested_boundary_case(boundary)
    _set_native_field(child, "manifest_id", "")
    repairing = RepairingTuple(
        candidate.resource_manifests,
        lambda: _set_native_field(child, "manifest_id", "primary"),
    )
    _set_native_field(candidate, "resource_manifests", repairing)

    with pytest.raises(ValidationError):
        invoke(candidate)

    assert repairing.calls == 1
    assert child.manifest_id == "primary"


@pytest.mark.parametrize("boundary", ["operation", "placement"])
def test_tenancy_boundaries_freeze_nested_models_before_extra_mapping_hooks(
    boundary: str,
) -> None:
    candidate, child, invoke = _tenancy_nested_boundary_case(boundary)
    _set_native_field(child, "manifest_id", "")
    extras = RepairingEmptyExtraMapping(
        lambda: _set_native_field(child, "manifest_id", "primary")
    )
    PYDANTIC_EXTRA_DESCRIPTOR.__set__(candidate, extras)

    with pytest.raises(ValidationError):
        invoke(candidate)

    assert extras.calls == 3
    assert child.manifest_id == "primary"


@pytest.mark.parametrize("boundary", ["operation", "placement"])
def test_tenancy_boundaries_preserve_strict_list_rejection_with_public_hook(
    boundary: str,
) -> None:
    candidate, child, invoke = _tenancy_nested_boundary_case(boundary)
    _set_native_field(child, "manifest_id", "")
    repairing = RepairingList(
        candidate.resource_manifests,
        lambda: _set_native_field(child, "manifest_id", "primary"),
    )
    _set_native_field(candidate, "resource_manifests", repairing)

    with pytest.raises(ValidationError) as exc_info:
        invoke(candidate)

    assert any(error["type"] == "tuple_type" for error in exc_info.value.errors())
    assert repairing.calls == 1
    assert child.manifest_id == "primary"


@pytest.mark.parametrize("boundary", ["operation", "placement"])
def test_tenancy_boundaries_leave_fields_set_hooks_inert(boundary: str) -> None:
    candidate, child, invoke = _tenancy_nested_boundary_case(boundary)
    _set_native_field(child, "manifest_id", "")
    name = RepairingName(
        "manifest_id",
        lambda: _set_native_field(child, "manifest_id", "primary"),
    )
    fields_set = PYDANTIC_FIELDS_SET_DESCRIPTOR.__get__(candidate, type(candidate))
    set.add(fields_set, name)
    name.calls = 0
    name.armed = True

    with pytest.raises(ValidationError):
        invoke(candidate)

    assert name.calls == 0
    assert child.manifest_id == ""


@pytest.mark.parametrize("boundary", ["operation", "placement"])
def test_tenancy_boundaries_retain_invalid_root_before_name_hook_repair(
    boundary: str,
) -> None:
    candidate, _child, invoke = _tenancy_nested_boundary_case(boundary)
    _set_native_field(candidate, "tenant_id", "")
    name = _install_repairing_stored_name(
        candidate,
        lambda: _set_native_field(candidate, "tenant_id", "tenant-a"),
    )

    with pytest.raises(ValidationError):
        invoke(candidate)

    assert name.calls > 0
    assert candidate.tenant_id == "tenant-a"


@pytest.mark.parametrize("boundary", ["operation", "placement"])
def test_tenancy_boundaries_preserve_valid_tuple_subclasses(boundary: str) -> None:
    candidate, _child, invoke = _tenancy_nested_boundary_case(boundary)
    calls: list[str] = []
    repairing = RepairingTuple(
        candidate.resource_manifests,
        lambda: calls.append("iterated"),
    )
    _set_native_field(candidate, "resource_manifests", repairing)

    result = invoke(candidate)

    assert result.resource_manifests[0].manifest_id == "primary"
    assert repairing.calls == 1
    assert calls == ["iterated"]


@pytest.mark.parametrize("boundary", ["operation", "placement"])
def test_tenancy_boundaries_preserve_divergent_tuple_public_view(
    boundary: str,
) -> None:
    candidate, _child, invoke = _tenancy_nested_boundary_case(boundary)
    divergent = DivergentTuple(candidate.resource_manifests, ())
    _set_native_field(candidate, "resource_manifests", divergent)

    result = invoke(candidate)

    assert result.resource_manifests == ()
    assert divergent.calls == 1


@pytest.mark.parametrize("boundary", ["operation", "placement"])
def test_tenancy_boundaries_reject_cross_tenant_public_tuple_target(
    boundary: str,
) -> None:
    candidate, _child, invoke = _tenancy_nested_boundary_case(boundary)
    cross_tenant = ResourceManifestReference.model_validate(
        manifest(tenant_id="tenant-b")
    )
    divergent = DivergentTuple(
        candidate.resource_manifests,
        (cross_tenant,),
    )
    _set_native_field(candidate, "resource_manifests", divergent)

    with pytest.raises(ValidationError, match="tenant_id must match"):
        invoke(candidate)

    assert divergent.calls == 1


@pytest.mark.parametrize("boundary", ["operation", "placement"])
def test_tenancy_boundaries_freeze_later_model_edge_before_name_callback(
    boundary: str,
) -> None:
    candidate, child, invoke = _tenancy_nested_boundary_case(boundary)
    original_edge = candidate.resource_manifests
    _set_native_field(child, "manifest_id", "")
    name = _install_repairing_stored_name(
        candidate,
        lambda: _set_native_field(candidate, "resource_manifests", ()),
    )

    with pytest.raises(ValidationError):
        invoke(candidate)

    assert name.calls > 0
    assert candidate.resource_manifests == ()
    assert original_edge[0].manifest_id == ""


@pytest.mark.parametrize("boundary", ["operation", "placement"])
def test_tenancy_boundaries_preserve_benign_string_subclass_names(
    boundary: str,
) -> None:
    candidate, _child, invoke = _tenancy_nested_boundary_case(boundary)
    name = _install_repairing_stored_name(candidate, lambda: None)

    result = invoke(candidate)

    assert result.tenant_id == "tenant-a"
    assert name.calls > 0


@pytest.mark.parametrize("boundary", ["operation", "placement"])
def test_tenancy_boundaries_preserve_absent_fields_set_policy(boundary: str) -> None:
    candidate, child, invoke = _tenancy_nested_boundary_case(boundary)
    PYDANTIC_FIELDS_SET_DESCRIPTOR.__delete__(candidate)
    PYDANTIC_FIELDS_SET_DESCRIPTOR.__delete__(child)

    assert invoke(candidate).tenant_id == "tenant-a"


@pytest.mark.parametrize("boundary", ["operation", "placement"])
def test_tenancy_boundaries_accept_truly_absent_extra_storage(boundary: str) -> None:
    candidate, child, invoke = _tenancy_nested_boundary_case(boundary)
    PYDANTIC_EXTRA_DESCRIPTOR.__delete__(candidate)
    PYDANTIC_EXTRA_DESCRIPTOR.__delete__(child)

    assert invoke(candidate).tenant_id == "tenant-a"
