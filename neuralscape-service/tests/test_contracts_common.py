"""Conformance tests for the shared candidate contract foundation."""

from collections.abc import ItemsView, Iterator, KeysView, Mapping
import json
from types import MappingProxyType
from typing import Callable, get_args

import pytest
from pydantic import (
    AliasChoices,
    AliasPath,
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    model_validator,
)

from contracts_common import (
    ContractModel,
    MAX_SAFE_INTEGER,
    OpaqueId,
    SafeCounter,
    VersionedContract,
    snapshot_contract_graph,
)
from contracts_errors import (
    InternalErrorCode,
    PublicErrorCode,
    public_error_for,
)
from contracts_references import ReferenceHandle, ReferenceKind, SourceVersion


_BASE_MODEL_DICT_DESCRIPTOR = vars(BaseModel)["__dict__"]
_BASE_MODEL_EXTRA_DESCRIPTOR = vars(BaseModel)["__pydantic_extra__"]
_BASE_MODEL_FIELDS_SET_DESCRIPTOR = vars(BaseModel)["__pydantic_fields_set__"]


class ExampleContract(ContractModel):
    count: SafeCounter


class DefaultedExampleContract(ContractModel):
    count: SafeCounter = 7


class AliasedExampleContract(ContractModel):
    count: SafeCounter = Field(alias="n")


class ChoiceAliasedExampleContract(ContractModel):
    count: SafeCounter = Field(
        validation_alias=AliasChoices(
            "n",
            "number",
            AliasPath("payload", "count"),
        )
    )


class PathAliasedExampleContract(ContractModel):
    count: SafeCounter = Field(validation_alias=AliasPath("payload", "count"))


class AliasedExampleEnvelope(ContractModel):
    item: AliasedExampleContract


class CrossStringAliasContract(ContractModel):
    primary: SafeCounter = Field(alias="shadow")
    shadow: SafeCounter


class CrossPathAliasContract(ContractModel):
    payload: dict[str, SafeCounter]
    count: SafeCounter = Field(validation_alias=AliasPath("payload", "count"))


class CrossChoiceAliasContract(ContractModel):
    primary: SafeCounter = Field(
        validation_alias=AliasChoices("primary", "shadow")
    )
    shadow: SafeCounter


class CrossChoicePathAliasContract(ContractModel):
    payload: dict[str, SafeCounter]
    count: SafeCounter = Field(
        validation_alias=AliasChoices(
            "count",
            AliasPath("payload", "count"),
        )
    )


class CrossStringAliasEnvelope(ContractModel):
    item: CrossStringAliasContract


class CrossPathAliasEnvelope(ContractModel):
    item: CrossPathAliasContract


class CrossChoiceAliasEnvelope(ContractModel):
    item: CrossChoiceAliasContract


class CrossChoicePathAliasEnvelope(ContractModel):
    item: CrossChoicePathAliasContract


class CrossDefaultAliasContract(ContractModel):
    primary: SafeCounter = Field(
        default=7,
        validation_alias=AliasChoices("primary", "shadow"),
    )
    shadow: SafeCounter = 11


class CrossAliasBaseContract(ContractModel):
    primary: SafeCounter = Field(
        validation_alias=AliasChoices("primary", "shadow")
    )
    shadow: SafeCounter


class CrossAliasDerivedContract(CrossAliasBaseContract):
    category: OpaqueId = Field(
        validation_alias=AliasChoices("category", "kind")
    )
    kind: OpaqueId


class SerializationAliasedExampleContract(ContractModel):
    count: SafeCounter = Field(serialization_alias="n")


class NameOnlyAliasedExampleContract(ContractModel):
    model_config = ConfigDict(validate_by_alias=False, validate_by_name=True)

    count: SafeCounter = Field(alias="n")


class InheritedAliasExampleContract(ContractModel):
    count: SafeCounter = Field(validation_alias=AliasChoices("n", "count"))


class DerivedAliasedExampleContract(InheritedAliasExampleContract):
    category: OpaqueId = Field(
        validation_alias=AliasChoices(
            "kind",
            "category",
            AliasPath("metadata", "category"),
        )
    )


class NestedExampleContract(ContractModel):
    identifier: OpaqueId
    count: SafeCounter


class DerivedNestedExampleContract(NestedExampleContract):
    category: OpaqueId


class ExampleContractGraph(ContractModel):
    primary: NestedExampleContract
    by_name: dict[str, NestedExampleContract]


class ExampleContractEnvelope(ContractModel):
    item: ExampleContract


class ExampleContractListEnvelope(ContractModel):
    items: list[ExampleContract]


class NestedInventoryEnvelope(ContractModel):
    tuple_trigger: tuple[int, ...]
    list_trigger: list[int]
    child: ExampleContract


class NativeEdgeEnvelope(ContractModel):
    trigger: tuple[int, ...]
    children: list[ExampleContract]


class TenantTargetEnvelope(ContractModel):
    tenant_id: OpaqueId
    target_refs: tuple[ReferenceHandle, ...]

    @model_validator(mode="after")
    def validate_target_tenants(self) -> "TenantTargetEnvelope":
        if any(item.tenant_id != self.tenant_id for item in self.target_refs):
            raise ValueError("target_refs must match tenant_id")
        return self


class DictionaryEnvelope(ContractModel):
    payload: dict[str, SafeCounter]


class DerivedExampleContractGraph(ContractModel):
    item: DerivedNestedExampleContract


class HiddenKeysExtra(dict[str, object]):
    def keys(self) -> KeysView[str]:
        return {}.keys()


class SingleReadExtra(dict[str, object]):
    item_reads = 0

    def keys(self) -> KeysView[str]:
        raise AssertionError("extra keys must come from captured entries")

    def items(self) -> ItemsView[str, object]:
        self.item_reads += 1
        if self.item_reads > 1:
            raise AssertionError("extra entries must be captured exactly once")
        return super().items()


class SingleReadMapping(Mapping[str, object]):
    def __init__(self, entries: dict[str, object]) -> None:
        self._entries = entries
        self.item_reads = 0

    def __getitem__(self, key: str) -> object:
        return self._entries[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def items(self) -> ItemsView[str, object]:
        self.item_reads += 1
        if self.item_reads > 1:
            raise AssertionError("mapping entries must be captured exactly once")
        return self._entries.items()


class DuplicateItemsMapping(Mapping[str, object]):
    def __init__(self, entries: tuple[tuple[str, object], ...]) -> None:
        self._entries = entries
        self.item_reads = 0

    def __getitem__(self, key: str) -> object:
        raise AssertionError("duplicate mapping getitem must not be called")

    def __iter__(self) -> Iterator[str]:
        raise AssertionError("duplicate mapping iterator must not be called")

    def __len__(self) -> int:
        raise AssertionError("duplicate mapping len must not be called")

    def items(self) -> tuple[tuple[str, object], ...]:
        self.item_reads += 1
        if self.item_reads > 1:
            raise AssertionError("duplicate mapping items must be captured once")
        return self._entries


class RepairingMapping(Mapping[str, object]):
    def __init__(self, callback: Callable[[], None]) -> None:
        self.callback = callback
        self.item_reads = 0

    def __getitem__(self, key: str) -> object:
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return iter(())

    def __len__(self) -> int:
        return 0

    def items(self) -> tuple[tuple[str, object], ...]:
        self.item_reads += 1
        callback = self.callback
        self.callback = lambda: None
        callback()
        return ()


class DictClassSpoofingMapping(SingleReadMapping):
    @property
    def __class__(self) -> type[dict]:
        return dict


class ClassReadRejectingMapping(SingleReadMapping):
    @property
    def __class__(self) -> type[object]:
        raise AssertionError("mapping __class__ override must not be read")


class ClassReadCountingMapping(SingleReadMapping):
    def __init__(self, entries: dict[str, object]) -> None:
        super().__init__(entries)
        self.class_reads = 0

    @property
    def __class__(self) -> type[dict]:
        self.class_reads += 1
        return dict


class ClassReadRejectingContract(ExampleContract):
    @property
    def __class__(self) -> type[object]:
        raise AssertionError("model __class__ override must not be read")


class ClassReadRejectingTuple(tuple):
    @property
    def __class__(self) -> type[object]:
        raise AssertionError("tuple __class__ override must not be read")


class ClassReadRejectingList(list):
    @property
    def __class__(self) -> type[object]:
        raise AssertionError("list __class__ override must not be read")


class ClassReadRejectingDict(dict[str, object]):
    @property
    def __class__(self) -> type[object]:
        raise AssertionError("dict __class__ override must not be read")


class ModelClassSpoofingDict(dict[str, object]):
    @property
    def __class__(self) -> type[ExampleContract]:
        return ExampleContract


class ModelDictViewHidingExampleContract(ExampleContract):
    def __getattribute__(self, name: str) -> object:
        if name == "__dict__":
            native = object.__getattribute__(self, "__dict__")
            return {
                key: item
                for key, item in dict.items(native)
                if key != "future_state"
            }
        return super().__getattribute__(name)


class ModelExtraViewHidingExampleContract(ExampleContract):
    def __getattribute__(self, name: str) -> object:
        if name == "__pydantic_extra__":
            return {}
        return super().__getattribute__(name)


class ModelDictDescriptorHidingExampleContract(ExampleContract):
    @property
    def __dict__(self):  # type: ignore[override]
        native = _BASE_MODEL_DICT_DESCRIPTOR.__get__(self, BaseModel)
        return {
            key: item
            for key, item in dict.items(native)
            if key != "future_state"
        }

    @__dict__.setter
    def __dict__(self, value):  # type: ignore[no-untyped-def]
        _BASE_MODEL_DICT_DESCRIPTOR.__set__(self, value)


class ModelExtraDescriptorHidingExampleContract(ExampleContract):
    @property
    def __pydantic_extra__(self):  # type: ignore[override]
        return None

    @__pydantic_extra__.setter
    def __pydantic_extra__(self, value):  # type: ignore[no-untyped-def]
        _BASE_MODEL_EXTRA_DESCRIPTOR.__set__(self, value)


class HiddenItemsDict(dict[str, object]):
    def items(self) -> tuple[tuple[str, object], ...]:
        return tuple(
            (key, item)
            for key, item in dict.items(self)
            if key not in {"injected", "n", "future_state", "hidden"}
        )


class EmptyItemsDict(dict[str, object]):
    def items(self) -> tuple[tuple[str, object], ...]:
        return ()


class RaisingInventoryDict(dict[str, object]):
    def items(self) -> ItemsView[str, object]:
        raise AssertionError("dict items override must not be called")

    def keys(self) -> KeysView[str]:
        raise AssertionError("dict keys override must not be called")

    def __iter__(self) -> Iterator[str]:
        raise AssertionError("dict iterator override must not be called")

    def __bool__(self) -> bool:
        raise AssertionError("dict truth override must not be called")


class FabricatingItemsDict(dict[str, object]):
    def items(self) -> tuple[tuple[str, object], ...]:
        return (*tuple(dict.items(self)), ("fabricated", 99))


class RepairingTuple(tuple[int, ...]):
    def __new__(
        cls,
        values: tuple[int, ...],
        callback: Callable[[], None],
    ) -> "RepairingTuple":
        instance = tuple.__new__(cls, values)
        instance.callback = callback
        instance.iteration_calls = 0
        return instance

    def __iter__(self):  # type: ignore[no-untyped-def]
        self.iteration_calls += 1
        callback = self.callback
        self.callback = lambda: None
        callback()
        return tuple.__iter__(self)


class RepairingList(list[int]):
    def __init__(self, values: list[int], callback: Callable[[], None]) -> None:
        list.__init__(self, values)
        self.callback = callback
        self.iteration_calls = 0

    def __iter__(self):  # type: ignore[no-untyped-def]
        self.iteration_calls += 1
        callback = self.callback
        self.callback = lambda: None
        callback()
        return list.__iter__(self)


class DivergentTuple(tuple[object, ...]):
    def __new__(
        cls,
        native_values: tuple[object, ...],
        public_values: tuple[object, ...],
    ) -> "DivergentTuple":
        instance = tuple.__new__(cls, native_values)
        instance.public_values = public_values
        instance.iteration_calls = 0
        return instance

    def __iter__(self):  # type: ignore[no-untyped-def]
        self.iteration_calls += 1
        return iter(self.public_values)


class DivergentList(list[object]):
    def __init__(
        self,
        native_values: list[object],
        public_values: list[object],
    ) -> None:
        list.__init__(self, native_values)
        self.public_values = public_values
        self.iteration_calls = 0

    def __iter__(self):  # type: ignore[no-untyped-def]
        self.iteration_calls += 1
        return iter(self.public_values)


class RepairingStoredName(str):
    def __new__(
        cls,
        value: str,
        callback: Callable[[], None],
        hook: str,
    ) -> "RepairingStoredName":
        instance = str.__new__(cls, value)
        instance.callback = callback
        instance.hook = hook
        instance.armed = False
        instance.hash_calls = 0
        instance.equality_calls = 0
        return instance

    def __hash__(self) -> int:
        self.hash_calls += 1
        if self.armed and self.hook == "hash":
            self.armed = False
            self.callback()
        return str.__hash__(self)

    def __eq__(self, other: object) -> bool:
        self.equality_calls += 1
        if self.armed and self.hook == "equality":
            self.armed = False
            self.callback()
        return str.__eq__(self, other)


def _invalid_nested_inventory_envelope() -> tuple[
    NestedInventoryEnvelope,
    ExampleContract,
]:
    child = ExampleContract(count=1).model_copy(update={"count": -1})
    envelope = NestedInventoryEnvelope(
        tuple_trigger=(),
        list_trigger=[],
        child=ExampleContract(count=1),
    ).model_copy(update={"child": child})
    return envelope, child


@pytest.mark.parametrize("container_kind", ["tuple", "list"])
def test_snapshot_contract_graph_freezes_later_model_before_container_hook(
    container_kind: str,
) -> None:
    ordinary, _ = _invalid_nested_inventory_envelope()
    ordinary_snapshot = snapshot_contract_graph(ordinary)
    with pytest.raises(ValidationError) as ordinary_error:
        NestedInventoryEnvelope.model_validate(ordinary_snapshot, strict=True)
    assert ordinary_error.value.errors()[0]["type"] == "greater_than_equal"
    assert ordinary_error.value.errors()[0]["loc"] == ("child", "count")
    assert ordinary_error.value.errors()[0]["input"] == -1

    value, child = _invalid_nested_inventory_envelope()
    native = _BASE_MODEL_DICT_DESCRIPTOR.__get__(value, BaseModel)

    def repair_child() -> None:
        child_native = _BASE_MODEL_DICT_DESCRIPTOR.__get__(child, BaseModel)
        dict.__setitem__(child_native, "count", 1)

    if container_kind == "tuple":
        trigger: RepairingTuple | RepairingList = RepairingTuple(
            native["tuple_trigger"],
            repair_child,
        )
        dict.__setitem__(native, "tuple_trigger", trigger)
    else:
        trigger = RepairingList(native["list_trigger"], repair_child)
        dict.__setitem__(native, "list_trigger", trigger)

    snapshot = snapshot_contract_graph(value)

    assert trigger.iteration_calls == 1
    assert child.count == 1
    assert snapshot["child"]["count"] == -1
    with pytest.raises(ValidationError) as repaired_error:
        NestedInventoryEnvelope.model_validate(snapshot, strict=True)
    assert repaired_error.value.errors() == ordinary_error.value.errors()


@pytest.mark.parametrize("container_kind", ["tuple", "list"])
def test_snapshot_contract_graph_preserves_sequence_subclass_public_view(
    container_kind: str,
) -> None:
    if container_kind == "tuple":
        value: DivergentTuple | DivergentList = DivergentTuple((1,), (2,))
        expected: tuple[int, ...] | list[int] = (2,)
        adapter = TypeAdapter(tuple[int, ...])
    else:
        value = DivergentList([1], [2])
        expected = [2]
        adapter = TypeAdapter(list[int])

    snapshot = snapshot_contract_graph(value)

    assert snapshot == expected
    assert type(snapshot) is type(expected)
    assert value.iteration_calls == 1
    assert adapter.validate_python(snapshot, strict=True) == expected


def test_snapshot_contract_graph_exposes_hidden_cross_tenant_target() -> None:
    cross_tenant = ReferenceHandle(
        kind="artifact",
        id="artifact-1",
        tenant_id="tenant-2",
        resolver="resolve_artifact",
    )
    target_refs = DivergentTuple((), (cross_tenant,))
    value = TenantTargetEnvelope(tenant_id="tenant-1", target_refs=())
    native = _BASE_MODEL_DICT_DESCRIPTOR.__get__(value, BaseModel)
    dict.__setitem__(native, "target_refs", target_refs)

    snapshot = snapshot_contract_graph(value)

    assert snapshot["target_refs"] == (
        {
            "kind": "artifact",
            "id": "artifact-1",
            "tenant_id": "tenant-2",
            "resolver": "resolve_artifact",
            "cross_tenant": False,
        },
    )
    assert target_refs.iteration_calls == 1
    with pytest.raises(ValidationError, match="target_refs must match tenant_id"):
        TenantTargetEnvelope.model_validate(snapshot, strict=True)


@pytest.mark.parametrize("mutation", ["replace", "remove"])
def test_snapshot_contract_graph_freezes_plain_list_edges_before_earlier_hook(
    mutation: str,
) -> None:
    invalid_child = ExampleContract(count=1).model_copy(update={"count": -1})
    children = [invalid_child]

    def mutate_edge() -> None:
        if mutation == "replace":
            list.__setitem__(children, 0, ExampleContract(count=1))
        else:
            list.clear(children)

    trigger = RepairingTuple((), mutate_edge)
    value = NativeEdgeEnvelope(trigger=(), children=[ExampleContract(count=1)])
    native = _BASE_MODEL_DICT_DESCRIPTOR.__get__(value, BaseModel)
    dict.__setitem__(native, "trigger", trigger)
    dict.__setitem__(native, "children", children)

    snapshot = snapshot_contract_graph(value)

    assert trigger.iteration_calls == 1
    assert snapshot["children"] == [{"count": -1}]
    if mutation == "replace":
        assert children[0].count == 1
    else:
        assert children == []
    with pytest.raises(ValidationError) as raised:
        NativeEdgeEnvelope.model_validate(snapshot, strict=True)
    assert raised.value.errors()[0]["type"] == "greater_than_equal"
    assert raised.value.errors()[0]["loc"] == ("children", 0, "count")
    assert raised.value.errors()[0]["input"] == -1


def test_snapshot_contract_graph_freezes_later_model_before_mapping_hook() -> None:
    ordinary, _ = _invalid_nested_inventory_envelope()
    ordinary_snapshot = snapshot_contract_graph(ordinary)
    with pytest.raises(ValidationError) as ordinary_error:
        NestedInventoryEnvelope.model_validate(ordinary_snapshot, strict=True)

    value, child = _invalid_nested_inventory_envelope()

    def repair_child() -> None:
        child_native = _BASE_MODEL_DICT_DESCRIPTOR.__get__(child, BaseModel)
        dict.__setitem__(child_native, "count", 1)

    extras = RepairingMapping(repair_child)
    _BASE_MODEL_EXTRA_DESCRIPTOR.__set__(value, extras)

    snapshot = snapshot_contract_graph(value)

    assert extras.item_reads == 1
    assert child.count == 1
    assert snapshot["child"]["count"] == -1
    with pytest.raises(ValidationError) as repaired_error:
        NestedInventoryEnvelope.model_validate(snapshot, strict=True)
    assert repaired_error.value.errors() == ordinary_error.value.errors()


@pytest.mark.parametrize("hook", ["hash", "equality"])
def test_snapshot_contract_graph_freezes_later_model_before_stored_name_hook(
    hook: str,
) -> None:
    ordinary, _ = _invalid_nested_inventory_envelope()
    ordinary_snapshot = snapshot_contract_graph(ordinary)
    with pytest.raises(ValidationError) as ordinary_error:
        NestedInventoryEnvelope.model_validate(ordinary_snapshot, strict=True)

    value, child = _invalid_nested_inventory_envelope()
    native = _BASE_MODEL_DICT_DESCRIPTOR.__get__(value, BaseModel)

    def repair_child() -> None:
        child_native = _BASE_MODEL_DICT_DESCRIPTOR.__get__(child, BaseModel)
        dict.__setitem__(child_native, "count", 1)

    stored_name = RepairingStoredName("tuple_trigger", repair_child, hook)
    stored_value = dict.pop(native, "tuple_trigger")
    dict.__setitem__(native, stored_name, stored_value)
    stored_name.armed = True
    stored_name.hash_calls = stored_name.equality_calls = 0  # storing it above hashed it once

    snapshot = snapshot_contract_graph(value)

    # A stored name is read as its text: its hash and equality hooks never run, so the
    # repair never happens and the frozen invalid child is what validation sees.
    assert stored_name.hash_calls == stored_name.equality_calls == 0
    assert child.count == -1
    assert snapshot["child"]["count"] == -1
    with pytest.raises(ValidationError) as repaired_error:
        NestedInventoryEnvelope.model_validate(snapshot, strict=True)
    assert repaired_error.value.errors() == ordinary_error.value.errors()


@pytest.mark.parametrize("hook", ["hash", "equality"])
def test_snapshot_contract_graph_freezes_invalid_root_before_name_hook(
    hook: str,
) -> None:
    value = ExampleContract(count=1).model_copy(update={"count": -1})
    native = _BASE_MODEL_DICT_DESCRIPTOR.__get__(value, BaseModel)

    def repair_root() -> None:
        dict.__setitem__(native, "count", 1)

    stored_name = RepairingStoredName("count", repair_root, hook)
    stored_value = dict.pop(native, "count")
    dict.__setitem__(native, stored_name, stored_value)
    stored_name.armed = True
    stored_name.hash_calls = stored_name.equality_calls = 0

    snapshot = snapshot_contract_graph(value)

    assert stored_name.hash_calls == stored_name.equality_calls == 0  # read as text, no hook
    assert snapshot["count"] == -1
    with pytest.raises(ValidationError) as raised:
        ExampleContract.model_validate(snapshot, strict=True)
    assert raised.value.errors()[0]["type"] == "greater_than_equal"
    assert raised.value.errors()[0]["loc"] == ("count",)
    assert raised.value.errors()[0]["input"] == -1


@pytest.mark.parametrize("hook", ["hash", "equality"])
def test_snapshot_contract_graph_preserves_valid_string_subclass_names(
    hook: str,
) -> None:
    value = ExampleContract(count=1)
    native = _BASE_MODEL_DICT_DESCRIPTOR.__get__(value, BaseModel)
    stored_name = RepairingStoredName("count", lambda: None, hook)
    stored_value = dict.pop(native, "count")
    dict.__setitem__(native, stored_name, stored_value)
    stored_name.armed = True

    snapshot = snapshot_contract_graph(value)

    assert snapshot == {"count": 1}
    assert ExampleContract.model_validate(snapshot, strict=True).count == 1


def test_snapshot_contract_graph_preserves_no_fields_set_policy() -> None:
    value = ExampleContract(count=1)
    fields_set = _BASE_MODEL_FIELDS_SET_DESCRIPTOR.__get__(value, BaseModel)
    set.add(fields_set, "future_constraint")

    snapshot = snapshot_contract_graph(value)

    assert snapshot == {"count": 1}
    assert ExampleContract.model_validate(snapshot, strict=True).count == 1


@pytest.mark.parametrize("source", ["copy", "construct"])
def test_snapshot_contract_graph_preserves_complete_stored_state(source: str) -> None:
    if source == "copy":
        value = ExampleContract(count=1).model_copy(
            update={"unreviewed_state": "preserved"}
        )
    else:
        value = ExampleContract.model_construct(count=1)
        value.__dict__["unreviewed_state"] = "preserved"

    snapshot = snapshot_contract_graph(value)

    assert snapshot == {"count": 1, "unreviewed_state": "preserved"}
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ExampleContract.model_validate(snapshot, strict=True)


@pytest.mark.parametrize("nested", [False, True], ids=["direct", "nested"])
def test_snapshot_contract_graph_reads_hidden_model_backing_entries(
    nested: bool,
) -> None:
    value = ExampleContract(count=1)
    hidden = HiddenItemsDict(count=1, injected=2)
    object.__setattr__(value, "__dict__", hidden)
    graph: object = {"item": value} if nested else value
    receiver = ExampleContractEnvelope if nested else ExampleContract

    assert list(hidden.items()) == [("count", 1)]
    assert list(dict.items(hidden)) == [("count", 1), ("injected", 2)]
    snapshot = snapshot_contract_graph(graph)

    expected = {"item": {"count": 1, "injected": 2}} if nested else {
        "count": 1,
        "injected": 2,
    }
    assert snapshot == expected
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        receiver.model_validate(snapshot, strict=True)


def test_snapshot_contract_graph_reads_hidden_stored_alias_collision() -> None:
    value = AliasedExampleContract(n=1)
    hidden = HiddenItemsDict(n=9)
    object.__setattr__(value, "__dict__", hidden)

    assert list(hidden.items()) == []
    assert list(dict.items(hidden)) == [("n", 9)]
    with pytest.raises(ValueError, match="conflicting declared and extra fields"):
        snapshot_contract_graph(value)


def test_snapshot_contract_graph_reads_hidden_cross_field_alias_root() -> None:
    value = CrossStringAliasContract(shadow=1)
    hidden = EmptyItemsDict(shadow=9)
    object.__setattr__(value, "__dict__", hidden)

    assert list(hidden.items()) == []
    assert list(dict.items(hidden)) == [("shadow", 9)]
    with pytest.raises(ValueError) as raised:
        snapshot_contract_graph(value)
    assert str(raised.value) == (
        "contract input contains stored validation alias roots "
        "with missing owning fields"
    )


@pytest.mark.parametrize(
    "stored",
    [RaisingInventoryDict(count=1), FabricatingItemsDict(count=1)],
    ids=["raising-overrides", "fabricated-entry"],
)
def test_snapshot_contract_graph_bypasses_model_dict_inventory_overrides(
    stored: dict[str, object],
) -> None:
    value = ExampleContract(count=1)
    object.__setattr__(value, "__dict__", stored)

    snapshot = snapshot_contract_graph(value)

    assert snapshot == {"count": 1}
    assert ExampleContract.model_validate(snapshot, strict=True) == value


@pytest.mark.parametrize("nested", [False, True], ids=["direct", "nested"])
def test_snapshot_contract_graph_uses_native_model_storage(
    nested: bool,
) -> None:
    value = ModelDictViewHidingExampleContract(count=1)
    native = RaisingInventoryDict(count=1, future_state="preserved")
    object.__setattr__(value, "__dict__", native)
    graph: object = {"item": value} if nested else value
    receiver = ExampleContractEnvelope if nested else ExampleContract

    assert value.__dict__ == {"count": 1}
    assert dict(dict.items(object.__getattribute__(value, "__dict__"))) == {
        "count": 1,
        "future_state": "preserved",
    }

    snapshot = snapshot_contract_graph(graph)

    expected = (
        {"item": {"count": 1, "future_state": "preserved"}}
        if nested
        else {"count": 1, "future_state": "preserved"}
    )
    assert snapshot == expected
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        receiver.model_validate(snapshot, strict=True)


@pytest.mark.parametrize("nested", [False, True], ids=["direct", "nested"])
def test_snapshot_contract_graph_uses_native_model_extra_storage(
    nested: bool,
) -> None:
    value = ModelExtraViewHidingExampleContract(count=1)
    native = RaisingInventoryDict(future_state="preserved")
    object.__setattr__(value, "__pydantic_extra__", native)
    graph: object = {"item": value} if nested else value
    receiver = ExampleContractEnvelope if nested else ExampleContract

    assert value.__pydantic_extra__ == {}
    assert dict(
        dict.items(object.__getattribute__(value, "__pydantic_extra__"))
    ) == {"future_state": "preserved"}

    snapshot = snapshot_contract_graph(graph)

    expected = (
        {"item": {"count": 1, "future_state": "preserved"}}
        if nested
        else {"count": 1, "future_state": "preserved"}
    )
    assert snapshot == expected
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        receiver.model_validate(snapshot, strict=True)


@pytest.mark.parametrize("nested", [False, True], ids=["direct", "nested"])
@pytest.mark.parametrize("storage", ["stored", "extra"])
def test_snapshot_contract_graph_bypasses_model_storage_descriptors(
    nested: bool,
    storage: str,
) -> None:
    descriptor_type = (
        ModelDictDescriptorHidingExampleContract
        if storage == "stored"
        else ModelExtraDescriptorHidingExampleContract
    )
    ordinary = ExampleContract(count=1)
    descriptor = descriptor_type(count=1)
    ordinary_graph: object = {"item": ordinary} if nested else ordinary
    descriptor_graph: object = {"item": descriptor} if nested else descriptor
    receiver = ExampleContractEnvelope if nested else ExampleContract
    valid_snapshot = {"item": {"count": 1}} if nested else {"count": 1}

    assert snapshot_contract_graph(ordinary_graph) == valid_snapshot
    assert snapshot_contract_graph(descriptor_graph) == valid_snapshot
    receiver.model_validate(valid_snapshot, strict=True)

    if storage == "stored":
        ordinary_backing = _BASE_MODEL_DICT_DESCRIPTOR.__get__(
            ordinary,
            BaseModel,
        )
        descriptor_backing = _BASE_MODEL_DICT_DESCRIPTOR.__get__(
            descriptor,
            BaseModel,
        )
        dict.__setitem__(ordinary_backing, "future_state", "preserved")
        dict.__setitem__(descriptor_backing, "future_state", "preserved")
        assert tuple(dict.items(descriptor_backing)) == (
            ("count", 1),
            ("future_state", "preserved"),
        )
        assert "future_state" not in descriptor.__dict__
    else:
        _BASE_MODEL_EXTRA_DESCRIPTOR.__set__(
            ordinary,
            {"future_state": "preserved"},
        )
        _BASE_MODEL_EXTRA_DESCRIPTOR.__set__(
            descriptor,
            {"future_state": "preserved"},
        )
        descriptor_backing = _BASE_MODEL_EXTRA_DESCRIPTOR.__get__(
            descriptor,
            BaseModel,
        )
        assert tuple(dict.items(descriptor_backing)) == (
            ("future_state", "preserved"),
        )
        assert descriptor.__pydantic_extra__ is None

    expected_snapshot = (
        {"item": {"count": 1, "future_state": "preserved"}}
        if nested
        else {"count": 1, "future_state": "preserved"}
    )
    ordinary_snapshot = snapshot_contract_graph(ordinary_graph)
    descriptor_snapshot = snapshot_contract_graph(descriptor_graph)

    assert descriptor_snapshot == ordinary_snapshot == expected_snapshot
    with pytest.raises(ValidationError) as ordinary_error:
        receiver.model_validate(ordinary_snapshot, strict=True)
    with pytest.raises(ValidationError) as descriptor_error:
        receiver.model_validate(descriptor_snapshot, strict=True)
    assert descriptor_error.value.errors() == ordinary_error.value.errors()


def test_snapshot_contract_graph_freezes_extras_before_nested_removal() -> None:
    ordinary = ExampleContractListEnvelope(items=[ExampleContract(count=1)])
    object.__setattr__(
        ordinary,
        "__pydantic_extra__",
        {"future_state": "preserved"},
    )
    ordinary_snapshot = snapshot_contract_graph(ordinary)

    parent: list[ExampleContractListEnvelope | None] = [None]

    class RemovingList(list[ExampleContract]):
        def __iter__(self) -> Iterator[ExampleContract]:
            if parent[0] is not None:
                extras = object.__getattribute__(
                    parent[0],
                    "__pydantic_extra__",
                )
                dict.clear(extras)
            return super().__iter__()

    value = ExampleContractListEnvelope(items=[ExampleContract(count=1)])
    object.__setattr__(value, "items", RemovingList([ExampleContract(count=1)]))
    object.__setattr__(
        value,
        "__pydantic_extra__",
        {"future_state": "preserved"},
    )
    parent[0] = value

    snapshot = snapshot_contract_graph(value)

    assert snapshot == ordinary_snapshot == {
        "items": [{"count": 1}],
        "future_state": "preserved",
    }
    assert object.__getattribute__(value, "__pydantic_extra__") == {}
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ExampleContractListEnvelope.model_validate(snapshot, strict=True)


def test_snapshot_contract_graph_freezes_extras_before_nested_addition() -> None:
    ordinary = ExampleContractListEnvelope(items=[ExampleContract(count=1)])
    ordinary_snapshot = snapshot_contract_graph(ordinary)

    parent: list[ExampleContractListEnvelope | None] = [None]

    class AddingList(list[ExampleContract]):
        def __iter__(self) -> Iterator[ExampleContract]:
            if parent[0] is not None:
                extras = object.__getattribute__(
                    parent[0],
                    "__pydantic_extra__",
                )
                dict.__setitem__(extras, "late_state", "not_in_snapshot")
            return super().__iter__()

    value = ExampleContractListEnvelope(items=[ExampleContract(count=1)])
    object.__setattr__(value, "items", AddingList([ExampleContract(count=1)]))
    object.__setattr__(value, "__pydantic_extra__", {})
    parent[0] = value

    snapshot = snapshot_contract_graph(value)

    assert snapshot == ordinary_snapshot == {"items": [{"count": 1}]}
    assert object.__getattribute__(value, "__pydantic_extra__") == {
        "late_state": "not_in_snapshot"
    }
    assert (
        ExampleContractListEnvelope.model_validate(snapshot, strict=True).items[0].count
        == 1
    )


def test_snapshot_contract_graph_preserves_nested_container_shapes() -> None:
    nested = NestedExampleContract(identifier="item-1", count=2)
    value = {
        "models": [nested],
        "tuple": ({"nested": nested},),
    }

    snapshot = snapshot_contract_graph(value)

    assert snapshot == {
        "models": [{"identifier": "item-1", "count": 2}],
        "tuple": (({"nested": {"identifier": "item-1", "count": 2}},)),
    }
    assert isinstance(snapshot, dict)
    assert isinstance(snapshot["models"], list)
    assert isinstance(snapshot["tuple"], tuple)
    assert isinstance(snapshot["tuple"][0], dict)


@pytest.mark.parametrize("extra", [None, MappingProxyType({})])
def test_snapshot_contract_graph_accepts_normal_empty_extra_storage(
    extra: object,
) -> None:
    value = ExampleContract(count=1)
    object.__setattr__(value, "__pydantic_extra__", extra)

    assert snapshot_contract_graph(value) == {"count": 1}


@pytest.mark.parametrize("extra", [[], ["unreviewed"]], ids=["falsey", "truthy"])
def test_snapshot_contract_graph_rejects_malformed_extra_storage(
    extra: object,
) -> None:
    value = ExampleContract(count=1)
    object.__setattr__(value, "__pydantic_extra__", extra)

    with pytest.raises(ValueError, match="extra storage must be a mapping"):
        snapshot_contract_graph(value)


def test_snapshot_contract_graph_rejects_moved_required_field() -> None:
    value = ExampleContract(count=1)
    value.__dict__.pop("count")
    object.__setattr__(value, "__pydantic_extra__", {"count": 1})

    with pytest.raises(ValueError, match="conflicting declared and extra fields"):
        snapshot_contract_graph(value)


def test_snapshot_contract_graph_preserves_normal_missing_required_behavior() -> None:
    value = ExampleContract(count=1)
    value.__dict__.pop("count")

    snapshot = snapshot_contract_graph(value)

    assert snapshot == {}
    with pytest.raises(ValidationError, match="Field required"):
        ExampleContract.model_validate(snapshot, strict=True)


def test_snapshot_contract_graph_preserves_normal_missing_default_behavior() -> None:
    value = DefaultedExampleContract()
    value.__dict__.pop("count")

    snapshot = snapshot_contract_graph(value)

    assert snapshot == {}
    assert DefaultedExampleContract.model_validate(snapshot, strict=True).count == 7


def test_snapshot_contract_graph_rejects_default_field_moved_to_extras() -> None:
    value = DefaultedExampleContract()
    value.__dict__.pop("count")
    object.__setattr__(value, "__pydantic_extra__", {"count": 9})

    with pytest.raises(ValueError, match="conflicting declared and extra fields"):
        snapshot_contract_graph(value)


def test_snapshot_contract_graph_rejects_subclass_field_moved_to_extras() -> None:
    value = DerivedNestedExampleContract(
        identifier="item-1",
        count=1,
        category="derived",
    )
    value.__dict__.pop("category")
    object.__setattr__(value, "__pydantic_extra__", {"category": "derived"})

    with pytest.raises(ValueError, match="conflicting declared and extra fields"):
        snapshot_contract_graph(value)


@pytest.mark.parametrize("duplicate", [1, 2], ids=["equal", "conflicting"])
def test_snapshot_contract_graph_rejects_declared_extra_overlap(
    duplicate: int,
) -> None:
    value = ExampleContract(count=1)
    object.__setattr__(value, "__pydantic_extra__", {"count": duplicate})

    with pytest.raises(ValueError, match="conflicting declared and extra fields"):
        snapshot_contract_graph(value)


def test_snapshot_contract_graph_reads_hidden_declared_extra_collision() -> None:
    value = ExampleContract(count=1)
    value.__dict__.pop("count")
    extra = EmptyItemsDict(count=9)
    object.__setattr__(value, "__pydantic_extra__", extra)

    assert list(extra.items()) == []
    assert list(dict.items(extra)) == [("count", 9)]
    with pytest.raises(ValueError, match="conflicting declared and extra fields"):
        snapshot_contract_graph(value)


def test_snapshot_contract_graph_rejects_duplicate_unknown_stored_name() -> None:
    value = ExampleContract(count=1).model_copy(update={"future_state": "stored"})
    object.__setattr__(value, "__pydantic_extra__", {"future_state": "extra"})

    with pytest.raises(ValueError, match="conflicting declared and extra fields"):
        snapshot_contract_graph(value)


@pytest.mark.parametrize(
    "extra",
    [
        {"future_state": "preserved"},
        MappingProxyType({"future_state": "preserved"}),
    ],
    ids=["dict", "mapping-proxy"],
)
def test_snapshot_contract_graph_preserves_nonconflicting_unknown_extra(
    extra: object,
) -> None:
    value = ExampleContract(count=1)
    object.__setattr__(value, "__pydantic_extra__", extra)

    snapshot = snapshot_contract_graph(value)

    assert snapshot == {"count": 1, "future_state": "preserved"}
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ExampleContract.model_validate(snapshot, strict=True)


@pytest.mark.parametrize(
    ("value", "field_name", "extra"),
    [
        (AliasedExampleContract(n=1), "count", {"n": 9}),
        (ChoiceAliasedExampleContract(n=1), "count", {"number": 9}),
        (
            ChoiceAliasedExampleContract(n=1),
            "count",
            {"payload": {"count": 9}},
        ),
        (
            PathAliasedExampleContract(payload={"count": 1}),
            "count",
            {"payload": {"count": 9}},
        ),
    ],
    ids=["alias", "choice-string", "choice-path", "path"],
)
def test_snapshot_contract_graph_rejects_declared_field_moved_to_validation_alias(
    value: ContractModel,
    field_name: str,
    extra: dict[str, object],
) -> None:
    value.__dict__.pop(field_name)
    object.__setattr__(value, "__pydantic_extra__", extra)

    assert type(value).model_validate(
        {**vars(value), **extra},
        strict=True,
    )
    with pytest.raises(ValueError, match="conflicting declared and extra fields"):
        snapshot_contract_graph(value)


@pytest.mark.parametrize(
    ("field_name", "extra"),
    [
        ("count", {"n": 9}),
        ("category", {"kind": "shadow"}),
        ("category", {"metadata": {"category": "shadow"}}),
    ],
    ids=["inherited", "subclass-choice", "subclass-path"],
)
def test_snapshot_contract_graph_reserves_concrete_model_aliases(
    field_name: str,
    extra: dict[str, object],
) -> None:
    value = DerivedAliasedExampleContract(n=1, kind="derived")
    value.__dict__.pop(field_name)
    object.__setattr__(value, "__pydantic_extra__", extra)

    assert type(value).model_validate(
        {**vars(value), **extra},
        strict=True,
    )
    with pytest.raises(ValueError, match="conflicting declared and extra fields"):
        snapshot_contract_graph(value)


def test_snapshot_contract_graph_preserves_stored_field_name_for_alias() -> None:
    value = AliasedExampleContract(n=1)

    snapshot = snapshot_contract_graph(value)

    assert snapshot == {"count": 1}
    with pytest.raises(ValidationError, match="Field required"):
        AliasedExampleContract.model_validate(snapshot, strict=True)


def test_snapshot_contract_graph_does_not_reserve_serialization_only_alias() -> None:
    value = SerializationAliasedExampleContract(count=1)
    value.__dict__.pop("count")
    object.__setattr__(value, "__pydantic_extra__", {"n": 9})

    snapshot = snapshot_contract_graph(value)

    assert snapshot == {"n": 9}
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        SerializationAliasedExampleContract.model_validate(snapshot, strict=True)


def test_snapshot_contract_graph_does_not_reserve_disabled_validation_alias() -> None:
    value = NameOnlyAliasedExampleContract(count=1)
    value.__dict__.pop("count")
    object.__setattr__(value, "__pydantic_extra__", {"n": 9})

    snapshot = snapshot_contract_graph(value)

    assert snapshot == {"n": 9}
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        NameOnlyAliasedExampleContract.model_validate(snapshot, strict=True)


def test_snapshot_contract_graph_preserves_name_only_field_input() -> None:
    value = NameOnlyAliasedExampleContract(count=1)

    snapshot = snapshot_contract_graph(value)

    assert snapshot == {"count": 1}
    assert NameOnlyAliasedExampleContract.model_validate(
        snapshot,
        strict=True,
    ) == NameOnlyAliasedExampleContract(count=1)


@pytest.mark.parametrize(
    ("value", "field_name", "stored_name", "stored_value"),
    [
        (AliasedExampleContract(n=1), "count", "n", 1),
        (AliasedExampleContract(n=1), "count", "n", 9),
        (ChoiceAliasedExampleContract(n=1), "count", "number", 9),
        (
            ChoiceAliasedExampleContract(n=1),
            "count",
            "payload",
            {"count": 9},
        ),
        (
            PathAliasedExampleContract(payload={"count": 1}),
            "count",
            "payload",
            {"count": 9},
        ),
    ],
    ids=[
        "alias-equal",
        "alias-conflicting",
        "choice-string",
        "choice-path",
        "path",
    ],
)
def test_snapshot_contract_graph_rejects_unknown_stored_validation_alias(
    value: ContractModel,
    field_name: str,
    stored_name: str,
    stored_value: object,
) -> None:
    corrupted = value.model_copy(update={stored_name: stored_value})
    corrupted.__dict__.pop(field_name)
    stored_snapshot = dict(vars(corrupted))

    assert corrupted.__pydantic_extra__ is None
    assert type(value).model_validate(stored_snapshot, strict=True)
    with pytest.raises(ValueError, match="conflicting declared and extra fields"):
        snapshot_contract_graph(corrupted)


def test_snapshot_contract_graph_rejects_nested_unknown_stored_alias() -> None:
    value = AliasedExampleContract(n=1).model_copy(update={"n": 9})
    value.__dict__.pop("count")
    stored_snapshot = {"item": dict(vars(value))}

    assert (
        AliasedExampleEnvelope.model_validate(
            stored_snapshot,
            strict=True,
        ).item.count
        == 9
    )
    with pytest.raises(ValueError, match="conflicting declared and extra fields"):
        snapshot_contract_graph({"item": value})


def test_snapshot_contract_graph_reserves_complete_stored_alias_path_root() -> None:
    value = PathAliasedExampleContract(payload={"count": 1}).model_copy(
        update={"payload": {"other": 9}}
    )
    value.__dict__.pop("count")

    with pytest.raises(ValidationError):
        PathAliasedExampleContract.model_validate(dict(vars(value)), strict=True)
    with pytest.raises(ValueError, match="conflicting declared and extra fields"):
        snapshot_contract_graph(value)


@pytest.mark.parametrize(
    "model_type",
    [NameOnlyAliasedExampleContract, SerializationAliasedExampleContract],
    ids=["disabled-validation-alias", "serialization-only-alias"],
)
def test_snapshot_contract_graph_preserves_inactive_unknown_stored_alias(
    model_type: type[ContractModel],
) -> None:
    value = model_type(count=1).model_copy(update={"n": 9})
    value.__dict__.pop("count")

    snapshot = snapshot_contract_graph(value)

    assert snapshot == {"n": 9}
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        model_type.model_validate(snapshot, strict=True)


@pytest.mark.parametrize(
    ("field_name", "stored_name", "stored_value"),
    [
        ("count", "n", 9),
        ("category", "kind", "shadow"),
        ("category", "metadata", {"category": "shadow"}),
    ],
    ids=["inherited", "subclass-choice", "subclass-path"],
)
def test_snapshot_contract_graph_rejects_concrete_model_stored_aliases(
    field_name: str,
    stored_name: str,
    stored_value: object,
) -> None:
    value = DerivedAliasedExampleContract(n=1, kind="derived").model_copy(
        update={stored_name: stored_value}
    )
    value.__dict__.pop(field_name)

    assert DerivedAliasedExampleContract.model_validate(
        dict(vars(value)),
        strict=True,
    )
    with pytest.raises(ValueError, match="conflicting declared and extra fields"):
        snapshot_contract_graph(value)


def test_snapshot_contract_graph_preserves_declared_active_alias_root() -> None:
    value = InheritedAliasExampleContract(n=3)

    snapshot = snapshot_contract_graph(value)

    assert snapshot == {"count": 3}
    assert InheritedAliasExampleContract.model_validate(
        snapshot,
        strict=True,
    ) == value


@pytest.mark.parametrize("replacement", [1, 9], ids=["equal", "conflicting"])
@pytest.mark.parametrize(
    ("value", "missing_field", "root_name", "receiver"),
    [
        (
            CrossStringAliasContract(shadow=1),
            "primary",
            "shadow",
            CrossStringAliasContract,
        ),
        (
            CrossPathAliasContract(payload={"count": 1}),
            "count",
            "payload",
            CrossPathAliasContract,
        ),
        (
            CrossChoiceAliasContract(primary=1, shadow=1),
            "primary",
            "shadow",
            CrossChoiceAliasContract,
        ),
        (
            CrossChoicePathAliasContract(count=1, payload={"count": 1}),
            "count",
            "payload",
            CrossChoicePathAliasContract,
        ),
    ],
    ids=["string", "path", "choice-string", "choice-path"],
)
def test_snapshot_contract_graph_rejects_cross_field_alias_root(
    value: ContractModel,
    missing_field: str,
    root_name: str,
    receiver: type[ContractModel],
    replacement: int,
) -> None:
    root_value: object = (
        {"count": replacement} if root_name == "payload" else replacement
    )
    corrupted = value.model_copy(update={root_name: root_value})
    corrupted.__dict__.pop(missing_field)
    stored_snapshot = dict(vars(corrupted))

    assert receiver.model_validate(stored_snapshot, strict=True)
    with pytest.raises(ValueError) as raised:
        snapshot_contract_graph(corrupted)
    assert str(raised.value) == (
        "contract input contains stored validation alias roots "
        "with missing owning fields"
    )


@pytest.mark.parametrize("replacement", [1, 9], ids=["equal", "conflicting"])
@pytest.mark.parametrize(
    ("value", "missing_field", "root_name", "receiver"),
    [
        (
            CrossStringAliasContract(shadow=1),
            "primary",
            "shadow",
            CrossStringAliasEnvelope,
        ),
        (
            CrossPathAliasContract(payload={"count": 1}),
            "count",
            "payload",
            CrossPathAliasEnvelope,
        ),
        (
            CrossChoiceAliasContract(primary=1, shadow=1),
            "primary",
            "shadow",
            CrossChoiceAliasEnvelope,
        ),
        (
            CrossChoicePathAliasContract(count=1, payload={"count": 1}),
            "count",
            "payload",
            CrossChoicePathAliasEnvelope,
        ),
    ],
    ids=["string", "path", "choice-string", "choice-path"],
)
def test_snapshot_contract_graph_rejects_nested_cross_field_alias_root(
    value: ContractModel,
    missing_field: str,
    root_name: str,
    receiver: type[ContractModel],
    replacement: int,
) -> None:
    root_value: object = (
        {"count": replacement} if root_name == "payload" else replacement
    )
    corrupted = value.model_copy(update={root_name: root_value})
    corrupted.__dict__.pop(missing_field)
    stored_snapshot = {"item": dict(vars(corrupted))}

    assert receiver.model_validate(stored_snapshot, strict=True)
    with pytest.raises(ValueError) as raised:
        snapshot_contract_graph({"item": corrupted})
    assert str(raised.value) == (
        "contract input contains stored validation alias roots "
        "with missing owning fields"
    )


@pytest.mark.parametrize(
    ("missing_field", "root_name", "root_value"),
    [
        ("primary", "shadow", 9),
        ("category", "kind", "shadow-kind"),
    ],
    ids=["inherited", "concrete-subclass"],
)
def test_snapshot_contract_graph_rejects_inherited_cross_field_alias_root(
    missing_field: str,
    root_name: str,
    root_value: object,
) -> None:
    value = CrossAliasDerivedContract(
        primary=1,
        shadow=1,
        category="original-kind",
        kind="original-kind",
    ).model_copy(update={root_name: root_value})
    value.__dict__.pop(missing_field)

    assert CrossAliasDerivedContract.model_validate(dict(vars(value)), strict=True)
    with pytest.raises(ValueError) as raised:
        snapshot_contract_graph(value)
    assert str(raised.value) == (
        "contract input contains stored validation alias roots "
        "with missing owning fields"
    )


def test_snapshot_contract_graph_preserves_existing_alias_conflict_messages() -> None:
    unknown_stored = AliasedExampleContract(n=1).model_copy(update={"n": 9})
    unknown_stored.__dict__.pop("count")
    extra_conflict = ExampleContract(count=1)
    object.__setattr__(extra_conflict, "__pydantic_extra__", {"count": 2})

    for value in (unknown_stored, extra_conflict):
        with pytest.raises(ValueError) as raised:
            snapshot_contract_graph(value)
        assert str(raised.value) == (
            "contract input contains conflicting declared and extra fields"
        )


@pytest.mark.parametrize(
    "value",
    [
        CrossChoiceAliasContract(primary=1, shadow=9),
        CrossChoicePathAliasContract(count=1, payload={"count": 9}),
        CrossAliasDerivedContract(
            primary=1,
            shadow=9,
            category="category-value",
            kind="kind-value",
        ),
    ],
    ids=["choice-string", "choice-path", "inherited-subclass"],
)
def test_snapshot_contract_graph_preserves_populated_cross_field_models(
    value: ContractModel,
) -> None:
    snapshot = snapshot_contract_graph(value)

    assert type(value).model_validate(snapshot, strict=True) == value


def test_snapshot_contract_graph_preserves_alias_only_populated_cross_field_model() -> None:
    value = CrossStringAliasContract(shadow=3)

    assert snapshot_contract_graph(value) == {
        "primary": 3,
        "shadow": 3,
    }


def test_snapshot_contract_graph_restores_absent_cross_field_defaults() -> None:
    value = CrossDefaultAliasContract()
    value.__dict__.pop("primary")
    value.__dict__.pop("shadow")

    snapshot = snapshot_contract_graph(value)

    assert snapshot == {}
    assert CrossDefaultAliasContract.model_validate(snapshot, strict=True) == (
        CrossDefaultAliasContract()
    )


@pytest.mark.parametrize("nested", [False, True], ids=["direct", "nested"])
def test_snapshot_contract_graph_rejects_key_hiding_alias_storage(
    nested: bool,
) -> None:
    value = AliasedExampleContract(n=1)
    value.__dict__.pop("count")
    extra = HiddenKeysExtra({"n": 9})
    object.__setattr__(value, "__pydantic_extra__", extra)
    graph: object = {"item": value} if nested else value

    assert list(extra.keys()) == []
    assert list(extra.items()) == [("n", 9)]
    with pytest.raises(ValueError, match="conflicting declared and extra fields"):
        snapshot_contract_graph(graph)


def test_snapshot_contract_graph_rejects_key_hiding_declared_name() -> None:
    value = ExampleContract(count=1)
    value.__dict__.pop("count")
    object.__setattr__(value, "__pydantic_extra__", HiddenKeysExtra({"count": 9}))

    with pytest.raises(ValueError, match="conflicting declared and extra fields"):
        snapshot_contract_graph(value)


@pytest.mark.parametrize("nested", [False, True], ids=["direct", "nested"])
def test_snapshot_contract_graph_reads_hidden_extra_backing_entries(
    nested: bool,
) -> None:
    value = ExampleContract(count=1)
    extra = HiddenItemsDict(future_state="preserved")
    object.__setattr__(value, "__pydantic_extra__", extra)
    graph: object = {"item": value} if nested else value
    receiver = ExampleContractEnvelope if nested else ExampleContract

    assert list(extra.items()) == []
    snapshot = snapshot_contract_graph(graph)

    expected = (
        {"item": {"count": 1, "future_state": "preserved"}}
        if nested
        else {"count": 1, "future_state": "preserved"}
    )
    assert snapshot == expected
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        receiver.model_validate(snapshot, strict=True)


def test_snapshot_contract_graph_bypasses_extra_dict_inventory_overrides() -> None:
    value = ExampleContract(count=1)
    extra = SingleReadExtra({"future_state": "preserved"})
    object.__setattr__(value, "__pydantic_extra__", extra)

    assert snapshot_contract_graph(value) == {
        "count": 1,
        "future_state": "preserved",
    }
    assert extra.item_reads == 0


@pytest.mark.parametrize(
    "extra",
    [
        RaisingInventoryDict(future_state="preserved"),
        FabricatingItemsDict(future_state="preserved"),
    ],
    ids=["raising-overrides", "fabricated-entry"],
)
def test_snapshot_contract_graph_uses_only_extra_dict_backing_entries(
    extra: dict[str, object],
) -> None:
    value = ExampleContract(count=1)
    object.__setattr__(value, "__pydantic_extra__", extra)

    snapshot = snapshot_contract_graph(value)

    assert snapshot == {"count": 1, "future_state": "preserved"}
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ExampleContract.model_validate(snapshot, strict=True)


def test_snapshot_contract_graph_captures_generic_mapping_extra_once() -> None:
    value = ExampleContract(count=1)
    extra = SingleReadMapping({"future_state": "preserved"})
    object.__setattr__(value, "__pydantic_extra__", extra)

    snapshot = snapshot_contract_graph(value)

    assert snapshot == {"count": 1, "future_state": "preserved"}
    assert extra.item_reads == 1
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ExampleContract.model_validate(snapshot, strict=True)


@pytest.mark.parametrize("nested", [False, True], ids=["direct", "nested"])
@pytest.mark.parametrize("second_value", [1, 2], ids=["equal", "different"])
def test_snapshot_contract_graph_rejects_duplicate_custom_extra_keys(
    nested: bool,
    second_value: int,
) -> None:
    value = ExampleContract(count=1)
    extra = DuplicateItemsMapping(
        (("future_state", 1), ("future_state", second_value))
    )
    object.__setattr__(value, "__pydantic_extra__", extra)
    graph: object = {"item": value} if nested else value

    with pytest.raises(ValueError) as raised:
        snapshot_contract_graph(graph)

    assert str(raised.value) == "contract extra storage contains duplicate keys"
    assert extra.item_reads == 1


def test_snapshot_contract_graph_preserves_unique_custom_extra_inventory() -> None:
    value = ExampleContract(count=1)
    extra = DuplicateItemsMapping((("future_state", "preserved"),))
    object.__setattr__(value, "__pydantic_extra__", extra)

    snapshot = snapshot_contract_graph(value)

    assert snapshot == {"count": 1, "future_state": "preserved"}
    assert extra.item_reads == 1
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ExampleContract.model_validate(snapshot, strict=True)


class ChangingIterationEntry(tuple):
    """A key/value entry whose iterator reports a different pair on every pass after the first."""

    passes = 0

    def __iter__(self) -> Iterator[object]:
        type(self).passes += 1
        if type(self).passes == 1:
            return iter(("future_state", 1))
        return iter(("count", 1))


def test_snapshot_contract_graph_freezes_each_custom_extra_entry_once() -> None:
    ChangingIterationEntry.passes = 0
    value = ExampleContract.model_construct(count=-1)  # stored state the validator would refuse
    extra = DuplicateItemsMapping((ChangingIterationEntry(("future_state", 1)),))
    object.__setattr__(value, "__pydantic_extra__", extra)

    snapshot = snapshot_contract_graph(value)

    # one pass over the entry: capture, collision checks and reconstruction all see the same
    # pair, so the stored declared value is never overwritten by a later, different pass
    assert snapshot == {"count": -1, "future_state": 1}
    assert ChangingIterationEntry.passes == 1


def test_snapshot_contract_graph_preserves_collision_priority_for_duplicate_extra() -> None:
    value = ExampleContract(count=1)
    extra = DuplicateItemsMapping((("count", 1), ("count", 2)))
    object.__setattr__(value, "__pydantic_extra__", extra)

    with pytest.raises(ValueError) as raised:
        snapshot_contract_graph(value)

    assert str(raised.value) == (
        "contract input contains conflicting declared and extra fields"
    )
    assert extra.item_reads == 1


@pytest.mark.parametrize("nested", [False, True], ids=["direct", "nested"])
def test_snapshot_contract_graph_uses_spoofed_class_mapping_protocol(
    nested: bool,
) -> None:
    value = ExampleContract(count=1)
    extra = DictClassSpoofingMapping({"future_state": "preserved"})
    object.__setattr__(value, "__pydantic_extra__", extra)
    graph: object = {"item": value} if nested else value
    receiver = ExampleContractEnvelope if nested else ExampleContract

    assert isinstance(extra, dict)
    assert type(extra) is DictClassSpoofingMapping
    snapshot = snapshot_contract_graph(graph)

    expected = (
        {"item": {"count": 1, "future_state": "preserved"}}
        if nested
        else {"count": 1, "future_state": "preserved"}
    )
    assert snapshot == expected
    assert extra.item_reads == 1
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        receiver.model_validate(snapshot, strict=True)


def test_snapshot_contract_graph_does_not_read_mapping_class_override() -> None:
    value = ExampleContract(count=1)
    extra = ClassReadRejectingMapping({"future_state": "preserved"})
    object.__setattr__(value, "__pydantic_extra__", extra)

    snapshot = snapshot_contract_graph(value)

    assert snapshot == {"count": 1, "future_state": "preserved"}
    assert extra.item_reads == 1


@pytest.mark.parametrize(
    "mapping_type",
    [ClassReadRejectingMapping, ClassReadCountingMapping],
    ids=["raising-class", "counted-class"],
)
@pytest.mark.parametrize("location", ["direct", "list", "dict"])
def test_snapshot_contract_graph_leaves_adversarial_non_dict_mapping_unchanged(
    mapping_type: type[SingleReadMapping],
    location: str,
) -> None:
    value = mapping_type({"future_state": "preserved"})
    graph: object
    if location == "direct":
        graph = value
    elif location == "list":
        graph = [value]
    else:
        graph = {"item": value}

    snapshot = snapshot_contract_graph(graph)

    retained = (
        snapshot
        if location == "direct"
        else snapshot[0]
        if location == "list"
        else snapshot["item"]
    )
    assert retained is value
    assert value.item_reads == 0
    if type(value) is ClassReadCountingMapping:
        assert value.class_reads == 0


@pytest.mark.parametrize("nested", [False, True], ids=["direct", "nested"])
def test_snapshot_contract_graph_dispatches_model_spoofing_dict_as_dict(
    nested: bool,
) -> None:
    value = ModelClassSpoofingDict(count=1)
    graph: object = {"item": value} if nested else value

    assert isinstance(value, ContractModel)
    snapshot = snapshot_contract_graph(graph)

    expected = {"item": {"count": 1}} if nested else {"count": 1}
    assert snapshot == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (ClassReadRejectingContract(count=1), {"count": 1}),
        (ClassReadRejectingTuple((ExampleContract(count=1),)), ({"count": 1},)),
        (ClassReadRejectingList([ExampleContract(count=1)]), [{"count": 1}]),
        (
            ClassReadRejectingDict(item=ExampleContract(count=1)),
            {"item": {"count": 1}},
        ),
    ],
    ids=["model-subclass", "tuple-subclass", "list-subclass", "dict-subclass"],
)
def test_snapshot_contract_graph_dispatches_concrete_container_subclasses(
    value: object,
    expected: object,
) -> None:
    assert snapshot_contract_graph(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (HiddenItemsDict(visible=1, hidden=2), {"visible": 1, "hidden": 2}),
        (RaisingInventoryDict(visible=1, hidden=2), {"visible": 1, "hidden": 2}),
        (FabricatingItemsDict(visible=1), {"visible": 1}),
    ],
    ids=["hidden-entry", "raising-overrides", "fabricated-entry"],
)
def test_snapshot_contract_graph_reads_nested_dict_backing_entries(
    value: dict[str, object],
    expected: dict[str, int],
) -> None:
    graph = {"payload": value}

    snapshot = snapshot_contract_graph(graph)

    assert snapshot == {"payload": expected}
    assert DictionaryEnvelope.model_validate(snapshot, strict=True).model_dump() == {
        "payload": expected
    }


def test_snapshot_contract_graph_rejects_dict_subclass_backing_cycle() -> None:
    value = FabricatingItemsDict()
    dict.__setitem__(value, "self", value)

    with pytest.raises(ValueError, match="cyclic contract input"):
        snapshot_contract_graph(value)


@pytest.mark.parametrize(
    "extra_kind",
    ["malformed", "overlap", "duplicate"],
)
def test_snapshot_contract_graph_preserves_stored_cycle_priority(
    extra_kind: str,
) -> None:
    value = ExampleContract(count=1)
    object.__getattribute__(value, "__dict__")["count"] = value
    duplicate: DuplicateItemsMapping | None = None
    if extra_kind == "malformed":
        extra: object = []
    elif extra_kind == "overlap":
        extra = {"count": 1}
    else:
        duplicate = DuplicateItemsMapping(
            (("future_state", 1), ("future_state", 2))
        )
        extra = duplicate
    object.__setattr__(value, "__pydantic_extra__", extra)

    with pytest.raises(ValueError) as raised:
        snapshot_contract_graph(value)

    assert str(raised.value) == "cyclic contract input is not supported"
    if duplicate is not None:
        assert duplicate.item_reads == 1


def test_snapshot_contract_graph_preserves_alias_guard_priority() -> None:
    value = AliasedExampleContract(n=1)
    stored = object.__getattribute__(value, "__dict__")
    stored.pop("count")
    stored["n"] = 9
    object.__setattr__(value, "__pydantic_extra__", [])

    with pytest.raises(ValueError) as raised:
        snapshot_contract_graph(value)

    assert str(raised.value) == (
        "contract input contains conflicting declared and extra fields"
    )


@pytest.mark.parametrize("container_type", [list, dict])
def test_snapshot_contract_graph_rejects_cycles(container_type: type) -> None:
    value: list[object] | dict[str, object] = container_type()
    if isinstance(value, list):
        value.append(value)
    else:
        value["self"] = value

    with pytest.raises(ValueError, match="cyclic contract input"):
        snapshot_contract_graph(value)


def test_snapshot_contract_graph_allows_repeated_noncyclic_objects() -> None:
    shared = [ExampleContract(count=1)]

    assert snapshot_contract_graph([shared, shared]) == [
        [{"count": 1}],
        [{"count": 1}],
    ]


def test_snapshot_contract_graph_leaves_non_dict_mappings_unchanged() -> None:
    value = MappingProxyType({"model": ExampleContract(count=1)})

    assert snapshot_contract_graph(value) is value


def test_contract_model_is_strict_and_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="valid integer"):
        ExampleContract(count="1")

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ExampleContract(count=1, authority="admin")


def test_nested_contract_instance_revalidates_invalid_known_field_in_mapping() -> None:
    valid = NestedExampleContract(identifier="value-1", count=1)
    unchecked = valid.model_copy(update={"count": True})

    assert unchecked.count is True
    with pytest.raises(ValidationError, match="valid integer"):
        ExampleContractGraph.model_validate(
            {
                "primary": valid,
                "by_name": {"unchecked": unchecked},
            }
        )


def test_nested_contract_instance_rejects_retained_unknown_field_in_mapping() -> None:
    valid = NestedExampleContract(identifier="value-1", count=1)
    unchecked = valid.model_copy(update={"future_constraint": "deny"})

    assert unchecked.__dict__["future_constraint"] == "deny"
    assert "future_constraint" in unchecked.model_fields_set
    assert "future_constraint" not in unchecked.model_dump()
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ExampleContractGraph.model_validate(
            {
                "primary": valid,
                "by_name": {"unchecked": unchecked},
            }
        )


def test_inherited_contract_revalidation_applies_at_its_declared_boundary() -> None:
    valid = DerivedNestedExampleContract(
        identifier="value-1",
        count=1,
        category="derived",
    )
    unchecked = valid.model_copy(update={"count": True})

    assert DerivedNestedExampleContract.model_config["extra"] == "forbid"
    assert DerivedNestedExampleContract.model_config["strict"] is True
    assert (
        DerivedNestedExampleContract.model_config["revalidate_instances"] == "always"
    )
    with pytest.raises(ValidationError, match="valid integer"):
        DerivedExampleContractGraph.model_validate({"item": unchecked})

    revalidated = DerivedExampleContractGraph.model_validate({"item": valid})
    assert revalidated.item == valid
    assert revalidated.item is not valid


def test_base_typed_boundary_rejects_subclass_fields_instead_of_dropping() -> None:
    derived = DerivedNestedExampleContract(
        identifier="value-1",
        count=1,
        category="derived",
    )

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ExampleContractGraph.model_validate(
            {
                "primary": derived,
                "by_name": {},
            }
        )


def test_valid_nested_instances_preserve_values_but_are_rebuilt_per_edge() -> None:
    nested = NestedExampleContract(identifier=" Value/01 ", count=1)

    graph = ExampleContractGraph.model_validate(
        {
            "primary": nested,
            "by_name": {"alias": nested},
        }
    )

    assert graph.primary == nested
    assert graph.by_name["alias"] == nested
    assert graph.primary is not nested
    assert graph.by_name["alias"] is not nested
    assert graph.primary is not graph.by_name["alias"]
    assert graph.primary.identifier == " Value/01 "


def test_version_is_required_and_only_candidate_v1_is_supported() -> None:
    assert VersionedContract(schema_version="candidate-v1").model_dump() == {
        "schema_version": "candidate-v1"
    }

    with pytest.raises(ValidationError, match="Field required"):
        VersionedContract()
    with pytest.raises(ValidationError, match="candidate-v1"):
        VersionedContract(schema_version="v1")


@pytest.mark.parametrize("value", ["", "x" * 257, 1, b"id"])
def test_opaque_id_rejects_empty_oversized_and_coerced_values(value: object) -> None:
    with pytest.raises(ValidationError):
        TypeAdapter(OpaqueId).validate_python(value)


def test_opaque_id_preserves_whitespace_and_case() -> None:
    value = " Tenant/A "
    assert TypeAdapter(OpaqueId).validate_python(value) == value


@pytest.mark.parametrize(
    "value",
    [-1, MAX_SAFE_INTEGER + 1, True, False, 1.0, 1.5, "1"],
)
def test_safe_counter_rejects_out_of_range_and_non_integer_values(value: object) -> None:
    with pytest.raises(ValidationError):
        TypeAdapter(SafeCounter).validate_python(value)


def test_safe_counter_accepts_both_exact_boundaries() -> None:
    adapter = TypeAdapter(SafeCounter)
    assert adapter.validate_python(0) == 0
    assert adapter.validate_python(MAX_SAFE_INTEGER) == MAX_SAFE_INTEGER


def test_reference_handle_supports_only_declared_kinds_and_required_scope() -> None:
    expected_kinds = {
        "memory",
        "source",
        "passage",
        "graph_node",
        "graph_edge",
        "episode",
        "artifact",
        "intent",
    }
    assert set(get_args(ReferenceKind)) == expected_kinds

    for kind in expected_kinds:
        handle = ReferenceHandle(
            kind=kind,
            id="Target-01",
            tenant_id="Tenant-A",
            resolver="exact_lookup",
        )
        assert handle.id == "Target-01"

    with pytest.raises(ValidationError):
        ReferenceHandle(
            kind="user",
            id="Target-01",
            tenant_id="Tenant-A",
            resolver="exact_lookup",
        )
    with pytest.raises(ValidationError, match="Field required"):
        ReferenceHandle(kind="memory", id="Target-01", resolver="exact_lookup")


def test_reference_is_a_value_not_an_authority_or_existence_claim() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ReferenceHandle(
            kind="memory",
            id="missing-or-denied",
            tenant_id="Tenant-A",
            resolver="exact_lookup",
            exists=True,
            authorized=True,
        )


def test_source_version_keeps_content_and_policy_progress_independent() -> None:
    version = SourceVersion(
        record_id="source-A",
        content_revision=3,
        policy_epoch=9,
    )
    changed_content = SourceVersion(
        record_id=version.record_id,
        content_revision=4,
        policy_epoch=version.policy_epoch,
    )

    assert changed_content.content_revision == 4
    assert changed_content.policy_epoch == 9
    assert version.content_revision == 3

    with pytest.raises(ValidationError):
        SourceVersion(record_id="source-A", content_revision=True, policy_epoch=9)


@pytest.mark.parametrize(
    ("value", "field", "replacement"),
    [
        (
            ReferenceHandle(
                kind="memory",
                id="memory-1",
                tenant_id="tenant-1",
                resolver="exact_lookup",
            ),
            "id",
            "memory-2",
        ),
        (
            SourceVersion(
                record_id="source-1",
                content_revision=3,
                policy_epoch=9,
            ),
            "content_revision",
            4,
        ),
    ],
)
def test_reference_value_snapshots_reject_assignment(
    value: ReferenceHandle | SourceVersion,
    field: str,
    replacement: object,
) -> None:
    with pytest.raises(ValidationError, match="Instance is frozen"):
        setattr(value, field, replacement)


def test_reference_handle_revalidates_invalid_known_field_copy() -> None:
    handle = ReferenceHandle(
        kind="memory",
        id="memory-1",
        tenant_id="tenant-1",
        resolver="exact_lookup",
    )
    copied = handle.model_copy(update={"kind": "user"})

    assert copied.kind == "user"
    with pytest.raises(ValidationError):
        ReferenceHandle.model_validate(copied)


def test_source_version_revalidates_invalid_known_field_copy() -> None:
    version = SourceVersion(
        record_id="source-1",
        content_revision=3,
        policy_epoch=9,
    )
    copied = version.model_copy(update={"content_revision": True})

    assert copied.content_revision is True
    with pytest.raises(ValidationError):
        SourceVersion.model_validate(copied)


def test_reference_handle_revalidation_rejects_injected_unknown_field() -> None:
    handle = ReferenceHandle(
        kind="memory",
        id="memory-1",
        tenant_id="tenant-1",
        resolver="exact_lookup",
    )
    copied = handle.model_copy(update={"authority": "admin"})

    assert "authority" not in copied.model_dump()
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ReferenceHandle.model_validate(copied)


def test_source_version_revalidation_rejects_injected_unknown_field() -> None:
    version = SourceVersion(
        record_id="source-1",
        content_revision=3,
        policy_epoch=9,
    )
    copied = version.model_copy(update={"global_revision": 12})

    assert "global_revision" not in copied.model_dump()
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        SourceVersion.model_validate(copied)


def test_valid_reference_instances_are_revalidated_without_value_changes() -> None:
    handle = ReferenceHandle(
        kind="graph_edge",
        id=" Edge/01 ",
        tenant_id="Tenant-A",
        resolver="graph_resolve",
    )
    version = SourceVersion(
        record_id="source-A",
        content_revision=3,
        policy_epoch=9,
    )

    revalidated_handle = ReferenceHandle.model_validate(handle)
    revalidated_version = SourceVersion.model_validate(version)

    assert revalidated_handle == handle
    assert revalidated_handle is not handle
    assert revalidated_version == version
    assert revalidated_version is not version


def test_native_and_json_representations_preserve_values() -> None:
    handle = ReferenceHandle(
        kind="graph_edge",
        id=" Edge/01 ",
        tenant_id="Tenant-A",
        resolver="graph_resolve",
    )

    native = handle.model_dump()
    wire = json.loads(handle.model_dump_json())
    assert native == wire == {
        "kind": "graph_edge",
        "id": " Edge/01 ",
        "tenant_id": "Tenant-A",
        "resolver": "graph_resolve",
        "cross_tenant": False,  # additive marker, default False
    }
    assert ReferenceHandle.model_validate_json(handle.model_dump_json()) == handle

    version = SourceVersion(
        record_id=" Source/01 ",
        content_revision=3,
        policy_epoch=9,
    )
    assert SourceVersion.model_validate_json(version.model_dump_json()) == version


def test_generated_json_schemas_are_closed_and_express_bounds() -> None:
    reference_schema = ReferenceHandle.model_json_schema()
    source_schema = SourceVersion.model_json_schema()

    assert reference_schema["additionalProperties"] is False
    assert reference_schema["required"] == ["kind", "id", "tenant_id", "resolver"]  # cross_tenant is optional (default False)
    assert reference_schema["properties"]["cross_tenant"]["default"] is False
    assert set(reference_schema["properties"]["kind"]["enum"]) == set(
        get_args(ReferenceKind)
    )
    assert reference_schema["properties"]["id"]["minLength"] == 1
    assert reference_schema["properties"]["id"]["maxLength"] == 256
    assert source_schema["additionalProperties"] is False
    assert source_schema["required"] == [
        "record_id",
        "content_revision",
        "policy_epoch",
    ]
    assert source_schema["properties"]["content_revision"] == {
        "maximum": MAX_SAFE_INTEGER,
        "minimum": 0,
        "title": "Content Revision",
        "type": "integer",
    }


def test_safe_error_mapping_collapses_existence_sensitive_outcomes() -> None:
    denied = public_error_for(InternalErrorCode.DENIED)
    missing = public_error_for(InternalErrorCode.MISSING_RESOURCE)
    wrong_kind = public_error_for(InternalErrorCode.WRONG_REFERENCE_KIND)
    stale_revision = public_error_for(InternalErrorCode.STALE_REVISION)

    assert denied == missing == wrong_kind == stale_revision
    assert denied.code is PublicErrorCode.NOT_FOUND
    assert "denied" not in denied.message.lower()


def test_authorized_error_mapping_can_return_richer_diagnostics() -> None:
    denied = public_error_for(
        InternalErrorCode.DENIED,
        disclose_existence=True,
    )
    wrong_kind = public_error_for(
        InternalErrorCode.WRONG_REFERENCE_KIND,
        disclose_existence=True,
    )
    stale_revision = public_error_for(
        InternalErrorCode.STALE_REVISION,
        disclose_existence=True,
    )

    assert denied.code is PublicErrorCode.PERMISSION_DENIED
    assert wrong_kind.code is PublicErrorCode.WRONG_REFERENCE_KIND
    assert stale_revision.code is PublicErrorCode.STALE_REVISION
    assert stale_revision.message == "The supplied revision is stale."
    assert public_error_for(
        InternalErrorCode.MISSING_RESOURCE,
        disclose_existence=True,
    ).code is PublicErrorCode.NOT_FOUND


def test_false_disclosure_decision_keeps_existence_sensitive_errors_collapsed() -> None:
    missing = public_error_for(
        InternalErrorCode.MISSING_RESOURCE,
        disclose_existence=False,
    )
    for code in (
        InternalErrorCode.DENIED,
        InternalErrorCode.WRONG_REFERENCE_KIND,
        InternalErrorCode.STALE_REVISION,
    ):
        assert public_error_for(code, disclose_existence=False) == missing


def test_stale_revision_default_and_false_are_both_nondisclosing() -> None:
    missing = public_error_for(InternalErrorCode.MISSING_RESOURCE)

    assert public_error_for(InternalErrorCode.STALE_REVISION) == missing
    assert public_error_for(
        InternalErrorCode.STALE_REVISION,
        disclose_existence=False,
    ) == missing


def test_stale_revision_true_discloses_existing_diagnostic_only() -> None:
    stale = public_error_for(
        InternalErrorCode.STALE_REVISION,
        disclose_existence=True,
    )

    assert stale.code is PublicErrorCode.STALE_REVISION
    assert stale.message == "The supplied revision is stale."
    assert stale.retryable is False
    for disclose_existence in (False, True):
        assert public_error_for(
            InternalErrorCode.STALE_POLICY,
            disclose_existence=disclose_existence,
        ).code is PublicErrorCode.STALE_POLICY


@pytest.mark.parametrize("value", ["true", "false", 0, 1, None])
def test_disclosure_decision_rejects_non_boolean_values(value: object) -> None:
    with pytest.raises(TypeError) as caught:
        public_error_for(
            InternalErrorCode.DENIED,
            disclose_existence=value,
        )
    assert str(caught.value) == "disclose_existence must be a bool"


@pytest.mark.parametrize("code", list(InternalErrorCode))
def test_error_code_boundary_accepts_every_internal_enum_variant(
    code: InternalErrorCode,
) -> None:
    assert isinstance(public_error_for(code).code, PublicErrorCode)


@pytest.mark.parametrize("value", [code.value for code in InternalErrorCode])
def test_error_code_boundary_rejects_known_plain_strings(value: object) -> None:
    with pytest.raises(TypeError) as caught:
        public_error_for(value)  # type: ignore[arg-type]
    assert str(caught.value) == "code must be an InternalErrorCode"


@pytest.mark.parametrize(
    "value",
    [
        "unknown_internal_error",
        PublicErrorCode.UNAUTHENTICATED,
        [],
        {},
        None,
        True,
        False,
    ],
)
def test_error_code_boundary_rejects_other_runtime_types(value: object) -> None:
    with pytest.raises(TypeError) as caught:
        public_error_for(value)  # type: ignore[arg-type]
    assert str(caught.value) == "code must be an InternalErrorCode"


def test_error_code_boundary_does_not_hash_an_invalid_value() -> None:
    class HashTrap:
        def __hash__(self) -> int:
            raise AssertionError("invalid values must not reach dictionary lookup")

    with pytest.raises(TypeError) as caught:
        public_error_for(HashTrap())  # type: ignore[arg-type]
    assert str(caught.value) == "code must be an InternalErrorCode"


@pytest.mark.parametrize("raw", ["denied", "unknown_internal_error"])
@pytest.mark.parametrize("set_enum_internals", [False, True])
def test_error_code_boundary_rejects_forged_exact_type_instances(
    raw: str,
    set_enum_internals: bool,
) -> None:
    forged = str.__new__(InternalErrorCode, raw)
    if set_enum_internals:
        forged._name_ = "FORGED"
        forged._value_ = raw

    assert type(forged) is InternalErrorCode
    assert not any(forged is member for member in InternalErrorCode)
    with pytest.raises(TypeError) as caught:
        public_error_for(forged)
    assert str(caught.value) == "code must be an InternalErrorCode"


def test_error_mapping_covers_every_internal_outcome_and_returns_copies() -> None:
    mapped = {code: public_error_for(code) for code in InternalErrorCode}
    assert set(mapped) == set(InternalErrorCode)
    assert mapped[InternalErrorCode.UNAVAILABLE_DEPENDENCY].retryable is True
    assert mapped[InternalErrorCode.TERMINAL_FAILURE].code is PublicErrorCode.INTERNAL_ERROR

    first = public_error_for(InternalErrorCode.CANCELLED)
    first.message = "mutated by a caller"
    assert public_error_for(InternalErrorCode.CANCELLED).message == "The operation was cancelled."


# ---------------------------------------------------------------------------
# Public-view pinning (public_first): a child that exists only in a sequence
# subclass's public iterator is captured when that view is first observed.  An
# earlier callback can therefore repair it, and it is then exposed in repaired
# form.  The view is read once per reconstruction.  Models reachable through
# native storage are fenced before any callback and are NOT repaired.  The
# tests pin this documented limit; they do not claim entry-time capture of
# children that were never exposed.
# ---------------------------------------------------------------------------


class PublicViewTupleEnvelope(ContractModel):
    trigger: tuple[int, ...]
    children: tuple[ExampleContract, ...]
    groups: tuple[tuple[ExampleContract, ...], ...] = ()


class PublicViewListEnvelope(ContractModel):
    trigger: tuple[int, ...]
    children: list[ExampleContract]
    groups: list[list[ExampleContract]] = []


def _invalid_example_child() -> ExampleContract:
    return ExampleContract(count=1).model_copy(update={"count": -1})


def _repair_example_child(child: ExampleContract) -> Callable[[], None]:
    def repair() -> None:
        child_native = _BASE_MODEL_DICT_DESCRIPTOR.__get__(child, BaseModel)
        dict.__setitem__(child_native, "count", 1)

    return repair


def _public_only_envelope(
    container_kind: str,
    placement: str,
) -> tuple[BaseModel, ExampleContract, DivergentTuple | DivergentList]:
    """Return an envelope whose only child is exposed by a public iterator."""

    child = _invalid_example_child()
    if container_kind == "tuple":
        public_view: DivergentTuple | DivergentList = DivergentTuple((), (child,))
        value: BaseModel = PublicViewTupleEnvelope(trigger=(), children=())
        nested_outer: object = (public_view,)
    else:
        public_view = DivergentList([], [child])
        value = PublicViewListEnvelope(trigger=(), children=[])
        nested_outer = [public_view]
    native = _BASE_MODEL_DICT_DESCRIPTOR.__get__(value, BaseModel)
    if placement == "direct":
        dict.__setitem__(native, "children", public_view)
    else:
        dict.__setitem__(native, "groups", nested_outer)
    return value, child, public_view


def _exposed_children(snapshot: dict, placement: str) -> object:
    children = snapshot["children"] if placement == "direct" else snapshot["groups"][0]
    return children


@pytest.mark.parametrize("placement", ["direct", "nested"])
@pytest.mark.parametrize("container_kind", ["tuple", "list"])
def test_public_first_unrepaired_public_only_child_is_exposed_and_rejected(
    container_kind: str,
    placement: str,
) -> None:
    value, child, public_view = _public_only_envelope(container_kind, placement)

    if placement == "direct":
        # The exposed entry is invalid as it is yielded: the whole input is refused (a prefix is never returned
        # in its place), after exactly one read of the view.
        with pytest.raises(ValueError, match="not valid"):
            snapshot_contract_graph(value, contract=type(value))
        assert public_view.iteration_calls == 1
        assert child.count == -1
        return
    # A view held inside an outer sequence has no single declared entry model: it is exposed and the receiving
    # model rejects it.
    snapshot = snapshot_contract_graph(value)
    assert list(_exposed_children(snapshot, placement)) == [{"count": -1}]
    assert public_view.iteration_calls == 1
    assert child.count == -1
    with pytest.raises(ValidationError) as raised:
        type(value).model_validate(snapshot, strict=True)
    assert raised.value.errors()[0]["type"] == "greater_than_equal"
    assert raised.value.errors()[0]["input"] == -1


@pytest.mark.parametrize("hook", ["hash", "equality"])
@pytest.mark.parametrize("placement", ["direct", "nested"])
@pytest.mark.parametrize("container_kind", ["tuple", "list"])
def test_public_first_public_only_child_repaired_by_earlier_stored_name_hook(
    container_kind: str,
    placement: str,
    hook: str,
) -> None:
    value, child, public_view = _public_only_envelope(container_kind, placement)
    native = _BASE_MODEL_DICT_DESCRIPTOR.__get__(value, BaseModel)
    stored_name = RepairingStoredName("trigger", _repair_example_child(child), hook)
    stored_value = dict.pop(native, "trigger")
    dict.__setitem__(native, stored_name, stored_value)
    stored_name.armed = True
    stored_name.hash_calls = stored_name.equality_calls = 0

    # A stored name is read as its text, so no hook runs before the public view is first
    # observed: the child is exposed as it is (invalid) and is refused (directly) or rejected by the model.
    if placement == "direct":
        with pytest.raises(ValueError, match="not valid"):
            snapshot_contract_graph(value, contract=type(value))
    else:
        snapshot = snapshot_contract_graph(value)
        assert list(_exposed_children(snapshot, placement)) == [{"count": -1}]
        with pytest.raises(ValidationError):
            type(value).model_validate(snapshot, strict=True)

    assert stored_name.hash_calls == stored_name.equality_calls == 0
    assert child.count == -1
    assert public_view.iteration_calls == 1


@pytest.mark.parametrize("placement", ["direct", "nested"])
@pytest.mark.parametrize("container_kind", ["tuple", "list"])
def test_public_first_public_only_child_repaired_by_earlier_extra_items_callback(
    container_kind: str,
    placement: str,
) -> None:
    value, child, public_view = _public_only_envelope(container_kind, placement)
    extras = RepairingMapping(_repair_example_child(child))
    _BASE_MODEL_EXTRA_DESCRIPTOR.__set__(value, extras)

    snapshot = snapshot_contract_graph(value)

    assert extras.item_reads == 1
    assert child.count == 1
    assert list(_exposed_children(snapshot, placement)) == [{"count": 1}]
    assert public_view.iteration_calls == 1
    assert type(value).model_validate(snapshot, strict=True) is not None


@pytest.mark.parametrize("container_kind", ["tuple", "list"])
def test_public_first_native_stored_child_stays_fenced_against_the_same_callbacks(
    container_kind: str,
) -> None:
    child = _invalid_example_child()
    if container_kind == "tuple":
        value: BaseModel = PublicViewTupleEnvelope(trigger=(), children=())
        native_children: object = (child,)
    else:
        value = PublicViewListEnvelope(trigger=(), children=[])
        native_children = [child]
    native = _BASE_MODEL_DICT_DESCRIPTOR.__get__(value, BaseModel)
    dict.__setitem__(native, "children", native_children)
    stored_name = RepairingStoredName("trigger", _repair_example_child(child), "hash")
    stored_value = dict.pop(native, "trigger")
    dict.__setitem__(native, stored_name, stored_value)
    stored_name.armed = True
    extras = RepairingMapping(_repair_example_child(child))
    _BASE_MODEL_EXTRA_DESCRIPTOR.__set__(value, extras)

    snapshot = snapshot_contract_graph(value)

    # The callbacks ran and repaired the live object, but the model was fenced
    # from native storage first, so the snapshot keeps the entry-time state.
    assert stored_name.hash_calls > 0
    assert extras.item_reads == 1
    assert child.count == 1
    assert list(snapshot["children"]) == [{"count": -1}]
    with pytest.raises(ValidationError) as raised:
        type(value).model_validate(snapshot, strict=True)
    assert raised.value.errors()[0]["input"] == -1


@pytest.mark.parametrize("container_kind", ["tuple", "list"])
def test_public_first_public_view_is_read_once_per_edge_not_per_snapshot(
    container_kind: str,
) -> None:
    _, _, public_view = _public_only_envelope(container_kind, "direct")

    # Not a model field, so there is no early check: two edges reach the one view.
    snapshot = snapshot_contract_graph({"one": public_view, "two": [public_view]})

    # One read for each edge that reaches the view (the direct one and the nested one); a single
    # edge reads it exactly once (see the tests above).
    assert public_view.iteration_calls == 2
    assert list(snapshot["one"]) == list(snapshot["two"][0]) == [{"count": -1}]


# ---------------------------------------------------------------------------
# PR #399 review regressions: strong references and sibling pre-capture.
# ---------------------------------------------------------------------------


class _TemporaryDictTuple(tuple[object, ...]):
    """Yields a brand-new dictionary on every public iteration."""

    def __iter__(self):  # type: ignore[no-untyped-def]
        self.reads = getattr(self, "reads", 0) + 1
        return iter(({"read": self.reads},))


class _TemporaryDictList(list[object]):
    def __iter__(self):  # type: ignore[no-untyped-def]
        self.reads = getattr(self, "reads", 0) + 1
        return iter([{"read": self.reads}])


@pytest.mark.parametrize("container_kind", ["tuple", "list"])
def test_public_view_temporaries_do_not_alias_through_reused_object_ids(
    container_kind: str,
) -> None:
    if container_kind == "tuple":
        view: object = _TemporaryDictTuple()
        outer: object = (view,) * 6
    else:
        view = _TemporaryDictList()
        outer = [view] * 6

    snapshot = snapshot_contract_graph(outer)

    # Each edge reads the view afresh; a released temporary's id must not make
    # a later edge reuse the earlier temporary's captured contents.
    assert [list(edge) for edge in snapshot] == [[{"read": n}] for n in range(1, 7)]


class _CallbackTuple(tuple[object, ...]):
    def __new__(cls, callback: Callable[[], None]) -> "_CallbackTuple":
        instance = tuple.__new__(cls, ())
        instance.callback = callback
        return instance

    def __iter__(self):  # type: ignore[no-untyped-def]
        self.callback()
        return iter(())


class _CallbackList(list[object]):
    def __init__(self, callback: Callable[[], None]) -> None:
        list.__init__(self, [])
        self.callback = callback

    def __iter__(self):  # type: ignore[no-untyped-def]
        self.callback()
        return iter(())


@pytest.mark.parametrize("sibling_kind", ["dict", "model"])
@pytest.mark.parametrize("container_kind", ["tuple", "list"])
def test_exposed_siblings_are_captured_before_the_first_is_reconstructed(
    container_kind: str,
    sibling_kind: str,
) -> None:
    if sibling_kind == "dict":
        sibling: object = {"count": -1}

        def repair() -> None:
            sibling["count"] = 1  # type: ignore[index]

        expected: object = {"count": -1}
    else:
        sibling = _invalid_example_child()
        repair = _repair_example_child(sibling)  # type: ignore[arg-type]
        expected = {"count": -1}

    if container_kind == "tuple":
        repairing: object = _CallbackTuple(repair)
        exposed: object = DivergentTuple((), (repairing, sibling))
    else:
        repairing = _CallbackList(repair)
        exposed = DivergentList([], [repairing, sibling])

    snapshot = snapshot_contract_graph((exposed,))

    first, second = snapshot[0]
    assert list(first) == []
    # The repair ran, but the sibling was frozen before the first item was
    # reconstructed, so the snapshot keeps the exposed (invalid) value.
    assert (sibling["count"] if sibling_kind == "dict" else sibling.count) == 1  # type: ignore[index,union-attr]
    assert second == expected


def _shared_ladder(depth: int) -> list:
    level: list = []
    for _ in range(depth):
        level = [level, level]
    return level


def _unique_lists(value: object) -> int:
    seen: set[int] = set()
    stack = [value]
    while stack:
        item = stack.pop()
        if type(item) is list and id(item) not in seen:
            seen.add(id(item))
            stack.extend(item)
    return len(seen)


def test_snapshot_contract_graph_reuses_completed_shared_subtrees() -> None:
    snapshot = snapshot_contract_graph(_shared_ladder(15))

    # One rebuilt list per distinct source list, not one per path (32,767).
    assert _unique_lists(snapshot) == 16


def test_snapshot_contract_graph_completes_deep_shared_ladder() -> None:
    snapshot = snapshot_contract_graph(_shared_ladder(60))

    assert _unique_lists(snapshot) == 61


def test_snapshot_contract_graph_shared_reuse_is_a_detached_copy() -> None:
    source = _shared_ladder(3)
    snapshot = snapshot_contract_graph(source)
    source[0].append("late")

    assert snapshot[0] is snapshot[1]
    assert snapshot[0] is not source[0]
    assert snapshot[0][0][0] == []


def test_snapshot_contract_graph_still_rejects_cycles_with_memo() -> None:
    shared: list = []
    cyclic: list = [shared, shared]
    shared.append(cyclic)

    with pytest.raises(Exception):
        snapshot_contract_graph(cyclic)


def test_snapshot_contract_graph_reads_public_view_once_per_edge() -> None:
    class CountingList(list):
        reads = 0

        def __iter__(self):
            type(self).reads += 1
            return super().__iter__()

    view = CountingList([1])
    snapshot = snapshot_contract_graph([[view], [view], {"k": [view]}])

    assert CountingList.reads == 3
    assert snapshot[0] == snapshot[1] == [[1]]


def test_snapshot_contract_graph_public_view_ancestors_are_not_reused() -> None:
    class CountingList(list):
        reads = 0

        def __iter__(self):
            type(self).reads += 1
            return super().__iter__()

    view = CountingList([1])
    parent = [view]
    snapshot_contract_graph([parent, parent, parent])

    assert CountingList.reads == 3


# ── one accounting: the shared discovery budget bounds capture and reconstruction ──

import time as _time  # noqa: E402

from contracts_limits import DISCOVERY_BUDGET_MESSAGE, STANDARD_MAX_DEPTH, STANDARD_MAX_EDGES  # noqa: E402
from contracts_references import SourceVersion  # noqa: E402


def _source_version() -> SourceVersion:
    return SourceVersion(record_id="rec-1", content_revision=1, policy_epoch=1)


def _pairs(depth: int, leaf: object = None) -> object:
    node: object = ({"x": 1},) if leaf is None else leaf
    for _ in range(depth):
        node = (node, node)
    return node


def test_shared_native_pairs_are_rebuilt_once_per_object(monkeypatch) -> None:
    """``node = (node, node)`` twelve times is 13 containers and 8,191 paths."""

    import contracts_common

    calls = 0
    original = contracts_common._snapshot_captured_graph

    def counting(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(contracts_common, "_snapshot_captured_graph", counting)
    snapshot = snapshot_contract_graph(_pairs(12))
    assert calls <= 40  # two edges per distinct container (the per-path copy made 16,383 calls)
    assert snapshot[0] is snapshot[1]


def test_fifteen_lists_of_two_references_are_not_expanded() -> None:
    node: object = []
    for _ in range(15):
        node = [node, node]
    started = _time.monotonic()
    snapshot = snapshot_contract_graph(node)
    assert _time.monotonic() - started < 1.0 and snapshot[0] is snapshot[1]


def test_a_forty_level_native_diamond_completes_quickly() -> None:
    started = _time.monotonic()
    snapshot_contract_graph(_pairs(40))
    assert _time.monotonic() - started < 2.0


def test_deep_nesting_is_the_stable_budget_error_not_a_recursion_error() -> None:
    for depth in (STANDARD_MAX_DEPTH + 10, 5_000):
        node: object = ()
        for _ in range(depth):
            node = (node,)
        with pytest.raises(ValueError, match=DISCOVERY_BUDGET_MESSAGE):
            snapshot_contract_graph(node)


def test_the_depth_limit_holds_on_every_path_to_a_reused_subtree() -> None:
    deep: object = ()
    for _ in range(STANDARD_MAX_DEPTH - 2):
        deep = (deep,)
    snapshot_contract_graph((deep,))  # fits at depth 1
    with pytest.raises(ValueError, match=DISCOVERY_BUDGET_MESSAGE):
        snapshot_contract_graph((deep, (((deep,),),)))  # the same subtree, three levels lower


def test_an_endless_public_sequence_is_cut_off_by_the_budget() -> None:
    produced = [0]

    class Endless(tuple):
        def __iter__(self):
            # Self-limiting so a tree without the budget fails instead of exhausting memory.
            while produced[0] < 4 * STANDARD_MAX_EDGES:
                produced[0] += 1
                yield produced[0]
            raise AssertionError("the discovery budget did not stop the view")

    with pytest.raises(ValueError, match=DISCOVERY_BUDGET_MESSAGE):
        snapshot_contract_graph(Endless())
    assert produced[0] <= 140_000  # the edge limit is 131,072


def test_a_diamond_of_public_views_is_charged_per_path() -> None:
    class View(tuple):
        def __iter__(self):
            return iter(tuple.__iter__(self))

    node: object = View((1,))
    for _ in range(30):
        node = View((node, node))
    started = _time.monotonic()
    with pytest.raises(ValueError, match=DISCOVERY_BUDGET_MESSAGE):
        snapshot_contract_graph(node)
    assert _time.monotonic() - started < 5.0


# ── purity is not "no public view was read": key hooks are callbacks too ──


def test_a_str_subclass_key_runs_no_hook_and_cannot_skew_later_views() -> None:
    counter = [0]

    class CountingKey(str):
        def __hash__(self):
            counter[0] += 1
            return str.__hash__(self)

        def __eq__(self, other):
            counter[0] += 1
            return str.__eq__(self, other)

    class CounterView(list):
        def __iter__(self):
            return iter([counter[0]])

    shared = {CountingKey("k"): 1}
    view = CounterView()
    counter[0] = 0  # building the dict above hashed the key once; only the snapshot is under test
    snapshot = snapshot_contract_graph([shared, view, shared, view])
    assert counter[0] == 0                      # the key is read as its text, never hashed through its hooks
    assert snapshot[1] == snapshot[3] == [0]    # both observations of the view agree with that
    assert [type(k) for k in snapshot[0]] == [str]


def test_two_keys_with_one_text_are_refused() -> None:
    class Twin(str):
        def __hash__(self):
            return hash("x" + str.__str__(self))

        def __eq__(self, other):
            return False

    value = {"k": 1}
    dict.__setitem__(value, Twin("k"), 2)
    with pytest.raises(ValueError, match="colliding"):
        snapshot_contract_graph(value)


def test_a_subtree_with_a_non_text_hashable_key_is_never_reused() -> None:
    counter = [0]

    class Key:
        def __hash__(self):
            counter[0] += 1
            return 7

        def __eq__(self, other):
            return self is other

    class CounterView(list):
        def __iter__(self):
            return iter([counter[0]])

    shared = {Key(): 1}
    view = CounterView()
    snapshot = snapshot_contract_graph([shared, view, shared, view])
    assert snapshot[0] is not snapshot[2]  # rebuilt per edge, like a public view
    assert snapshot[1] != snapshot[3]       # so each later observation sees its own hook run


def test_str_subclass_model_field_names_are_read_as_text() -> None:
    counter = [0]

    class CountingName(str):
        def __hash__(self):
            counter[0] += 1
            return str.__hash__(self)

    value = _source_version()
    stored = vars(BaseModel)["__dict__"].__get__(value, BaseModel)
    renamed = {CountingName(name): item for name, item in dict.items(stored)}
    stored.clear()
    counter[0] = 0
    stored.update(renamed)
    counter[0] = 0
    snapshot = snapshot_contract_graph(value)
    assert counter[0] == 0
    assert all(type(name) is str for name in snapshot)


# ── type checks by native MRO, no metaclass hook (from the snapshot hardening) ──


def _hooked(events: list[str]):
    class Meta(type):
        def __hash__(cls):
            events.append("hash")
            return id(cls)

        def __eq__(cls, other):
            events.append("eq")
            return False

        def __ne__(cls, other):
            events.append("ne")
            return True

        def __subclasscheck__(cls, subclass):
            events.append("subclasscheck")
            return False

        def __instancecheck__(cls, instance):
            events.append("instancecheck")
            return False

    return Meta


@pytest.mark.parametrize("base", [tuple, list, dict])
def test_the_shared_snapshot_runs_no_metaclass_hook_on_a_hostile_container_class(base) -> None:
    events: list[str] = []
    hostile = _hooked(events)("Hostile", (base,), {})
    value = hostile()
    events.clear()
    try:
        snapshot_contract_graph(value)
    except Exception:  # an unsupported subclass may be refused; running its hooks is not allowed
        pass
    assert events == []


def test_a_class_only_registered_as_a_mapping_is_not_trusted_as_model_extras() -> None:
    from collections.abc import Mapping as _Mapping

    touched: list[str] = []

    class RegisteredOnly:
        def items(self):
            touched.append("items")
            return [("smuggled", 1)]

        def __iter__(self):
            touched.append("iter")
            return iter(())

        def __len__(self):
            return 0

    _Mapping.register(RegisteredOnly)
    value = _source_version()
    object.__setattr__(value, "__pydantic_extra__", RegisteredOnly())
    with pytest.raises(ValueError):
        snapshot_contract_graph(value)
    assert touched == []


def test_ordinary_graphs_are_far_inside_the_budget() -> None:
    snapshot = snapshot_contract_graph([_source_version() for _ in range(500)])
    assert len(snapshot) == 500

# ── discovery budget: design-neutral limits (depth, nodes, edges, views, hygiene) ──

from contracts_limits import STANDARD  # noqa: E402


def _nested_tuple(depth: int) -> tuple:
    value: tuple = ()
    for _ in range(depth):
        value = (value,)
    return value


def test_snapshot_contract_graph_rejects_unbounded_depth_without_recursion_error() -> None:
    with pytest.raises(ValueError, match=DISCOVERY_BUDGET_MESSAGE):
        snapshot_contract_graph(_nested_tuple(5000))


def test_snapshot_contract_graph_accepts_the_depth_limit_and_rejects_one_more() -> None:
    limit = STANDARD.max_depth

    snapshot = snapshot_contract_graph(_nested_tuple(limit))
    assert snapshot == _nested_tuple(limit)
    with pytest.raises(ValueError, match=DISCOVERY_BUDGET_MESSAGE):
        snapshot_contract_graph(_nested_tuple(limit + 1))


@pytest.mark.parametrize("shallow_first", [True, False])
def test_snapshot_contract_graph_depth_does_not_depend_on_visit_order(
    shallow_first: bool,
) -> None:
    tall = _nested_tuple(STANDARD.max_depth - 10)
    deep_path = tall
    for _ in range(20):
        deep_path = (deep_path,)
    # `tall` is reached at depth 1 and again at depth 21 through a chain.
    value = (tall, deep_path) if shallow_first else (deep_path, tall)

    with pytest.raises(ValueError, match=DISCOVERY_BUDGET_MESSAGE):
        snapshot_contract_graph(value)


def test_snapshot_contract_graph_depth_holds_for_an_impure_subtree_reached_deeper() -> None:
    # The subtree reads a public view, so it is rebuilt (not reused) on every path; it fits
    # where it is first captured but not where the second path reaches it.
    class View(list):
        pass

    tall: list = [View([1])]
    for _ in range(STANDARD.max_depth - 10):
        tall = [tall]
    deep_path: object = tall
    for _ in range(20):
        deep_path = [deep_path]

    snapshot_contract_graph([tall])
    with pytest.raises(ValueError, match=DISCOVERY_BUDGET_MESSAGE):
        snapshot_contract_graph([tall, deep_path])


def test_snapshot_contract_graph_rejects_too_many_nodes() -> None:
    with pytest.raises(ValueError, match=DISCOVERY_BUDGET_MESSAGE):
        snapshot_contract_graph([[] for _ in range(STANDARD.max_nodes)])
    assert len(snapshot_contract_graph([[] for _ in range(STANDARD.max_nodes - 1)])) == (
        STANDARD.max_nodes - 1
    )


@pytest.mark.parametrize(
    "build",
    [list, tuple, lambda items: dict.fromkeys(items)],
    ids=["list", "tuple", "dict"],
)
def test_snapshot_contract_graph_rejects_too_many_edges(
    build: Callable[[range], object],
) -> None:
    with pytest.raises(ValueError, match=DISCOVERY_BUDGET_MESSAGE):
        snapshot_contract_graph(build(range(STANDARD.max_edges + 1)))


@pytest.mark.parametrize(
    "build",
    [list, tuple, lambda items: dict.fromkeys(items)],
    ids=["list", "tuple", "dict"],
)
def test_snapshot_contract_graph_charges_each_native_entry_once(
    build: Callable[[range], object],
) -> None:
    # One entry is one edge: capturing it and rebuilding it once is one traversal, so a
    # container just inside the edge limit is accepted.
    accepted = snapshot_contract_graph(build(range(STANDARD.max_edges - 1)))
    assert len(accepted) == STANDARD.max_edges - 1  # type: ignore[arg-type]
    shared: list = []
    assert len(snapshot_contract_graph([shared] * (STANDARD.max_edges - 2))) == (
        STANDARD.max_edges - 2
    )


def test_snapshot_contract_graph_rejects_too_many_model_edges() -> None:
    class Open(ContractModel):
        model_config = ConfigDict(extra="allow")

    model = Open.model_validate({f"k{index}": index for index in range(STANDARD.max_edges + 1)})

    with pytest.raises(ValueError, match=DISCOVERY_BUDGET_MESSAGE):
        snapshot_contract_graph(model)


class _EmptyView(list):
    pass


def test_snapshot_contract_graph_charges_ancestor_copies_that_cannot_be_reused() -> None:
    # 4,096 paths each observe the view once (8,192 charges), but every copy of
    # the wide ancestors on those paths must be rebuilt too and is charged.
    level: list = [_EmptyView()]
    for _ in range(12):
        level = [level, level, *range(30)]

    with pytest.raises(ValueError, match=DISCOVERY_BUDGET_MESSAGE):
        snapshot_contract_graph(level)

    narrow: list = [_EmptyView()]
    for _ in range(12):
        narrow = [narrow, narrow]
    snapshot_contract_graph(narrow)


def test_snapshot_contract_graph_bounds_depth_of_a_chain_formed_by_views() -> None:
    # Every view is captured at depth 1, but each one's public iteration exposes
    # the next, so the reconstruction path is far deeper than any native path.
    class Link(list):
        successor: object = None

        def __iter__(self):
            if self.successor is None:
                return iter(())
            return iter((self.successor,))

    links = [Link() for _ in range(STANDARD.max_depth + 20)]
    for current, successor in zip(links, links[1:]):
        current.successor = successor

    with pytest.raises(ValueError, match=DISCOVERY_BUDGET_MESSAGE):
        snapshot_contract_graph(links)


def test_snapshot_contract_graph_charges_aliased_edges() -> None:
    shared: list = []
    with pytest.raises(ValueError, match=DISCOVERY_BUDGET_MESSAGE):
        snapshot_contract_graph([shared] * (STANDARD.max_edges + 1))


def test_snapshot_contract_graph_budgets_a_shared_ladder_over_a_public_view() -> None:
    class View(list):
        pass

    level: list = [View([1])]
    for _ in range(40):
        level = [level, level]

    # Each path observes the view again, so the work is charged and stops.
    with pytest.raises(ValueError, match=DISCOVERY_BUDGET_MESSAGE):
        snapshot_contract_graph(level)


_ENDLESS_GUARD = 4 * STANDARD.max_edges


def test_snapshot_contract_graph_stops_an_endless_public_view() -> None:
    class Endless(list):
        def __iter__(self):
            # Self-limiting so a tree without the budget fails instead of exhausting memory.
            for _ in range(_ENDLESS_GUARD):
                yield 0
            raise AssertionError("the discovery budget did not stop the view")

    with pytest.raises(ValueError, match=DISCOVERY_BUDGET_MESSAGE):
        snapshot_contract_graph([Endless()])


def test_snapshot_contract_graph_stops_an_endless_mapping_view() -> None:
    class EndlessItems(Mapping):
        def __getitem__(self, key):
            raise KeyError(key)

        def __iter__(self):
            return iter(())

        def __len__(self):
            return 0

        def items(self):
            for index in range(_ENDLESS_GUARD):
                yield index, 0
            raise AssertionError("the discovery budget did not stop the view")

    class Open(ContractModel):
        pass

    model = Open()
    object.__setattr__(model, "__pydantic_extra__", EndlessItems())

    with pytest.raises(ValueError, match=DISCOVERY_BUDGET_MESSAGE):
        snapshot_contract_graph(model)


def test_snapshot_contract_graph_native_accounting_runs_no_caller_code() -> None:
    calls: list[str] = []

    class Hostile(list):
        def __len__(self):
            calls.append("len")
            return 0

        def __bool__(self):
            calls.append("bool")
            return True

        def __hash__(self):
            calls.append("hash")
            return 0

        def __eq__(self, other):
            calls.append("eq")
            return True

        def __getattribute__(self, name):
            if name == "__class__":
                calls.append("class")
            return super().__getattribute__(name)

    snapshot = snapshot_contract_graph([Hostile([1, 2]), [Hostile([3])]])

    assert snapshot == [[1, 2], [[3]]]
    assert calls == []


def test_snapshot_contract_graph_budget_error_is_value_free() -> None:
    marker = "very-secret-marker"
    value: object = (marker,)
    for _ in range(STANDARD.max_depth + 1):
        value = (value,)

    with pytest.raises(ValueError) as caught:
        snapshot_contract_graph(value)

    assert str(caught.value) == DISCOVERY_BUDGET_MESSAGE
    assert marker not in repr(caught.value)



# ── an admitted depth never overflows the stack, whatever the caller's own depth ──


def _with_caller_frames(headroom: int, call):
    """Run ``call`` with only ``headroom`` frames left below the interpreter's limit."""

    import sys as _sys

    def descend(remaining: int):
        if remaining <= 0:
            return call()
        return descend(remaining - 1)

    frame, used = _sys._getframe(), 0
    while frame is not None:
        frame, used = frame.f_back, used + 1
    return descend(_sys.getrecursionlimit() - used - headroom)


@pytest.mark.parametrize("kind", ["tuple", "list", "dict", "model-and-dict"])
def test_the_deepest_admitted_input_snapshots_with_a_deep_caller_stack(kind) -> None:
    depth = STANDARD_MAX_DEPTH - 1
    node: object = None
    for level in range(depth):
        if kind == "tuple":
            node = (node,)
        elif kind == "list":
            node = [node]
        elif kind == "dict" or level % 2:
            node = {"k": node}
        else:
            node = ExampleContractEnvelope.model_construct(item=node)
    # 300 frames of headroom: a reconstruction that costs three frames per level needs ~770
    snapshot = _with_caller_frames(300, lambda: snapshot_contract_graph(node))
    assert snapshot is not None


def test_a_recursion_error_is_never_what_a_caller_receives() -> None:
    node: object = None
    for _ in range(STANDARD_MAX_DEPTH - 1):
        node = [node]
    # far too little headroom for any reconstruction: the stable budget error, never RecursionError
    with pytest.raises(ValueError, match=DISCOVERY_BUDGET_MESSAGE):
        _with_caller_frames(40, lambda: snapshot_contract_graph(node))


# ── model field names are read as text without hooks, and duplicates are refused ──


class _CountingName(str):
    """Same text as a plain name but a different hash, so both can live in one dict."""

    calls: list = []

    def __hash__(self):
        _CountingName.calls.append("hash")
        return hash("other:" + str.__str__(self))

    def __eq__(self, other):
        _CountingName.calls.append("eq")
        return self is other


def test_a_colliding_stored_name_runs_no_hook_and_is_refused() -> None:
    value = ExampleContract(count=1)
    stored = vars(BaseModel)["__dict__"].__get__(value, BaseModel)
    dict.__setitem__(stored, _CountingName("count"), 5)  # a second "count" with a different hash path
    _CountingName.calls.clear()
    with pytest.raises(ValueError, match="colliding"):
        snapshot_contract_graph(value)
    assert _CountingName.calls == []


@pytest.mark.parametrize("order", ["plain-first", "subclass-first"])
def test_a_colliding_extra_name_runs_no_hook_and_is_refused(order) -> None:
    value = ExampleContract(count=1)
    pairs = [("note", 1), (_CountingName("note"), 2)]
    extra: dict = {}
    for key, item in pairs if order == "plain-first" else reversed(pairs):
        dict.__setitem__(extra, key, item)
    _BASE_MODEL_EXTRA_DESCRIPTOR.__set__(value, extra)
    _CountingName.calls.clear()
    with pytest.raises(ValueError, match="duplicate keys"):
        snapshot_contract_graph(value)
    assert _CountingName.calls == []


def test_snapshot_capture_stack_is_bounded_by_the_tier_depth_not_the_recursion_limit() -> None:
    # The stable error must come from the depth limit, not from running into the
    # interpreter's recursion limit, so a caller that raised that limit is still bounded.
    import sys

    import contracts_common

    deepest = 0
    current = 0
    capture_code = contracts_common._capture_native_graph.__code__

    def profile(frame, event, arg):
        nonlocal deepest, current
        if frame.f_code is capture_code:
            if event == "call":
                current += 1
                deepest = max(deepest, current)
            elif event == "return":
                current -= 1

    value: object = ()
    for _ in range(3_000):
        value = (value,)
    limit = sys.getrecursionlimit()
    sys.setrecursionlimit(5_000)
    sys.setprofile(profile)
    try:
        with pytest.raises(ValueError, match=DISCOVERY_BUDGET_MESSAGE):
            snapshot_contract_graph(value)
    finally:
        sys.setprofile(None)
        sys.setrecursionlimit(limit)
    assert deepest <= STANDARD.max_depth + 2


_EXTRA = BaseModel.__dict__["__pydantic_extra__"]


# ── purity: reconstruction that runs key code is never reused, on every path that reads keys ──


class _CountingKey:
    """A key of a non-plain type: building a dict with it runs this hash."""

    hashes = 0

    def __init__(self, name: str) -> None:
        self.name = name

    def __hash__(self) -> int:
        _CountingKey.hashes += 1
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _CountingKey) and other.name == self.name


class _PublicExtras(Mapping):
    """Non-dict extra storage whose public items carry the counting key."""

    def __init__(self, key: object) -> None:
        self.key = key

    def __getitem__(self, key):
        raise KeyError(key)

    def __iter__(self):
        return iter(())

    def __len__(self) -> int:
        return 0

    def items(self):
        return [(self.key, 1)]


def _hashes_for_references(build, count: int) -> int:
    _CountingKey.hashes = 0
    value = build()
    try:
        snapshot_contract_graph([value] * count)
    except ValueError:
        pass  # a model with surplus names is refused at the end; the walk still ran
    return _CountingKey.hashes


def _model_with_public_extras():
    value = SourceVersion(record_id="r1", content_revision=1, policy_epoch=1)
    _EXTRA.__set__(value, _PublicExtras(_CountingKey("surplus")))
    return value


def _model_with_native_extras():
    value = SourceVersion(record_id="r1", content_revision=1, policy_epoch=1)
    _EXTRA.__set__(value, {_CountingKey("surplus"): 1})
    return value


def _model_with_a_stored_key():
    value = SourceVersion(record_id="r1", content_revision=1, policy_epoch=1)
    dict.__setitem__(_BASE_MODEL_DICT_DESCRIPTOR.__get__(value, BaseModel), _CountingKey("surplus"), 1)
    return value


def _plain_dict_with_the_key():
    return {_CountingKey("k"): 1}


@pytest.mark.parametrize(
    "build",
    [_model_with_public_extras, _model_with_native_extras, _model_with_a_stored_key, _plain_dict_with_the_key],
    ids=["non-dict-extras", "dict-extras", "stored-name", "plain-dict-value"],
)
def test_key_code_runs_for_every_reference_not_just_the_first_two(build) -> None:
    one, two, three = (_hashes_for_references(build, count) for count in (1, 2, 3))
    assert one > 0  # the key's code is reached at all
    # capture may hash once; every further reference rebuilds, and so hashes, again
    assert two - one > 0
    assert three - two == two - one, (one, two, three)  # a later reference never skips the callbacks


# ── the native depth limit does not depend on which sibling reaches a shared subtree first ──


class _EmptyView(list):
    """A list whose public view yields nothing while its native backing holds the child."""

    def __iter__(self):
        return iter(())


def _deep_subtree(levels: int) -> list:
    node: list = []
    for _ in range(levels):
        node = [node]
    return node


def _hidden_path(subtree: object, wrappers: int) -> object:
    node = subtree
    for _ in range(wrappers):
        node = _EmptyView([node])
    return node


@pytest.mark.parametrize("hidden_first", [False, True], ids=["direct-first", "hidden-first"])
def test_a_subtree_reached_deep_through_empty_public_views_is_refused_in_either_sibling_order(hidden_first) -> None:
    subtree = _deep_subtree(250)  # 251 levels on its own: inside the 256 limit
    snapshot_contract_graph([subtree])  # alone it is accepted
    hidden = _hidden_path(subtree, 10)  # the same subtree 11 levels down: 261 native levels
    siblings = [hidden, subtree] if hidden_first else [subtree, hidden]
    with pytest.raises(ValueError, match="budget"):
        snapshot_contract_graph(siblings)


def test_a_shared_subtree_that_stays_inside_the_limit_on_every_path_is_still_accepted() -> None:
    subtree = _deep_subtree(200)
    hidden = _hidden_path(subtree, 10)  # 211 levels at the deepest
    for siblings in ([subtree, hidden], [hidden, subtree], [subtree, subtree, hidden]):
        assert snapshot_contract_graph(siblings) is not None


@pytest.mark.parametrize("wrap", [list, tuple], ids=["list", "tuple"])
def test_a_revisit_in_a_deep_position_through_model_and_dict_edges_is_refused_in_either_order(wrap) -> None:
    subtree = _deep_subtree(250)
    deep: object = {"k": subtree}
    for _ in range(10):
        deep = {"k": wrap([deep])}
    for siblings in ([subtree, deep], [deep, subtree]):
        with pytest.raises(ValueError, match="budget"):
            snapshot_contract_graph(wrap(siblings))


# ── extras found late (through a public mapping) respect every depth the model was reached at ──


def _model_with_deep_extras(levels: int, *, native: bool):
    value = SourceVersion(record_id="r1", content_revision=1, policy_epoch=1)
    extras = {"deep": _deep_subtree(levels)}
    _EXTRA.__set__(value, extras if native else MappingProxyType(extras))
    return value


@pytest.mark.parametrize("native", [False, True], ids=["public-mapping-extras", "dict-extras"])
@pytest.mark.parametrize("hidden_first", [False, True], ids=["direct-first", "hidden-first"])
def test_extras_are_checked_at_the_deepest_depth_the_model_was_reached(native, hidden_first) -> None:
    model = _model_with_deep_extras(250, native=native)  # 252 levels below the model: fine on its own
    snapshot_contract_graph([model])
    hidden = _hidden_path(model, 10)  # the same model 11 levels down through empty public views
    siblings = [hidden, model] if hidden_first else [model, hidden]
    with pytest.raises(ValueError, match="budget"):
        snapshot_contract_graph(siblings)


@pytest.mark.parametrize("native", [False, True], ids=["public-mapping-extras", "dict-extras"])
def test_extras_found_late_on_a_shallow_path_still_pass_when_every_path_fits(native) -> None:
    model = _model_with_deep_extras(200, native=native)
    hidden = _hidden_path(model, 10)
    for siblings in ([model, hidden], [hidden, model]):
        assert snapshot_contract_graph(siblings) is not None


@pytest.mark.parametrize("native", [False, True], ids=["public-mapping-extras", "dict-extras"])
@pytest.mark.parametrize("deep_first", [False, True], ids=["shallow-first", "deep-first"])
def test_a_late_height_reaches_ancestors_that_a_deeper_path_visits(native, deep_first) -> None:
    model = _model_with_deep_extras(200, native=native)
    head = _hidden_path(model, 5)  # an ancestor of the model through native edges
    deeper: object = head
    for _ in range(60):  # a second, deeper path to the same ancestor
        deeper = [deeper]
    siblings = [model, head, deeper]
    if deep_first:
        siblings = [deeper, model, head]
    with pytest.raises(ValueError, match="budget"):
        snapshot_contract_graph(siblings)


def test_the_ancestor_of_a_late_height_is_accepted_when_the_deeper_path_fits() -> None:
    model = _model_with_deep_extras(100, native=False)
    head = _hidden_path(model, 5)
    deeper: object = head
    for _ in range(40):
        deeper = [deeper]
    for siblings in ([model, head, deeper], [deeper, model, head]):
        assert snapshot_contract_graph(siblings) is not None


class _ViewOf(list):
    """A list whose public view yields one item that its native backing does not hold."""

    def __init__(self, shown: object) -> None:
        super().__init__()
        self._shown = shown

    def __iter__(self):
        return iter((self._shown,))


@pytest.mark.parametrize("model_first", [False, True], ids=["view-first", "model-first"])
def test_a_late_height_is_checked_at_the_depth_an_ancestor_reached_through_a_public_view(model_first) -> None:
    model = _model_with_deep_extras(150, native=False)
    ancestor = _hidden_path(model, 5)  # holds the model natively, five levels down
    shown: object = _ViewOf(ancestor)  # reached only through this view, one hundred levels deep
    for _ in range(100):
        shown = [shown]
    siblings = [model, shown] if model_first else [shown, model]
    with pytest.raises(ValueError, match="budget"):
        snapshot_contract_graph(siblings)


def test_the_deepest_visit_counts_even_when_a_shallower_one_comes_later() -> None:
    model = _model_with_deep_extras(150, native=False)
    ancestor = _hidden_path(model, 5)
    deep: object = _ViewOf(ancestor)
    for _ in range(100):
        deep = [deep]
    first, last = _ViewOf(ancestor), _ViewOf(ancestor)  # the same ancestor, near the top before and after
    with pytest.raises(ValueError, match="budget"):
        snapshot_contract_graph([first, deep, last, model])  # the extras are found after every visit


@pytest.mark.parametrize("hidden_first", [False, True], ids=["direct-first", "hidden-first"])
def test_a_height_found_late_inside_late_extras_reaches_the_outer_model_and_its_ancestors(hidden_first) -> None:
    inner = _model_with_deep_extras(250, native=False)  # 252 levels once its extras are read
    outer = SourceVersion(record_id="r2", content_revision=1, policy_epoch=1)
    _EXTRA.__set__(outer, MappingProxyType({"inner": inner}))  # found late too: one more level
    snapshot_contract_graph([outer])  # 254 levels on its own: inside the limit
    hidden = _hidden_path(outer, 10)
    siblings = [hidden, outer] if hidden_first else [outer, hidden]
    with pytest.raises(ValueError, match="budget"):
        snapshot_contract_graph(siblings)


# ── model metadata is read without caller hooks, so a reused subtree ran no caller code ──


def _counter_view(counter):
    class CounterView(list):
        def __iter__(self):
            return iter([counter[0]])

    return CounterView()


def test_a_metaclass_hook_on_model_metadata_never_runs_and_cannot_skew_later_views() -> None:
    from pydantic._internal._model_construction import ModelMetaclass

    counter = [0]

    class Meta(ModelMetaclass):
        def __getattribute__(cls, name):
            if name in ("model_fields", "model_config", "__pydantic_fields__"):
                counter[0] += 1
            return super().__getattribute__(name)

    class Hooked(BaseModel, metaclass=Meta):
        a: int = 1

    model = Hooked()
    view = _counter_view(counter)
    counter[0] = 0
    snapshot = snapshot_contract_graph([model, view, model, view])
    assert counter[0] == 0                     # the fields and configuration are read natively
    assert snapshot[1] == snapshot[3] == [0]   # so both observations of the view agree
    assert snapshot[0] == snapshot[2] == {"a": 1}


@pytest.mark.parametrize("shape", ["alias", "path-root", "choices"])
def test_a_str_subclass_validation_alias_runs_no_hook_and_cannot_skew_later_views(shape) -> None:
    from pydantic import AliasChoices, AliasPath, Field

    counter = [0]

    class CountingAlias(str):
        def __hash__(self):
            counter[0] += 1
            return str.__hash__(self)

        def __eq__(self, other):
            counter[0] += 1
            return str.__eq__(self, other)

    alias = {
        "alias": CountingAlias("alias_a"),
        "path-root": AliasPath(CountingAlias("root_a"), 0),
        "choices": AliasChoices(CountingAlias("a1"), AliasPath(CountingAlias("a2"), "x")),
    }[shape]

    class Aliased(BaseModel):
        a: int = Field(1, validation_alias=alias)

    model = Aliased()
    view = _counter_view(counter)
    counter[0] = 0
    snapshot = snapshot_contract_graph([model, view, model, view])
    assert counter[0] == 0                     # an alias is read as its text, never hashed through its hooks
    assert snapshot[1] == snapshot[3] == [0]


def test_an_alias_whose_read_runs_code_is_refused_without_running_it() -> None:
    from pydantic import AliasPath, Field

    counter = [0]

    class CountingPath(AliasPath):
        @property
        def path(self):
            counter[0] += 1
            return ["root_a", 0]

        @path.setter
        def path(self, value):
            pass

    class Aliased(BaseModel):
        a: int = Field(1, validation_alias=CountingPath("root_a", 0))

    model = Aliased()
    counter[0] = 0  # pydantic read the alias while it built the class; only the snapshot is under test
    with pytest.raises(ValueError, match="metadata"):
        snapshot_contract_graph([model, _counter_view(counter), model])
    assert counter[0] == 0  # refused by its type, never read through its property


# ── reading configuration and alias storage runs no caller hook (or the subtree is not reused) ──


def _hooked_text(counter, text):
    class HookedText(str):
        def __hash__(self):
            counter[0] += 1
            return str.__hash__(str.__str__(self))

        def __eq__(self, other):
            counter[0] += 1
            return str.__eq__(str.__str__(self), other)

    return HookedText(text)


def _assert_consistent(counter, model) -> None:
    """Metadata that could only be read through hook-capable storage or keys is refused, and no hook runs."""

    view = _counter_view(counter)
    counter[0] = 0
    with pytest.raises(ValueError, match="metadata"):
        snapshot_contract_graph([model, view, model, view])
    assert counter[0] == 0


def test_a_str_subclass_key_in_the_model_configuration_is_refused_without_running_it() -> None:
    counter = [0]

    class Configured(BaseModel):
        a: int = 1

    config = vars(Configured)["model_config"]
    dict.__setitem__(config, _hooked_text(counter, "validate_by_alias"), True)
    _assert_consistent(counter, Configured())


@pytest.mark.parametrize("shape", ["choices", "path"])
def test_alias_storage_that_is_a_list_subclass_is_refused_without_running_it(shape) -> None:
    from pydantic import AliasChoices, AliasPath, Field

    counter = [0]

    class HookedList(list):
        def __iter__(self):
            counter[0] += 1
            return list.__iter__(self)

        def __len__(self):
            counter[0] += 1
            return list.__len__(self)

        def __getitem__(self, index):
            counter[0] += 1
            return list.__getitem__(self, index)

    if shape == "choices":
        alias = AliasChoices("a1", "a2")
        object.__setattr__(alias, "choices", HookedList(["a1", "a2"]))
    else:
        alias = AliasPath("root_a", 0)
        object.__setattr__(alias, "path", HookedList(["root_a", 0]))

    class Aliased(BaseModel):
        a: int = Field(1, validation_alias=alias)

    _assert_consistent(counter, Aliased())


def test_a_hooked_key_in_a_model_class_namespace_is_refused_without_running_it() -> None:
    counter = [0]

    class Colliding(str):  # hashes like "model_config", so looking that name up compares against it
        def __hash__(self):
            return str.__hash__("model_config")

        def __eq__(self, other):
            counter[0] += 1
            return False

    class Meta(type(BaseModel)):
        @classmethod
        def __prepare__(mcs, name, bases, **kwargs):
            namespace = dict(super().__prepare__(name, bases, **kwargs))
            dict.__setitem__(namespace, Colliding("__ns_shadow__"), None)
            return namespace

    class Hooked(BaseModel, metaclass=Meta):
        a: int = 1

    _assert_consistent(counter, Hooked())


# ── model metadata is scanned once per class per call, bounded, and a non-text field key runs no hook ──


def test_model_metadata_is_scanned_once_per_class_however_many_instances(monkeypatch) -> None:
    import contracts_common

    attributes = {f"helper_{index}": staticmethod(lambda: None) for index in range(2_000)}
    Wide = type("Wide", (BaseModel,), {"__annotations__": {"a": int}, "a": 1, **attributes})
    scanned = [0]
    real = contracts_common._exact_entry

    def counting(items, name):
        listed = list(items)
        scanned[0] += len(listed)
        return real(listed, name)

    monkeypatch.setattr(contracts_common, "_exact_entry", counting)
    snapshot_contract_graph([Wide() for _ in range(500)])
    # one class: its namespace is scanned for the fields and the configuration once, not once per instance
    assert scanned[0] < 3 * 2_100, scanned[0]


def test_an_oversized_alias_list_is_refused_before_it_is_copied() -> None:
    import tracemalloc

    from pydantic import AliasChoices, Field

    alias = AliasChoices("a1", "a2")
    choices = ["a1"] * 2_000_000  # far past the edge allowance
    object.__setattr__(alias, "choices", choices)

    class Aliased(BaseModel):
        a: int = Field(1, validation_alias=alias)

    model = Aliased()
    tracemalloc.start()
    try:
        with pytest.raises(ValueError, match=DISCOVERY_BUDGET_MESSAGE):
            snapshot_contract_graph(model)
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 4_000_000, peak  # the 16 MB copy of the list was never made


def test_a_non_text_key_among_the_declared_fields_is_refused_without_running_it() -> None:
    counter = [0]

    class HookedKey:
        def __hash__(self):
            counter[0] += 1
            return 7

        def __eq__(self, other):
            counter[0] += 1
            return self is other

    class Declared(BaseModel):
        a: int = 1

    model = Declared()
    dict.__setitem__(vars(Declared)["__pydantic_fields__"], HookedKey(), vars(Declared)["__pydantic_fields__"]["a"])
    _assert_consistent(counter, model)


# ── cached class metadata is not reused across caller code, and the declared-field scans are charged ──


def _aliased_pair():
    from collections.abc import Mapping

    from pydantic import ConfigDict, Field

    class Aliased(BaseModel):
        model_config = ConfigDict(validate_by_alias=False, validate_by_name=True, extra="allow")
        a: int = Field(1, validation_alias="alias_a")

    config = vars(Aliased)["model_config"]
    first = Aliased()
    second = Aliased()
    vars(BaseModel)["__dict__"].__get__(second, BaseModel)["alias_a"] = 2  # an alias root, stored
    return Aliased, config, first, second, Mapping


def test_every_instance_is_judged_by_the_class_as_it_was_before_caller_code_ran() -> None:
    """One snapshot of the class per call, taken before any caller code runs: a callback later in the call
    (here a carrier's extra mapping, read while the graph is rebuilt) cannot change how any instance is judged."""

    Aliased, config, first, second, Mapping = _aliased_pair()

    class Flipping(Mapping):
        def __getitem__(self, key):
            raise KeyError(key)

        def __iter__(self):
            return iter(())

        def __len__(self):
            return 0

        def items(self):
            dict.__setitem__(config, "validate_by_alias", True)
            return iter(())

    carrier = Aliased()
    vars(BaseModel)["__pydantic_extra__"].__set__(carrier, Flipping())
    combined = snapshot_contract_graph([first, carrier, second])
    assert combined[2] == {"a": 1, "alias_a": 2}  # judged as the class was when the call began
    # a later call takes its own snapshot, of the class as it is then
    with pytest.raises(ValueError, match="conflicting declared and extra fields"):
        snapshot_contract_graph(second)


def test_a_key_hook_between_two_instances_cannot_change_how_the_second_is_judged() -> None:
    """The dict path: the first value's class is read, then a key hook flips the configuration before the
    second value of the same class is rebuilt.  Both are judged by the one snapshot."""

    Aliased, config, first, second, _ = _aliased_pair()

    armed = [False]

    class FlippingKey:  # a key the rebuilt dict hashes (a text key would be read as its text instead)
        def __hash__(self):
            if armed[0]:  # only once the call is running (building the dict below hashes the key too)
                dict.__setitem__(config, "validate_by_alias", True)
            return 1

        def __eq__(self, other):
            return self is other

    key = FlippingKey()
    graph = {key: first, "k2": second}
    armed[0] = True
    snapshot = snapshot_contract_graph(graph)
    assert config["validate_by_alias"] is True  # the hook did run during the call
    assert snapshot["k2"] == {"a": 1, "alias_a": 2}  # the same judgment as without the hook
    armed[0] = False
    dict.__setitem__(config, "validate_by_alias", False)
    assert snapshot_contract_graph(graph)["k2"] == snapshot["k2"]


def test_declared_fields_are_scanned_and_charged_once_per_class_per_call(monkeypatch) -> None:
    import contracts_common
    from pydantic import create_model

    Many = create_model("Many", **{f"f{index}": (int, 0) for index in range(2_000)})

    def sparse():
        instance = Many()
        vars(BaseModel)["__dict__"].__get__(instance, BaseModel).clear()  # almost no captured storage
        return instance

    reads = [0]
    real = contracts_common._model_metadata

    def counting(model_type, budget):
        reads[0] += 1
        return real(model_type, budget)

    monkeypatch.setattr(contracts_common, "_model_metadata", counting)
    snapshot_contract_graph([sparse() for _ in range(200)])
    assert reads[0] == 1  # one charged scan for the class, not one per occurrence


# ── the declared-field table is charged before it is read, and read from one copy ──


def _impure_wide_model(field_count: int) -> type[BaseModel]:
    """A model with ``field_count`` declared fields whose metadata is impure (one non-text key), so it is
    read again for every instance."""

    class Hashless:
        def __hash__(self):
            return 7

    Wide = type("Wide", (BaseModel,), {"__annotations__": {"a": int}, "a": 1})
    fields = vars(Wide)["__pydantic_fields__"]
    for index in range(field_count):
        dict.__setitem__(fields, f"f{index}", fields["a"])
    dict.__setitem__(fields, Hashless(), fields["a"])
    return Wide


def test_wide_model_metadata_is_read_once_per_call_and_charged() -> None:
    def wide(field_count):
        Wide = type("Wide", (BaseModel,), {"__annotations__": {"a": int}, "a": 1})
        fields = vars(Wide)["__pydantic_fields__"]
        for index in range(field_count):
            dict.__setitem__(fields, f"f{index}", fields["a"])
        return Wide

    Wide = wide(2_000)
    # 200 instances of a class with a 2,001-entry declared table: the table is read (and charged) once
    assert len(snapshot_contract_graph([Wide() for _ in range(200)])) == 200
    with pytest.raises(ValueError, match=DISCOVERY_BUDGET_MESSAGE):
        snapshot_contract_graph([wide(200_000)()])  # one read of a table past the edge allowance is refused
    with pytest.raises(ValueError, match="metadata"):
        snapshot_contract_graph([_impure_wide_model(10)()])  # a non-text declared key is refused


def test_a_declared_field_table_shared_by_many_classes_costs_its_size_per_class() -> None:
    class Base(BaseModel):
        a: int = 1

    def classes(field_count: int) -> list[BaseModel]:
        table = dict(vars(Base)["__pydantic_fields__"])
        for index in range(field_count):
            table[f"f{index}"] = table["a"]
        instances = []
        for index in range(100):
            Shared = type(f"Shared{index}", (Base,), {})
            type.__setattr__(Shared, "__pydantic_fields__", table)
            instances.append(Shared())
        return instances

    with pytest.raises(ValueError, match=DISCOVERY_BUDGET_MESSAGE):
        snapshot_contract_graph(classes(2_000))  # 100 classes, one 2,001-entry table each
    assert len(snapshot_contract_graph(classes(10))) == 100


def test_a_replaced_declared_field_table_is_refused_before_it_is_copied() -> None:
    import tracemalloc

    class Table(dict):
        pass

    class Replaced(BaseModel):
        a: int = 1

    table = Table(vars(Replaced)["__pydantic_fields__"])
    for index in range(300_000):
        dict.__setitem__(table, f"f{index}", table["a"])
    type.__setattr__(Replaced, "__pydantic_fields__", table)
    model = Replaced()
    tracemalloc.start()
    try:
        with pytest.raises(ValueError, match="metadata"):  # a replaced table is refused by its type
            snapshot_contract_graph(model)
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 4_000_000, peak  # the table was never copied


@pytest.mark.parametrize("hook", ["field", "choices", "path"])
def test_a_metadata_hook_that_grows_the_declared_table_cannot_break_the_scan(hook) -> None:
    from pydantic import AliasChoices, AliasPath
    from pydantic.fields import FieldInfo

    class Model(BaseModel):
        a: int = 1
        b: int = 2

    fields = vars(Model)["__pydantic_fields__"]

    def grow():
        dict.__setitem__(fields, f"late{len(fields)}", fields["b"])

    if hook == "field":
        class GrowingField(FieldInfo):
            __slots__ = ()

            @property
            def validation_alias(self):
                grow()
                return None

        replacement = GrowingField.__new__(GrowingField)
        for slot in FieldInfo.__slots__:
            try:
                object.__setattr__(replacement, slot, getattr(fields["a"], slot))
            except AttributeError:
                pass
        dict.__setitem__(fields, "a", replacement)
    elif hook == "choices":
        class GrowingChoices(AliasChoices):
            @property
            def choices(self):
                grow()
                return ["a1"]

        object.__setattr__(fields["a"], "validation_alias", GrowingChoices.__new__(GrowingChoices))
    else:
        class GrowingPath(AliasPath):
            @property
            def path(self):
                grow()
                return ["root"]

        object.__setattr__(fields["a"], "validation_alias", GrowingPath.__new__(GrowingPath))

    size = len(fields)
    with pytest.raises(ValueError, match="metadata"):  # hook-capable metadata types are refused, never read
        snapshot_contract_graph(Model())
    assert len(fields) == size  # no hook ran


# ── metadata reading runs no caller code at all: hook-capable metadata types are refused ──


def test_a_metadata_hook_cannot_repair_a_sibling_before_it_is_captured() -> None:
    from pydantic import AliasPath, Field

    class Counted(BaseModel):
        count: int = 0

    sibling = Counted()
    vars(BaseModel)["__dict__"].__get__(sibling, BaseModel)["count"] = -1  # unchecked, invalid

    armed = [False]

    class RepairingPath(AliasPath):
        @property
        def path(self):
            if armed[0]:  # pydantic reads the alias while it builds the class: only the snapshot is under test
                vars(BaseModel)["__dict__"].__get__(sibling, BaseModel)["count"] = 1  # repairs the sibling
            return ["root_a", 0]

        @path.setter
        def path(self, value):
            pass

    class Aliased(BaseModel):
        a: int = Field(1, validation_alias=RepairingPath("root_a", 0))

    model = Aliased()
    armed[0] = True
    with pytest.raises(ValueError, match="metadata"):
        snapshot_contract_graph([model, sibling])
    assert vars(BaseModel)["__dict__"].__get__(sibling, BaseModel)["count"] == -1  # no hook ran


@pytest.mark.parametrize("shape", ["field-subclass", "choices-subclass", "fields-dict-subclass", "config-not-dict"])
def test_hook_capable_metadata_types_are_refused(shape) -> None:
    from pydantic import AliasChoices, Field
    from pydantic.fields import FieldInfo

    class Model(BaseModel):
        a: int = Field(1, validation_alias="alias_a")

    if shape == "field-subclass":
        class Sub(FieldInfo):
            __slots__ = ()
        fields = vars(Model)["__pydantic_fields__"]
        replacement = Sub.__new__(Sub)
        for slot in FieldInfo.__slots__:
            try:
                object.__setattr__(replacement, slot, object.__getattribute__(fields["a"], slot))
            except AttributeError:
                pass
        dict.__setitem__(fields, "a", replacement)
    elif shape == "choices-subclass":
        class Choices(AliasChoices):
            pass
        object.__setattr__(vars(Model)["__pydantic_fields__"]["a"], "validation_alias", Choices("x", "y"))
    elif shape == "fields-dict-subclass":
        class Table(dict):
            pass
        type.__setattr__(Model, "__pydantic_fields__", Table(vars(Model)["__pydantic_fields__"]))
    else:
        from types import MappingProxyType
        type.__setattr__(Model, "model_config", MappingProxyType(dict(vars(Model)["model_config"])))
    with pytest.raises(ValueError, match="metadata"):
        snapshot_contract_graph(Model())


# ── alias storage that is not a native list or tuple is refused, never read as empty ──


@pytest.mark.parametrize("shape", ["choices", "path"])
def test_alias_storage_that_cannot_be_read_natively_is_refused_not_dropped(shape) -> None:
    from collections import UserList

    from pydantic import AliasChoices, AliasPath, ConfigDict, Field

    alias = AliasChoices("alias_a") if shape == "choices" else AliasPath("alias_a", 0)

    class Aliased(BaseModel):
        model_config = ConfigDict(validate_by_alias=True, validate_by_name=True, extra="allow")
        a: int = Field(1, validation_alias=alias)

    if shape == "choices":
        object.__setattr__(alias, "choices", UserList(["alias_a"]))
    else:
        object.__setattr__(alias, "path", UserList(["alias_a", 0]))
    both = Aliased()
    vars(BaseModel)["__dict__"].__get__(both, BaseModel)["alias_a"] = 2  # the field and its alias root, stored
    with pytest.raises(ValueError, match="metadata"):
        snapshot_contract_graph(both)  # used to drop the alias silently and accept the conflicting storage


# ── metadata is judged when a model is projected, not when it is merely captured ──


class _HiddenBacking(list):
    """A list whose public view is empty: what its native backing holds is never projected."""

    def __iter__(self):
        return iter(())


def _model_with_refused_metadata() -> BaseModel:
    class NonText:
        def __hash__(self):
            return 5

        def __eq__(self, other):
            return self is other

    class Refused(BaseModel):
        a: int = 1

    fields = vars(Refused)["__pydantic_fields__"]
    model = Refused()
    dict.__setitem__(fields, NonText(), fields["a"])  # metadata that is refused when the model is projected
    return model


def test_a_model_that_is_never_projected_is_not_refused_for_its_metadata() -> None:
    hidden = _HiddenBacking([_model_with_refused_metadata()])
    assert snapshot_contract_graph({"items": hidden}) == {"items": []}  # the public view is what is rebuilt


def test_the_same_model_is_refused_when_it_is_projected() -> None:
    with pytest.raises(ValueError, match="metadata"):
        snapshot_contract_graph({"items": [_model_with_refused_metadata()]})


# ── a missing slot on pydantic's own metadata types is the same deferred metadata refusal ──


def _model_missing_slot(shape) -> BaseModel:
    from pydantic import AliasChoices, AliasPath, Field

    alias = {"path": AliasPath("alias_a", 0), "choices": AliasChoices("alias_a"), "field": "alias_a"}[shape]

    class Slotted(BaseModel):
        a: int = Field(1, validation_alias=alias)

    model = Slotted()
    if shape == "path":
        object.__delattr__(alias, "path")
    elif shape == "choices":
        object.__delattr__(alias, "choices")
    else:
        object.__delattr__(vars(Slotted)["__pydantic_fields__"]["a"], "validation_alias")
    return model


@pytest.mark.parametrize("shape", ["path", "choices", "field"])
def test_a_missing_metadata_slot_on_a_hidden_model_does_not_abort_capture(shape) -> None:
    assert snapshot_contract_graph({"items": _HiddenBacking([_model_missing_slot(shape)])}) == {"items": []}


@pytest.mark.parametrize("shape", ["path", "choices", "field"])
def test_a_missing_metadata_slot_on_a_projected_model_is_the_metadata_refusal(shape) -> None:
    with pytest.raises(ValueError, match="metadata"):
        snapshot_contract_graph({"items": [_model_missing_slot(shape)]})


# ---- the enum-annotation helper runs on every supported Python (on 3.10 ``list[int]`` is an instance of ``type``)

def test_the_enum_annotation_helper_handles_generic_aliases_and_unions() -> None:
    import enum
    import typing

    from contracts_common import _annotation_holds_enum

    class Color(enum.Enum):
        RED = "red"

    class Plain:
        pass

    holds = [
        Color,
        list[Color],
        tuple[Color, ...],
        tuple[str, Color],
        dict[str, Color],
        dict[Color, int],
        Color | None,
        typing.Optional[Color],
        typing.Union[int, Color],
        typing.Annotated[Color, "note"],
        list[dict[str, tuple[Color, ...]]],
        typing.Annotated[list[Color], "note"],
    ]
    does_not = [
        Plain,
        str,
        int,
        list,
        list[int],
        tuple[str, ...],
        dict[str, int],
        int | None,
        typing.Optional[str],
        typing.Annotated[str, "note"],
        typing.Any,
        typing.Literal["a"],
        None,
        list[Plain],
    ]
    for annotation in holds:
        assert _annotation_holds_enum(annotation) is True, annotation
    for annotation in does_not:
        assert _annotation_holds_enum(annotation) is False, annotation


def test_a_model_with_generic_enum_annotations_defines_and_records_its_enum_fields() -> None:
    import enum
    import typing

    from contracts_common import ContractModel

    class Color(enum.Enum):
        RED = "red"

    class Holder(ContractModel):
        one: Color
        many: list[Color]
        fixed: tuple[Color, ...]
        mapping: dict[str, Color]
        maybe: Color | None = None
        optional: typing.Optional[Color] = None
        annotated: typing.Annotated[Color, "note"] = Color.RED
        text: str = ""
        numbers: list[int] = []

    assert set(Holder._enum_field_names) == {"one", "many", "fixed", "mapping", "maybe", "optional", "annotated"}


# ── a shared text-subclass alias is read as text once per call ──


class _SharedAlias(str):
    pass


def _alias_texts_seen(classes: int) -> tuple[int, int, bool]:
    """``(distinct alias texts across the class snapshots, classes snapshotted, all exact text)`` for ``classes``
    model classes whose every field shares one long text-subclass alias (twice in each choice list)."""

    import contracts_common
    from pydantic import Field

    alias = _SharedAlias("a" * 4096)
    models = []
    for index in range(classes):
        namespace = {
            "__annotations__": {"first": int, "second": int},
            "first": Field(0, validation_alias=AliasChoices(alias, alias)),
            "second": Field(0, validation_alias=AliasPath(alias, 0)),
        }
        models.append(type(f"Shares{index}", (BaseModel,), namespace))
    inventories = []
    real = contracts_common._GraphInventory

    def recording(*args, **kwargs):
        inventories.append(real(*args, **kwargs))
        return inventories[-1]

    contracts_common._GraphInventory = recording
    try:
        snapshot_contract_graph([model() for model in models])
    finally:
        contracts_common._GraphInventory = real
    (inventory,) = inventories
    texts = {
        id(text): text
        for entry in inventory.metadata.values()
        if entry[2] is not None
        for text in entry[2]
    }
    snapshotted = sum(1 for entry in inventory.metadata.values() if entry[2] is not None)
    return len(texts), snapshotted, all(type(text) is str for text in texts.values())


def test_a_text_subclass_alias_shared_by_many_classes_is_copied_once_per_call() -> None:
    # The copies grew with the classes (4 -> 4, 8 -> 8): each class snapshot made and kept its own.
    for classes in (4, 8):
        distinct, snapshotted, exact = _alias_texts_seen(classes)
        assert snapshotted == classes and exact
        assert distinct == 1


def _refused_key_text_checks(classes: int) -> int:
    """How many times the metadata reads look at one long ``str``-subclass configuration key shared by
    ``classes`` model classes (each check of it as text preceded a full copy of it)."""

    import contracts_common

    key = _SharedAlias("c" * 4096)
    models = [type(f"Refused{index}", (BaseModel,), {"__annotations__": {"x": int}, "x": 0}) for index in range(classes)]
    for model in models:
        dict.__setitem__(vars(model)["model_config"], key, True)
    checks = [0]
    real = contracts_common._is_subclass

    def counting(kind, bases):
        if kind is _SharedAlias:
            checks[0] += 1
        return real(kind, bases)

    contracts_common._is_subclass = counting
    try:
        with pytest.raises(ValueError):
            snapshot_contract_graph([model() for model in models])
    finally:
        contracts_common._is_subclass = real
    return checks[0]


def test_a_key_that_refuses_the_metadata_is_not_copied_as_text() -> None:
    # Each class copied the shared refused key (4 -> 4, 8 -> 8) before refusing its metadata.
    for classes in (4, 8):
        assert _refused_key_text_checks(classes) == 0
