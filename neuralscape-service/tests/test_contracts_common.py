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

    snapshot = snapshot_contract_graph(value)

    if hook == "hash":
        assert stored_name.hash_calls > 0
    else:
        assert stored_name.equality_calls > 0
    assert child.count == 1
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

    snapshot = snapshot_contract_graph(value)

    assert value.count == 1
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
    assert reference_schema["required"] == ["kind", "id", "tenant_id", "resolver"]
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
