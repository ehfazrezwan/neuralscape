"""Strict primitives shared by candidate product contracts.

Contract models reject unknown fields and coercion when validating Python
objects.  JSON validation accepts only the corresponding JSON scalar types.
``model_dump()`` is the native Python representation; ``model_dump_json()``
is the wire representation.  Neither representation grants identity or
authority merely because a value validates.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Annotated, Literal

from pydantic import (
    AliasChoices,
    AliasPath,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
)


MAX_SAFE_INTEGER = 9_007_199_254_740_991

_BASE_MODEL_DICT_DESCRIPTOR = vars(BaseModel)["__dict__"]
_BASE_MODEL_EXTRA_DESCRIPTOR = vars(BaseModel)["__pydantic_extra__"]


OpaqueId = Annotated[
    str,
    StringConstraints(strict=True, min_length=1, max_length=256),
]
"""An opaque, non-empty identifier that is preserved exactly as supplied."""


SafeCounter = Annotated[
    int,
    Field(strict=True, ge=0, le=MAX_SAFE_INTEGER),
]
"""A non-negative integer exactly representable by JavaScript number values."""


class ContractModel(BaseModel):
    """Base for strict contracts with a closed set of fields."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        revalidate_instances="always",
    )


class VersionedContract(ContractModel):
    """A candidate contract whose version must be explicit on every input."""

    schema_version: Literal["candidate-v1"]


def _validation_alias_owners(model_type: type[BaseModel]) -> dict[str, set[str]]:
    if model_type.model_config.get("validate_by_alias") is False:
        return {}

    owners: dict[str, set[str]] = {}
    for field_name, field in model_type.model_fields.items():
        validation_alias = field.validation_alias
        choices = (
            validation_alias.choices
            if isinstance(validation_alias, AliasChoices)
            else (validation_alias,)
        )
        for choice in choices:
            if isinstance(choice, str):
                owners.setdefault(choice, set()).add(field_name)
            elif isinstance(choice, AliasPath):
                owners.setdefault(choice.path[0], set()).add(field_name)
    return owners


def _mapping_entries(
    value: Mapping[object, object],
    value_type: type[object],
) -> tuple[tuple[object, object], ...]:
    if issubclass(value_type, dict):
        return tuple(dict.items(value))
    return tuple(value.items())


@dataclass
class _ModelInventory:
    model_type: type[BaseModel]
    stored_entries: tuple[tuple[object, object], ...]
    extra: object | None
    extra_entries: tuple[tuple[object, object], ...] | None = None
    malformed_extra: bool = False


@dataclass
class _GraphInventory:
    models: dict[int, _ModelInventory] = field(default_factory=dict)
    sequences: dict[int, tuple[object, ...]] = field(default_factory=dict)
    dictionaries: dict[int, tuple[tuple[object, object], ...]] = field(
        default_factory=dict
    )
    visited: set[int] = field(default_factory=set)


def _capture_native_graph(value: object, inventory: _GraphInventory) -> None:
    """Freeze supported native backing without invoking public protocols."""

    value_type = type(value)
    if not issubclass(value_type, (BaseModel, tuple, list, dict)):
        return

    value_id = id(value)
    if value_id in inventory.visited:
        return
    inventory.visited.add(value_id)

    if issubclass(value_type, BaseModel):
        stored = _BASE_MODEL_DICT_DESCRIPTOR.__get__(value, BaseModel)
        stored_entries = tuple(dict.items(stored))
        try:
            extra = _BASE_MODEL_EXTRA_DESCRIPTOR.__get__(value, BaseModel)
        except AttributeError:
            extra = None
        model_inventory = _ModelInventory(
            model_type=value_type,
            stored_entries=stored_entries,
            extra=extra,
        )
        inventory.models[value_id] = model_inventory

        # Capture every model reachable through native stored backing before
        # classifying extras or hashing any stored name.
        for _, item in stored_entries:
            _capture_native_graph(item, inventory)

        if extra is None:
            model_inventory.extra_entries = ()
            return
        extra_type = type(extra)
        if issubclass(extra_type, dict):
            extra_entries = tuple(dict.items(extra))
            model_inventory.extra_entries = extra_entries
            for _, item in extra_entries:
                _capture_native_graph(item, inventory)
        elif not issubclass(extra_type, Mapping):
            model_inventory.extra_entries = ()
            model_inventory.malformed_extra = True
        # A non-dict Mapping retains its established public items protocol.
        # Its inventory is captured later, after all natively reachable model
        # backing has already been frozen.
        return

    if issubclass(value_type, tuple):
        items = tuple(tuple.__iter__(value))
        inventory.sequences[value_id] = items
        for item in items:
            _capture_native_graph(item, inventory)
        return

    if issubclass(value_type, list):
        items = tuple(list.__iter__(value))
        inventory.sequences[value_id] = items
        for item in items:
            _capture_native_graph(item, inventory)
        return

    entries = tuple(dict.items(value))
    inventory.dictionaries[value_id] = entries
    for _, item in entries:
        _capture_native_graph(item, inventory)


def _snapshot_captured_graph(
    value: object,
    active_containers: set[int],
    inventory: _GraphInventory,
) -> object:
    """Reconstruct one graph after its reachable native backing is frozen."""

    value_type = type(value)
    if not issubclass(value_type, (BaseModel, tuple, list, dict)):
        return value

    value_id = id(value)
    if value_id not in inventory.visited:
        _capture_native_graph(value, inventory)
    if value_id in active_containers:
        raise ValueError("cyclic contract input is not supported")
    active_containers.add(value_id)
    try:
        if issubclass(value_type, BaseModel):
            captured = inventory.models[value_id]
            model_type = captured.model_type
            stored_entries = captured.stored_entries
            stored_names = {key for key, _ in stored_entries}
            declared_names = set(model_type.model_fields)
            alias_owners = _validation_alias_owners(model_type)
            alias_roots = set(alias_owners)
            unknown_stored_names = stored_names - declared_names
            missing_owner_roots = {
                root
                for root in stored_names & alias_roots
                if alias_owners[root] - stored_names
            }
            if unknown_stored_names & alias_roots:
                raise ValueError(
                    "contract input contains conflicting declared and extra fields"
                )
            if missing_owner_roots:
                raise ValueError(
                    "contract input contains stored validation alias roots "
                    "with missing owning fields"
                )

            extra_entries = captured.extra_entries
            if extra_entries is None:
                extra_entries = _mapping_entries(
                    captured.extra,  # type: ignore[arg-type]
                    type(captured.extra),
                )
                captured.extra_entries = extra_entries
                for _, item in extra_entries:
                    _capture_native_graph(item, inventory)

            fields = {
                key: _snapshot_captured_graph(item, active_containers, inventory)
                for key, item in stored_entries
            }
            if captured.malformed_extra:
                raise ValueError("contract extra storage must be a mapping")
            if captured.extra is not None:
                extra_keys = {key for key, _ in extra_entries}
                reserved_names = stored_names | declared_names | alias_roots
                if reserved_names & extra_keys:
                    raise ValueError(
                        "contract input contains conflicting declared and extra fields"
                    )
                if len(extra_keys) != len(extra_entries):
                    raise ValueError("contract extra storage contains duplicate keys")
            fields.update(
                {
                    key: _snapshot_captured_graph(
                        item,
                        active_containers,
                        inventory,
                    )
                    for key, item in extra_entries
                }
            )
            return fields

        if issubclass(value_type, tuple):
            items = (
                inventory.sequences[value_id]
                if value_type is tuple
                else tuple(iter(value))
            )
            return tuple(
                _snapshot_captured_graph(item, active_containers, inventory)
                for item in items
            )
        if issubclass(value_type, list):
            items = (
                inventory.sequences[value_id]
                if value_type is list
                else tuple(iter(value))
            )
            return [
                _snapshot_captured_graph(item, active_containers, inventory)
                for item in items
            ]
        entries = inventory.dictionaries[value_id]
        return {
            key: _snapshot_captured_graph(item, active_containers, inventory)
            for key, item in entries
        }
    finally:
        active_containers.remove(value_id)


def snapshot_contract_graph(
    value: object, active_containers: set[int] | None = None
) -> object:
    """Reconstruct stored contract graphs while rejecting structural defects.

    Malformed extra storage, declared/extra overlap, and cycles are rejected.
    This is neither schema validation nor authorization: callers must strictly
    reconstruct the intended target model before trusting the result.
    """

    if active_containers is None:
        active_containers = set()
    inventory = _GraphInventory()
    _capture_native_graph(value, inventory)
    return _snapshot_captured_graph(value, active_containers, inventory)


__all__ = [
    "ContractModel",
    "MAX_SAFE_INTEGER",
    "OpaqueId",
    "SafeCounter",
    "VersionedContract",
    "snapshot_contract_graph",
]
