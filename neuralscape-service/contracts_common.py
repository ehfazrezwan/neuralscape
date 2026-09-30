"""Strict primitives shared by candidate product contracts.

Contract models reject unknown fields and coercion when validating Python
objects.  JSON validation accepts only the corresponding JSON scalar types.
``model_dump()`` is the native Python representation; ``model_dump_json()``
is the wire representation.  Neither representation grants identity or
authority merely because a value validates.
"""

from collections.abc import Mapping
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


def snapshot_contract_graph(
    value: object, active_containers: set[int] | None = None
) -> object:
    """Reconstruct stored contract graphs while rejecting structural defects.

    Malformed extra storage, declared/extra overlap, and cycles are rejected.
    This is neither schema validation nor authorization: callers must strictly
    reconstruct the intended target model before trusting the result.
    """

    if not isinstance(value, (BaseModel, tuple, list, dict)):
        return value

    if active_containers is None:
        active_containers = set()
    value_id = id(value)
    if value_id in active_containers:
        raise ValueError("cyclic contract input is not supported")
    active_containers.add(value_id)
    try:
        if isinstance(value, BaseModel):
            model_type = type(value)
            stored_entries = tuple(vars(value).items())
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
            if unknown_stored_names & alias_roots or missing_owner_roots:
                raise ValueError(
                    "contract input contains conflicting declared and extra fields"
                )
            fields = {
                key: snapshot_contract_graph(item, active_containers)
                for key, item in stored_entries
            }
            extra = getattr(value, "__pydantic_extra__", None)
            if extra is not None:
                if not isinstance(extra, Mapping):
                    raise ValueError("contract extra storage must be a mapping")
                extra_entries = tuple(extra.items())
                extra_keys = {key for key, _ in extra_entries}
                reserved_names = (
                    fields.keys()
                    | declared_names
                    | alias_roots
                )
                if reserved_names & extra_keys:
                    raise ValueError(
                        "contract input contains conflicting declared and extra fields"
                    )
                fields.update(
                    {
                        key: snapshot_contract_graph(item, active_containers)
                        for key, item in extra_entries
                    }
                )
            return fields
        if isinstance(value, tuple):
            return tuple(
                snapshot_contract_graph(item, active_containers) for item in value
            )
        if isinstance(value, list):
            return [
                snapshot_contract_graph(item, active_containers) for item in value
            ]
        return {
            key: snapshot_contract_graph(item, active_containers)
            for key, item in value.items()
        }
    finally:
        active_containers.remove(value_id)


__all__ = [
    "ContractModel",
    "MAX_SAFE_INTEGER",
    "OpaqueId",
    "SafeCounter",
    "VersionedContract",
    "snapshot_contract_graph",
]
