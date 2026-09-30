"""Strict primitives shared by candidate product contracts.

Contract models reject unknown fields and coercion when validating Python
objects.  JSON validation accepts only the corresponding JSON scalar types.
``model_dump()`` is the native Python representation; ``model_dump_json()``
is the wire representation.  Neither representation grants identity or
authority merely because a value validates.
"""

from collections.abc import Mapping
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints


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
            fields = {
                key: snapshot_contract_graph(item, active_containers)
                for key, item in vars(value).items()
            }
            extra = getattr(value, "__pydantic_extra__", None)
            if extra is not None:
                if not isinstance(extra, Mapping):
                    raise ValueError("contract extra storage must be a mapping")
                declared_fields = type(value).model_fields.keys()
                if (fields.keys() | declared_fields) & extra.keys():
                    raise ValueError(
                        "contract input contains conflicting declared and extra fields"
                    )
                fields.update(
                    {
                        key: snapshot_contract_graph(item, active_containers)
                        for key, item in extra.items()
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
