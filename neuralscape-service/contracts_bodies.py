"""Nested body contracts separating readable content from opaque envelopes.

The ``opaque_envelope`` tag records only that the body is represented by an
opaque envelope identifier.  It does not assert that an envelope exists, that
it is well formed, or that any cryptographic property has been verified.

Body variants inherit the schema version of their enclosing versioned record;
they are not standalone wire messages.  :func:`parse_memory_body` validates
only this nested value and revalidates model instances, including unchecked
copies, before they cross the body boundary.
"""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, TypeAdapter

from contracts_common import ContractModel, OpaqueId


class PlaintextBody(ContractModel):
    """Readable body content available inside its permitted boundary."""

    model_config = ConfigDict(frozen=True, revalidate_instances="always")

    kind: Literal["plaintext"]
    text: Annotated[str, Field(min_length=1)]


class OpaqueEnvelopeBody(ContractModel):
    """Reference to an envelope whose contents are not represented here."""

    model_config = ConfigDict(frozen=True, revalidate_instances="always")

    kind: Literal["opaque_envelope"]
    envelope_id: OpaqueId


_BASE_MODEL_DICT_DESCRIPTOR = vars(BaseModel)["__dict__"]
_BASE_MODEL_EXTRA_DESCRIPTOR = vars(BaseModel)["__pydantic_extra__"]


def _with_native_dict_extra_backing(value: object) -> object:
    """Project a model with native dict extra storage for closed revalidation."""
    value_type = type(value)
    if not issubclass(value_type, BaseModel):
        return value
    extras = _BASE_MODEL_EXTRA_DESCRIPTOR.__get__(value, BaseModel)
    if extras is None:
        captured = ()
    elif issubclass(type(extras), dict):
        # Bypass every overridable view on a dict subclass and capture its
        # actual backing exactly once.
        captured = tuple(dict.items(extras))
    else:
        # Not a dict: none of its views can be trusted and none is read.  Fence it the
        # same way as any other native storage by projecting a marker the closed model
        # rejects, instead of handing the original model (and its masked descriptors)
        # back to revalidation.
        captured = (("unreadable_extra_storage", True),)

    # Detach every pair before projected-dict hashing can run a hostile stored
    # name callback that repairs a later value in the live model backing.
    stored = tuple(
        dict.items(_BASE_MODEL_DICT_DESCRIPTOR.__get__(value, BaseModel))
    )
    projected = dict(stored)
    if captured:
        projected["native_extra_backing"] = dict(captured)
    return projected


def _project_body_storage(value: object) -> object:
    """Before-validator: project a body model's native storage before the tag is read.

    Attached to the union itself, so the projection applies on every route into it:
    the shared adapter, the parser, and any contract that nests a ``MemoryBody``.
    """

    if issubclass(type(value), (PlaintextBody, OpaqueEnvelopeBody)):
        return _with_native_dict_extra_backing(value)
    return value


MemoryBody = Annotated[
    Union[PlaintextBody, OpaqueEnvelopeBody],
    Field(discriminator="kind"),
    BeforeValidator(_project_body_storage),
]
"""Tagged body value; plaintext and opaque-envelope fields cannot be mixed.

Model instances are revalidated from their native storage (not from any overridden
descriptor), before the discriminator is evaluated, wherever this type is used."""


MEMORY_BODY_ADAPTER: TypeAdapter[MemoryBody] = TypeAdapter(MemoryBody)
"""Reusable validator and JSON-schema adapter for :data:`MemoryBody`."""


def parse_memory_body(value: object) -> MemoryBody:
    """Validate one nested body value, including an unchecked model copy.

    The caller remains responsible for validating the enclosing contract and
    its explicit schema version before treating this value as wire data.
    """

    return MEMORY_BODY_ADAPTER.validate_python(value)


__all__ = [
    "MEMORY_BODY_ADAPTER",
    "MemoryBody",
    "OpaqueEnvelopeBody",
    "PlaintextBody",
    "parse_memory_body",
]
