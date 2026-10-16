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

from pydantic import BeforeValidator, ConfigDict, Field, TypeAdapter

from contracts_common import ContractModel, OpaqueId
from contracts_native import is_subclass, plain_keyed, project_model_storage


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


def _project_body_storage(value: object) -> object:
    """Before-validator: project a body model's native storage before the tag is read.

    Attached to the union itself, so the projection applies on every route into it:
    the shared adapter, the parser, and any contract that nests a ``MemoryBody``.  Only a
    genuine body model (by its native MRO) or an exact dict goes on to Pydantic; any other
    object is refused before its ``__class__`` can be read.
    """

    kind = type(value)
    if is_subclass(kind, (PlaintextBody, OpaqueEnvelopeBody)):
        return project_model_storage(value)
    if kind is dict:
        try:
            return plain_keyed(dict.items(value))
        except TypeError as error:
            raise ValueError(f"a body dict has an unusable key: {error}") from None
    raise ValueError("a body must be a body model or a plain dict")


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
