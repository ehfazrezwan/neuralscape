"""Nested body contracts separating readable content from opaque envelopes.

The ``encrypted`` tag records only that the body is represented by an opaque
envelope identifier.  It does not assert that an envelope exists, that it is
well formed, or that any cryptographic property has been verified.

Body variants inherit the schema version of their enclosing versioned record;
they are not standalone wire messages.  :func:`parse_memory_body` validates
only this nested value and revalidates model instances, including unchecked
copies, before they cross the body boundary.
"""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import ConfigDict, Field, TypeAdapter

from contracts_common import ContractModel, OpaqueId


class PlaintextBody(ContractModel):
    """Readable body content available inside its permitted boundary."""

    model_config = ConfigDict(frozen=True, revalidate_instances="always")

    kind: Literal["plaintext"]
    text: Annotated[str, Field(min_length=1)]


class OpaqueEnvelopeBody(ContractModel):
    """Reference to an envelope whose contents are not represented here."""

    model_config = ConfigDict(frozen=True, revalidate_instances="always")

    kind: Literal["encrypted"]
    envelope_id: OpaqueId


MemoryBody = Annotated[
    Union[PlaintextBody, OpaqueEnvelopeBody],
    Field(discriminator="kind"),
]
"""Tagged body value; plaintext and opaque-envelope fields cannot be mixed."""


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
