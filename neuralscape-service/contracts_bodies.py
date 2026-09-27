"""Body contracts that keep readable content separate from opaque envelopes.

The ``encrypted`` tag records only that the body is represented by an opaque
envelope identifier.  It does not assert that an envelope exists, that it is
well formed, or that any cryptographic property has been verified.
"""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import Field, TypeAdapter

from contracts_common import ContractModel, OpaqueId


class PlaintextBody(ContractModel):
    """Readable body content available inside its permitted boundary."""

    kind: Literal["plaintext"]
    text: Annotated[str, Field(min_length=1)]


class OpaqueEnvelopeBody(ContractModel):
    """Reference to an envelope whose contents are not represented here."""

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
    """Validate an untrusted native value as exactly one supported body mode."""

    return MEMORY_BODY_ADAPTER.validate_python(value)


__all__ = [
    "MEMORY_BODY_ADAPTER",
    "MemoryBody",
    "OpaqueEnvelopeBody",
    "PlaintextBody",
    "parse_memory_body",
]
