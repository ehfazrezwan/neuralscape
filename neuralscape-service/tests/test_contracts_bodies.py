"""Tests for the plaintext/opaque body boundary."""

import pytest
from pydantic import ValidationError

from contracts_bodies import (
    MEMORY_BODY_ADAPTER,
    OpaqueEnvelopeBody,
    PlaintextBody,
    parse_memory_body,
)


def test_plaintext_body_preserves_exact_text() -> None:
    body = parse_memory_body({"kind": "plaintext", "text": "  exact text  "})

    assert isinstance(body, PlaintextBody)
    assert body.text == "  exact text  "
    assert body.model_dump(mode="json") == {
        "kind": "plaintext",
        "text": "  exact text  ",
    }


def test_encrypted_body_is_only_an_opaque_reference() -> None:
    body = parse_memory_body(
        {"kind": "encrypted", "envelope_id": "envelope/customer-value"}
    )

    assert isinstance(body, OpaqueEnvelopeBody)
    assert body.envelope_id == "envelope/customer-value"
    assert set(body.model_dump()) == {"kind", "envelope_id"}


@pytest.mark.parametrize(
    "mixed_body",
    [
        {"kind": "plaintext", "text": "readable", "envelope_id": "env-1"},
        {"kind": "encrypted", "envelope_id": "env-1", "text": "leak"},
    ],
)
def test_mixed_body_modes_are_rejected(mixed_body: dict[str, object]) -> None:
    with pytest.raises(ValidationError, match="extra"):
        parse_memory_body(mixed_body)


@pytest.mark.parametrize("kind", ["opaque", "ciphertext", "unknown", ""])
def test_unsupported_body_modes_are_rejected(kind: str) -> None:
    with pytest.raises(ValidationError):
        parse_memory_body({"kind": kind, "envelope_id": "env-1"})


@pytest.mark.parametrize(
    "body",
    [
        {"kind": "plaintext", "text": ""},
        {"kind": "encrypted", "envelope_id": ""},
        {"kind": "encrypted"},
        {"text": "missing discriminator"},
    ],
)
def test_incomplete_body_values_are_rejected(body: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        parse_memory_body(body)


def test_body_schema_has_a_closed_discriminated_union() -> None:
    schema = MEMORY_BODY_ADAPTER.json_schema()

    assert set(schema["discriminator"]["mapping"]) == {"encrypted", "plaintext"}
    assert len(schema["oneOf"]) == 2
    for definition in schema["$defs"].values():
        assert definition["additionalProperties"] is False
