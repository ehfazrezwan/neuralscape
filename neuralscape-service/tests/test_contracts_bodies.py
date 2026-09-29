"""Tests for the plaintext/opaque body boundary."""

import pytest
from pydantic import ValidationError

from contracts_common import VersionedContract
from contracts_bodies import (
    MEMORY_BODY_ADAPTER,
    MemoryBody,
    OpaqueEnvelopeBody,
    PlaintextBody,
    parse_memory_body,
)


class VersionedBodyOwner(VersionedContract):
    """Synthetic enclosing record that owns its nested body's version."""

    body: MemoryBody


def test_plaintext_body_preserves_exact_text() -> None:
    body = parse_memory_body({"kind": "plaintext", "text": "  exact text  "})

    assert isinstance(body, PlaintextBody)
    assert body.text == "  exact text  "
    assert body.model_dump(mode="json") == {
        "kind": "plaintext",
        "text": "  exact text  ",
    }


def test_opaque_envelope_body_is_only_an_opaque_reference() -> None:
    body = parse_memory_body(
        {"kind": "opaque_envelope", "envelope_id": "envelope/customer-value"}
    )

    assert isinstance(body, OpaqueEnvelopeBody)
    assert body.envelope_id == "envelope/customer-value"
    assert set(body.model_dump()) == {"kind", "envelope_id"}


def test_enclosing_contract_owns_body_version() -> None:
    owner = VersionedBodyOwner.model_validate(
        {
            "schema_version": "candidate-v1",
            "body": {"kind": "opaque_envelope", "envelope_id": "env-1"},
        }
    )

    assert isinstance(owner.body, OpaqueEnvelopeBody)
    assert "schema_version" not in owner.body.model_dump()
    with pytest.raises(ValidationError, match="extra"):
        parse_memory_body(
            {
                "schema_version": "candidate-v1",
                "kind": "opaque_envelope",
                "envelope_id": "env-1",
            }
        )


@pytest.mark.parametrize(
    "body",
    [
        PlaintextBody(kind="plaintext", text="secret"),
        OpaqueEnvelopeBody(kind="opaque_envelope", envelope_id="env-1"),
    ],
)
def test_body_variants_are_frozen(body: MemoryBody) -> None:
    with pytest.raises(ValidationError, match="frozen"):
        body.kind = "unsupported"  # type: ignore[assignment]


def test_parser_revalidates_unchecked_body_copy() -> None:
    body = PlaintextBody(kind="plaintext", text="secret")
    unchecked = body.model_copy(update={"kind": "opaque_envelope"})

    with pytest.raises(ValidationError):
        parse_memory_body(unchecked)


@pytest.mark.parametrize(
    "mixed_body",
    [
        {"kind": "plaintext", "text": "readable", "envelope_id": "env-1"},
        {"kind": "opaque_envelope", "envelope_id": "env-1", "text": "leak"},
    ],
)
def test_mixed_body_modes_are_rejected(mixed_body: dict[str, object]) -> None:
    with pytest.raises(ValidationError, match="extra"):
        parse_memory_body(mixed_body)


@pytest.mark.parametrize(
    "kind", ["encrypted", "opaque", "ciphertext", "unknown", ""]
)
def test_unsupported_body_modes_are_rejected(kind: str) -> None:
    with pytest.raises(ValidationError):
        parse_memory_body({"kind": kind, "envelope_id": "env-1"})


@pytest.mark.parametrize(
    "body",
    [
        {"kind": "plaintext", "text": ""},
        {"kind": "opaque_envelope", "envelope_id": ""},
        {"kind": "opaque_envelope"},
        {"text": "missing discriminator"},
    ],
)
def test_incomplete_body_values_are_rejected(body: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        parse_memory_body(body)


def test_body_schema_has_a_closed_discriminated_union() -> None:
    schema = MEMORY_BODY_ADAPTER.json_schema()

    assert set(schema["discriminator"]["mapping"]) == {
        "opaque_envelope",
        "plaintext",
    }
    assert len(schema["oneOf"]) == 2
    for definition in schema["$defs"].values():
        assert definition["additionalProperties"] is False
