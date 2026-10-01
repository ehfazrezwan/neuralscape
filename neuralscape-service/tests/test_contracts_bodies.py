"""Tests for the plaintext/opaque body boundary."""

from collections.abc import Iterator, Mapping

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


class _HiddenBackingDict(dict[str, object]):
    def __init__(self, values: dict[str, object]) -> None:
        super().__init__(values)
        self.view_calls = 0

    def __len__(self) -> int:
        self.view_calls += 1
        return 0

    def __bool__(self) -> bool:
        self.view_calls += 1
        return False

    def __iter__(self) -> Iterator[str]:
        self.view_calls += 1
        return iter(())

    def keys(self):
        self.view_calls += 1
        return {}.keys()

    def items(self):
        self.view_calls += 1
        return {}.items()


class _IterationVisibleItemsEmptyMapping(Mapping[str, object]):
    def __init__(self) -> None:
        self.items_calls = 0

    def __getitem__(self, key: str) -> object:
        if key == "future_constraint":
            return "deny"
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return iter(("future_constraint",))

    def __len__(self) -> int:
        return 1

    def items(self):
        self.items_calls += 1
        return {}.items()


class _SpoofedDictMapping(_IterationVisibleItemsEmptyMapping):
    def __init__(self) -> None:
        super().__init__()
        self.class_reads = 0

    @property
    def __class__(self):
        self.class_reads += 1
        return dict


class _RaisingClassHiddenDict(_HiddenBackingDict):
    def __init__(self, values: dict[str, object]) -> None:
        super().__init__(values)
        self.class_reads = 0

    @property
    def __class__(self):
        self.class_reads += 1
        raise AssertionError("__class__ must not be read")


def _attach_extra_storage(body: MemoryBody, storage: object) -> MemoryBody:
    object.__setattr__(body, "__pydantic_extra__", storage)
    return body


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
    ("body", "backing"),
    [
        (
            PlaintextBody(kind="plaintext", text="secret"),
            {"future_constraint": "deny"},
        ),
        (
            PlaintextBody(kind="plaintext", text="secret"),
            {"text": "replacement"},
        ),
        (
            OpaqueEnvelopeBody(kind="opaque_envelope", envelope_id="env-1"),
            {"future_constraint": "deny"},
        ),
        (
            OpaqueEnvelopeBody(kind="opaque_envelope", envelope_id="env-1"),
            {"envelope_id": "replacement"},
        ),
    ],
)
def test_parser_rejects_hidden_native_dict_backing(
    body: MemoryBody,
    backing: dict[str, object],
) -> None:
    hidden = _HiddenBackingDict(backing)
    _attach_extra_storage(body, hidden)

    with pytest.raises(ValidationError, match="extra"):
        parse_memory_body(body)

    assert hidden.view_calls == 0


def test_parser_extra_storage_controls_preserve_existing_behavior() -> None:
    empty = _attach_extra_storage(
        PlaintextBody(kind="plaintext", text="secret"),
        {},
    )
    parsed = parse_memory_body(empty)
    assert isinstance(parsed, PlaintextBody)
    assert parsed.text == "secret"

    ordinary = _attach_extra_storage(
        PlaintextBody(kind="plaintext", text="secret"),
        {"future_constraint": "deny"},
    )
    with pytest.raises(ValidationError, match="extra"):
        parse_memory_body(ordinary)

    inverse = _IterationVisibleItemsEmptyMapping()
    inverted = _attach_extra_storage(
        PlaintextBody(kind="plaintext", text="secret"),
        inverse,
    )
    with pytest.raises(ValidationError):
        parse_memory_body(inverted)
    assert inverse.items_calls == 0


def test_parser_uses_concrete_extra_storage_type() -> None:
    spoofed = _SpoofedDictMapping()
    spoofed_body = _attach_extra_storage(
        PlaintextBody(kind="plaintext", text="secret"),
        spoofed,
    )
    with pytest.raises(ValidationError):
        parse_memory_body(spoofed_body)
    assert spoofed.class_reads == 1

    hidden = _RaisingClassHiddenDict({"future_constraint": "deny"})
    hidden_body = _attach_extra_storage(
        PlaintextBody(kind="plaintext", text="secret"),
        hidden,
    )
    with pytest.raises(ValidationError, match="extra"):
        parse_memory_body(hidden_body)
    assert hidden.class_reads == 0
    assert hidden.view_calls == 0


def test_parser_malformed_construct_with_hidden_backing_is_canonical() -> None:
    malformed = PlaintextBody.model_construct(kind="plaintext")
    _attach_extra_storage(malformed, {"future_constraint": "deny"})

    with pytest.raises(ValidationError) as exc:
        parse_memory_body(malformed)

    assert any(error["loc"][-1] == "text" for error in exc.value.errors())


def test_parser_declared_subclass_field_remains_invalid_at_union_boundary() -> None:
    class SpecializedPlaintextBody(PlaintextBody):
        declared_note: str

    body = SpecializedPlaintextBody(
        kind="plaintext",
        text="secret",
        declared_note="subclass-only",
    )

    with pytest.raises(ValidationError, match="extra"):
        parse_memory_body(body)


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
