"""Tests for processing-policy and plaintext-recipient boundaries."""

from collections.abc import Iterator, Mapping
from typing import ClassVar

import pytest
from pydantic import BaseModel, ValidationError

from contracts_processing import (
    PlaintextDispatchRejected,
    ProcessingPolicy,
    validate_plaintext_dispatch,
)


_BASE_MODEL_DICT_DESCRIPTOR = vars(BaseModel)["__dict__"]
_BASE_MODEL_EXTRA_DESCRIPTOR = vars(BaseModel)["__pydantic_extra__"]


class _HiddenBackingDict(dict[str, object]):
    """A real dict backing whose overridable views all claim to be empty."""

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
    """Non-dict control retaining the existing iteration-authoritative path."""

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


class _EmptyNativeRaisingViews(dict[str, object]):
    """Actually empty native storage whose overridden views must stay unused."""

    def __init__(self) -> None:
        super().__init__()
        self.view_calls = 0

    def _called(self):
        self.view_calls += 1
        raise AssertionError("overridden dict view must not be read")

    def __len__(self) -> int:
        return self._called()

    def __bool__(self) -> bool:
        return self._called()

    def __iter__(self) -> Iterator[str]:
        return self._called()

    def __getitem__(self, key: str) -> object:
        return self._called()

    def keys(self):
        return self._called()

    def items(self):
        return self._called()


class _EmptyNativeFabricatedViews(dict[str, object]):
    """Actually empty native storage with fabricated nonempty public views."""

    def __init__(self) -> None:
        super().__init__()
        self.view_calls = 0

    def __len__(self) -> int:
        self.view_calls += 1
        return 1

    def __bool__(self) -> bool:
        self.view_calls += 1
        return True

    def __iter__(self) -> Iterator[str]:
        self.view_calls += 1
        return iter(("fabricated_unknown",))

    def __getitem__(self, key: str) -> object:
        self.view_calls += 1
        if key == "fabricated_unknown":
            return True
        raise KeyError(key)

    def keys(self):
        self.view_calls += 1
        return {"fabricated_unknown": True}.keys()

    def items(self):
        self.view_calls += 1
        return {"fabricated_unknown": True}.items()


class _AttributeHidingPolicy(ProcessingPolicy):
    mode_reads: ClassVar[int] = 0

    def __getattribute__(self, name: str):
        if name == "mode":
            type(self).mode_reads += 1
            return "strict_local"
        return super().__getattribute__(name)


class _DescriptorMaskedPolicy(ProcessingPolicy):
    @property
    def __dict__(self):
        native = _BASE_MODEL_DICT_DESCRIPTOR.__get__(self, BaseModel)
        return {
            name: value
            for name, value in dict.items(native)
            if name != "future_constraint"
        }

    @__dict__.setter
    def __dict__(self, value):
        _BASE_MODEL_DICT_DESCRIPTOR.__set__(self, value)

    @property
    def __pydantic_extra__(self):
        return None

    @__pydantic_extra__.setter
    def __pydantic_extra__(self, value):
        _BASE_MODEL_EXTRA_DESCRIPTOR.__set__(self, value)


def _replace_model_backing(
    policy: ProcessingPolicy,
    changes: dict[str, object],
) -> _HiddenBackingDict:
    stored = dict(
        dict.items(object.__getattribute__(policy, "__dict__"))
    )
    for name, value in changes.items():
        dict.__setitem__(stored, name, value)
    hidden = _HiddenBackingDict(stored)
    object.__setattr__(policy, "__dict__", hidden)
    return hidden


def _attach_extra_storage(
    policy: ProcessingPolicy,
    storage: object,
) -> ProcessingPolicy:
    object.__setattr__(policy, "__pydantic_extra__", storage)
    return policy


def _strict_local_policy(**changes: object) -> ProcessingPolicy:
    values: dict[str, object] = {
        "schema_version": "candidate-v1",
        "mode": "strict_local",
        "allowed_execution_locations": ("endpoint",),
        "approved_recipient_ids": (),
        "fallback_policy": "deny",
        "policy_epoch": 7,
    }
    values.update(changes)
    return ProcessingPolicy.model_validate(values)


def _external_policy(**changes: object) -> ProcessingPolicy:
    values: dict[str, object] = {
        "schema_version": "candidate-v1",
        "mode": "operator_trusted",
        "allowed_execution_locations": ("endpoint", "external_provider"),
        "approved_recipient_ids": ("provider-a",),
        "fallback_policy": "within_approved_recipients",
        "policy_epoch": 7,
    }
    values.update(changes)
    return ProcessingPolicy.model_validate(values)


def test_strict_local_accepts_only_local_primary_dispatch() -> None:
    policy = _strict_local_policy()

    assert (
        validate_plaintext_dispatch(
            policy,
            execution_location="endpoint",
            recipient_id=None,
            current_policy_epoch=7,
            currently_authorized_recipient_ids=set(),
            is_fallback=False,
        )
        is None
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"allowed_execution_locations": ("endpoint", "external_provider")},
        {"allowed_execution_locations": ("external_provider",)},
        {"approved_recipient_ids": ("provider-a",)},
        {"fallback_policy": "within_approved_recipients"},
    ],
)
def test_strict_local_rejects_external_or_fallback_policy(
    changes: dict[str, object],
) -> None:
    with pytest.raises(ValidationError, match="strict_local|recipient fallback"):
        _strict_local_policy(**changes)


def test_non_endpoint_policy_requires_explicit_recipient() -> None:
    with pytest.raises(ValidationError, match="approved recipient"):
        _external_policy(approved_recipient_ids=())


def test_recipient_without_non_endpoint_location_is_rejected() -> None:
    with pytest.raises(ValidationError, match="non-endpoint execution location"):
        _external_policy(allowed_execution_locations=("endpoint",))


def test_operator_blind_policy_cannot_select_operator_worker() -> None:
    with pytest.raises(ValidationError, match="operator_worker"):
        _external_policy(
            mode="operator_blind",
            allowed_execution_locations=("operator_worker",),
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("allowed_execution_locations", ("endpoint", "endpoint")),
        ("approved_recipient_ids", ("provider-a", "provider-a")),
    ],
)
def test_policy_rejects_duplicate_boundary_entries(field: str, value: object) -> None:
    with pytest.raises(ValidationError, match="unique"):
        _external_policy(**{field: value})


def test_policy_rejects_unknown_fields_and_boolean_epoch() -> None:
    with pytest.raises(ValidationError):
        _strict_local_policy(policy_epoch=True)
    with pytest.raises(ValidationError, match="extra"):
        _strict_local_policy(read_access_implies_processing=True)


def test_policy_requires_explicit_candidate_version() -> None:
    policy = _strict_local_policy()
    values = policy.model_dump(mode="python")
    values.pop("schema_version")

    assert policy.schema_version == "candidate-v1"
    assert policy.model_json_schema()["properties"]["schema_version"]["const"] == (
        "candidate-v1"
    )
    assert "schema_version" in policy.model_json_schema()["required"]
    with pytest.raises(ValidationError):
        ProcessingPolicy.model_validate(values)
    with pytest.raises(ValidationError):
        _strict_local_policy(schema_version=None)
    with pytest.raises(ValidationError):
        _strict_local_policy(schema_version="future-v2")


def test_policy_and_boundary_collections_are_frozen() -> None:
    policy = _strict_local_policy()

    with pytest.raises(ValidationError, match="frozen"):
        policy.mode = "operator_trusted"  # type: ignore[misc]
    with pytest.raises(AttributeError):
        policy.allowed_execution_locations.append("external_provider")  # type: ignore[attr-defined]
    with pytest.raises(AttributeError):
        policy.approved_recipient_ids.append("provider-a")  # type: ignore[attr-defined]


def test_dispatch_revalidates_unchecked_strict_local_copy() -> None:
    unchecked = _strict_local_policy().model_copy(
        update={
            "allowed_execution_locations": ("endpoint", "external_provider"),
            "approved_recipient_ids": ("provider-a",),
        }
    )

    with pytest.raises(PlaintextDispatchRejected, match="policy is invalid"):
        validate_plaintext_dispatch(
            unchecked,
            execution_location="external_provider",
            recipient_id="provider-a",
            current_policy_epoch=7,
            currently_authorized_recipient_ids={"provider-a"},
            is_fallback=False,
        )


@pytest.mark.parametrize(
    "backing",
    [
        {"future_constraint": "deny"},
        {"mode": "operator_trusted"},
    ],
)
def test_dispatch_rejects_hidden_native_dict_backing(
    backing: dict[str, object],
) -> None:
    hidden = _HiddenBackingDict(backing)
    policy = _attach_extra_storage(_strict_local_policy(), hidden)

    with pytest.raises(PlaintextDispatchRejected, match="policy is invalid"):
        validate_plaintext_dispatch(
            policy,
            execution_location="endpoint",
            recipient_id=None,
            current_policy_epoch=7,
            currently_authorized_recipient_ids=set(),
            is_fallback=False,
        )

    assert hidden.view_calls == 0


@pytest.mark.parametrize(
    "extra_storage",
    [None, {}],
    ids=["none-extras", "empty-dict-extras"],
)
def test_dispatch_projects_hidden_native_model_backing(
    extra_storage: object,
) -> None:
    policy = _strict_local_policy()
    hidden = _replace_model_backing(policy, {"hidden_unknown": "deny"})
    _attach_extra_storage(policy, extra_storage)

    with pytest.raises(PlaintextDispatchRejected, match="policy is invalid") as exc:
        validate_plaintext_dispatch(
            policy,
            execution_location="endpoint",
            recipient_id=None,
            current_policy_epoch=7,
            currently_authorized_recipient_ids=set(),
            is_fallback=False,
        )

    assert isinstance(exc.value.__cause__, ValidationError)
    assert any(
        error["loc"] == ("hidden_unknown",)
        for error in exc.value.__cause__.errors()
    )
    assert hidden.view_calls == 0


@pytest.mark.parametrize(
    "extra_storage",
    [None, {}],
    ids=["none-extras", "empty-dict-extras"],
)
def test_dispatch_native_model_backing_ignores_attribute_override(
    extra_storage: object,
) -> None:
    policy = _AttributeHidingPolicy(
        schema_version="candidate-v1",
        mode="strict_local",
        allowed_execution_locations=("endpoint",),
        approved_recipient_ids=(),
        fallback_policy="deny",
        policy_epoch=7,
    )
    _AttributeHidingPolicy.mode_reads = 0
    hidden = _replace_model_backing(policy, {"mode": "invalid"})
    _attach_extra_storage(policy, extra_storage)

    with pytest.raises(PlaintextDispatchRejected, match="policy is invalid") as exc:
        validate_plaintext_dispatch(
            policy,
            execution_location="endpoint",
            recipient_id=None,
            current_policy_epoch=7,
            currently_authorized_recipient_ids=set(),
            is_fallback=False,
        )

    assert isinstance(exc.value.__cause__, ValidationError)
    assert any(error["loc"] == ("mode",) for error in exc.value.__cause__.errors())
    assert _AttributeHidingPolicy.mode_reads == 0
    assert hidden.view_calls == 0


@pytest.mark.parametrize("inventory", ["stored", "extra"])
def test_dispatch_uses_base_model_storage_descriptors(
    inventory: str,
) -> None:
    ordinary = _strict_local_policy()
    masked = _DescriptorMaskedPolicy(
        schema_version="candidate-v1",
        mode="strict_local",
        allowed_execution_locations=("endpoint",),
        approved_recipient_ids=(),
        fallback_policy="deny",
        policy_epoch=7,
    )
    hidden_stores: list[_HiddenBackingDict] = []
    for policy in (ordinary, masked):
        if inventory == "stored":
            stored = dict(
                dict.items(_BASE_MODEL_DICT_DESCRIPTOR.__get__(policy, BaseModel))
            )
            stored["future_constraint"] = "deny"
            hidden = _HiddenBackingDict(stored)
            _BASE_MODEL_DICT_DESCRIPTOR.__set__(policy, hidden)
        else:
            hidden = _HiddenBackingDict({"future_constraint": "deny"})
            _BASE_MODEL_EXTRA_DESCRIPTOR.__set__(policy, hidden)
        hidden_stores.append(hidden)

    diagnostics = []
    for policy in (ordinary, masked):
        with pytest.raises(
            PlaintextDispatchRejected,
            match="policy is invalid",
        ) as exc:
            validate_plaintext_dispatch(
                policy,
                execution_location="endpoint",
                recipient_id=None,
                current_policy_epoch=7,
                currently_authorized_recipient_ids=set(),
                is_fallback=False,
            )
        assert isinstance(exc.value.__cause__, ValidationError)
        diagnostics.append(
            [
                (error["loc"], error["type"])
                for error in exc.value.__cause__.errors()
            ]
        )

    assert diagnostics[0] == diagnostics[1]
    assert all(hidden.view_calls == 0 for hidden in hidden_stores)


def test_dispatch_descriptor_masked_valid_and_absent_extra_controls() -> None:
    valid = _DescriptorMaskedPolicy(
        schema_version="candidate-v1",
        mode="strict_local",
        allowed_execution_locations=("endpoint",),
        approved_recipient_ids=(),
        fallback_policy="deny",
        policy_epoch=7,
    )
    validate_plaintext_dispatch(
        valid,
        execution_location="endpoint",
        recipient_id=None,
        current_policy_epoch=7,
        currently_authorized_recipient_ids=set(),
        is_fallback=False,
    )

    _BASE_MODEL_EXTRA_DESCRIPTOR.__delete__(valid)
    with pytest.raises(AttributeError, match="__pydantic_extra__"):
        validate_plaintext_dispatch(
            valid,
            execution_location="endpoint",
            recipient_id=None,
            current_policy_epoch=7,
            currently_authorized_recipient_ids=set(),
            is_fallback=False,
        )


def test_dispatch_extra_storage_controls_preserve_existing_behavior() -> None:
    empty = _attach_extra_storage(_strict_local_policy(), {})
    validate_plaintext_dispatch(
        empty,
        execution_location="endpoint",
        recipient_id=None,
        current_policy_epoch=7,
        currently_authorized_recipient_ids=set(),
        is_fallback=False,
    )

    ordinary = _attach_extra_storage(
        _strict_local_policy(),
        {"future_constraint": "deny"},
    )
    with pytest.raises(PlaintextDispatchRejected, match="policy is invalid"):
        validate_plaintext_dispatch(
            ordinary,
            execution_location="endpoint",
            recipient_id=None,
            current_policy_epoch=7,
            currently_authorized_recipient_ids=set(),
            is_fallback=False,
        )

    inverse = _IterationVisibleItemsEmptyMapping()
    inverted = _attach_extra_storage(_strict_local_policy(), inverse)
    with pytest.raises(PlaintextDispatchRejected, match="policy is invalid"):
        validate_plaintext_dispatch(
            inverted,
            execution_location="endpoint",
            recipient_id=None,
            current_policy_epoch=7,
            currently_authorized_recipient_ids=set(),
            is_fallback=False,
        )
    assert inverse.items_calls == 0


@pytest.mark.parametrize(
    "extra_type",
    [_EmptyNativeRaisingViews, _EmptyNativeFabricatedViews],
    ids=["raising-views", "fabricated-views"],
)
def test_dispatch_accepts_empty_native_backing_without_overridden_views(
    extra_type: type[_EmptyNativeRaisingViews | _EmptyNativeFabricatedViews],
) -> None:
    extras = extra_type()
    policy = _attach_extra_storage(_strict_local_policy(), extras)

    validate_plaintext_dispatch(
        policy,
        execution_location="endpoint",
        recipient_id=None,
        current_policy_epoch=7,
        currently_authorized_recipient_ids=set(),
        is_fallback=False,
    )

    assert tuple(dict.items(extras)) == ()
    assert extras.view_calls == 0


def test_dispatch_empty_native_projection_preserves_injected_unknown_field() -> None:
    policy = _strict_local_policy()
    object.__setattr__(policy, "injected_unknown", "deny")
    extras = _EmptyNativeRaisingViews()
    _attach_extra_storage(policy, extras)

    with pytest.raises(PlaintextDispatchRejected, match="policy is invalid") as exc:
        validate_plaintext_dispatch(
            policy,
            execution_location="endpoint",
            recipient_id=None,
            current_policy_epoch=7,
            currently_authorized_recipient_ids=set(),
            is_fallback=False,
        )

    assert isinstance(exc.value.__cause__, ValidationError)
    assert any(
        error["loc"] == ("injected_unknown",)
        for error in exc.value.__cause__.errors()
    )
    assert extras.view_calls == 0


def test_dispatch_empty_native_projection_preserves_subclass_field() -> None:
    class SpecializedPolicy(ProcessingPolicy):
        declared_note: str

    policy = SpecializedPolicy(
        schema_version="candidate-v1",
        mode="strict_local",
        allowed_execution_locations=("endpoint",),
        approved_recipient_ids=(),
        fallback_policy="deny",
        policy_epoch=7,
        declared_note="subclass-only",
    )
    extras = _EmptyNativeRaisingViews()
    _attach_extra_storage(policy, extras)

    with pytest.raises(PlaintextDispatchRejected, match="policy is invalid") as exc:
        validate_plaintext_dispatch(
            policy,
            execution_location="endpoint",
            recipient_id=None,
            current_policy_epoch=7,
            currently_authorized_recipient_ids=set(),
            is_fallback=False,
        )

    assert isinstance(exc.value.__cause__, ValidationError)
    assert any(
        error["loc"] == ("declared_note",)
        for error in exc.value.__cause__.errors()
    )
    assert extras.view_calls == 0


def test_dispatch_empty_native_projection_preserves_missing_fields() -> None:
    policy = ProcessingPolicy.model_construct(schema_version="candidate-v1")
    extras = _EmptyNativeRaisingViews()
    _attach_extra_storage(policy, extras)

    with pytest.raises(PlaintextDispatchRejected, match="policy is invalid") as exc:
        validate_plaintext_dispatch(
            policy,
            execution_location="endpoint",
            recipient_id=None,
            current_policy_epoch=7,
            currently_authorized_recipient_ids=set(),
            is_fallback=False,
        )

    assert isinstance(exc.value.__cause__, ValidationError)
    missing_locations = {
        error["loc"] for error in exc.value.__cause__.errors()
    }
    assert ("mode",) in missing_locations
    assert ("policy_epoch",) in missing_locations
    assert extras.view_calls == 0


def test_dispatch_uses_concrete_extra_storage_type() -> None:
    spoofed = _SpoofedDictMapping()
    spoofed_policy = _attach_extra_storage(_strict_local_policy(), spoofed)
    with pytest.raises(
        PlaintextDispatchRejected,
        match="policy is invalid",
    ) as exc:
        validate_plaintext_dispatch(
            spoofed_policy,
            execution_location="endpoint",
            recipient_id=None,
            current_policy_epoch=7,
            currently_authorized_recipient_ids=set(),
            is_fallback=False,
        )
    assert isinstance(exc.value.__cause__, ValidationError)
    assert spoofed.class_reads == 1

    hidden = _RaisingClassHiddenDict({"future_constraint": "deny"})
    hidden_policy = _attach_extra_storage(_strict_local_policy(), hidden)
    with pytest.raises(PlaintextDispatchRejected, match="policy is invalid"):
        validate_plaintext_dispatch(
            hidden_policy,
            execution_location="endpoint",
            recipient_id=None,
            current_policy_epoch=7,
            currently_authorized_recipient_ids=set(),
            is_fallback=False,
        )
    assert hidden.class_reads == 0
    assert hidden.view_calls == 0


def test_dispatch_malformed_construct_with_hidden_backing_is_canonical() -> None:
    malformed = ProcessingPolicy.model_construct(schema_version="candidate-v1")
    _attach_extra_storage(malformed, {"future_constraint": "deny"})

    with pytest.raises(PlaintextDispatchRejected, match="policy is invalid") as exc:
        validate_plaintext_dispatch(
            malformed,
            execution_location="endpoint",
            recipient_id=None,
            current_policy_epoch=7,
            currently_authorized_recipient_ids=set(),
            is_fallback=False,
        )

    assert isinstance(exc.value.__cause__, ValidationError)
    assert any(error["loc"] == ("mode",) for error in exc.value.__cause__.errors())


def test_dispatch_declared_subclass_field_remains_invalid_at_base_boundary() -> None:
    class SpecializedPolicy(ProcessingPolicy):
        declared_note: str

    policy = SpecializedPolicy(
        schema_version="candidate-v1",
        mode="strict_local",
        allowed_execution_locations=("endpoint",),
        approved_recipient_ids=(),
        fallback_policy="deny",
        policy_epoch=7,
        declared_note="subclass-only",
    )

    with pytest.raises(PlaintextDispatchRejected, match="policy is invalid"):
        validate_plaintext_dispatch(
            policy,
            execution_location="endpoint",
            recipient_id=None,
            current_policy_epoch=7,
            currently_authorized_recipient_ids=set(),
            is_fallback=False,
        )


def test_external_dispatch_requires_policy_and_current_authority() -> None:
    policy = _external_policy()

    validate_plaintext_dispatch(
        policy,
        execution_location="external_provider",
        recipient_id="provider-a",
        current_policy_epoch=7,
        currently_authorized_recipient_ids={"provider-a"},
        is_fallback=False,
    )

    with pytest.raises(PlaintextDispatchRejected, match="current processing authority"):
        validate_plaintext_dispatch(
            policy,
            execution_location="external_provider",
            recipient_id="provider-a",
            current_policy_epoch=7,
            currently_authorized_recipient_ids=set(),
            is_fallback=False,
        )


def test_current_authority_cannot_add_recipient_missing_from_policy() -> None:
    with pytest.raises(PlaintextDispatchRejected, match="not approved by policy"):
        validate_plaintext_dispatch(
            _external_policy(),
            execution_location="external_provider",
            recipient_id="provider-b",
            current_policy_epoch=7,
            currently_authorized_recipient_ids={"provider-a", "provider-b"},
            is_fallback=False,
        )


def test_stale_policy_epoch_fails_closed() -> None:
    with pytest.raises(PlaintextDispatchRejected, match="not current"):
        validate_plaintext_dispatch(
            _external_policy(),
            execution_location="external_provider",
            recipient_id="provider-a",
            current_policy_epoch=8,
            currently_authorized_recipient_ids={"provider-a"},
            is_fallback=False,
        )


def test_unapproved_location_fails_even_for_authorized_recipient() -> None:
    with pytest.raises(PlaintextDispatchRejected, match="location"):
        validate_plaintext_dispatch(
            _external_policy(),
            execution_location="customer_worker",
            recipient_id="provider-a",
            current_policy_epoch=7,
            currently_authorized_recipient_ids={"provider-a"},
            is_fallback=False,
        )


@pytest.mark.parametrize(
    ("execution_location", "recipient_id"),
    [("endpoint", None), ("external_provider", "provider-a")],
)
def test_deny_policy_blocks_fallback_after_primary_failure(
    execution_location: str,
    recipient_id: str | None,
) -> None:
    policy = _external_policy(fallback_policy="deny")

    with pytest.raises(PlaintextDispatchRejected, match="fallback is denied"):
        validate_plaintext_dispatch(
            policy,
            execution_location=execution_location,  # type: ignore[arg-type]
            recipient_id=recipient_id,
            current_policy_epoch=7,
            currently_authorized_recipient_ids={"provider-a"},
            is_fallback=True,
        )


def test_dispatch_requires_explicit_fallback_classification() -> None:
    policy = _external_policy(fallback_policy="deny")

    with pytest.raises(TypeError, match="is_fallback"):
        validate_plaintext_dispatch(  # type: ignore[call-arg]
            policy,
            execution_location="external_provider",
            recipient_id="provider-a",
            current_policy_epoch=7,
            currently_authorized_recipient_ids={"provider-a"},
        )

    assert (
        validate_plaintext_dispatch(
            policy,
            execution_location="external_provider",
            recipient_id="provider-a",
            current_policy_epoch=7,
            currently_authorized_recipient_ids={"provider-a"},
            is_fallback=False,
        )
        is None
    )
    with pytest.raises(PlaintextDispatchRejected, match="fallback is denied"):
        validate_plaintext_dispatch(
            policy,
            execution_location="external_provider",
            recipient_id="provider-a",
            current_policy_epoch=7,
            currently_authorized_recipient_ids={"provider-a"},
            is_fallback=True,
        )


def test_approved_fallback_is_revalidated_against_current_authority() -> None:
    policy = _external_policy(
        approved_recipient_ids=("provider-a", "provider-b"),
    )

    validate_plaintext_dispatch(
        policy,
        execution_location="external_provider",
        recipient_id="provider-b",
        current_policy_epoch=7,
        currently_authorized_recipient_ids={"provider-b"},
        is_fallback=True,
    )


def test_recipient_only_policy_rejects_endpoint_fallback_but_allows_primary() -> None:
    policy = _external_policy()

    assert (
        validate_plaintext_dispatch(
            policy,
            execution_location="endpoint",
            recipient_id=None,
            current_policy_epoch=7,
            currently_authorized_recipient_ids=set(),
            is_fallback=False,
        )
        is None
    )
    with pytest.raises(PlaintextDispatchRejected, match="non-endpoint"):
        validate_plaintext_dispatch(
            policy,
            execution_location="endpoint",
            recipient_id=None,
            current_policy_epoch=7,
            currently_authorized_recipient_ids=set(),
            is_fallback=True,
        )


def test_endpoint_dispatch_cannot_smuggle_a_recipient_identity() -> None:
    with pytest.raises(PlaintextDispatchRejected, match="must not name"):
        validate_plaintext_dispatch(
            _external_policy(),
            execution_location="endpoint",
            recipient_id="provider-a",
            current_policy_epoch=7,
            currently_authorized_recipient_ids={"provider-a"},
            is_fallback=False,
        )
