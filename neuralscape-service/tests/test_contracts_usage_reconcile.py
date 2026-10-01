"""Adversarial correction and aggregation tests for usage reconciliation."""

import copy
import gc
import json
import weakref
from collections.abc import Callable, ItemsView, Iterator, KeysView, Mapping
from itertools import permutations
from types import MappingProxyType

import pytest
from pydantic import BaseModel, ValidationError

import contracts_usage_reconcile as usage_reconcile_contracts
from contracts_common import MAX_SAFE_INTEGER
from contracts_usage import AttributionSnapshot, TokenQuantity, TokenUsage, UsageEvent
from contracts_usage_reconcile import (
    ReconciledLedger,
    ReconciledUsageStream,
    UsageReconciliation,
    UsageReconciliationError,
    reconcile_usage_events,
)


_MODEL_DICT_DESCRIPTOR = BaseModel.__dict__["__dict__"]
_MODEL_EXTRAS_DESCRIPTOR = BaseModel.__dict__["__pydantic_extra__"]


def _quantity(value: int | None, reason: str | None = None) -> TokenQuantity:
    return TokenQuantity(value=value, missing_reason=reason)


def _usage(
    *,
    input_tokens: int | None = 10,
    output_tokens: int | None = 5,
    cache_read_tokens: int | None = 4,
    reasoning_tokens: int | None = 2,
    cache_read_inclusion: str = "included_in_input",
    reasoning_inclusion: str = "included_in_output",
) -> TokenUsage:
    return TokenUsage(
        measurement="actual",
        input_tokens=_quantity(
            input_tokens, None if input_tokens is not None else "provider_did_not_report"
        ),
        output_tokens=_quantity(
            output_tokens, None if output_tokens is not None else "provider_did_not_report"
        ),
        cache_read_tokens=_quantity(
            cache_read_tokens,
            None if cache_read_tokens is not None else "provider_did_not_report",
        ),
        cache_write_tokens=_quantity(None, "not_applicable"),
        reasoning_tokens=_quantity(
            reasoning_tokens,
            None if reasoning_tokens is not None else "provider_did_not_report",
        ),
        cache_read_inclusion=cache_read_inclusion,
        cache_write_inclusion="not_applicable",
        reasoning_inclusion=reasoning_inclusion,
    )


def _attribution(**updates: object) -> AttributionSnapshot:
    values = {
        "producer": "producer-1",
        "bill_owner": "billing-account-1",
        "bill_owner_missing_reason": None,
        "execution_location": "worker-1",
        "provider": "provider-1",
        "provider_missing_reason": None,
        "model": "model-revision-1",
        "model_missing_reason": None,
        "recipients": ("provider-1",),
        "recipients_missing_reason": None,
    }
    values.update(updates)
    return AttributionSnapshot(**values)


def _event(**updates: object) -> UsageEvent:
    values = {
        "schema_version": "candidate-v1",
        "event_id": "event-1",
        "tenant_id": "tenant-1",
        "task_id": "task-1",
        "session_id": "session-1",
        "intent_id": "intent-1",
        "attempt_id": "attempt-1",
        "ledger": "service",
        "operation": "extract",
        "predecessor_event_id": None,
        "status": "final",
        "attempt_outcome": "completed",
        "late_after_cancellation": False,
        "attribution": _attribution(),
        "usage": _usage(),
        "usage_missing_reason": None,
    }
    values.update(updates)
    return UsageEvent(**values)


class _SplitViewExtras(Mapping[str, object]):
    """Expose entries through items while hiding them from iteration/truthiness."""

    def __init__(self, entries: dict[str, object]) -> None:
        self._entries = entries
        self.iteration_calls = 0
        self.items_calls = 0

    def __getitem__(self, key: str) -> object:
        return self._entries[key]

    def __iter__(self) -> Iterator[str]:
        self.iteration_calls += 1
        return iter(())

    def __len__(self) -> int:
        return 0

    def items(self) -> ItemsView[str, object]:
        self.items_calls += 1
        return self._entries.items()


class _InverseItemsExtras(dict[str, object]):
    """Expose keys through iteration while hiding every items() pair."""

    def __init__(self, entries: dict[str, object]) -> None:
        super().__init__(entries)
        self.iteration_calls = 0
        self.items_calls = 0

    def __iter__(self) -> Iterator[str]:
        self.iteration_calls += 1
        return super().__iter__()

    def items(self) -> ItemsView[str, object]:
        self.items_calls += 1
        return {}.items()


class _NonDictInverseItemsExtras(Mapping[str, object]):
    """Expose iteration names while reporting an empty authoritative items view."""

    def __init__(self, entries: dict[str, object]) -> None:
        self._entries = entries
        self.iteration_calls = 0
        self.items_calls = 0

    def __getitem__(self, key: str) -> object:
        return self._entries[key]

    def __iter__(self) -> Iterator[str]:
        self.iteration_calls += 1
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def items(self) -> ItemsView[str, object]:
        self.items_calls += 1
        return {}.items()


class _HiddenBackingExtras(dict[str, object]):
    """Hide native dict backing from every overridable inventory view."""

    def __init__(self, entries: dict[str, object]) -> None:
        super().__init__(entries)
        self.len_calls = 0
        self.iteration_calls = 0
        self.keys_calls = 0
        self.items_calls = 0

    def __len__(self) -> int:
        self.len_calls += 1
        return 0

    def __iter__(self) -> Iterator[str]:
        self.iteration_calls += 1
        return iter(())

    def keys(self) -> KeysView[str]:
        self.keys_calls += 1
        return {}.keys()

    def items(self) -> ItemsView[str, object]:
        self.items_calls += 1
        return {}.items()

    def __bool__(self) -> bool:
        raise AssertionError("extra truthiness must not be read")


class _HidingStorageUsageEvent(UsageEvent):
    def __getattribute__(self, name: str) -> object:
        if name == "__dict__":
            backing = object.__getattribute__(self, "__dict__")
            return {
                key: item
                for key, item in dict.items(backing)
                if key != "future_constraint"
            }
        return super().__getattribute__(name)


class _HidingStorageAttribution(AttributionSnapshot):
    def __getattribute__(self, name: str) -> object:
        if name == "__dict__":
            backing = object.__getattribute__(self, "__dict__")
            return {
                key: item
                for key, item in dict.items(backing)
                if key != "future_constraint"
            }
        return super().__getattribute__(name)


class _DescriptorMaskedAttribution(AttributionSnapshot):
    @property
    def __dict__(self):
        native = _MODEL_DICT_DESCRIPTOR.__get__(self, BaseModel)
        visible = {
            name: value
            for name, value in dict.items(native)
            if name != "future_constraint"
        }
        if visible.get("provider") == "":
            visible["provider"] = "provider-1"
        return visible

    @__dict__.setter
    def __dict__(self, value):
        _MODEL_DICT_DESCRIPTOR.__set__(self, value)

    @property
    def __pydantic_extra__(self):
        return None

    @__pydantic_extra__.setter
    def __pydantic_extra__(self, value):
        _MODEL_EXTRAS_DESCRIPTOR.__set__(self, value)


class _DescriptorMaskedReconciledUsageStream(ReconciledUsageStream):
    @property
    def __dict__(self):
        native = _MODEL_DICT_DESCRIPTOR.__get__(self, BaseModel)
        visible = {
            name: value
            for name, value in dict.items(native)
            if name != "future_constraint"
        }
        if visible.get("known_token_subtotal") == 0:
            visible["known_token_subtotal"] = 15
        return visible

    @__dict__.setter
    def __dict__(self, value):
        _MODEL_DICT_DESCRIPTOR.__set__(self, value)

    @property
    def __pydantic_extra__(self):
        return None

    @__pydantic_extra__.setter
    def __pydantic_extra__(self, value):
        _MODEL_EXTRAS_DESCRIPTOR.__set__(self, value)


class _DescriptorMaskedUsageReconciliation(UsageReconciliation):
    @property
    def __dict__(self):
        native = _MODEL_DICT_DESCRIPTOR.__get__(self, BaseModel)
        visible = {
            name: value
            for name, value in dict.items(native)
            if name != "future_constraint"
        }
        if visible.get("tenant_id") == "":
            visible["tenant_id"] = "tenant-1"
        return visible

    @__dict__.setter
    def __dict__(self, value):
        _MODEL_DICT_DESCRIPTOR.__set__(self, value)

    @property
    def __pydantic_extra__(self):
        return None

    @__pydantic_extra__.setter
    def __pydantic_extra__(self, value):
        _MODEL_EXTRAS_DESCRIPTOR.__set__(self, value)


class _DescriptorMaskedTokenUsage(TokenUsage):
    @property
    def __dict__(self):
        native = _MODEL_DICT_DESCRIPTOR.__get__(self, BaseModel)
        visible = {
            name: value
            for name, value in dict.items(native)
            if name != "future_constraint"
        }
        if visible.get("measurement") == "future_measurement":
            visible["measurement"] = "actual"
        return visible

    @__dict__.setter
    def __dict__(self, value):
        _MODEL_DICT_DESCRIPTOR.__set__(self, value)

    @property
    def __pydantic_extra__(self):
        return None

    @__pydantic_extra__.setter
    def __pydantic_extra__(self, value):
        _MODEL_EXTRAS_DESCRIPTOR.__set__(self, value)


class _DescriptorMaskedReconciledLedger(ReconciledLedger):
    @property
    def __dict__(self):
        native = _MODEL_DICT_DESCRIPTOR.__get__(self, BaseModel)
        visible = {
            name: value
            for name, value in dict.items(native)
            if name != "future_constraint"
        }
        if visible.get("coverage") == "future_coverage":
            visible["coverage"] = "reported"
        return visible

    @__dict__.setter
    def __dict__(self, value):
        _MODEL_DICT_DESCRIPTOR.__set__(self, value)

    @property
    def __pydantic_extra__(self):
        return None

    @__pydantic_extra__.setter
    def __pydantic_extra__(self, value):
        _MODEL_EXTRAS_DESCRIPTOR.__set__(self, value)


class _ChangingItemsExtras(Mapping[str, object]):
    """Return a different items view after the first captured snapshot."""

    def __init__(self) -> None:
        self.iteration_calls = 0
        self.items_calls = 0

    def __getitem__(self, key: str) -> object:
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        self.iteration_calls += 1
        return iter(())

    def __len__(self) -> int:
        return 0

    def items(self) -> ItemsView[str, object]:
        self.items_calls += 1
        if self.items_calls == 1:
            return {}.items()
        return {"tenant_id": "shadow-tenant"}.items()


class _ReplacingItemsInput(dict[str, object]):
    """Replace one native edge from the receiver's public items callback."""

    def __init__(self, entries: dict[str, object]) -> None:
        super().__init__(entries)
        self.action: Callable[[], None] | None = None
        self.items_calls = 0

    def items(self) -> ItemsView[str, object]:
        self.items_calls += 1
        if self.action is not None:
            action, self.action = self.action, None
            action()
        return super().items()


class _DivergentItemsInput(dict[str, object]):
    """Expose a public view that differs from authoritative native dict edges."""

    def __init__(
        self,
        native_entries: dict[str, object],
        public_entries: dict[str, object],
    ) -> None:
        super().__init__(native_entries)
        self.public_entries = public_entries
        self.items_calls = 0

    def items(self) -> ItemsView[str, object]:
        self.items_calls += 1
        return self.public_entries.items()


class _DivergentPublicMapping(Mapping[str, object]):
    """Use the public items view for a non-dict Mapping input."""

    def __init__(
        self,
        lookup_entries: dict[str, object],
        public_entries: dict[str, object],
    ) -> None:
        self.lookup_entries = lookup_entries
        self.public_entries = public_entries
        self.items_calls = 0

    def __getitem__(self, key: str) -> object:
        return self.lookup_entries[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.lookup_entries)

    def __len__(self) -> int:
        return len(self.lookup_entries)

    def items(self) -> ItemsView[str, object]:
        self.items_calls += 1
        return self.public_entries.items()


class _AdjacentItemsMapping(Mapping[str, object]):
    """Yield one field twice with a callback between adjacent values."""

    def __init__(
        self,
        entries: dict[str, object],
        target: str,
        first: object,
        second: object,
        *,
        repeat: bool,
        action: Callable[[], None] | None = None,
    ) -> None:
        self.entries = entries
        self.target = target
        self.first = first
        self.second = second
        self.repeat = repeat
        self.action = action
        self.items_calls = 0
        self.action_calls = 0

    def __getitem__(self, key: str) -> object:
        return self.entries[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.entries)

    def __len__(self) -> int:
        return len(self.entries)

    def items(self) -> Iterator[tuple[str, object]]:
        self.items_calls += 1

        def iterate() -> Iterator[tuple[str, object]]:
            for key, value in self.entries.items():
                if key != self.target:
                    yield key, value
                    continue
                yield key, self.first
                if self.action is not None:
                    self.action_calls += 1
                    action, self.action = self.action, None
                    action()
                if self.repeat:
                    yield key, self.second

        return iterate()


class _DivergentTuple(tuple):
    """Expose a public iteration view distinct from native tuple storage."""

    def __new__(
        cls,
        native_entries: tuple[object, ...],
        public_entries: tuple[object, ...],
    ):
        instance = super().__new__(cls, native_entries)
        instance.public_entries = public_entries
        instance.iteration_calls = 0
        return instance

    def __iter__(self) -> Iterator[object]:
        self.iteration_calls += 1
        return iter(self.public_entries)


class _TraversalItemsInput(dict[str, object]):
    """Raise immediately or lazily from one public items traversal."""

    def __init__(self, entries: dict[str, object], behavior: str) -> None:
        super().__init__(entries)
        self.behavior = behavior
        self.items_calls = 0
        self.iterator_entries = 0

    def items(self) -> Iterator[tuple[str, object]]:
        self.items_calls += 1
        if self.behavior == "immediate":
            raise RuntimeError("items immediate sentinel")

        def iterate() -> Iterator[tuple[str, object]]:
            first = next(iter(dict.items(self)))
            self.iterator_entries += 1
            yield first
            if self.behavior == "lazy":
                raise RuntimeError("items lazy sentinel")

        return iterate()


class _TraversalPublicMapping(Mapping[str, object]):
    """Fail at one precise stage of a non-dict public items traversal."""

    def __init__(self, entries: dict[str, object], behavior: str) -> None:
        self.entries = entries
        self.behavior = behavior
        self.items_calls = 0
        self.iterator_calls = 0
        self.iterator_entries = 0

    def __getitem__(self, key: str) -> object:
        return self.entries[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.entries)

    def __len__(self) -> int:
        return len(self.entries)

    def items(self) -> Iterator[tuple[str, object]]:
        self.items_calls += 1
        if self.behavior == "items":
            raise RuntimeError("public mapping items sentinel")
        return _TraversalPublicItemsIterator(self)


class _TraversalPublicItemsIterator:
    def __init__(self, owner: _TraversalPublicMapping) -> None:
        self.owner = owner
        self.entries = tuple(dict.items(owner.entries))
        self.index = 0

    def __iter__(self) -> "_TraversalPublicItemsIterator":
        self.owner.iterator_calls += 1
        if self.owner.behavior == "iter":
            raise RuntimeError("public mapping iter sentinel")
        return self

    def __next__(self) -> tuple[str, object]:
        if self.owner.behavior == "next" and self.index == 0:
            raise RuntimeError("public mapping next sentinel")
        if self.owner.behavior == "lazy" and self.index == 1:
            raise RuntimeError("public mapping lazy sentinel")
        if self.index == len(self.entries):
            raise StopIteration
        entry = self.entries[self.index]
        self.index += 1
        self.owner.iterator_entries += 1
        return entry


class _StopIterationUnpackPair:
    def __init__(self) -> None:
        self.iteration_calls = 0

    def __iter__(self) -> Iterator[object]:
        self.iteration_calls += 1
        raise StopIteration("pair unpack sentinel")


class _LateMalformedPairMapping(Mapping[str, object]):
    """Yield a malformed pair after every otherwise valid public entry."""

    def __init__(self, entries: dict[str, object], malformed_pair: object) -> None:
        self.entries = entries
        self.malformed_pair = malformed_pair
        self.items_calls = 0
        self.iterator_calls = 0
        self.iterator_entries = 0

    def __getitem__(self, key: str) -> object:
        return self.entries[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.entries)

    def __len__(self) -> int:
        return len(self.entries)

    def items(self) -> Iterator[object]:
        self.items_calls += 1
        return _LateMalformedPairIterator(self)


class _LateMalformedPairIterator:
    def __init__(self, owner: _LateMalformedPairMapping) -> None:
        self.owner = owner
        self.entries: tuple[object, ...] = (
            *tuple(dict.items(owner.entries)),
            owner.malformed_pair,
        )
        self.index = 0

    def __iter__(self) -> "_LateMalformedPairIterator":
        self.owner.iterator_calls += 1
        return self

    def __next__(self) -> object:
        if self.index == len(self.entries):
            raise StopIteration
        entry = self.entries[self.index]
        self.index += 1
        self.owner.iterator_entries += 1
        return entry


class _FalseyPopulatedExtras(dict[str, object]):
    def __bool__(self) -> bool:
        return False


def _error_code(events: list[UsageEvent]) -> str:
    with pytest.raises(UsageReconciliationError) as caught:
        reconcile_usage_events(events)
    return caught.value.code


def _error_signature(events: tuple[UsageEvent, ...]) -> tuple[str, str]:
    with pytest.raises(UsageReconciliationError) as caught:
        reconcile_usage_events(events)
    return caught.value.code, caught.value.detail


def _assert_canonical_invalid_event(event: UsageEvent) -> None:
    with pytest.raises(UsageReconciliationError) as caught:
        reconcile_usage_events([event])

    assert caught.value.code == "invalid_event"
    assert str(caught.value) == (
        "invalid_event: an input event failed contract validation"
    )


def _assert_no_hidden_backing_override_reads(extras: _HiddenBackingExtras) -> None:
    assert extras.len_calls == 0
    assert extras.iteration_calls == 0
    assert extras.keys_calls == 0
    assert extras.items_calls == 0


def _assert_result_payload_rejected(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        UsageReconciliation.model_validate(payload)
    with pytest.raises(ValidationError):
        UsageReconciliation.model_validate_json(json.dumps(payload))


def _assert_stream_payload_rejected(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        ReconciledUsageStream.model_validate(payload)
    with pytest.raises(ValidationError):
        ReconciledUsageStream.model_validate_json(json.dumps(payload))


def _assert_attribution_authority_rejected(value: object) -> None:
    with pytest.raises(ValidationError) as caught:
        ReconciledUsageStream.model_validate(value)

    assert [
        (error["type"], error["loc"])
        for error in caught.value.errors(include_url=False)
    ] == [("extra_forbidden", ("attribution", "authority"))]


def _set_native_unknown(target: BaseModel, storage: str) -> None:
    if storage == "stored":
        backing = _MODEL_DICT_DESCRIPTOR.__get__(target, BaseModel)
        dict.__setitem__(backing, "future_constraint", "deny")
    else:
        backing = {"future_constraint": "deny"}
        _MODEL_EXTRAS_DESCRIPTOR.__set__(target, backing)
    assert dict.__getitem__(backing, "future_constraint") == "deny"


def _validation_signature(
    receiver: type[BaseModel], value: object
) -> list[tuple[str, tuple[object, ...], str]]:
    with pytest.raises(ValidationError) as caught:
        receiver.model_validate(value)
    return [
        (error["type"], error["loc"], error["msg"])
        for error in caught.value.errors(include_url=False)
    ]


def _python_receive(
    receiver: type[BaseModel],
    value: Mapping[str, object],
    entry: str,
) -> BaseModel:
    if entry == "model_validate":
        return receiver.model_validate(value)
    return receiver(**value)


def _python_validation_signature(
    receiver: type[BaseModel],
    value: Mapping[str, object],
    entry: str,
) -> list[tuple[str, tuple[object, ...], str]]:
    with pytest.raises(ValidationError) as caught:
        _python_receive(receiver, value, entry)
    return [
        (error["type"], error["loc"], error["msg"])
        for error in caught.value.errors(include_url=False)
    ]


def test_public_mapping_entry_validator_accepts_ordinary_entries() -> None:
    class LifetimeMarker:
        pass

    class LifetimePair:
        def __init__(self, key: object, value: object) -> None:
            self.key = key
            self.value = value

        def __iter__(self):
            return iter((self.key, self.value))

    first_key = LifetimeMarker()
    first_value = LifetimeMarker()
    second_key = LifetimeMarker()
    second_value = LifetimeMarker()
    first_pair = LifetimePair(first_key, first_value)
    second_pair = LifetimePair(second_key, second_value)
    entries = (pair for pair in (first_pair, second_pair))
    lifetime_refs = tuple(
        weakref.ref(item)
        for item in (
            entries,
            first_pair,
            first_key,
            first_value,
            second_pair,
            second_key,
            second_value,
        )
    )

    validated = usage_reconcile_contracts._validate_public_mapping_entries(entries)

    assert isinstance(
        validated,
        usage_reconcile_contracts._ValidatedPublicMappingEntries,
    )
    assert validated.entries is entries
    assert validated.iterator is entries
    assert validated.retained_keys == {first_key: None, second_key: None}
    assert validated.retained_entries == (
        (first_pair, first_key, first_value),
        (second_pair, second_key, second_value),
    )

    del (
        entries,
        first_pair,
        first_key,
        first_value,
        second_pair,
        second_key,
        second_value,
    )
    gc.collect()
    assert all(reference() is not None for reference in lifetime_refs)

    del validated
    gc.collect()
    assert all(reference() is None for reference in lifetime_refs)


@pytest.mark.parametrize(
    ("entries", "error_type", "error_message"),
    [
        (
            [("only-item",)],
            ValueError,
            "not enough values to unpack (expected 2, got 1)",
        ),
        (
            [["key", "value", "extra"]],
            ValueError,
            "too many values to unpack (expected 2)",
        ),
        (
            [([], "value")],
            TypeError,
            "unhashable type: 'list'",
        ),
    ],
)
def test_public_mapping_entry_validator_translates_ordinary_invalid_entries(
    entries: list[object],
    error_type: type[Exception],
    error_message: str,
) -> None:
    failed = usage_reconcile_contracts._validate_public_mapping_entries(entries)

    assert isinstance(failed, usage_reconcile_contracts._FailedNativeDictTraversal)
    assert isinstance(failed.error, error_type)
    assert str(failed.error) == error_message


def test_native_dict_snapshot_calls_public_entry_validator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {"field": "value"}
    observed_entries: list[tuple[tuple[str, object], ...]] = []
    validate_entries = usage_reconcile_contracts._validate_public_mapping_entries

    def observe(entries: object):
        captured = tuple(entries)
        observed_entries.append(captured)
        return validate_entries(captured)

    monkeypatch.setattr(
        usage_reconcile_contracts,
        "_validate_public_mapping_entries",
        observe,
    )

    assert usage_reconcile_contracts._native_snapshot(payload) == payload
    assert observed_entries == [(("field", "value"),)]


def test_native_dict_snapshot_retains_validation_owner_through_nested_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class LifetimeMarker:
        pass

    nested = {"leaf": "value"}
    payload = {"nested": nested}
    validate_entries = usage_reconcile_contracts._validate_public_mapping_entries
    snapshot = usage_reconcile_contracts._native_snapshot
    lifetime_refs: list[tuple[Callable[[], object | None], ...]] = []
    observed_liveness: list[tuple[bool, ...]] = []
    validation_calls = 0

    def retain_synthetic_owner(entries: object):
        nonlocal validation_calls
        validation_calls += 1
        validated = validate_entries(entries)
        if validation_calls != 1:
            return validated
        assert isinstance(
            validated,
            usage_reconcile_contracts._ValidatedPublicMappingEntries,
        )

        first_pair = LifetimeMarker()
        first_key = LifetimeMarker()
        first_value = LifetimeMarker()
        second_pair = LifetimeMarker()
        second_key = LifetimeMarker()
        second_value = LifetimeMarker()
        iterator = (item for item in ())
        lifetime_refs.append(
            tuple(
                weakref.ref(item)
                for item in (
                    iterator,
                    first_pair,
                    first_key,
                    first_value,
                    second_pair,
                    second_key,
                    second_value,
                )
            )
        )
        return usage_reconcile_contracts._ValidatedPublicMappingEntries(
            entries=(first_pair, second_pair),
            iterator=iterator,
            retained_keys={first_key: None, second_key: None},
            retained_entries=(
                (first_pair, first_key, first_value),
                (second_pair, second_key, second_value),
            ),
        )

    def observe_nested_projection(value, *args, **kwargs):
        if value is nested:
            observed_liveness.append(
                tuple(reference() is not None for reference in lifetime_refs[0])
            )
        return snapshot(value, *args, **kwargs)

    monkeypatch.setattr(
        usage_reconcile_contracts,
        "_validate_public_mapping_entries",
        retain_synthetic_owner,
    )
    monkeypatch.setattr(
        usage_reconcile_contracts,
        "_native_snapshot",
        observe_nested_projection,
    )

    assert usage_reconcile_contracts._native_snapshot(payload) == payload
    assert validation_calls == 2
    assert observed_liveness == [(True, True, True, True, True, True, True)]

    gc.collect()
    assert all(reference() is None for reference in lifetime_refs[0])


def test_native_dict_snapshot_propagates_public_entry_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {"field": "value"}
    ledger = reconcile_usage_events([_event()]).ledgers[0]
    ledger_payload = ledger.model_dump(mode="python")
    failure = usage_reconcile_contracts._FailedNativeDictTraversal(
        ValueError("public entry sentinel")
    )
    observed_entries: list[tuple[tuple[str, object], ...]] = []

    def fail(entries: object):
        observed_entries.append(tuple(entries))
        return failure

    monkeypatch.setattr(
        usage_reconcile_contracts,
        "_validate_public_mapping_entries",
        fail,
    )

    assert usage_reconcile_contracts._native_snapshot(payload) is failure
    assert observed_entries == [(("field", "value"),)]

    observed_entries.clear()
    assert _validation_signature(ReconciledLedger, ledger_payload) == [
        (
            "mapping_type",
            (),
            (
                "Input should be a valid mapping, error: ValueError: "
                "public entry sentinel"
            ),
        )
    ]
    assert observed_entries == [tuple(ledger_payload.items())]


def test_reconciled_ledger_python_before_schema_is_present_and_invoked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schema = ReconciledLedger.__pydantic_core_schema__
    assert schema["type"] == "json-or-python"
    python_schema = schema["python_schema"]
    assert python_schema["type"] == "function-before"
    assert (
        python_schema["function"]["function"]
        is usage_reconcile_contracts._snapshot_python_input
    )

    ledger = reconcile_usage_events([_event()]).ledgers[0]
    payload = ledger.model_dump(mode="python")
    snapshot = usage_reconcile_contracts._native_snapshot
    observed_inputs: list[object] = []
    depth = 0

    def observe(value, *args, **kwargs):
        nonlocal depth
        if depth == 0:
            observed_inputs.append(value)
        depth += 1
        try:
            return snapshot(value, *args, **kwargs)
        finally:
            depth -= 1

    monkeypatch.setattr(usage_reconcile_contracts, "_native_snapshot", observe)

    assert ReconciledLedger.model_validate(payload) == ledger
    assert observed_inputs == [payload]

    observed_inputs.clear()
    assert ReconciledLedger.model_validate_json(ledger.model_dump_json()) == ledger
    assert observed_inputs == []

    invalid_payload = {**payload, "coverage": "future_coverage"}
    assert _validation_signature(ReconciledLedger, invalid_payload) == [
        (
            "literal_error",
            ("coverage",),
            "Input should be 'reported' or 'unreported'",
        )
    ]
    assert observed_inputs == [invalid_payload]


def test_reconciled_ledger_validates_returned_python_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger = reconcile_usage_events([_event()]).ledgers[0]
    payload = ledger.model_dump(mode="python")
    returned_snapshot = {**payload, "coverage": "future_coverage"}
    observed_inputs: list[object] = []

    def replace(value, *args, **kwargs):
        observed_inputs.append(value)
        return returned_snapshot

    monkeypatch.setattr(usage_reconcile_contracts, "_native_snapshot", replace)

    assert _validation_signature(ReconciledLedger, payload) == [
        (
            "literal_error",
            ("coverage",),
            "Input should be 'reported' or 'unreported'",
        )
    ]
    assert observed_inputs == [payload]

    observed_inputs.clear()
    assert ReconciledLedger.model_validate_json(ledger.model_dump_json()) == ledger
    assert observed_inputs == []


def test_reconciles_three_ledgers_without_cross_ledger_relabelling() -> None:
    service = _event()
    agent = _event(
        event_id="event-agent",
        attempt_id="attempt-agent",
        ledger="consuming_agent",
        operation="agent-turn",
        usage=_usage(
            input_tokens=7,
            output_tokens=3,
            cache_read_tokens=2,
            reasoning_tokens=1,
            cache_read_inclusion="additional_to_input",
            reasoning_inclusion="additional_to_output",
        ),
    )
    evaluation = _event(
        event_id="event-eval",
        attempt_id="attempt-eval",
        ledger="evaluation",
        operation="judge",
        status="unavailable",
        attempt_outcome="failed",
        usage=None,
        usage_missing_reason="dependency_unavailable",
    )

    result = reconcile_usage_events([evaluation, agent, service])
    ledgers = {ledger.ledger: ledger for ledger in result.ledgers}

    assert ledgers["service"].total_tokens == 15
    assert ledgers["consuming_agent"].total_tokens == 13
    assert {ledger.coverage for ledger in ledgers.values()} == {"reported"}
    assert ledgers["evaluation"].known_token_subtotal == 0
    assert ledgers["evaluation"].total_tokens is None
    assert ledgers["evaluation"].incomplete_attempt_ids == ("attempt-eval",)


def test_unavailable_streams_preserve_distinct_head_missing_reasons() -> None:
    initial_missing = _event(
        event_id="initial-missing",
        attempt_id="attempt-provider",
        status="unavailable",
        attempt_outcome="failed",
        usage=None,
        usage_missing_reason="not_yet_reported",
    )
    provider_missing = _event(
        event_id="provider-missing",
        attempt_id="attempt-provider",
        predecessor_event_id="initial-missing",
        status="unavailable",
        attempt_outcome="failed",
        usage=None,
        usage_missing_reason="provider_did_not_report",
    )
    dependency_missing = _event(
        event_id="dependency-missing",
        attempt_id="attempt-dependency",
        status="unavailable",
        attempt_outcome="failed",
        usage=None,
        usage_missing_reason="dependency_unavailable",
    )

    forward = reconcile_usage_events(
        [initial_missing, provider_missing, dependency_missing]
    )
    reverse = reconcile_usage_events(
        [dependency_missing, provider_missing, initial_missing]
    )

    assert forward == reverse
    assert {
        stream.head_event_id: stream.usage_missing_reason
        for stream in forward.streams
    } == {
        "provider-missing": "provider_did_not_report",
        "dependency-missing": "dependency_unavailable",
    }


def test_stream_deserialization_rejects_usage_reason_contradictions() -> None:
    unavailable = reconcile_usage_events(
        [
            _event(
                status="unavailable",
                attempt_outcome="failed",
                usage=None,
                usage_missing_reason="dependency_unavailable",
            )
        ]
    ).streams[0].model_dump(mode="python")
    final = reconcile_usage_events([_event()]).streams[0].model_dump(mode="python")
    pending = reconcile_usage_events(
        [
            _event(
                status="pending",
                attempt_outcome="in_progress",
                usage=None,
            )
        ]
    ).streams[0].model_dump(mode="python")

    unavailable_without_reason = copy.deepcopy(unavailable)
    unavailable_without_reason["usage_missing_reason"] = None
    unavailable_with_usage = copy.deepcopy(unavailable)
    unavailable_with_usage["usage"] = _usage().model_dump(mode="python")
    final_with_reason = copy.deepcopy(final)
    final_with_reason["usage_missing_reason"] = "dependency_unavailable"
    pending_with_reason = copy.deepcopy(pending)
    pending_with_reason["usage_missing_reason"] = "not_yet_reported"

    for payload in (
        unavailable_without_reason,
        unavailable_with_usage,
        final_with_reason,
        pending_with_reason,
    ):
        _assert_stream_payload_rejected(payload)


def test_result_rejects_top_tenant_mismatching_stream() -> None:
    payload = copy.deepcopy(
        reconcile_usage_events([_event()]).model_dump(mode="python")
    )
    payload["tenant_id"] = "tenant-other"

    _assert_result_payload_rejected(payload)


def test_result_requires_exactly_one_canonical_summary_per_ledger() -> None:
    payload = copy.deepcopy(
        reconcile_usage_events([_event()]).model_dump(mode="python")
    )
    service = payload["ledgers"][0]
    payload["ledgers"] = (service, copy.deepcopy(service), copy.deepcopy(service))

    _assert_result_payload_rejected(payload)


def test_result_rejects_ledger_total_mismatching_streams() -> None:
    payload = copy.deepcopy(
        reconcile_usage_events([_event()]).model_dump(mode="python")
    )
    payload["ledgers"][0]["known_token_subtotal"] = 999
    payload["ledgers"][0]["total_tokens"] = 999

    _assert_result_payload_rejected(payload)


def test_result_rejects_reported_ledger_without_streams() -> None:
    payload = copy.deepcopy(
        reconcile_usage_events([_event()]).model_dump(mode="python")
    )
    payload["streams"] = ()

    _assert_result_payload_rejected(payload)


def test_result_rejects_unavailable_stream_retaining_usage_and_total() -> None:
    payload = copy.deepcopy(
        reconcile_usage_events([_event()]).model_dump(mode="python")
    )
    payload["streams"][0]["status"] = "unavailable"

    _assert_result_payload_rejected(payload)


def test_valid_result_round_trips_through_public_json_contract() -> None:
    result = reconcile_usage_events([_event()])

    assert UsageReconciliation.model_validate_json(result.model_dump_json()) == result


def test_result_revalidation_rejects_hidden_extra_in_nested_copy() -> None:
    result = reconcile_usage_events([_event()])
    invalid_stream = result.streams[0].model_copy(update={"authority": True})
    invalid_result = result.model_copy(update={"streams": (invalid_stream,)})

    with pytest.raises(ValidationError):
        UsageReconciliation.model_validate(invalid_result)


@pytest.mark.parametrize("storage", ["stored", "extra"])
@pytest.mark.parametrize("receiver_name", ["stream", "result"])
def test_result_receivers_reject_descriptor_hidden_root_storage_like_ordinary(
    storage: str,
    receiver_name: str,
) -> None:
    result = reconcile_usage_events([_event()])
    if receiver_name == "stream":
        receiver = ReconciledUsageStream
        ordinary = result.streams[0].model_copy()
        descriptor = _DescriptorMaskedReconciledUsageStream.model_validate(
            result.streams[0].model_dump(mode="python")
        )
    else:
        receiver = UsageReconciliation
        ordinary = result.model_copy()
        descriptor = _DescriptorMaskedUsageReconciliation.model_validate(
            result.model_dump(mode="python")
        )

    _set_native_unknown(ordinary, storage)
    _set_native_unknown(descriptor, storage)
    if storage == "stored":
        assert "future_constraint" not in descriptor.__dict__
    else:
        assert descriptor.__pydantic_extra__ is None

    expected = [
        (
            "extra_forbidden",
            ("future_constraint",),
            "Extra inputs are not permitted",
        )
    ]
    assert _validation_signature(receiver, ordinary) == expected
    assert _validation_signature(receiver, descriptor) == expected


@pytest.mark.parametrize("storage", ["stored", "extra"])
@pytest.mark.parametrize("target_name", ["attribution", "stream"])
def test_result_receivers_reject_descriptor_hidden_nested_storage_like_ordinary(
    storage: str,
    target_name: str,
) -> None:
    result = reconcile_usage_events([_event()])
    stream = result.streams[0]
    if target_name == "attribution":
        receiver = ReconciledUsageStream
        ordinary_target = stream.attribution.model_copy()
        descriptor_target = _DescriptorMaskedAttribution.model_validate(
            stream.attribution.model_dump(mode="python")
        )
        expected_location = ("attribution", "future_constraint")
        ordinary_parent = stream.model_copy(
            update={"attribution": ordinary_target}
        )
        descriptor_parent = stream.model_copy(
            update={"attribution": descriptor_target}
        )
    else:
        receiver = UsageReconciliation
        ordinary_target = stream.model_copy()
        descriptor_target = _DescriptorMaskedReconciledUsageStream.model_validate(
            stream.model_dump(mode="python")
        )
        expected_location = ("streams", 0, "future_constraint")
        ordinary_parent = result.model_copy(update={"streams": (ordinary_target,)})
        descriptor_parent = result.model_copy(
            update={"streams": (descriptor_target,)}
        )

    _set_native_unknown(ordinary_target, storage)
    _set_native_unknown(descriptor_target, storage)
    if storage == "stored":
        assert "future_constraint" not in descriptor_target.__dict__
    else:
        assert descriptor_target.__pydantic_extra__ is None

    expected = [
        (
            "extra_forbidden",
            expected_location,
            "Extra inputs are not permitted",
        )
    ]
    assert _validation_signature(receiver, ordinary_parent) == expected
    assert _validation_signature(receiver, descriptor_parent) == expected


@pytest.mark.parametrize("entry", ["model_validate", "constructor"])
@pytest.mark.parametrize("root_kind", ["dict", "mapping-proxy"])
@pytest.mark.parametrize("storage", ["stored", "extra"])
@pytest.mark.parametrize(
    ("target_name", "expected_location"),
    [
        ("attribution", ("attribution", "future_constraint")),
        ("usage", ("usage", "future_constraint")),
        ("stream", ("streams", 0, "future_constraint")),
        ("ledger", ("ledgers", 0, "future_constraint")),
    ],
)
def test_python_result_inputs_reject_nested_descriptor_unknowns_like_ordinary(
    entry: str,
    root_kind: str,
    storage: str,
    target_name: str,
    expected_location: tuple[object, ...],
) -> None:
    result = reconcile_usage_events([_event()])
    stream = result.streams[0]
    if target_name == "attribution":
        receiver = ReconciledUsageStream
        ordinary_target = stream.attribution.model_copy()
        descriptor_target = _DescriptorMaskedAttribution.model_validate(
            stream.attribution.model_dump(mode="python")
        )
        ordinary_payload = stream.model_dump(mode="python")
        descriptor_payload = stream.model_dump(mode="python")
        ordinary_payload["attribution"] = ordinary_target
        descriptor_payload["attribution"] = descriptor_target
    elif target_name == "usage":
        receiver = ReconciledUsageStream
        assert stream.usage is not None
        ordinary_target = stream.usage.model_copy()
        descriptor_target = _DescriptorMaskedTokenUsage.model_validate(
            stream.usage.model_dump(mode="python")
        )
        ordinary_payload = stream.model_dump(mode="python")
        descriptor_payload = stream.model_dump(mode="python")
        ordinary_payload["usage"] = ordinary_target
        descriptor_payload["usage"] = descriptor_target
    elif target_name == "stream":
        receiver = UsageReconciliation
        ordinary_target = stream.model_copy()
        descriptor_target = _DescriptorMaskedReconciledUsageStream.model_validate(
            stream.model_dump(mode="python")
        )
        ordinary_payload = result.model_dump(mode="python")
        descriptor_payload = result.model_dump(mode="python")
        ordinary_payload["streams"] = (ordinary_target,)
        descriptor_payload["streams"] = (descriptor_target,)
    else:
        receiver = UsageReconciliation
        ordinary_target = result.ledgers[0].model_copy()
        descriptor_target = _DescriptorMaskedReconciledLedger.model_validate(
            result.ledgers[0].model_dump(mode="python")
        )
        ordinary_payload = result.model_dump(mode="python")
        descriptor_payload = result.model_dump(mode="python")
        ordinary_payload["ledgers"] = (
            ordinary_target,
            *result.ledgers[1:],
        )
        descriptor_payload["ledgers"] = (
            descriptor_target,
            *result.ledgers[1:],
        )

    _set_native_unknown(ordinary_target, storage)
    _set_native_unknown(descriptor_target, storage)
    if storage == "stored":
        assert "future_constraint" not in descriptor_target.__dict__
    else:
        assert descriptor_target.__pydantic_extra__ is None

    if root_kind == "mapping-proxy":
        ordinary_input = MappingProxyType(ordinary_payload)
        descriptor_input = MappingProxyType(descriptor_payload)
    else:
        ordinary_input = ordinary_payload
        descriptor_input = descriptor_payload
    expected = [
        (
            "extra_forbidden",
            expected_location,
            "Extra inputs are not permitted",
        )
    ]
    assert _python_validation_signature(receiver, ordinary_input, entry) == expected
    assert _python_validation_signature(receiver, descriptor_input, entry) == expected


@pytest.mark.parametrize("entry", ["model_validate", "constructor"])
@pytest.mark.parametrize("root_kind", ["dict", "mapping-proxy"])
@pytest.mark.parametrize("receiver_name", ["stream", "result"])
def test_python_result_inputs_accept_valid_nested_descriptor_controls(
    entry: str,
    root_kind: str,
    receiver_name: str,
) -> None:
    result = reconcile_usage_events([_event()])
    stream = result.streams[0]
    if receiver_name == "stream":
        receiver = ReconciledUsageStream
        expected = stream
        attribution = _DescriptorMaskedAttribution.model_validate(
            stream.attribution.model_dump(mode="python")
        )
        assert stream.usage is not None
        usage = _DescriptorMaskedTokenUsage.model_validate(
            stream.usage.model_dump(mode="python")
        )
        payload = stream.model_dump(mode="python")
        payload.update(attribution=attribution, usage=usage)
    else:
        receiver = UsageReconciliation
        expected = result
        nested_stream = _DescriptorMaskedReconciledUsageStream.model_validate(
            stream.model_dump(mode="python")
        )
        ledger = _DescriptorMaskedReconciledLedger.model_validate(
            result.ledgers[0].model_dump(mode="python")
        )
        payload = result.model_dump(mode="python")
        payload["streams"] = (nested_stream,)
        payload["ledgers"] = (ledger, *result.ledgers[1:])

    received = _python_receive(
        receiver,
        MappingProxyType(payload) if root_kind == "mapping-proxy" else payload,
        entry,
    )

    assert received == expected


@pytest.mark.parametrize("entry", ["model_validate", "constructor"])
@pytest.mark.parametrize(
    ("target_name", "invalid_field", "invalid_value", "error_type", "location"),
    [
        (
            "attribution",
            "provider",
            "",
            "string_too_short",
            ("attribution", "provider"),
        ),
        (
            "usage",
            "measurement",
            "future_measurement",
            "literal_error",
            ("usage", "measurement"),
        ),
        (
            "stream",
            "known_token_subtotal",
            0,
            "value_error",
            ("streams", 0),
        ),
        (
            "ledger",
            "coverage",
            "future_coverage",
            "literal_error",
            ("ledgers", 0, "coverage"),
        ),
    ],
)
def test_python_result_inputs_reject_nested_descriptor_invalid_fields_like_ordinary(
    entry: str,
    target_name: str,
    invalid_field: str,
    invalid_value: object,
    error_type: str,
    location: tuple[object, ...],
) -> None:
    result = reconcile_usage_events([_event()])
    stream = result.streams[0]
    if target_name == "attribution":
        receiver = ReconciledUsageStream
        ordinary_target = stream.attribution.model_copy(
            update={invalid_field: invalid_value}
        )
        descriptor_target = _DescriptorMaskedAttribution.model_validate(
            stream.attribution.model_dump(mode="python")
        )
        ordinary_payload = stream.model_dump(mode="python")
        descriptor_payload = stream.model_dump(mode="python")
        ordinary_payload["attribution"] = ordinary_target
        descriptor_payload["attribution"] = descriptor_target
    elif target_name == "usage":
        receiver = ReconciledUsageStream
        assert stream.usage is not None
        ordinary_target = stream.usage.model_copy(
            update={invalid_field: invalid_value}
        )
        descriptor_target = _DescriptorMaskedTokenUsage.model_validate(
            stream.usage.model_dump(mode="python")
        )
        ordinary_payload = stream.model_dump(mode="python")
        descriptor_payload = stream.model_dump(mode="python")
        ordinary_payload["usage"] = ordinary_target
        descriptor_payload["usage"] = descriptor_target
    elif target_name == "stream":
        receiver = UsageReconciliation
        ordinary_target = stream.model_copy(update={invalid_field: invalid_value})
        descriptor_target = _DescriptorMaskedReconciledUsageStream.model_validate(
            stream.model_dump(mode="python")
        )
        ordinary_payload = result.model_dump(mode="python")
        descriptor_payload = result.model_dump(mode="python")
        ordinary_payload["streams"] = (ordinary_target,)
        descriptor_payload["streams"] = (descriptor_target,)
    else:
        receiver = UsageReconciliation
        ordinary_target = result.ledgers[0].model_copy(
            update={invalid_field: invalid_value}
        )
        descriptor_target = _DescriptorMaskedReconciledLedger.model_validate(
            result.ledgers[0].model_dump(mode="python")
        )
        ordinary_payload = result.model_dump(mode="python")
        descriptor_payload = result.model_dump(mode="python")
        ordinary_payload["ledgers"] = (
            ordinary_target,
            *result.ledgers[1:],
        )
        descriptor_payload["ledgers"] = (
            descriptor_target,
            *result.ledgers[1:],
        )

    descriptor_backing = _MODEL_DICT_DESCRIPTOR.__get__(
        descriptor_target,
        BaseModel,
    )
    dict.__setitem__(descriptor_backing, invalid_field, invalid_value)
    assert dict.__getitem__(descriptor_backing, invalid_field) == invalid_value
    assert descriptor_target.__dict__[invalid_field] != invalid_value

    ordinary_signature = _python_validation_signature(
        receiver,
        ordinary_payload,
        entry,
    )
    descriptor_signature = _python_validation_signature(
        receiver,
        descriptor_payload,
        entry,
    )
    assert descriptor_signature == ordinary_signature
    assert descriptor_signature[0][:2] == (error_type, location)


@pytest.mark.parametrize("receiver_name", ["stream", "result"])
def test_result_receivers_reject_descriptor_hidden_invalid_root(
    receiver_name: str,
) -> None:
    result = reconcile_usage_events([_event()])
    if receiver_name == "stream":
        receiver = ReconciledUsageStream
        ordinary = result.streams[0].model_copy(
            update={"known_token_subtotal": 0}
        )
        descriptor = _DescriptorMaskedReconciledUsageStream.model_validate(
            result.streams[0].model_dump(mode="python")
        )
        field_name = "known_token_subtotal"
        invalid_value: object = 0
        expected_type = "value_error"
        expected_location: tuple[object, ...] = ()
    else:
        receiver = UsageReconciliation
        ordinary = result.model_copy(update={"tenant_id": ""})
        descriptor = _DescriptorMaskedUsageReconciliation.model_validate(
            result.model_dump(mode="python")
        )
        field_name = "tenant_id"
        invalid_value = ""
        expected_type = "string_too_short"
        expected_location = ("tenant_id",)

    backing = _MODEL_DICT_DESCRIPTOR.__get__(descriptor, BaseModel)
    dict.__setitem__(backing, field_name, invalid_value)
    assert descriptor.__dict__[field_name] != invalid_value
    ordinary_signature = _validation_signature(receiver, ordinary)
    descriptor_signature = _validation_signature(receiver, descriptor)
    assert descriptor_signature == ordinary_signature
    assert descriptor_signature[0][:2] == (expected_type, expected_location)


@pytest.mark.parametrize("receiver_name", ["stream", "result"])
def test_python_prefreeze_precedes_mapping_key_hooks(receiver_name: str) -> None:
    result = reconcile_usage_events([_event()])

    class RepairingFieldName(str):
        def __new__(cls, value: str):
            instance = super().__new__(cls, value)
            instance.action = None
            instance.calls = 0
            return instance

        def __hash__(self):
            if self.action is not None:
                self.calls += 1
                action, self.action = self.action, None
                action()
            return super().__hash__()

    if receiver_name == "stream":
        receiver = ReconciledUsageStream
        child = result.streams[0].attribution.model_copy(update={"provider": ""})
        child_backing = _MODEL_DICT_DESCRIPTOR.__get__(child, BaseModel)
        payload = result.streams[0].model_dump(mode="python")
        payload["attribution"] = child
        expected_location = ("attribution", "provider")
        repair = lambda: dict.__setitem__(
            child_backing,
            "provider",
            "provider-1",
        )
    else:
        receiver = UsageReconciliation
        child = result.streams[0].model_copy(
            update={"known_token_subtotal": 0}
        )
        child_backing = _MODEL_DICT_DESCRIPTOR.__get__(child, BaseModel)
        payload = result.model_dump(mode="python")
        payload["streams"] = (child,)
        expected_location = ("streams", 0)
        repair = lambda: dict.__setitem__(
            child_backing,
            "known_token_subtotal",
            15,
        )

    tenant_id = dict.pop(payload, "tenant_id")
    key = RepairingFieldName("tenant_id")
    dict.__setitem__(payload, key, tenant_id)
    key.action = repair

    signature = _python_validation_signature(receiver, payload, "model_validate")

    assert key.calls == 1
    assert signature[0][1] == expected_location
    if receiver_name == "stream":
        assert child_backing["provider"] == "provider-1"
    else:
        assert child_backing["known_token_subtotal"] == 15


@pytest.mark.parametrize(
    ("target_name", "error_type", "location"),
    [
        ("attribution", "string_too_short", ("attribution", "provider")),
        ("usage", "literal_error", ("usage", "measurement")),
        ("stream", "value_error", ("streams", 0)),
        ("ledger", "literal_error", ("ledgers", 0, "coverage")),
    ],
)
def test_python_prefreeze_retains_native_dict_edges_before_items_callback(
    target_name: str,
    error_type: str,
    location: tuple[object, ...],
) -> None:
    result = reconcile_usage_events([_event()])
    stream = result.streams[0]
    if target_name == "attribution":
        receiver = ReconciledUsageStream
        expected = stream
        field_name = "attribution"
        invalid_child = stream.attribution.model_copy(update={"provider": ""})
        valid_child = stream.attribution.model_copy()
        payload = stream.model_dump(mode="python")
    elif target_name == "usage":
        receiver = ReconciledUsageStream
        expected = stream
        field_name = "usage"
        assert stream.usage is not None
        invalid_child = stream.usage.model_copy(
            update={"measurement": "future_measurement"}
        )
        valid_child = stream.usage.model_copy()
        payload = stream.model_dump(mode="python")
    elif target_name == "stream":
        receiver = UsageReconciliation
        expected = result
        field_name = "streams"
        invalid_child = (
            stream.model_copy(update={"known_token_subtotal": 0}),
        )
        valid_child = (stream.model_copy(),)
        payload = result.model_dump(mode="python")
    else:
        receiver = UsageReconciliation
        expected = result
        field_name = "ledgers"
        invalid_child = (
            result.ledgers[0].model_copy(update={"coverage": "future_coverage"}),
            *result.ledgers[1:],
        )
        valid_child = (result.ledgers[0].model_copy(), *result.ledgers[1:])
        payload = result.model_dump(mode="python")

    ordinary = _ReplacingItemsInput(payload)
    dict.__setitem__(ordinary, field_name, invalid_child)
    ordinary_signature = _validation_signature(receiver, ordinary)

    replacement = _ReplacingItemsInput(payload)
    dict.__setitem__(replacement, field_name, invalid_child)
    replacement.action = lambda: dict.__setitem__(
        replacement,
        field_name,
        valid_child,
    )
    replacement_signature = _validation_signature(receiver, replacement)

    valid = _ReplacingItemsInput(payload)
    dict.__setitem__(valid, field_name, valid_child)
    valid.action = lambda: dict.__setitem__(valid, field_name, valid_child)
    received = receiver.model_validate(valid)

    assert ordinary.items_calls == 1
    assert replacement.items_calls == 1
    assert valid.items_calls == 1
    assert replacement_signature == ordinary_signature
    assert replacement_signature[0][:2] == (error_type, location)
    assert dict.__getitem__(replacement, field_name) is valid_child
    assert received == expected


def test_python_prefreeze_keeps_native_dict_and_public_mapping_authority() -> None:
    stream = reconcile_usage_events([_event()]).streams[0]
    valid_payload = stream.model_dump(mode="python")
    invalid_payload = stream.model_dump(mode="python")
    invalid_payload["attribution"] = stream.attribution.model_copy(
        update={"provider": ""}
    )
    expected = [
        (
            "string_too_short",
            ("attribution", "provider"),
            "String should have at least 1 character",
        )
    ]

    native_invalid = _DivergentItemsInput(invalid_payload, valid_payload)
    assert _validation_signature(ReconciledUsageStream, native_invalid) == expected
    assert native_invalid.items_calls == 1

    native_valid = _DivergentItemsInput(valid_payload, invalid_payload)
    assert ReconciledUsageStream.model_validate(native_valid) == stream
    assert native_valid.items_calls == 1

    public_valid = _DivergentPublicMapping(invalid_payload, valid_payload)
    assert ReconciledUsageStream.model_validate(public_valid) == stream
    assert public_valid.items_calls == 1

    public_invalid = _DivergentPublicMapping(valid_payload, invalid_payload)
    assert _validation_signature(ReconciledUsageStream, public_invalid) == expected
    assert public_invalid.items_calls == 1


@pytest.mark.parametrize("receiver_name", ["result", "stream"])
def test_python_public_mapping_keeps_first_observed_shared_descendant(
    receiver_name: str,
) -> None:
    def run_case(
        case: str,
    ) -> tuple[
        _AdjacentItemsMapping,
        BaseModel | list[tuple[str, tuple[object, ...], str]],
    ]:
        result = reconcile_usage_events([_event()])
        stream = result.streams[0]
        if receiver_name == "result":
            receiver = UsageReconciliation
            expected: BaseModel = result
            payload = result.model_dump(mode="python")
            target = "streams"
            valid_parent: object = tuple([stream.model_copy()])
            shared = stream.model_copy(update={"known_token_subtotal": 0})
            first_parent: object = tuple([shared])
            second_parent: object = (
                first_parent if case == "same-parent" else tuple([shared])
            )
            shared_backing = _MODEL_DICT_DESCRIPTOR.__get__(shared, BaseModel)

            def repair() -> None:
                dict.__setitem__(shared_backing, "known_token_subtotal", 15)

        else:
            receiver = ReconciledUsageStream
            expected = stream
            payload = stream.model_dump(mode="python")
            target = "usage"
            assert stream.usage is not None
            valid_usage = stream.usage.model_copy()
            valid_parent = valid_usage
            shared = valid_usage.input_tokens.model_copy(update={"value": -1})
            first_parent = valid_usage.model_copy(
                update={"input_tokens": shared}
            )
            second_parent = (
                first_parent
                if case == "same-parent"
                else valid_usage.model_copy(update={"input_tokens": shared})
            )
            shared_backing = _MODEL_DICT_DESCRIPTOR.__get__(shared, BaseModel)

            def repair() -> None:
                dict.__setitem__(shared_backing, "value", 10)

        if case == "valid":
            first = valid_parent
            second = valid_parent
            repeat = False
            action = None
        elif case == "ordinary-invalid":
            first = first_parent
            second = first_parent
            repeat = False
            action = None
        else:
            first = first_parent
            second = second_parent
            repeat = True
            action = repair
        value = _AdjacentItemsMapping(
            payload,
            target,
            first,
            second,
            repeat=repeat,
            action=action,
        )
        if case == "valid":
            outcome: BaseModel | list[tuple[str, tuple[object, ...], str]] = (
                receiver.model_validate(value)
            )
        else:
            outcome = _validation_signature(receiver, value)
        assert value.items_calls == 1
        assert value.action_calls == (1 if action is not None else 0)
        if case == "valid":
            assert outcome == expected
        return value, outcome

    _valid_mapping, _valid_outcome = run_case("valid")
    _ordinary_mapping, ordinary = run_case("ordinary-invalid")
    _same_mapping, same_parent = run_case("same-parent")
    _distinct_mapping, distinct_parent = run_case("distinct-parent")

    assert same_parent == ordinary
    assert distinct_parent == ordinary
    if receiver_name == "result":
        assert ordinary[0][:2] == ("value_error", ("streams", 0))
    else:
        assert ordinary[0][:2] == (
            "greater_than_equal",
            ("usage", "input_tokens", "value"),
        )


def test_python_prefreeze_retains_native_root_tuple_edges_without_iteration() -> None:
    result = reconcile_usage_events([_event()])
    valid_stream = result.streams[0].model_copy()
    invalid_stream = result.streams[0].model_copy(
        update={"known_token_subtotal": 0}
    )
    payload = result.model_dump(mode="python")

    ordinary_payload = dict(payload)
    ordinary_payload["streams"] = (invalid_stream,)
    ordinary_signature = _validation_signature(
        UsageReconciliation,
        ordinary_payload,
    )

    divergent = _DivergentTuple((invalid_stream,), (valid_stream,))
    divergent_payload = dict(payload)
    divergent_payload["streams"] = divergent
    divergent_signature = _validation_signature(
        UsageReconciliation,
        divergent_payload,
    )

    control = _DivergentTuple((valid_stream,), (invalid_stream,))
    control_payload = dict(payload)
    control_payload["streams"] = control
    received = UsageReconciliation.model_validate(control_payload)

    assert divergent_signature == ordinary_signature
    assert divergent_signature[0][:2] == ("value_error", ("streams", 0))
    assert divergent.iteration_calls == 0
    assert control.iteration_calls == 0
    assert received == result


def test_python_prefreeze_retains_nested_native_tuple_edges_without_iteration() -> None:
    stream = reconcile_usage_events([_event()]).streams[0]

    ordinary_attribution = stream.attribution.model_copy()
    ordinary_backing = _MODEL_DICT_DESCRIPTOR.__get__(
        ordinary_attribution,
        BaseModel,
    )
    dict.__setitem__(ordinary_backing, "recipients", ("",))
    ordinary_payload = stream.model_dump(mode="python")
    ordinary_payload["attribution"] = ordinary_attribution
    ordinary_signature = _validation_signature(
        ReconciledUsageStream,
        ordinary_payload,
    )

    divergent = _DivergentTuple(("",), ("provider-1",))
    divergent_attribution = stream.attribution.model_copy()
    divergent_backing = _MODEL_DICT_DESCRIPTOR.__get__(
        divergent_attribution,
        BaseModel,
    )
    dict.__setitem__(divergent_backing, "recipients", divergent)
    divergent_payload = stream.model_dump(mode="python")
    divergent_payload["attribution"] = divergent_attribution
    divergent_signature = _validation_signature(
        ReconciledUsageStream,
        divergent_payload,
    )

    control = _DivergentTuple(("provider-1",), ("",))
    control_attribution = stream.attribution.model_copy()
    control_backing = _MODEL_DICT_DESCRIPTOR.__get__(control_attribution, BaseModel)
    dict.__setitem__(control_backing, "recipients", control)
    control_payload = stream.model_dump(mode="python")
    control_payload["attribution"] = control_attribution
    received = ReconciledUsageStream.model_validate(control_payload)

    assert divergent_signature == ordinary_signature
    assert divergent_signature[0][:2] == (
        "string_too_short",
        ("attribution", "recipients", 0),
    )
    assert divergent.iteration_calls == 0
    assert control.iteration_calls == 0
    assert received == stream


@pytest.mark.parametrize("behavior", ["immediate", "lazy"])
@pytest.mark.parametrize("receiver_name", ["stream", "result"])
def test_python_result_root_dict_traversal_errors_remain_validation_errors(
    behavior: str,
    receiver_name: str,
) -> None:
    result = reconcile_usage_events([_event()])
    if receiver_name == "stream":
        receiver = ReconciledUsageStream
        valid = result.streams[0]
    else:
        receiver = UsageReconciliation
        valid = result

    failing = _TraversalItemsInput(valid.model_dump(mode="python"), behavior)
    assert _validation_signature(receiver, failing) == [
        (
            "mapping_type",
            (),
            (
                "Input should be a valid mapping, error: RuntimeError: "
                f"items {behavior} sentinel"
            ),
        )
    ]
    assert failing.items_calls == 1
    assert failing.iterator_entries == (1 if behavior == "lazy" else 0)

    control = _TraversalItemsInput(valid.model_dump(mode="python"), "valid")
    assert receiver.model_validate(control) == valid
    assert control.items_calls == 1
    assert control.iterator_entries == 1


@pytest.mark.parametrize("behavior", ["immediate", "lazy"])
@pytest.mark.parametrize("entry", ["model_validate", "constructor"])
@pytest.mark.parametrize(
    ("target_name", "location"),
    [
        ("attribution", ("attribution",)),
        ("usage", ("usage",)),
        ("stream", ("streams", 0)),
        ("ledger", ("ledgers", 0)),
    ],
)
def test_python_nested_dict_traversal_errors_keep_category_and_location(
    behavior: str,
    entry: str,
    target_name: str,
    location: tuple[object, ...],
) -> None:
    result = reconcile_usage_events([_event()])
    stream = result.streams[0]
    if target_name == "attribution":
        receiver = ReconciledUsageStream
        expected = stream
        field_name = "attribution"
        child = stream.attribution
        payload = stream.model_dump(mode="python")
    elif target_name == "usage":
        receiver = ReconciledUsageStream
        expected = stream
        field_name = "usage"
        assert stream.usage is not None
        child = stream.usage
        payload = stream.model_dump(mode="python")
    elif target_name == "stream":
        receiver = UsageReconciliation
        expected = result
        field_name = "streams"
        child = stream
        payload = result.model_dump(mode="python")
    else:
        receiver = UsageReconciliation
        expected = result
        field_name = "ledgers"
        child = result.ledgers[0]
        payload = result.model_dump(mode="python")

    failing = _TraversalItemsInput(child.model_dump(mode="python"), behavior)
    control = _TraversalItemsInput(child.model_dump(mode="python"), "valid")
    if target_name in {"stream", "ledger"}:
        remaining = (
            () if target_name == "stream" else result.ledgers[1:]
        )
        payload[field_name] = (failing, *remaining)
    else:
        payload[field_name] = failing

    assert _python_validation_signature(receiver, payload, entry) == [
        (
            "mapping_type",
            location,
            (
                "Input should be a valid mapping, error: RuntimeError: "
                f"items {behavior} sentinel"
            ),
        )
    ]
    assert failing.items_calls == 1
    assert failing.iterator_entries == (1 if behavior == "lazy" else 0)

    if target_name in {"stream", "ledger"}:
        payload[field_name] = (control, *remaining)
    else:
        payload[field_name] = control
    assert _python_receive(receiver, payload, entry) == expected
    assert control.items_calls == 1
    assert control.iterator_entries == 1


@pytest.mark.parametrize(
    ("behavior", "expected_iterator_calls", "expected_entries"),
    [
        ("items", 0, 0),
        ("iter", 1, 0),
        ("next", 1, 0),
        ("lazy", 1, 1),
    ],
)
@pytest.mark.parametrize("receiver_name", ["stream", "result"])
def test_python_result_root_public_mapping_traversal_errors_are_located(
    behavior: str,
    expected_iterator_calls: int,
    expected_entries: int,
    receiver_name: str,
) -> None:
    result = reconcile_usage_events([_event()])
    if receiver_name == "stream":
        receiver = ReconciledUsageStream
        valid = result.streams[0]
    else:
        receiver = UsageReconciliation
        valid = result

    payload = valid.model_dump(mode="python")
    failing = _TraversalPublicMapping(payload, behavior)
    assert _validation_signature(receiver, failing) == [
        (
            "mapping_type",
            (),
            (
                "Input should be a valid mapping, error: RuntimeError: "
                f"public mapping {behavior} sentinel"
            ),
        )
    ]
    assert failing.items_calls == 1
    assert failing.iterator_calls == expected_iterator_calls
    assert failing.iterator_entries == expected_entries

    control = _TraversalPublicMapping(payload, "valid")
    assert receiver.model_validate(control) == valid
    assert control.items_calls == 1
    assert control.iterator_calls == 1
    assert control.iterator_entries == len(payload)


@pytest.mark.parametrize(
    ("behavior", "expected_iterator_calls", "expected_entries"),
    [
        ("items", 0, 0),
        ("iter", 1, 0),
        ("next", 1, 0),
        ("lazy", 1, 1),
    ],
)
@pytest.mark.parametrize("receiver_name", ["stream", "result"])
def test_python_nested_public_mapping_traversal_errors_keep_location(
    behavior: str,
    expected_iterator_calls: int,
    expected_entries: int,
    receiver_name: str,
) -> None:
    result = reconcile_usage_events([_event()])
    stream = result.streams[0]
    if receiver_name == "stream":
        receiver = ReconciledUsageStream
        expected = stream
        payload = stream.model_dump(mode="python")
        field_name = "attribution"
        child = stream.attribution
        location = ("attribution",)
    else:
        receiver = UsageReconciliation
        expected = result
        payload = result.model_dump(mode="python")
        field_name = "streams"
        child = stream
        location = ("streams", 0)

    failing = _TraversalPublicMapping(child.model_dump(mode="python"), behavior)
    payload[field_name] = failing if receiver_name == "stream" else (failing,)
    assert _validation_signature(receiver, payload) == [
        (
            "mapping_type",
            location,
            (
                "Input should be a valid mapping, error: RuntimeError: "
                f"public mapping {behavior} sentinel"
            ),
        )
    ]
    assert failing.items_calls == 1
    assert failing.iterator_calls == expected_iterator_calls
    assert failing.iterator_entries == expected_entries

    control = _TraversalPublicMapping(child.model_dump(mode="python"), "valid")
    payload[field_name] = control if receiver_name == "stream" else (control,)
    assert receiver.model_validate(payload) == expected
    assert control.items_calls == 1
    assert control.iterator_calls == 1
    assert control.iterator_entries == len(child.model_dump(mode="python"))


@pytest.mark.parametrize("receiver_name", ["stream", "result"])
def test_python_public_mapping_does_not_remap_recursive_cycle_error(
    receiver_name: str,
) -> None:
    result = reconcile_usage_events([_event()])
    stream = result.streams[0]
    cycle: list[object] = []
    cycle.append(cycle)
    if receiver_name == "stream":
        receiver = ReconciledUsageStream
        payload = stream.model_dump(mode="python")
        field_name = "attribution"
    else:
        receiver = UsageReconciliation
        payload = result.model_dump(mode="python")
        field_name = "streams"
    payload[field_name] = cycle if receiver_name == "stream" else (cycle,)
    expected_entries = tuple(payload).index(field_name) + 1
    value = _TraversalPublicMapping(payload, "valid")

    assert _validation_signature(receiver, value) == [
        ("value_error", (), "Value error, cyclic input graph")
    ]
    assert value.items_calls == 1
    assert value.iterator_calls == 1
    assert value.iterator_entries == expected_entries


@pytest.mark.parametrize(
    ("pair_kind", "error_detail"),
    [
        ("stop-iteration", "StopIteration: pair unpack sentinel"),
        (
            "one-item",
            "ValueError: not enough values to unpack (expected 2, got 1)",
        ),
        ("three-item", "ValueError: too many values to unpack (expected 2)"),
    ],
)
@pytest.mark.parametrize("placement", ["root", "nested"])
@pytest.mark.parametrize("receiver_name", ["stream", "result"])
def test_python_public_mapping_late_malformed_pair_is_located(
    pair_kind: str,
    error_detail: str,
    placement: str,
    receiver_name: str,
) -> None:
    result = reconcile_usage_events([_event()])
    stream = result.streams[0]
    if receiver_name == "stream":
        receiver = ReconciledUsageStream
        valid = stream
        if placement == "root":
            outer_payload: dict[str, object] | None = None
            entries = stream.model_dump(mode="python")
            location: tuple[object, ...] = ()
        else:
            outer_payload = stream.model_dump(mode="python")
            entries = stream.attribution.model_dump(mode="python")
            location = ("attribution",)
    else:
        receiver = UsageReconciliation
        valid = result
        if placement == "root":
            outer_payload = None
            entries = result.model_dump(mode="python")
            location = ()
        else:
            outer_payload = result.model_dump(mode="python")
            entries = stream.model_dump(mode="python")
            location = ("streams", 0)

    if pair_kind == "stop-iteration":
        malformed_pair: object = _StopIterationUnpackPair()
    elif pair_kind == "one-item":
        malformed_pair = ("only-item",)
    else:
        malformed_pair = ("key", "value", "extra")
    failing = _LateMalformedPairMapping(entries, malformed_pair)
    if placement == "root":
        value: object = failing
    else:
        assert outer_payload is not None
        if receiver_name == "stream":
            outer_payload["attribution"] = failing
        else:
            outer_payload["streams"] = (failing,)
        value = outer_payload

    assert _validation_signature(receiver, value) == [
        (
            "mapping_type",
            location,
            f"Input should be a valid mapping, error: {error_detail}",
        )
    ]
    assert failing.items_calls == 1
    assert failing.iterator_calls == 1
    assert failing.iterator_entries == len(entries) + 1
    if isinstance(malformed_pair, _StopIterationUnpackPair):
        assert malformed_pair.iteration_calls == 1

    assert receiver.model_validate(valid.model_dump(mode="python")) == valid


@pytest.mark.parametrize("placement", ["root", "nested"])
@pytest.mark.parametrize("receiver_name", ["stream", "result"])
def test_python_public_mapping_unhashable_key_is_located(
    placement: str,
    receiver_name: str,
) -> None:
    result = reconcile_usage_events([_event()])
    stream = result.streams[0]
    if receiver_name == "stream":
        receiver = ReconciledUsageStream
        valid = stream
        if placement == "root":
            outer_payload: dict[str, object] | None = None
            entries = stream.model_dump(mode="python")
            location: tuple[object, ...] = ()
        else:
            outer_payload = stream.model_dump(mode="python")
            entries = stream.attribution.model_dump(mode="python")
            location = ("attribution",)
    else:
        receiver = UsageReconciliation
        valid = result
        if placement == "root":
            outer_payload = None
            entries = result.model_dump(mode="python")
            location = ()
        else:
            outer_payload = result.model_dump(mode="python")
            entries = stream.model_dump(mode="python")
            location = ("streams", 0)

    failing = _LateMalformedPairMapping(entries, ([], "ignored"))
    control = _TraversalPublicMapping(entries, "valid")
    if placement == "root":
        failing_value: object = failing
        control_value: object = control
    else:
        assert outer_payload is not None
        failing_payload = dict(outer_payload)
        control_payload = dict(outer_payload)
        if receiver_name == "stream":
            failing_payload["attribution"] = failing
            control_payload["attribution"] = control
        else:
            failing_payload["streams"] = (failing,)
            control_payload["streams"] = (control,)
        failing_value = failing_payload
        control_value = control_payload

    assert _validation_signature(receiver, failing_value) == [
        (
            "mapping_type",
            location,
            (
                "Input should be a valid mapping, error: TypeError: "
                "unhashable type: 'list'"
            ),
        )
    ]
    assert failing.items_calls == 1
    assert failing.iterator_calls == 1
    assert failing.iterator_entries == len(entries) + 1

    assert receiver.model_validate(control_value) == valid
    assert control.items_calls == 1
    assert control.iterator_calls == 1
    assert control.iterator_entries == len(entries)


@pytest.mark.parametrize("placement", ["root", "nested"])
@pytest.mark.parametrize("receiver_name", ["stream", "result"])
def test_frozen_native_dict_projection_unhashable_key_is_located(
    placement: str,
    receiver_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = reconcile_usage_events([_event()])
    stream = result.streams[0]
    if receiver_name == "stream":
        receiver = ReconciledUsageStream
        valid = stream
        failing_value = stream.model_dump(mode="python")
        target = (
            failing_value
            if placement == "root"
            else failing_value["attribution"]
        )
        location: tuple[object, ...] = (
            () if placement == "root" else ("attribution",)
        )
    else:
        receiver = UsageReconciliation
        valid = result
        failing_value = result.model_dump(mode="python")
        target = (
            failing_value
            if placement == "root"
            else failing_value["streams"][0]
        )
        location = () if placement == "root" else ("streams", 0)

    assert type(target) is dict
    original_freeze = usage_reconcile_contracts._freeze_model_storage
    injected_targets: list[dict[object, object]] = []

    def inject_unhashable_frozen_key(
        value,
        frozen_models,
        frozen_dicts,
        frozen_tuples,
        visited,
    ) -> None:
        original_freeze(
            value,
            frozen_models,
            frozen_dicts,
            frozen_tuples,
            visited,
        )
        if value is target:
            injected_targets.append(value)
            frozen_dicts[id(value)] = (value, (([], "ignored"),))

    monkeypatch.setattr(
        usage_reconcile_contracts,
        "_freeze_model_storage",
        inject_unhashable_frozen_key,
    )

    assert _validation_signature(receiver, failing_value) == [
        (
            "mapping_type",
            location,
            (
                "Input should be a valid mapping, error: TypeError: "
                "unhashable type: 'list'"
            ),
        )
    ]
    assert injected_targets == [target]

    control = valid.model_dump(mode="python")
    assert receiver.model_validate(control) == valid
    assert injected_targets == [target]


@pytest.mark.parametrize(
    "target_name",
    ["stream", "attribution", "result", "nested_stream"],
)
def test_result_receivers_accept_valid_descriptor_subclass_controls(
    target_name: str,
) -> None:
    result = reconcile_usage_events([_event()])
    stream = result.streams[0]
    if target_name == "stream":
        value = _DescriptorMaskedReconciledUsageStream.model_validate(
            stream.model_dump(mode="python")
        )
        received = ReconciledUsageStream.model_validate(value)
        assert type(received) is ReconciledUsageStream
        assert received == stream
    elif target_name == "attribution":
        attribution = _DescriptorMaskedAttribution.model_validate(
            stream.attribution.model_dump(mode="python")
        )
        value = stream.model_copy(update={"attribution": attribution})
        received = ReconciledUsageStream.model_validate(value)
        assert type(received.attribution) is AttributionSnapshot
        assert received == stream
    elif target_name == "result":
        value = _DescriptorMaskedUsageReconciliation.model_validate(
            result.model_dump(mode="python")
        )
        received = UsageReconciliation.model_validate(value)
        assert type(received) is UsageReconciliation
        assert received == result
    else:
        nested = _DescriptorMaskedReconciledUsageStream.model_validate(
            stream.model_dump(mode="python")
        )
        value = result.model_copy(update={"streams": (nested,)})
        received = UsageReconciliation.model_validate(value)
        assert type(received.streams[0]) is ReconciledUsageStream
        assert received == result


@pytest.mark.parametrize("construction", ["copy", "construct"])
@pytest.mark.parametrize("receiver_name", ["stream", "result"])
def test_result_receivers_reject_unknown_from_unchecked_model_construction(
    construction: str,
    receiver_name: str,
) -> None:
    result = reconcile_usage_events([_event()])
    valid = result.streams[0] if receiver_name == "stream" else result
    receiver = (
        ReconciledUsageStream
        if receiver_name == "stream"
        else UsageReconciliation
    )
    if construction == "copy":
        invalid = valid.model_copy(update={"future_constraint": "deny"})
    else:
        stored = _MODEL_DICT_DESCRIPTOR.__get__(valid, BaseModel)
        invalid = receiver.model_construct(**dict(dict.items(stored)))
        invalid_stored = _MODEL_DICT_DESCRIPTOR.__get__(invalid, BaseModel)
        dict.__setitem__(invalid_stored, "future_constraint", "deny")

    assert _validation_signature(receiver, invalid) == [
        (
            "extra_forbidden",
            ("future_constraint",),
            "Extra inputs are not permitted",
        )
    ]


@pytest.mark.parametrize("receiver_name", ["stream", "ledger", "result"])
def test_result_receivers_preserve_dict_and_json_validation_paths(
    receiver_name: str,
) -> None:
    result = reconcile_usage_events([_event()])
    if receiver_name == "stream":
        valid = result.streams[0]
        receiver = ReconciledUsageStream
    elif receiver_name == "ledger":
        valid = result.ledgers[0]
        receiver = ReconciledLedger
    else:
        valid = result
        receiver = UsageReconciliation

    payload = valid.model_dump(mode="python")
    assert receiver.model_validate(payload) == valid
    assert receiver.model_validate(MappingProxyType(payload)) == valid
    assert receiver(**payload) == valid
    assert receiver.model_validate_json(valid.model_dump_json()) == valid


@pytest.mark.parametrize("receiver_name", ["stream", "result"])
@pytest.mark.parametrize(
    "extra_storage",
    [
        pytest.param("absent", id="absent"),
        pytest.param(None, id="none"),
        pytest.param({}, id="empty-dict"),
        pytest.param(MappingProxyType({}), id="empty-mapping"),
        pytest.param([], id="malformed"),
    ],
)
def test_result_receivers_preserve_native_extra_storage_protocol(
    receiver_name: str,
    extra_storage: object,
) -> None:
    result = reconcile_usage_events([_event()])
    valid = result.streams[0] if receiver_name == "stream" else result
    receiver = (
        ReconciledUsageStream
        if receiver_name == "stream"
        else UsageReconciliation
    )
    value = valid.model_copy()
    if extra_storage == "absent":
        _MODEL_EXTRAS_DESCRIPTOR.__delete__(value)
        with pytest.raises(AttributeError):
            _MODEL_EXTRAS_DESCRIPTOR.__get__(value, BaseModel)
    else:
        _MODEL_EXTRAS_DESCRIPTOR.__set__(value, extra_storage)

    if isinstance(extra_storage, list):
        with pytest.raises(
            ValueError, match="contract extra storage must be a mapping"
        ):
            receiver.model_validate(value)
    else:
        assert receiver.model_validate(value) == valid


def test_result_receivers_preserve_semantic_recomputation() -> None:
    result = reconcile_usage_events([_event()])
    invalid_stream = result.streams[0].model_copy(
        update={"known_token_subtotal": 0}
    )
    with pytest.raises(
        ValidationError, match="stream subtotal does not match token usage"
    ):
        ReconciledUsageStream.model_validate(invalid_stream)

    invalid_ledger = result.ledgers[0].model_copy(
        update={"known_token_subtotal": 999, "total_tokens": 999}
    )
    invalid_result = result.model_copy(
        update={"ledgers": (invalid_ledger, *result.ledgers[1:])}
    )
    with pytest.raises(
        ValidationError,
        match="service ledger known_token_subtotal does not match streams",
    ):
        UsageReconciliation.model_validate(invalid_result)


@pytest.mark.parametrize("receiver_name", ["stream", "result"])
def test_result_receivers_reject_native_cycles(receiver_name: str) -> None:
    result = reconcile_usage_events([_event()])
    if receiver_name == "stream":
        value = result.streams[0].model_copy()
        stored = _MODEL_DICT_DESCRIPTOR.__get__(value, BaseModel)
        dict.__setitem__(stored, "attribution", value)
        receiver = ReconciledUsageStream
    else:
        value = result.model_copy()
        stored = _MODEL_DICT_DESCRIPTOR.__get__(value, BaseModel)
        dict.__setitem__(stored, "streams", (value,))
        receiver = UsageReconciliation

    with pytest.raises(ValueError, match="cyclic input graph"):
        receiver.model_validate(value)


def test_result_snapshot_freezes_root_extras_before_nested_traversal() -> None:
    result = reconcile_usage_events([_event()]).model_copy()
    extras = {"future_constraint": "deny"}
    _MODEL_EXTRAS_DESCRIPTOR.__set__(result, extras)

    class ClearingTuple(tuple):
        def __iter__(self):
            self.iteration_calls += 1
            dict.clear(extras)
            return super().__iter__()

    streams = ClearingTuple(result.streams)
    streams.iteration_calls = 0
    stored = _MODEL_DICT_DESCRIPTOR.__get__(result, BaseModel)
    dict.__setitem__(stored, "streams", streams)

    assert tuple(dict.items(extras)) == (("future_constraint", "deny"),)
    assert _validation_signature(UsageReconciliation, result) == [
        (
            "extra_forbidden",
            ("future_constraint",),
            "Extra inputs are not permitted",
        )
    ]
    assert streams.iteration_calls == 0
    assert tuple(dict.items(extras)) == (("future_constraint", "deny"),)


def test_stream_rejects_existing_attribution_with_hidden_extra() -> None:
    stream = reconcile_usage_events([_event()]).streams[0]
    invalid_attribution = stream.attribution.model_copy(
        update={"authority": "admin"}
    )
    payload = stream.model_dump(mode="python")
    payload["attribution"] = invalid_attribution

    _assert_attribution_authority_rejected(payload)


def test_existing_stream_revalidation_rejects_attribution_hidden_extra() -> None:
    stream = reconcile_usage_events([_event()]).streams[0]
    invalid_attribution = stream.attribution.model_copy(
        update={"authority": "admin"}
    )
    invalid_stream = stream.model_copy(
        update={"attribution": invalid_attribution}
    )

    _assert_attribution_authority_rejected(invalid_stream)


def test_empty_input_cannot_fabricate_tenant_or_ledger_completeness() -> None:
    assert _error_code([]) == "empty_reconciliation"


def test_absent_ledgers_are_unreported_not_complete_zeroes() -> None:
    result = reconcile_usage_events([_event()])
    ledgers = {ledger.ledger: ledger for ledger in result.ledgers}

    assert result.tenant_id == "tenant-1"
    assert ledgers["service"].coverage == "reported"
    assert ledgers["service"].missing_reason is None
    assert ledgers["service"].total_tokens == 15
    for ledger_name in ("consuming_agent", "evaluation"):
        ledger = ledgers[ledger_name]
        assert ledger.coverage == "unreported"
        assert ledger.missing_reason == "no_events"
        assert ledger.known_token_subtotal == 0
        assert ledger.total_tokens is None
        assert ledger.incomplete_attempt_ids == ()


def test_explicit_measured_zero_is_a_complete_reported_zero() -> None:
    zero = _event(
        usage=_usage(
            input_tokens=0,
            output_tokens=0,
            cache_read_tokens=0,
            reasoning_tokens=0,
        )
    )
    result = reconcile_usage_events([zero])
    service = result.ledgers[0]

    assert service.coverage == "reported"
    assert service.missing_reason is None
    assert service.known_token_subtotal == 0
    assert service.total_tokens == 0


def test_reconciliation_rejects_unchecked_negative_nested_counter() -> None:
    event = _event()
    assert event.usage is not None
    negative_input = event.usage.input_tokens.model_copy(update={"value": -1})
    invalid_usage = event.usage.model_copy(update={"input_tokens": negative_input})
    invalid_event = event.model_copy(update={"usage": invalid_usage})

    assert _error_code([invalid_event]) == "invalid_event"


def test_reconciliation_rejects_unchecked_contradictory_event_state() -> None:
    invalid_event = _event().model_copy(
        update={
            "status": "unavailable",
            "usage_missing_reason": "provider_did_not_report",
        }
    )

    assert invalid_event.usage is not None
    assert _error_code([invalid_event]) == "invalid_event"


def test_reconciliation_rejects_direct_nested_mutation() -> None:
    invalid_event = _event()
    assert invalid_event.usage is not None
    invalid_event.usage.input_tokens.value = -1

    assert _error_code([invalid_event]) == "invalid_event"


def test_reconciliation_rejects_hidden_extra_from_unchecked_copy() -> None:
    event = _event()
    assert event.usage is not None
    invalid_input = event.usage.input_tokens.model_copy(update={"authority": True})
    invalid_usage = event.usage.model_copy(update={"input_tokens": invalid_input})
    invalid_event = event.model_copy(update={"usage": invalid_usage})

    assert _error_code([invalid_event]) == "invalid_event"


@pytest.mark.parametrize("target_name", ["event", "attribution"])
def test_reconciliation_reads_native_model_storage_without_instance_dispatch(
    target_name: str,
) -> None:
    if target_name == "event":
        event: UsageEvent = _HidingStorageUsageEvent.model_validate(
            _event().model_dump(mode="python")
        )
        target = event
    else:
        attribution = _HidingStorageAttribution.model_validate(
            _attribution().model_dump(mode="python")
        )
        event = _event().model_copy(update={"attribution": attribution})
        target = attribution
    native_storage = object.__getattribute__(target, "__dict__")
    dict.__setitem__(native_storage, "future_constraint", "deny")

    assert ("future_constraint", "deny") in tuple(dict.items(native_storage))
    assert "future_constraint" not in target.__dict__
    _assert_canonical_invalid_event(event)


def test_reconciliation_freezes_parent_extras_before_child_traversal() -> None:
    parent_extras: dict[str, object] = {"future_constraint": "deny"}

    class ClearingRecipients(tuple):
        def __iter__(self) -> Iterator[object]:
            self.iteration_calls += 1
            dict.clear(parent_extras)
            return super().__iter__()

    class MutatingAttribution(AttributionSnapshot):
        pass

    attribution = MutatingAttribution.model_validate(
        _attribution().model_dump(mode="python")
    )
    attribution_storage = object.__getattribute__(attribution, "__dict__")
    recipients = ClearingRecipients(("provider-1",))
    recipients.iteration_calls = 0
    dict.__setitem__(attribution_storage, "recipients", recipients)
    event = _event().model_copy(update={"attribution": attribution})
    object.__setattr__(event, "__pydantic_extra__", parent_extras)

    assert tuple(dict.items(parent_extras)) == (("future_constraint", "deny"),)
    _assert_canonical_invalid_event(event)
    assert recipients.iteration_calls == 0
    assert tuple(dict.items(parent_extras)) == (("future_constraint", "deny"),)

    dict.clear(parent_extras)
    result = reconcile_usage_events([event])
    assert result.streams[0].head_event_id == "event-1"
    assert result.streams[0].total_tokens == 15


def test_reconciliation_accepts_valid_model_storage_subclass() -> None:
    event = _HidingStorageUsageEvent.model_validate(
        _event().model_dump(mode="python")
    )

    result = reconcile_usage_events([event])

    assert result.streams[0].head_event_id == "event-1"
    assert result.streams[0].total_tokens == 15


@pytest.mark.parametrize(
    "extra_storage",
    [
        pytest.param([], id="empty-list"),
        pytest.param((), id="empty-tuple"),
        pytest.param("", id="empty-string"),
        pytest.param(0, id="zero"),
        pytest.param(False, id="false"),
        pytest.param(["hidden"], id="nonempty-list"),
    ],
)
def test_reconciliation_rejects_malformed_event_extra_storage(
    extra_storage: object,
) -> None:
    event = _event()
    object.__setattr__(event, "__pydantic_extra__", extra_storage)

    with pytest.raises(UsageReconciliationError) as caught:
        reconcile_usage_events([event])

    assert caught.value.code == "invalid_event"
    assert str(caught.value) == (
        "invalid_event: an input event failed contract validation"
    )


@pytest.mark.parametrize(
    "extra_storage",
    [None, {}, MappingProxyType({})],
    ids=["none", "empty-dict", "empty-mapping"],
)
def test_reconciliation_accepts_absent_or_empty_mapping_extra_storage(
    extra_storage: object,
) -> None:
    event = _event()
    object.__setattr__(event, "__pydantic_extra__", extra_storage)

    result = reconcile_usage_events([event])

    assert result.streams[0].head_event_id == event.event_id
    assert result.streams[0].total_tokens == 15


@pytest.mark.parametrize(
    "extra_storage",
    [
        pytest.param({"authority": True}, id="unknown-field"),
        pytest.param({"event_id": "event-1"}, id="overlapping-field"),
    ],
)
def test_reconciliation_rejects_mapping_event_extra_storage(
    extra_storage: object,
) -> None:
    event = _event()
    object.__setattr__(event, "__pydantic_extra__", extra_storage)

    assert _error_code([event]) == "invalid_event"


@pytest.mark.parametrize("target_name", ["event", "attribution", "quantity"])
def test_reconciliation_rejects_hidden_native_extra_backing(
    target_name: str,
) -> None:
    event = _event()
    assert event.usage is not None
    targets = {
        "event": event,
        "attribution": event.attribution,
        "quantity": event.usage.input_tokens,
    }
    extras = _HiddenBackingExtras({"authority": True})
    object.__setattr__(targets[target_name], "__pydantic_extra__", extras)

    assert tuple(dict.items(extras)) == (("authority", True),)
    _assert_canonical_invalid_event(event)
    _assert_no_hidden_backing_override_reads(extras)


@pytest.mark.parametrize(
    ("target_name", "field_name", "extra_value"),
    [
        ("event", "tenant_id", "shadow-tenant"),
        ("attribution", "producer", "shadow-producer"),
        ("quantity", "value", 99),
    ],
)
def test_reconciliation_rejects_hidden_native_declared_overlap(
    target_name: str,
    field_name: str,
    extra_value: object,
) -> None:
    event = _event()
    assert event.usage is not None
    targets = {
        "event": event,
        "attribution": event.attribution,
        "quantity": event.usage.input_tokens,
    }
    extras = _HiddenBackingExtras({field_name: extra_value})
    object.__setattr__(targets[target_name], "__pydantic_extra__", extras)

    _assert_canonical_invalid_event(event)
    _assert_no_hidden_backing_override_reads(extras)


def test_reconciliation_rejects_hidden_native_subclass_field_overlap() -> None:
    class ExtendedUsageEvent(UsageEvent):
        audit_marker: str = "marker"

    event = ExtendedUsageEvent.model_validate(_event().model_dump(mode="python"))
    event.__dict__.pop("audit_marker")
    extras = _HiddenBackingExtras({"audit_marker": "override"})
    object.__setattr__(event, "__pydantic_extra__", extras)

    _assert_canonical_invalid_event(event)
    _assert_no_hidden_backing_override_reads(extras)


def test_reconciliation_accepts_empty_hidden_native_backing() -> None:
    event = _event()
    extras = _HiddenBackingExtras({})
    object.__setattr__(event, "__pydantic_extra__", extras)

    result = reconcile_usage_events([event])

    assert result.streams[0].head_event_id == event.event_id
    _assert_no_hidden_backing_override_reads(extras)


def test_reconciliation_rejects_split_view_declared_extra_storage() -> None:
    event = _event(tenant_id="wrong-tenant")
    extras = _SplitViewExtras({"tenant_id": "tenant-1"})
    object.__setattr__(
        event,
        "__pydantic_extra__",
        extras,
    )

    _assert_canonical_invalid_event(event)
    assert extras.iteration_calls == 1
    assert extras.items_calls == 1


def test_reconciliation_retains_split_view_unknown_extra_for_rejection() -> None:
    event = _event()
    extras = _SplitViewExtras({"authority": True})
    object.__setattr__(
        event,
        "__pydantic_extra__",
        extras,
    )

    _assert_canonical_invalid_event(event)
    assert extras.iteration_calls == 1
    assert extras.items_calls == 1


def test_reconciliation_uses_one_stable_extra_items_snapshot() -> None:
    event = _event()
    extras = _ChangingItemsExtras()
    object.__setattr__(event, "__pydantic_extra__", extras)

    result = reconcile_usage_events([event])

    assert extras.iteration_calls == 1
    assert extras.items_calls == 1
    assert result.streams[0].tenant_id == "tenant-1"


def test_reconciliation_rejects_falsey_populated_extra_storage() -> None:
    event = _event()
    object.__setattr__(
        event,
        "__pydantic_extra__",
        _FalseyPopulatedExtras({"authority": True}),
    )

    _assert_canonical_invalid_event(event)


def test_reconciliation_accepts_genuinely_empty_split_view_mapping() -> None:
    event = _event()
    extras = _SplitViewExtras({})
    object.__setattr__(
        event,
        "__pydantic_extra__",
        extras,
    )

    result = reconcile_usage_events([event])

    assert result.streams[0].head_event_id == "event-1"
    assert extras.iteration_calls == 1
    assert extras.items_calls == 1


@pytest.mark.parametrize(
    ("target_name", "field_name", "extra_value"),
    [
        ("event", "tenant_id", "shadow-tenant"),
        ("attribution", "producer", "shadow-producer"),
    ],
)
def test_reconciliation_rejects_inverse_items_declared_overlap(
    target_name: str,
    field_name: str,
    extra_value: str,
) -> None:
    event = _event()
    targets = {
        "event": event,
        "attribution": event.attribution,
    }
    extras = _NonDictInverseItemsExtras({field_name: extra_value})
    object.__setattr__(targets[target_name], "__pydantic_extra__", extras)

    _assert_canonical_invalid_event(event)
    assert extras.iteration_calls == 1
    assert extras.items_calls == 1


@pytest.mark.parametrize("target_name", ["event", "attribution"])
def test_reconciliation_rejects_inverse_items_moved_subclass_default(
    target_name: str,
) -> None:
    class ExtendedUsageEvent(UsageEvent):
        audit_marker: str = "marker"

    class ExtendedAttribution(AttributionSnapshot):
        audit_marker: str = "marker"

    if target_name == "event":
        event = ExtendedUsageEvent.model_validate(_event().model_dump(mode="python"))
        target = event
    else:
        attribution = ExtendedAttribution.model_validate(
            _attribution().model_dump(mode="python")
        )
        event = _event().model_copy(update={"attribution": attribution})
        target = attribution
    target.__dict__.pop("audit_marker")
    extras = _NonDictInverseItemsExtras({"audit_marker": "override"})
    object.__setattr__(target, "__pydantic_extra__", extras)

    _assert_canonical_invalid_event(event)
    assert extras.iteration_calls == 1
    assert extras.items_calls == 1


@pytest.mark.parametrize("target_name", ["event", "attribution"])
def test_reconciliation_preserves_inverse_items_unknown_omission(
    target_name: str,
) -> None:
    event = _event()
    targets = {
        "event": event,
        "attribution": event.attribution,
    }
    extras = _NonDictInverseItemsExtras({"future_semantics": "omit"})
    object.__setattr__(targets[target_name], "__pydantic_extra__", extras)

    result = reconcile_usage_events([event])

    assert result.streams[0].head_event_id == "event-1"
    assert "future_semantics" not in result.model_dump_json()
    assert extras.iteration_calls == 1
    assert extras.items_calls == 1


@pytest.mark.parametrize("target_name", ["event", "attribution"])
def test_reconciliation_rejects_dict_inverse_items_unknown_backing(
    target_name: str,
) -> None:
    event = _event()
    targets = {
        "event": event,
        "attribution": event.attribution,
    }
    extras = _InverseItemsExtras({"future_semantics": "reject"})
    object.__setattr__(targets[target_name], "__pydantic_extra__", extras)

    _assert_canonical_invalid_event(event)
    assert extras.iteration_calls == 0
    assert extras.items_calls == 0


@pytest.mark.parametrize(
    ("target_name", "field_name"),
    [
        ("event", "event_id"),
        ("attribution", "producer"),
    ],
)
def test_reconciliation_preserves_inverse_items_required_missing_rejection(
    target_name: str,
    field_name: str,
) -> None:
    event = _event()
    targets = {
        "event": event,
        "attribution": event.attribution,
    }
    target = targets[target_name]
    stored_value = target.__dict__.pop(field_name)
    extras = _NonDictInverseItemsExtras({field_name: stored_value})
    object.__setattr__(target, "__pydantic_extra__", extras)

    _assert_canonical_invalid_event(event)
    assert extras.iteration_calls == 1
    assert extras.items_calls == 1


@pytest.mark.parametrize(
    ("target_name", "field_name"),
    [
        ("event", "event_id"),
        ("quantity", "value"),
        ("attribution", "producer"),
    ],
)
def test_reconciliation_rejects_declared_field_moved_to_extra_storage(
    target_name: str,
    field_name: str,
) -> None:
    event = _event()
    assert event.usage is not None
    targets = {
        "event": event,
        "quantity": event.usage.input_tokens,
        "attribution": event.attribution,
    }
    target = targets[target_name]
    stored_value = target.__dict__.pop(field_name)
    object.__setattr__(target, "__pydantic_extra__", {field_name: stored_value})

    _assert_canonical_invalid_event(event)


def test_reconciliation_rejects_constructed_missing_field_supplied_by_extras() -> None:
    values = dict(_event().__dict__)
    event_id = values.pop("event_id")
    event = UsageEvent.model_construct(**values)
    object.__setattr__(event, "__pydantic_extra__", {"event_id": event_id})

    _assert_canonical_invalid_event(event)


@pytest.mark.parametrize(
    ("target_name", "field_name"),
    [
        ("event", "event_id"),
        ("quantity", "value"),
        ("attribution", "producer"),
    ],
)
def test_reconciliation_rejects_missing_declared_field_without_extras(
    target_name: str,
    field_name: str,
) -> None:
    event = _event()
    assert event.usage is not None
    targets = {
        "event": event,
        "quantity": event.usage.input_tokens,
        "attribution": event.attribution,
    }
    targets[target_name].__dict__.pop(field_name)

    _assert_canonical_invalid_event(event)


def test_snapshot_rejects_subclass_default_field_supplied_by_extras() -> None:
    class DefaultedEventId(UsageEvent):
        event_id: str = "default-event"

    values = _event().model_dump()
    values.pop("event_id")
    event = DefaultedEventId.model_validate(values)
    event.__dict__.pop("event_id")
    object.__setattr__(
        event,
        "__pydantic_extra__",
        {"event_id": "default-event"},
    )

    with pytest.raises(ValueError, match="extra storage overlaps stored fields"):
        usage_reconcile_contracts._native_snapshot(event)
    _assert_canonical_invalid_event(event)


def test_snapshot_rejects_subclass_only_field_supplied_by_extras() -> None:
    class ExtendedUsageEvent(UsageEvent):
        audit_marker: str = "marker"

    event = ExtendedUsageEvent.model_validate(_event().model_dump())
    event.__dict__.pop("audit_marker")
    object.__setattr__(
        event,
        "__pydantic_extra__",
        {"audit_marker": "marker"},
    )

    with pytest.raises(ValueError, match="extra storage overlaps stored fields"):
        usage_reconcile_contracts._native_snapshot(event)
    _assert_canonical_invalid_event(event)


@pytest.mark.parametrize(
    "extra_storage",
    [
        pytest.param([], id="malformed"),
        pytest.param({"authority": True}, id="unknown-field"),
        pytest.param({"value": 10}, id="overlapping-field"),
    ],
)
def test_reconciliation_rejects_nested_quantity_extra_storage(
    extra_storage: object,
) -> None:
    event = _event()
    assert event.usage is not None
    object.__setattr__(
        event.usage.input_tokens,
        "__pydantic_extra__",
        extra_storage,
    )

    assert _error_code([event]) == "invalid_event"


def test_reconciliation_rejects_nested_split_view_declared_extra_storage() -> None:
    event = _event()
    assert event.usage is not None
    object.__setattr__(
        event.usage.input_tokens,
        "__pydantic_extra__",
        _SplitViewExtras({"value": 99}),
    )

    _assert_canonical_invalid_event(event)


def test_reconciliation_rejects_cyclic_mutated_input_graph() -> None:
    invalid_event = _event()
    invalid_event.usage = invalid_event  # type: ignore[assignment]

    assert _error_code([invalid_event]) == "invalid_event"


def test_reconciliation_result_is_isolated_from_later_input_mutation() -> None:
    event = _event()
    result = reconcile_usage_events([event])
    assert event.usage is not None
    assert result.streams[0].usage is not None

    event.usage.input_tokens.value = 0

    assert result.streams[0].usage.input_tokens.value == 10
    assert result.streams[0].total_tokens == 15


@pytest.mark.parametrize(
    "changes",
    [
        {"missing_reason": None},
        {"known_token_subtotal": 1},
        {"total_tokens": 0},
        {"incomplete_attempt_ids": ("attempt-1",)},
    ],
)
def test_unreported_ledger_cannot_claim_observed_or_complete_usage(
    changes: dict[str, object],
) -> None:
    values: dict[str, object] = {
        "ledger": "consuming_agent",
        "coverage": "unreported",
        "missing_reason": "no_events",
        "known_token_subtotal": 0,
        "total_tokens": None,
        "incomplete_attempt_ids": (),
    }
    values.update(changes)

    with pytest.raises(ValidationError):
        ReconciledLedger(**values)


def test_correction_is_order_independent_and_exact_duplicate_is_idempotent() -> None:
    pending = _event(
        event_id="pending",
        status="pending",
        attempt_outcome="in_progress",
        usage=None,
    )
    final = _event(event_id="final", predecessor_event_id="pending")

    forward = reconcile_usage_events([pending, final, final])
    reverse = reconcile_usage_events([final, pending])

    assert forward == reverse
    assert len(forward.streams) == 1
    assert forward.streams[0].head_event_id == "final"
    assert forward.streams[0].total_tokens == 15


def test_three_event_correction_chain_is_permutation_invariant() -> None:
    root = _event(
        event_id="root",
        status="pending",
        attempt_outcome="in_progress",
        usage=None,
    )
    update = _event(
        event_id="update",
        predecessor_event_id="root",
        status="pending",
        attempt_outcome="in_progress",
        usage=_usage(input_tokens=8),
    )
    final = _event(event_id="final", predecessor_event_id="update")

    results = [
        reconcile_usage_events(order)
        for order in permutations((root, update, final))
    ]

    assert all(result == results[0] for result in results)
    assert results[0].streams[0].head_event_id == "final"


def test_same_event_id_with_different_payload_is_a_conflict() -> None:
    first = _event()
    conflicting = _event(attempt_id="attempt-elsewhere")
    assert _error_code([first, conflicting]) == "conflicting_event_id"


def test_unknown_predecessor_is_invalid() -> None:
    correction = _event(event_id="correction", predecessor_event_id="not-present")
    assert _error_code([correction]) == "unknown_predecessor"


def test_correction_cycle_is_invalid() -> None:
    first = _event(event_id="a", predecessor_event_id="b")
    second = _event(event_id="b", predecessor_event_id="a")
    assert _error_code([first, second]) == "correction_cycle"


def test_cycle_diagnostic_is_stable_across_event_permutations() -> None:
    first = _event(event_id="a", predecessor_event_id="b")
    second = _event(event_id="b", predecessor_event_id="a")

    signatures = {
        _error_signature(order) for order in permutations((first, second))
    }

    assert signatures == {
        ("correction_cycle", "correction chain containing a cycles")
    }


def test_competing_graph_diagnostics_are_stable_across_event_permutations() -> None:
    root = _event(
        event_id="z-root",
        status="pending",
        attempt_outcome="in_progress",
        usage=None,
    )
    cross_stream = _event(
        event_id="a-cross",
        predecessor_event_id="z-root",
        task_id="task-2",
    )
    branch = _event(event_id="b-branch", predecessor_event_id="z-root")
    unknown = _event(event_id="c-unknown", predecessor_event_id="not-present")

    signatures = {
        _error_signature(order)
        for order in permutations((root, cross_stream, branch, unknown))
    }

    assert signatures == {
        (
            "cross_stream_correction",
            "a-cross does not share its predecessor's stream identity",
        )
    }


def test_valid_independent_stream_output_and_idempotence_are_permutation_stable(
) -> None:
    events = (
        _event(event_id="z-event", attempt_id="attempt-3"),
        _event(event_id="a-event", attempt_id="attempt-1"),
        _event(event_id="m-event", attempt_id="attempt-2"),
    )
    expected = reconcile_usage_events(events)

    results = [
        reconcile_usage_events((*order, order[0]))
        for order in permutations(events)
    ]

    assert all(result == expected for result in results)


def test_cycle_validation_visits_each_event_once_for_long_valid_chain() -> None:
    class CountingEventMap(dict[str, UsageEvent]):
        reads = 0

        def __getitem__(self, event_id: str) -> UsageEvent:
            self.reads += 1
            return super().__getitem__(event_id)

    chain_length = 128
    events = CountingEventMap()
    predecessor_id = None
    for index in range(chain_length):
        event_id = f"event-{index}"
        events[event_id] = _event(
            event_id=event_id,
            predecessor_event_id=predecessor_id,
            status="pending",
            attempt_outcome="in_progress",
            usage=None,
        )
        predecessor_id = event_id

    usage_reconcile_contracts._validate_no_correction_cycles(events)

    assert events.reads == chain_length


def test_branched_correction_is_invalid() -> None:
    root = _event(event_id="root", status="pending", attempt_outcome="in_progress", usage=None)
    first = _event(event_id="first", predecessor_event_id="root")
    second = _event(event_id="second", predecessor_event_id="root")
    assert _error_code([root, first, second]) == "branched_correction"


def test_multiple_independent_roots_for_one_stream_are_invalid() -> None:
    first = _event(event_id="first-root")
    second = _event(event_id="second-root")

    assert _error_code([second, first]) == "multiple_stream_roots"


def test_mixed_tenants_cannot_be_aggregated() -> None:
    first = _event()
    second = _event(event_id="other", tenant_id="tenant-2", attempt_id="attempt-2")
    assert _error_code([first, second]) == "mixed_tenants"


@pytest.mark.parametrize("changed", ["tenant_id", "task_id", "attempt_id", "ledger"])
def test_correction_cannot_move_between_tenants_tasks_attempts_or_ledgers(
    changed: str,
) -> None:
    root = _event(event_id="root", status="pending", attempt_outcome="in_progress", usage=None)
    updates: dict[str, object] = {
        "event_id": "correction",
        "predecessor_event_id": "root",
    }
    updates[changed] = {
        "tenant_id": "tenant-2",
        "task_id": "task-2",
        "attempt_id": "attempt-2",
        "ledger": "consuming_agent",
    }[changed]
    correction = _event(**updates)

    expected = "mixed_tenants" if changed == "tenant_id" else "cross_stream_correction"
    assert _error_code([root, correction]) == expected


def test_distinct_attempts_remain_distinct_even_with_same_attribution() -> None:
    first = _event(event_id="one", attempt_id="attempt-1")
    second = _event(event_id="two", attempt_id="attempt-2")
    result = reconcile_usage_events([second, first])

    assert [stream.attempt_id for stream in result.streams] == [
        "attempt-1",
        "attempt-2",
    ]
    assert result.ledgers[0].total_tokens == 30


def test_unknown_additional_category_makes_total_unknown_not_zero() -> None:
    event = _event(
        usage=_usage(
            cache_read_tokens=None,
            cache_read_inclusion="additional_to_input",
        )
    )
    result = reconcile_usage_events([event])
    stream = result.streams[0]

    assert stream.known_token_subtotal == 15
    assert stream.total_tokens is None
    assert stream.incomplete_categories == ("cache_read_tokens",)
    assert result.ledgers[0].known_token_subtotal == 15
    assert result.ledgers[0].total_tokens is None


def test_unknown_included_subcategory_does_not_double_count_or_hide_base_total() -> None:
    event = _event(usage=_usage(cache_read_tokens=None))
    result = reconcile_usage_events([event])

    assert result.streams[0].total_tokens == 15
    assert result.streams[0].incomplete_categories == ()


def test_per_stream_token_subtotal_overflow_is_rejected() -> None:
    event = _event(
        usage=_usage(
            input_tokens=MAX_SAFE_INTEGER,
            output_tokens=1,
        )
    )

    assert _error_code([event]) == "counter_overflow"


def test_ledger_token_subtotal_overflow_is_rejected() -> None:
    first = _event(
        event_id="first",
        attempt_id="attempt-first",
        usage=_usage(input_tokens=MAX_SAFE_INTEGER, output_tokens=0),
    )
    second = _event(
        event_id="second",
        attempt_id="attempt-second",
        usage=_usage(input_tokens=1, output_tokens=0),
    )

    assert _error_code([second, first]) == "counter_overflow"


def test_late_usage_after_cancellation_replaces_unavailable_observation() -> None:
    unavailable = _event(
        event_id="cancelled",
        status="unavailable",
        attempt_outcome="cancelled",
        usage=None,
        usage_missing_reason="not_yet_reported",
    )
    late = _event(
        event_id="late",
        predecessor_event_id="cancelled",
        attempt_outcome="cancelled",
        late_after_cancellation=True,
    )
    result = reconcile_usage_events([late, unavailable])

    assert result.streams[0].head_event_id == "late"
    assert result.streams[0].late_after_cancellation is True
    assert result.streams[0].total_tokens == 15


def test_late_marker_is_rejected_on_correction_chain_root() -> None:
    fabricated_root = _event(
        attempt_outcome="cancelled",
        late_after_cancellation=True,
    )

    assert _error_code([fabricated_root]) == "invalid_transition"


def test_late_marker_requires_recorded_predecessor_cancellation() -> None:
    pending = _event(
        event_id="pending",
        status="pending",
        attempt_outcome="in_progress",
        usage=None,
    )
    fabricated_late = _event(
        event_id="fabricated-late",
        predecessor_event_id="pending",
        attempt_outcome="cancelled",
        late_after_cancellation=True,
    )

    assert _error_code([fabricated_late, pending]) == "invalid_transition"


def test_valid_late_chain_is_order_independent_and_duplicate_idempotent() -> None:
    pending = _event(
        event_id="pending",
        status="pending",
        attempt_outcome="in_progress",
        usage=None,
    )
    cancelled = _event(
        event_id="cancelled",
        predecessor_event_id="pending",
        status="unavailable",
        attempt_outcome="cancelled",
        usage=None,
        usage_missing_reason="not_yet_reported",
    )
    late = _event(
        event_id="late",
        predecessor_event_id="cancelled",
        attempt_outcome="cancelled",
        late_after_cancellation=True,
    )
    corrected = _event(
        event_id="corrected",
        predecessor_event_id="late",
        attempt_outcome="cancelled",
        late_after_cancellation=True,
        usage=_usage(input_tokens=12),
    )

    forward_with_duplicate = reconcile_usage_events(
        [pending, cancelled, late, corrected, corrected]
    )
    reverse = reconcile_usage_events([corrected, late, cancelled, pending])

    assert forward_with_duplicate == reverse
    assert reverse.streams[0].head_event_id == "corrected"
    assert reverse.streams[0].late_after_cancellation is True
    assert reverse.streams[0].total_tokens == 17


def test_late_marker_cannot_be_removed_by_later_correction() -> None:
    cancelled = _event(
        event_id="cancelled",
        status="unavailable",
        attempt_outcome="cancelled",
        usage=None,
        usage_missing_reason="not_yet_reported",
    )
    late = _event(
        event_id="late",
        predecessor_event_id="cancelled",
        attempt_outcome="cancelled",
        late_after_cancellation=True,
    )
    removed = _event(
        event_id="removed",
        predecessor_event_id="late",
        attempt_outcome="cancelled",
        late_after_cancellation=False,
    )

    assert _error_code([removed, late, cancelled]) == "invalid_transition"


def test_late_usage_after_cancellation_must_remain_explicit() -> None:
    unavailable = _event(
        event_id="cancelled",
        status="unavailable",
        attempt_outcome="cancelled",
        usage=None,
        usage_missing_reason="not_yet_reported",
    )
    incorrectly_final = _event(
        event_id="late",
        predecessor_event_id="cancelled",
        attempt_outcome="cancelled",
    )
    assert _error_code([unavailable, incorrectly_final]) == "invalid_transition"


def test_final_usage_after_pending_cancellation_must_be_explicit() -> None:
    cancelled = _event(
        event_id="cancelled",
        status="pending",
        attempt_outcome="cancelled",
        usage=None,
    )
    incorrectly_final = _event(
        event_id="late",
        predecessor_event_id="cancelled",
        attempt_outcome="cancelled",
    )

    assert _error_code([incorrectly_final, cancelled]) == "invalid_transition"


def test_explicit_final_usage_after_pending_cancellation_is_retained() -> None:
    cancelled = _event(
        event_id="cancelled",
        status="pending",
        attempt_outcome="cancelled",
        usage=None,
    )
    late = _event(
        event_id="late",
        predecessor_event_id="cancelled",
        attempt_outcome="cancelled",
        late_after_cancellation=True,
    )

    result = reconcile_usage_events([late, cancelled])

    assert result.streams[0].head_event_id == "late"
    assert result.streams[0].late_after_cancellation is True


def test_final_cancellation_correction_requires_explicit_late_marker() -> None:
    cancelled = _event(
        event_id="cancelled",
        attempt_outcome="cancelled",
    )
    nonexplicit_correction = _event(
        event_id="nonexplicit",
        predecessor_event_id="cancelled",
        attempt_outcome="cancelled",
    )
    explicit_correction = _event(
        event_id="explicit",
        predecessor_event_id="cancelled",
        attempt_outcome="cancelled",
        late_after_cancellation=True,
    )

    assert _error_code([cancelled, nonexplicit_correction]) == "invalid_transition"
    result = reconcile_usage_events([cancelled, explicit_correction])
    assert result.streams[0].head_event_id == "explicit"
    assert result.streams[0].late_after_cancellation is True


def test_independent_final_cancelled_root_does_not_infer_late_usage() -> None:
    independent = _event(
        attempt_outcome="cancelled",
        late_after_cancellation=False,
    )

    result = reconcile_usage_events([independent])

    assert result.streams[0].late_after_cancellation is False
