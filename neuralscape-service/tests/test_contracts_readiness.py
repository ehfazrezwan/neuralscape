"""Contract tests for capability-specific readiness evidence."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from types import MappingProxyType

import pytest
from pydantic import BaseModel, ValidationError

from contracts_readiness import (
    CapabilityReadiness,
    TenantReadinessPublication,
    capability_is_ready,
    validate_readiness_publication,
)


VERSION = "candidate-v1"
OBSERVED_AT = datetime(2026, 9, 28, 0, 0, tzinfo=timezone.utc)


class InconsistentExtraMapping(Mapping[object, object]):
    """Expose entries through items while hiding them from normal iteration."""

    def __init__(self, entries: dict[object, object]) -> None:
        self._entries = tuple(entries.items())

    def __getitem__(self, key: object) -> object:
        return dict(self._entries)[key]

    def __iter__(self):
        return iter(())

    def __len__(self) -> int:
        return 0

    def items(self):
        return self._entries


class ChangingItemsMapping(Mapping[object, object]):
    """Return one entry snapshot first and a different one thereafter."""

    def __init__(
        self,
        first: tuple[tuple[object, object], ...],
        later: tuple[tuple[object, object], ...],
    ) -> None:
        self._first = first
        self._later = later
        self.calls = 0

    def __getitem__(self, key: object) -> object:
        raise KeyError(key)

    def __iter__(self):
        return iter(())

    def __len__(self) -> int:
        return 0

    def items(self):
        self.calls += 1
        return self._first if self.calls == 1 else self._later


class FalseyEntries(dict[object, object]):
    def __bool__(self) -> bool:
        return False


class HiddenNativeBacking(dict[object, object]):
    """Hide native dict storage from every overridable mapping view."""

    def __init__(self, entries: dict[object, object]) -> None:
        dict.__init__(self, entries)
        self.length_calls = 0
        self.iteration_calls = 0
        self.keys_calls = 0
        self.items_calls = 0
        self.boolean_calls = 0

    def __len__(self) -> int:
        self.length_calls += 1
        return 0

    def __iter__(self):
        self.iteration_calls += 1
        return iter(())

    def keys(self):
        self.keys_calls += 1
        return ()

    def items(self):
        self.items_calls += 1
        return ()

    def __bool__(self) -> bool:
        self.boolean_calls += 1
        return False


class ClassTrapNativeBacking(HiddenNativeBacking):
    @property
    def __class__(self):
        raise AssertionError("native dict dispatch must not inspect __class__")


class DictSpoofMapping(Mapping[object, object]):
    """A non-dict mapping whose dynamic class view claims to be dict."""

    def __init__(self, entries: dict[object, object]) -> None:
        self._entries = tuple(entries.items())
        self.class_calls = 0
        self.length_calls = 0
        self.iteration_calls = 0
        self.items_calls = 0
        self.boolean_calls = 0

    @property
    def __class__(self):
        self.class_calls += 1
        return dict

    def __getitem__(self, key: object) -> object:
        return dict(self._entries)[key]

    def __iter__(self):
        self.iteration_calls += 1
        return (name for name, _field_value in self._entries)

    def __len__(self) -> int:
        self.length_calls += 1
        return len(self._entries)

    def items(self):
        self.items_calls += 1
        return self._entries

    def __bool__(self) -> bool:
        self.boolean_calls += 1
        return False


class HiddenModelStorage(dict[str, object]):
    """Retain native model state while exposing a sanitized public view."""

    def __init__(
        self,
        actual: dict[str, object],
        visible: dict[str, object],
    ) -> None:
        dict.__init__(self, actual)
        self._visible = tuple(visible.items())
        self.keys_calls = 0
        self.items_calls = 0

    def keys(self):
        self.keys_calls += 1
        return dict(self._visible).keys()

    def items(self):
        self.items_calls += 1
        return self._visible


def replace_model_storage(
    model: object,
    *,
    stored_name: str,
    stored_value: object,
) -> HiddenModelStorage:
    visible = dict(
        dict.items(object.__getattribute__(model, "__dict__"))
    )
    actual = dict(visible)
    actual[stored_name] = stored_value
    backing = HiddenModelStorage(actual, visible)
    object.__setattr__(model, "__dict__", backing)
    return backing


PYDANTIC_DICT_DESCRIPTOR = BaseModel.__dict__["__dict__"]
PYDANTIC_EXTRA_DESCRIPTOR = BaseModel.__dict__["__pydantic_extra__"]
PYDANTIC_FIELDS_SET_DESCRIPTOR = BaseModel.__dict__["__pydantic_fields_set__"]


class RepairingName(str):
    def __new__(cls, value: str, repair):
        instance = super().__new__(cls, value)
        instance.repair = repair
        instance.armed = False
        instance.calls = 0
        return instance

    def __hash__(self) -> int:
        if self.armed:
            self.calls += 1
            self.repair()
        return str.__hash__(self)


class RepairingTuple(tuple):
    def __new__(cls, values: tuple[object, ...], repair):
        instance = super().__new__(cls, values)
        instance.repair = repair
        instance.calls = 0
        return instance

    def __iter__(self):
        self.calls += 1
        self.repair()
        return tuple.__iter__(self)


class RepairingList(list):
    def __init__(self, values: tuple[object, ...], repair) -> None:
        super().__init__(values)
        self.repair = repair
        self.calls = 0

    def __iter__(self):
        self.calls += 1
        self.repair()
        return list.__iter__(self)


class DivergentTuple(tuple):
    def __new__(
        cls,
        native_values: tuple[object, ...],
        visible_values: tuple[object, ...],
    ):
        instance = super().__new__(cls, native_values)
        instance.visible_values = visible_values
        instance.calls = 0
        return instance

    def __iter__(self):
        self.calls += 1
        return iter(self.visible_values)


class RepairingEmptyExtraMapping(Mapping[object, object]):
    def __init__(self, repair) -> None:
        self.repair = repair
        self.calls = 0

    def _run(self) -> None:
        self.calls += 1
        self.repair()

    def __getitem__(self, key: object) -> object:
        raise KeyError(key)

    def __iter__(self):
        self._run()
        return iter(())

    def __len__(self) -> int:
        self._run()
        return 0

    def items(self):
        self._run()
        return ()


def model_with_hidden_extra(
    model: BaseModel,
    *,
    reported_view: str,
) -> tuple[BaseModel, list[str]]:
    calls: list[str] = []
    target = model
    if reported_view != "ordinary":
        reported = None if reported_view == "none" else {}
        model_type = type(model)

        def masked_getattribute(self, name: str):
            if name == "__pydantic_extra__":
                calls.append(name)
                return reported
            return super(masked_type, self).__getattribute__(name)

        masked_type = type(
            f"Masked{model_type.__name__}",
            (model_type,),
            {"__getattribute__": masked_getattribute},
        )
        target = masked_type.model_validate(
            dict(dict.items(object.__getattribute__(model, "__dict__")))
        )
    PYDANTIC_EXTRA_DESCRIPTOR.__set__(
        target,
        {"hidden_unknown": "deny"},
    )
    return target, calls


def model_with_descriptor_storage(
    model: BaseModel,
    *,
    representation: str,
    include_unknown: bool,
) -> tuple[BaseModel, list[str]]:
    calls: list[str] = []
    target = model
    if representation == "property":
        model_type = type(model)

        def stored_getter(self):
            calls.append("__dict__")
            native = PYDANTIC_DICT_DESCRIPTOR.__get__(self, BaseModel)
            return {
                name: field_value
                for name, field_value in dict.items(native)
                if name != "hidden_unknown"
            }

        def stored_setter(self, value):
            PYDANTIC_DICT_DESCRIPTOR.__set__(self, value)

        masked_type = type(
            f"Descriptor{model_type.__name__}",
            (model_type,),
            {"__dict__": property(stored_getter, stored_setter)},
        )
        target = masked_type.model_validate(
            dict(
                dict.items(
                    PYDANTIC_DICT_DESCRIPTOR.__get__(model, BaseModel)
                )
            )
        )
    native = PYDANTIC_DICT_DESCRIPTOR.__get__(target, BaseModel)
    if include_unknown:
        dict.__setitem__(native, "hidden_unknown", "deny")
    calls.clear()
    return target, calls


class InverseItemsMapping(dict[object, object]):
    """Expose iterated keys while hiding every entry from items()."""

    def __init__(self, entries: dict[object, object]) -> None:
        super().__init__(entries)
        self.iteration_calls = 0
        self.items_calls = 0

    def __iter__(self):
        self.iteration_calls += 1
        return super().__iter__()

    def items(self):
        self.items_calls += 1
        return ()


class ObservedLengthMapping(Mapping[object, object]):
    """Expose only a configured length, with counters for every observed view."""

    def __init__(self, observed_length: int) -> None:
        self.observed_length = observed_length
        self.length_calls = 0
        self.iteration_calls = 0
        self.items_calls = 0
        self.boolean_calls = 0

    def __getitem__(self, key: object) -> object:
        raise KeyError(key)

    def __iter__(self):
        self.iteration_calls += 1
        return iter(())

    def __len__(self) -> int:
        self.length_calls += 1
        return self.observed_length

    def items(self):
        self.items_calls += 1
        return ()


class FalseBooleanLengthMapping(ObservedLengthMapping):
    def __bool__(self) -> bool:
        self.boolean_calls += 1
        return False


class TrueBooleanEmptyMapping(ObservedLengthMapping):
    def __bool__(self) -> bool:
        self.boolean_calls += 1
        return True


class DefaultItemsLengthProbe(Mapping[object, object]):
    """Instrument the standard Mapping.items() view and its source mapping."""

    def __init__(self) -> None:
        self.length_calls = 0
        self.iteration_calls = 0
        self.items_calls = 0

    def __getitem__(self, key: object) -> object:
        raise KeyError(key)

    def __iter__(self):
        self.iteration_calls += 1
        return iter(())

    def __len__(self) -> int:
        self.length_calls += 1
        return 0

    def items(self):
        self.items_calls += 1
        return super().items()


def observation(
    capability: str = "exact_reads",
    status: str = "healthy",
    observed_at: str = "2026-09-28T00:00:00Z",
    **updates: object,
) -> CapabilityReadiness:
    document: dict[str, object] = {
        "schema_version": VERSION,
        "capability": capability,
        "status": status,
        "observed_at": observed_at,
    }
    document.update(updates)
    return CapabilityReadiness.model_validate_json(json.dumps(document))


def publication(**updates: object) -> TenantReadinessPublication:
    document: dict[str, object] = {
        "schema_version": VERSION,
        "tenant_id": "tenant-a",
        "placement_generation": 12,
        "capabilities": [
            observation("api_acceptance", "healthy").model_dump(mode="json"),
            observation("exact_reads", "healthy").model_dump(mode="json"),
            observation(
                "graph_reads", "unavailable", detail_code="store-timeout"
            ).model_dump(mode="json"),
            observation("worker_processing", "unknown").model_dump(mode="json"),
        ],
    }
    document.update(updates)
    return TenantReadinessPublication.model_validate_json(json.dumps(document))


def evaluate(
    item: CapabilityReadiness,
    *,
    elapsed_seconds: int = 3,
    max_age_seconds: object = 10,
) -> bool:
    return capability_is_ready(
        item,
        evaluated_at=OBSERVED_AT + timedelta(seconds=elapsed_seconds),
        max_probe_age_seconds=max_age_seconds,
    )


def test_observation_requires_an_aware_timestamp() -> None:
    assert observation().observed_at == OBSERVED_AT

    with pytest.raises(ValidationError):
        observation(observed_at="2026-09-28T00:00:00")
    with pytest.raises(ValidationError):
        CapabilityReadiness.model_validate(
            {
                "schema_version": VERSION,
                "capability": "exact_reads",
                "status": "healthy",
                "observed_at": datetime(2026, 9, 28, 0, 0),
            }
        )


def test_unavailable_and_unknown_are_preserved_as_distinct_states() -> None:
    unavailable = observation(status="unavailable")
    unknown = observation(status="unknown")

    assert unavailable.status == "unavailable"
    assert unknown.status == "unknown"
    assert not evaluate(unavailable)
    assert not evaluate(unknown)


def test_readiness_is_evaluated_per_capability_and_explicit_time() -> None:
    exact_reads = observation("exact_reads", "healthy")
    graph_reads = observation("graph_reads", "unavailable")

    assert evaluate(exact_reads, elapsed_seconds=9, max_age_seconds=9)
    assert not evaluate(exact_reads, elapsed_seconds=10, max_age_seconds=9)
    assert not evaluate(graph_reads, elapsed_seconds=1, max_age_seconds=9)


def test_replayed_observation_ages_out_without_a_generation_change() -> None:
    exact_reads = observation("exact_reads", "healthy")

    assert evaluate(exact_reads, elapsed_seconds=1, max_age_seconds=5)
    assert not evaluate(exact_reads, elapsed_seconds=6, max_age_seconds=5)


@pytest.mark.parametrize(
    "status", ["degraded", "unavailable", "unknown", "disabled", "unconfigured"]
)
def test_nonhealthy_status_never_becomes_ready_from_recency(status: str) -> None:
    assert not evaluate(observation(status=status), elapsed_seconds=0)


def test_evaluation_time_cannot_precede_observation() -> None:
    with pytest.raises(ValueError, match="cannot precede"):
        evaluate(observation(), elapsed_seconds=-1)


def test_evaluation_time_must_be_explicit_and_aware() -> None:
    item = observation()

    with pytest.raises(ValidationError):
        capability_is_ready(
            item,
            evaluated_at=datetime(2026, 9, 28, 0, 0),
            max_probe_age_seconds=10,
        )


@pytest.mark.parametrize(
    "invalid_evaluated_at",
    ["2026-09-28T00:00:03Z", OBSERVED_AT.timestamp(), True],
)
def test_evaluation_time_rejects_coercible_non_datetime_scalars(
    invalid_evaluated_at: object,
) -> None:
    with pytest.raises(ValidationError):
        capability_is_ready(
            observation(),
            evaluated_at=invalid_evaluated_at,
            max_probe_age_seconds=10,
        )


@pytest.mark.parametrize(
    "invalid_max_age",
    [True, 1.0, -1, 9_007_199_254_740_992],
)
def test_readiness_helper_strictly_validates_max_age(
    invalid_max_age: object,
) -> None:
    with pytest.raises(ValidationError):
        evaluate(observation(), max_age_seconds=invalid_max_age)


def test_publication_keeps_partial_outage_visible() -> None:
    report = publication()
    by_capability = {item.capability: item for item in report.capabilities}

    assert evaluate(by_capability["exact_reads"])
    assert not evaluate(by_capability["graph_reads"])
    assert not evaluate(by_capability["worker_processing"])


def test_publication_rejects_duplicate_capability_observations() -> None:
    with pytest.raises(ValidationError, match="must be unique"):
        publication(
            capabilities=[
                observation("exact_reads", "healthy").model_dump(mode="json"),
                observation("exact_reads", "unavailable").model_dump(mode="json"),
            ]
        )


def test_publication_requires_at_least_one_capability() -> None:
    with pytest.raises(ValidationError, match="at least one"):
        publication(capabilities=[])


def test_publication_rejects_stale_and_future_generations() -> None:
    report = publication(placement_generation=12)

    for current_generation in (11, 13):
        with pytest.raises(ValueError, match="stale or unexpected"):
            validate_readiness_publication(
                report,
                expected_tenant_id="tenant-a",
                current_placement_generation=current_generation,
            )


def test_publication_accepts_only_exact_tenant_and_generation() -> None:
    report = publication()

    result = validate_readiness_publication(
        report,
        expected_tenant_id="tenant-a",
        current_placement_generation=12,
    )
    assert result == report
    assert result is not report
    with pytest.raises(ValueError, match="tenant_id does not match"):
        validate_readiness_publication(
            report,
            expected_tenant_id="tenant-b",
            current_placement_generation=12,
        )


def test_publication_revalidates_copied_empty_capability_set() -> None:
    corrupted = publication().model_copy(update={"capabilities": ()})

    with pytest.raises(ValidationError, match="at least one"):
        validate_readiness_publication(
            corrupted,
            expected_tenant_id="tenant-a",
            current_placement_generation=12,
        )


def test_publication_revalidates_copied_duplicate_capabilities() -> None:
    exact_reads = observation("exact_reads")
    corrupted = publication().model_copy(
        update={"capabilities": (exact_reads, exact_reads.model_copy())}
    )

    with pytest.raises(ValidationError, match="must be unique"):
        validate_readiness_publication(
            corrupted,
            expected_tenant_id="tenant-a",
            current_placement_generation=12,
        )


def test_publication_revalidates_nested_capability_graph() -> None:
    invalid = observation("exact_reads").model_copy(
        update={"status": "process_running"}
    )
    corrupted = publication().model_copy(update={"capabilities": (invalid,)})

    with pytest.raises(ValidationError):
        validate_readiness_publication(
            corrupted,
            expected_tenant_id="tenant-a",
            current_placement_generation=12,
        )


def test_publication_rejects_retained_copied_extra_fields() -> None:
    report = publication()
    top_level_extra = report.model_copy(update={"process_healthy": True})
    nested_extra = report.capabilities[0].model_copy(
        update={"process_healthy": True}
    )
    nested_corruption = report.model_copy(
        update={"capabilities": (nested_extra, *report.capabilities[1:])}
    )

    for corrupted in (top_level_extra, nested_corruption):
        with pytest.raises(ValueError, match="undeclared fields"):
            validate_readiness_publication(
                corrupted,
                expected_tenant_id="tenant-a",
                current_placement_generation=12,
            )


@pytest.mark.parametrize(
    "malformed_extra",
    [[], (), "", 0, False, ["unreviewed"]],
    ids=["list", "tuple", "string", "zero", "false", "nonempty-list"],
)
def test_publication_rejects_each_malformed_extra_storage_representation(
    malformed_extra: object,
) -> None:
    report = publication()
    object.__setattr__(report, "__pydantic_extra__", malformed_extra)

    with pytest.raises(ValueError, match="extra storage must be a mapping"):
        validate_readiness_publication(
            report,
            expected_tenant_id="tenant-a",
            current_placement_generation=12,
        )


@pytest.mark.parametrize(
    "extra_storage",
    [None, {}, MappingProxyType({})],
    ids=["none", "empty-dict", "empty-custom-mapping"],
)
def test_publication_accepts_absent_or_empty_mapping_extra_storage(
    extra_storage: object,
) -> None:
    report = publication()
    object.__setattr__(report, "__pydantic_extra__", extra_storage)

    result = validate_readiness_publication(
        report,
        expected_tenant_id="tenant-a",
        current_placement_generation=12,
    )

    assert result.tenant_id == "tenant-a"


@pytest.mark.parametrize(
    "stored_extra",
    [{"process_healthy": True}, {"tenant_id": "tenant-a"}],
    ids=["unknown", "declared-overlap"],
)
def test_publication_rejects_nonempty_mapping_extra_storage(
    stored_extra: dict[str, object],
) -> None:
    report = publication()
    object.__setattr__(report, "__pydantic_extra__", stored_extra)

    with pytest.raises(ValueError, match="undeclared fields"):
        validate_readiness_publication(
            report,
            expected_tenant_id="tenant-a",
            current_placement_generation=12,
        )


@pytest.mark.parametrize("location", ["direct", "nested"])
@pytest.mark.parametrize(
    "extra_kind",
    ["unknown", "declared"],
    ids=["unknown", "declared"],
)
def test_publication_rejects_entries_hidden_from_mapping_iteration(
    location: str,
    extra_kind: str,
) -> None:
    report = publication()
    target = report if location == "direct" else report.capabilities[0]
    declared_extra = (
        {"tenant_id": "tenant-a"}
        if location == "direct"
        else {"status": "healthy"}
    )
    stored_extra = (
        {"future_constraint": "reject"}
        if extra_kind == "unknown"
        else declared_extra
    )
    object.__setattr__(
        target,
        "__pydantic_extra__",
        InconsistentExtraMapping(stored_extra),
    )

    with pytest.raises(ValueError, match="undeclared fields"):
        validate_readiness_publication(
            report,
            expected_tenant_id="tenant-a",
            current_placement_generation=12,
        )


@pytest.mark.parametrize(
    "stored_extra",
    [{"future_constraint": "reject"}, {"tenant_id": "tenant-a"}],
    ids=["unknown", "declared"],
)
def test_publication_rejects_populated_falsey_extra_storage(
    stored_extra: dict[object, object],
) -> None:
    report = publication()
    object.__setattr__(report, "__pydantic_extra__", FalseyEntries(stored_extra))

    with pytest.raises(ValueError, match="undeclared fields"):
        validate_readiness_publication(
            report,
            expected_tenant_id="tenant-a",
            current_placement_generation=12,
        )


def test_publication_consumes_one_stable_extra_entry_snapshot() -> None:
    report = publication()
    populated_first = ChangingItemsMapping(
        (("future_constraint", "reject"),),
        (),
    )
    object.__setattr__(report, "__pydantic_extra__", populated_first)

    with pytest.raises(ValueError, match="undeclared fields"):
        validate_readiness_publication(
            report,
            expected_tenant_id="tenant-a",
            current_placement_generation=12,
        )
    assert populated_first.calls == 1

    report = publication()
    empty_first = ChangingItemsMapping(
        (),
        (("future_constraint", "reject"),),
    )
    object.__setattr__(report, "__pydantic_extra__", empty_first)

    result = validate_readiness_publication(
        report,
        expected_tenant_id="tenant-a",
        current_placement_generation=12,
    )

    assert result.tenant_id == "tenant-a"
    assert empty_first.calls == 1


@pytest.mark.parametrize(
    ("boundary", "location"),
    [
        ("publication", "direct"),
        ("publication", "nested"),
        ("capability", "direct"),
    ],
)
def test_readiness_boundaries_reject_iteration_visible_items_empty_extras(
    boundary: str,
    location: str,
) -> None:
    extras = InverseItemsMapping({"future_constraint": "reject"})
    if boundary == "publication":
        report = publication()
        target = report if location == "direct" else report.capabilities[0]
        object.__setattr__(target, "__pydantic_extra__", extras)

        with pytest.raises(ValueError, match="undeclared fields"):
            validate_readiness_publication(
                report,
                expected_tenant_id="tenant-a",
                current_placement_generation=12,
            )
    else:
        item = observation()
        object.__setattr__(item, "__pydantic_extra__", extras)

        with pytest.raises(ValueError, match="undeclared fields"):
            evaluate(item)

    assert extras.iteration_calls == 0
    assert extras.items_calls == 0


@pytest.mark.parametrize(
    ("boundary", "location"),
    [
        ("publication", "direct"),
        ("publication", "nested"),
        ("capability", "direct"),
    ],
)
@pytest.mark.parametrize(
    "extra_kind",
    ["unknown", "declared"],
    ids=["unknown", "declared-overlap"],
)
def test_readiness_boundaries_reject_hidden_native_dict_backing(
    boundary: str,
    location: str,
    extra_kind: str,
) -> None:
    if boundary == "publication":
        candidate = publication()
        target = candidate if location == "direct" else candidate.capabilities[0]
        declared_name = "tenant_id" if location == "direct" else "status"

        def validate() -> object:
            return validate_readiness_publication(
                candidate,
                expected_tenant_id="tenant-a",
                current_placement_generation=12,
            )

    else:
        target = observation()
        declared_name = "status"

        def validate() -> object:
            return evaluate(target)

    name = "future_constraint" if extra_kind == "unknown" else declared_name
    extras = HiddenNativeBacking({name: "reject"})
    object.__setattr__(target, "__pydantic_extra__", extras)

    with pytest.raises(ValueError, match="undeclared fields"):
        validate()

    assert tuple(dict.items(extras)) == ((name, "reject"),)
    assert (
        extras.length_calls,
        extras.iteration_calls,
        extras.keys_calls,
        extras.items_calls,
        extras.boolean_calls,
    ) == (0, 0, 0, 0, 0)


def test_publication_accepts_empty_native_dict_subclass_backing() -> None:
    candidate = publication()
    extras = HiddenNativeBacking({})
    object.__setattr__(candidate, "__pydantic_extra__", extras)

    result = validate_readiness_publication(
        candidate,
        expected_tenant_id="tenant-a",
        current_placement_generation=12,
    )

    assert result.tenant_id == "tenant-a"
    assert (
        extras.length_calls,
        extras.iteration_calls,
        extras.keys_calls,
        extras.items_calls,
        extras.boolean_calls,
    ) == (0, 0, 0, 0, 0)


@pytest.mark.parametrize(
    ("boundary", "location"),
    [
        ("publication", "direct"),
        ("publication", "nested"),
        ("capability", "direct"),
    ],
)
@pytest.mark.parametrize(
    ("extra_kind", "should_reject"),
    [("empty", False), ("unknown", True), ("declared", True)],
)
def test_readiness_boundaries_dispatch_native_dict_before_mapping_abc(
    boundary: str,
    location: str,
    extra_kind: str,
    should_reject: bool,
) -> None:
    if boundary == "publication":
        candidate = publication()
        target = candidate if location == "direct" else candidate.capabilities[0]

        def validate() -> object:
            return validate_readiness_publication(
                candidate,
                expected_tenant_id="tenant-a",
                current_placement_generation=12,
            )

    else:
        target = observation()

        def validate() -> object:
            return evaluate(target)

    declared_name = (
        "tenant_id"
        if boundary == "publication" and location == "direct"
        else "status"
    )
    stored = {
        "unknown": {"future_constraint": "reject"},
        "declared": {declared_name: "reject"},
    }.get(extra_kind, {})
    extras = ClassTrapNativeBacking(stored)
    object.__setattr__(target, "__pydantic_extra__", extras)

    if should_reject:
        with pytest.raises(ValueError, match="undeclared fields"):
            validate()
    else:
        validate()

    assert (
        extras.length_calls,
        extras.iteration_calls,
        extras.keys_calls,
        extras.items_calls,
        extras.boolean_calls,
    ) == (0, 0, 0, 0, 0)


@pytest.mark.parametrize(
    ("boundary", "location"),
    [
        ("publication", "direct"),
        ("publication", "nested"),
        ("capability", "direct"),
    ],
)
@pytest.mark.parametrize(
    ("extra_kind", "should_reject"),
    [("empty", False), ("unknown", True), ("declared", True)],
)
def test_readiness_boundaries_keep_spoofing_non_dict_on_mapping_path(
    boundary: str,
    location: str,
    extra_kind: str,
    should_reject: bool,
) -> None:
    if boundary == "publication":
        candidate = publication()
        target = candidate if location == "direct" else candidate.capabilities[0]

        def validate() -> object:
            return validate_readiness_publication(
                candidate,
                expected_tenant_id="tenant-a",
                current_placement_generation=12,
            )

    else:
        target = observation()

        def validate() -> object:
            return evaluate(target)

    declared_name = (
        "tenant_id"
        if boundary == "publication" and location == "direct"
        else "status"
    )
    stored = {
        "unknown": {"future_constraint": "reject"},
        "declared": {declared_name: "reject"},
    }.get(extra_kind, {})
    extras = DictSpoofMapping(stored)
    object.__setattr__(target, "__pydantic_extra__", extras)

    if should_reject:
        with pytest.raises(ValueError, match="undeclared fields"):
            validate()
    else:
        validate()

    assert extras.class_calls == 0  # the claimed class is never consulted: the real type decides
    assert extras.length_calls == 1
    assert extras.iteration_calls == 1
    assert extras.items_calls == 1
    assert extras.boolean_calls == 0


@pytest.mark.parametrize(
    ("boundary", "location"),
    [
        ("publication", "direct"),
        ("publication", "nested"),
        ("capability", "direct"),
    ],
)
@pytest.mark.parametrize(
    "corruption",
    ["hidden-unknown", "invalid-declared"],
)
def test_readiness_boundaries_use_frozen_native_model_storage(
    boundary: str,
    location: str,
    corruption: str,
) -> None:
    if boundary == "publication":
        candidate = publication()
        target = candidate if location == "direct" else candidate.capabilities[0]

        def validate() -> object:
            return validate_readiness_publication(
                candidate,
                expected_tenant_id="tenant-a",
                current_placement_generation=12,
            )

    else:
        target = observation()

        def validate() -> object:
            return evaluate(target)

    if corruption == "hidden-unknown":
        stored_name = "hidden_unknown"
        stored_value: object = "deny"
    elif boundary == "publication" and location == "direct":
        stored_name = "tenant_id"
        stored_value = ""
    else:
        stored_name = "status"
        stored_value = "not-a-status"
    backing = replace_model_storage(
        target,
        stored_name=stored_name,
        stored_value=stored_value,
    )

    if corruption == "hidden-unknown":
        with pytest.raises(ValueError, match="undeclared fields"):
            validate()
    else:
        with pytest.raises(ValidationError):
            validate()

    assert dict.__getitem__(backing, stored_name) == stored_value
    assert backing.keys_calls == 0
    assert backing.items_calls == 0


@pytest.mark.parametrize(
    ("boundary", "location"),
    [
        ("publication", "direct"),
        ("publication", "nested"),
        ("capability", "direct"),
    ],
)
@pytest.mark.parametrize("reported_view", ["ordinary", "none", "empty"])
def test_readiness_boundaries_read_model_owned_extra_storage(
    boundary: str,
    location: str,
    reported_view: str,
) -> None:
    if boundary == "publication":
        candidate = publication()
        if location == "direct":
            target, calls = model_with_hidden_extra(
                candidate,
                reported_view=reported_view,
            )
            candidate = target
        else:
            target, calls = model_with_hidden_extra(
                candidate.capabilities[0],
                reported_view=reported_view,
            )
            candidate = candidate.model_copy(
                update={
                    "capabilities": (
                        target,
                        *candidate.capabilities[1:],
                    )
                }
            )

        def validate() -> object:
            return validate_readiness_publication(
                candidate,
                expected_tenant_id="tenant-a",
                current_placement_generation=12,
            )

    else:
        target, calls = model_with_hidden_extra(
            observation(),
            reported_view=reported_view,
        )

        def validate() -> object:
            return evaluate(target)

    actual_extras = PYDANTIC_EXTRA_DESCRIPTOR.__get__(target, type(target))
    assert dict.__getitem__(actual_extras, "hidden_unknown") == "deny"
    with pytest.raises(ValueError, match="undeclared fields"):
        validate()
    assert calls == []


@pytest.mark.parametrize(
    ("boundary", "location"),
    [
        ("publication", "direct"),
        ("publication", "nested"),
        ("capability", "direct"),
    ],
)
@pytest.mark.parametrize("representation", ["ordinary", "property"])
@pytest.mark.parametrize("include_unknown", [False, True], ids=["valid", "unknown"])
def test_readiness_boundaries_read_base_model_dict_descriptor(
    boundary: str,
    location: str,
    representation: str,
    include_unknown: bool,
) -> None:
    if boundary == "publication":
        candidate = publication()
        if location == "direct":
            target, calls = model_with_descriptor_storage(
                candidate,
                representation=representation,
                include_unknown=include_unknown,
            )
            candidate = target
        else:
            target, calls = model_with_descriptor_storage(
                candidate.capabilities[0],
                representation=representation,
                include_unknown=include_unknown,
            )
            candidate = candidate.model_copy(
                update={
                    "capabilities": (
                        target,
                        *candidate.capabilities[1:],
                    )
                }
            )

        def validate() -> object:
            return validate_readiness_publication(
                candidate,
                expected_tenant_id="tenant-a",
                current_placement_generation=12,
            )

    else:
        target, calls = model_with_descriptor_storage(
            observation(),
            representation=representation,
            include_unknown=include_unknown,
        )

        def validate() -> object:
            return evaluate(target)

    native = PYDANTIC_DICT_DESCRIPTOR.__get__(target, BaseModel)
    assert dict.__contains__(native, "hidden_unknown") is include_unknown
    if include_unknown:
        with pytest.raises(ValueError, match="undeclared fields"):
            validate()
    else:
        validate()
    assert calls == []


@pytest.mark.parametrize(
    ("boundary", "location"),
    [
        ("publication", "direct"),
        ("publication", "nested"),
        ("capability", "direct"),
    ],
)
@pytest.mark.parametrize(
    ("case", "should_reject"),
    [
        ("length-one", True),
        ("length-one-false-bool", True),
        ("length-zero-true-bool", False),
    ],
)
def test_readiness_boundaries_use_length_without_mapping_truthiness(
    boundary: str,
    location: str,
    case: str,
    should_reject: bool,
) -> None:
    if case == "length-one":
        extras = ObservedLengthMapping(1)
    elif case == "length-one-false-bool":
        extras = FalseBooleanLengthMapping(1)
    else:
        extras = TrueBooleanEmptyMapping(0)

    if boundary == "publication":
        report = publication()
        target = report if location == "direct" else report.capabilities[0]
        object.__setattr__(target, "__pydantic_extra__", extras)

        def validate_publication() -> object:
            return validate_readiness_publication(
                report,
                expected_tenant_id="tenant-a",
                current_placement_generation=12,
            )

        validate = validate_publication
    else:
        item = observation()
        object.__setattr__(item, "__pydantic_extra__", extras)

        def validate_capability() -> object:
            return evaluate(item)

        validate = validate_capability

    if should_reject:
        with pytest.raises(ValueError, match="undeclared fields"):
            validate()
    else:
        validate()

    assert extras.length_calls == 1
    assert extras.iteration_calls == 1
    assert extras.items_calls == 1
    assert extras.boolean_calls == 0


def test_publication_default_items_view_does_not_repeat_length_hint() -> None:
    report = publication()
    extras = DefaultItemsLengthProbe()
    object.__setattr__(report, "__pydantic_extra__", extras)

    result = validate_readiness_publication(
        report,
        expected_tenant_id="tenant-a",
        current_placement_generation=12,
    )

    assert result.tenant_id == "tenant-a"
    assert extras.length_calls == 1
    assert extras.iteration_calls == 2
    assert extras.items_calls == 1


def test_publication_rejects_malformed_nested_extra_storage() -> None:
    report = publication()
    object.__setattr__(
        report.capabilities[0],
        "__pydantic_extra__",
        (),
    )

    with pytest.raises(ValueError, match="extra storage must be a mapping"):
        validate_readiness_publication(
            report,
            expected_tenant_id="tenant-a",
            current_placement_generation=12,
        )


def test_publication_rejects_cyclic_mutated_input_graph() -> None:
    report = publication()
    report.capabilities = (report,)  # type: ignore[assignment]

    with pytest.raises(ValueError, match="must be acyclic"):
        validate_readiness_publication(
            report,
            expected_tenant_id="tenant-a",
            current_placement_generation=12,
        )


def test_readiness_evaluation_revalidates_copied_observation_graph() -> None:
    invalid_status = observation(status="unavailable").model_copy(
        update={"status": "process_running"}
    )
    naive_timestamp = observation().model_copy(
        update={"observed_at": datetime(2026, 9, 28, 0, 0)}
    )

    with pytest.raises(ValidationError):
        evaluate(invalid_status)
    with pytest.raises(ValidationError):
        evaluate(naive_timestamp)


def test_readiness_evaluation_rejects_retained_copied_extra() -> None:
    corrupted = observation().model_copy(update={"probe_verified": False})

    with pytest.raises(ValueError, match="undeclared fields"):
        evaluate(corrupted)


def test_valid_explicit_healthy_copy_has_same_semantics_as_fresh_input() -> None:
    copied = observation(status="unavailable").model_copy(
        update={"status": "healthy"}
    )
    fresh = observation(status="healthy")

    assert copied.model_dump() == fresh.model_dump()
    assert evaluate(copied) == evaluate(fresh) is True


def test_readiness_evaluation_revalidates_post_construction_mutation() -> None:
    item = observation()
    item.observed_at = datetime(2026, 9, 28, 0, 0)

    with pytest.raises(ValidationError):
        evaluate(item)


@pytest.mark.parametrize(
    "invalid_generation",
    [True, 12.0, -1, 9_007_199_254_740_992],
)
def test_publication_helper_strictly_validates_generation(
    invalid_generation: object,
) -> None:
    with pytest.raises(ValidationError):
        validate_readiness_publication(
            publication(),
            expected_tenant_id="tenant-a",
            current_placement_generation=invalid_generation,
        )


@pytest.mark.parametrize("invalid_tenant_id", [1, b"tenant-a", ""])
def test_publication_helper_strictly_validates_tenant_id(
    invalid_tenant_id: object,
) -> None:
    with pytest.raises(ValidationError):
        validate_readiness_publication(
            publication(),
            expected_tenant_id=invalid_tenant_id,
            current_placement_generation=12,
        )


def test_generic_process_health_cannot_stand_in_for_capability_evidence() -> None:
    with pytest.raises(ValidationError):
        TenantReadinessPublication.model_validate_json(
            json.dumps(
                {
                    "schema_version": VERSION,
                    "tenant_id": "tenant-a",
                    "placement_generation": 12,
                    "capabilities": [],
                    "process_healthy": True,
                }
            )
        )


def test_version_and_unknown_status_are_rejected() -> None:
    with pytest.raises(ValidationError):
        CapabilityReadiness.model_validate_json(
            json.dumps(
                {
                    "capability": "exact_reads",
                    "status": "healthy",
                    "observed_at": "2026-09-28T00:00:00Z",
                }
            )
        )
    with pytest.raises(ValidationError):
        observation(status="process_running")


def _set_native_field(model: BaseModel, name: str, value: object) -> None:
    native = PYDANTIC_DICT_DESCRIPTOR.__get__(model, BaseModel)
    dict.__setitem__(native, name, value)


def _install_repairing_stored_name(
    model: BaseModel, repair
) -> RepairingName:
    native = PYDANTIC_DICT_DESCRIPTOR.__get__(model, BaseModel)
    stored_value = dict.__getitem__(native, "schema_version")
    dict.__delitem__(native, "schema_version")
    name = RepairingName("schema_version", repair)
    dict.__setitem__(native, name, stored_value)
    name.calls = 0
    name.armed = True
    return name


def _validate_publication(candidate: TenantReadinessPublication):
    return validate_readiness_publication(
        candidate,
        expected_tenant_id="tenant-a",
        current_placement_generation=12,
    )


def test_readiness_publication_freezes_nested_model_before_stored_name_hook() -> None:
    ordinary = publication()
    _set_native_field(ordinary.capabilities[0], "capability", "")
    with pytest.raises(ValidationError):
        _validate_publication(ordinary)

    candidate = publication()
    child = candidate.capabilities[0]
    _set_native_field(child, "capability", "")
    name = _install_repairing_stored_name(
        candidate,
        lambda: _set_native_field(child, "capability", "api_acceptance"),
    )

    with pytest.raises(ValidationError):
        _validate_publication(candidate)

    assert name.calls > 0
    assert child.capability == "api_acceptance"


def test_readiness_publication_freezes_tuple_before_traversal_hook() -> None:
    candidate = publication()
    child = candidate.capabilities[0]
    _set_native_field(child, "capability", "")
    repairing = RepairingTuple(
        candidate.capabilities,
        lambda: _set_native_field(child, "capability", "api_acceptance"),
    )
    _set_native_field(candidate, "capabilities", repairing)

    with pytest.raises(ValidationError):
        _validate_publication(candidate)

    assert repairing.calls == 1
    assert child.capability == "api_acceptance"


def test_readiness_publication_freezes_nested_model_before_extra_mapping_hook() -> None:
    candidate = publication()
    child = candidate.capabilities[0]
    _set_native_field(child, "capability", "")
    extras = RepairingEmptyExtraMapping(
        lambda: _set_native_field(child, "capability", "api_acceptance")
    )
    PYDANTIC_EXTRA_DESCRIPTOR.__set__(candidate, extras)

    with pytest.raises(ValidationError):
        _validate_publication(candidate)

    assert extras.calls == 3
    assert child.capability == "api_acceptance"


def test_readiness_publication_preserves_strict_list_rejection_with_hook() -> None:
    candidate = publication()
    child = candidate.capabilities[0]
    _set_native_field(child, "capability", "")
    repairing = RepairingList(
        candidate.capabilities,
        lambda: _set_native_field(child, "capability", "api_acceptance"),
    )
    _set_native_field(candidate, "capabilities", repairing)

    with pytest.raises(ValidationError) as exc_info:
        _validate_publication(candidate)

    assert any(error["type"] == "tuple_type" for error in exc_info.value.errors())
    assert repairing.calls == 1
    assert child.capability == "api_acceptance"


def test_readiness_publication_leaves_fields_set_hook_inert() -> None:
    candidate = publication()
    child = candidate.capabilities[0]
    _set_native_field(child, "capability", "")
    name = RepairingName(
        "capability",
        lambda: _set_native_field(child, "capability", "api_acceptance"),
    )
    fields_set = PYDANTIC_FIELDS_SET_DESCRIPTOR.__get__(candidate, type(candidate))
    set.add(fields_set, name)
    name.calls = 0
    name.armed = True

    with pytest.raises(ValidationError):
        _validate_publication(candidate)

    assert name.calls == 0
    assert child.capability == ""


def test_readiness_publication_retains_invalid_root_before_name_hook_repair() -> None:
    candidate = publication()
    _set_native_field(candidate, "tenant_id", "")
    name = _install_repairing_stored_name(
        candidate,
        lambda: _set_native_field(candidate, "tenant_id", "tenant-a"),
    )

    with pytest.raises(ValidationError):
        _validate_publication(candidate)

    assert name.calls > 0
    assert candidate.tenant_id == "tenant-a"


def test_capability_receiver_retains_invalid_scalar_before_name_hook_repair() -> None:
    candidate = observation()
    _set_native_field(candidate, "capability", "")
    name = _install_repairing_stored_name(
        candidate,
        lambda: _set_native_field(candidate, "capability", "exact_reads"),
    )

    with pytest.raises(ValidationError):
        evaluate(candidate)

    assert name.calls > 0
    assert candidate.capability == "exact_reads"


def test_readiness_publication_preserves_valid_tuple_subclass() -> None:
    candidate = publication()
    calls: list[str] = []
    repairing = RepairingTuple(
        candidate.capabilities,
        lambda: calls.append("iterated"),
    )
    _set_native_field(candidate, "capabilities", repairing)

    result = _validate_publication(candidate)

    assert result.capabilities[0].capability == "api_acceptance"
    assert repairing.calls == 1
    assert calls == ["iterated"]


def test_readiness_publication_preserves_divergent_tuple_public_view() -> None:
    candidate = publication()
    visible = (observation("exact_reads", "healthy"),)
    divergent = DivergentTuple(candidate.capabilities, visible)
    _set_native_field(candidate, "capabilities", divergent)

    result = _validate_publication(candidate)

    assert result.capabilities == visible
    assert divergent.calls == 1


def test_readiness_publication_freezes_later_model_edge_before_name_callback() -> None:
    candidate = publication()
    original_edge = candidate.capabilities
    child = original_edge[0]
    _set_native_field(child, "capability", "")
    replacement = (observation("exact_reads", "healthy"),)
    name = _install_repairing_stored_name(
        candidate,
        lambda: _set_native_field(candidate, "capabilities", replacement),
    )

    with pytest.raises(ValidationError):
        _validate_publication(candidate)

    assert name.calls > 0
    assert candidate.capabilities is replacement
    assert original_edge[0].capability == ""


def test_readiness_publication_preserves_benign_string_subclass_name() -> None:
    candidate = publication()
    name = _install_repairing_stored_name(candidate, lambda: None)

    result = _validate_publication(candidate)

    assert result.tenant_id == "tenant-a"
    assert name.calls > 0


def test_readiness_boundaries_preserve_absent_fields_set_policy() -> None:
    candidate = publication()
    child = candidate.capabilities[0]
    PYDANTIC_FIELDS_SET_DESCRIPTOR.__delete__(candidate)
    PYDANTIC_FIELDS_SET_DESCRIPTOR.__delete__(child)

    assert _validate_publication(candidate).tenant_id == "tenant-a"

    item = observation()
    PYDANTIC_FIELDS_SET_DESCRIPTOR.__delete__(item)
    assert evaluate(item)


def test_readiness_publication_accepts_truly_absent_extra_storage() -> None:
    candidate = publication()
    child = candidate.capabilities[0]
    PYDANTIC_EXTRA_DESCRIPTOR.__delete__(candidate)
    PYDANTIC_EXTRA_DESCRIPTOR.__delete__(child)

    assert _validate_publication(candidate).tenant_id == "tenant-a"


def _public_only_invalid_capability_case():
    candidate = publication()
    public_child = observation("exact_reads", "healthy")
    _set_native_field(public_child, "capability", "")
    divergent = DivergentTuple(candidate.capabilities, (public_child,))
    _set_native_field(candidate, "capabilities", divergent)
    return candidate, public_child, divergent


def test_public_first_rejects_invalid_public_only_capability_without_callbacks() -> (
    None
):
    candidate, child, divergent = _public_only_invalid_capability_case()

    with pytest.raises(ValidationError):
        _validate_publication(candidate)

    assert divergent.calls == 1
    assert child.capability == ""


def test_public_first_lazy_capability_is_captured_at_first_observation_not_entry() -> (
    None
):
    # Documented limit: a child exposed only by a public iterator is captured
    # when the iterator first exposes it, so a callback that runs earlier in the
    # same call may change it first.  Entry-time capture is not promised.
    candidate, child, divergent = _public_only_invalid_capability_case()
    name = _install_repairing_stored_name(
        candidate,
        lambda: _set_native_field(child, "capability", "exact_reads"),
    )

    result = _validate_publication(candidate)

    assert name.calls > 0
    assert divergent.calls == 1
    assert result.capabilities[0].capability == "exact_reads"


def test_public_first_lazy_capability_is_frozen_once_first_observed() -> None:
    candidate, child, divergent = _public_only_invalid_capability_case()
    name = _install_repairing_stored_name(
        child,
        lambda: _set_native_field(child, "capability", "exact_reads"),
    )

    with pytest.raises(ValidationError):
        _validate_publication(candidate)

    assert name.calls > 0
    assert divergent.calls == 1
    assert child.capability == "exact_reads"


# ---------------------------------------------------------------------------
# Bounded native discovery (shared STANDARD budget, same walk as the tenant
# contracts): iterative, callback-free, stable value-free rejection.
# ---------------------------------------------------------------------------

import contracts_tenancy  # noqa: E402
from contracts_limits import DiscoveryTier  # noqa: E402

BUDGET_MESSAGE = "contract graph exceeds the native discovery budget"


def _validate(candidate: TenantReadinessPublication) -> TenantReadinessPublication:
    return validate_readiness_publication(
        candidate,
        expected_tenant_id="tenant-a",
        current_placement_generation=12,
    )


def _with_capabilities(candidate: TenantReadinessPublication, value: object):
    return candidate.model_copy(update={"capabilities": value})


def _nested(kind: type, depth: int) -> object:
    nested: object = kind()
    for _ in range(depth - 1):
        nested = (nested,) if kind is tuple else [nested]
    return nested


def _fits(candidate: object, monkeypatch, **limits: int) -> bool:
    tier = {"nodes": 10**6, "edges": 10**9, "depth": 256}
    tier.update(limits)
    monkeypatch.setattr(
        contracts_tenancy,
        "STANDARD",
        DiscoveryTier("test", tier["nodes"], tier["edges"], tier["depth"]),
    )
    from contracts_readiness import _snapshot_closed_graph

    try:
        _snapshot_closed_graph(candidate)
    except ValueError as error:
        if str(error) == BUDGET_MESSAGE:
            return False
        raise
    return True


@pytest.mark.parametrize("kind", [tuple, list])
def test_depth_1000_native_graph_is_a_stable_budget_error_not_recursion(kind) -> None:
    deep = _with_capabilities(publication(), _nested(kind, 1000))

    with pytest.raises(ValueError) as raised:
        _validate(deep)
    assert type(raised.value) is ValueError
    assert str(raised.value) == BUDGET_MESSAGE


def test_discovery_limits_are_exact_for_nodes_edges_and_depth(monkeypatch) -> None:
    candidate = publication()
    # publication + capabilities tuple + four observations
    # edges: 4 publication fields + 4 tuple entries + 5 stored fields x 4
    # depth: publication (0) -> capabilities tuple (1) -> observation (2)
    for key, exact in (("nodes", 6), ("edges", 28), ("depth", 2)):
        assert _fits(candidate, monkeypatch, **{key: exact})
        assert not _fits(candidate, monkeypatch, **{key: exact - 1})


def test_production_depth_limit_is_exact_at_the_real_tier() -> None:
    with pytest.raises(ValidationError):  # within budget, then structurally invalid
        _validate(_with_capabilities(publication(), _nested(list, 256)))
    with pytest.raises(ValueError, match=BUDGET_MESSAGE) as over:
        _validate(_with_capabilities(publication(), _nested(list, 257)))
    assert type(over.value) is ValueError


class CountingHugeList(list):
    """Native storage far over budget; every public protocol is counted."""

    def __init__(self, size: int) -> None:
        super().__init__(range(size))
        self.public_calls: list[str] = []

    def __iter__(self):
        self.public_calls.append("iter")
        return list.__iter__(self)

    def __len__(self) -> int:
        self.public_calls.append("len")
        return list.__len__(self)

    def __getitem__(self, index):
        self.public_calls.append("getitem")
        return list.__getitem__(self, index)

    def __eq__(self, other: object) -> bool:
        self.public_calls.append("eq")
        return list.__eq__(self, other)

    __hash__ = None  # type: ignore[assignment]


def test_hidden_huge_native_graph_is_rejected_with_zero_public_reads() -> None:
    huge = CountingHugeList(200_000)

    with pytest.raises(ValueError, match=BUDGET_MESSAGE):
        _validate(_with_capabilities(publication(), huge))
    assert huge.public_calls == []


class InfinitePublicView(list):
    """Empty native storage; the public view never ends."""

    def __init__(self) -> None:
        super().__init__()
        self.reads = 0

    def __iter__(self):
        self.reads += 1
        count = 0
        while True:
            count += 1
            yield count


def test_public_view_entries_are_charged_when_first_observed() -> None:
    endless = InfinitePublicView()

    with pytest.raises(ValueError, match=BUDGET_MESSAGE):
        _validate(_with_capabilities(publication(), endless))
    assert endless.reads == 1


def test_budget_error_is_value_free() -> None:
    secret = "tenant-secret-value"
    candidate = publication().model_copy(
        update={"tenant_id": secret, "capabilities": list(range(200_000))}
    )
    with pytest.raises(ValueError) as raised:
        _validate(candidate)
    assert secret not in str(raised.value)
    assert str(raised.value) == BUDGET_MESSAGE


# ---------------------------------------------------------------------------
# Ordering and age use absolute time, not wall time in a shared zone.
# ---------------------------------------------------------------------------

from zoneinfo import ZoneInfo  # noqa: E402

_NEW_YORK = ZoneInfo("America/New_York")


def _fold_observation(observed_at: datetime) -> CapabilityReadiness:
    # Built from the datetime itself so the observation keeps its zone object.
    return CapabilityReadiness(
        schema_version="candidate-v1",
        capability="exact_reads",
        status="healthy",
        observed_at=observed_at,
    )


def test_fold_does_not_make_a_stale_probe_ready() -> None:
    # 2026-11-01 01:30 happens twice in New York: 05:30Z (fold=0) and 06:30Z (fold=1).
    observed = datetime(2026, 11, 1, 1, 30, tzinfo=_NEW_YORK, fold=0)
    evaluated = datetime(2026, 11, 1, 1, 35, tzinfo=_NEW_YORK, fold=1)  # 06:35Z: 65 minutes later
    assert (evaluated - observed).total_seconds() == 300  # wall-time difference is wrong
    observation = _fold_observation(observed)

    assert capability_is_ready(observation, evaluated_at=evaluated, max_probe_age_seconds=600) is False
    assert capability_is_ready(observation, evaluated_at=evaluated, max_probe_age_seconds=3900) is True


def test_fold_does_not_reject_a_later_evaluation_as_preceding() -> None:
    observed = datetime(2026, 11, 1, 1, 30, tzinfo=_NEW_YORK, fold=0)  # 05:30Z
    evaluated = datetime(2026, 11, 1, 1, 10, tzinfo=_NEW_YORK, fold=1)  # 06:10Z: 40 minutes later
    observation = _fold_observation(observed)

    assert capability_is_ready(observation, evaluated_at=evaluated, max_probe_age_seconds=3600) is True


def test_a_truly_earlier_evaluation_in_the_fold_is_still_rejected() -> None:
    observed = datetime(2026, 11, 1, 1, 30, tzinfo=_NEW_YORK, fold=1)  # 06:30Z
    evaluated = datetime(2026, 11, 1, 1, 45, tzinfo=_NEW_YORK, fold=0)  # 05:45Z: before the probe
    with pytest.raises(ValueError, match="cannot precede"):
        capability_is_ready(_fold_observation(observed), evaluated_at=evaluated, max_probe_age_seconds=3600)


def test_mixed_zones_and_the_spring_gap_use_absolute_time() -> None:
    observed = datetime(2026, 3, 8, 1, 30, tzinfo=_NEW_YORK)  # 06:30Z, before the gap
    evaluated = datetime(2026, 3, 8, 3, 30, tzinfo=_NEW_YORK)  # 07:30Z, one hour later
    observation = _fold_observation(observed)
    assert capability_is_ready(observation, evaluated_at=evaluated, max_probe_age_seconds=3600) is True
    assert capability_is_ready(observation, evaluated_at=evaluated, max_probe_age_seconds=3599) is False
    utc_evaluated = datetime(2026, 3, 8, 7, 30, tzinfo=timezone.utc)
    assert capability_is_ready(observation, evaluated_at=utc_evaluated, max_probe_age_seconds=3600) is True


# ---- a zone that answers differently on each read (a caller-owned tzinfo is arbitrary code)

from datetime import tzinfo  # noqa: E402


class FlappingZone(tzinfo):
    """Reports ``script[0]`` on the first read after ``arm()``, then ``script[1]`` on every later read."""

    def __init__(self, first: timedelta, later: timedelta) -> None:
        self.first, self.later = first, later
        self.reads: int | None = None

    def arm(self) -> None:
        self.reads = 0

    def utcoffset(self, dt):
        if self.reads is None:
            return self.first
        self.reads += 1
        return self.first if self.reads == 1 else self.later

    def dst(self, dt):
        return timedelta(0)

    def tzname(self, dt):
        return "flap"


def _flapping_observation(zone: FlappingZone, *, observed: datetime) -> CapabilityReadiness:
    item = _fold_observation(observed.replace(tzinfo=zone))  # built while the zone is steady
    zone.arm()
    return item


def test_an_observed_at_zone_that_changes_its_offset_between_reads_cannot_freshen_a_stale_probe() -> None:
    # First read: UTC, so the probe is 2026-01-01 12:00Z.  Every later read: UTC-12, which would move it to
    # 2026-01-02 00:00Z and make a probe 12 hours old look brand new at that evaluation time.
    zone = FlappingZone(timedelta(0), timedelta(hours=-12))
    observation = _flapping_observation(zone, observed=datetime(2026, 1, 1, 12, 0))
    evaluated = datetime(2026, 1, 2, 0, 0, tzinfo=timezone.utc)
    assert capability_is_ready(observation, evaluated_at=evaluated, max_probe_age_seconds=3600) is False
    assert zone.reads == 1  # the caller's zone was asked once


def test_an_evaluated_at_zone_that_changes_its_offset_between_reads_cannot_age_a_fresh_probe() -> None:
    zone = FlappingZone(timedelta(0), timedelta(hours=-12))
    observation = _fold_observation(datetime(2026, 1, 1, 11, 30, tzinfo=timezone.utc))
    evaluated = datetime(2026, 1, 1, 12, 0, tzinfo=zone)
    zone.arm()
    assert capability_is_ready(observation, evaluated_at=evaluated, max_probe_age_seconds=3600) is True
    assert zone.reads == 1


def test_a_publication_pins_each_observation_time_once() -> None:
    zone = FlappingZone(timedelta(0), timedelta(hours=-12))
    item = _fold_observation(datetime(2026, 1, 1, 12, 0, tzinfo=zone))
    publication = TenantReadinessPublication(
        schema_version="candidate-v1", tenant_id="tenant-a", placement_generation=1, capabilities=(item,)
    )
    zone.arm()
    checked = validate_readiness_publication(
        publication, expected_tenant_id="tenant-a", current_placement_generation=1
    )
    assert zone.reads == 1
    kept = checked.capabilities[0].observed_at
    assert kept == datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)  # the first read decided the instant
    # and the value handed back no longer depends on the caller's zone
    assert capability_is_ready(
        checked.capabilities[0], evaluated_at=datetime(2026, 1, 1, 12, 30, tzinfo=timezone.utc), max_probe_age_seconds=3600
    ) is True
    assert zone.reads == 1


def test_pinning_keeps_ordinary_zones_equal_and_naive_values_refused() -> None:
    ordinary = _fold_observation(datetime(2026, 7, 1, 9, 0, tzinfo=_NEW_YORK))
    pub = TenantReadinessPublication(schema_version="candidate-v1", tenant_id="t", placement_generation=1, capabilities=(ordinary,))
    checked = validate_readiness_publication(pub, expected_tenant_id="t", current_placement_generation=1)
    assert checked == pub and checked.capabilities[0].observed_at.utcoffset() == timedelta(hours=-4)
    # an ambiguous (fold) time keeps its instant and offset; only the zone object is replaced by the fixed offset
    observation = _fold_observation(datetime(2026, 11, 1, 1, 30, tzinfo=_NEW_YORK, fold=1))
    folded = TenantReadinessPublication(schema_version="candidate-v1", tenant_id="t", placement_generation=1, capabilities=(observation,))
    kept = validate_readiness_publication(folded, expected_tenant_id="t", current_placement_generation=1).capabilities[0].observed_at
    assert kept.utcoffset() == timedelta(hours=-5) and kept.astimezone(timezone.utc) == datetime(2026, 11, 1, 6, 30, tzinfo=timezone.utc)
    with pytest.raises(ValueError):
        capability_is_ready(observation, evaluated_at=datetime(2026, 11, 2), max_probe_age_seconds=1)  # naive


# ---- shared subtrees are pinned once; a zone with no offset on its first read is refused

class CountingZone(tzinfo):
    """UTC, counting every offset read."""

    def __init__(self) -> None:
        self.reads = 0

    def utcoffset(self, dt):
        self.reads += 1
        return timedelta(0)

    def dst(self, dt):
        return timedelta(0)

    def tzname(self, dt):
        return "count"


def _shared_node(depth: int, leaf: object) -> object:
    node = leaf
    for _ in range(depth):
        node = [node, node]  # 2**depth paths, depth + 1 distinct lists
    return node


@pytest.mark.parametrize("depth", [12, 30])
def test_a_shared_nested_graph_is_pinned_once_per_object(depth: int) -> None:
    zone = CountingZone()
    item = _fold_observation(datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc))
    dict.__setitem__(vars(item), "detail_code", _shared_node(depth, datetime(2026, 1, 1, 12, 0, tzinfo=zone)))
    started = time.perf_counter()
    with pytest.raises(ValidationError):  # a nested list is not an id; the pinning must not blow up first
        capability_is_ready(item, evaluated_at=datetime(2026, 1, 1, 12, 30, tzinfo=timezone.utc), max_probe_age_seconds=3600)
    assert zone.reads == 1  # not 2**depth
    assert time.perf_counter() - started < 2.0


def test_pinning_keeps_the_sharing_of_the_snapshot() -> None:
    import contracts_readiness as readiness

    shared = [datetime(2026, 1, 1, tzinfo=timezone.utc)]
    pinned = readiness._pin_datetimes({"a": shared, "b": (shared, shared), "c": [shared, "x"]})
    assert pinned["a"] is pinned["b"][0] is pinned["b"][1] is pinned["c"][0]
    untouched = {"k": ["plain", ("tuple", 1)]}
    assert readiness._pin_datetimes(untouched) is untouched  # nothing to pin: nothing copied


class NoOffsetFirstZone(tzinfo):
    """No offset on the first read after ``arm()``, UTC on every later read."""

    def __init__(self) -> None:
        self.reads: int | None = None

    def arm(self) -> None:
        self.reads = 0

    def utcoffset(self, dt):
        if self.reads is None:
            return timedelta(0)
        self.reads += 1
        return None if self.reads == 1 else timedelta(0)

    def dst(self, dt):
        return timedelta(0)

    def tzname(self, dt):
        return "none-first"


def test_an_observed_at_with_no_offset_on_its_first_read_is_refused_and_read_once() -> None:
    zone = NoOffsetFirstZone()
    item = _fold_observation(datetime(2026, 1, 1, 12, 0, tzinfo=zone))
    zone.arm()
    with pytest.raises(ValidationError):
        capability_is_ready(item, evaluated_at=datetime(2026, 1, 1, 12, 30, tzinfo=timezone.utc), max_probe_age_seconds=3600)
    assert zone.reads == 1


def test_an_evaluated_at_with_no_offset_on_its_first_read_is_refused_and_read_once() -> None:
    zone = NoOffsetFirstZone()
    item = _fold_observation(datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc))
    evaluated = datetime(2026, 1, 1, 12, 30, tzinfo=zone)
    zone.arm()
    with pytest.raises(ValidationError):
        capability_is_ready(item, evaluated_at=evaluated, max_probe_age_seconds=3600)
    assert zone.reads == 1


def test_a_publication_with_a_first_read_of_none_is_refused() -> None:
    zone = NoOffsetFirstZone()
    item = _fold_observation(datetime(2026, 1, 1, 12, 0, tzinfo=zone))
    publication = TenantReadinessPublication(schema_version="candidate-v1", tenant_id="t", placement_generation=1, capabilities=(item,))
    zone.arm()
    with pytest.raises(ValidationError):
        validate_readiness_publication(publication, expected_tenant_id="t", current_placement_generation=1)
    assert zone.reads == 1


# ---- type gates classify by the object's real type, never by its own ``__class__``

class _ClaimsObservation:
    """Not a ``CapabilityReadiness``: its ``__class__`` says it is, and its ``__dict__`` holds a flapping zone."""

    def __init__(self, observed_at: datetime) -> None:
        self.__dict__.update(
            schema_version="candidate-v1", capability="exact_reads", status="healthy",
            observed_at=observed_at, detail_code=None,
        )
        self.__pydantic_fields_set__ = set(self.__dict__)

    @property
    def __class__(self):  # type: ignore[override]
        return CapabilityReadiness


def test_an_object_claiming_the_observation_class_cannot_skip_the_timestamp_pinning() -> None:
    zone = FlappingZone(timedelta(0), timedelta(hours=-12))
    spoof = _ClaimsObservation(datetime(2026, 1, 1, 12, 0, tzinfo=zone))
    zone.arm()
    with pytest.raises(TypeError, match="observation must be a CapabilityReadiness"):
        capability_is_ready(spoof, evaluated_at=datetime(2026, 1, 2, 0, 0, tzinfo=timezone.utc), max_probe_age_seconds=60)  # type: ignore[arg-type]
    assert zone.reads in (0, None)  # the caller's zone was never consulted


def test_the_genuine_observation_with_the_same_zone_is_not_ready() -> None:
    zone = FlappingZone(timedelta(0), timedelta(hours=-12))
    observation = _fold_observation(datetime(2026, 1, 1, 12, 0, tzinfo=zone))
    zone.arm()
    assert capability_is_ready(observation, evaluated_at=datetime(2026, 1, 2, 0, 0, tzinfo=timezone.utc), max_probe_age_seconds=60) is False
    assert zone.reads == 1


def test_an_object_claiming_the_publication_class_is_refused_by_the_gate() -> None:
    class Spoof:
        @property
        def __class__(self):  # type: ignore[override]
            return TenantReadinessPublication
    with pytest.raises(TypeError, match="publication must be a TenantReadinessPublication"):
        validate_readiness_publication(Spoof(), expected_tenant_id="t", current_placement_generation=1)  # type: ignore[arg-type]


# ---- a hostile zone is a value error, never the caller's exception or a TypeError

class _HostileZone(tzinfo):
    def __init__(self, mode: str) -> None:
        self.mode, self.armed, self.reads = mode, False, 0

    def utcoffset(self, dt):
        if not self.armed:
            return timedelta(0)
        self.reads += 1
        if self.mode == "raises":
            raise RuntimeError("zone trap")
        return "not an offset"

    def dst(self, dt):
        return timedelta(0)

    def tzname(self, dt):
        return "hostile"


@pytest.mark.parametrize("mode", ["raises", "non-offset"])
def test_a_hostile_zone_on_observed_at_is_a_value_error(mode: str) -> None:
    zone = _HostileZone(mode)
    item = _fold_observation(datetime(2026, 1, 1, 12, 0, tzinfo=zone))
    zone.armed = True
    with pytest.raises(ValidationError):
        capability_is_ready(item, evaluated_at=datetime(2026, 1, 1, 12, 30, tzinfo=timezone.utc), max_probe_age_seconds=3600)
    assert zone.reads == 1


@pytest.mark.parametrize("mode", ["raises", "non-offset"])
def test_a_hostile_zone_on_evaluated_at_is_a_value_error(mode: str) -> None:
    zone = _HostileZone(mode)
    item = _fold_observation(datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc))
    evaluated = datetime(2026, 1, 1, 12, 30, tzinfo=zone)
    zone.armed = True
    with pytest.raises(ValidationError):
        capability_is_ready(item, evaluated_at=evaluated, max_probe_age_seconds=3600)
    assert zone.reads == 1


@pytest.mark.parametrize("mode", ["raises", "non-offset"])
def test_a_hostile_zone_inside_a_publication_is_a_value_error(mode: str) -> None:
    zone = _HostileZone(mode)
    item = _fold_observation(datetime(2026, 1, 1, 12, 0, tzinfo=zone))
    publication = TenantReadinessPublication(schema_version="candidate-v1", tenant_id="t", placement_generation=1, capabilities=(item,))
    zone.armed = True
    with pytest.raises(ValidationError):
        validate_readiness_publication(publication, expected_tenant_id="t", current_placement_generation=1)
    assert zone.reads == 1


# ---- one timestamp object has one interpretation throughout a call


def test_an_evaluation_time_that_is_the_observation_time_object_is_read_once() -> None:
    zone = FlappingZone(timedelta(0), timedelta(hours=-12))
    moment = datetime(2026, 1, 1, 12, 0, tzinfo=zone)
    observation = _fold_observation(moment)
    assert observation.observed_at is moment  # the same caller object at both positions
    zone.arm()
    # Read once, the probe is exactly as old as the evaluation (age 0): ready.  A second read of the same
    # object (UTC-12) would put the evaluation 12 hours after the probe and call it stale.
    assert capability_is_ready(observation, evaluated_at=moment, max_probe_age_seconds=3600) is True
    assert zone.reads == 1


def test_distinct_timestamp_objects_are_still_each_read_once() -> None:
    zone = FlappingZone(timedelta(0), timedelta(hours=-12))
    observation = _fold_observation(datetime(2026, 1, 1, 12, 0, tzinfo=zone))
    evaluated = datetime(2026, 1, 1, 12, 30, tzinfo=zone)  # another object on the same zone
    zone.arm()
    # each object is read once; the second object's first read is already a later read of the zone (UTC-12)
    assert capability_is_ready(observation, evaluated_at=evaluated, max_probe_age_seconds=3600) is False
    assert zone.reads == 2
