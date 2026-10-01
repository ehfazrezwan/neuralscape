"""Contract tests for capability-specific readiness evidence."""

from __future__ import annotations

import json
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

    assert extras.class_calls == 1
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
