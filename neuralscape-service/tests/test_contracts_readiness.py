"""Contract tests for capability-specific readiness evidence."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from types import MappingProxyType

import pytest
from pydantic import ValidationError

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
