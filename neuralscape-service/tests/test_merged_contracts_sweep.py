"""Hostile-input regressions for the merged tenancy, readiness and usage contracts.

Each test reproduces a way a caller-supplied object could change a validation
outcome (or an error class) and pins the rule that now prevents it.  Every
test named for a defect fails on the tree before the matching change.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping

import pytest
from pydantic import BaseModel, ValidationError

import contracts_usage_reconcile as usage_reconcile
from contracts_readiness import CapabilityReadiness, capability_is_ready
from contracts_tenancy import ResourceManifestReference, TenantPlacement, validate_placement_publication
from tests.test_contracts_usage_reconcile import _event

VERSION = "candidate-v1"
_STORED = BaseModel.__dict__["__dict__"]
_EXTRA = BaseModel.__dict__["__pydantic_extra__"]


def native(model: BaseModel) -> dict:
    return _STORED.__get__(model, BaseModel)


def reference(index: int = 1, generation: int = 1) -> ResourceManifestReference:
    return ResourceManifestReference(
        schema_version=VERSION, tenant_id="t", manifest_id=f"m{index}", manifest_revision=1, placement_generation=generation
    )


def placement() -> TenantPlacement:
    return TenantPlacement(schema_version=VERSION, tenant_id="t", generation=1, resource_manifests=(reference(1),))


def publish(value: TenantPlacement) -> TenantPlacement:
    return validate_placement_publication(value, expected_tenant_id="t", current_generation=1)


class RegisteredOnlyMapping:
    """Not a mapping subclass: it only registers with the ABC."""

    def items(self):
        return ()

    def __iter__(self):
        return iter(())

    def __len__(self) -> int:
        return 0


Mapping.register(RegisteredOnlyMapping)


class RealEmptyMapping(Mapping):
    def __getitem__(self, key):
        raise KeyError(key)

    def __iter__(self):
        return iter(())

    def __len__(self) -> int:
        return 0


# ---- a class that only registers with an ABC is not a mapping


def test_tenancy_a_registered_only_mapping_is_not_valid_extra_storage() -> None:
    value = placement()
    _EXTRA.__set__(value, RegisteredOnlyMapping())
    with pytest.raises(ValueError, match="must be a mapping"):
        publish(value)


def test_tenancy_a_real_empty_mapping_is_still_accepted() -> None:
    value = placement()
    _EXTRA.__set__(value, RealEmptyMapping())
    assert publish(value).generation == 1


def test_usage_a_registered_only_mapping_is_not_valid_extra_storage() -> None:
    event = _event()
    _EXTRA.__set__(event, RegisteredOnlyMapping())
    with pytest.raises(usage_reconcile.UsageReconciliationError):
        usage_reconcile.reconcile_usage_events([event])


def test_usage_a_real_empty_mapping_is_still_accepted() -> None:
    event = _event()
    _EXTRA.__set__(event, RealEmptyMapping())
    assert usage_reconcile.reconcile_usage_events([event]).streams[0].status == "final"


# ---- exposed siblings are captured before an earlier one is rebuilt


class ExposedView(tuple):
    def __new__(cls, items):
        instance = tuple.__new__(cls, ())
        instance.exposed = items
        return instance

    def __iter__(self):
        return iter(self.exposed)


class RepairingExtras(Mapping):
    def __init__(self, callback) -> None:
        self.callback = callback

    def __getitem__(self, key):
        raise KeyError(key)

    def __iter__(self):
        return iter(())

    def __len__(self) -> int:
        return 0

    def items(self):
        self.callback()
        return ()


def _sibling_scenario(*, repair: bool) -> TenantPlacement:
    sibling = ResourceManifestReference.model_construct(
        schema_version=VERSION, tenant_id="t", manifest_id="m9", manifest_revision=1, placement_generation=99
    )
    first = reference(2)
    if repair:
        _EXTRA.__set__(first, RepairingExtras(lambda: dict.__setitem__(native(sibling), "placement_generation", 1)))
    value = placement()
    dict.__setitem__(native(value), "resource_manifests", ExposedView((first, sibling)))
    return value


def test_an_invalid_exposed_sibling_cannot_be_repaired_by_an_earlier_siblings_callback() -> None:
    with pytest.raises(ValidationError):
        publish(_sibling_scenario(repair=True))


def test_the_same_siblings_without_a_callback_are_rejected_too() -> None:
    with pytest.raises(ValidationError):
        publish(_sibling_scenario(repair=False))


def test_valid_exposed_siblings_are_accepted() -> None:
    value = placement()
    dict.__setitem__(native(value), "resource_manifests", ExposedView((reference(2), reference(3))))
    assert [item.manifest_id for item in publish(value).resource_manifests] == ["m2", "m3"]


# ---- the clock of an observation is read on the absolute timeline, not through its subclass


class ForgedClock(dt.datetime):
    def astimezone(self, tz=None):
        return dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=1)


def _observation(observed_at) -> CapabilityReadiness:
    value = CapabilityReadiness(
        schema_version=VERSION, capability="exact_reads", status="healthy",
        observed_at=dt.datetime(2000, 1, 1, tzinfo=dt.timezone.utc),
    )
    dict.__setitem__(native(value), "observed_at", observed_at)
    return value


def test_a_datetime_subclass_cannot_make_a_stale_observation_fresh() -> None:
    forged = ForgedClock(2000, 1, 1, tzinfo=dt.timezone.utc)
    now = dt.datetime.now(dt.timezone.utc)
    assert capability_is_ready(_observation(forged), evaluated_at=now, max_probe_age_seconds=60) is False


def test_a_genuine_fresh_observation_is_ready_and_a_genuine_stale_one_is_not() -> None:
    now = dt.datetime.now(dt.timezone.utc)
    fresh = _observation(now - dt.timedelta(seconds=5))
    stale = _observation(now - dt.timedelta(hours=1))
    assert capability_is_ready(fresh, evaluated_at=now, max_probe_age_seconds=60) is True
    assert capability_is_ready(stale, evaluated_at=now, max_probe_age_seconds=60) is False


def test_a_tzinfo_that_changes_its_offset_is_read_once() -> None:
    import contracts_readiness

    class Flapping(dt.tzinfo):
        calls = 0

        def utcoffset(self, value):
            Flapping.calls += 1
            return dt.timedelta(0) if Flapping.calls % 2 else dt.timedelta(hours=-12)

        def dst(self, value):
            return dt.timedelta(0)

    value = dt.datetime(2026, 1, 1, 12, tzinfo=Flapping())
    Flapping.calls = 0
    instant = contracts_readiness._exact_utc(value)
    assert Flapping.calls == 1  # the offset is asked for once, so the instant cannot be two different times
    assert instant == dt.datetime(2026, 1, 1, 12, tzinfo=dt.timezone.utc)
    assert type(instant) is dt.datetime


# ---- stored field names are exact text


class SameTextKey(str):
    def __hash__(self) -> int:
        return 1234

    def __eq__(self, other: object) -> bool:
        return self is other

    def __ne__(self, other: object) -> bool:
        return self is not other


def test_tenancy_a_text_subclass_field_name_is_refused() -> None:
    value = placement()
    dict.__setitem__(native(value), SameTextKey("generation"), 7)
    with pytest.raises(ValueError, match="unique text"):
        publish(value)


def test_usage_a_text_subclass_field_name_is_refused() -> None:
    event = _event()
    dict.__setitem__(native(event), SameTextKey("status"), "partial")
    with pytest.raises(usage_reconcile.UsageReconciliationError):
        usage_reconcile.reconcile_usage_events([event])


# ---- a pair a view yields is unpacked exactly once, and non-text names are a ValueError


class ScriptedPair(tuple):
    """A (key, value) pair whose value changes between unpackings, or that can be unpacked only once."""

    def __new__(cls, key, values):
        instance = tuple.__new__(cls, (key,))
        instance.key, instance.values, instance.unpacked = key, list(values), 0
        return instance

    def __iter__(self):
        self.unpacked += 1
        value = self.values.pop(0) if len(self.values) > 1 or self.once else self.values[0]
        return iter((self.key, value))

    once = False


class OnceOnlyPair(ScriptedPair):
    once = True

    def __iter__(self):
        if self.unpacked:
            raise RuntimeError("a one-shot pair was unpacked a second time")
        return super().__iter__()


class PairView(dict):
    """A dict subclass whose public items are scripted pair objects."""

    def __init__(self, pairs):
        super().__init__()
        self.pairs = pairs

    def items(self):
        return list(self.pairs)


def _reference_view(pair_type, generation_values):
    pairs = [
        pair_type("schema_version", [VERSION]), pair_type("tenant_id", ["t"]), pair_type("manifest_id", ["m1"]),
        pair_type("manifest_revision", [1]), pair_type("placement_generation", generation_values),
    ]
    return PairView(pairs), pairs


def _placement_with(view) -> TenantPlacement:
    value = placement()
    dict.__setitem__(native(value), "resource_manifests", (view,))
    return value


def test_a_pair_that_changes_between_unpackings_cannot_turn_an_invalid_value_valid() -> None:
    view, pairs = _reference_view(ScriptedPair, [99, 1])  # first read 99 (invalid), a second read would say 1
    with pytest.raises(ValidationError):
        publish(_placement_with(view))
    assert all(pair.unpacked == 1 for pair in pairs)


def test_a_one_shot_pair_is_read_once_and_the_value_is_accepted() -> None:
    view, pairs = _reference_view(OnceOnlyPair, [1])
    assert publish(_placement_with(view)).resource_manifests[0].manifest_id == "m1"
    assert all(pair.unpacked == 1 for pair in pairs)


def test_a_non_text_stored_name_is_a_value_error_not_a_type_error_in_tenancy_and_readiness() -> None:
    import datetime as dt

    from contracts_readiness import (
        CapabilityReadiness,
        TenantReadinessPublication,
        validate_readiness_publication,
    )

    value = placement()
    dict.__setitem__(native(value), 7, "x")
    with pytest.raises(ValueError):
        publish(value)
    observation = CapabilityReadiness(
        schema_version=VERSION, capability="exact_reads", status="healthy",
        observed_at=dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc),
    )
    publication = TenantReadinessPublication(
        schema_version=VERSION, tenant_id="t", placement_generation=1, capabilities=(observation,)
    )
    dict.__setitem__(native(publication), 7, "x")
    with pytest.raises(ValueError):
        validate_readiness_publication(publication, expected_tenant_id="t", current_placement_generation=1)


def test_a_non_text_stored_name_in_a_usage_result_is_a_validation_error_for_direct_receivers() -> None:
    result = usage_reconcile.reconcile_usage_events([_event()])
    stream = result.streams[0]
    dict.__setitem__(native(stream), 7, "x")
    with pytest.raises(ValidationError):
        usage_reconcile.ReconciledUsageStream.model_validate(stream)
    other = usage_reconcile.reconcile_usage_events([_event()])
    dict.__setitem__(native(other), 7, "x")
    with pytest.raises(ValidationError):
        usage_reconcile.UsageReconciliation.model_validate(other)


# ---- a caller-owned (key, value) pair is unpacked exactly once at every boundary

class FlippingPair(tuple):
    """A pair that reads as ``first`` on its first unpacking and as ``later`` on every later one."""

    def __new__(cls, first, later):
        instance = tuple.__new__(cls, ())
        instance.first, instance.later, instance.unpacked = first, later, 0
        return instance

    def __iter__(self):
        self.unpacked += 1
        return iter(self.first if self.unpacked == 1 else self.later)


class PairExtras(Mapping):
    """Real mapping whose iteration hides its entries and whose items() yields the scripted pairs."""

    def __init__(self, pairs):
        self.pairs = pairs

    def __iter__(self):
        return iter(())

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, key):
        raise KeyError(key)

    def items(self):
        return list(self.pairs)


def _usage_event_with_extras(first, later) -> tuple:
    event = _event()
    dict.__delitem__(native(event), "event_id")  # the stored fields lack a required field
    pair = FlippingPair(first, later)
    _EXTRA.__set__(event, PairExtras([pair]))
    return event, pair


def test_usage_extra_pair_that_supplies_a_removed_field_on_its_second_unpacking_is_refused() -> None:
    # first reading: an undeclared name (invalid); a second reading would supply the missing event_id
    event, pair = _usage_event_with_extras(("junk_undeclared", 1), ("event_id", "event-1"))
    with pytest.raises(ValueError):
        usage_reconcile.reconcile_usage_events([event])
    assert pair.unpacked == 1


def test_usage_extra_pair_with_only_the_undeclared_name_is_refused_as_before() -> None:
    event, pair = _usage_event_with_extras(("junk_undeclared", 1), ("junk_undeclared", 1))
    with pytest.raises(ValueError):
        usage_reconcile.reconcile_usage_events([event])
    assert pair.unpacked == 1


def test_usage_extra_pair_is_unpacked_once_even_when_the_event_is_otherwise_valid() -> None:
    event = _event()
    pair = FlippingPair(("junk_undeclared", 1), ("junk_undeclared", 2))
    _EXTRA.__set__(event, PairExtras([pair]))
    with pytest.raises(ValueError):
        usage_reconcile.reconcile_usage_events([event])
    assert pair.unpacked == 1


def test_readiness_pair_that_repairs_an_invalid_status_on_the_second_unpacking_is_refused() -> None:
    from contracts_readiness import (
        CapabilityReadiness,
        TenantReadinessPublication,
        validate_readiness_publication,
    )

    observed = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)
    observation = CapabilityReadiness(schema_version=VERSION, capability="exact_reads", status="healthy", observed_at=observed)
    pairs = [
        FlippingPair(("schema_version", VERSION), ("schema_version", VERSION)),
        FlippingPair(("capability", "exact_reads"), ("capability", "exact_reads")),
        FlippingPair(("status", "not-a-status"), ("status", "healthy")),
        FlippingPair(("observed_at", observed), ("observed_at", observed)),
    ]
    publication = TenantReadinessPublication(
        schema_version=VERSION, tenant_id="t", placement_generation=1, capabilities=(observation,)
    )
    dict.__setitem__(native(publication), "capabilities", (PairView(pairs),))
    with pytest.raises(ValueError):
        validate_readiness_publication(publication, expected_tenant_id="t", current_placement_generation=1)
    assert all(pair.unpacked == 1 for pair in pairs)
