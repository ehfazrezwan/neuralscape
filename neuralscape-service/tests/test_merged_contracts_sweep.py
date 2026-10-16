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
