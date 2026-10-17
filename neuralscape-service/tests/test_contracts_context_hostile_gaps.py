"""Caller-owned zones and keys at the context and freshness boundaries: one read, no caller code, bounded memory."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone, tzinfo

import pytest
from pydantic import BaseModel, ValidationError

from contracts_context import ContextBundle, ContextRequest, bundle_matches_reported_evidence
from contracts_freshness import (
    FreshnessWitness,
    SourceCheckpoint,
    pin_datetime,
    pin_native_snapshot,
)
from tests.test_contracts_context import _bundle, _item, _request, _validate
from tests.test_contracts_freshness import _checkpoint

_STORED = BaseModel.__dict__["__dict__"].__get__


class FlappingZone(tzinfo):
    """Steady until ``arm()``; then ``first`` on the first read and ``later`` (or an error) on every later one."""

    def __init__(self, first: timedelta, later: timedelta | Exception) -> None:
        self.first, self.later, self.reads = first, later, None

    def arm(self) -> None:
        self.reads = 0

    def utcoffset(self, dt):
        if self.reads is None:
            return self.first
        self.reads += 1
        if self.reads == 1:
            return self.first
        if isinstance(self.later, Exception):
            raise self.later
        return self.later

    def dst(self, dt):
        return timedelta(0)

    def tzname(self, dt):
        return "flap"


class OffsetWithOwnArithmetic(timedelta):
    def __sub__(self, other):
        return timedelta(days=3)

    def __neg__(self):
        return timedelta(days=3)


class DatetimeWithOwnOffset(datetime):
    def utcoffset(self):  # a method of the caller's own object: claims a different zone than the real one
        return timedelta(hours=23)


UTC = timezone.utc
_NOON = (2026, 9, 28, 12)
_EVENING = (2026, 9, 28, 18)


def _current_item() -> dict[str, object]:
    item = _item()
    freshness = item["freshness"]
    freshness["upstream_status"] = "current"
    checkpoint = freshness["source_checkpoints"][0]
    checkpoint.update(verification_status="current", last_successful_verification_at="2026-09-28T11:00:00Z")
    return item


def _matching_pair():
    request = _validate(ContextRequest, _request(freshness_requirement="current_known"))
    bundle = _validate(ContextBundle, _bundle(selected_items=[_current_item()]))
    return request, bundle


def _stored_checkpoint(bundle: ContextBundle):
    return bundle.selected_items[0].freshness.source_checkpoints[0]


# ---- F-1: a zone read twice

def test_the_matching_control_accepts_a_verification_before_the_observation() -> None:
    request, bundle = _matching_pair()
    assert bundle_matches_reported_evidence(request, bundle)


@pytest.mark.parametrize("later", [timedelta(hours=12), timedelta(hours=-12)], ids=["plus-12", "minus-12"])
def test_a_zone_read_twice_cannot_place_the_verification_before_the_observation(later) -> None:
    request, bundle = _matching_pair()
    zone = FlappingZone(timedelta(0), later)
    stored = _STORED(_stored_checkpoint(bundle), BaseModel)
    dict.__setitem__(stored, "observed_at", datetime(*_NOON, tzinfo=zone))
    dict.__setitem__(stored, "last_successful_verification_at", datetime(*_EVENING, tzinfo=UTC))  # six hours after
    zone.arm()
    assert bundle_matches_reported_evidence(request, bundle) is False
    assert zone.reads == 1  # the zone was asked once


def test_a_receipt_style_snapshot_pins_every_datetime_once() -> None:
    zone = FlappingZone(timedelta(0), timedelta(hours=9))
    leaf = datetime(*_NOON, tzinfo=zone)
    zone.arm()
    pinned = pin_native_snapshot({"a": [leaf, {"b": leaf}], "c": (leaf,)})
    assert zone.reads == 1  # one datetime object is read once however many places hold it
    assert pinned["a"][0] == pinned["a"][1]["b"] == pinned["c"][0] == datetime(*_NOON, tzinfo=UTC)
    assert type(pinned["a"][0]) is datetime and pinned["a"][0].tzinfo == UTC


def test_an_offset_with_its_own_arithmetic_cannot_reorder_the_instants() -> None:
    request, bundle = _matching_pair()

    class Zone(tzinfo):
        def utcoffset(self, dt):
            return OffsetWithOwnArithmetic(0)

        def dst(self, dt):
            return timedelta(0)

    stored = _STORED(_stored_checkpoint(bundle), BaseModel)
    dict.__setitem__(stored, "observed_at", datetime(*_NOON, tzinfo=UTC))
    dict.__setitem__(stored, "last_successful_verification_at", datetime(*_EVENING, tzinfo=Zone()))  # six hours after
    assert bundle_matches_reported_evidence(request, bundle) is False


# ---- F-2: direct receivers of the checkpoint

def _checkpoint_with(observed: datetime, verified: datetime) -> dict[str, object]:
    value = _checkpoint()
    value["observed_at"], value["last_successful_verification_at"] = observed, verified
    return value


def _direct(observed: datetime, verified: datetime) -> SourceCheckpoint:
    """Validate a checkpoint the way a direct receiver does, from Python objects (enum names as text)."""

    return SourceCheckpoint.model_validate(_checkpoint_with(observed, verified), strict=False)


def test_a_datetime_subclass_cannot_reorder_the_instants_for_a_direct_receiver() -> None:
    late = DatetimeWithOwnOffset(*_EVENING, tzinfo=UTC)  # really six hours after the observation
    with pytest.raises(ValidationError):
        _direct(datetime(*_NOON, tzinfo=UTC), late)


def test_a_zone_failing_on_a_later_read_is_a_value_error_for_a_direct_receiver() -> None:
    zone = FlappingZone(timedelta(0), RuntimeError("the zone went away"))
    observed = datetime(*_NOON, tzinfo=zone)
    zone.arm()
    with pytest.raises(ValidationError):  # not the caller's RuntimeError
        _direct(observed, datetime(2026, 9, 28, 11, tzinfo=UTC))


def test_a_direct_receiver_stores_a_plain_pinned_datetime() -> None:
    zone = FlappingZone(timedelta(hours=2), timedelta(hours=-5))
    checkpoint = _direct(datetime(*_NOON, tzinfo=zone), datetime(2026, 9, 28, 9, tzinfo=UTC))
    zone.arm()
    assert type(checkpoint.observed_at) is datetime and checkpoint.observed_at.tzinfo is not zone
    assert checkpoint.observed_at == datetime(*_NOON, tzinfo=timezone(timedelta(hours=2)))
    assert zone.reads == 0  # nothing reads the caller's zone after validation


def test_a_json_checkpoint_still_validates_and_pins_to_its_own_offset() -> None:
    checkpoint = SourceCheckpoint.model_validate_json(json.dumps(_checkpoint()))
    assert checkpoint.observed_at == datetime(2026, 9, 28, 12, tzinfo=UTC)
    witness = FreshnessWitness.model_validate_json(json.dumps({
        "upstream_status": "unverified", "source_checkpoints": [],
        "projection": {"status": "current", "applied_sources": [{"record_id": "r", "content_revision": 1, "policy_epoch": 1}],
                       "verified_at": "2026-09-28T12:00:00+02:00"},
    }))
    assert witness.projection.verified_at.utcoffset() == timedelta(hours=2)


def test_pin_datetime_leaves_non_datetimes_and_naive_values_for_validation_to_refuse() -> None:
    assert pin_datetime("2026-09-28T12:00:00Z") == "2026-09-28T12:00:00Z"
    naive = datetime(*_NOON)
    assert pin_datetime(naive) is naive
    assert pin_datetime(7) == 7


def test_the_verification_time_is_pinned_like_the_observation() -> None:
    zone = FlappingZone(timedelta(hours=1), timedelta(hours=-9))
    checkpoint = _direct(datetime(*_NOON, tzinfo=UTC), datetime(2026, 9, 28, 11, tzinfo=zone))
    zone.arm()
    assert type(checkpoint.last_successful_verification_at) is datetime
    assert checkpoint.last_successful_verification_at.tzinfo is not zone and zone.reads == 0


def test_a_pinned_offset_is_an_exact_timedelta_whatever_the_zone_returned() -> None:
    class Zone(tzinfo):
        def utcoffset(self, dt):
            return OffsetWithOwnArithmetic(hours=2)

        def dst(self, dt):
            return timedelta(0)

    pinned = pin_datetime(datetime(*_NOON, tzinfo=Zone()))
    assert type(pinned.utcoffset()) is timedelta and pinned.utcoffset() == timedelta(hours=2)
    assert pinned - datetime(*_NOON, tzinfo=UTC) == timedelta(hours=-2)
