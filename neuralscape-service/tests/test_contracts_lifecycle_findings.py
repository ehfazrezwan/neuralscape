"""Lifecycle contract pins: one offset read per timestamp, bounded receipt-inventory retention, and work
per distinct object (not per path) when materializing shared native structure."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone, tzinfo

import pytest
from pydantic import ValidationError

from contracts_lifecycle import IntentStatus, ProcessingStage, StageReceipt, StageStatus
from tests.test_contracts_lifecycle import NOW, intent, receipt

CANONICAL = ProcessingStage.CANONICAL


class ShiftingZone(tzinfo):
    """Answers UTC-5 on its first ``utcoffset`` read and UTC on every later one (outside the envelope)."""

    def __init__(self) -> None:
        self.reads = 0

    def utcoffset(self, dt):
        self.reads += 1
        return timedelta(hours=-5) if self.reads == 1 else timedelta(0)

    def dst(self, dt):
        return timedelta(0)

    def tzname(self, dt):
        return "shifting"


def _with_times(started_at: datetime, finished_at: datetime) -> dict:
    values = receipt(CANONICAL, StageStatus.APPLIED).model_dump()
    values["started_at"], values["finished_at"] = started_at, finished_at
    return values


def test_finished_at_ordering_uses_the_same_offset_read_as_validation() -> None:
    """Start 12:00 in a zone first read as UTC-5 is 17:00Z; a finish at 14:00Z precedes it.  The ordering
    must use that read, not ask the zone again (which would answer UTC and make the start 12:00Z)."""

    zone = ShiftingZone()
    started = datetime(2026, 9, 28, 12, 0, tzinfo=zone)
    with pytest.raises(ValidationError, match="finished_at cannot precede started_at"):
        StageReceipt.model_validate(_with_times(started, datetime(2026, 9, 28, 14, 0, tzinfo=timezone.utc)))
    assert zone.reads == 1  # read once, and that read decides both validation and ordering


def test_each_timestamp_is_pinned_to_its_one_offset_read() -> None:
    zone = ShiftingZone()
    started = datetime(2026, 9, 28, 12, 0, tzinfo=zone)  # 17:00Z on its one read
    checked = StageReceipt.model_validate(_with_times(started, datetime(2026, 9, 28, 18, 0, tzinfo=timezone.utc)))
    assert checked.started_at.utcoffset() == timedelta(hours=-5) and zone.reads == 1
    assert checked.started_at.tzinfo is not zone  # the validated copy no longer asks the caller's zone
    assert checked.started_at == datetime(2026, 9, 28, 17, 0, tzinfo=timezone.utc)


def test_ordinary_zones_and_the_repeated_hour_are_unchanged() -> None:
    plain = StageReceipt.model_validate(_with_times(NOW, NOW + timedelta(seconds=1)))
    assert plain.started_at == NOW and plain.started_at.utcoffset() == timedelta(0)
    with pytest.raises(ValidationError, match="finished_at cannot precede started_at"):
        StageReceipt.model_validate(_with_times(NOW + timedelta(seconds=1), NOW))
    naive = _with_times(NOW, NOW + timedelta(seconds=1))
    naive["started_at"] = datetime(2026, 9, 28, 12, 0)
    with pytest.raises(ValidationError):
        StageReceipt.model_validate(naive)  # a naive timestamp is still refused


# ------------------------------------------------ receipt inventory: one retained capture error
class _CollidingKey(str):
    """Spells a stored field's name but hashes and compares as a different key."""

    def __hash__(self) -> int:
        return 7

    def __eq__(self, other: object) -> bool:
        return self is other


def _unreadable_receipt() -> StageReceipt:
    from pydantic import BaseModel

    from tests.test_contracts_lifecycle import BASE_MODEL_DICT_DESCRIPTOR

    broken = receipt(CANONICAL, StageStatus.APPLIED)
    dict.__setitem__(BASE_MODEL_DICT_DESCRIPTOR.__get__(broken, BaseModel), _CollidingKey("status"), "applied")
    return broken


def _claim(receipts):
    from contracts_lifecycle import validate_required_stage_claim

    validate_required_stage_claim(claimed_status=IntentStatus.APPLIED, intent=intent(CANONICAL), receipts=receipts)


def test_distinct_unreadable_receipts_do_not_each_keep_an_exception() -> None:
    """Only the first capture error of an exact root tuple is ever raised; the inventory must not keep one
    exception (with its traceback) per later unreadable receipt."""

    import gc
    import tracemalloc

    receipts = tuple(_unreadable_receipt() for _ in range(32_000))
    gc.collect()
    tracemalloc.start()
    try:
        with pytest.raises(ValueError, match="colliding keys"):
            _claim(receipts)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 40 * 1024 * 1024, peak


def test_the_first_unreadable_receipt_still_reports_its_own_error_in_order() -> None:
    readable = receipt(CANONICAL, StageStatus.APPLIED)
    with pytest.raises(ValueError, match="colliding keys"):
        _claim((readable, _unreadable_receipt(), _unreadable_receipt()))
    with pytest.raises(TypeError, match="value must be a StageReceipt"):
        _claim((object(), _unreadable_receipt()))  # an earlier receipt's own error comes first
