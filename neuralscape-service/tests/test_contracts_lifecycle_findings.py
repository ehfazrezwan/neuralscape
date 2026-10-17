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


# ------------------------------------- shared native structure: work per object, not per path
def _diamond(depth: int, kind=tuple):
    node = kind()
    for _ in range(depth):
        node = kind((node, node)) if kind is not dict else {"a": node, "b": node}
    return node


@pytest.mark.parametrize("kind", [tuple, list, dict])
def test_a_tiny_shared_graph_costs_work_per_object_not_per_path(kind) -> None:
    """20 objects, each holding the next twice (2^20 paths): materialization must reuse what it already
    built for an object instead of rebuilding (and charging) every path through it."""

    import time

    from contracts_limits import DISCOVERY_BUDGET_MESSAGE

    from pydantic import BaseModel

    from tests.test_contracts_lifecycle import BASE_MODEL_DICT_DESCRIPTOR

    shared = receipt(CANONICAL, StageStatus.APPLIED)
    dict.__setitem__(BASE_MODEL_DICT_DESCRIPTOR.__get__(shared, BaseModel), "output_refs", (_diamond(20, kind),))
    started = time.monotonic()
    with pytest.raises(ValueError) as refused:
        _claim((shared,))
    rendered = str(refused.value)  # an error naming the input must not expand it either
    elapsed = time.monotonic() - started
    # Strict validation would walk all 2^20 paths, so the reused copy is charged per occurrence and the
    # input is refused by the budget; the charge is arithmetic over the 20 objects, so it is immediate.
    assert DISCOVERY_BUDGET_MESSAGE in rendered
    assert elapsed < 0.5, elapsed
    assert len(rendered) < 10_000


def test_a_shared_subtree_too_deep_on_one_path_is_still_refused_by_depth() -> None:
    """Reuse keeps the depth check per path: a subtree built first on a shallow path and then reached on a
    deep one is refused when the deep path exceeds the depth limit, exactly as without reuse."""

    from pydantic import BaseModel

    from contracts_limits import DISCOVERY_BUDGET_MESSAGE, PORTABILITY_MAX_DEPTH
    from tests.test_contracts_lifecycle import BASE_MODEL_DICT_DESCRIPTOR

    def claim_with(subtree):
        deep = subtree
        for _ in range(PORTABILITY_MAX_DEPTH - 6):  # the wrappers alone stay within the limit ...
            deep = (deep,)
        shared = receipt(CANONICAL, StageStatus.APPLIED)
        dict.__setitem__(BASE_MODEL_DICT_DESCRIPTOR.__get__(shared, BaseModel), "output_refs", (subtree, deep))
        _claim((shared,))

    with pytest.raises(ValueError) as shallow:
        claim_with(_diamond(0))
    assert DISCOVERY_BUDGET_MESSAGE not in str(shallow.value)  # ... refused only as invalid refs
    with pytest.raises(ValueError, match=DISCOVERY_BUDGET_MESSAGE):
        claim_with(_diamond(8))  # ... but a subtree of height 8 under them is too deep on that path


def test_a_malformed_first_receipt_does_not_copy_an_uncharged_tail() -> None:
    """The inventory stops at the first capture error, so the tail is never charged; reading the exact
    root must not copy it either (the error is raised at the first receipt)."""

    import gc
    import tracemalloc

    receipts = (_unreadable_receipt(),) + (None,) * 5_000_000
    gc.collect()
    tracemalloc.start()
    try:
        with pytest.raises(ValueError, match="colliding keys"):
            _claim(receipts)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 8 * 1024 * 1024, peak  # a copy of the tail alone is 40 MB


def _bad_intent_id(value):
    from pydantic import ValidationError

    from contracts_lifecycle import validate_required_stage_claim

    bad = intent(CANONICAL).model_copy(update={"id": value})
    with pytest.raises(ValidationError) as refused:
        validate_required_stage_claim(claimed_status=IntentStatus.APPLIED, intent=bad, receipts=())
    return refused.value


def test_a_flat_container_repeated_many_times_is_not_echoed_in_the_error() -> None:
    """A container with no nested containers is reused too, so (flat,) * n holds one copy of flat; an error that
    named the input would render n * n entries."""

    n = 300
    flat = ("x",) * n
    error = _bad_intent_id((flat,) * n)
    assert len(repr(error.errors())) < 10_000  # the input would be about 450,000 characters
    assert "omitted: input shares subtrees" in repr(error.errors())


def test_an_input_that_shares_nothing_keeps_its_full_diagnostic() -> None:
    error = _bad_intent_id(("x",) * 3)
    assert "('x', 'x', 'x')" in repr(error.errors())


@pytest.mark.parametrize("kind", [list, tuple, dict])
def test_a_flat_container_of_any_kind_repeated_many_times_is_not_echoed(kind) -> None:
    n = 300
    flat = {"k": "x"} if kind is dict else kind(("x",) * n)
    if kind is dict:
        flat = {f"k{i}": "x" for i in range(n)}
    wrapper = [flat] * n if kind is list else tuple([flat] * n)
    error = _bad_intent_id(wrapper)
    assert len(repr(error.errors())) < 10_000


# ---- strict validation walks a shared raw container once per occurrence: it is charged per occurrence


def _requirements_sharing_one_raw_dict(n: int, ids: tuple[str, ...] | None = None):
    from contracts_lifecycle import ProcessingStage as Stage

    command = intent(CANONICAL)
    shared = {  # strict input: the exact types the model expects
        "stage": Stage.CANONICAL, "effect_id": "effect-canonical",
        "precondition_source_ids": ids if ids is not None else tuple(f"memory-{i}" for i in range(n)),
        "upstream_stages": (),
        "committed_sources": (), "minimum_output_refs": 0, "no_source_witness": False,
    }
    return command.model_copy(update={"stage_requirements": (shared,) * n})


def test_many_references_to_one_raw_requirement_are_charged_for_every_occurrence() -> None:
    """N requirement entries naming one raw dict of N source ids: one dict to the graph walk, N * N entries
    for strict validation.  Refused by the budget before validation, in bounded time."""

    import time

    from contracts_limits import DISCOVERY_BUDGET_MESSAGE

    command = _requirements_sharing_one_raw_dict(2_000)
    started = time.monotonic()
    with pytest.raises(ValueError, match=DISCOVERY_BUDGET_MESSAGE):
        _claim_with(command, ())
    assert time.monotonic() - started < 2


def test_receipts_sharing_one_raw_output_tuple_are_charged_for_every_occurrence() -> None:
    import time

    from pydantic import BaseModel

    from contracts_limits import DISCOVERY_BUDGET_MESSAGE
    from tests.test_contracts_lifecycle import BASE_MODEL_DICT_DESCRIPTOR

    handles = tuple({"kind": "memory", "id": f"m{i}", "tenant_id": "tenant-1", "resolver": "ns:memory"}
                    for i in range(1_500))
    receipts = []
    for attempt in range(1, 1_501):
        one = receipt(CANONICAL, StageStatus.APPLIED).model_copy(update={"attempt": attempt})
        dict.__setitem__(BASE_MODEL_DICT_DESCRIPTOR.__get__(one, BaseModel), "output_refs", handles)
        receipts.append(one)
    started = time.monotonic()
    with pytest.raises(ValueError, match=DISCOVERY_BUDGET_MESSAGE):
        _claim_with(intent(CANONICAL), tuple(receipts))
    assert time.monotonic() - started < 3


def test_a_raw_container_shared_a_few_times_still_validates() -> None:
    from contracts_limits import DISCOVERY_BUDGET_MESSAGE

    command = _requirements_sharing_one_raw_dict(1, ("memory-1",))  # one entry: nothing is repeated
    _claim_with(command, (), IntentStatus.ACCEPTED)
    shared = _requirements_sharing_one_raw_dict(3, ("memory-1",))
    with pytest.raises(ValueError) as refused:  # three entries for one stage: the contract refuses it
        _claim_with(shared, (), IntentStatus.ACCEPTED)
    assert DISCOVERY_BUDGET_MESSAGE not in str(refused.value)


def _claim_with(command, receipts, status=IntentStatus.APPLIED):
    from contracts_lifecycle import validate_required_stage_claim

    validate_required_stage_claim(claimed_status=status, intent=command, receipts=receipts)
