"""Caller-owned zones and keys at the context and freshness boundaries: one read, no caller code, bounded memory."""

from __future__ import annotations

import json
import tracemalloc
from datetime import datetime, timedelta, timezone, tzinfo

import pytest
from pydantic import BaseModel, ValidationError

from contracts_context import ContextBundle, ContextRequest, bundle_matches_reported_evidence
from contracts_freshness import (
    FreshnessWitness,
    SourceCheckpoint,
    pin_datetime,
    pin_native_snapshot,
    require_native_snapshot,
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


# ---- F-3: the path strings

def test_the_native_snapshot_check_does_not_copy_a_key_into_every_descendant_path() -> None:
    key = "k" * (1 << 20)  # 1 MiB
    snapshot = {key: [{"leaf": index} for index in range(200)]}
    tracemalloc.start()
    try:
        require_native_snapshot(snapshot)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 8 * (1 << 20), peak  # 200 copies of the key (about 200 MiB) before the fix


def test_a_refusal_names_a_bounded_path_not_the_whole_key() -> None:
    key = "k" * (1 << 20)
    with pytest.raises(ValueError) as refused:
        require_native_snapshot({key: [object()]})
    assert len(str(refused.value)) < 400 and "unsupported value type at $" in str(refused.value)


# ---- F-4: keys

@pytest.mark.parametrize(
    "make_key", [lambda: object(), lambda: type("Text", (str,), {"__hash__": lambda self: 1})("a")],
    ids=["non-text-key", "text-subclass-key-with-hooks"],
)
def test_the_native_snapshot_check_refuses_unsupported_keys_as_a_value_error(make_key) -> None:
    with pytest.raises(ValueError, match="unsupported key type"):
        require_native_snapshot({make_key(): 1})


def test_a_key_whose_repr_raises_never_reaches_the_caller() -> None:
    class Hostile:
        def __repr__(self):
            raise RuntimeError("repr ran")

        __hash__ = object.__hash__

    with pytest.raises(ValueError, match="unsupported key type"):
        require_native_snapshot({Hostile(): 1})


def test_exact_native_scalar_keys_are_still_left_to_strict_validation() -> None:
    assert require_native_snapshot({7: "x", "a": 1, None: 2}) == {7: "x", "a": 1, None: 2}


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


# ---- review round: offsets with their own subtraction, hostile and foreign-enum keys, the combined entry budget

import enum  # noqa: E402

import contracts_freshness  # noqa: E402
from contracts_freshness import projection_status_for  # noqa: E402
from contracts_references import SourceVersion  # noqa: E402


def test_an_offset_with_its_own_subtraction_cannot_make_a_later_verification_look_earlier_for_a_direct_receiver() -> None:
    class Zone(tzinfo):
        def utcoffset(self, dt):
            return OffsetWithOwnArithmetic(0)  # really UTC; its subtraction claims otherwise

        def dst(self, dt):
            return timedelta(0)

    with pytest.raises(ValidationError):  # 13:00 is after the 12:00 observation, whatever the offset's arithmetic says
        _direct(datetime(2026, 9, 28, 12, tzinfo=UTC), datetime(2026, 9, 28, 13, tzinfo=Zone()))
    with pytest.raises(ValidationError):
        _direct(datetime(2026, 9, 28, 12, tzinfo=Zone()), datetime(2026, 9, 28, 13, tzinfo=Zone()))


def test_a_hostile_key_runs_none_of_its_callbacks_and_is_refused() -> None:
    calls: list[str] = []

    class HostileKey:
        def __repr__(self):
            calls.append("repr")
            return "key"

        def __hash__(self):
            calls.append("hash")
            return 1

        def __eq__(self, other):
            calls.append("eq")
            return False

        def __str__(self):
            calls.append("str")
            return "key"

    snapshot = {HostileKey(): 1}
    calls.clear()  # inserting the key hashed it; only what the check does counts
    with pytest.raises(ValueError, match="unsupported key type"):
        require_native_snapshot({"outer": [snapshot]})
    assert calls == []


def test_a_foreign_enum_key_is_refused_like_a_foreign_enum_value() -> None:
    class Foreign(enum.Enum):
        A = "a"

    with pytest.raises(ValueError, match="unsupported key type"):
        require_native_snapshot({Foreign.A: 1})
    with pytest.raises(ValueError, match="unsupported value type"):
        require_native_snapshot({"a": Foreign.A})


def _versions(count: int, start: int = 0) -> list[SourceVersion]:
    return [SourceVersion(record_id=f"r{start + index}", content_revision=1, policy_epoch=1) for index in range(count)]


class _View(list):
    """A list subclass: its public iterator is the only way in, so every entry counts against the budget."""


def test_native_entries_count_against_the_combined_entry_budget(monkeypatch) -> None:
    monkeypatch.setattr(contracts_freshness, "_MAX_SEQUENCE_ENTRIES", 10)
    native = _versions(3)
    assert projection_status_for(_View(_versions(7, 100)), native) is not None  # 3 native + 7 iterated = 10: allowed
    with pytest.raises(ValueError, match="budget"):
        projection_status_for(_View(_versions(8, 100)), native)  # 11 in all: one over


def test_a_native_group_over_the_budget_is_refused_before_anything_is_copied(monkeypatch) -> None:
    monkeypatch.setattr(contracts_freshness, "_MAX_SEQUENCE_ENTRIES", 10)
    captured: list[object] = []
    real = contracts_freshness.snapshot_contract_graph
    monkeypatch.setattr(contracts_freshness, "snapshot_contract_graph", lambda value, *a, **k: captured.append(value) or real(value, *a, **k))
    with pytest.raises(ValueError, match="budget"):
        projection_status_for(_versions(11), _versions(1, 100))
    assert captured == []


def test_the_budget_counts_both_native_sequences_together(monkeypatch) -> None:
    monkeypatch.setattr(contracts_freshness, "_MAX_SEQUENCE_ENTRIES", 10)
    assert projection_status_for(_versions(5), _versions(5, 100)) is not None
    with pytest.raises(ValueError, match="budget"):
        projection_status_for(_versions(6), _versions(5, 100))


def test_the_documented_limit_holds_at_its_real_size_for_one_native_and_a_full_iterator() -> None:
    limit = contracts_freshness._MAX_SEQUENCE_ENTRIES
    entries = _versions(limit + 1)  # distinct sources, one more than the limit allows in all

    class Lazy(_View):
        def __iter__(self):
            return iter(entries[1:])  # `limit` iterated entries: with one native entry, one over

    with pytest.raises(ValueError, match="budget"):
        projection_status_for(Lazy(), [entries[0]])


# ---- sequence class gate


def test_a_sequence_class_with_a_metaclass_eq_runs_no_code_before_the_arguments_are_captured() -> None:
    from tests.test_contracts_freshness import _invalid, _sv

    for hold in (tuple, list):
        bad = _invalid()
        calls: list[str] = []

        class Meta(type):
            def __eq__(cls, other):
                calls.append("eq")
                object.__setattr__(bad, "content_revision", 1)  # repairs the invalid model the other argument holds
                return NotImplemented

            __hash__ = type.__hash__

        class Public(list, metaclass=Meta):
            pass

        with pytest.raises(Exception):
            projection_status_for(Public([_sv("rec-1", 1)]), hold([bad]))
        assert calls == []  # the class was judged by identity: its hook never ran


# ---- a zone that stops answering
class _StopsAnswering(tzinfo):
    """Gives an offset for the first ``answers`` reads (validation), then ``None`` (the pin)."""

    def __init__(self, answers: int) -> None:
        self.answers, self.reads = answers, 0

    def utcoffset(self, dt):
        self.reads += 1
        return timedelta(0) if self.reads <= self.answers else None

    def dst(self, dt):
        return timedelta(0)


@pytest.mark.parametrize("answers", [1, 2, 3])
def test_a_zone_that_stops_answering_never_leaves_a_caller_zone_on_a_stored_datetime(answers) -> None:
    from pydantic import BaseModel

    from contracts_freshness import PinnedAwareDatetime

    class Holder(BaseModel):
        at: PinnedAwareDatetime

    zone = _StopsAnswering(answers)
    try:
        held = Holder(at=datetime(*_NOON, tzinfo=zone))
    except ValidationError:
        return  # refused is fine
    assert type(held.at) is datetime and type(held.at.tzinfo) is timezone, type(held.at.tzinfo)


def test_ordering_two_datetimes_refuses_a_zone_that_stops_answering() -> None:
    from contracts_freshness import _instant_after

    zone = _StopsAnswering(0)
    with pytest.raises(ValueError):
        _instant_after(datetime(*_NOON, tzinfo=zone), datetime(*_EVENING, tzinfo=UTC))


# ---- shared containers
def test_a_datetime_in_a_container_shared_by_two_parents_is_pinned_on_every_path() -> None:
    zone = FlappingZone(timedelta(0), timedelta(hours=9))
    shared = [datetime(*_NOON, tzinfo=zone)]
    zone.arm()
    pinned = pin_native_snapshot({"a": {"k": shared}, "b": {"k": shared}})
    assert pinned["a"]["k"] is pinned["b"]["k"]  # the sharing is kept
    assert type(pinned["a"]["k"][0].tzinfo) is timezone and zone.reads == 1


def test_a_shared_container_reached_before_its_other_parent_is_pinned_on_both_paths() -> None:
    zone = FlappingZone(timedelta(0), timedelta(hours=9))
    shared = {"when": datetime(*_NOON, tzinfo=zone)}
    zone.arm()
    pinned = pin_native_snapshot({"a": [shared], "b": shared})
    for found in (pinned["a"][0]["when"], pinned["b"]["when"]):
        assert type(found) is datetime and type(found.tzinfo) is timezone
    assert pinned["a"][0] is pinned["b"] and zone.reads == 1


def test_pinning_a_snapshot_that_contains_itself_terminates() -> None:
    loop: list[object] = [datetime(*_NOON, tzinfo=UTC)]
    loop.append(loop)
    pinned = pin_native_snapshot({"loop": loop})
    assert type(pinned["loop"][0]) is datetime


# ---- both predicates capture their two arguments together, before any callback runs


def _bundle_with_a_negative_source_revision():
    """A bundle that is invalid on entry (an unchecked copy holds source revision -1) and the object to repair."""

    from contracts_context import ContextAssemblyReceipt
    from tests.test_contracts_context import _receipt

    request, _ = _matching_pair()
    bundle = _validate(ContextBundle, _bundle())  # the bundle the default receipt describes
    item = bundle.selected_items[0]
    bad = item.source_versions[0].model_copy(update={"content_revision": -1})
    broken = bundle.model_copy(update={"selected_items": (item.model_copy(update={"source_versions": (bad,)}),)})
    receipt = _validate(ContextAssemblyReceipt, _receipt())
    return request, receipt, broken, bad


def _repair(bad) -> None:
    object.__setattr__(bad, "content_revision", 4)  # the revision the matching receipt names


class _IteratorThatRepairs(tuple):
    """A public tuple whose iterator repairs an invalid model held natively by the other argument."""

    def __new__(cls, items, callback):
        instance = super().__new__(cls, items)
        instance.callback = callback
        return instance

    def __iter__(self):
        self.callback()
        return iter(tuple.__iter__(self))


class _ZoneThatRepairs(tzinfo):
    """A zone whose offset read repairs an invalid model held natively by the other argument."""

    def __init__(self, callback) -> None:
        self.callback = callback

    def utcoffset(self, dt):
        self.callback()
        return timedelta(0)

    def dst(self, dt):
        return timedelta(0)


def test_a_public_iterator_in_the_receipt_cannot_repair_the_bundle_before_it_is_captured() -> None:
    from contracts_context import receipt_matches_bundle

    _, receipt, broken, bad = _bundle_with_a_negative_source_revision()
    assert not receipt_matches_bundle(receipt, broken)  # control: invalid on entry
    object.__setattr__(
        receipt, "source_versions", _IteratorThatRepairs(receipt.source_versions, lambda: _repair(bad))
    )
    assert not receipt_matches_bundle(receipt, broken)
    assert bad.content_revision == 4  # the callback did run, after the bundle had been captured


def test_a_deadline_zone_callback_cannot_repair_the_bundle_before_it_is_captured() -> None:
    request, _, broken, bad = _bundle_with_a_negative_source_revision()
    assert not bundle_matches_reported_evidence(request, broken)  # control: invalid on entry
    stored = _STORED(request, BaseModel)
    dict.__setitem__(stored, "deadline", datetime(*_NOON, tzinfo=_ZoneThatRepairs(lambda: _repair(bad))))
    assert not bundle_matches_reported_evidence(request, broken)
    assert bad.content_revision == 4


def test_the_pair_snapshot_keeps_the_predicates_true_for_valid_arguments_with_callbacks() -> None:
    from contracts_context import receipt_matches_bundle

    request, receipt, _, _ = _bundle_with_a_negative_source_revision()
    _, bundle = _matching_pair()
    plain_bundle = _validate(ContextBundle, _bundle())
    stored = _STORED(request, BaseModel)
    seen = []
    dict.__setitem__(stored, "deadline", datetime(*_NOON, tzinfo=_ZoneThatRepairs(lambda: seen.append(1))))
    assert bundle_matches_reported_evidence(request, bundle)
    assert seen  # the zone was consulted
    object.__setattr__(receipt, "source_versions", _IteratorThatRepairs(receipt.source_versions, lambda: seen.append(2)))
    assert receipt_matches_bundle(receipt, plain_bundle)


# ---- an entry exposed only by a public iterator is captured before that iterator advances


class _EmptyBackedGenerated(tuple):
    """A tuple with nothing in its native backing: every item exists only through the generator ``make`` returns."""

    def __new__(cls, make):
        instance = super().__new__(cls, ())
        instance.make = make
        return instance

    def __iter__(self):
        return self.make()


def _receipt_and_bundle_with_source_versions(make):
    from contracts_context import ContextAssemblyReceipt
    from tests.test_contracts_context import _receipt

    receipt = _validate(ContextAssemblyReceipt, _receipt())
    bundle = _validate(ContextBundle, _bundle())
    object.__setattr__(receipt, "source_versions", _EmptyBackedGenerated(make))
    return receipt, bundle


def test_an_entry_yielded_then_repaired_on_resume_is_judged_as_it_was_yielded() -> None:
    from contracts_context import receipt_matches_bundle

    def make():
        entry = {"record_id": "source-a", "content_revision": -1, "policy_epoch": 2}  # invalid as yielded
        yield entry
        entry["content_revision"] = 4  # repaired when the consumer asks for the next item (the bundle's value)

    receipt, bundle = _receipt_and_bundle_with_source_versions(make)
    assert receipt_matches_bundle(receipt, bundle) is False


def test_an_honest_generated_entry_still_matches() -> None:
    from contracts_context import receipt_matches_bundle

    def make():
        yield {"record_id": "source-a", "content_revision": 4, "policy_epoch": 2}

    receipt, bundle = _receipt_and_bundle_with_source_versions(make)
    assert receipt_matches_bundle(receipt, bundle) is True


def test_a_generated_entry_repaired_on_resume_in_the_bundle_is_judged_as_it_was_yielded() -> None:
    """The bundle's own public sequence is read the same way."""

    from contracts_context import ContextAssemblyReceipt, receipt_matches_bundle
    from tests.test_contracts_context import _receipt

    receipt = _validate(ContextAssemblyReceipt, _receipt())
    bundle = _validate(ContextBundle, _bundle())
    item = bundle.selected_items[0]

    def make():
        entry = {"record_id": "source-a", "content_revision": -1, "policy_epoch": 2}
        yield entry
        entry["content_revision"] = 4

    object.__setattr__(item, "source_versions", _EmptyBackedGenerated(make))
    assert receipt_matches_bundle(receipt, bundle) is False


# ---- every enum-typed input is a declared member by identity (``str.__new__(Enum, "value")`` is refused)


def _every_enum():
    import enum as _enum

    import contracts_context
    import contracts_freshness

    return [
        value
        for module in (contracts_context, contracts_freshness)
        for value in vars(module).values()
        if isinstance(value, type) and issubclass(value, _enum.Enum) and value.__module__ == module.__name__
    ]


def _spoof(member):
    return str.__new__(type(member), member.value)  # right class and value, but not a declared member


def test_the_enum_sweep_covers_every_enum_of_both_modules() -> None:
    names = {kind.__name__ for kind in _every_enum()}
    assert {"FreshnessRequirement", "UpstreamVerificationStatus", "ProjectionStatus", "VerificationMethod",
            "ReconciliationState", "TimePerspective", "ContextOutcome", "CapabilityStatus", "TemporalStatus",
            "EvidenceForm", "Uncertainty", "SafeOmissionCode", "SafeDegradationCode"} <= names


@pytest.mark.parametrize("kind", _every_enum(), ids=lambda kind: kind.__name__)
def test_a_contract_field_refuses_a_noncanonical_enum_instance_and_keeps_every_member(kind) -> None:
    from pydantic import create_model

    from contracts_common import ContractModel

    holder = create_model("Holder", __base__=ContractModel, value=(kind, ...), many=(tuple[kind, ...], ()))
    for member in kind:
        assert holder.model_validate({"value": member}).value is member
        assert holder.model_validate_json(json.dumps({"value": member.value})).value is member  # JSON: the member
        with pytest.raises(ValidationError):
            holder.model_validate({"value": _spoof(member)})
        with pytest.raises(ValidationError):
            holder.model_validate({"value": member, "many": (member, _spoof(member))})  # also inside a tuple


@pytest.mark.parametrize("kind", _every_enum(), ids=lambda kind: kind.__name__)
def test_the_native_snapshot_refuses_a_noncanonical_enum_instance_and_keeps_every_member(kind) -> None:
    from contracts_freshness import require_native_snapshot

    for member in kind:
        assert require_native_snapshot({"x": [member]}) == {"x": [member]}
        with pytest.raises(ValueError, match="noncanonical enum"):
            require_native_snapshot({"x": [_spoof(member)]})


def test_upstream_degradation_for_refuses_a_noncanonical_status_and_maps_every_member() -> None:
    from contracts_context import SafeDegradationCode, upstream_degradation_for
    from contracts_freshness import UpstreamVerificationStatus

    assert upstream_degradation_for(UpstreamVerificationStatus.STALE) is SafeDegradationCode.UPSTREAM_VERIFICATION_STALE
    assert upstream_degradation_for(UpstreamVerificationStatus.CURRENT) is None
    for member in UpstreamVerificationStatus:
        with pytest.raises(ValueError, match="declared member"):
            upstream_degradation_for(_spoof(member))  # was None: the stale status reported no degradation


def _stale_bundle():
    item = _item()
    freshness = item["freshness"]
    freshness["upstream_status"] = "stale"
    freshness["source_checkpoints"][0].update(verification_status="stale", last_successful_verification_at="2026-09-28T11:00:00Z")
    return _validate(ContextBundle, _bundle(selected_items=[item]))


def test_a_noncanonical_current_verified_request_cannot_match_stale_evidence() -> None:
    from contracts_context import FreshnessRequirement

    request, _ = _matching_pair()
    bundle = _stale_bundle()
    canonical = request.model_copy(update={"freshness_requirement": FreshnessRequirement.CURRENT_VERIFIED})
    spoofed = request.model_copy(update={"freshness_requirement": _spoof(FreshnessRequirement.CURRENT_VERIFIED)})
    stale_ok = request.model_copy(update={"freshness_requirement": FreshnessRequirement.LABELLED_STALE_ACCEPTABLE})
    assert bundle_matches_reported_evidence(canonical, bundle) is False
    assert bundle_matches_reported_evidence(spoofed, bundle) is False  # was True: the stale-acceptable branch
    assert bundle_matches_reported_evidence(stale_ok, bundle) is True  # control: the labelled request still matches


@pytest.mark.parametrize(
    "where",
    ["time_perspective", "freshness_requirement", "outcome", "temporal_status", "upstream_status", "projection_status",
     "verification_method"],
)
def test_a_noncanonical_enum_in_either_argument_makes_the_predicates_false(where) -> None:
    from contracts_context import receipt_matches_bundle
    from tests.test_contracts_context import _receipt
    from contracts_context import ContextAssemblyReceipt

    request, bundle = _matching_pair()
    assert bundle_matches_reported_evidence(request, bundle) is True
    item = bundle.selected_items[0]
    if where in ("time_perspective", "freshness_requirement"):
        request = request.model_copy(update={where: _spoof(getattr(request, where))})
    elif where == "outcome":
        bundle = bundle.model_copy(update={"outcome": _spoof(bundle.outcome)})
    else:
        freshness = item.freshness
        if where == "temporal_status":
            item = item.model_copy(update={"temporal_status": _spoof(item.temporal_status)})
        elif where == "upstream_status":
            item = item.model_copy(update={"freshness": freshness.model_copy(update={"upstream_status": _spoof(freshness.upstream_status)})})
        elif where == "projection_status":
            projection = freshness.projection.model_copy(update={"status": _spoof(freshness.projection.status)})
            item = item.model_copy(update={"freshness": freshness.model_copy(update={"projection": projection})})
        else:
            checkpoint = freshness.source_checkpoints[0]
            checkpoint = checkpoint.model_copy(update={"verification_method": _spoof(checkpoint.verification_method)})
            item = item.model_copy(update={"freshness": freshness.model_copy(update={"source_checkpoints": (checkpoint,)})})
        bundle = bundle.model_copy(update={"selected_items": (item,)})
    assert bundle_matches_reported_evidence(request, bundle) is False
    receipt = _validate(ContextAssemblyReceipt, _receipt())
    assert receipt_matches_bundle(receipt, _validate(ContextBundle, _bundle())) in (True, False)  # control: callable


# ---- a key's label in a diagnostic is fixed or bounded, and a pin must succeed before it is cached


class _ZoneWithHostileText(tzinfo):
    """Fixed offset, but its repr and str are caller code that raises (and counts)."""

    texts = 0

    def utcoffset(self, dt):
        return timedelta(0)

    def dst(self, dt):
        return None

    def __repr__(self):
        _ZoneWithHostileText.texts += 1
        raise RuntimeError("caller zone repr")

    __str__ = __repr__


@pytest.mark.parametrize(
    "build",
    [
        lambda key: {key: {"x": object()}},
        lambda key: {key: {object(): 1}},
        lambda key: {key: {key: [object()]}},
        lambda key: {"a": [{key: {"b": object()}}]},
    ],
    ids=["value-beneath", "key-beneath", "two-levels", "nested-in-a-list"],
)
def test_a_datetime_key_is_labelled_without_running_its_zone_text(build) -> None:
    from contracts_freshness import require_native_snapshot

    key = datetime(2026, 1, 1, tzinfo=_ZoneWithHostileText())
    _ZoneWithHostileText.texts = 0
    with pytest.raises(ValueError) as caught:
        require_native_snapshot(build(key))
    assert _ZoneWithHostileText.texts == 0
    assert "unsupported" in str(caught.value) and len(str(caught.value)) < 400


def test_other_native_key_kinds_keep_a_short_label_and_a_huge_int_key_is_bounded() -> None:
    from contracts_freshness import require_native_snapshot

    for key, shown in ((True, "True"), (None, "None"), (7, "7"), (1.5, "1.5"), ("name", "'name'")):
        with pytest.raises(ValueError, match=r"unsupported value type at \$\[" + shown.replace(".", r"\.") + r"\]"):
            require_native_snapshot({key: object()})
    with pytest.raises(ValueError) as caught:
        require_native_snapshot({10**5000: object()})
    assert len(str(caught.value)) < 400 and "unsupported value type" in str(caught.value)


class _NoOffsetOnce(tzinfo):
    """The first offset read answers ``None``; every later one answers +3 h."""

    reads = 0

    def utcoffset(self, dt):
        _NoOffsetOnce.reads += 1
        return None if _NoOffsetOnce.reads == 1 else timedelta(hours=3)

    def dst(self, dt):
        return None


def test_pinning_a_snapshot_refuses_a_zone_that_reported_no_offset_at_the_pin() -> None:
    zone = _NoOffsetOnce()
    when = datetime(2026, 1, 1, 8, 0, tzinfo=zone)
    _NoOffsetOnce.reads = 0
    with pytest.raises(ValueError, match="usable UTC offset"):
        pin_native_snapshot({"a": when, "b": [when]})
    assert _NoOffsetOnce.reads == 1  # asked once; the unusable answer is never cached as a result


def test_pinning_still_passes_a_naive_datetime_on_to_validation_and_pins_a_good_one() -> None:
    naive = datetime(2026, 1, 1, 8, 0)
    good = datetime(2026, 1, 1, 8, 0, tzinfo=timezone(timedelta(hours=1)))
    out = pin_native_snapshot({"n": naive, "g": good})
    assert out["n"] is naive  # refused later by validation, as before
    assert type(out["g"]) is datetime and type(out["g"].tzinfo) is timezone and out["g"] == good
