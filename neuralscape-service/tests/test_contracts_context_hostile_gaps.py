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


def _spoofed_pair(where: str):
    """A matching current (request, bundle) pair with one field replaced by a look-alike enum instance."""

    request, bundle = _matching_pair()
    item = bundle.selected_items[0]
    if where in ("time_perspective", "freshness_requirement"):
        return request.model_copy(update={where: _spoof(getattr(request, where))}), bundle
    if where == "outcome":
        return request, bundle.model_copy(update={"outcome": _spoof(bundle.outcome)})
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
    return request, bundle.model_copy(update={"selected_items": (item,)})


_REQUEST_FIELDS = ["time_perspective", "freshness_requirement"]
_BUNDLE_FIELDS = ["outcome", "temporal_status", "upstream_status", "projection_status", "verification_method"]


@pytest.mark.parametrize("where", _REQUEST_FIELDS + _BUNDLE_FIELDS)
def test_a_noncanonical_enum_makes_the_evidence_predicate_false_where_the_unmodified_pair_matches(where) -> None:
    request, bundle = _matching_pair()
    assert bundle_matches_reported_evidence(request, bundle) is True  # the control: this very pair matches
    spoofed_request, spoofed_bundle = _spoofed_pair(where)
    assert bundle_matches_reported_evidence(spoofed_request, spoofed_bundle) is False


@pytest.mark.parametrize("where", _BUNDLE_FIELDS)
def test_a_noncanonical_enum_in_the_bundle_makes_the_receipt_predicate_false_where_the_unmodified_bundle_matches(where) -> None:
    from contracts_context import ContextAssemblyReceipt, receipt_matches_bundle
    from tests.test_contracts_context import _receipt

    receipt = _validate(ContextAssemblyReceipt, _receipt())
    bundle = _validate(ContextBundle, _bundle())
    assert receipt_matches_bundle(receipt, bundle) is True  # the control: this very receipt and bundle match
    item = bundle.selected_items[0]
    freshness = item.freshness
    if where == "outcome":
        bad = bundle.model_copy(update={"outcome": _spoof(bundle.outcome)})
    else:
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
        bad = bundle.model_copy(update={"selected_items": (item,)})
    assert receipt_matches_bundle(receipt, bad) is False


def test_the_enum_check_visits_a_shared_container_once_not_once_per_path(monkeypatch) -> None:
    """16 tuple layers each holding the previous twice are 2**16 paths to one leaf; the check keeps what it has seen."""

    import contracts_common

    checked = [0]
    original = contracts_common.is_canonical_enum_member

    def counting(value):
        checked[0] += 1
        return original(value)

    monkeypatch.setattr(contracts_common, "is_canonical_enum_member", counting)
    from contracts_context import TimePerspective

    layer: object = TimePerspective.CURRENT
    for _ in range(20):
        layer = (layer, layer)
    contracts_common.require_canonical_enum_members(layer)
    assert checked[0] < 100, checked[0]  # per path it was 2**20 = 1,048,576
    bad: object = _spoof(TimePerspective.CURRENT)
    for _ in range(20):
        bad = (bad, bad)
    with pytest.raises(ValueError, match="declared member"):
        contracts_common.require_canonical_enum_members(bad)


def test_an_invalid_yielded_entry_stops_the_view_before_its_tail_is_read() -> None:
    """The first source version is invalid and 50,000 valid ones follow: the invalid entry is validated as it is
    captured, before the iterator is advanced, so the tail is never read."""

    from contracts_context import ContextAssemblyReceipt, receipt_matches_bundle
    from tests.test_contracts_context import _receipt

    read = [0]

    def make():
        read[0] += 1
        yield {"record_id": "source-a", "content_revision": -1, "policy_epoch": 2}  # invalid
        for index in range(50_000):
            read[0] += 1
            yield {"record_id": f"tail-{index}", "content_revision": 1, "policy_epoch": 1}

    receipt, bundle = _receipt_and_bundle_with_source_versions(make)
    assert receipt_matches_bundle(receipt, bundle) is False
    assert read[0] <= 2, read[0]  # the generator was started; the tail was never produced



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


# ---- a datetime is pinned when its entry is captured, before the public iterator that yielded it advances


class _SteerableZone(tzinfo):
    """A fixed offset that the test changes from outside (as an iterator can when it is resumed)."""

    def __init__(self, offset: timedelta) -> None:
        self.offset, self.reads = offset, 0

    def utcoffset(self, dt):
        self.reads += 1
        return self.offset

    def dst(self, dt):
        return timedelta(0)


class _GeneratedItems(tuple):
    """A tuple with nothing in its native backing: its items exist only through the generator ``make`` returns."""

    def __new__(cls, make):
        instance = super().__new__(cls, ())
        instance.make = make
        return instance

    def __iter__(self):
        return self.make()


def _bundle_whose_item_is_generated(observed_utc, verified_wall, zone, after_resume):
    request, bundle = _matching_pair()
    item = bundle.selected_items[0]
    checkpoint = _stored_checkpoint(bundle)
    dict.__setitem__(_STORED(checkpoint, BaseModel), "observed_at", observed_utc)
    dict.__setitem__(_STORED(checkpoint, BaseModel), "last_successful_verification_at", datetime(*verified_wall, tzinfo=zone))

    def make():
        yield item
        after_resume()  # the consumer asked for the next entry: the iterator changes what it just yielded

    dict.__setitem__(_STORED(bundle, BaseModel), "selected_items", _GeneratedItems(make))
    return request, bundle


# A verification must not be later than the observation it reports (the matching control: verification first).
@pytest.mark.parametrize(
    "verified, changed_to, expected",
    [
        ((2026, 9, 28, 6), timedelta(hours=-12), True),  # 06:00Z as yielded (before 12:00Z); -12 h would make it 18:00Z
        (_EVENING, timedelta(hours=12), False),  # 18:00Z as yielded (after 12:00Z); +12 h would make it 06:00Z
    ],
    ids=["valid-then-moved-after", "invalid-then-moved-before"],
)
def test_a_zone_changed_when_the_iterator_resumes_is_judged_as_it_was_yielded(verified, changed_to, expected) -> None:
    zone = _SteerableZone(timedelta(0))
    request, bundle = _bundle_whose_item_is_generated(
        datetime(*_NOON, tzinfo=UTC), verified, zone, lambda: setattr(zone, "offset", changed_to)
    )
    assert bundle_matches_reported_evidence(request, bundle) is expected


def test_an_honest_generated_item_with_a_steady_zone_still_matches() -> None:
    zone = _SteerableZone(timedelta(0))
    request, bundle = _bundle_whose_item_is_generated(datetime(*_NOON, tzinfo=UTC), (2026, 9, 28, 6), zone, lambda: None)
    assert bundle_matches_reported_evidence(request, bundle) is True
    assert zone.reads >= 1


def test_a_native_datetime_is_pinned_before_any_public_iterator_runs() -> None:
    from contracts_freshness import snapshot_pinned

    zone = _SteerableZone(timedelta(0))
    leaf = datetime(*_NOON, tzinfo=zone)

    def make():
        yield {"x": 1}
        zone.offset = timedelta(hours=12)  # a different view is resumed: the native datetime's zone changes

    snapshot = snapshot_pinned(({"leaf": leaf}, _GeneratedItems(make)))
    assert snapshot[0]["leaf"] == datetime(*_NOON, tzinfo=UTC)
    assert type(snapshot[0]["leaf"]) is datetime and type(snapshot[0]["leaf"].tzinfo) is timezone


def test_every_datetime_is_asked_once_and_an_unusable_zone_is_refused_at_capture() -> None:
    from contracts_freshness import snapshot_pinned

    zone = _SteerableZone(timedelta(hours=2))
    shared = datetime(*_NOON, tzinfo=zone)
    out = snapshot_pinned({"a": shared, "b": [shared]})
    assert zone.reads == 1 and out["a"] is out["b"][0]

    class NoOffset(tzinfo):
        def utcoffset(self, dt):
            return None

        def dst(self, dt):
            return None

    with pytest.raises(ValueError, match="usable UTC offset"):
        snapshot_pinned({"a": datetime(*_NOON, tzinfo=NoOffset())})
    naive = datetime(*_NOON)
    assert snapshot_pinned({"n": naive})["n"] is naive


def test_a_snapshot_without_a_pin_function_still_returns_the_callers_datetime() -> None:
    from contracts_common import snapshot_contract_graph

    leaf = datetime(*_NOON, tzinfo=UTC)
    assert snapshot_contract_graph({"a": leaf})["a"] is leaf


# ---- a datetime is never a mapping key: strict validation would format it (and its zone) in its error


class _ZoneThatCountsItsText(tzinfo):
    reprs = 0

    def utcoffset(self, dt):
        return timedelta(0)

    def dst(self, dt):
        return None

    def __repr__(self):
        _ZoneThatCountsItsText.reprs += 1
        raise RuntimeError("zone repr")

    __str__ = __repr__


def _entry_with_a_datetime_key():
    key = datetime(2026, 1, 1, tzinfo=_ZoneThatCountsItsText())
    return {"record_id": "r1", "content_revision": 1, "policy_epoch": 1, key: 1}


def test_a_snapshot_with_a_datetime_key_is_refused_before_validation_can_format_it() -> None:
    entry = _entry_with_a_datetime_key()
    _ZoneThatCountsItsText.reprs = 0
    with pytest.raises(ValueError, match="unsupported key type"):
        require_native_snapshot({"a": [entry]})
    assert _ZoneThatCountsItsText.reprs == 0


def test_a_datetime_keyed_source_entry_never_reaches_the_validator_error_that_formats_the_zone() -> None:
    from contracts_freshness import projection_status_for

    _ZoneThatCountsItsText.reprs = 0
    with pytest.raises(ValueError):
        projection_status_for((_entry_with_a_datetime_key(),), ())
    assert _ZoneThatCountsItsText.reprs == 0


def test_the_other_scalar_kinds_are_still_accepted_as_keys() -> None:
    snapshot = {"text": 1, 7: 2, 1.5: 3, True: 4, None: 5}
    assert require_native_snapshot(snapshot) is snapshot


# ---- labels: text is cut, and a kind with no label never falls back to its repr


def test_a_long_text_key_is_cut_in_the_label_and_a_short_one_is_not() -> None:
    from contracts_freshness import _label_text

    assert _label_text("k" * 500) == repr("k" * 40) + "..."
    assert _label_text("short") == "'short'"
    assert _label_text("k" * 40) == repr("k" * 40)  # exactly at the bound: not cut


def test_a_kind_with_no_label_is_a_fixed_word_whatever_its_repr_says() -> None:
    from contracts_freshness import _label_text

    class Loud:
        def __repr__(self):
            raise RuntimeError("repr ran")

        __str__ = __format__ = __repr__

    assert _label_text(Loud()) == "<key>"
    assert _label_text(datetime(2026, 1, 1, tzinfo=_ZoneThatCountsItsText())) == "<datetime key>"


def test_pin_datetime_keeps_returning_the_callers_own_object_for_an_unusable_zone_and_says_so() -> None:
    class NoOffset(tzinfo):
        def utcoffset(self, dt):
            return None

        def dst(self, dt):
            return None

    value = datetime(2026, 1, 1, tzinfo=NoOffset())
    assert pin_datetime(value) is value  # documented: the caller must check the result (see the docstring)
    assert "never cache" in (pin_datetime.__doc__ or "")


# ---- early entry validation: only homogeneous sequences, and a bounded per-class cache

def _holder_models():
    from contracts_common import ContractModel

    class Left(ContractModel):
        x: int

    class Right(ContractModel):
        y: str

    class Fixed(ContractModel):
        pair: tuple[Left, Right]

    class Open(ContractModel):
        many: tuple[Left, ...]

    class Listed(ContractModel):
        many: list[Left]

    return Left, Right, Fixed, Open, Listed


def _view(entries, read):
    def make():
        for entry in entries:
            read[0] += 1
            yield entry

    return _EmptyBackedGenerated(make)


def test_a_fixed_heterogeneous_tuple_is_read_whole_not_cut_at_its_second_entry() -> None:
    from contracts_common import snapshot_contract_graph

    _, _, Fixed, _, _ = _holder_models()
    read = [0]
    # a Left, a Right, then a third entry the contract does not allow: the whole view must reach validation
    holder = Fixed.model_construct(pair=_view([{"x": 1}, {"y": "b"}, {"x": 2}], read))
    snapshot = snapshot_contract_graph(holder)
    assert len(snapshot["pair"]) == 3 and read[0] == 3  # was 2: the Right entry failed the Left check and ended the read
    with pytest.raises(ValidationError):
        type(holder).model_validate(snapshot, strict=True)


def test_a_valid_fixed_tuple_still_validates_and_homogeneous_views_stop_early_and_refuse() -> None:
    from contracts_common import snapshot_contract_graph

    Left, Right, Fixed, Open, Listed = _holder_models()
    read = [0]
    good = Fixed.model_construct(pair=_view([{"x": 1}, {"y": "b"}], read))
    assert type(good).model_validate(snapshot_contract_graph(good), strict=True).pair[1].y == "b"
    for model, field in ((Open, "many"), (Listed, "many")):
        read = [0]
        bad = model.model_construct(**{field: _view([{"x": -1.5}] + [{"x": 1}] * 1_000, read)})
        with pytest.raises(ValueError, match="not valid"):  # the invalid first entry refuses the whole input
            snapshot_contract_graph(bad, contract=model)
        assert read[0] == 1  # and ends the read: the other 1,000 entries are never produced


def test_the_item_model_cache_does_not_grow_with_caller_created_classes() -> None:
    import gc
    import weakref

    from contracts_common import ContractModel, _item_models
    from pydantic import create_model

    references = []
    for index in range(400):
        model = create_model(f"Dynamic{index}", __base__=ContractModel, items=(tuple[int, ...], ()))
        _item_models(model)
        references.append(weakref.ref(model))
        del model
    gc.collect()
    assert sum(1 for reference in references if reference() is not None) <= 256


# ---- an entry exposed by a public view: captured whole before its datetimes are pinned; kept by position; guarded before validation


def _generated_item_view(items):
    return _GeneratedItems(lambda: iter(items))


def test_a_zone_callback_cannot_repair_a_later_part_of_the_entry_it_is_pinned_in() -> None:
    """The checkpoint's zone (captured first) repairs a negative revision in the same item's later projection."""

    request, bundle = _matching_pair()
    item = bundle.selected_items[0]
    applied = item.freshness.projection.applied_sources[0]
    good = applied.content_revision
    dict.__setitem__(_STORED(applied, BaseModel), "content_revision", -1)  # invalid as yielded

    class Repairing(tzinfo):
        def utcoffset(self, dt):
            dict.__setitem__(_STORED(applied, BaseModel), "content_revision", good)
            return timedelta(0)

        def dst(self, dt):
            return None

    checkpoint = _stored_checkpoint(bundle)
    dict.__setitem__(_STORED(checkpoint, BaseModel), "last_successful_verification_at", datetime(2026, 9, 28, 6, tzinfo=Repairing()))
    dict.__setitem__(_STORED(bundle, BaseModel), "selected_items", _generated_item_view([item]))
    assert bundle_matches_reported_evidence(request, bundle) is False


def test_the_same_item_yielded_twice_is_snapshotted_at_each_position() -> None:
    """The item's public views report one revision the first time it is yielded and another the second time (each
    entry consistent in itself): the two positions disagree about source-a, whichever order they come in."""

    from contracts_references import SourceVersion

    class Occurrence:
        number = 0

    def views(revisions):
        class ByOccurrence(tuple):
            def __iter__(self):
                revision = revisions[min(Occurrence.number, len(revisions) - 1)]
                return iter([{"record_id": "source-a", "content_revision": revision, "policy_epoch": 2}])

        return ByOccurrence

    def run(revisions) -> bool:
        request, bundle = _matching_pair()
        item = bundle.selected_items[0]
        proto = (SourceVersion(record_id="source-a", content_revision=4, policy_epoch=2),)
        kind = views(revisions)
        dict.__setitem__(_STORED(item, BaseModel), "source_versions", kind(proto))
        dict.__setitem__(_STORED(item.freshness.projection, BaseModel), "applied_sources", kind(proto))

        class Twice(tuple):
            def __new__(cls):
                return super().__new__(cls, ())

            def __iter__(self):
                Occurrence.number = 0
                yield item
                Occurrence.number = 1
                yield item

        dict.__setitem__(_STORED(bundle, BaseModel), "selected_items", Twice())
        return bundle_matches_reported_evidence(request, bundle)

    assert run([4, 4]) is True  # the same entry twice agrees with itself
    assert run([4, 5]) is False
    assert run([5, 4]) is False


def test_the_same_pure_item_yielded_twice_still_matches() -> None:
    request, bundle = _matching_pair()
    item = bundle.selected_items[0]
    dict.__setitem__(_STORED(bundle, BaseModel), "selected_items", _generated_item_view([item, item]))
    assert bundle_matches_reported_evidence(request, bundle) is True


def test_a_proxy_yielded_by_a_public_view_is_refused_before_pydantic_reads_its_class() -> None:
    from contracts_context import receipt_matches_bundle
    from contracts_references import SourceVersion

    reads = {"class": 0}

    class Proxy:
        def __init__(self, target):
            object.__setattr__(self, "_target", target)

        def __getattribute__(self, name):
            if name == "__class__":
                reads["class"] += 1
                return SourceVersion
            return getattr(object.__getattribute__(self, "_target"), name)

    def make():
        yield Proxy(SourceVersion(record_id="source-a", content_revision=4, policy_epoch=2))

    receipt, bundle = _receipt_and_bundle_with_source_versions(make)
    assert receipt_matches_bundle(receipt, bundle) is False
    assert reads["class"] == 0, "pydantic read the proxy's __class__ before the supported-type guard refused it"


def test_a_noncanonical_enum_yielded_by_a_public_view_is_refused_before_validation() -> None:
    from contracts_context import receipt_matches_bundle

    from contracts_freshness import ProjectionStatus

    fake = str.__new__(ProjectionStatus, "current")

    def make():
        yield {"record_id": "source-a", "content_revision": 4, "policy_epoch": 2, "extra": fake}

    receipt, bundle = _receipt_and_bundle_with_source_versions(make)
    assert receipt_matches_bundle(receipt, bundle) is False


# ---- an early refusal of a yielded entry refuses the whole input, and early validation uses the contract the helper uses


def _strict_receipt_view(*entries):
    """A ``ContextAssemblyReceipt`` subclass instance whose ``source_versions`` view yields ``entries`` and that
    requires a stricter ``SourceVersion`` (revision at least 5) than the base contract the helper validates."""

    from typing import Annotated

    from pydantic import Field

    from contracts_context import ContextAssemblyReceipt
    from contracts_references import SourceVersion
    from tests.test_contracts_context import _receipt

    class StrictVersion(SourceVersion):
        content_revision: Annotated[int, Field(strict=True, ge=5)]

    class StrictReceipt(ContextAssemblyReceipt):
        source_versions: tuple[StrictVersion, ...]

    base = _validate(ContextAssemblyReceipt, _receipt())
    values = dict(_STORED(base, BaseModel))
    values["source_versions"] = _GeneratedItems(lambda: iter(list(entries)))
    receipt = StrictReceipt.model_construct(**values)
    return receipt


def test_a_stricter_subclass_does_not_truncate_the_snapshot_the_base_contract_sees() -> None:
    from contracts_context import receipt_matches_bundle

    bundle = _validate(ContextBundle, _bundle())  # its only source is source-a at revision 4
    extra = {"record_id": "source-b", "content_revision": 1, "policy_epoch": 1}
    first = {"record_id": "source-a", "content_revision": 4, "policy_epoch": 2}
    receipt = _strict_receipt_view(first, extra)
    # The base contract accepts revision 4, so the receipt lists source-a AND source-b: it is not the bundle's evidence.
    assert receipt_matches_bundle(receipt, bundle) is False


def test_the_same_receipt_without_the_extra_source_still_matches_under_the_base_contract() -> None:
    from contracts_context import receipt_matches_bundle

    bundle = _validate(ContextBundle, _bundle())
    receipt = _strict_receipt_view({"record_id": "source-a", "content_revision": 4, "policy_epoch": 2})
    assert receipt_matches_bundle(receipt, bundle) is True  # the subclass's stricter rule is not the helper's contract


def test_an_entry_refused_early_refuses_the_input_instead_of_returning_a_prefix() -> None:
    from contracts_context import receipt_matches_bundle

    bundle = _validate(ContextBundle, _bundle())
    bad = {"record_id": "source-a", "content_revision": -1, "policy_epoch": 2}
    later = {"record_id": "source-a", "content_revision": 4, "policy_epoch": 2}
    receipt, _ = _receipt_and_bundle_with_source_versions(lambda: iter([bad, later]))
    assert receipt_matches_bundle(receipt, bundle) is False
    from contracts_common import snapshot_contract_graph
    from contracts_context import ContextAssemblyReceipt

    with pytest.raises(ValueError):
        snapshot_contract_graph(receipt, contract=ContextAssemblyReceipt)


def test_the_declared_contract_is_used_all_the_way_down_not_a_narrower_nested_class() -> None:
    """A nested model that is an instance of a stricter subclass is still judged by what the root declares."""

    from typing import Annotated

    from pydantic import Field

    from contracts_references import SourceVersion

    request, bundle = _matching_pair()
    item = bundle.selected_items[0]
    projection = item.freshness.projection

    class StrictVersion(SourceVersion):
        content_revision: Annotated[int, Field(strict=True, ge=5)]

    class StrictProjection(type(projection)):
        applied_sources: tuple[StrictVersion, ...]

    values = dict(_STORED(projection, BaseModel))
    values["applied_sources"] = _GeneratedItems(lambda: iter([{"record_id": "source-a", "content_revision": 4, "policy_epoch": 2}]))
    dict.__setitem__(_STORED(item.freshness, BaseModel), "projection", StrictProjection.model_construct(**values))
    # the base contract accepts revision 4; the narrower nested class does not, and it is not the helper's contract
    assert bundle_matches_reported_evidence(request, bundle) is True


# ---- eager validation never runs a caller's class on the retained snapshot, and every snapshot call site passes the guard


def _receipt_whose_version_class_rewrites_what_it_validates():
    """A ``ContextAssemblyReceipt`` subclass redeclaring ``source_versions`` with a ``SourceVersion`` subclass whose
    ``model_validate`` repairs the snapshot it is handed (and counts the calls)."""

    from typing import Annotated

    from contracts_context import ContextAssemblyReceipt
    from contracts_references import SourceVersion
    from tests.test_contracts_context import _receipt

    calls = {"n": 0}

    class Repairing(SourceVersion):
        @classmethod
        def model_validate(cls, obj, *args, **kwargs):
            calls["n"] += 1
            if isinstance(obj, dict) and obj.get("content_revision") == -1:
                obj["content_revision"] = 4  # the bundle's revision: repairs the retained snapshot in place
            return super().model_validate(obj, *args, **kwargs)

    class RepairingReceipt(ContextAssemblyReceipt):
        source_versions: tuple[Repairing, ...]

    base = _validate(ContextAssemblyReceipt, _receipt())
    values = dict(_STORED(base, BaseModel))
    values["source_versions"] = _GeneratedItems(lambda: iter([{"record_id": "source-a", "content_revision": -1, "policy_epoch": 2}]))
    return RepairingReceipt.model_construct(**values), calls


def test_a_callers_subclass_cannot_repair_the_snapshot_through_its_own_eager_validator() -> None:
    from contracts_context import ContextAssemblyReceipt, receipt_matches_bundle

    receipt, calls = _receipt_whose_version_class_rewrites_what_it_validates()
    bundle = _validate(ContextBundle, _bundle())
    assert receipt_matches_bundle(receipt, bundle) is False  # revision -1 was captured; nothing repaired it
    assert calls["n"] == 0  # the caller's class never ran
    from contracts_common import snapshot_contract_graph

    with pytest.raises(ValueError, match="not valid"):  # refused by the trusted contract, as yielded (-1)
        snapshot_contract_graph(receipt, contract=ContextAssemblyReceipt)
    assert calls["n"] == 0


def test_without_a_declared_contract_no_class_of_the_caller_runs_during_the_snapshot() -> None:
    from contracts_common import snapshot_contract_graph

    receipt, calls = _receipt_whose_version_class_rewrites_what_it_validates()
    snapshot = snapshot_contract_graph(receipt)  # no trusted contract: no early validation at all
    assert [entry["content_revision"] for entry in snapshot["source_versions"]] == [-1]
    assert calls["n"] == 0


def test_the_trusted_contracts_own_validator_works_on_a_private_copy() -> None:
    from typing import Annotated

    from pydantic import model_validator

    from contracts_common import ContractModel, snapshot_contract_graph
    from contracts_references import SourceVersion

    class Mutating(SourceVersion):
        @model_validator(mode="before")
        @classmethod
        def rewrite(cls, data):
            if isinstance(data, dict):
                data["content_revision"] = 4  # a validator that edits its input in place
            return data

    class Holder(ContractModel):
        versions: tuple[Mutating, ...]

    holder = Holder.model_construct(versions=_GeneratedItems(lambda: iter([{"record_id": "s", "content_revision": 1, "policy_epoch": 1}])))
    snapshot = snapshot_contract_graph(holder, contract=Holder)
    assert [entry["content_revision"] for entry in snapshot["versions"]] == [1]  # the retained snapshot is untouched


def _unchecked_source_version_with_a_proxy_in_a_nested_view(reads):
    from contracts_common import ContractModel
    from contracts_references import SourceVersion

    class Proxy:
        def __getattribute__(self, name):
            if name == "__class__":
                reads["class"] += 1
                return SourceVersion
            raise AttributeError(name)

    class Nested(ContractModel):
        many: tuple[SourceVersion, ...]

    nested = Nested.model_construct(many=_GeneratedItems(lambda: iter([Proxy()])))
    version = SourceVersion.model_construct(record_id="r1", content_revision=1, policy_epoch=1)
    dict.__setitem__(_STORED(version, BaseModel), "nested", nested)
    return version


@pytest.mark.parametrize("path", ["exact_sequence", "lazy_sequence"])
def test_a_proxy_in_a_nested_view_is_refused_before_pydantic_reads_its_class_on_both_dependency_paths(path) -> None:
    from contracts_freshness import projection_status_for

    reads = {"class": 0}
    version = _unchecked_source_version_with_a_proxy_in_a_nested_view(reads)
    sequence = (version,) if path == "exact_sequence" else _GeneratedItems(lambda: iter([version]))
    with pytest.raises(ValueError):
        projection_status_for(sequence, ())
    assert reads["class"] == 0


def test_every_snapshot_call_site_in_the_service_passes_the_entry_guard() -> None:
    """A call to ``snapshot_contract_graph`` outside tests must give the receiving boundary's guard."""

    import ast
    from pathlib import Path

    service = Path(__file__).resolve().parents[1]
    offenders = []
    for path in sorted(service.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, "id", getattr(node.func, "attr", None)) == "snapshot_contract_graph":
                if not any(keyword.arg == "entry_guard" for keyword in node.keywords):
                    offenders.append(f"{path.name}:{node.lineno}")
    assert offenders == []
