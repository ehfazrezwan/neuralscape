"""Deterministic, product-independent records for reproducible benchmark runs.

The models in this module describe evidence about a run.  They do not confer
authorization and they make no claim that a workload, measurement, or resource
limit is suitable for production.  An absent measurement is unknown; it must
not be synthesized as a zero.
"""

from __future__ import annotations

import heapq
import json
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from types import MappingProxyType
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Annotated, Any, Literal, NamedTuple, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationInfo,
    field_validator,
    model_validator,
)
from pydantic_core import core_schema


_MAX_SAFE_COUNTER = 9_007_199_254_740_991
_OpaqueId = Annotated[
    str,
    StringConstraints(strict=True, min_length=1, max_length=256),
]
_SafeCounter = Annotated[
    int,
    Field(strict=True, ge=0, le=_MAX_SAFE_COUNTER),
]
_BASE_MODEL_DICT_DESCRIPTOR = vars(BaseModel)["__dict__"]
_BASE_MODEL_EXTRA_DESCRIPTOR = vars(BaseModel)["__pydantic_extra__"]

_FrozenModelStorage = tuple[
    BaseModel,
    tuple[tuple[Any, Any], ...],
    Any,
    tuple[tuple[Any, Any], ...] | None,
]


class _FrozenExtraInventory(NamedTuple):
    owner: Mapping[Any, Any]
    iterated_names: tuple[Any, ...]
    item_entries: tuple[Any, ...]
    failure: Exception | None


class _FrozenPublicMapping(NamedTuple):
    owner: Mapping[Any, Any]
    entries: tuple[tuple[Any, Any], ...]
    failure: Exception | None
    complete: bool


class _FrozenPublicSequence(NamedTuple):
    owner: object
    entries: tuple[Any, ...]
    failure: Exception | None
    complete: bool


class _FrozenGraph(NamedTuple):
    models: dict[int, _FrozenModelStorage]
    native_containers: dict[int, tuple[object, tuple[Any, ...]]]
    replay_entries: dict[int, tuple[object, tuple[Any, ...]]]
    completed_traversals: dict[int, object]
    # id -> containers on the longest path below a completed subtree (itself included): a later wrapper
    # reaching it is not walked again but is still depth-checked where it is reached.
    completed_heights: dict[int, int]
    extra_mapping_owners: dict[int, Mapping[Any, Any]]
    extra_mapping_queue: list[Mapping[Any, Any]]
    extra_mapping_cursor: list[int]
    extra_inventories: dict[int, _FrozenExtraInventory]
    public_mapping_owners: dict[int, Mapping[Any, Any]]
    public_mapping_queue: list[Mapping[Any, Any]]
    public_mappings: dict[int, _FrozenPublicMapping]
    completed_mapping_discoveries: dict[int, tuple[object, bool]]
    public_sequence_owners: dict[int, object]
    public_sequence_queue: list[object]
    public_sequence_cursor: list[int]
    public_sequences: dict[int, _FrozenPublicSequence]
    visits: list[int]
    # Entries of native containers charged before they are copied (see ``_charge_native``): its own
    # counter, so the pre-copy check does not spend the visit budget the snapshot uses for the same nodes.
    native_entries: list[int]
    # id -> (the caller's datetime, its fixed-offset copy): one offset per object per call.  The
    # original is retained so its id cannot be reused by another object during the call.
    fixed_datetimes: dict[int, tuple[datetime, datetime]]


class _ManifestContract(BaseModel):
    """Strict, immutable base local to the standalone benchmark package."""

    # ``revalidate_instances`` is defense in depth: a model instance that ever reached
    # validation would have its fields checked again instead of being trusted.
    model_config = ConfigDict(
        extra="forbid", strict=True, frozen=True, revalidate_instances="always"
    )
    schema_version: Literal["candidate-v1"]


class RunState(str, Enum):
    """Lifecycle state of a run record."""

    PLANNED = "planned"
    COMPLETED = "completed"
    ABORTED = "aborted"


class CacheCondition(str, Enum):
    """Declared cache/startup condition; mixed runs must be split."""

    COLD = "cold"
    WARM = "warm"


class LimitKind(str, Enum):
    """Whether a resource was capped, explicitly uncapped, or unknown."""

    BOUNDED = "bounded"
    UNBOUNDED = "unbounded"
    UNKNOWN = "unknown"


class ObservationKind(str, Enum):
    """Availability of an observed resource value."""

    PENDING = "pending"
    MEASURED = "measured"
    UNKNOWN = "unknown"


class BuildIdentity(_ManifestContract):
    """Immutable source and image identities used by a run."""

    git_commit: str = Field(pattern=r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
    image_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    dependency_lock_sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")


class WorkloadIdentity(_ManifestContract):
    """Exact workload definition and input corpus bytes."""

    workload_id: _OpaqueId
    workload_sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    corpus_sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")


class ConcurrencySetting(_ManifestContract):
    """One independently named concurrency control."""

    scope: _OpaqueId
    value: _SafeCounter

    @field_validator("value")
    @classmethod
    def require_positive_value(cls, value: int) -> int:
        if value < 1:
            raise ValueError("concurrency values must be at least one; omit inactive controls")
        return value


class ResourceReading(_ManifestContract):
    """A comparable resource limit and peak observation in one declared unit.

    ``observed_value`` is present only for a measured observation.  ``unknown``
    explicitly preserves a missing measurement instead of turning it into zero.
    """

    scope: _OpaqueId
    resource: _OpaqueId
    unit: _OpaqueId
    limit_kind: LimitKind
    limit_value: _SafeCounter | None = None
    observation_kind: ObservationKind
    observed_value: _SafeCounter | None = None
    unknown_reason: _OpaqueId | None = None

    @model_validator(mode="after")
    def validate_availability(self) -> Self:
        if self.limit_kind is LimitKind.BOUNDED:
            if self.limit_value is None or self.limit_value < 1:
                raise ValueError("a bounded resource requires a positive limit_value")
        elif self.limit_value is not None:
            raise ValueError("limit_value is valid only when limit_kind is bounded")

        if self.observation_kind is ObservationKind.MEASURED:
            if self.observed_value is None:
                raise ValueError("a measured resource requires observed_value")
            if self.unknown_reason is not None:
                raise ValueError("a measured resource cannot have unknown_reason")
        elif self.observed_value is not None:
            raise ValueError("observed_value is valid only for a measured resource")
        elif self.observation_kind is ObservationKind.UNKNOWN:
            if self.unknown_reason is None:
                raise ValueError("an unknown observation requires unknown_reason")
        elif self.unknown_reason is not None:
            raise ValueError("unknown_reason is valid only for an unknown observation")
        return self


class MeasurementProvenance(_ManifestContract):
    """How and where a set of samples was collected."""

    collector: _OpaqueId
    collector_version: _OpaqueId
    clock: Literal["monotonic", "monotonic_raw", "perf_counter"]
    placement: _OpaqueId


class SampleExclusion(_ManifestContract):
    """Samples deliberately omitted for a bounded, non-outcome reason.

    Errors and timeouts are deliberately not exclusion kinds: they remain part
    of the workload outcome accounting.
    """

    kind: Literal["warmup", "fixture_setup", "invalid_instrumentation"]
    count: _SafeCounter
    reason: _OpaqueId

    @field_validator("count")
    @classmethod
    def require_positive_count(cls, value: int) -> int:
        if value < 1:
            raise ValueError("an exclusion must remove at least one attempted sample")
        return value


class FailureOutcome(_ManifestContract):
    """Attempt outcomes that failed the workload rather than measurement setup."""

    kind: Literal["timeout", "error"]
    count: _SafeCounter

    @field_validator("count")
    @classmethod
    def require_positive_count(cls, value: int) -> int:
        if value < 1:
            raise ValueError("a failure outcome must account for at least one attempt")
        return value


class TimingMeasurement(_ManifestContract):
    """Sampling boundary for one timing metric, without benchmark results."""

    name: _OpaqueId
    time_unit: Literal["nanoseconds", "microseconds", "milliseconds", "seconds"]
    attempted_sample_count: _SafeCounter
    included_sample_count: _SafeCounter
    provenance: MeasurementProvenance
    exclusions: tuple[SampleExclusion, ...] = ()
    failure_outcomes: tuple[FailureOutcome, ...] = ()

    @model_validator(mode="after")
    def validate_sample_accounting(self) -> Self:
        if self.attempted_sample_count < 1:
            raise ValueError("attempted_sample_count must be at least one")
        excluded = sum(item.count for item in self.exclusions)
        failed = sum(item.count for item in self.failure_outcomes)
        if self.included_sample_count + failed + excluded != self.attempted_sample_count:
            raise ValueError(
                "included samples, failure outcomes, and exclusions must equal "
                "attempted samples"
            )
        exclusion_kinds = [item.kind for item in self.exclusions]
        if len(exclusion_kinds) != len(set(exclusion_kinds)):
            raise ValueError("exclusion kinds must be unique within a measurement")
        failure_kinds = [item.kind for item in self.failure_outcomes]
        if len(failure_kinds) != len(set(failure_kinds)):
            raise ValueError("failure outcome kinds must be unique within a measurement")
        return self


def _exact_offset(offset: object) -> timedelta:
    """An exact ``timedelta`` with the value of ``offset``, read through the built-in slots.

    A ``tzinfo`` may return a ``timedelta`` subclass whose arithmetic or attributes run caller code;
    the three fields are read with the base type's own accessors and a fresh exact value is built.
    """

    return timedelta(
        days=timedelta.days.__get__(offset),  # type: ignore[arg-type]
        seconds=timedelta.seconds.__get__(offset),  # type: ignore[arg-type]
        microseconds=timedelta.microseconds.__get__(offset),  # type: ignore[arg-type]
    )


def _exact_datetime(value: datetime, zone: timezone) -> datetime:
    """An exact ``datetime`` with the wall time and ``fold`` of ``value`` under ``zone``.

    ``datetime.replace`` builds an instance of the caller's subclass (running its ``__new__``), and a
    kept subclass would run caller arithmetic later; the fields are read with the base type's own
    accessors and a fresh exact value is built.
    """

    return datetime(
        datetime.year.__get__(value),  # type: ignore[attr-defined]
        datetime.month.__get__(value),  # type: ignore[attr-defined]
        datetime.day.__get__(value),  # type: ignore[attr-defined]
        datetime.hour.__get__(value),  # type: ignore[attr-defined]
        datetime.minute.__get__(value),  # type: ignore[attr-defined]
        datetime.second.__get__(value),  # type: ignore[attr-defined]
        datetime.microsecond.__get__(value),  # type: ignore[attr-defined]
        tzinfo=zone,
        fold=datetime.fold.__get__(value),  # type: ignore[attr-defined]
    )


def _instant_before(left: datetime, right: datetime) -> bool:
    """Whether ``left`` is an earlier instant than ``right``, without leaving the datetime range.

    Converting to UTC raises ``OverflowError`` at the ends of the range.  The difference of the two
    naive wall times and the difference of the two offsets cannot overflow, and ``fold`` is honoured
    because each offset is the datetime's own.
    """

    offsets = (datetime.utcoffset(left), datetime.utcoffset(right))
    if offsets[0] is None or offsets[1] is None:
        raise ValueError("run timestamps must include an explicit UTC offset")
    left_offset, right_offset = _exact_offset(offsets[0]), _exact_offset(offsets[1])
    # Built-in arithmetic only: no operand's own __sub__ or __rsub__ can run.
    wall = datetime.__sub__(datetime.replace(left, tzinfo=None), datetime.replace(right, tzinfo=None))
    return timedelta.__sub__(wall, timedelta.__sub__(left_offset, right_offset)) < timedelta(0)


# id -> (the caller's datetime, its fixed-offset copy) for the manifest validation in progress.  One
# offset is read per object per call, so the same datetime supplied as both timestamps cannot be
# given two different readings.  The original is retained so its id cannot be reused mid-call.
_TIMESTAMP_PINS: ContextVar[dict[int, tuple[datetime, datetime]] | None] = ContextVar(
    "_TIMESTAMP_PINS", default=None
)


# The started_at reading of the manifest whose fields are being validated, when no per-call pins are
# open (a TypeAdapter, or a model that nests RunManifest, does not run RunManifest's own __init__ or
# model_validate).  Fields are validated in declaration order, so finished_at of the same manifest
# reuses the reading if it is the same datetime object; the slot is cleared once finished_at is done
# (both fields validate their defaults, and a before-validator on finished_at empties it whether or not
# the value is valid, so a rejected validation leaves nothing behind).
_PENDING_STARTED_AT: ContextVar[tuple[datetime, datetime] | None] = ContextVar(
    "_PENDING_STARTED_AT", default=None
)


@contextmanager
def _pinned_timestamp_offsets() -> Iterator[None]:
    """Open the per-call offset pins for one manifest validation (nested calls get their own).

    A model-level validator cannot do this: it would hand JSON input to the strict field checks as
    Python objects.  The two entry points that take Python objects open the scope instead.
    """

    token = _TIMESTAMP_PINS.set({})
    try:
        yield
    finally:
        _TIMESTAMP_PINS.reset(token)


def _take_started_reading(value: Any, handler: Any) -> Any:
    """Check ``finished_at`` after taking, and in every case emptying, the started_at reading held for it.

    The slot is emptied before the value is checked and again afterwards, so it is empty whether the
    value is valid or not: a rejected validation leaves nothing behind.  The same object supplied as
    both timestamps is handed on as the reading already taken.  (A function validator on the whole
    field would make JSON input reach the strict checks as Python objects; see ``_PairedFinish``.)
    """

    held = _PENDING_STARTED_AT.get()
    _PENDING_STARTED_AT.set(None)
    try:
        if held is not None and _TIMESTAMP_PINS.get() is None and held[0] is value:
            value = held[1]
        return handler(value)
    finally:
        _PENDING_STARTED_AT.set(None)


class _PairedFinish:
    """Annotation for ``finished_at``: the pairing step applies to Python input only.

    JSON input keeps the field's own schema (each value is a new object, so there is nothing to pair)
    and so stays on the strict JSON path.
    """

    @classmethod
    def __get_pydantic_core_schema__(cls, source: Any, handler: Any) -> Any:
        inner = handler(source)
        return core_schema.json_or_python_schema(
            json_schema=inner,
            python_schema=core_schema.no_info_wrap_validator_function(_take_started_reading, inner),
        )


class RunManifest(_ManifestContract):
    """A reproducible benchmark run envelope with explicit evidence state."""

    run_id: _OpaqueId
    state: RunState
    build: BuildIdentity
    workload: WorkloadIdentity
    cache_condition: CacheCondition
    concurrency: tuple[ConcurrencySetting, ...]
    resources: tuple[ResourceReading, ...]
    # Validated even when omitted, so the started_at/finished_at pairing slot is always cleared.
    started_at: datetime | None = Field(default=None, validate_default=True)
    finished_at: Annotated[datetime | None, _PairedFinish] = Field(default=None, validate_default=True)
    measurements: tuple[TimingMeasurement, ...] = ()

    def __init__(self, /, **data: Any) -> None:
        with _pinned_timestamp_offsets():
            super().__init__(**data)

    # An ``__init__`` that is not marked as pass-through makes pydantic route JSON input through it
    # as Python objects, which would defeat the strict JSON checks.
    __init__.__pydantic_base_init__ = True  # type: ignore[attr-defined]

    @classmethod
    def model_validate(cls, obj: Any, *args: Any, **kwargs: Any) -> Self:
        with _pinned_timestamp_offsets():
            return super().model_validate(obj, *args, **kwargs)

    @field_validator("started_at", "finished_at")
    @classmethod
    def require_aware_timestamp(cls, value: datetime | None, info: ValidationInfo) -> datetime | None:
        pins = _TIMESTAMP_PINS.get()
        if pins is None and info.field_name == "started_at":
            _PENDING_STARTED_AT.set(None)  # nothing from an earlier validation is ever paired with this one
        if value is None:
            return None
        # Read the caller's tzinfo once and keep an exact fixed-offset copy, so nothing later (the
        # ordering check, serialization) asks caller code again or sees a different answer.
        pinned = pins.get(id(value)) if pins is not None else None
        if pinned is not None and pinned[0] is value:
            return pinned[1]
        try:
            offset = datetime.utcoffset(value)
            if offset is None:
                raise ValueError("run timestamps must include an explicit UTC offset")
            fixed = _exact_datetime(value, timezone(_exact_offset(offset)))
            if pins is not None:
                pins[id(value)] = (value, fixed)
            elif info.field_name == "started_at":
                _PENDING_STARTED_AT.set((value, fixed))
            return fixed
        except ValueError:
            raise
        except Exception:  # a tzinfo is caller code: it can raise, or return something that is no offset
            raise ValueError("run timestamps must have a usable UTC offset") from None

    @model_validator(mode="after")
    def validate_run_evidence(self) -> Self:
        if not self.concurrency:
            raise ValueError("at least one concurrency setting is required")
        concurrency_scopes = [item.scope for item in self.concurrency]
        if len(concurrency_scopes) != len(set(concurrency_scopes)):
            raise ValueError("concurrency scopes must be unique")

        if not self.resources:
            raise ValueError("at least one resource reading is required")
        resource_keys = [(item.scope, item.resource) for item in self.resources]
        if len(resource_keys) != len(set(resource_keys)):
            raise ValueError("resource scope/name pairs must be unique")
        measurement_names = [item.name for item in self.measurements]
        if len(measurement_names) != len(set(measurement_names)):
            raise ValueError("measurement names must be unique")

        if self.finished_at is not None and self.started_at is not None:
            # Same-zone datetimes compare by wall time and ignore fold, so
            # order the instants themselves.
            if _instant_before(self.finished_at, self.started_at):
                raise ValueError("finished_at cannot precede started_at")

        if self.state is RunState.PLANNED:
            if self.started_at is not None or self.finished_at is not None:
                raise ValueError("a planned run cannot contain execution timestamps")
            if self.measurements:
                raise ValueError("a planned run cannot contain measurements")
            if any(
                item.observation_kind is not ObservationKind.PENDING
                for item in self.resources
            ):
                raise ValueError("planned resource observations must be pending")
        else:
            if self.started_at is None or self.finished_at is None:
                raise ValueError("a finished run requires started_at and finished_at")
            if any(
                item.observation_kind is ObservationKind.PENDING
                for item in self.resources
            ):
                raise ValueError("a finished run cannot retain pending observations")
            if self.state is RunState.COMPLETED and not self.measurements:
                raise ValueError("a completed run requires at least one measurement")

        if self.cache_condition is CacheCondition.COLD:
            if any(
                exclusion.kind == "warmup"
                for measurement in self.measurements
                for exclusion in measurement.exclusions
            ):
                raise ValueError("cold runs cannot exclude warmup samples")
        return self


def _retain_public_mapping(value: Any, frozen_graph: _FrozenGraph) -> bool:
    """Retain a strong owner for a supported nonexact public mapping."""

    if (
        _is_instance(value, BaseModel)
        or type(value) is dict
        or not _is_instance(value, Mapping)
    ):
        return False
    identity = id(value)
    owner = frozen_graph.public_mapping_owners.get(identity)
    if owner is value:
        return False
    frozen_graph.public_mapping_owners[identity] = value
    frozen_graph.public_mapping_queue.append(value)
    frozen_graph.public_mappings.pop(identity, None)
    return True


_TYPE_MRO = vars(type)["__mro__"]
_MAPPING_BASES = (Mapping, dict, MappingProxyType)


def _is_subclass(cls: type, types: type | tuple[type, ...]) -> bool:
    """Whether ``cls`` inherits from one of ``types``, decided by identity on the native MRO.

    ``issubclass`` against an ABC (``Mapping``) or pydantic's ABC-based metaclass
    (``BaseModel``) runs ``ABCMeta.__subclasscheck__``: registry lookups that hash and
    compare ``cls`` and honour ``__subclasshook__``, so a class with a hostile metaclass
    ``__hash__``/``__eq__``/``__subclasscheck__`` could run code or be matched by
    spoofing.  This reads the class's real MRO through the base ``type`` descriptor and
    compares with ``is``.  ``Mapping`` also recognises ``dict`` and ``mappingproxy``
    bases explicitly; a class only *registered* as a mapping is not treated as one.
    """

    wanted: list[type] = []
    for candidate in types if type(types) is tuple else (types,):
        if candidate is Mapping:
            wanted.extend(_MAPPING_BASES)
        else:
            wanted.append(candidate)
    for base in _TYPE_MRO.__get__(cls):
        for candidate in wanted:
            if base is candidate:
                return True
    return False


def _is_instance(value: Any, types: type | tuple[type, ...]) -> bool:
    """``isinstance`` that runs no caller-controlled code: no ``__class__``, no ABC registry.

    ``isinstance`` consults ``value.__class__`` (a hostile container can make that a
    callback that runs before its contents are captured), and ABC or pydantic metaclass
    checks run ``__subclasscheck__`` machinery.  ``type(value)`` is not overridable and
    :func:`_is_subclass` compares native MRO entries by identity.
    """

    return _is_subclass(type(value), types)


def _is_exact_type(value_type: type, exact: tuple[type, ...]) -> bool:
    """Whether ``value_type`` *is* one of ``exact``, by identity.

    ``value_type in (a, b)`` compares with ``==``, which for a class runs its
    metaclass ``__eq__``: hostile caller code that could report itself as exact.
    """

    return any(value_type is candidate for candidate in exact)


def _retain_public_sequence(value: Any, frozen_graph: _FrozenGraph) -> bool:
    """Retain a strong owner for one nonexact public sequence.

    A tuple/list subclass that is also a Mapping is observed as a mapping, the
    same way every other dispatcher treats it, so its sequence iterator is not
    consulted before its mapping items.
    """

    if (
        _is_instance(value, (BaseModel, Mapping))
        or _is_exact_type(type(value), (tuple, list))
        or not _is_instance(value, (tuple, list))
    ):
        return False
    identity = id(value)
    owner = frozen_graph.public_sequence_owners.get(identity)
    if owner is value:
        return False
    frozen_graph.public_sequence_owners[identity] = value
    frozen_graph.public_sequence_queue.append(value)
    frozen_graph.public_sequences.pop(identity, None)
    return True


def _retain_extra_mapping(
    value: Any,
    frozen_graph: _FrozenGraph,
) -> bool:
    """Retain one supported non-dict model-extra mapping owner."""

    if (
        value is None
        or _is_subclass(type(value), dict)
        or not _is_instance(value, Mapping)
    ):
        return False
    identity = id(value)
    owner = frozen_graph.extra_mapping_owners.get(identity)
    if owner is value:
        return False
    frozen_graph.extra_mapping_owners[identity] = value
    frozen_graph.extra_mapping_queue.append(value)
    frozen_graph.extra_inventories.pop(identity, None)
    return True


def _inventory_extra_mapping(
    value: Mapping[Any, Any],
    frozen_graph: _FrozenGraph,
) -> _FrozenExtraInventory:
    """Retain one model-extra mapping's key and item views."""

    identity = id(value)
    frozen = frozen_graph.extra_inventories.get(identity)
    if frozen is not None and frozen.owner is value:
        return frozen

    iterated_names: list[Any] = []
    item_entries: list[Any] = []
    failure: Exception | None = None
    try:
        name_iterator = iter(value)
    except Exception as error:
        failure = error
    else:
        while True:
            try:
                name = next(name_iterator)
            except StopIteration:
                break
            except Exception as error:
                failure = error
                break
            _charge_capture(frozen_graph)
            iterated_names.append(name)

    if failure is None:
        try:
            item_iterator = iter(value.items())
        except Exception as error:
            failure = error
        else:
            while True:
                try:
                    entry = next(item_iterator)
                except StopIteration:
                    break
                except Exception as error:
                    failure = error
                    break
                _charge_capture(frozen_graph)
                item_entries.append(entry)

    frozen = _FrozenExtraInventory(
        owner=value,
        iterated_names=tuple(iterated_names),
        item_entries=tuple(item_entries),
        failure=failure,
    )
    frozen_graph.extra_inventories[identity] = frozen
    return frozen


def _inventory_retained_extra_mappings(frozen_graph: _FrozenGraph) -> None:
    """Observe each retained model-extra mapping before public traversal."""

    while frozen_graph.extra_mapping_cursor[0] < len(
        frozen_graph.extra_mapping_queue
    ):
        cursor = frozen_graph.extra_mapping_cursor[0]
        owner = frozen_graph.extra_mapping_queue[cursor]
        frozen_graph.extra_mapping_cursor[0] = cursor + 1
        if frozen_graph.extra_mapping_owners.get(id(owner)) is not owner:
            continue
        _inventory_extra_mapping(owner, frozen_graph)


def _freeze_model_storage(
    value: Any,
    frozen_graph: _FrozenGraph,
    visited: set[int],
) -> None:
    """Capture native graph entries before overridable graph traversal."""

    pending = [value]
    while pending:
        current = pending.pop()
        _retain_public_mapping(current, frozen_graph)
        value_type = type(current)
        is_native_container = _is_subclass(
            value_type, (BaseModel, tuple, list, dict)
        )
        if not is_native_container:
            continue

        identity = id(current)
        if identity in visited:
            continue
        visited.add(identity)

        existing_replay = frozen_graph.replay_entries.get(identity)
        if existing_replay is not None and existing_replay[0] is current:
            continue
        frozen_graph.completed_traversals.pop(identity, None)
        frozen_graph.completed_heights.pop(identity, None)
        frozen_graph.completed_mapping_discoveries.pop(identity, None)

        if _is_subclass(value_type, BaseModel):
            existing_model = frozen_graph.models.get(identity)
            if existing_model is not None and existing_model[0] is current:
                continue
            stored = _BASE_MODEL_DICT_DESCRIPTOR.__get__(current, BaseModel)
            try:
                extras = _BASE_MODEL_EXTRA_DESCRIPTOR.__get__(current, BaseModel)
            except AttributeError:
                extras = None
            extra_count = (
                dict.__len__(extras) if extras is not None and _is_subclass(type(extras), dict) else 0
            )
            _charge_native(frozen_graph, dict.__len__(stored) + extra_count)
            stored_entries = tuple(dict.items(stored))
            native_extra_entries = (
                tuple(dict.items(extras))
                if extras is not None and _is_subclass(type(extras), dict)
                else None
            )
            _retain_extra_mapping(extras, frozen_graph)
            frozen_graph.models[identity] = (
                current,
                stored_entries,
                extras,
                native_extra_entries,
            )
            entries = tuple(item for _name, item in stored_entries)
        else:
            existing_container = frozen_graph.native_containers.get(identity)
            if (
                _is_exact_type(value_type, (dict, list, tuple))
                and existing_container is not None
                and existing_container[0] is current
            ):
                continue

            if _is_subclass(value_type, tuple):
                _charge_native(frozen_graph, tuple.__len__(current))
                entries = tuple(tuple.__iter__(current))
            elif _is_subclass(value_type, list):
                _charge_native(frozen_graph, list.__len__(current))
                entries = tuple(list.__iter__(current))
            else:
                _charge_native(frozen_graph, dict.__len__(current))
                dict_entries = tuple(dict.items(current))
                entries = tuple(item for _key, item in dict_entries)
            if _is_exact_type(value_type, (dict, list, tuple)):
                frozen_graph.native_containers[identity] = (
                    current,
                    dict_entries if value_type is dict else entries,
                )
        frozen_graph.replay_entries[identity] = (current, entries)
        pending.extend(reversed(entries))


def _retain_reachable_public_sequences(
    value: Any,
    frozen_graph: _FrozenGraph,
) -> None:
    """Retain public sequences on authoritative frozen-storage paths."""

    pending = [(value, True)]
    visited: dict[int, bool] = {}
    while pending:
        current, authoritative = pending.pop()
        identity = id(current)
        previous_authority = visited.get(identity)
        if identity in visited and (
            previous_authority is True or not authoritative
        ):
            continue
        visited[identity] = authoritative

        is_model = _is_instance(current, BaseModel)
        if not is_model and authoritative:
            _retain_public_sequence(current, frozen_graph)

        value_type = type(current)
        if not _is_subclass(value_type, (BaseModel, tuple, list, dict)):
            continue
        frozen_replay = frozen_graph.replay_entries.get(identity)
        if frozen_replay is None or frozen_replay[0] is not current:
            _freeze_model_storage(current, frozen_graph, set())
            frozen_replay = frozen_graph.replay_entries[identity]
        _owner, entries = frozen_replay
        local_authority = _is_subclass(
            value_type, BaseModel
        ) or _is_exact_type(value_type, (dict, list, tuple))
        child_authority = authoritative and local_authority
        pending.extend(
            (item, child_authority) for item in reversed(entries)
        )


def _traverse_model_storage(
    value: Any,
    frozen_graph: _FrozenGraph,
    visited: set[int],
) -> None:
    """Retain historical native-recursion failures at validation order.

    A subtree already walked (in this call or an earlier one on the same graph) is not walked again:
    its recorded height is added to the depth where it is reached, so a subtree first seen shallow
    and reached deep later is still refused, and the replay work of a call is bounded by the
    containers it has not seen.
    """

    heights: dict[int, int] = {}
    depth = [0]

    def visit(current: Any) -> int:
        """The height of ``current`` in containers (0 for a leaf)."""

        value_type = type(current)
        is_native_container = _is_subclass(
            value_type, (BaseModel, tuple, list, dict)
        )
        if not is_native_container:
            return 0

        identity = id(current)
        if identity in visited:
            known = heights.get(identity)
            if known is not None and depth[0] + known > MAX_SNAPSHOT_DEPTH:
                raise ValueError(SNAPSHOT_BUDGET_MESSAGE)
            return known if known is not None else 0
        completed_height = frozen_graph.completed_heights.get(identity)
        # The root is always walked (its own entries); only the subtrees below it are reused.
        if (
            depth[0] > 0
            and completed_height is not None
            and frozen_graph.completed_traversals.get(identity) is current
        ):
            visited.add(identity)
            heights[identity] = completed_height
            if depth[0] + completed_height > MAX_SNAPSHOT_DEPTH:
                raise ValueError(SNAPSHOT_BUDGET_MESSAGE)
            return completed_height
        visited.add(identity)

        frozen_replay = frozen_graph.replay_entries.get(identity)
        if frozen_replay is None or frozen_replay[0] is not current:
            _freeze_model_storage(current, frozen_graph, set())
            frozen_replay = frozen_graph.replay_entries[identity]
        owner, entries = frozen_replay
        # The walk recurses; the snapshot's depth bound applies here too, so nesting deeper than
        # the manifest allows is a stable budget error and never a RecursionError.
        depth[0] += 1
        if depth[0] > MAX_SNAPSHOT_DEPTH:
            raise ValueError(SNAPSHOT_BUDGET_MESSAGE)
        height = 0
        for item in entries:
            height = max(height, visit(item))
        depth[0] -= 1
        heights[identity] = height + 1
        traversed[identity] = owner
        traversed_heights[identity] = height + 1
        return height + 1

    traversed: dict[int, object] = {}
    traversed_heights: dict[int, int] = {}
    visit(value)
    frozen_graph.completed_traversals.update(traversed)
    frozen_graph.completed_heights.update(traversed_heights)


def _protect_yielded_model_storage(
    value: Any,
    frozen_graph: _FrozenGraph,
) -> bool:
    """Capture and successfully traverse yielded supported storage once."""

    value_type = type(value)
    if not _is_subclass(value_type, (BaseModel, tuple, list, dict)):
        return False

    identity = id(value)
    frozen_replay = frozen_graph.replay_entries.get(identity)
    if frozen_replay is None or frozen_replay[0] is not value:
        _freeze_model_storage(value, frozen_graph, set())
        _inventory_retained_extra_mappings(frozen_graph)

    if frozen_graph.completed_traversals.get(identity) is value:
        return False

    _traverse_model_storage(value, frozen_graph, set())
    return True


def _inventory_reachable_public_mappings(
    value: Any,
    frozen_graph: _FrozenGraph,
    observing: dict[int, object],
) -> bool:
    """Inventory public mappings reachable through retained native storage."""

    pending = [(value, True)]
    visited: dict[int, bool] = {}
    discovered: dict[int, tuple[object, bool]] = {}
    while pending:
        current, authoritative = pending.pop()
        identity = id(current)
        previous_authority = visited.get(identity)
        if identity in visited and (
            previous_authority is True or not authoritative
        ):
            continue
        visited[identity] = authoritative

        is_model = _is_instance(current, BaseModel)
        if (
            not is_model
            and type(current) is not dict
            and _is_instance(current, Mapping)
        ):
            _retain_public_mapping(current, frozen_graph)
            active_owner = observing.get(identity)
            if active_owner is current:
                if authoritative:
                    return False
            elif not _inventory_public_mapping(
                current,
                frozen_graph,
                observing,
            ) and authoritative:
                return False
        elif (
            not is_model
            and authoritative
            and not _is_exact_type(type(current), (tuple, list))
            and _is_instance(current, (tuple, list))
        ):
            _retain_public_sequence(current, frozen_graph)
            active_owner = observing.get(identity)
            if active_owner is current:
                return False
            if not _inventory_public_sequence(
                current,
                frozen_graph,
                observing,
            ):
                return False

        prior_discovery = frozen_graph.completed_mapping_discoveries.get(
            identity
        )
        if (
            prior_discovery is not None
            and prior_discovery[0] is current
            and (prior_discovery[1] or not authoritative)
        ):
            continue

        value_type = type(current)
        if not _is_subclass(value_type, (BaseModel, tuple, list, dict)):
            continue
        frozen_replay = frozen_graph.replay_entries.get(identity)
        if frozen_replay is None or frozen_replay[0] is not current:
            _freeze_model_storage(current, frozen_graph, set())
            frozen_replay = frozen_graph.replay_entries[identity]
        _owner, entries = frozen_replay
        local_authority = _is_subclass(
            value_type, BaseModel
        ) or _is_exact_type(value_type, (dict, list, tuple))
        child_authority = authoritative and local_authority
        prior = discovered.get(identity)
        discovered[identity] = (
            current,
            child_authority or (prior is not None and prior[1]),
        )
        pending.extend(
            (item, child_authority) for item in reversed(entries)
        )
    for identity, discovery in discovered.items():
        prior = frozen_graph.completed_mapping_discoveries.get(identity)
        if prior is None or prior[0] is not discovery[0] or discovery[1]:
            frozen_graph.completed_mapping_discoveries[identity] = discovery
    return True


def _inventory_public_sequence(
    value: tuple[Any, ...] | list[Any],
    frozen_graph: _FrozenGraph,
    observing: dict[int, object],
) -> bool:
    """Retain one yielded sequence subclass's authoritative public view."""

    identity = id(value)
    frozen = frozen_graph.public_sequences.get(identity)
    if frozen is not None and frozen.owner is value:
        return frozen.complete
    if observing.get(identity) is value:
        return False

    observing[identity] = value
    entries: list[Any] = []
    failure: Exception | None = None
    complete = False
    try:
        try:
            iterator = iter(value)
        except Exception as error:
            failure = error
        else:
            while True:
                try:
                    item = next(iterator)
                except StopIteration:
                    complete = True
                    break
                except Exception as error:
                    failure = error
                    break
                _charge_capture(frozen_graph)
                entries.append(item)
                try:
                    _protect_yielded_model_storage(item, frozen_graph)
                    public_shape_complete = _inventory_yielded_public_shape(
                        item,
                        frozen_graph,
                        observing,
                    )
                except Exception as error:
                    failure = error
                    break
                if not public_shape_complete:
                    break
    finally:
        observing.pop(identity, None)

    frozen_graph.public_sequences[identity] = _FrozenPublicSequence(
        owner=value,
        entries=tuple(entries),
        failure=failure,
        complete=complete,
    )
    return complete


def _inventory_retained_public_sequences(frozen_graph: _FrozenGraph) -> None:
    """Observe retained public sequences in deterministic discovery order."""

    observing: dict[int, object] = {}
    while frozen_graph.public_sequence_cursor[0] < len(
        frozen_graph.public_sequence_queue
    ):
        cursor = frozen_graph.public_sequence_cursor[0]
        owner = frozen_graph.public_sequence_queue[cursor]
        frozen_graph.public_sequence_cursor[0] = cursor + 1
        identity = id(owner)
        if frozen_graph.public_sequence_owners.get(identity) is not owner:
            continue
        frozen = frozen_graph.public_sequences.get(identity)
        if frozen is not None and frozen.owner is owner:
            continue
        _inventory_public_sequence(owner, frozen_graph, observing)


def _inventory_yielded_public_shape(
    value: Any,
    frozen_graph: _FrozenGraph,
    observing: dict[int, object],
) -> bool:
    """Protect one yielded value's authoritative public shape."""

    identity = id(value)
    if not _is_instance(value, BaseModel):
        if type(value) is not dict and _is_instance(value, Mapping):
            if observing.get(identity) is value:
                return False
            _retain_public_mapping(value, frozen_graph)
            if not _inventory_public_mapping(value, frozen_graph, observing):
                return False
        elif not _is_exact_type(type(value), (tuple, list)) and _is_instance(value, (tuple, list)):
            if observing.get(identity) is value:
                return False
            _retain_public_sequence(value, frozen_graph)
            if not _inventory_public_sequence(value, frozen_graph, observing):
                return False

    return _inventory_reachable_public_mappings(
        value,
        frozen_graph,
        observing,
    )


def _inventory_public_mapping(
    value: Mapping[Any, Any],
    frozen_graph: _FrozenGraph,
    observing: dict[int, object],
) -> bool:
    """Observe one public mapping stream without performing key operations."""

    identity = id(value)
    frozen = frozen_graph.public_mappings.get(identity)
    if frozen is not None and frozen.owner is value:
        return frozen.complete

    active_owner = observing.get(identity)
    if active_owner is value:
        return False
    observing[identity] = value
    entries: list[tuple[Any, Any]] = []
    failure: Exception | None = None
    complete = False
    try:
        try:
            mapping_items = value.items()
            iterator = iter(mapping_items)
        except Exception as error:
            failure = error
        else:
            while True:
                try:
                    pair = next(iterator)
                except StopIteration:
                    complete = True
                    break
                except Exception as error:
                    failure = error
                    break

                try:
                    key, item = pair
                except Exception as error:
                    failure = error
                    break
                _charge_capture(frozen_graph)
                entries.append((key, item))

                try:
                    _protect_yielded_model_storage(item, frozen_graph)
                    public_shape_complete = _inventory_yielded_public_shape(
                        item,
                        frozen_graph,
                        observing,
                    )
                except Exception as error:
                    failure = error
                    break
                if not public_shape_complete:
                    break
    finally:
        observing.pop(identity, None)

    frozen_graph.public_mappings[identity] = _FrozenPublicMapping(
        owner=value,
        entries=tuple(entries),
        failure=failure,
        complete=complete,
    )
    return complete


def _inventory_retained_public_mappings(frozen_graph: _FrozenGraph) -> None:
    """Observe retained public mappings in deterministic discovery order."""

    observing: dict[int, object] = {}
    cursor = 0
    while cursor < len(frozen_graph.public_mapping_queue):
        pending_owner = frozen_graph.public_mapping_queue[cursor]
        cursor += 1
        identity = id(pending_owner)
        if frozen_graph.public_mapping_owners.get(identity) is not pending_owner:
            continue
        frozen = frozen_graph.public_mappings.get(identity)
        if frozen is not None and frozen.owner is pending_owner:
            continue
        if not _inventory_public_mapping(
            pending_owner,
            frozen_graph,
            observing,
        ):
            return


def _freeze_model_graphs(*values: Any) -> _FrozenGraph:
    """Capture every supplied native graph in one shared inventory.

    Supported inputs are JSON-compatible data, the project's own models and
    exact built-in containers, plus container subclasses read through their
    public views once at first observation.  Native backing that a public view
    hides is discovered defensively but is never authoritative.

    Hidden-backing aliases can make the shared discovery conservative: for
    example a shared exclusions tuple reachable from two measurements, whose
    public item has hidden backing that refers to the other measurement, is
    rejected with ``incomplete public mapping inventory`` even though its
    public views are acyclic.  That is a supported fail-closed outcome for
    in-process objects outside the input envelope (equivalent plain JSON is
    accepted); what must never happen is acceptance of data that differs from
    the public view captured at first observation.
    ``test_shared_exclusions_alias_through_hidden_backing_*`` pin this shape.
    """

    frozen_graph = _FrozenGraph(
        models={},
        native_containers={},
        replay_entries={},
        completed_traversals={},
        completed_heights={},
        extra_mapping_owners={},
        extra_mapping_queue=[],
        extra_mapping_cursor=[0],
        extra_inventories={},
        public_mapping_owners={},
        public_mapping_queue=[],
        public_mappings={},
        completed_mapping_discoveries={},
        public_sequence_owners={},
        public_sequence_queue=[],
        public_sequence_cursor=[0],
        public_sequences={},
        visits=[0],
        native_entries=[0],
        fixed_datetimes={},
    )
    visited: set[int] = set()
    for value in values:
        _freeze_model_storage(value, frozen_graph, visited)
    for value in values:
        _retain_reachable_public_sequences(value, frozen_graph)
    _inventory_retained_extra_mappings(frozen_graph)
    _inventory_retained_public_sequences(frozen_graph)
    _inventory_retained_public_mappings(frozen_graph)
    return frozen_graph


_LEAF_TYPES = (type(None), bool, int, float, str, datetime)
# The manifest has a handful of levels and a few thousand values.  The snapshot visits each
# *occurrence* (shared subtrees are not memoized because a public view must be observed once
# per occurrence), so an unbounded input could do exponential work or exhaust the stack.
MAX_SNAPSHOT_VISITS = 65_536
MAX_SNAPSHOT_DEPTH = 64
SNAPSHOT_BUDGET_MESSAGE = "manifest input exceeds the native snapshot budget"


def _charge_capture(frozen_graph: _FrozenGraph) -> None:
    """Charge one public-view entry as it is read, so an iterator that never ends is cut off.

    The capture and the snapshot share one budget: an entry read here is charged before it is
    kept, and the same counter then bounds the snapshot that follows.
    """

    frozen_graph.visits[0] += 1
    if frozen_graph.visits[0] > MAX_SNAPSHOT_VISITS:
        raise ValueError(SNAPSHOT_BUDGET_MESSAGE)


def _charge_native(frozen_graph: _FrozenGraph, count: int) -> None:
    """Charge a native container's entries *before* they are copied and retained.

    The count comes from the container's own length slot (no caller code runs), so an oversized
    native graph is refused before it is materialized.  Every container but the root is an entry of
    its parent, so counting entries bounds the nodes that are copied.  The limit is the snapshot's own
    (``MAX_SNAPSHOT_VISITS``) but the counter is separate: a graph the snapshot accepts has at most that
    many entries, so the check never refuses an input the snapshot would take, and the two phases do
    not add up.
    """

    if count > MAX_SNAPSHOT_VISITS - frozen_graph.native_entries[0]:
        frozen_graph.native_entries[0] = MAX_SNAPSHOT_VISITS + 1
        frozen_graph.visits[0] = MAX_SNAPSHOT_VISITS + 1
        raise ValueError(SNAPSHOT_BUDGET_MESSAGE)
    frozen_graph.native_entries[0] += count


def _charge_snapshot(frozen_graph: _FrozenGraph, active: set[int]) -> None:
    """Charge one visit and enforce the depth bound (depth = containers on the active path)."""

    frozen_graph.visits[0] += 1
    if frozen_graph.visits[0] > MAX_SNAPSHOT_VISITS or len(active) > MAX_SNAPSHOT_DEPTH:
        raise ValueError(SNAPSHOT_BUDGET_MESSAGE)


_ALLOWED_ENUMS = (RunState, CacheCondition, LimitKind, ObservationKind)


def _is_supported_leaf(value: Any) -> bool:
    """Whether ``value`` is a JSON-style scalar, a datetime or a member of a project enum.

    Everything is decided on the value's real type by identity.  Only members of the
    manifest's own enum classes are accepted, never a generic ``Enum`` subclass: an enum
    member can define a ``__class__`` callback and delegate model storage to another
    object, and Pydantic would read that ``__class__`` while validating.
    """

    kind = type(value)
    return _is_exact_type(kind, _LEAF_TYPES) or _is_exact_type(kind, _ALLOWED_ENUMS)


_KEY_LABEL_CHARS = 64
_UNDECLARED_NAMES_LISTED = 5
_KEY_LABEL_INT_BITS = 128


def _key_label(key: object) -> str:
    """A short label for a mapping key, built only from its exact text or number.

    A key can be a ``str`` subclass (or any hashable) whose ``__repr__`` is caller code that may run
    side effects or return megabytes; only the base types' own methods are used here, and long text
    is cut.
    """

    kind = type(key)
    if _is_instance(key, str):
        text = str.__getitem__(key, slice(0, _KEY_LABEL_CHARS))  # type: ignore[arg-type]
        cut = str.__len__(key) > _KEY_LABEL_CHARS  # type: ignore[arg-type]
        return str.__repr__(text) + ("…" if cut else "")
    if kind is bool:
        return "True" if key is True else "False"
    if kind is int:
        bits = int.bit_length(key)  # type: ignore[arg-type]
        if bits > _KEY_LABEL_INT_BITS:
            return f"<int key of {bits} bits>"
        return int.__repr__(key)  # type: ignore[arg-type]
    # The class name is a native read, but it can be a ``str`` subclass (its __format__ is caller code) of any
    # length: copy it to an exact str with the base slice, and cut it like a key's own text.
    name = type.__dict__["__name__"].__get__(kind)
    if not _is_instance(name, str):
        return "<key>"
    text = str.__getitem__(name, slice(0, _KEY_LABEL_CHARS))  # type: ignore[arg-type]
    cut = "…" if str.__len__(name) > _KEY_LABEL_CHARS else ""  # type: ignore[arg-type]
    return "<" + text + cut + " key>"


class _Path:
    """A location in the caller's input, rendered only when an error names it.

    Building ``f"{path}[{key!r}]"`` for every visited entry ran each key's ``__repr__`` on the success
    path too; a ``_Path`` keeps the parent and the key and labels it with ``_key_label`` on demand.
    """

    __slots__ = ("parent", "step")

    def __init__(self, parent: "_Path | str", step: object) -> None:
        self.parent = parent
        self.step = step  # ("field", name), ("key", key) or ("index", int)

    def __str__(self) -> str:
        parts: list[str] = []
        node: _Path | str = self
        while type(node) is _Path:
            kind, item = node.step  # type: ignore[misc]
            if kind == "field":
                parts.append("." + str.__str__(item))
            elif kind == "index":
                parts.append(f"[{item}]")
            else:
                parts.append(f"[{_key_label(item)}]")
            node = node.parent
        parts.append(node)  # type: ignore[arg-type]
        return "".join(reversed(parts))


def _require_native_snapshot(snapshot: Any) -> Any:
    """Reject, with a stable error and before Pydantic sees it, any unsupported value.

    ``_snapshot_native`` rebuilds models, mappings and sequences as exact dicts, lists and
    tuples and returns every other value untouched.  An object that merely *reports* a model
    class through ``__class__`` (or delegates attributes to one) would otherwise reach
    Pydantic, which can recognise it as an already-validated instance and skip revalidating
    its fields.  The walk is iterative and reads only real types.
    """

    pending: list[tuple[Any, _Path | str]] = [(snapshot, "$")]
    seen: set[int] = set()
    while pending:
        value, path = pending.pop()
        kind = type(value)
        if _is_exact_type(kind, (dict, list, tuple)):
            if id(value) in seen:
                continue
            seen.add(id(value))
            if kind is dict:
                pending.extend((item, _Path(path, ("key", key))) for key, item in dict.items(value))
            else:
                pending.extend((item, _Path(path, ("index", index))) for index, item in enumerate(value))
        elif not _is_supported_leaf(value):
            raise ValueError(f"unsupported value type at {path}")
    return snapshot


def _fixed_offset(value: datetime, path: _Path | str) -> datetime:
    """Read a datetime's UTC offset exactly once and keep a fixed-offset copy.

    A ``tzinfo`` is caller code: its ``utcoffset`` can return a different answer on every
    call (or depend on ``fold``), so a value validated against one reading could serialize
    or compare against another.  The offset is read once here, at capture, and the
    manifest keeps the same wall time under that fixed offset (an exact ``timedelta``).  A value with
    no offset becomes naive, which validation refuses.
    """

    try:
        offset = datetime.utcoffset(value)
        if offset is None:
            # No offset at capture: drop the caller's tzinfo, so validation refuses the naive value
            # and a later read cannot find an offset that was not there.
            return datetime.replace(value, tzinfo=None)
        return datetime.replace(value, tzinfo=timezone(_exact_offset(offset)))
    except Exception:  # a tzinfo is caller code: it can raise, or return something that is no offset
        raise ValueError(f"unsupported UTC offset at {path}") from None


def _snapshot_native(
    value: Any,
    *,
    path: _Path | str = "$",
    active: set[int] | None = None,
    _frozen_graph: _FrozenGraph | None = None,
) -> Any:
    """Copy native input without coercion while enforcing closed model storage.

    Pydantic's unchecked construction and copy APIs can create model instances
    whose stored values have never passed field validation.  They can also add
    undeclared keys that ``model_dump`` omits.  This snapshot preserves native
    tuple/list distinctions and mapping keys so strict revalidation sees the
    caller's actual structure instead of a normalized substitute.
    """

    if _frozen_graph is None:
        _frozen_graph = _freeze_model_graphs(value)
        _traverse_model_storage(value, _frozen_graph, set())
    if active is None:
        active = set()

    is_container = _is_instance(value, (BaseModel, Mapping, tuple, list))
    identity = id(value)
    if is_container:
        if identity in active:
            raise ValueError(f"cyclic native input at {path}")
        active.add(identity)
    _charge_snapshot(_frozen_graph, active)

    try:
        if _is_instance(value, BaseModel):
            fields = type(value).model_fields
            frozen = _frozen_graph.models.get(identity)
            if frozen is None or frozen[0] is not value:
                _freeze_model_storage(value, _frozen_graph, set())
                _inventory_retained_extra_mappings(_frozen_graph)
                _traverse_model_storage(value, _frozen_graph, set())
                frozen = _frozen_graph.models[identity]
            _model, stored_entries, extras, native_extra_entries = frozen
            stored_values = dict(stored_entries)
            undeclared = set(stored_values).difference(fields)
            if extras is not None:
                if native_extra_entries is not None:
                    extra_entries = native_extra_entries
                    iterated_extra_names = tuple(key for key, _ in extra_entries)
                else:
                    if not _is_instance(extras, Mapping):
                        raise ValueError(f"malformed stored extras at {path}")
                    _retain_extra_mapping(extras, _frozen_graph)
                    _inventory_retained_extra_mappings(_frozen_graph)
                    extra_inventory = _frozen_graph.extra_inventories[
                        id(extras)
                    ]
                    if extra_inventory.failure is not None:
                        raise extra_inventory.failure
                    iterated_extra_names = extra_inventory.iterated_names
                    extra_entries = extra_inventory.item_entries
                undeclared.update(iterated_extra_names)
                undeclared.update(key for key, _ in extra_entries)
            if undeclared:
                # Name the first few (sorted by label) and count the rest, so the error stays small.
                labels = heapq.nsmallest(_UNDECLARED_NAMES_LISTED, map(_key_label, undeclared))
                names = ", ".join(labels)
                if len(undeclared) > len(labels):
                    names += f", … and {len(undeclared) - len(labels)} more"
                raise ValueError(f"undeclared stored fields at {path}: {names}")
            return {
                name: _snapshot_native(
                    stored_values[name],
                    path=_Path(path, ("field", name)),
                    active=active,
                    _frozen_graph=_frozen_graph,
                )
                for name in fields
                if name in stored_values
            }

        if _is_instance(value, Mapping):
            if type(value) is dict:
                frozen_container = _frozen_graph.native_containers.get(identity)
                if frozen_container is None or frozen_container[0] is not value:
                    _freeze_model_storage(value, _frozen_graph, set())
                    _traverse_model_storage(value, _frozen_graph, set())
                    frozen_container = _frozen_graph.native_containers[identity]
                mapping_items = frozen_container[1]
                return {
                    key: _snapshot_native(
                        item,
                        path=_Path(path, ("key", key)),
                        active=active,
                        _frozen_graph=_frozen_graph,
                    )
                    for key, item in mapping_items
                }

            _retain_public_mapping(value, _frozen_graph)
            frozen_mapping = _frozen_graph.public_mappings.get(identity)
            if frozen_mapping is None or frozen_mapping.owner is not value:
                _inventory_public_mapping(value, _frozen_graph, {})
                frozen_mapping = _frozen_graph.public_mappings[identity]

            projected: dict[Any, Any] = {}
            for key, item in frozen_mapping.entries:
                _protect_yielded_model_storage(item, _frozen_graph)
                item_path = _Path(path, ("key", key))
                projected[key] = _snapshot_native(
                    item,
                    path=item_path,
                    active=active,
                    _frozen_graph=_frozen_graph,
                )
            if frozen_mapping.failure is not None:
                raise frozen_mapping.failure
            if not frozen_mapping.complete:
                raise ValueError(f"incomplete public mapping inventory at {path}")
            return projected
        if _is_instance(value, tuple):
            if type(value) is tuple:
                frozen_container = _frozen_graph.native_containers.get(identity)
                if frozen_container is None or frozen_container[0] is not value:
                    _freeze_model_storage(value, _frozen_graph, set())
                    _traverse_model_storage(value, _frozen_graph, set())
                    frozen_container = _frozen_graph.native_containers[identity]
                tuple_items = frozen_container[1]
            else:
                frozen_sequence = _frozen_graph.public_sequences.get(identity)
                if (
                    frozen_sequence is None
                    or frozen_sequence.owner is not value
                ):
                    _retain_public_sequence(value, _frozen_graph)
                    _inventory_public_sequence(value, _frozen_graph, {})
                    frozen_sequence = _frozen_graph.public_sequences[identity]
                projected = tuple(
                    _snapshot_native(
                        item,
                        path=_Path(path, ("index", index)),
                        active=active,
                        _frozen_graph=_frozen_graph,
                    )
                    for index, item in enumerate(frozen_sequence.entries)
                )
                if frozen_sequence.failure is not None:
                    raise frozen_sequence.failure
                if not frozen_sequence.complete:
                    raise ValueError(
                        f"incomplete public sequence inventory at {path}"
                    )
                return projected
            return tuple(
                _snapshot_native(
                    item,
                    path=_Path(path, ("index", index)),
                    active=active,
                    _frozen_graph=_frozen_graph,
                )
                for index, item in enumerate(tuple_items)
            )
        if _is_instance(value, list):
            if type(value) is list:
                frozen_container = _frozen_graph.native_containers.get(identity)
                if frozen_container is None or frozen_container[0] is not value:
                    _freeze_model_storage(value, _frozen_graph, set())
                    _traverse_model_storage(value, _frozen_graph, set())
                    frozen_container = _frozen_graph.native_containers[identity]
                list_items = frozen_container[1]
            else:
                frozen_sequence = _frozen_graph.public_sequences.get(identity)
                if (
                    frozen_sequence is None
                    or frozen_sequence.owner is not value
                ):
                    _retain_public_sequence(value, _frozen_graph)
                    _inventory_public_sequence(value, _frozen_graph, {})
                    frozen_sequence = _frozen_graph.public_sequences[identity]
                projected = [
                    _snapshot_native(
                        item,
                        path=_Path(path, ("index", index)),
                        active=active,
                        _frozen_graph=_frozen_graph,
                    )
                    for index, item in enumerate(frozen_sequence.entries)
                ]
                if frozen_sequence.failure is not None:
                    raise frozen_sequence.failure
                if not frozen_sequence.complete:
                    raise ValueError(
                        f"incomplete public sequence inventory at {path}"
                    )
                return projected
            return [
                _snapshot_native(
                    item,
                    path=_Path(path, ("index", index)),
                    active=active,
                    _frozen_graph=_frozen_graph,
                )
                for index, item in enumerate(list_items)
            ]
        if type(value) is datetime:
            # One offset per object for the whole call: the same datetime supplied twice (for
            # example as both timestamps) is read once, at its first occurrence.
            cached = _frozen_graph.fixed_datetimes.get(identity)
            if cached is not None and cached[0] is value:
                return cached[1]
            fixed = _fixed_offset(value, path)
            _frozen_graph.fixed_datetimes[identity] = (value, fixed)
            return fixed
        return value
    finally:
        if is_container:
            active.remove(identity)


def _validated_manifest_snapshot(
    manifest: RunManifest,
    *,
    _frozen_graph: _FrozenGraph | None = None,
) -> RunManifest:
    """Return a fresh, deeply validated manifest graph from native evidence."""

    if not _is_instance(manifest, RunManifest):
        raise TypeError("manifest must be a RunManifest")
    return RunManifest.model_validate(
        _require_native_snapshot(_snapshot_native(manifest, _frozen_graph=_frozen_graph))
    )


def finish_run(
    planned: RunManifest,
    *,
    state: Literal[RunState.COMPLETED, RunState.ABORTED],
    started_at: datetime,
    finished_at: datetime,
    resources: tuple[ResourceReading, ...],
    measurements: tuple[TimingMeasurement, ...] = (),
) -> RunManifest:
    """Finish a planned run without allowing its declared envelope to change."""

    frozen_graph = _freeze_model_graphs(
        planned,
        state,
        started_at,
        finished_at,
        resources,
        measurements,
    )
    _traverse_model_storage(planned, frozen_graph, set())
    validated_planned = _validated_manifest_snapshot(
        planned,
        _frozen_graph=frozen_graph,
    )
    if validated_planned.state is not RunState.PLANNED:
        raise ValueError("only a planned run can be finished")

    _traverse_model_storage(validated_planned, frozen_graph, set())
    candidate_data = _snapshot_native(
        validated_planned,
        _frozen_graph=frozen_graph,
    )
    _traverse_model_storage(state, frozen_graph, set())
    frozen_state = _snapshot_native(
        state,
        path="$.state",
        _frozen_graph=frozen_graph,
    )
    _traverse_model_storage(started_at, frozen_graph, set())
    frozen_started_at = _snapshot_native(
        started_at,
        path="$.started_at",
        _frozen_graph=frozen_graph,
    )
    _traverse_model_storage(finished_at, frozen_graph, set())
    frozen_finished_at = _snapshot_native(
        finished_at,
        path="$.finished_at",
        _frozen_graph=frozen_graph,
    )
    _traverse_model_storage(resources, frozen_graph, set())
    frozen_resources = _snapshot_native(
        resources,
        path="$.resources",
        _frozen_graph=frozen_graph,
    )
    _traverse_model_storage(measurements, frozen_graph, set())
    frozen_measurements = _snapshot_native(
        measurements,
        path="$.measurements",
        _frozen_graph=frozen_graph,
    )
    candidate_data.update(
        state=frozen_state,
        started_at=frozen_started_at,
        finished_at=frozen_finished_at,
        resources=frozen_resources,
        measurements=frozen_measurements,
    )
    candidate = RunManifest.model_validate(_require_native_snapshot(candidate_data))
    if candidate.state is RunState.PLANNED:
        raise ValueError("finish_run requires a terminal state (completed or aborted)")

    planned_resources = {
        (item.scope, item.resource): (item.unit, item.limit_kind, item.limit_value)
        for item in validated_planned.resources
    }
    finished_resources = {
        (item.scope, item.resource): (item.unit, item.limit_kind, item.limit_value)
        for item in candidate.resources
    }
    if finished_resources != planned_resources:
        raise ValueError("finishing a run cannot change its declared resource envelope")
    return candidate


def serialize_run_manifest(manifest: RunManifest) -> bytes:
    """Return canonical UTF-8 JSON bytes for hashing, comparison, and storage."""

    payload = _validated_manifest_snapshot(manifest).model_dump(mode="json")
    return json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def validate_run_manifest_json(data: str | bytes) -> RunManifest:
    """Parse one manifest, rejecting duplicate JSON keys and invalid contracts."""

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    decoded = json.loads(data, object_pairs_hook=reject_duplicate_keys)
    if not isinstance(decoded, dict):
        raise ValueError("a run manifest must be a JSON object")
    # Strict native validation deliberately rejects JSON representations such
    # as arrays for tuples and strings for timestamps.  Validate the original
    # bytes in JSON mode after the structural duplicate-key check.
    return RunManifest.model_validate_json(data)
