"""Deterministic, product-independent records for reproducible benchmark runs.

The models in this module describe evidence about a run.  They do not confer
authorization and they make no claim that a workload, measurement, or resource
limit is suitable for production.  An absent measurement is unknown; it must
not be synthesized as a zero.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime
from enum import Enum
from typing import Annotated, Any, Literal, NamedTuple, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)


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
    public_mapping_owners: dict[int, Mapping[Any, Any]]
    public_mapping_queue: list[Mapping[Any, Any]]
    public_mappings: dict[int, _FrozenPublicMapping]
    completed_mapping_discoveries: dict[int, tuple[object, bool]]
    public_sequences: dict[int, _FrozenPublicSequence]


class _ManifestContract(BaseModel):
    """Strict, immutable base local to the standalone benchmark package."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
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


class RunManifest(_ManifestContract):
    """A reproducible benchmark run envelope with explicit evidence state."""

    run_id: _OpaqueId
    state: RunState
    build: BuildIdentity
    workload: WorkloadIdentity
    cache_condition: CacheCondition
    concurrency: tuple[ConcurrencySetting, ...]
    resources: tuple[ResourceReading, ...]
    started_at: datetime | None = None
    finished_at: datetime | None = None
    measurements: tuple[TimingMeasurement, ...] = ()

    @field_validator("started_at", "finished_at")
    @classmethod
    def require_aware_timestamp(cls, value: datetime | None) -> datetime | None:
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("run timestamps must include an explicit UTC offset")
        return value

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
            if self.finished_at < self.started_at:
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
        isinstance(value, BaseModel)
        or type(value) is dict
        or not isinstance(value, Mapping)
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
        is_native_container = issubclass(
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
        frozen_graph.completed_mapping_discoveries.pop(identity, None)

        if issubclass(value_type, BaseModel):
            existing_model = frozen_graph.models.get(identity)
            if existing_model is not None and existing_model[0] is current:
                continue
            stored = _BASE_MODEL_DICT_DESCRIPTOR.__get__(current, BaseModel)
            stored_entries = tuple(dict.items(stored))
            try:
                extras = _BASE_MODEL_EXTRA_DESCRIPTOR.__get__(current, BaseModel)
            except AttributeError:
                extras = None
            native_extra_entries = (
                tuple(dict.items(extras))
                if extras is not None and issubclass(type(extras), dict)
                else None
            )
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
                value_type in (dict, list, tuple)
                and existing_container is not None
                and existing_container[0] is current
            ):
                continue

            if issubclass(value_type, tuple):
                entries = tuple(tuple.__iter__(current))
            elif issubclass(value_type, list):
                entries = tuple(list.__iter__(current))
            else:
                dict_entries = tuple(dict.items(current))
                entries = tuple(item for _key, item in dict_entries)
            if value_type in (dict, list, tuple):
                frozen_graph.native_containers[identity] = (
                    current,
                    dict_entries if value_type is dict else entries,
                )
        frozen_graph.replay_entries[identity] = (current, entries)
        pending.extend(reversed(entries))


def _traverse_model_storage(
    value: Any,
    frozen_graph: _FrozenGraph,
    visited: set[int],
) -> None:
    """Retain historical native-recursion failures at validation order."""

    traversed: dict[int, object] = {}

    def visit(current: Any) -> None:
        value_type = type(current)
        is_native_container = issubclass(
            value_type, (BaseModel, tuple, list, dict)
        )
        if not is_native_container:
            return

        identity = id(current)
        if identity in visited:
            return
        visited.add(identity)

        frozen_replay = frozen_graph.replay_entries.get(identity)
        if frozen_replay is None or frozen_replay[0] is not current:
            _freeze_model_storage(current, frozen_graph, set())
            frozen_replay = frozen_graph.replay_entries[identity]
        owner, entries = frozen_replay
        traversed[identity] = owner
        for item in entries:
            visit(item)

    visit(value)
    frozen_graph.completed_traversals.update(traversed)


def _protect_yielded_model_storage(
    value: Any,
    frozen_graph: _FrozenGraph,
) -> bool:
    """Capture and successfully traverse yielded supported storage once."""

    value_type = type(value)
    if not issubclass(value_type, (BaseModel, tuple, list, dict)):
        return False

    identity = id(value)
    frozen_replay = frozen_graph.replay_entries.get(identity)
    if frozen_replay is None or frozen_replay[0] is not value:
        _freeze_model_storage(value, frozen_graph, set())

    completed_owner = frozen_graph.completed_traversals.get(identity)
    if completed_owner is value:
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

        is_model = isinstance(current, BaseModel)
        if (
            not is_model
            and type(current) is not dict
            and isinstance(current, Mapping)
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
            and type(current) not in (tuple, list)
            and isinstance(current, (tuple, list))
        ):
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
        if not issubclass(value_type, (BaseModel, tuple, list, dict)):
            continue
        frozen_replay = frozen_graph.replay_entries.get(identity)
        if frozen_replay is None or frozen_replay[0] is not current:
            _freeze_model_storage(current, frozen_graph, set())
            frozen_replay = frozen_graph.replay_entries[identity]
        _owner, entries = frozen_replay
        local_authority = issubclass(value_type, BaseModel) or value_type in (
            dict,
            list,
            tuple,
        )
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


def _inventory_yielded_public_shape(
    value: Any,
    frozen_graph: _FrozenGraph,
    observing: dict[int, object],
) -> bool:
    """Protect one yielded value's authoritative public shape."""

    identity = id(value)
    if not isinstance(value, BaseModel):
        if type(value) is not dict and isinstance(value, Mapping):
            if observing.get(identity) is value:
                return False
            _retain_public_mapping(value, frozen_graph)
            if not _inventory_public_mapping(value, frozen_graph, observing):
                return False
        elif type(value) not in (tuple, list) and isinstance(
            value, (tuple, list)
        ):
            if observing.get(identity) is value:
                return False
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
    """Capture every supplied native graph in one shared inventory."""

    frozen_graph = _FrozenGraph(
        models={},
        native_containers={},
        replay_entries={},
        completed_traversals={},
        public_mapping_owners={},
        public_mapping_queue=[],
        public_mappings={},
        completed_mapping_discoveries={},
        public_sequences={},
    )
    visited: set[int] = set()
    for value in values:
        _freeze_model_storage(value, frozen_graph, visited)
    _inventory_retained_public_mappings(frozen_graph)
    return frozen_graph


def _snapshot_native(
    value: Any,
    *,
    path: str = "$",
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

    is_container = isinstance(value, (BaseModel, Mapping, tuple, list))
    identity = id(value)
    if is_container:
        if identity in active:
            raise ValueError(f"cyclic native input at {path}")
        active.add(identity)

    try:
        if isinstance(value, BaseModel):
            fields = type(value).model_fields
            frozen = _frozen_graph.models.get(identity)
            if frozen is None or frozen[0] is not value:
                _freeze_model_storage(value, _frozen_graph, set())
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
                    if not isinstance(extras, Mapping):
                        raise ValueError(f"malformed stored extras at {path}")
                    iterated_extra_names = tuple(extras)
                    extra_entries = tuple(extras.items())
                undeclared.update(iterated_extra_names)
                undeclared.update(key for key, _ in extra_entries)
            if undeclared:
                names = ", ".join(sorted(repr(name) for name in undeclared))
                raise ValueError(f"undeclared stored fields at {path}: {names}")
            return {
                name: _snapshot_native(
                    stored_values[name],
                    path=f"{path}.{name}",
                    active=active,
                    _frozen_graph=_frozen_graph,
                )
                for name in fields
                if name in stored_values
            }

        if isinstance(value, Mapping):
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
                        path=f"{path}[{key!r}]",
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
                item_path = f"{path}[{key!r}]"
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
        if isinstance(value, tuple):
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
                    frozen_sequence is not None
                    and frozen_sequence.owner is value
                ):
                    projected = tuple(
                        _snapshot_native(
                            item,
                            path=f"{path}[{index}]",
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
                tuple_items = value
            return tuple(
                _snapshot_native(
                    item,
                    path=f"{path}[{index}]",
                    active=active,
                    _frozen_graph=_frozen_graph,
                )
                for index, item in enumerate(tuple_items)
            )
        if isinstance(value, list):
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
                    frozen_sequence is not None
                    and frozen_sequence.owner is value
                ):
                    projected = [
                        _snapshot_native(
                            item,
                            path=f"{path}[{index}]",
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
                list_items = value
            return [
                _snapshot_native(
                    item,
                    path=f"{path}[{index}]",
                    active=active,
                    _frozen_graph=_frozen_graph,
                )
                for index, item in enumerate(list_items)
            ]
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

    if not isinstance(manifest, RunManifest):
        raise TypeError("manifest must be a RunManifest")
    return RunManifest.model_validate(
        _snapshot_native(manifest, _frozen_graph=_frozen_graph)
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
    candidate = RunManifest.model_validate(candidate_data)

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
