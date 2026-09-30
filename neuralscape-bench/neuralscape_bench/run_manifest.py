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
from typing import Annotated, Any, Literal, Self

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


def _snapshot_native(value: Any, *, path: str = "$", active: set[int] | None = None) -> Any:
    """Copy native input without coercion while enforcing closed model storage.

    Pydantic's unchecked construction and copy APIs can create model instances
    whose stored values have never passed field validation.  They can also add
    undeclared keys that ``model_dump`` omits.  This snapshot preserves native
    tuple/list distinctions and mapping keys so strict revalidation sees the
    caller's actual structure instead of a normalized substitute.
    """

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
            stored = vars(value)
            undeclared = set(stored).difference(fields)
            extras = getattr(value, "__pydantic_extra__", None)
            if extras is not None:
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
                name: _snapshot_native(stored[name], path=f"{path}.{name}", active=active)
                for name in fields
                if name in stored
            }

        if isinstance(value, Mapping):
            return {
                key: _snapshot_native(item, path=f"{path}[{key!r}]", active=active)
                for key, item in value.items()
            }
        if isinstance(value, tuple):
            return tuple(
                _snapshot_native(item, path=f"{path}[{index}]", active=active)
                for index, item in enumerate(value)
            )
        if isinstance(value, list):
            return [
                _snapshot_native(item, path=f"{path}[{index}]", active=active)
                for index, item in enumerate(value)
            ]
        return value
    finally:
        if is_container:
            active.remove(identity)


def _validated_manifest_snapshot(manifest: RunManifest) -> RunManifest:
    """Return a fresh, deeply validated manifest graph from native evidence."""

    if not isinstance(manifest, RunManifest):
        raise TypeError("manifest must be a RunManifest")
    return RunManifest.model_validate(_snapshot_native(manifest))


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

    validated_planned = _validated_manifest_snapshot(planned)
    if validated_planned.state is not RunState.PLANNED:
        raise ValueError("only a planned run can be finished")

    candidate_data = _snapshot_native(validated_planned)
    candidate_data.update(
        state=_snapshot_native(state, path="$.state"),
        started_at=_snapshot_native(started_at, path="$.started_at"),
        finished_at=_snapshot_native(finished_at, path="$.finished_at"),
        resources=_snapshot_native(resources, path="$.resources"),
        measurements=_snapshot_native(measurements, path="$.measurements"),
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
