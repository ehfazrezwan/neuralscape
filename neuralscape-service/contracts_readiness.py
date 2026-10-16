"""Capability-specific readiness observations and publication validation.

Readiness is evidence about a named capability.  It is deliberately not
derived from process liveness, object construction, or another capability.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import AwareDatetime, TypeAdapter, model_validator

from contracts_common import OpaqueId, SafeCounter, VersionedContract
from contracts_tenancy import _closed_graph_snapshot_session, _inherits


ReadinessCapability = Literal[
    "api_acceptance", "exact_reads", "graph_reads", "worker_processing"
]
"""Independently probed service capabilities."""

CapabilityStatus = Literal[
    "disabled", "unconfigured", "healthy", "degraded", "unavailable", "unknown"
]
"""Operational status without coercing unknown or unavailable to healthy."""

_AWARE_DATETIME_ADAPTER = TypeAdapter(AwareDatetime)
_OPAQUE_ID_ADAPTER = TypeAdapter(OpaqueId)
_SAFE_COUNTER_ADAPTER = TypeAdapter(SafeCounter)


class CapabilityReadiness(VersionedContract):
    """One capability observation timestamped by its bounded real probe."""

    capability: ReadinessCapability
    status: CapabilityStatus
    observed_at: AwareDatetime
    detail_code: OpaqueId | None = None


class TenantReadinessPublication(VersionedContract):
    """Readiness observations bound to one tenant placement generation."""

    tenant_id: OpaqueId
    placement_generation: SafeCounter
    capabilities: tuple[CapabilityReadiness, ...]

    @model_validator(mode="after")
    def validate_capability_set(self) -> TenantReadinessPublication:
        if not self.capabilities:
            raise ValueError(
                "readiness publication must contain at least one capability"
            )
        names = [observation.capability for observation in self.capabilities]
        if len(names) != len(set(names)):
            raise ValueError("readiness publication capabilities must be unique")
        return self


def _snapshot_closed_graph(value: object) -> object:
    """Copy a model/native graph without dropping stored undeclared fields.

    Uses the same bounded, callback-free capture and public-view rebuild as the
    tenant contracts, so both receiving boundaries share one set of limits.
    """

    freeze_native_graph, rebuild = _closed_graph_snapshot_session()
    freeze_native_graph(value)
    return rebuild(value)


_DATETIME_PARTS = (
    datetime.year, datetime.month, datetime.day, datetime.hour, datetime.minute, datetime.second,
    datetime.microsecond,
)


def _pin_datetime(value: object) -> object:
    """A caller-owned aware ``datetime`` as a plain one with a fixed offset, read exactly once.

    The offset is asked of the caller's ``tzinfo`` a single time (through the built-in accessor) and the
    fields are read through the built-in descriptors, so a ``tzinfo`` or ``datetime`` subclass that answers
    differently on each call cannot give validation one instant and the comparison another.  A
    ``datetime`` that has no offset on that one read becomes a plain naive ``datetime`` (the caller's
    zone is dropped, so a later read cannot find an offset that was not there) and validation refuses it.
    Anything that is not a ``datetime`` is returned unchanged for validation to refuse.
    """

    if not _inherits(type(value), datetime):
        return value
    offset = datetime.utcoffset(value)  # type: ignore[arg-type]
    fields = [part.__get__(value, datetime) for part in _DATETIME_PARTS]
    if offset is None:
        return datetime(*fields)
    return datetime(*fields, tzinfo=timezone(offset))


def _pin_datetimes(value: object, memo: dict[int, tuple[object, object]] | None = None) -> object:
    """Pin every ``datetime`` in a captured (plain dict, tuple or list) snapshot, once per object.

    The snapshot shares a subtree that several edges name; the pinned copy is shared the same way
    (memoized by identity, the original retained so its id cannot be reused), so each timestamp is read
    once however many paths reach it and the work is proportional to the distinct objects.  A subtree
    holding no datetime is returned as it is.
    """

    if memo is None:
        memo = {}
    kind = type(value)
    leaf = kind is not dict and kind is not tuple and kind is not list
    if leaf and not _inherits(kind, datetime):
        return value  # nothing to pin and nothing to share
    cached = memo.get(id(value))
    if cached is not None and cached[0] is value:
        return cached[1]
    if leaf:
        pinned_leaf = _pin_datetime(value)  # the same datetime object is read once, at its first occurrence
        memo[id(value)] = (value, pinned_leaf)
        return pinned_leaf
    if kind is dict:
        items = [(key, _pin_datetimes(item, memo)) for key, item in value.items()]
        changed = any(new is not old for (_k, new), old in zip(items, value.values()))
        pinned: object = dict(items) if changed else value
    else:
        elements = [_pin_datetimes(item, memo) for item in value]
        changed = any(new is not old for new, old in zip(elements, value))
        pinned = kind(elements) if changed else value
    memo[id(value)] = (value, pinned)
    return pinned


def _validated_observation_snapshot(
    observation: CapabilityReadiness,
) -> CapabilityReadiness:
    if not isinstance(observation, CapabilityReadiness):
        raise TypeError("observation must be a CapabilityReadiness")
    return CapabilityReadiness.model_validate(
        _pin_datetimes(_snapshot_closed_graph(observation))
    )


def _validated_publication_snapshot(
    publication: TenantReadinessPublication,
) -> TenantReadinessPublication:
    if not isinstance(publication, TenantReadinessPublication):
        raise TypeError("publication must be a TenantReadinessPublication")
    return TenantReadinessPublication.model_validate(
        _pin_datetimes(_snapshot_closed_graph(publication))
    )


def validate_readiness_publication(
    publication: TenantReadinessPublication,
    *,
    expected_tenant_id: OpaqueId,
    current_placement_generation: SafeCounter,
) -> TenantReadinessPublication:
    """Reject cross-tenant and stale-generation readiness publication."""

    publication = _validated_publication_snapshot(publication)
    validated_tenant_id = _OPAQUE_ID_ADAPTER.validate_python(
        expected_tenant_id, strict=True
    )
    validated_generation = _SAFE_COUNTER_ADAPTER.validate_python(
        current_placement_generation, strict=True
    )
    if publication.tenant_id != validated_tenant_id:
        raise ValueError("readiness publication tenant_id does not match")
    if publication.placement_generation != validated_generation:
        raise ValueError(
            "readiness publication placement_generation is stale or unexpected"
        )
    return publication


def _exact_utc(value: datetime) -> datetime:
    """The instant on the UTC timeline as a plain ``datetime``.

    Callers pass a value already pinned by ``_pin_datetime`` (a plain ``datetime`` with a fixed offset), so
    this read is the only one the compared instant depends on; the built-in accessors are used so a
    ``datetime`` subclass (an overridden ``astimezone`` or field) cannot change it either.
    """

    offset = datetime.utcoffset(value)
    if offset is None:
        raise ValueError("a timezone-aware datetime is required")
    fields = [part.__get__(value, datetime) for part in _DATETIME_PARTS]
    try:
        return (datetime(*fields) - offset).replace(tzinfo=timezone.utc)
    except OverflowError as error:
        raise ValueError("datetime is outside the supported range") from error


def capability_is_ready(
    observation: CapabilityReadiness,
    *,
    evaluated_at: datetime,
    max_probe_age_seconds: SafeCounter,
) -> bool:
    """Evaluate explicit capability evidence at a supplied time and age limit.

    Structural validation cannot prove that the represented probe occurred.
    """

    observation = _validated_observation_snapshot(observation)
    # The caller's timestamp is read once (offset and fields) before validation, and the pinned copy is
    # what is validated and compared: a zone that answers differently per read cannot split the two.
    validated_evaluation_time = _AWARE_DATETIME_ADAPTER.validate_python(
        _pin_datetime(evaluated_at), strict=True
    )
    validated_max_age = _SAFE_COUNTER_ADAPTER.validate_python(
        max_probe_age_seconds, strict=True
    )
    # Aware datetimes that share one tzinfo object are compared and subtracted
    # in wall time, which misorders and mis-ages values around a daylight-saving
    # fold or gap.  Order and age are measured on the absolute (UTC) timeline.
    evaluated_utc = _exact_utc(validated_evaluation_time)
    observed_utc = _exact_utc(observation.observed_at)
    if evaluated_utc < observed_utc:
        raise ValueError("evaluated_at cannot precede observed_at")
    age_seconds = (evaluated_utc - observed_utc).total_seconds()
    return observation.status == "healthy" and age_seconds <= validated_max_age


__all__ = [
    "CapabilityReadiness",
    "CapabilityStatus",
    "ReadinessCapability",
    "TenantReadinessPublication",
    "capability_is_ready",
    "validate_readiness_publication",
]
