"""Capability-specific readiness observations and publication validation.

Readiness is evidence about a named capability.  It is deliberately not
derived from process liveness, object construction, or another capability.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Literal

from pydantic import AwareDatetime, BaseModel, TypeAdapter, model_validator

from contracts_common import OpaqueId, SafeCounter, VersionedContract


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
_PYDANTIC_DICT_DESCRIPTOR = BaseModel.__dict__["__dict__"]
_PYDANTIC_EXTRA_DESCRIPTOR = BaseModel.__dict__["__pydantic_extra__"]


def _model_extra_storage(value: BaseModel) -> object:
    """Read Pydantic-owned extra storage without model instance dispatch."""

    try:
        return _PYDANTIC_EXTRA_DESCRIPTOR.__get__(value, type(value))
    except AttributeError:
        return None


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
    """Copy a model/native graph without dropping stored undeclared fields."""

    frozen_models: dict[int, tuple[object, bool, object]] = {}
    frozen_containers: dict[int, object] = {}
    discovered: set[int] = set()

    def freeze_native_graph(item: object) -> None:
        """Discover model state and exact-container edges without callbacks."""

        item_type = type(item)
        if not (
            issubclass(item_type, BaseModel)
            or issubclass(item_type, dict)
            or issubclass(item_type, list)
            or issubclass(item_type, tuple)
        ):
            return

        identity = id(item)
        if identity in discovered:
            return
        discovered.add(identity)

        if issubclass(item_type, BaseModel):
            stored_entries = tuple(
                dict.items(_PYDANTIC_DICT_DESCRIPTOR.__get__(item, BaseModel))
            )
            extras = _model_extra_storage(item)
            if extras is not None and issubclass(type(extras), dict):
                frozen_extras: object = tuple(dict.items(extras))
                extras_are_native = True
            else:
                frozen_extras = extras
                extras_are_native = False
            frozen_models[identity] = (
                stored_entries,
                extras_are_native,
                frozen_extras,
            )
            for _name, field_value in stored_entries:
                freeze_native_graph(field_value)
            return

        if issubclass(item_type, dict):
            entries = tuple(dict.items(item))
            if item_type is dict:
                frozen_containers[identity] = entries
            for _key, field_value in entries:
                freeze_native_graph(field_value)
            return

        if issubclass(item_type, list):
            items = tuple(list.__iter__(item))
            if item_type is list:
                frozen_containers[identity] = items
        else:
            items = tuple(tuple.__iter__(item))
            if item_type is tuple:
                frozen_containers[identity] = items
        for child in items:
            freeze_native_graph(child)

    freeze_native_graph(value)
    active: set[int] = set()

    def rebuild(item: object) -> object:
        item_type = type(item)
        if not (
            issubclass(item_type, BaseModel)
            or issubclass(item_type, dict)
            or issubclass(item_type, list)
            or issubclass(item_type, tuple)
        ):
            return item

        identity = id(item)
        if identity not in discovered:
            freeze_native_graph(item)
        if identity in active:
            raise ValueError("contract input graph must be acyclic")
        active.add(identity)
        try:
            if isinstance(item, BaseModel):
                stored_entries, extras_are_native, frozen_extras = frozen_models[
                    identity
                ]
                declared = item_type.model_fields
                if extras_are_native:
                    extra_entries = frozen_extras
                    observed_extra_length = len(extra_entries)
                    iterated_extra_names = tuple(
                        name for name, _field_value in extra_entries
                    )
                elif frozen_extras is None:
                    observed_extra_length = 0
                    iterated_extra_names = ()
                    extra_entries = ()
                elif not isinstance(frozen_extras, Mapping):
                    raise ValueError("contract extra storage must be a mapping")
                else:
                    observed_extra_length = len(frozen_extras)
                    iterated_extra_names = tuple(name for name in frozen_extras)
                    extra_entries = tuple(
                        entry for entry in frozen_extras.items()
                    )
                stored_names = {name for name, _field_value in stored_entries}
                undeclared = stored_names - declared.keys()
                if (
                    undeclared
                    or observed_extra_length
                    or iterated_extra_names
                    or extra_entries
                ):
                    raise ValueError("contract input contains undeclared fields")
                return {
                    name: rebuild(field_value)
                    for name, field_value in stored_entries
                }
            if isinstance(item, dict):
                entries = (
                    frozen_containers[identity]
                    if item_type is dict
                    else item.items()
                )
                return {
                    key: rebuild(field_value)
                    for key, field_value in entries
                }
            if isinstance(item, list):
                items = (
                    frozen_containers[identity]
                    if item_type is list
                    else item
                )
                return [rebuild(child) for child in items]
            items = (
                frozen_containers[identity]
                if item_type is tuple
                else item
            )
            return tuple(rebuild(child) for child in items)
        finally:
            active.remove(identity)

    return rebuild(value)


def _validated_observation_snapshot(
    observation: CapabilityReadiness,
) -> CapabilityReadiness:
    if not isinstance(observation, CapabilityReadiness):
        raise TypeError("observation must be a CapabilityReadiness")
    return CapabilityReadiness.model_validate(_snapshot_closed_graph(observation))


def _validated_publication_snapshot(
    publication: TenantReadinessPublication,
) -> TenantReadinessPublication:
    if not isinstance(publication, TenantReadinessPublication):
        raise TypeError("publication must be a TenantReadinessPublication")
    return TenantReadinessPublication.model_validate(
        _snapshot_closed_graph(publication)
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
    validated_evaluation_time = _AWARE_DATETIME_ADAPTER.validate_python(
        evaluated_at, strict=True
    )
    validated_max_age = _SAFE_COUNTER_ADAPTER.validate_python(
        max_probe_age_seconds, strict=True
    )
    if validated_evaluation_time < observation.observed_at:
        raise ValueError("evaluated_at cannot precede observed_at")
    age_seconds = (
        validated_evaluation_time - observation.observed_at
    ).total_seconds()
    return observation.status == "healthy" and age_seconds <= validated_max_age


__all__ = [
    "CapabilityReadiness",
    "CapabilityStatus",
    "ReadinessCapability",
    "TenantReadinessPublication",
    "capability_is_ready",
    "validate_readiness_publication",
]
