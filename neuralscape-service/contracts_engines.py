"""Engine capability declarations that keep operational facts distinct.

A manifest describes an implementation; it does not select a provider, grant
access, or prove current usability. Qualification references make a claim
reviewable but are not themselves proof that the referenced evidence is valid.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, NamedTuple, TypeVar

from pydantic import BaseModel, TypeAdapter, model_validator

from contracts_common import ContractModel, OpaqueId, VersionedContract
from contracts_references import ReferenceHandle


_ContractT = TypeVar("_ContractT", bound=BaseModel)
_MAX_SNAPSHOT_DEPTH = 256
_OPAQUE_ID_ADAPTER = TypeAdapter(OpaqueId)
_MODEL_DICT_DESCRIPTOR = BaseModel.__dict__["__dict__"]
_MODEL_FIELDS_SET_DESCRIPTOR = BaseModel.__dict__["__pydantic_fields_set__"]
_MODEL_EXTRAS_DESCRIPTOR = BaseModel.__dict__["__pydantic_extra__"]
_MISSING_MODEL_STORAGE = object()


class _FrozenModelState(NamedTuple):
    """Callback-free entry state for one natively reachable model."""

    owner: BaseModel
    storage_is_dict: bool
    stored_items: tuple[tuple[Any, Any], ...]
    fields_set_is_set: bool
    fields_set_members: tuple[Any, ...]
    extras_state: str
    extra_items: tuple[tuple[Any, Any], ...]


def _model_storage(descriptor: Any, value: BaseModel) -> Any:
    """Read a BaseModel-owned store without subclass attribute dispatch."""

    try:
        return descriptor.__get__(value, BaseModel)
    except AttributeError:
        return _MISSING_MODEL_STORAGE


def _native_set_members(value: set[Any] | frozenset[Any]) -> tuple[Any, ...]:
    """Capture concrete set members without subclass iteration hooks."""

    if isinstance(value, set):
        return tuple(set.__iter__(value))
    return tuple(frozenset.__iter__(value))


def _freeze_native_model_graph(
    value: Any,
    frozen_models: dict[int, _FrozenModelState],
    discovered: dict[int, Any],
) -> None:
    """Freeze models reachable through native container edges without callbacks."""

    pending = [value]
    while pending:
        current = pending.pop()
        if not isinstance(current, (BaseModel, dict, list, tuple)):
            continue
        identity = id(current)
        if identity in discovered and discovered[identity] is current:
            continue
        discovered[identity] = current

        if isinstance(current, BaseModel):
            frozen = frozen_models.get(identity)
            if frozen is not None and frozen.owner is current:
                continue

            storage = _model_storage(_MODEL_DICT_DESCRIPTOR, current)
            fields_set_value = _model_storage(
                _MODEL_FIELDS_SET_DESCRIPTOR,
                current,
            )
            extras_value = _model_storage(_MODEL_EXTRAS_DESCRIPTOR, current)
            storage_is_dict = isinstance(storage, dict)
            fields_set_is_set = isinstance(fields_set_value, (set, frozenset))
            stored_items = (
                tuple(dict.items(storage)) if storage_is_dict else ()
            )
            fields_set_members = (
                _native_set_members(fields_set_value) if fields_set_is_set else ()
            )
            if extras_value is _MISSING_MODEL_STORAGE:
                extras_state = "missing"
                extra_items: tuple[tuple[Any, Any], ...] = ()
            elif extras_value is None:
                extras_state = "none"
                extra_items = ()
            elif type(extras_value) is dict:
                extras_state = "dict"
                extra_items = tuple(dict.items(extras_value))
            else:
                extras_state = "malformed"
                extra_items = ()
            frozen = _FrozenModelState(
                owner=current,
                storage_is_dict=storage_is_dict,
                stored_items=stored_items,
                fields_set_is_set=fields_set_is_set,
                fields_set_members=fields_set_members,
                extras_state=extras_state,
                extra_items=extra_items,
            )
            frozen_models[identity] = frozen
            pending.extend(item for _, item in stored_items)
        elif isinstance(current, dict):
            pending.extend(item for _, item in tuple(dict.items(current)))
        elif isinstance(current, list):
            pending.extend(tuple(list.__iter__(current)))
        else:
            pending.extend(tuple(tuple.__iter__(current)))


def _normalize_field_names(
    names: tuple[Any, ...],
    *,
    location: str,
    storage_kind: str,
) -> tuple[str, ...]:
    """Normalize string-subclass names without invoking their hooks."""

    normalized: list[str] = []
    seen: set[str] = set()
    for candidate in names:
        if not isinstance(candidate, str):
            raise ValueError(
                f"malformed contract {storage_kind} at {location}: "
                "field names must be strings"
            )
        name = str.__str__(candidate)
        if name in seen:
            raise ValueError(
                f"malformed contract {storage_kind} at {location}: "
                "field names collide after normalization"
            )
        seen.add(name)
        normalized.append(name)
    return tuple(normalized)


def _normalize_field_items(
    items: tuple[tuple[Any, Any], ...],
    *,
    location: str,
    storage_kind: str,
) -> tuple[tuple[str, Any], ...]:
    """Attach normalized names to values captured from native storage."""

    names = _normalize_field_names(
        tuple(name for name, _ in items),
        location=location,
        storage_kind=storage_kind,
    )
    return tuple(
        (name, item)
        for name, (_, item) in zip(names, items)
    )


def _snapshot_native_value(
    value: Any,
    *,
    active: set[int],
    frozen_models: dict[int, _FrozenModelState],
    discovered: dict[int, Any],
    depth: int,
    location: str,
) -> Any:
    """Copy a contract graph without discarding unchecked model state."""

    if depth > _MAX_SNAPSHOT_DEPTH:
        raise ValueError(f"contract graph nesting exceeds the limit at {location}")

    is_container = isinstance(value, (BaseModel, dict, list, tuple))
    identity = id(value)
    if is_container:
        _freeze_native_model_graph(value, frozen_models, discovered)
        if identity in active:
            raise ValueError(f"cyclic contract graph at {location}")
        active.add(identity)

    try:
        if isinstance(value, BaseModel):
            frozen = frozen_models[identity]
            if not frozen.storage_is_dict or not frozen.fields_set_is_set:
                raise ValueError(f"malformed contract model at {location}")
            if frozen.extras_state == "malformed":
                raise ValueError(f"malformed contract extras at {location}")
            stored_items = _normalize_field_items(
                frozen.stored_items,
                location=location,
                storage_kind="model",
            )
            fields_set_names = _normalize_field_names(
                frozen.fields_set_members,
                location=location,
                storage_kind="model",
            )
            extra_items = _normalize_field_items(
                frozen.extra_items,
                location=location,
                storage_kind="extras",
            )
            declared_names = tuple(type(value).model_fields)
            declared = set(declared_names)
            stored = {name for name, _ in stored_items}
            fields_set = set(fields_set_names)
            extras = {name for name, _ in extra_items}
            duplicated = extras & (stored | declared)
            if duplicated:
                names = ", ".join(sorted(duplicated))
                raise ValueError(
                    f"malformed contract extras at {location}: "
                    f"duplicate field(s): {names}"
                )
            undeclared = (stored | fields_set | extras) - declared
            if undeclared:
                names = ", ".join(sorted(undeclared))
                raise ValueError(
                    f"undeclared contract field(s) at {location}: {names}"
                )
            stored_by_name = dict(stored_items)
            stored_values = tuple(
                (name, stored_by_name[name])
                for name in declared_names
                if name in stored_by_name
            )
            return {
                name: _snapshot_native_value(
                    item,
                    active=active,
                    frozen_models=frozen_models,
                    discovered=discovered,
                    depth=depth + 1,
                    location=f"{location}.{name}",
                )
                for name, item in stored_values
            }
        if isinstance(value, dict):
            return {
                key: _snapshot_native_value(
                    item,
                    active=active,
                    frozen_models=frozen_models,
                    discovered=discovered,
                    depth=depth + 1,
                    location=f"{location}[key]",
                )
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [
                _snapshot_native_value(
                    item,
                    active=active,
                    frozen_models=frozen_models,
                    discovered=discovered,
                    depth=depth + 1,
                    location=f"{location}[{index}]",
                )
                for index, item in enumerate(value)
            ]
        if isinstance(value, tuple):
            return tuple(
                _snapshot_native_value(
                    item,
                    active=active,
                    frozen_models=frozen_models,
                    discovered=discovered,
                    depth=depth + 1,
                    location=f"{location}[{index}]",
                )
                for index, item in enumerate(value)
            )
        return value
    finally:
        if is_container:
            active.remove(identity)


def _validated_contract_snapshot(
    value: Any,
    expected_type: type[_ContractT],
    *,
    label: str,
) -> _ContractT:
    """Return a fresh, fully validated closed-contract snapshot."""

    if not isinstance(value, expected_type):
        raise TypeError(f"{label} must be a {expected_type.__name__}")
    frozen_models: dict[int, _FrozenModelState] = {}
    discovered: dict[int, Any] = {}
    _freeze_native_model_graph(value, frozen_models, discovered)
    return _validated_contract_snapshot_from_frozen(
        value,
        expected_type,
        label=label,
        frozen_models=frozen_models,
        discovered=discovered,
    )


def _validated_contract_snapshot_from_frozen(
    value: Any,
    expected_type: type[_ContractT],
    *,
    label: str,
    frozen_models: dict[int, _FrozenModelState],
    discovered: dict[int, Any],
) -> _ContractT:
    """Validate using entry state already frozen for the enclosing boundary."""

    if not isinstance(value, expected_type):
        raise TypeError(f"{label} must be a {expected_type.__name__}")
    try:
        native = _snapshot_native_value(
            value,
            active=set(),
            frozen_models=frozen_models,
            discovered=discovered,
            depth=0,
            location=label,
        )
    except RecursionError as exc:
        raise ValueError(f"contract graph nesting is invalid at {label}") from exc
    return expected_type.model_validate(native)


def _validated_contract_tuple(
    values: Any,
    expected_type: type[_ContractT],
    *,
    label: str,
) -> tuple[_ContractT, ...]:
    """Validate a tuple argument without normalizing another container type."""

    if not isinstance(values, tuple):
        raise TypeError(f"{label} must be a tuple")
    frozen_models: dict[int, _FrozenModelState] = {}
    discovered: dict[int, Any] = {}
    _freeze_native_model_graph(values, frozen_models, discovered)
    return _validated_contract_tuple_from_frozen(
        values,
        expected_type,
        label=label,
        frozen_models=frozen_models,
        discovered=discovered,
    )


def _validated_contract_tuple_from_frozen(
    values: Any,
    expected_type: type[_ContractT],
    *,
    label: str,
    frozen_models: dict[int, _FrozenModelState],
    discovered: dict[int, Any],
) -> tuple[_ContractT, ...]:
    """Validate a tuple using entry state frozen for the enclosing boundary."""

    if not isinstance(values, tuple):
        raise TypeError(f"{label} must be a tuple")
    return tuple(
        _validated_contract_snapshot_from_frozen(
            value,
            expected_type,
            label=f"{label}[{index}]",
            frozen_models=frozen_models,
            discovered=discovered,
        )
        for index, value in enumerate(values)
    )


class EngineAdapterKind(str, Enum):
    """The architectural seam implemented by an engine adapter."""

    VECTOR_STORE = "vector_store"
    GRAPH_STORE = "graph_store"
    MEMORY_ENGINE = "memory_engine"
    INFERENCE = "inference"


class CapabilityHealth(str, Enum):
    """Observed health of one configured operation."""

    UNKNOWN = "unknown"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class QualificationClaim(ContractModel):
    """Versioned references supporting a qualification claim for one operation.

    Validation preserves the claim's identity. It does not resolve the
    references or prove that the evidence qualifies the operation.
    """

    operation: OpaqueId
    profile_reference: ReferenceHandle
    profile_version: OpaqueId
    evidence_reference: ReferenceHandle
    evidence_version: OpaqueId


class CapabilityOperationState(ContractModel):
    """Independent deployment and qualification facts for one operation."""

    operation: OpaqueId
    supported: bool
    configured: bool
    health: CapabilityHealth
    qualification: QualificationClaim | None

    @model_validator(mode="after")
    def validate_state(self) -> "CapabilityOperationState":
        if self.configured and not self.supported:
            raise ValueError("an unsupported operation cannot be configured")
        if self.qualification is not None and not self.supported:
            raise ValueError("an unsupported operation cannot be qualified")
        if (
            self.qualification is not None
            and self.qualification.operation != self.operation
        ):
            raise ValueError("qualification operation must match the capability operation")
        if self.health is not CapabilityHealth.UNKNOWN and not self.configured:
            raise ValueError("an unconfigured operation must have unknown health")
        return self

    def has_declared_prerequisites(self) -> bool:
        """Return whether structural dispatch prerequisites are declared.

        This does not resolve the qualification evidence and therefore is not a
        runtime authorization or proof of qualification.
        """

        state = _validated_contract_snapshot(
            self,
            CapabilityOperationState,
            label="capability operation state",
        )
        return (
            state.supported
            and state.configured
            and state.health is CapabilityHealth.HEALTHY
            and state.qualification is not None
        )


class CapabilityManifest(VersionedContract):
    """A versioned, operation-specific capability declaration."""

    implementation_id: OpaqueId
    implementation_version: OpaqueId
    adapter_kind: EngineAdapterKind
    contract_schema_versions: tuple[OpaqueId, ...]
    operations: tuple[CapabilityOperationState, ...]

    @model_validator(mode="after")
    def validate_unique_values(self) -> "CapabilityManifest":
        if not self.contract_schema_versions:
            raise ValueError("at least one contract schema version is required")
        if len(set(self.contract_schema_versions)) != len(self.contract_schema_versions):
            raise ValueError("contract schema versions must be unique")
        operation_names = [operation.operation for operation in self.operations]
        if len(set(operation_names)) != len(operation_names):
            raise ValueError("operation declarations must be unique")
        return self

    def operation_state(self, operation: str) -> CapabilityOperationState | None:
        """Return the exact declared operation, without aliases or approximation."""

        operation = _OPAQUE_ID_ADAPTER.validate_python(operation, strict=True)
        manifest = _validated_contract_snapshot(
            self,
            CapabilityManifest,
            label="capability manifest",
        )
        return next(
            (state for state in manifest.operations if state.operation == operation),
            None,
        )


class CapabilityRequirement(ContractModel):
    """Exact contract and qualification profile required for an operation."""

    operation: OpaqueId
    contract_schema_version: OpaqueId
    require_configured: bool
    require_healthy: bool
    qualification_profile_reference: ReferenceHandle | None
    qualification_profile_version: OpaqueId | None

    @model_validator(mode="after")
    def validate_qualification_requirement(self) -> "CapabilityRequirement":
        supplied = (
            self.qualification_profile_reference is not None,
            self.qualification_profile_version is not None,
        )
        if supplied[0] != supplied[1]:
            raise ValueError("qualification profile reference and version are paired")
        return self


class CapabilityViolation(ContractModel):
    """A stable explanation of one unmet capability requirement."""

    operation: OpaqueId
    fact: OpaqueId


def _operation_state_lookup(
    manifest: CapabilityManifest,
) -> dict[str, CapabilityOperationState]:
    """Build the exact operation index for one already-validated manifest."""

    return {state.operation: state for state in manifest.operations}


def validate_capability_requirements(
    manifest: CapabilityManifest,
    requirements: tuple[CapabilityRequirement, ...],
) -> tuple[CapabilityViolation, ...]:
    """Return structurally unmet facts without treating references as proof.

    An empty result means the declarations agree. Runtime dispatch must still
    resolve and evaluate the referenced qualification evidence.
    """

    if not isinstance(manifest, CapabilityManifest):
        raise TypeError("capability manifest must be a CapabilityManifest")

    requirements_is_tuple = isinstance(requirements, tuple)
    frozen_models: dict[int, _FrozenModelState] = {}
    discovered: dict[int, Any] = {}
    _freeze_native_model_graph(manifest, frozen_models, discovered)
    if requirements_is_tuple:
        _freeze_native_model_graph(requirements, frozen_models, discovered)

    manifest = _validated_contract_snapshot_from_frozen(
        manifest,
        CapabilityManifest,
        label="capability manifest",
        frozen_models=frozen_models,
        discovered=discovered,
    )
    requirements = _validated_contract_tuple_from_frozen(
        requirements,
        CapabilityRequirement,
        label="capability requirements",
        frozen_models=frozen_models,
        discovered=discovered,
    )
    states_by_operation = _operation_state_lookup(manifest)

    violations: list[CapabilityViolation] = []
    for requirement in requirements:
        state = states_by_operation.get(requirement.operation)
        if state is None or not state.supported:
            violations.append(
                CapabilityViolation(operation=requirement.operation, fact="supported")
            )
            continue
        if requirement.contract_schema_version not in manifest.contract_schema_versions:
            violations.append(
                CapabilityViolation(
                    operation=requirement.operation,
                    fact="contract_schema_version",
                )
            )
        if requirement.require_configured and not state.configured:
            violations.append(
                CapabilityViolation(operation=requirement.operation, fact="configured")
            )
        if requirement.require_healthy and state.health is not CapabilityHealth.HEALTHY:
            violations.append(
                CapabilityViolation(operation=requirement.operation, fact="healthy")
            )
        if requirement.qualification_profile_reference is not None:
            claim = state.qualification
            if claim is None:
                violations.append(
                    CapabilityViolation(operation=requirement.operation, fact="qualified")
                )
            elif (
                claim.profile_reference != requirement.qualification_profile_reference
                or claim.profile_version != requirement.qualification_profile_version
            ):
                violations.append(
                    CapabilityViolation(
                        operation=requirement.operation,
                        fact="qualification_profile",
                    )
                )
    return tuple(violations)


__all__ = [
    "CapabilityHealth",
    "CapabilityManifest",
    "CapabilityOperationState",
    "CapabilityRequirement",
    "CapabilityViolation",
    "EngineAdapterKind",
    "QualificationClaim",
    "validate_capability_requirements",
]
