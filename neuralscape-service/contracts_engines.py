"""Engine capability declarations that keep operational facts distinct.

A manifest describes an implementation; it does not select a provider, grant
access, or prove current usability. Qualification references make a claim
reviewable but are not themselves proof that the referenced evidence is valid.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Callable, Iterator, NamedTuple, TypeVar

from pydantic import BaseModel, TypeAdapter, model_validator

from contracts_common import ContractModel, OpaqueId, VersionedContract
from contracts_limits import DISCOVERY_BUDGET_MESSAGE, DiscoveryBudget
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


class _FrozenContainerState(NamedTuple):
    """Callback-free entry edges for one exact native container."""

    owner: Any
    kind: str
    entries: tuple[Any, ...]


def _model_storage(descriptor: Any, value: BaseModel) -> Any:
    """Read a BaseModel-owned store without subclass attribute dispatch."""

    try:
        return descriptor.__get__(value, BaseModel)
    except AttributeError:
        return _MISSING_MODEL_STORAGE


def _native_set_members(value: set[Any] | frozenset[Any]) -> tuple[Any, ...]:
    """Capture concrete set members without subclass iteration hooks."""

    if issubclass(type(value), set):
        return tuple(set.__iter__(value))
    return tuple(frozenset.__iter__(value))


def _native_len(value: Any) -> int:
    """Length of a concrete native container without subclass hooks."""

    value_type = type(value)
    if issubclass(value_type, dict):
        return dict.__len__(value)
    if issubclass(value_type, list):
        return list.__len__(value)
    if issubclass(value_type, tuple):
        return tuple.__len__(value)
    if issubclass(value_type, set):
        return set.__len__(value)
    return frozenset.__len__(value)


def _freeze_native_model_graph(
    value: Any,
    frozen_models: dict[int, _FrozenModelState],
    frozen_containers: dict[int, _FrozenContainerState],
    discovered: dict[int, Any],
    budget: DiscoveryBudget,
    depth: int = 0,
) -> None:
    """Freeze models reachable through native container edges without callbacks.

    One ``budget`` is shared by every root of a validation call.  Nodes are
    charged once by identity and every captured entry is charged as an edge
    *before* it is copied, so an oversized graph is rejected with the stable
    budget error before any public callback runs and without materializing it.
    The depth frontier carries ``(object, depth, authoritative)``: model fields
    and exact built-in containers are authoritative; native backing behind a
    container subclass is hidden and clears the flag.  An authoritative node at
    the snapshot depth limit is captured but its children are not enqueued, so
    the ordered snapshot still reports the established nesting error.
    """

    pending: list[tuple[Any, int, bool]] = [(value, depth, True)]
    while pending:
        current, depth, authoritative = pending.pop()
        current_type = type(current)
        if not issubclass(current_type, (BaseModel, dict, list, tuple)):
            continue
        identity = id(current)
        if not budget.visit(current, depth, authoritative):
            continue
        discovered[identity] = current
        enqueue = not (authoritative and depth >= _MAX_SNAPSHOT_DEPTH)
        children: tuple[Any, ...] = ()
        children_authoritative = authoritative

        if issubclass(current_type, BaseModel):
            frozen = frozen_models.get(identity)
            if frozen is None or frozen.owner is not current:
                storage = _model_storage(_MODEL_DICT_DESCRIPTOR, current)
                fields_set_value = _model_storage(
                    _MODEL_FIELDS_SET_DESCRIPTOR,
                    current,
                )
                extras_value = _model_storage(_MODEL_EXTRAS_DESCRIPTOR, current)
                storage_is_dict = issubclass(type(storage), dict)
                fields_set_is_set = issubclass(
                    type(fields_set_value),
                    (set, frozenset),
                )
                extras_are_dict = type(extras_value) is dict
                budget.charge_edge(
                    (_native_len(storage) if storage_is_dict else 0)
                    + (_native_len(fields_set_value) if fields_set_is_set else 0)
                    + (_native_len(extras_value) if extras_are_dict else 0)
                )
                stored_items = (
                    tuple(dict.items(storage)) if storage_is_dict else ()
                )
                fields_set_members = (
                    _native_set_members(fields_set_value)
                    if fields_set_is_set
                    else ()
                )
                if extras_value is _MISSING_MODEL_STORAGE:
                    extras_state = "missing"
                    extra_items: tuple[tuple[Any, Any], ...] = ()
                elif extras_value is None:
                    extras_state = "none"
                    extra_items = ()
                elif extras_are_dict:
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
            else:
                # Re-expansion through a shallower authoritative path: the
                # captured edges are charged again, nothing is re-read.
                budget.charge_edge(
                    len(frozen.stored_items)
                    + len(frozen.fields_set_members)
                    + len(frozen.extra_items)
                )
            children = tuple(item for _, item in frozen.stored_items)
        else:
            if issubclass(current_type, dict):
                kind, exact = "dict", current_type is dict
                state = frozen_containers.get(identity)
                if state is None or state.owner is not current:
                    budget.charge_edge(_native_len(current))
                    entries: tuple[Any, ...] = tuple(dict.items(current))
                else:
                    budget.charge_edge(len(state.entries))
                    entries = state.entries
                children = tuple(item for _, item in entries)
            else:
                if issubclass(current_type, list):
                    kind, exact = "list", current_type is list
                    reader = list.__iter__
                else:
                    kind, exact = "tuple", current_type is tuple
                    reader = tuple.__iter__
                state = frozen_containers.get(identity)
                if state is None or state.owner is not current:
                    budget.charge_edge(_native_len(current))
                    entries = tuple(reader(current))
                else:
                    budget.charge_edge(len(state.entries))
                    entries = state.entries
                children = entries
            if exact:
                frozen_containers[identity] = _FrozenContainerState(
                    owner=current,
                    kind=kind,
                    entries=entries,
                )
            else:
                # Native backing behind a subclass is hidden: it is accounted
                # for but never authoritative.
                children_authoritative = False

        if enqueue:
            pending.extend((child, depth + 1, children_authoritative) for child in children)


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
        if not issubclass(type(candidate), str):
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


def _charged_entries(read: Callable[[], Any], budget: DiscoveryBudget) -> Iterator[Any]:
    """Observe a public view: one charge for the observation itself (an empty view is
    not free), taken before the view is read, then one charge per entry observed."""

    budget.charge_edge()
    for entry in read():
        budget.charge_edge()
        yield entry


def _snapshot_native_value(
    value: Any,
    *,
    active: set[int],
    frozen_models: dict[int, _FrozenModelState],
    frozen_containers: dict[int, _FrozenContainerState],
    discovered: dict[int, Any],
    budget: DiscoveryBudget,
    memo: dict[int, tuple[Any, Any, int]],
    depth: int,
    location: str,
) -> tuple[Any, int, bool]:
    """Copy a contract graph without discarding unchecked model state.

    Returns ``(copy, height, memoizable)``.  A subtree built only from native
    storage (models and exact built-in containers) is a pure function of the
    entry state frozen for the call, so it is copied once per object identity
    and shared by every path that reaches it; the work is bounded by the
    distinct objects and edges rather than the number of paths.  ``height`` is
    the deepest descendant relative to this value, so the nesting limit still
    applies to every path.  A subtree that observed a public view (a container
    subclass) is never memoized: each edge that reaches such a view observes
    it again, and every entry observed is charged to the shared budget.
    """

    if depth > _MAX_SNAPSHOT_DEPTH:
        raise ValueError(f"contract graph nesting exceeds the limit at {location}")
    if depth > budget.tier.max_depth:
        raise ValueError(DISCOVERY_BUDGET_MESSAGE)  # the tier applies to every occurrence, not only to discovery

    value_type = type(value)
    is_container = issubclass(value_type, (BaseModel, dict, list, tuple))
    identity = id(value)
    if is_container:
        cached = memo.get(identity)
        if cached is not None and cached[0] is value:
            _, copied, height = cached
            if depth + height > _MAX_SNAPSHOT_DEPTH:
                raise ValueError(f"contract graph nesting exceeds the limit at {location}")
            if depth + height > budget.tier.max_depth:
                raise ValueError(DISCOVERY_BUDGET_MESSAGE)
            return copied, height, True
        # A child first exposed through a public view is captured now; it is
        # charged to the same envelope as the entry discovery.
        _freeze_native_model_graph(
            value,
            frozen_models,
            frozen_containers,
            discovered,
            budget,
            depth,
        )
        if identity in active:
            raise ValueError(f"cyclic contract graph at {location}")
        active.add(identity)

    height = 0
    memoizable = True
    own_cost = 0  # copying an exact node that cannot be reused is work the capture phase did not charge

    def child(item: Any, child_location: str) -> Any:
        nonlocal height, memoizable
        copied_item, item_height, item_memoizable = _snapshot_native_value(
            item,
            active=active,
            frozen_models=frozen_models,
            frozen_containers=frozen_containers,
            discovered=discovered,
            budget=budget,
            memo=memo,
            depth=depth + 1,
            location=child_location,
        )
        height = max(height, item_height + 1)
        memoizable = memoizable and item_memoizable
        return copied_item

    try:
        if issubclass(value_type, BaseModel):
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
            own_cost = 1 + len(stored_values)
            result: Any = {
                name: child(item, f"{location}.{name}") for name, item in stored_values
            }
        elif issubclass(value_type, dict):
            frozen = frozen_containers.get(identity)
            exact = (
                frozen is not None and frozen.owner is value and frozen.kind == "dict"
            )
            memoizable = exact
            if exact:
                own_cost = 1 + len(frozen.entries)
            result = {
                key: child(item, f"{location}[key]")
                for key, item in (
                    frozen.entries if exact else _charged_entries(value.items, budget)
                )
            }
        elif issubclass(value_type, (list, tuple)):
            is_list = issubclass(value_type, list)
            frozen = frozen_containers.get(identity)
            exact = (
                frozen is not None
                and frozen.owner is value
                and frozen.kind == ("list" if is_list else "tuple")
            )
            memoizable = exact
            if exact:
                own_cost = 1 + len(frozen.entries)
            source_items = frozen.entries if exact else _charged_entries(lambda: value, budget)
            items = [
                child(item, f"{location}[{index}]")
                for index, item in enumerate(source_items)
            ]
            result = items if is_list else tuple(items)
        else:
            return value, 0, True
        if is_container and memoizable:
            memo[identity] = (value, result, height)
        elif own_cost:
            budget.charge_edge(own_cost)  # every non-reused ancestor copy is charged
        return result, height, memoizable
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

    if not issubclass(type(value), expected_type):
        raise TypeError(f"{label} must be a {expected_type.__name__}")
    frozen_models: dict[int, _FrozenModelState] = {}
    frozen_containers: dict[int, _FrozenContainerState] = {}
    discovered: dict[int, Any] = {}
    budget = DiscoveryBudget()
    memo: dict[int, tuple[Any, Any, int]] = {}
    _freeze_native_model_graph(
        value,
        frozen_models,
        frozen_containers,
        discovered,
        budget,
    )
    return _validated_contract_snapshot_from_frozen(
        value,
        expected_type,
        label=label,
        frozen_models=frozen_models,
        frozen_containers=frozen_containers,
        discovered=discovered,
        budget=budget,
        memo=memo,
    )


def _validated_contract_snapshot_from_frozen(
    value: Any,
    expected_type: type[_ContractT],
    *,
    label: str,
    frozen_models: dict[int, _FrozenModelState],
    frozen_containers: dict[int, _FrozenContainerState],
    discovered: dict[int, Any],
    budget: DiscoveryBudget,
    memo: dict[int, tuple[Any, Any, int]],
) -> _ContractT:
    """Validate using entry state already frozen for the enclosing boundary."""

    if not issubclass(type(value), expected_type):
        raise TypeError(f"{label} must be a {expected_type.__name__}")
    try:
        native, _, _ = _snapshot_native_value(
            value,
            active=set(),
            frozen_models=frozen_models,
            frozen_containers=frozen_containers,
            discovered=discovered,
            budget=budget,
            memo=memo,
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

    if not issubclass(type(values), tuple):
        raise TypeError(f"{label} must be a tuple")
    frozen_models: dict[int, _FrozenModelState] = {}
    frozen_containers: dict[int, _FrozenContainerState] = {}
    discovered: dict[int, Any] = {}
    budget = DiscoveryBudget()
    memo: dict[int, tuple[Any, Any, int]] = {}
    _freeze_native_model_graph(
        values,
        frozen_models,
        frozen_containers,
        discovered,
        budget,
    )
    return _validated_contract_tuple_from_frozen(
        values,
        expected_type,
        label=label,
        frozen_models=frozen_models,
        frozen_containers=frozen_containers,
        discovered=discovered,
        budget=budget,
        memo=memo,
    )


def _validated_contract_tuple_from_frozen(
    values: Any,
    expected_type: type[_ContractT],
    *,
    label: str,
    frozen_models: dict[int, _FrozenModelState],
    frozen_containers: dict[int, _FrozenContainerState],
    discovered: dict[int, Any],
    budget: DiscoveryBudget,
    memo: dict[int, tuple[Any, Any, int]],
) -> tuple[_ContractT, ...]:
    """Validate a tuple using entry state frozen for the enclosing boundary."""

    if not issubclass(type(values), tuple):
        raise TypeError(f"{label} must be a tuple")
    return tuple(
        _validated_contract_snapshot_from_frozen(
            value,
            expected_type,
            label=f"{label}[{index}]",
            frozen_models=frozen_models,
            frozen_containers=frozen_containers,
            discovered=discovered,
            budget=budget,
            memo=memo,
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

    if not issubclass(type(manifest), CapabilityManifest):
        raise TypeError("capability manifest must be a CapabilityManifest")

    requirements_is_tuple = issubclass(type(requirements), tuple)
    frozen_models: dict[int, _FrozenModelState] = {}
    frozen_containers: dict[int, _FrozenContainerState] = {}
    discovered: dict[int, Any] = {}
    budget = DiscoveryBudget()  # one envelope across manifest and requirements
    memo: dict[int, tuple[Any, Any, int]] = {}  # one copy per object across both roots
    try:
        _freeze_native_model_graph(
            manifest,
            frozen_models,
            frozen_containers,
            discovered,
            budget,
        )
    except ValueError as exc:
        # A strict top-level type error outranks the budget; ordinary manifest
        # diagnostics keep their established priority over it.
        if not requirements_is_tuple and str(exc) == DISCOVERY_BUDGET_MESSAGE:
            raise TypeError("capability requirements must be a tuple") from None
        raise
    if requirements_is_tuple:
        _freeze_native_model_graph(
            requirements,
            frozen_models,
            frozen_containers,
            discovered,
            budget,
        )

    manifest = _validated_contract_snapshot_from_frozen(
        manifest,
        CapabilityManifest,
        label="capability manifest",
        frozen_models=frozen_models,
        frozen_containers=frozen_containers,
        discovered=discovered,
        budget=budget,
        memo=memo,
    )
    requirements = _validated_contract_tuple_from_frozen(
        requirements,
        CapabilityRequirement,
        label="capability requirements",
        frozen_models=frozen_models,
        frozen_containers=frozen_containers,
        discovered=discovered,
        budget=budget,
        memo=memo,
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
