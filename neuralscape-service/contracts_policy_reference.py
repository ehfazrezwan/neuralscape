"""Deterministic, side-effect-free reference policy evaluator."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from types import MappingProxyType
from typing import Any, NamedTuple, cast

from pydantic import BaseModel

from contracts_policy import (
    SUPPORTED_POLICY_ACTIONS,
    AccessPolicy,
    PolicyDecision,
    PolicyEvaluationInput,
    PolicyOutcome,
    PolicyReasonCode,
    PrincipalContext,
)
from contracts_limits import STANDARD, DiscoveryBudget, is_exact_type
from contracts_references import ReferenceHandle


_BASE_MODEL_DICT_DESCRIPTOR = vars(BaseModel)["__dict__"]
_BASE_MODEL_EXTRA_DESCRIPTOR = vars(BaseModel)["__pydantic_extra__"]
_BASE_MODEL_FIELDS_SET_DESCRIPTOR = vars(BaseModel)["__pydantic_fields_set__"]


class _CapturedFailure(NamedTuple):
    error: Exception


class _ModelInventory(NamedTuple):
    owner: BaseModel
    stored_storage: object
    stored_items: tuple[tuple[object, object], ...] | None
    extra_storage: object
    extra_items: tuple[tuple[object, object], ...] | None
    fields_set_storage: object
    fields_set_names: tuple[object, ...] | None


class _MappingEntry(NamedTuple):
    key: object
    value: object


class _MappingInventory(NamedTuple):
    owner: Mapping[object, object]
    entries: tuple[_MappingEntry | _CapturedFailure, ...] | _CapturedFailure


_TYPE_MRO = vars(type)["__mro__"]


def _inherits(cls: type, base: type) -> bool:
    """Whether ``base`` is on ``cls``'s own method resolution order, by identity.

    ``issubclass`` against an ABC also honours ``register()`` and ``__subclasshook__``, so a
    class that merely registered itself would be trusted as the real protocol.  The native MRO
    names only what the class actually inherits from.
    """

    return any(candidate is base for candidate in _TYPE_MRO.__get__(cls))


def _is_mapping(value: object) -> bool:
    """A real mapping: a dict, inherited from ``Mapping``, or the one read-only view that only registers."""

    kind = type(value)
    return kind is MappingProxyType or issubclass(kind, dict) or _inherits(kind, Mapping)


def _is_native_dict(value: object) -> bool:
    """Recognize dict storage without consulting an instance-level class view."""

    return issubclass(type(value), dict)


def _new_budget(budget: DiscoveryBudget | None) -> DiscoveryBudget:
    """One shared envelope per validation call; a fresh standard one when none is given."""

    return DiscoveryBudget(STANDARD) if budget is None else budget


def _precheck(budget: DiscoveryBudget, count: int) -> None:
    """Refuse a native copy that could not fit the remaining edge allowance.

    Nothing is charged here (the walk charges the edges it traverses); this only
    stops a huge native container from being copied before it is accounted.
    """

    if count > budget.tier.max_edges - budget.edges:
        budget.charge_edge(count)  # raises the stable budget error


def _mapping_items(
    value: Mapping[object, object],
) -> tuple[tuple[object, object], ...]:
    """Capture one mapping inventory without trusting native-dict overrides."""

    if _is_native_dict(value):
        return tuple(dict.items(value))
    return tuple(value.items())


def _fields_set_names(
    value: object, budget: DiscoveryBudget | None = None
) -> tuple[object, ...]:
    """Capture native set backing; any other iterable is charged per item it yields."""

    budget = _new_budget(budget)
    if issubclass(type(value), set):
        _precheck(budget, set.__len__(value))
        return tuple(set.__iter__(value))
    names: list[object] = []
    for item in cast(Iterable[object], value):
        budget.charge_edge()
        names.append(item)
    return tuple(names)


def _normalize_model_names(names: tuple[object, ...]) -> tuple[str, ...]:
    """Normalize model-owned string names without invoking subclass hooks."""

    normalized: list[str] = []
    seen: set[str] = set()
    for candidate in names:
        if not issubclass(type(candidate), str):
            raise ValueError("contract input model field names must be strings")
        name = str.__str__(candidate)
        if name in seen:
            raise ValueError("contract input contains colliding model field names")
        seen.add(name)
        normalized.append(name)
    return tuple(normalized)


def _normalize_model_items(
    items: tuple[tuple[object, object], ...],
) -> tuple[tuple[str, object], ...]:
    """Pair native model values with callback-free normalized names."""

    names = _normalize_model_names(tuple(name for name, _ in items))
    return tuple((name, item) for name, (_, item) in zip(names, items))


def _capture_model_store(descriptor: object, value: BaseModel) -> object:
    """Capture a model-owned slot while deferring its existing error order."""

    try:
        return descriptor.__get__(value, BaseModel)  # type: ignore[attr-defined]
    except Exception as error:
        return _CapturedFailure(error)


def _capture_model_inventory(
    value: BaseModel, budget: DiscoveryBudget | None = None
) -> _ModelInventory:
    """Capture native inventories without invoking supported fallback protocols."""

    budget = _new_budget(budget)
    stored_storage = _capture_model_store(_BASE_MODEL_DICT_DESCRIPTOR, value)
    if _is_native_dict(stored_storage):
        _precheck(budget, dict.__len__(stored_storage))
    stored_items = (
        tuple(dict.items(stored_storage))
        if _is_native_dict(stored_storage)
        else None
    )
    extra_storage = _capture_model_store(_BASE_MODEL_EXTRA_DESCRIPTOR, value)
    if _is_native_dict(extra_storage):
        _precheck(budget, dict.__len__(extra_storage))
    extra_items = (
        tuple(dict.items(extra_storage))
        if _is_native_dict(extra_storage)
        else (() if extra_storage is None else None)
    )
    fields_set_storage = _capture_model_store(
        _BASE_MODEL_FIELDS_SET_DESCRIPTOR,
        value,
    )
    if issubclass(type(fields_set_storage), set):
        _precheck(budget, set.__len__(fields_set_storage))
    fields_set_names = (
        tuple(set.__iter__(fields_set_storage))
        if issubclass(type(fields_set_storage), set)
        else None
    )
    return _ModelInventory(
        value,
        stored_storage,
        stored_items,
        extra_storage,
        extra_items,
        fields_set_storage,
        fields_set_names,
    )


def _validated_model_inventory(
    value: BaseModel,
    inventories: dict[int, _ModelInventory],
) -> _ModelInventory | None:
    """Return only an inventory owned by this exact model instance."""

    inventory = inventories.get(id(value))
    if inventory is not None and inventory.owner is not value:
        raise ValueError("contract input model inventory owner mismatch")
    return inventory


def _retain_mapping_owner(
    value: Mapping[object, object],
    owners: dict[int, Mapping[object, object]],
) -> None:
    """Retain and authenticate a mapping reached without public callbacks."""

    identity = id(value)
    owner = owners.get(identity)
    if owner is not None and owner is not value:
        raise ValueError("contract input mapping owner mismatch")
    owners[identity] = value


def _validated_mapping_inventory(
    value: Mapping[object, object],
    inventories: dict[int, _MappingInventory],
) -> _MappingInventory | None:
    """Return only an inventory owned by this exact mapping instance."""

    inventory = inventories.get(id(value))
    if inventory is not None and inventory.owner is not value:
        raise ValueError("contract input mapping inventory owner mismatch")
    return inventory


def _freeze_reachable_model_inventories(
    value: object,
    inventories: dict[int, _ModelInventory],
    visited_containers: set[int],
    mapping_owners: dict[int, Mapping[object, object]] | None = None,
    native_dicts: dict[int, _MappingInventory] | None = None,
    budget: DiscoveryBudget | None = None,
    depth: int = 0,
    authoritative: bool = True,
) -> None:
    """Freeze natively reachable models before callback-bearing traversal.

    Native dict entries are retained as they are first observed here, so the
    completion pass replays them instead of re-reading the live dict after any
    public callback has run (first observation, then replay).

    Every container, model and mapping is charged to the shared ``budget`` (one
    node by identity, one edge per entry, depth along the path), using native
    sizes only: no hook of the caller's objects runs for the accounting.  A
    container subclass walked through its native backing is not authoritative
    (its public view is); an object first reached deep is expanded again, from
    the retained inventory, when a shallower authoritative path reaches it.
    """

    if mapping_owners is None:
        mapping_owners = {}
    if native_dicts is None:
        native_dicts = {}
    budget = _new_budget(budget)

    # One stack frame per level (no helper closure): the depth limit then bounds
    # the interpreter stack as well as the walk.
    below = (inventories, visited_containers, mapping_owners, native_dicts, budget, depth + 1)

    if _inherits(type(value), BaseModel):
        if not budget.visit(value, depth, authoritative):
            return
        identity = id(value)
        inventory = _validated_model_inventory(value, inventories)
        if inventory is None:
            inventory = _capture_model_inventory(value, budget)
            inventories[identity] = inventory
        if inventory.stored_items is not None:
            budget.charge_edge(len(inventory.stored_items))
            for _, item in inventory.stored_items:
                _freeze_reachable_model_inventories(item, *below, authoritative)
        elif _is_mapping(inventory.stored_storage):
            _retain_mapping_owner(inventory.stored_storage, mapping_owners)
        if inventory.extra_items is not None:
            budget.charge_edge(len(inventory.extra_items))
            for _, item in inventory.extra_items:
                _freeze_reachable_model_inventories(item, *below, authoritative)
        elif _is_mapping(inventory.extra_storage):
            _retain_mapping_owner(inventory.extra_storage, mapping_owners)
        return

    if _is_native_dict(value):
        if not budget.visit(value, depth, authoritative):
            return
        identity = id(value)
        visited_containers.add(identity)
        existing = native_dicts.get(identity)
        if existing is not None:
            # Already observed: keep that snapshot (and the children frozen with
            # it); never replace it with a re-read taken after callbacks ran.
            if existing.owner is not value:
                raise ValueError("contract input mapping inventory owner mismatch")
            retained = cast(tuple[_MappingEntry, ...], existing.entries)
        else:
            _precheck(budget, dict.__len__(value))
            retained = tuple(_MappingEntry(key, item) for key, item in dict.items(value))
            native_dicts[identity] = _MappingInventory(value, retained)
        budget.charge_edge(len(retained))
        for entry in retained:
            _freeze_reachable_model_inventories(entry.value, *below, authoritative)
        return

    if _is_mapping(value):
        budget.visit(value, depth, True)  # its pairs are its public view: authoritative
        _retain_mapping_owner(value, mapping_owners)
        return

    if issubclass(type(value), list):
        if not budget.visit(value, depth, authoritative):
            return
        visited_containers.add(id(value))
        _precheck(budget, list.__len__(value))
        items = tuple(list.__iter__(value))
        budget.charge_edge(len(items))
        for item in items:
            _freeze_reachable_model_inventories(item, *below, authoritative and is_exact_type(value, list))
        return

    if issubclass(type(value), tuple):
        if not budget.visit(value, depth, authoritative):
            return
        visited_containers.add(id(value))
        _precheck(budget, tuple.__len__(value))
        items = tuple(tuple.__iter__(value))
        budget.charge_edge(len(items))
        for item in items:
            _freeze_reachable_model_inventories(item, *below, authoritative and is_exact_type(value, tuple))
        return


def _open_mapping_items(value: Mapping[object, object]) -> Iterator[object]:
    """The mapping's public pair iterator, opened but NOT consumed."""

    if _is_native_dict(value):
        return iter(tuple(dict.items(value)))
    return iter(value.items())


def _freeze_mapping_inventories(
    owners: dict[int, Mapping[object, object]],
    mapping_inventories: dict[int, _MappingInventory],
    model_inventories: dict[int, _ModelInventory],
    visited_containers: set[int],
    native_dicts: dict[int, _MappingInventory] | None = None,
    in_progress: set[int] | None = None,
    budget: DiscoveryBudget | None = None,
) -> None:
    """Observe each retained public mapping inventory exactly once.

    Each pair's reachable children are frozen BEFORE the pair iterator is
    advanced, so an iterator (for example a generator) cannot repair or alter an
    earlier entry's children after they were exposed.  Failures stay deferred:
    an error from opening or advancing the iterator replaces the whole
    inventory; the first malformed pair is terminal, and the rest of the
    iterator is still drained so a later iteration error keeps its precedence.

    A mapping that is not a native dict (a proxy, a custom mapping) exposed by a
    pair is captured, items frozen, at that same point, before the parent iterator
    advances, even if the entry walk had already queued it; merely queueing it
    would read its items after the parent's later steps could have changed what
    it exposes.  An inventory, once taken, is never replaced: an owner that was
    captured while another was in progress is skipped by the outer loop.
    """

    if in_progress is None:
        in_progress = set()
    budget = _new_budget(budget)
    while True:
        pending = tuple(
            owner
            for owner in owners.values()
            if id(owner) not in in_progress
            and _validated_mapping_inventory(owner, mapping_inventories) is None
        )
        if not pending:
            return

        for owner in pending:
            if _validated_mapping_inventory(owner, mapping_inventories) is not None:
                continue  # captured while an earlier owner was in progress: never read twice
            try:
                iterator = _open_mapping_items(owner)
            except Exception as error:
                mapping_inventories[id(owner)] = _MappingInventory(
                    owner,
                    _CapturedFailure(error),
                )
                continue

            entries: list[_MappingEntry | _CapturedFailure] = []
            failure: Exception | None = None
            terminal = False
            in_progress.add(id(owner))
            child_depth = (budget.authoritative_depth(owner) or 0) + 1
            while True:
                try:
                    raw_entry = next(iterator)
                except StopIteration:
                    break
                except Exception as error:
                    failure = error
                    break
                # Every pair the public iterator produces is an edge, including the
                # ones drained after a terminal pair, so an endless iterator is
                # refused with the budget error (never deferred as a pair failure).
                budget.charge_edge()
                if terminal:
                    continue  # drain only: nothing after the terminal pair is retained
                try:
                    key, item = raw_entry  # type: ignore[misc]
                except Exception as error:
                    entries.append(_CapturedFailure(error))
                    terminal = True
                    continue
                entries.append(_MappingEntry(key, item))
                # The mappings THIS pair exposes, whether or not the entry walk had
                # already queued them: any that has no inventory yet is captured now,
                # at its first exposure, before this iterator is advanced.
                exposed: dict[int, Mapping[object, object]] = {}
                _freeze_reachable_model_inventories(
                    item,
                    model_inventories,
                    visited_containers,
                    exposed,
                    native_dicts,
                    budget,
                    child_depth,
                )
                for found in exposed.values():
                    _retain_mapping_owner(found, owners)
                discovered = {
                    identity: found
                    for identity, found in exposed.items()
                    if identity not in in_progress
                    and _validated_mapping_inventory(found, mapping_inventories) is None
                }
                if discovered:
                    _freeze_mapping_inventories(
                        discovered,
                        mapping_inventories,
                        model_inventories,
                        visited_containers,
                        native_dicts,
                        in_progress,
                        budget,
                    )
            in_progress.discard(id(owner))
            mapping_inventories[id(owner)] = _MappingInventory(
                owner,
                _CapturedFailure(failure) if failure is not None else tuple(entries),
            )


def _freeze_mapping_entries(
    raw_entries: tuple[object, ...],
    owners: dict[int, Mapping[object, object]],
    model_inventories: dict[int, _ModelInventory],
    visited_containers: set[int],
    native_dicts: dict[int, _MappingInventory] | None = None,
    budget: DiscoveryBudget | None = None,
    depth: int = 1,
) -> tuple[_MappingEntry | _CapturedFailure, ...]:
    """Retain the successful prefix and first terminal pair failure."""

    budget = _new_budget(budget)
    budget.charge_edge(len(raw_entries))
    entries: list[_MappingEntry | _CapturedFailure] = []
    for raw_entry in raw_entries:
        try:
            key, item = raw_entry  # type: ignore[misc]
        except Exception as error:
            entries.append(_CapturedFailure(error))
            break
        entries.append(_MappingEntry(key, item))
        _freeze_reachable_model_inventories(
            item,
            model_inventories,
            visited_containers,
            owners,
            native_dicts,
            budget,
            depth,
        )
    return tuple(entries)


def _raise_captured_failure(value: object) -> object:
    if type(value) is _CapturedFailure:
        raise value.error
    return value


def _retained_mapping_entries(
    value: Mapping[object, object],
    mapping_inventories: dict[int, _MappingInventory],
    model_inventories: dict[int, _ModelInventory],
    budget: DiscoveryBudget | None = None,
) -> tuple[_MappingEntry | _CapturedFailure, ...]:
    """Return one authenticated public inventory, capturing it if newly seen."""

    budget = _new_budget(budget)

    inventory = _validated_mapping_inventory(value, mapping_inventories)
    if inventory is None:
        owners: dict[int, Mapping[object, object]] = {}
        _retain_mapping_owner(value, owners)
        _freeze_mapping_inventories(
            owners,
            mapping_inventories,
            model_inventories,
            set(),
            mapping_inventories,
            None,
            budget,
        )
        inventory = _validated_mapping_inventory(value, mapping_inventories)
        if inventory is None:
            raise ValueError("contract input mapping inventory is missing")
    return cast(
        tuple[_MappingEntry | _CapturedFailure, ...],
        _raise_captured_failure(inventory.entries),
    )


def _retained_mapping_items(
    value: Mapping[object, object],
    mapping_inventories: dict[int, _MappingInventory],
    model_inventories: dict[int, _ModelInventory],
    budget: DiscoveryBudget | None = None,
) -> tuple[tuple[object, object], ...]:
    """Rebuild model-owned items while retaining per-entry error priority."""

    items: list[tuple[object, object]] = []
    for entry in _retained_mapping_entries(
        value,
        mapping_inventories,
        model_inventories,
        budget,
    ):
        retained = cast(_MappingEntry, _raise_captured_failure(entry))
        items.append((retained.key, retained.value))
    return tuple(items)


def _native_entries(
    value: Mapping[object, object], budget: DiscoveryBudget
) -> tuple[_MappingEntry, ...]:
    """The entries of a native dict that discovery did not already retain."""

    _precheck(budget, dict.__len__(value))
    return tuple(_MappingEntry(key, item) for key, item in dict.items(value))


_DEEPEST = "deepest"  # memo bookkeeping keys (the memo's other keys are object ids)
_IMPURE = "impure"


def _remember(
    memo: dict[object, Any],
    value: object,
    result: object,
    depth: int,
    saved_deepest: int,
    impure_before: int,
    *,
    pure: bool,
) -> object:
    """Record a finished subtree for reuse if nothing in it observed a public view."""

    deepest = memo.get(_DEEPEST, depth)
    if not pure:
        memo[_IMPURE] = memo.get(_IMPURE, 0) + 1
    elif memo.get(_IMPURE, 0) == impure_before:
        memo[id(value)] = (value, result, deepest - depth)
    memo[_DEEPEST] = max(saved_deepest, deepest)
    return result


def _complete_contract_input(
    value: object,
    active_ids: set[int],
    inventories: dict[int, _ModelInventory],
    mapping_inventories: dict[int, _MappingInventory] | None = None,
    budget: DiscoveryBudget | None = None,
    depth: int = 0,
    memo: dict[object, Any] | None = None,
) -> object:
    """Copy a native input graph without normalizing away invalid data.

    Every container, model and mapping copied is checked against the shared
    ``budget`` (depth along the path, nodes by identity) and every edge copied is
    charged, repeats included, so a graph that shares substructure cannot make
    the copy exponentially larger than what discovery paid for, and a public
    iterator that never ends is refused rather than followed.

    A subtree built only from native storage (models and native containers, no public
    view observed) is a pure function of state already frozen for the call, so it is
    copied once per object identity and shared by every path that reaches it; the work
    is bounded by distinct objects, not by paths.  A subtree that observed a public view
    is never reused: each edge that reaches such a view observes it again and is charged.
    """

    if mapping_inventories is None:
        mapping_inventories = {}
    budget = _new_budget(budget)
    if memo is None:
        memo = {}
    cached = memo.get(id(value))
    if cached is not None and cached[0] is value:  # the entry keeps the object alive, so the id is its own
        budget.visit(value, depth, True)
        if depth + cached[2] > budget.tier.max_depth:
            budget.charge_edge(budget.tier.max_edges + 1)  # the stable budget error: the nesting limit holds on every path
        memo[_DEEPEST] = max(memo.get(_DEEPEST, 0), depth + cached[2])
        return cached[1]
    saved_deepest = memo.get(_DEEPEST, 0)
    memo[_DEEPEST] = depth
    impure_before = memo.get(_IMPURE, 0)

    if _inherits(type(value), BaseModel):
        identity = id(value)
        if identity in active_ids:
            raise ValueError("cyclic contract input is not supported")
        active_ids.add(identity)
        try:
            inventory = _validated_model_inventory(value, inventories)
            if inventory is None:
                _freeze_reachable_model_inventories(
                    value, inventories, set(), None, mapping_inventories, budget, depth
                )
                inventory = _validated_model_inventory(value, inventories)
                if inventory is None:
                    raise ValueError("contract input model inventory is missing")
            budget.visit(value, depth, True)
            stored_storage = _raise_captured_failure(inventory.stored_storage)
            native_stored_items = (
                inventory.stored_items
                if inventory.stored_items is not None
                else _retained_mapping_items(
                    stored_storage,  # type: ignore[arg-type]
                    mapping_inventories,
                    inventories,
                    budget,
                )
            )
            stored_items = _normalize_model_items(native_stored_items)
            stored_values = dict(stored_items)
            declared_fields = type(value).model_fields
            extra_values = _raise_captured_failure(inventory.extra_storage)
            if extra_values is not None:
                if not (
                    _is_native_dict(extra_values)
                    or _is_mapping(extra_values)
                ):
                    raise ValueError(
                        "contract input extra storage must be a mapping"
                    )
                native_extra_items = (
                    inventory.extra_items
                    if inventory.extra_items is not None
                    else _retained_mapping_items(
                        extra_values,  # type: ignore[arg-type]
                        mapping_inventories,
                        inventories,
                        budget,
                    )
                )
                extra_items = _normalize_model_items(native_extra_items)
                for key, item in extra_items:
                    if key in stored_values or key in declared_fields:
                        raise ValueError(
                            "contract input contains duplicate stored fields"
                        )
                    stored_values[key] = item

            # An unchecked copy normally stores its update in ``__dict__``.
            # Retain even an anomalous set-only field so closed-model validation
            # cannot silently erase evidence of unknown input semantics.
            fields_set_storage = _raise_captured_failure(
                inventory.fields_set_storage
            )
            fields_set_names = _normalize_model_names(
                inventory.fields_set_names
                if inventory.fields_set_names is not None
                else _fields_set_names(fields_set_storage, budget)
            )
            for field_name in fields_set_names:
                if (
                    field_name not in declared_fields
                    and field_name not in stored_values
                ):
                    stored_values[field_name] = None

            completed_model: dict[object, object] = {}
            for key, item in stored_values.items():
                budget.charge_edge()  # one stack frame per level: no helper closure
                completed_model[key] = _complete_contract_input(
                    item, active_ids, inventories, mapping_inventories, budget, depth + 1, memo
                )
            return _remember(
                memo, value, completed_model, depth, saved_deepest, impure_before,
                pure=(
                    inventory.stored_items is not None
                    and inventory.extra_items is not None
                    and inventory.fields_set_names is not None
                ),
            )
        finally:
            active_ids.remove(identity)

    if _is_mapping(value):
        identity = id(value)
        if identity in active_ids:
            raise ValueError("cyclic contract input is not supported")
        active_ids.add(identity)
        try:
            budget.visit(value, depth, True)
            # Keys are deliberately not stringified or otherwise normalized.
            if _is_native_dict(value):
                retained_dict = _validated_mapping_inventory(value, mapping_inventories)
                if retained_dict is None:
                    # First met here, through a public view: this read is its first
                    # observation, kept for every later edge that reaches the same dict.
                    retained_dict = _MappingInventory(value, _native_entries(value, budget))
                    mapping_inventories[id(value)] = retained_dict
                entries = cast(
                    tuple[_MappingEntry | _CapturedFailure, ...],
                    retained_dict.entries,
                )
            else:
                entries = _retained_mapping_entries(
                    value,
                    mapping_inventories,
                    inventories,
                    budget,
                )
            completed: dict[object, object] = {}
            for entry in entries:
                retained = cast(
                    _MappingEntry,
                    _raise_captured_failure(entry),
                )
                key = retained.key
                if issubclass(type(key), str):
                    # One key by its text: read without running the subclass's hooks and stored
                    # as a plain str, so a lookup cannot find a different key than the text says.
                    key = str.__str__(key)
                    if key in completed:
                        raise ValueError("contract input mapping has colliding keys")
                budget.charge_edge()  # one stack frame per level: no helper closure
                completed[key] = _complete_contract_input(
                    retained.value, active_ids, inventories, mapping_inventories, budget, depth + 1, memo
                )
            return _remember(
                memo, value, completed, depth, saved_deepest, impure_before,
                pure=_is_native_dict(value),
            )
        finally:
            active_ids.remove(identity)

    if issubclass(type(value), list):
        identity = id(value)
        if identity in active_ids:
            raise ValueError("cyclic contract input is not supported")
        active_ids.add(identity)
        try:
            budget.visit(value, depth, True)
            completed_list: list[object] = []
            for item in value:  # the public iterator: every item it yields is charged
                budget.charge_edge()  # one stack frame per level: no helper closure
                completed_list.append(
                    _complete_contract_input(
                        item, active_ids, inventories, mapping_inventories, budget, depth + 1, memo
                    )
                )
            return _remember(
                memo, value, completed_list, depth, saved_deepest, impure_before,
                pure=is_exact_type(value, list),
            )
        finally:
            active_ids.remove(identity)

    if issubclass(type(value), tuple):
        identity = id(value)
        if identity in active_ids:
            raise ValueError("cyclic contract input is not supported")
        active_ids.add(identity)
        try:
            budget.visit(value, depth, True)
            completed_items: list[object] = []
            for item in value:  # the public iterator: every item it yields is charged
                budget.charge_edge()  # one stack frame per level: no helper closure
                completed_items.append(
                    _complete_contract_input(
                        item, active_ids, inventories, mapping_inventories, budget, depth + 1, memo
                    )
                )
            return _remember(
                memo, value, tuple(completed_items), depth, saved_deepest, impure_before,
                pure=is_exact_type(value, tuple),
            )
        finally:
            active_ids.remove(identity)

    memo[_DEEPEST] = saved_deepest  # a scalar is not a level
    return value


def _validated_principal_snapshot(
    principal: PrincipalContext,
    inventories: dict[int, _ModelInventory],
    mapping_inventories: dict[int, _MappingInventory],
    budget: DiscoveryBudget | None = None,
) -> PrincipalContext:
    return PrincipalContext.model_validate(
        _complete_contract_input(
            principal,
            set(),
            inventories,
            mapping_inventories,
            budget,
        )
    )


def _validated_evaluation_snapshot(
    evaluation: PolicyEvaluationInput,
    inventories: dict[int, _ModelInventory],
    mapping_inventories: dict[int, _MappingInventory],
    budget: DiscoveryBudget | None = None,
) -> PolicyEvaluationInput:
    return PolicyEvaluationInput.model_validate(
        _complete_contract_input(
            evaluation,
            set(),
            inventories,
            mapping_inventories,
            budget,
        )
    )


def _validated_policy_snapshot(
    policy: AccessPolicy,
    inventories: dict[int, _ModelInventory],
    mapping_inventories: dict[int, _MappingInventory],
    budget: DiscoveryBudget | None = None,
) -> AccessPolicy:
    return AccessPolicy.model_validate(
        _complete_contract_input(
            policy,
            set(),
            inventories,
            mapping_inventories,
            budget,
        )
    )


def _decision(
    *,
    principal: PrincipalContext,
    action: str,
    resource: ReferenceHandle,
    policy: AccessPolicy,
    outcome: PolicyOutcome,
    reason_code: PolicyReasonCode,
) -> PolicyDecision:
    return PolicyDecision(
        schema_version="candidate-v1",
        action=action,
        resource=resource,
        outcome=outcome,
        reason_code=reason_code,
        policy_epoch=policy.policy_epoch,
        evaluated_tenant_id=principal.tenant_id,
        evaluated_subject_id=principal.subject_id,
        evaluated_credential_id=principal.credential_id,
        evaluated_membership_version=principal.membership_version,
    )


def evaluate_policy(
    *,
    principal: PrincipalContext,
    evaluation: PolicyEvaluationInput,
    policy: AccessPolicy,
) -> PolicyDecision:
    """Evaluate exact policy statements, returning a safe allow/deny decision.

    All three input graphs are reconstructed and validated before authorization
    logic.  Evaluation is deliberately limited to the action vocabulary and
    exact reference matching implemented here.  Unknown actions or policy terms
    fail closed.  The function neither resolves the resource nor checks
    existence.

    Container inputs are authoritative through their public view captured once
    at first observation (see ``AccessPolicy``).  A tuple subclass whose native
    backing disagrees with its public iterator is evaluated from the iterator;
    unexposed lazy children are not promised at entry.
    """

    inventories: dict[int, _ModelInventory] = {}
    mapping_owners: dict[int, Mapping[object, object]] = {}
    mapping_inventories: dict[int, _MappingInventory] = {}
    native_dicts: dict[int, _MappingInventory] = {}
    visited_containers: set[int] = set()
    budget = DiscoveryBudget(STANDARD)  # one envelope across all three roots
    for value in (principal, evaluation, policy):
        _freeze_reachable_model_inventories(
            value,
            inventories,
            visited_containers,
            mapping_owners,
            native_dicts,
            budget,
        )
    _freeze_mapping_inventories(
        mapping_owners,
        mapping_inventories,
        inventories,
        visited_containers,
        native_dicts,
        None,
        budget,
    )
    mapping_inventories.update(native_dicts)
    principal = _validated_principal_snapshot(
        principal,
        inventories,
        mapping_inventories,
        budget,
    )
    evaluation = _validated_evaluation_snapshot(
        evaluation,
        inventories,
        mapping_inventories,
        budget,
    )
    policy = _validated_policy_snapshot(
        policy,
        inventories,
        mapping_inventories,
        budget,
    )

    action = evaluation.action
    resource = evaluation.resource

    def deny(reason_code: PolicyReasonCode = "not_authorized") -> PolicyDecision:
        return _decision(
            principal=principal,
            action=action,
            resource=resource,
            policy=policy,
            outcome="deny",
            reason_code=reason_code,
        )

    if action not in SUPPORTED_POLICY_ACTIONS:
        return deny("unsupported_action")

    # A snapshot containing terms this evaluator does not understand cannot be
    # partially interpreted as permissive, even when the unknown term is on an
    # otherwise unrelated statement.
    if any(
        statement.action not in SUPPORTED_POLICY_ACTIONS
        for statement in policy.statements
    ):
        return deny("unsupported_policy_semantics")
    if principal.delegation is not None and any(
        delegated_action not in SUPPORTED_POLICY_ACTIONS
        for delegated_action in principal.delegation.allowed_actions
    ):
        return deny("unsupported_policy_semantics")

    if not (
        principal.tenant_id == policy.tenant_id
        and resource.tenant_id == policy.tenant_id
    ):
        return deny()

    current_membership = next(
        (
            membership
            for membership in policy.membership_versions
            if membership.subject_id == principal.subject_id
        ),
        None,
    )
    if (
        current_membership is None
        or current_membership.version != principal.membership_version
    ):
        return deny()

    if principal.delegation is not None:
        if action not in principal.delegation.allowed_actions:
            return deny()
        if resource not in principal.delegation.allowed_resources:
            return deny()

    matching_statements = tuple(
        statement
        for statement in policy.statements
        if statement.subject_id == principal.subject_id
        and statement.action == action
        and statement.resource == resource
        and (
            statement.credential_id is None
            or statement.credential_id == principal.credential_id
        )
    )

    # Denial precedence is independent of statement ordering.
    if any(statement.effect == "deny" for statement in matching_statements):
        return deny()
    if any(statement.effect == "allow" for statement in matching_statements):
        return _decision(
            principal=principal,
            action=action,
            resource=resource,
            policy=policy,
            outcome="allow",
            reason_code="explicit_grant",
        )
    return deny()


def validate_policy_decision(decision: PolicyDecision) -> PolicyDecision:
    """Return an independent, deeply validated receiving snapshot.

    Contract instances and unchecked copies are not trusted as provenance.  A
    consumer should use the returned snapshot before serialization or use.
    Validation does not prove that the decision came from an authority.
    """

    inventories: dict[int, _ModelInventory] = {}
    mapping_owners: dict[int, Mapping[object, object]] = {}
    mapping_inventories: dict[int, _MappingInventory] = {}
    native_dicts: dict[int, _MappingInventory] = {}
    visited_containers: set[int] = set()
    budget = DiscoveryBudget(STANDARD)
    _freeze_reachable_model_inventories(
        decision,
        inventories,
        visited_containers,
        mapping_owners,
        native_dicts,
        budget,
    )
    _freeze_mapping_inventories(
        mapping_owners,
        mapping_inventories,
        inventories,
        visited_containers,
        native_dicts,
        None,
        budget,
    )
    mapping_inventories.update(native_dicts)
    return PolicyDecision.model_validate(
        _complete_contract_input(
            decision,
            set(),
            inventories,
            mapping_inventories,
            budget,
        )
    )


__all__ = ["evaluate_policy", "validate_policy_decision"]
