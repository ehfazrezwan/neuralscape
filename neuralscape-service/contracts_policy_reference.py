"""Deterministic, side-effect-free reference policy evaluator."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from types import MappingProxyType
from typing import Any, NamedTuple, cast

from pydantic import BaseModel, ValidationError
from pydantic_core import PydanticCustomError

from contracts_policy import (
    SUPPORTED_POLICY_ACTIONS,
    AccessPolicy,
    PolicyDecision,
    PolicyEvaluationInput,
    PolicyOutcome,
    PolicyReasonCode,
    PolicyStatement,
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
_TYPE_NAME = vars(type)["__name__"]  # a class's name, read without any metaclass hook


_MAX_TYPE_NAME = 200  # a class name is caller-controlled: a diagnostic quotes at most this much of it


def _type_name(value: object) -> str:
    """The name of ``value``'s class as exact, bounded text: a class may be named by a ``str`` subclass,
    whose ``__format__``/``__str__`` would otherwise run when the name is interpolated, and by a name of
    any length."""

    name = str.__str__(_TYPE_NAME.__get__(type(value)))
    return name if len(name) <= _MAX_TYPE_NAME else name[:_MAX_TYPE_NAME] + "..."


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
    if fields_set_names is not None:
        # A legitimate set names fields the model stores, which are charged as entries when the
        # model is expanded.  Only names beyond those are charged here, so a model cannot keep a
        # large private copy of a set (or share one among many models) for free, while an
        # ordinary model pays nothing twice.
        stored_count = len(stored_items) if stored_items is not None else 0
        extra_count = len(extra_items) if extra_items is not None else 0
        budget.charge_edge(max(0, len(fields_set_names) - stored_count - extra_count))
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
    visited_containers: dict[int, int],
    mapping_owners: dict[int, Mapping[object, object]] | None = None,
    native_dicts: dict[int, _MappingInventory] | None = None,
    budget: DiscoveryBudget | None = None,
    depth: int = 0,
    authoritative: bool = True,
    reach: dict[int, tuple[Mapping[object, object], ...]] | None = None,
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

    ``reach[id(value)]`` records the public mappings reachable at or beneath ``value``,
    including through containers an earlier path already visited (charged once per
    container), so a caller that exposes ``value`` can capture every mapping it exposes
    even when the walk itself skips what it has seen.
    """

    if mapping_owners is None:
        mapping_owners = {}
    if native_dicts is None:
        native_dicts = {}
    if reach is None:
        reach = {}
    budget = _new_budget(budget)

    # One stack frame per level (no helper closure): the depth limit then bounds
    # the interpreter stack as well as the walk.
    below = (inventories, visited_containers, mapping_owners, native_dicts, budget, depth + 1)
    parts: list[tuple[Mapping[object, object], ...]] = []  # the reach of each child, by reference

    def finish() -> None:
        # Remember what was found beneath this value; no recursion here.  A single source is
        # held by reference (a container shared by many parents costs nothing per parent);
        # only a genuine merge builds, and is charged for, a new tuple of distinct mappings.
        # Repeated references to one child share its tuple, so sources are distinct by identity
        # first (no per-reference work); a merge is charged for every entry it will read
        # *before* it reads them, so the work is bounded by the budget, not by the aliasing.
        distinct = {id(part): part for part in parts if part}
        sources = list(distinct.values())
        if len(sources) <= 1:
            beneath = sources[0] if sources else ()
        else:
            budget.charge_edge(sum(len(part) for part in sources))
            beneath = tuple({id(mapping): mapping for part in sources for mapping in part}.values())
        reach[id(value)] = beneath

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
                _freeze_reachable_model_inventories(item, *below, authoritative, reach)
                parts.append(reach.get(id(item), ()))
        elif _is_mapping(inventory.stored_storage):
            _own_fallback_mapping(inventory.stored_storage, mapping_owners, budget, depth + 1, parts)
        if inventory.extra_items is not None:
            budget.charge_edge(len(inventory.extra_items))
            for _, item in inventory.extra_items:
                _freeze_reachable_model_inventories(item, *below, authoritative, reach)
                parts.append(reach.get(id(item), ()))
        elif _is_mapping(inventory.extra_storage):
            _own_fallback_mapping(inventory.extra_storage, mapping_owners, budget, depth + 1, parts)
        finish()
        return

    if _is_native_dict(value):
        if not budget.visit(value, depth, authoritative):
            return
        identity = id(value)
        visited_containers[identity] = dict.__len__(value)
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
            _freeze_reachable_model_inventories(entry.value, *below, authoritative, reach)
            parts.append(reach.get(id(entry.value), ()))
        finish()
        return

    if _is_mapping(value):
        budget.visit(value, depth, True)  # its pairs are its public view: authoritative
        _retain_mapping_owner(value, mapping_owners)
        reach[id(value)] = (value,)
        return

    if issubclass(type(value), list):
        if not budget.visit(value, depth, authoritative):
            return
        _precheck(budget, list.__len__(value))
        items = tuple(list.__iter__(value))
        budget.charge_edge(len(items))
        visited_containers[id(value)] = len(items)  # completion's first copy gets these entries free
        for item in items:
            _freeze_reachable_model_inventories(
                item, *below, authoritative and is_exact_type(value, list), reach
            )
            parts.append(reach.get(id(item), ()))
        finish()
        return

    if issubclass(type(value), tuple):
        if not budget.visit(value, depth, authoritative):
            return
        _precheck(budget, tuple.__len__(value))
        items = tuple(tuple.__iter__(value))
        budget.charge_edge(len(items))
        visited_containers[id(value)] = len(items)
        for item in items:
            _freeze_reachable_model_inventories(
                item, *below, authoritative and is_exact_type(value, tuple), reach
            )
            parts.append(reach.get(id(item), ()))
        finish()
        return


def _own_fallback_mapping(
    storage: Mapping[object, object],
    mapping_owners: dict[int, Mapping[object, object]],
    budget: DiscoveryBudget,
    depth: int,
    parts: list[tuple[Mapping[object, object], ...]],
) -> None:
    """A model's own non-dict storage is a public mapping one level below the model.

    Its depth is recorded in the budget like any other node, so a chain of models linked
    through such storage cannot reset the depth at every link.
    """

    budget.charge_edge()
    budget.visit(storage, depth, True)
    _retain_mapping_owner(storage, mapping_owners)
    parts.append((storage,))


def _open_mapping_items(value: Mapping[object, object]) -> Iterator[object]:
    """The mapping's public pair iterator, opened but NOT consumed."""

    if _is_native_dict(value):
        return iter(tuple(dict.items(value)))
    return iter(value.items())


def _open_pair(raw_entry: object) -> tuple[object, object, Iterator[object] | None]:
    """Read a key/value pair's first two items: ``(key, value, rest)``.

    Unpacking (``key, value = pair``) asks a pair's iterator for a third item, to check that
    there is none, *before* the caller can freeze the value, and a pair's public iterator can
    run code at that point.  This reads exactly two items; the caller freezes the value and then
    calls ``_close_pair(rest)``.  An exact tuple is read natively (no hook runs, ``rest`` is
    ``None``).  The errors and messages are the ones unpacking raises.
    """

    if type(raw_entry) is tuple:
        if tuple.__len__(raw_entry) != 2:
            size = tuple.__len__(raw_entry)
            if size < 2:
                raise ValueError(f"not enough values to unpack (expected 2, got {size})")
            raise ValueError("too many values to unpack (expected 2)")
        return tuple.__getitem__(raw_entry, 0), tuple.__getitem__(raw_entry, 1), None
    try:
        iterator = iter(raw_entry)  # type: ignore[call-overload]
    except TypeError:
        if _defines_iter(type(raw_entry)):
            raise  # unpacking reports the iteration protocol's own error unchanged
        raise TypeError(f"cannot unpack non-iterable {_type_name(raw_entry)} object") from None
    try:
        key = next(iterator)
    except StopIteration:
        raise ValueError("not enough values to unpack (expected 2, got 0)") from None
    try:
        value = next(iterator)
    except StopIteration:
        raise ValueError("not enough values to unpack (expected 2, got 1)") from None
    return key, value, iterator


_CLASS_DICT = vars(type)["__dict__"]


def _defines_iter(cls: type) -> bool:
    """Whether any class on ``cls``'s own MRO defines ``__iter__`` (even as ``None``).

    That is when unpacking reports ``iter()``'s own ``TypeError``; a type with no ``__iter__`` at
    all gets "cannot unpack non-iterable".  The class dictionaries are read through the base
    ``type`` descriptor, so no metaclass hook runs.
    """

    return any("__iter__" in _CLASS_DICT.__get__(base) for base in _TYPE_MRO.__get__(cls))


def _close_pair(rest: Iterator[object] | None) -> None:
    """The exhaustion check unpacking would have done, after the value was frozen."""

    if rest is None:
        return
    try:
        next(rest)
    except StopIteration:
        return
    raise ValueError("too many values to unpack (expected 2)")


def _freeze_mapping_inventories(
    owners: dict[int, Mapping[object, object]],
    mapping_inventories: dict[int, _MappingInventory],
    model_inventories: dict[int, _ModelInventory],
    visited_containers: dict[int, int],
    native_dicts: dict[int, _MappingInventory] | None = None,
    in_progress: set[int] | None = None,
    budget: DiscoveryBudget | None = None,
    reach: dict[int, tuple[Mapping[object, object], ...]] | None = None,
    active_depth: int = 0,
    scanned: dict[int, tuple[Mapping[object, object], ...]] | None = None,
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
    if reach is None:
        reach = {}
    if scanned is None:
        # Reach tuples already merged into ``owners`` during this capture, held by reference for the
        # whole call: an id alone could be reused once a shallower visit releases the tuple.
        scanned = {}
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
            # The deeper of the owner's recorded depth and the depth of the capture path that
            # reached it: a shallow alias records depth 1 for every node of a long chain, but
            # capturing that chain recursively still nests, so the path depth is what bounds the
            # stack (each nested capture is one level, checked by the budget like any other).
            child_depth = max(budget.authoritative_depth(owner) or 0, active_depth) + 1
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
                    key, item, rest = _open_pair(raw_entry)
                except Exception as error:
                    entries.append(_CapturedFailure(error))
                    terminal = True
                    continue
                # The pair's value is frozen (below) before the pair's iterator is asked
                # whether it has a third item, so it cannot repair what it just exposed.
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
                    True,
                    reach,
                )
                beneath = reach.get(id(item), ())
                if beneath and scanned.get(id(beneath)) is not beneath:
                    # Including those behind containers the walk had already visited.  A reach
                    # tuple is scanned (and charged) once per capture: every mapping in it was
                    # retained and captured, or is in progress, the first time it was exposed,
                    # so a later pair exposing the same tuple has nothing new to find in it.
                    scanned[id(beneath)] = beneath
                    budget.charge_edge(len(beneath))
                    for found in beneath:
                        _retain_mapping_owner(found, exposed)
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
                        reach,
                        child_depth,
                        scanned,
                    )
                try:
                    _close_pair(rest)
                except Exception as error:
                    entries.append(_CapturedFailure(error))
                    terminal = True
                    continue
                entries.append(_MappingEntry(key, item))
            in_progress.discard(id(owner))
            mapping_inventories[id(owner)] = _MappingInventory(
                owner,
                _CapturedFailure(failure) if failure is not None else tuple(entries),
            )


def _freeze_mapping_entries(
    raw_entries: tuple[object, ...],
    owners: dict[int, Mapping[object, object]],
    model_inventories: dict[int, _ModelInventory],
    visited_containers: dict[int, int],
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
            key, item, rest = _open_pair(raw_entry)
        except Exception as error:
            entries.append(_CapturedFailure(error))
            break
        _freeze_reachable_model_inventories(  # frozen before the pair's exhaustion check
            item,
            model_inventories,
            visited_containers,
            owners,
            native_dicts,
            budget,
            depth,
        )
        try:
            _close_pair(rest)
        except Exception as error:
            entries.append(_CapturedFailure(error))
            break
        entries.append(_MappingEntry(key, item))
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
    paid: dict[int, int] | None = None,
) -> tuple[_MappingEntry | _CapturedFailure, ...]:
    """Return one authenticated public inventory, capturing it if newly seen.

    ``paid`` is completion's map of what discovery already charged: a mapping first captured here
    records its native descendants in it, so completion does not charge their entries a second time.
    """

    budget = _new_budget(budget)

    inventory = _validated_mapping_inventory(value, mapping_inventories)
    if inventory is None:
        owners: dict[int, Mapping[object, object]] = {}
        _retain_mapping_owner(value, owners)
        _freeze_mapping_inventories(
            owners,
            mapping_inventories,
            model_inventories,
            {} if paid is None else paid,
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
    paid: dict[int, int] | None = None,
) -> tuple[tuple[object, object], ...]:
    """Rebuild model-owned items while retaining per-entry error priority."""

    items: list[tuple[object, object]] = []
    for entry in _retained_mapping_entries(
        value,
        mapping_inventories,
        model_inventories,
        budget,
        paid,
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
_PAID = "paid"  # id -> how many entries discovery charged, for each native container it walked
# Non-text key types whose hashing and equality run no caller code and never equal a string key.
_PLAIN_NON_TEXT_KEYS = (int, float, bool, type(None), bytes)


class _OpaqueKey:
    """A non-text key held by identity: it hashes and compares as itself, never as the key.

    Validation reports it as an invalid (non-string) key at the key's place; its representation
    names only the key's type, so no hook of the key runs.
    """

    __slots__ = ("_key",)

    def __init__(self, key: object) -> None:
        self._key = key

    def __repr__(self) -> str:
        return f"<{_type_name(self._key)} key>"
_COPIED = "copied"  # ids of containers completion has copied at least once


def _first_paid_copy(memo: dict[object, Any], value: object, paid: bool) -> bool:
    """Whether copying ``value`` now needs no edge charges.

    Discovery charges every entry of a container it walks.  The first copy of that container
    replays exactly those entries, so charging them again would count each edge twice.  Any
    later copy (an impure subtree rebuilt on another path) is new work and is charged in full,
    which is what bounds exponential re-copying.
    """

    copied = memo.setdefault(_COPIED, set())
    first = id(value) not in copied
    copied.add(id(value))
    return first and paid
_REUSED = "reused"  # how many times a finished native subtree was shared between paths


# At most this many validation errors are reported in full; the rest are counted, not listed.
MAX_ERROR_ENTRIES = 100


def _validated_without_echo(
    model_type: type[BaseModel], native: object, memo: dict[object, Any]
) -> Any:
    """Strictly validate ``native``; if it shares subtrees, never echo it in an error.

    A validation error normally repeats the offending input.  When the copy shares native
    subtrees between paths, the input's text is exponentially larger than the graph (a
    few hundred bytes of input can take minutes to render), so that error is rebuilt
    without the input: the same types, locations and messages, nothing embedded.  An error with
    more than ``MAX_ERROR_ENTRIES`` entries keeps that many (without inputs) plus one entry
    counting the rest.  Ordinary inputs keep the full diagnostic.
    """

    try:
        return model_type.model_validate(native)
    except ValidationError as error:
        count = error.error_count()
        if not memo.get(_REUSED) and count <= MAX_ERROR_ENTRIES:
            raise
        # A shared input is never echoed; a very long error list keeps its first entries (without
        # their inputs) and says how many more there were, so its text cannot grow with the input.
        marker = "<omitted: input shares subtrees>" if memo.get(_REUSED) else "<omitted: error list bounded>"
        lines: list[dict[str, Any]] = [
            {"type": e["type"], "loc": e["loc"], "input": marker, **({"ctx": e["ctx"]} if "ctx" in e else {})}
            for e in error.errors(include_input=False, include_url=False)[:MAX_ERROR_ENTRIES]
        ]
        if count > MAX_ERROR_ENTRIES:
            lines.append({
                "type": PydanticCustomError(
                    "too_many_errors", "{omitted} further validation errors omitted",
                    {"omitted": count - MAX_ERROR_ENTRIES},
                ),
                "loc": (),
                "input": marker,
            })
        try:
            bounded: Exception = ValidationError.from_exception_data(error.title, lines)  # type: ignore[arg-type]
        except Exception:  # noqa: BLE001 - a type that cannot be rebuilt still gets a bounded, value-free error
            bounded = ValueError(f"{error.title}: {error.error_count()} validation error(s)")
        raise bounded from None


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
        memo[_REUSED] = memo.get(_REUSED, 0) + 1
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
                # A model first exposed here (by a container's public view) is discovered like one the entry
                # walk found: the public mappings it reaches (a proxy field, an extra-storage mapping) are
                # captured now, before any callback of the model's own storage runs.
                lazy_owners: dict[int, Mapping[object, object]] = {}
                lazy_reach: dict[int, tuple[Mapping[object, object], ...]] = {}
                lazy_paid = memo.setdefault(_PAID, {})
                _freeze_reachable_model_inventories(
                    value, inventories, lazy_paid, lazy_owners, mapping_inventories, budget, depth, True, lazy_reach
                )
                lazy_pending = {
                    identity: found
                    for identity, found in lazy_owners.items()
                    if _validated_mapping_inventory(found, mapping_inventories) is None
                }
                if lazy_pending:
                    _freeze_mapping_inventories(
                        lazy_pending,
                        mapping_inventories,
                        inventories,
                        lazy_paid,
                        mapping_inventories,
                        None,
                        budget,
                        lazy_reach,
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
                    memo.setdefault(_PAID, {}),
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
                        memo.setdefault(_PAID, {}),
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
            free = _first_paid_copy(memo, value, True)  # a captured model's entries were charged
            if free:
                # Discovery paid for the stored and extra entries, and for the fields-set names
                # beyond their count (``max`` of the two); names read just above from a non-set
                # storage were each charged as read.  Names synthesized from a fields-set that does
                # not name stored fields were not all paid for: charge the entries past that.
                entries = (len(inventory.stored_items) if inventory.stored_items is not None else 0) + (
                    len(inventory.extra_items) if inventory.extra_items is not None else 0
                )
                paid = (
                    max(entries, len(fields_set_names))
                    if inventory.fields_set_names is not None
                    else entries + len(fields_set_names)
                )
                budget.charge_edge(max(0, len(stored_values) - paid))
            for key, item in stored_values.items():
                if not free:
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
            paid = True  # a public mapping's pairs were charged as it was captured
            if _is_native_dict(value):
                retained_dict = _validated_mapping_inventory(value, mapping_inventories)
                paid = retained_dict is not None and identity in memo.get(_PAID, ())
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
                    memo.setdefault(_PAID, {}),
                )
            completed: dict[object, object] = {}
            free = _first_paid_copy(memo, value, paid)
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
                elif not is_exact_type(key, *_PLAIN_NON_TEXT_KEYS):
                    # Any other key is invalid for a contract, but inserting it would run its hash
                    # and equality and could overwrite a real key; an identity box keeps it
                    # separate so validation reports it as an invalid key at the same place.
                    key = _OpaqueKey(key)
                if not free:
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
            # An exact list discovery walked was charged entry by entry, but only for the entries
            # it had then: the live list can grow afterwards (an earlier public iterator, or an
            # item's own iterator during this loop), so every entry past that count is charged.
            # A public view was never charged.
            paid = memo.get(_PAID, {})
            prepaid = paid[identity] if (
                _first_paid_copy(memo, value, is_exact_type(value, list) and identity in paid)
            ) else 0
            for index, item in enumerate(value):  # the public iterator: each item it yields
                if index >= prepaid:
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
            free = _first_paid_copy(
                memo, value, is_exact_type(value, tuple) and identity in memo.get(_PAID, ())
            )
            for item in value:  # the public iterator: every item it yields is charged
                if not free:
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
    paid: dict[int, int] | None = None,
) -> PrincipalContext:
    memo: dict[object, Any] = {} if paid is None else {_PAID: paid}
    native = _complete_contract_input(
        principal, set(), inventories, mapping_inventories, budget, 0, memo
    )
    return _validated_without_echo(PrincipalContext, native, memo)


def _validated_evaluation_snapshot(
    evaluation: PolicyEvaluationInput,
    inventories: dict[int, _ModelInventory],
    mapping_inventories: dict[int, _MappingInventory],
    budget: DiscoveryBudget | None = None,
    paid: dict[int, int] | None = None,
) -> PolicyEvaluationInput:
    memo: dict[object, Any] = {} if paid is None else {_PAID: paid}
    native = _complete_contract_input(
        evaluation, set(), inventories, mapping_inventories, budget, 0, memo
    )
    return _validated_without_echo(PolicyEvaluationInput, native, memo)


def _validated_policy_snapshot(
    policy: AccessPolicy,
    inventories: dict[int, _ModelInventory],
    mapping_inventories: dict[int, _MappingInventory],
    budget: DiscoveryBudget | None = None,
    paid: dict[int, int] | None = None,
) -> AccessPolicy:
    memo: dict[object, Any] = {} if paid is None else {_PAID: paid}
    native = _complete_contract_input(
        policy, set(), inventories, mapping_inventories, budget, 0, memo
    )
    return _validated_without_echo(AccessPolicy, native, memo)


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


# The statement fields ``evaluate_policy`` judges.  A field added to ``PolicyStatement`` must be
# judged here (and listed) before a statement that sets it can authorize anything.
_JUDGED_STATEMENT_FIELDS = frozenset({"id", "effect", "subject_id", "credential_id", "action", "resource"})


def _is_set_unjudged(statement: PolicyStatement, name: str, field: Any) -> bool:
    """Whether an unjudged field carries a term: a required field always does (it has no default, so
    any value, ``None`` included, was provided); an optional one does when it differs from its real
    default, a default factory's product included."""

    if field.is_required():
        return True
    return getattr(statement, name, None) != field.get_default(call_default_factory=True)


def _unjudged_terms(statement: PolicyStatement) -> tuple[str, ...]:
    """The fields of ``statement`` outside ``_JUDGED_STATEMENT_FIELDS`` that carry a term."""

    return tuple(
        name
        for name, field in type(statement).model_fields.items()
        if name not in _JUDGED_STATEMENT_FIELDS and _is_set_unjudged(statement, name, field)
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
    visited_containers: dict[int, int] = {}
    budget = DiscoveryBudget(STANDARD)  # one envelope across all three roots
    reach: dict[int, tuple[Mapping[object, object], ...]] = {}
    for value in (principal, evaluation, policy):
        _freeze_reachable_model_inventories(
            value,
            inventories,
            visited_containers,
            mapping_owners,
            native_dicts,
            budget,
            0,
            True,
            reach,
        )
    _freeze_mapping_inventories(
        mapping_owners,
        mapping_inventories,
        inventories,
        visited_containers,
        native_dicts,
        None,
        budget,
        reach,
    )
    mapping_inventories.update(native_dicts)
    principal = _validated_principal_snapshot(
        principal,
        inventories,
        mapping_inventories,
        budget,
        visited_containers,
    )
    evaluation = _validated_evaluation_snapshot(
        evaluation,
        inventories,
        mapping_inventories,
        budget,
        visited_containers,
    )
    policy = _validated_policy_snapshot(
        policy,
        inventories,
        mapping_inventories,
        budget,
        visited_containers,
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
    # A statement field this evaluator does not judge (a condition, for example) narrows or
    # qualifies the grant; ignoring it would turn a conditioned allow into an unconditional one.
    # Any statement that sets such a field makes the whole snapshot unsupported.
    if any(_unjudged_terms(statement) for statement in policy.statements):
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
    visited_containers: dict[int, int] = {}
    budget = DiscoveryBudget(STANDARD)
    reach: dict[int, tuple[Mapping[object, object], ...]] = {}
    _freeze_reachable_model_inventories(
        decision,
        inventories,
        visited_containers,
        mapping_owners,
        native_dicts,
        budget,
        0,
        True,
        reach,
    )
    _freeze_mapping_inventories(
        mapping_owners,
        mapping_inventories,
        inventories,
        visited_containers,
        native_dicts,
        None,
        budget,
        reach,
    )
    mapping_inventories.update(native_dicts)
    memo: dict[object, Any] = {_PAID: visited_containers}
    native = _complete_contract_input(
        decision, set(), inventories, mapping_inventories, budget, 0, memo
    )
    return _validated_without_echo(PolicyDecision, native, memo)


__all__ = ["evaluate_policy", "validate_policy_decision"]
