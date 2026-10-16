"""Strict primitives shared by candidate product contracts.

Contract models reject unknown fields and coercion when validating Python
objects.  JSON validation accepts only the corresponding JSON scalar types.
``model_dump()`` is the native Python representation; ``model_dump_json()``
is the wire representation.  Neither representation grants identity or
authority merely because a value validates.
"""

from collections.abc import Mapping
from types import MappingProxyType
from dataclasses import dataclass, field
from typing import Annotated, Literal

from contracts_limits import STANDARD, DiscoveryBudget, is_exact_type
from pydantic import (
    AliasChoices,
    AliasPath,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
)


MAX_SAFE_INTEGER = 9_007_199_254_740_991

_BASE_MODEL_DICT_DESCRIPTOR = vars(BaseModel)["__dict__"]
_BASE_MODEL_EXTRA_DESCRIPTOR = vars(BaseModel)["__pydantic_extra__"]


OpaqueId = Annotated[
    str,
    StringConstraints(strict=True, min_length=1, max_length=256),
]
"""An opaque, non-empty identifier that is preserved exactly as supplied."""


SafeCounter = Annotated[
    int,
    Field(strict=True, ge=0, le=MAX_SAFE_INTEGER),
]
"""A non-negative integer exactly representable by JavaScript number values."""


class ContractModel(BaseModel):
    """Base for strict contracts with a closed set of fields."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        revalidate_instances="always",
    )


class VersionedContract(ContractModel):
    """A candidate contract whose version must be explicit on every input."""

    schema_version: Literal["candidate-v1"]


def _validation_alias_owners(model_type: type[BaseModel]) -> dict[str, set[str]]:
    if model_type.model_config.get("validate_by_alias") is False:
        return {}

    owners: dict[str, set[str]] = {}
    for field_name, field in model_type.model_fields.items():
        validation_alias = field.validation_alias
        choices = (
            validation_alias.choices
            if isinstance(validation_alias, AliasChoices)
            else (validation_alias,)
        )
        for choice in choices:
            if isinstance(choice, str):
                owners.setdefault(choice, set()).add(field_name)
            elif isinstance(choice, AliasPath):
                owners.setdefault(choice.path[0], set()).add(field_name)
    return owners


_TYPE_MRO = vars(type)["__mro__"]
_MAPPING_BASES = (Mapping, dict, MappingProxyType)
# Key types whose hashing and equality run no caller code.  A ``str`` subclass is read as its
# text instead; any other key type can run a hook while a dict is rebuilt, so a subtree that
# holds one is never reused (see ``_snapshot_captured_graph``).
_PLAIN_KEY_TYPES = (str, int, float, bool, type(None), bytes)


def _is_subclass(cls: type, types: type | tuple[type, ...]) -> bool:
    """Whether ``cls`` inherits from one of ``types``, decided by identity on the native MRO.

    ``issubclass`` against an ABC (``Mapping``) or pydantic's ABC-based metaclass runs
    ``ABCMeta.__subclasscheck__``: registry lookups that hash and compare ``cls`` and honour
    ``__subclasshook__`` and virtual registration, so a class with a hostile metaclass could run
    code or be matched by spoofing.  This reads the real MRO through the base ``type`` descriptor.
    ``Mapping`` also recognises ``dict`` and ``mappingproxy``; a class only *registered* as a
    mapping is not one.
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


def _plain_keys(
    entries: tuple[tuple[object, object], ...],
) -> tuple[tuple[tuple[object, object], ...], bool, bool]:
    """Read ``str``-subclass keys as their text, without running their hooks.

    Returns ``(entries, pure, colliding)``.  ``pure`` is false when a key of a non-plain type
    other than a ``str`` subclass remains (rebuilding the dict hashes it, which runs its code).
    When two keys would share one text the original keys are kept and ``colliding`` is set, so
    the established diagnostics for models report exactly as before and a plain dict is refused
    when it is rebuilt (see ``_reconstruct_captured_graph``).
    """

    plain: list[tuple[object, object]] = []
    texts: set[str] = set()
    pure = True
    for key, item in entries:
        if not is_exact_type(key, *_PLAIN_KEY_TYPES):
            if _is_subclass(type(key), str):
                key = str.__str__(key)  # type: ignore[arg-type]
            else:
                pure = False
        if type(key) is str:
            if key in texts:
                return entries, False, True
            texts.add(key)  # type: ignore[arg-type]
        plain.append((key, item))
    return tuple(plain), pure, False


def _mapping_entries(
    value: Mapping[object, object],
    value_type: type[object],
    budget: DiscoveryBudget,
) -> tuple[tuple[object, object], ...]:
    if _is_subclass(value_type, dict):
        _precheck(budget, dict.__len__(value))  # type: ignore[arg-type]
        return tuple(dict.items(value))  # type: ignore[arg-type]
    # Rebuild every returned entry as a plain tuple, reading each one exactly once: an entry that
    # is a tuple subclass with a changing iterator would otherwise show a different pair to
    # capture, collision checking and reconstruction.  Each entry the view yields is charged.
    budget.charge_edge()
    entries: list[tuple[object, object]] = []
    for key, item in value.items():
        budget.charge_edge()
        entries.append((key, item))
    return tuple(entries)


def _precheck(budget: DiscoveryBudget, count: int) -> None:
    """Refuse to copy a native container that cannot fit the remaining edge allowance."""

    if count > budget.tier.max_edges - budget.edges:
        budget.charge_edge(count)  # raises the stable budget error


@dataclass
class _ModelInventory:
    model_type: type[BaseModel]
    stored_entries: tuple[tuple[object, object], ...]
    extra: object | None
    extra_entries: tuple[tuple[object, object], ...] | None = None
    malformed_extra: bool = False
    pure_names: bool = True


@dataclass
class _GraphInventory:
    budget: DiscoveryBudget = field(default_factory=lambda: DiscoveryBudget(STANDARD))
    models: dict[int, _ModelInventory] = field(default_factory=dict)
    sequences: dict[int, tuple[object, ...]] = field(default_factory=dict)
    dictionaries: dict[int, tuple[tuple[object, object], ...]] = field(
        default_factory=dict
    )
    pure_dictionaries: set[int] = field(default_factory=set)
    colliding_dictionaries: set[int] = field(default_factory=set)
    visited: set[int] = field(default_factory=set)
    # Strong references keep every captured object alive for the whole
    # reconstruction, so a released temporary's id cannot be reused by a later
    # object and alias its captured contents.
    retained: list[object] = field(default_factory=list)
    # Completed reconstructions of pure subtrees, keyed by identity:
    # ``(object, copy, height)``.  A descendant shared by many paths is rebuilt once
    # and shared; ``height`` keeps the depth limit enforced on every path.
    completed: dict[int, tuple[object, object, int]] = field(default_factory=dict)
    # Counts every callback-capable read so far: a public view (a sequence subclass's
    # iteration, a non-dict mapping's items) or a dict rebuilt with a hook-capable key.
    # A subtree during which it did not move ran no caller code, so it is pure.
    impure_reads: int = 0
    # Containers rebuilt at least once.  The first rebuild walks entries capture already
    # charged; every later rebuild of the same container (only an impure one is rebuilt
    # again) is new work, and its entries are charged again.
    rebuilt: set[int] = field(default_factory=set)


def _capture_native_graph(value: object, inventory: _GraphInventory, depth: int = 0) -> None:
    """Freeze supported native backing without invoking public protocols.

    Every captured node and entry is charged to the call's budget, and the depth is checked
    along every path, so a deep or wide input is the stable budget error, never a crash.
    """

    value_type = type(value)
    if not _is_subclass(value_type, (BaseModel, tuple, list, dict)):
        return
    budget = inventory.budget
    if depth > budget.tier.max_depth:
        budget.charge_edge(budget.tier.max_edges + 1)  # the stable budget error

    value_id = id(value)
    if value_id in inventory.visited:
        return
    budget.charge_node(value)
    inventory.visited.add(value_id)
    inventory.retained.append(value)
    below = depth + 1

    if _is_subclass(value_type, BaseModel):
        stored = _BASE_MODEL_DICT_DESCRIPTOR.__get__(value, BaseModel)
        _precheck(budget, dict.__len__(stored))
        stored_entries, pure_stored, _ = _plain_keys(tuple(dict.items(stored)))
        budget.charge_edge(len(stored_entries))
        try:
            extra = _BASE_MODEL_EXTRA_DESCRIPTOR.__get__(value, BaseModel)
        except AttributeError:
            extra = None
        model_inventory = _ModelInventory(
            model_type=value_type,
            stored_entries=stored_entries,
            extra=extra,
            pure_names=pure_stored,
        )
        inventory.models[value_id] = model_inventory

        # Capture every model reachable through native stored backing before
        # classifying extras or hashing any stored name.
        for _, item in stored_entries:
            _capture_native_graph(item, inventory, below)

        if extra is None:
            model_inventory.extra_entries = ()
            return
        extra_type = type(extra)
        if _is_subclass(extra_type, dict):
            _precheck(budget, dict.__len__(extra))
            extra_entries, pure_extra, _ = _plain_keys(tuple(dict.items(extra)))
            budget.charge_edge(len(extra_entries))
            model_inventory.extra_entries = extra_entries
            model_inventory.pure_names = model_inventory.pure_names and pure_extra
            for _, item in extra_entries:
                _capture_native_graph(item, inventory, below)
        elif not _is_subclass(extra_type, Mapping):
            model_inventory.extra_entries = ()
            model_inventory.malformed_extra = True
        # A non-dict Mapping retains its established public items protocol.
        # Its inventory is captured later, after all natively reachable model
        # backing has already been frozen.
        return

    if _is_subclass(value_type, tuple):
        _precheck(budget, tuple.__len__(value))
        items = tuple(tuple.__iter__(value))
        budget.charge_edge(len(items))
        inventory.sequences[value_id] = items
        for item in items:
            _capture_native_graph(item, inventory, below)
        return

    if _is_subclass(value_type, list):
        _precheck(budget, list.__len__(value))
        items = tuple(list.__iter__(value))
        budget.charge_edge(len(items))
        inventory.sequences[value_id] = items
        for item in items:
            _capture_native_graph(item, inventory, below)
        return

    _precheck(budget, dict.__len__(value))
    entries, pure, colliding = _plain_keys(tuple(dict.items(value)))
    budget.charge_edge(len(entries))
    inventory.dictionaries[value_id] = entries
    if pure:
        inventory.pure_dictionaries.add(value_id)
    if colliding:
        inventory.colliding_dictionaries.add(value_id)
    for _, item in entries:
        _capture_native_graph(item, inventory, below)


def _exposed_items(value: object, inventory: _GraphInventory, depth: int) -> tuple[object, ...]:
    """Read a sequence subclass's public view and freeze every sibling.

    The native backing of all exposed items is captured before the first one is
    reconstructed, so a callback run while reconstructing an earlier item cannot
    change a later sibling's captured state.  The observation and each item it yields
    are charged as they are read, so an endless view is cut off by the budget.
    """

    budget = inventory.budget
    inventory.impure_reads += 1
    budget.charge_edge()
    collected: list[object] = []
    for item in iter(value):  # type: ignore[call-overload]
        budget.charge_edge()
        collected.append(item)
    items = tuple(collected)
    inventory.retained.append(items)
    for item in items:
        _capture_native_graph(item, inventory, depth + 1)
    return items


def _snapshot_captured_graph(
    value: object,
    active_containers: set[int],
    inventory: _GraphInventory,
    depth: int = 0,
) -> object:
    """Reconstruct one graph after its reachable native backing is frozen.

    Discovery deduplicates identities, so reconstruction must too: a completed
    subtree is reused by every later path that reaches it, rather than rebuilt per
    path (which makes work and output grow exponentially with depth for a graph that
    shares descendants), and the depth limit is checked on every such path
    (``depth + height``).  A subtree is reused only if it ran no caller code: it read
    no public view and rebuilt no dict whose keys can run a hook.  Any other subtree is
    rebuilt and charged on every edge, so each edge reaching a public view observes it
    again, as the contract promises.
    """

    value_type = type(value)
    if not _is_subclass(value_type, (BaseModel, tuple, list, dict)):
        return value

    budget = inventory.budget
    if depth > budget.tier.max_depth:
        budget.charge_edge(budget.tier.max_edges + 1)  # the stable budget error
    value_id = id(value)
    if value_id not in inventory.visited:
        _capture_native_graph(value, inventory, depth)
    cached = inventory.completed.get(value_id)
    if cached is not None and cached[0] is value:
        # The edge to a reused subtree is the parent's entry, already charged.
        if depth + cached[2] > budget.tier.max_depth:
            budget.charge_edge(budget.tier.max_edges + 1)
        return cached[1]
    reads_before = inventory.impure_reads
    result, height = _reconstruct_captured_graph(value, active_containers, inventory, depth)
    if inventory.impure_reads == reads_before:
        inventory.completed[value_id] = (value, result, height)
    return result


def _child(
    item: object,
    active_containers: set[int],
    inventory: _GraphInventory,
    depth: int,
    height: int,
    charge: bool,
) -> tuple[object, int]:
    """Rebuild ``item`` at ``depth`` and return ``(copy, parent height)``.

    ``charge`` is set when this edge was not already charged: an entry of a container
    rebuilt again.  An entry rebuilt for the first time was charged by capture, and an
    item of a public view was charged when the view was read.
    """

    if charge:
        inventory.budget.charge_edge()
    copied = _snapshot_captured_graph(item, active_containers, inventory, depth)
    if _is_subclass(type(item), (BaseModel, tuple, list, dict)):
        # A reused child knows its height; an impure child is never reused, and then
        # neither is its parent, so the parent's height is not needed.
        cached = inventory.completed.get(id(item))
        item_height = cached[2] if cached is not None and cached[0] is item else 0
        height = max(height, item_height + 1)
    return copied, height


def _reconstruct_captured_graph(
    value: object,
    active_containers: set[int],
    inventory: _GraphInventory,
    depth: int,
) -> tuple[object, int]:
    """Rebuild one container; returns ``(copy, height)`` (children via ``_snapshot_captured_graph``)."""

    value_type = type(value)
    value_id = id(value)
    budget = inventory.budget
    if value_id in active_containers:
        raise ValueError("cyclic contract input is not supported")
    active_containers.add(value_id)
    height = 0
    below = depth + 1
    again = value_id in inventory.rebuilt
    inventory.rebuilt.add(value_id)
    # Children are rebuilt in plain loops (no closure, no generator), so one level of nesting
    # costs two interpreter frames and the depth limit keeps the stack far from its limit.
    try:
        if _is_subclass(value_type, BaseModel):
            captured = inventory.models[value_id]
            if not captured.pure_names:
                inventory.impure_reads += 1
            model_type = captured.model_type
            stored_entries = captured.stored_entries
            stored_names = {key for key, _ in stored_entries}
            declared_names = set(model_type.model_fields)
            alias_owners = _validation_alias_owners(model_type)
            alias_roots = set(alias_owners)
            unknown_stored_names = stored_names - declared_names
            missing_owner_roots = {
                root
                for root in stored_names & alias_roots
                if alias_owners[root] - stored_names
            }
            if unknown_stored_names & alias_roots:
                raise ValueError(
                    "contract input contains conflicting declared and extra fields"
                )
            if missing_owner_roots:
                raise ValueError(
                    "contract input contains stored validation alias roots "
                    "with missing owning fields"
                )

            extra_entries = captured.extra_entries
            if extra_entries is None:
                inventory.impure_reads += 1
                extra_entries, _pure, _colliding = _plain_keys(
                    _mapping_entries(
                        captured.extra,  # type: ignore[arg-type]
                        type(captured.extra),
                        budget,
                    )
                )
                captured.extra_entries = extra_entries
                for _, item in extra_entries:
                    _capture_native_graph(item, inventory, depth + 1)

            fields = {}
            for key, item in stored_entries:
                fields[key], height = _child(
                    item, active_containers, inventory, below, height, again
                )
            if captured.malformed_extra:
                raise ValueError("contract extra storage must be a mapping")
            if captured.extra is not None:
                extra_keys = {key for key, _ in extra_entries}
                reserved_names = stored_names | declared_names | alias_roots
                if reserved_names & extra_keys:
                    raise ValueError(
                        "contract input contains conflicting declared and extra fields"
                    )
                if len(extra_keys) != len(extra_entries):
                    raise ValueError("contract extra storage contains duplicate keys")
            for key, item in extra_entries:
                fields[key], height = _child(
                    item, active_containers, inventory, below, height, again
                )
            return fields, height

        if _is_subclass(value_type, tuple):
            exact = value_type is tuple
            items = (
                inventory.sequences[value_id]
                if exact
                else _exposed_items(value, inventory, depth)
            )
            copies = []
            for item in items:
                copied, height = _child(
                    item, active_containers, inventory, below, height, exact and again
                )
                copies.append(copied)
            return tuple(copies), height
        if _is_subclass(value_type, list):
            exact = value_type is list
            items = (
                inventory.sequences[value_id]
                if exact
                else _exposed_items(value, inventory, depth)
            )
            copies = []
            for item in items:
                copied, height = _child(
                    item, active_containers, inventory, below, height, exact and again
                )
                copies.append(copied)
            return copies, height
        if value_id in inventory.colliding_dictionaries:
            raise ValueError("contract input contains colliding keys")
        if value_id not in inventory.pure_dictionaries:
            inventory.impure_reads += 1  # rebuilding it hashes keys that can run code
        entries = inventory.dictionaries[value_id]
        rebuilt = {}
        for key, item in entries:
            rebuilt[key], height = _child(
                item, active_containers, inventory, below, height, again
            )
        return rebuilt, height
    finally:
        active_containers.remove(value_id)


def snapshot_contract_graph(
    value: object, active_containers: set[int] | None = None
) -> object:
    """Reconstruct stored contract graphs while rejecting structural defects.

    Malformed extra storage, declared/extra overlap, and cycles are rejected.
    This is neither schema validation nor authorization: callers must strictly
    reconstruct the intended target model before trusting the result.

    Stored models and exact built-in containers are fenced from their native
    storage before any user callback runs.  A tuple or list subclass is
    different: its public iteration is the authoritative view and is read once
    for each edge that reaches it, when that edge is first observed.  A child
    that exists only in such a public view cannot be fenced earlier, so an
    earlier callback can change it and it is then exposed in its changed form.
    No view is read ahead of the callbacks that precede it.  Once a view is
    read, the native backing of every item it exposes is frozen before the
    first item is reconstructed, and all captured objects stay referenced until
    reconstruction finishes.
    """

    if active_containers is None:
        active_containers = set()
    inventory = _GraphInventory()  # one discovery budget for the whole call
    _capture_native_graph(value, inventory)
    return _snapshot_captured_graph(value, active_containers, inventory)


__all__ = [
    "ContractModel",
    "MAX_SAFE_INTEGER",
    "OpaqueId",
    "SafeCounter",
    "VersionedContract",
    "snapshot_contract_graph",
]
