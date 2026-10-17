"""Strict primitives shared by candidate product contracts.

Contract models reject unknown fields and coercion when validating Python
objects.  JSON validation accepts only the corresponding JSON scalar types.
``model_dump()`` is the native Python representation; ``model_dump_json()``
is the wire representation.  Neither representation grants identity or
authority merely because a value validates.
"""

import enum
import functools
import typing
from collections.abc import Callable, Mapping
from datetime import datetime
from types import MappingProxyType
from dataclasses import dataclass, field
from typing import Annotated, Literal

from contracts_limits import DISCOVERY_BUDGET_MESSAGE, STANDARD, DiscoveryBudget, is_exact_type
from pydantic import (
    AliasChoices,
    AliasPath,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    model_validator,
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


def is_canonical_enum_member(value: object) -> bool:
    """Whether an enum instance is one of its class's declared members, decided by identity.

    ``str.__new__(MyEnum, "value")`` builds an object that passes ``isinstance(value, MyEnum)`` (and
    pydantic's strict enum check) but is no member: every ``value is MyEnum.MEMBER`` test then fails,
    so a decision written as a chain of identity tests silently falls through to its default branch.
    """

    for member in type(value).__members__.values():  # type: ignore[attr-defined]
        if member is value:
            return True
    return False


def require_canonical_enum_members(value: object, *, _depth: int = 0) -> None:
    """Raise ``ValueError`` if ``value`` is, or natively contains, an enum instance that is no member.

    Looks through exact tuples, lists, sets, frozensets and dicts (what validation produces); nested
    models validate themselves.  A container reached by several paths is checked once (the check keeps
    what it has seen, alive, for the call), so a graph that shares its layers costs its objects, not its
    paths.
    """

    seen: dict[int, object] = {}
    _scan_enum_members(value, _depth, seen)


def _scan_enum_members(value: object, depth: int, seen: dict[int, object]) -> None:
    if any(base is enum.Enum for base in _TYPE_MRO.__get__(type(value))):
        if not is_canonical_enum_member(value):
            raise ValueError("an enum value must be a declared member of its enum")
        return
    kind = type(value)
    if not (kind is dict or kind is tuple or kind is list or kind is set or kind is frozenset):  # identity
        return
    if depth > 32:
        raise ValueError("contract value nesting exceeds the enum check depth")
    if id(value) in seen:
        return
    seen[id(value)] = value  # kept alive: its id stays its own
    if kind is dict:
        for key, item in dict.items(value):  # type: ignore[arg-type]
            _scan_enum_members(key, depth + 1, seen)
            _scan_enum_members(item, depth + 1, seen)
    else:
        for item in value:  # type: ignore[attr-defined]
            _scan_enum_members(item, depth + 1, seen)


def _annotation_holds_enum(annotation: object, _depth: int = 0) -> bool:
    if _depth > 8:
        return True  # unknown shape: check it
    if isinstance(annotation, type):
        return any(base is enum.Enum for base in _TYPE_MRO.__get__(annotation))
    return any(_annotation_holds_enum(arg, _depth + 1) for arg in typing.get_args(annotation))


class ContractModel(BaseModel):
    """Base for strict contracts with a closed set of fields."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        revalidate_instances="always",
    )

    _enum_field_names: typing.ClassVar[tuple[str, ...]] = ()

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: typing.Any) -> None:
        super().__pydantic_init_subclass__(**kwargs)
        cls._enum_field_names = tuple(
            name for name, info in cls.model_fields.items() if _annotation_holds_enum(info.annotation)
        )

    @model_validator(mode="after")
    def _enum_members_are_canonical(self) -> "ContractModel":
        # Every enum-typed field must hold a declared member (see ``is_canonical_enum_member``): a
        # look-alike passes the type check and then falls through every identity comparison.
        stored = self.__dict__
        for name in type(self)._enum_field_names:
            if name in stored:
                require_canonical_enum_members(stored[name])
        return self


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

    Returns ``(entries, pure, colliding)``.  Every ``str``-subclass key is replaced by its text
    (``str.__str__``, which runs no hook), so nothing later hashes or compares the subclass.
    ``pure`` is false when a key of another non-plain type remains (rebuilding a dict hashes it,
    which runs its code).  ``colliding`` is set when two keys share one text; the entries keep
    both, and the caller refuses them where its established diagnostics expect it (a plain dict
    when it is rebuilt, model names after their fields, extras through the duplicate-key check).
    """

    plain: list[tuple[object, object]] = []
    texts: set[str] = set()
    pure = True
    colliding = False
    for key, item in entries:
        if not is_exact_type(key, *_PLAIN_KEY_TYPES):
            if _is_subclass(type(key), str):
                key = str.__str__(key)  # type: ignore[arg-type]
            else:
                pure = False
        if type(key) is str:
            if key in texts:
                colliding = True
            texts.add(key)  # type: ignore[arg-type]
        plain.append((key, item))
    return tuple(plain), pure and not colliding, colliding


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
    colliding_names: bool = False


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
    # The native height of every captured container: how many container levels lie below it
    # along native edges (0 for one holding no container).  A subtree reached again by a deeper
    # path is checked against the depth limit with this, not skipped, so the limit does not
    # depend on which path reached the shared subtree first.
    native_heights: dict[int, int] = field(default_factory=dict)
    # The deepest level at which each container was reached on any path, in capture or in
    # reconstruction.  Extras found late (through a public mapping) are checked at this depth,
    # not at the depth of the path that happened to find them.
    reached: dict[int, int] = field(default_factory=dict)
    # Native parents of each container (the containers holding it as a stored entry or extra), so
    # a height found late can be carried to every ancestor and checked at that ancestor's depth.
    parents: dict[int, set[int]] = field(default_factory=dict)
    # Strong references keep every captured object alive for the whole
    # reconstruction, so a released temporary's id cannot be reused by a later
    # object and alias its captured contents.
    retained: list[object] = field(default_factory=list)
    # An optional function applied to every ``datetime`` leaf the moment its entry is captured (before any public
    # iterator is advanced), and the results by identity: ``(the caller's datetime, its pinned copy)``.  The
    # original stays referenced, so its id cannot be reused by another object during the call.
    pin_leaf: "Callable[[object], object] | None" = None
    pinned: dict[int, tuple[object, object]] = field(default_factory=dict)
    # While the initial native capture runs, a datetime is only recorded here (pinning asks its zone, which is
    # caller code, and every native backing must be frozen before any caller code runs); the recorded ones are
    # pinned once that capture is complete, and any later one (captured as a public view yields it) at once.
    collecting: bool = True
    collected: list[object] = field(default_factory=list)
    # An optional check applied to each snapshotted entry of a public view before it is validated eagerly (the
    # receiving boundary's supported-type and enum check): a refusal ends the read like an invalid entry.
    entry_guard: "Callable[[object], object] | None" = None
    # The contract each root is validated against by the receiving boundary, by object identity (see ``contract``).
    contracts: dict[int, type] = field(default_factory=dict)
    # Snapshots already validated early, by identity (held, so the identity cannot be reused): a pure subtree
    # is reconstructed once and shared, so the same item yielded many times is judged once, not once per occurrence.
    validated: dict[int, object] = field(default_factory=dict)
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
    rebuilt_containers: set[int] = field(default_factory=set)


def _reach(inventory: _GraphInventory, value_id: int, depth: int) -> None:
    """Record that a container was reached at ``depth`` (a leaf call: it adds no nesting frame)."""

    if depth > inventory.reached.get(value_id, -1):
        inventory.reached[value_id] = depth


def _adopt(inventory: _GraphInventory, parent_id: int, item: object) -> None:
    """Record ``parent_id`` as a native parent of ``item`` when the item is a container."""

    if _is_subclass(type(item), (BaseModel, tuple, list, dict)):
        inventory.parents.setdefault(id(item), set()).add(parent_id)


def _grow_height(inventory: _GraphInventory, value_id: int, height: int) -> None:
    """Raise a container's native height and carry the increase to its ancestors.

    Each container whose height grows is checked against the depth limit at the deepest level it
    was reached on any path (``reached + height``), so a height found late is never judged at only
    the shallow path that found it.  Heights only increase and the check fails once one exceeds
    the limit, so the walk is bounded.
    """

    budget = inventory.budget
    work = [(value_id, height)]
    while work:
        node, grown = work.pop()
        if grown <= inventory.native_heights.get(node, 0):
            continue
        inventory.native_heights[node] = grown
        if inventory.reached.get(node, 0) + grown > budget.tier.max_depth:
            budget.charge_edge(budget.tier.max_edges + 1)  # the stable budget error
        for parent in inventory.parents.get(node, ()):
            work.append((parent, grown + 1))


def _capture_native_graph(value: object, inventory: _GraphInventory, depth: int = 0) -> int:
    """Freeze supported native backing without invoking public protocols.

    Every captured node and entry is charged to the call's budget, and the depth is checked
    along every path, so a deep or wide input is the stable budget error, never a crash.
    Returns the container's native height (``-1`` for a value that is not a container), which
    a later, deeper visit of the same container uses to check the depth of everything below it.
    """

    value_type = type(value)
    if not _is_subclass(value_type, (BaseModel, tuple, list, dict)):
        if inventory.pin_leaf is not None and _is_subclass(value_type, datetime) and id(value) not in inventory.pinned:
            if inventory.collecting:
                inventory.collected.append(value)  # pinned when the native capture is complete
            else:
                inventory.pinned[id(value)] = (value, inventory.pin_leaf(value))  # pinned now, read once
        return -1
    budget = inventory.budget
    if depth > budget.tier.max_depth:
        budget.charge_edge(budget.tier.max_edges + 1)  # the stable budget error

    value_id = id(value)
    _reach(inventory, value_id, depth)
    if value_id in inventory.visited:
        known = inventory.native_heights.get(value_id)
        if known is None:
            return 0  # still being captured: a cycle, refused when it is rebuilt
        if depth + known > budget.tier.max_depth:
            budget.charge_edge(budget.tier.max_edges + 1)  # a deeper path reaches a deep subtree
        return known
    budget.charge_node(value)
    inventory.visited.add(value_id)
    inventory.retained.append(value)
    below = depth + 1

    if _is_subclass(value_type, BaseModel):
        stored = _BASE_MODEL_DICT_DESCRIPTOR.__get__(value, BaseModel)
        _precheck(budget, dict.__len__(stored))
        stored_entries, pure_stored, stored_colliding = _plain_keys(tuple(dict.items(stored)))
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
            colliding_names=stored_colliding,
        )
        inventory.models[value_id] = model_inventory

        # Capture every model reachable through native stored backing before
        # classifying extras or hashing any stored name.
        height = 0  # plain loops, not a generator: one stack frame per level
        for _, item in stored_entries:
            _adopt(inventory, value_id, item)
            below_height = _capture_native_graph(item, inventory, below) + 1
            if below_height > height:
                height = below_height
        if extra is None:
            model_inventory.extra_entries = ()
            inventory.native_heights[value_id] = height
            return height
        extra_type = type(extra)
        if _is_subclass(extra_type, dict):
            _precheck(budget, dict.__len__(extra))
            extra_entries, pure_extra, _ = _plain_keys(tuple(dict.items(extra)))
            budget.charge_edge(len(extra_entries))
            model_inventory.extra_entries = extra_entries
            model_inventory.pure_names = model_inventory.pure_names and pure_extra
            for _, item in extra_entries:
                _adopt(inventory, value_id, item)
                below_height = _capture_native_graph(item, inventory, below) + 1
                if below_height > height:
                    height = below_height
        elif not _is_subclass(extra_type, Mapping):
            model_inventory.extra_entries = ()
            model_inventory.malformed_extra = True
        # A non-dict Mapping retains its established public items protocol.
        # Its inventory is captured later, after all natively reachable model
        # backing has already been frozen.
        inventory.native_heights[value_id] = height
        return height

    if _is_subclass(value_type, tuple):
        _precheck(budget, tuple.__len__(value))
        items = tuple(tuple.__iter__(value))
        budget.charge_edge(len(items))
        inventory.sequences[value_id] = items
        height = 0
        for item in items:
            _adopt(inventory, value_id, item)
            below_height = _capture_native_graph(item, inventory, below) + 1
            if below_height > height:
                height = below_height
        inventory.native_heights[value_id] = height
        return height

    if _is_subclass(value_type, list):
        _precheck(budget, list.__len__(value))
        items = tuple(list.__iter__(value))
        budget.charge_edge(len(items))
        inventory.sequences[value_id] = items
        height = 0
        for item in items:
            _adopt(inventory, value_id, item)
            below_height = _capture_native_graph(item, inventory, below) + 1
            if below_height > height:
                height = below_height
        inventory.native_heights[value_id] = height
        return height

    _precheck(budget, dict.__len__(value))
    entries, pure, colliding = _plain_keys(tuple(dict.items(value)))
    budget.charge_edge(len(entries))
    inventory.dictionaries[value_id] = entries
    if pure:
        inventory.pure_dictionaries.add(value_id)
    if colliding:
        inventory.colliding_dictionaries.add(value_id)
    height = 0
    for _, item in entries:
        _adopt(inventory, value_id, item)
        below_height = _capture_native_graph(item, inventory, below) + 1
        if below_height > height:
            height = below_height
    inventory.native_heights[value_id] = height
    return height


def _private_copy(value: object, limit: int) -> object:
    """A structural copy of a snapshot (exact dicts, lists and tuples rebuilt; every other value shared as it is).

    The snapshot stores an identical subtree once and shares it, so the copy does the same: each distinct container
    is copied once (a memo by identity, holding the original so its identity cannot be reused) and shared in the
    copy as it is in the snapshot, which makes the work linear in the snapshot's distinct nodes.  The strict
    validation that follows walks a shared subtree once per path, so the expanded size (nodes counted per path,
    computed from the same memo) is held to ``limit``, the tier's edge limit: a snapshot that would cost more to
    validate is refused with the stable budget error before it is copied further or validated.  No value's own
    copy hook runs, and the snapshot's depth is already bounded.
    """

    memo: dict[int, tuple[object, object, int]] = {}

    def copy(node: object) -> tuple[object, int]:
        kind = type(node)
        if kind is not dict and kind is not list and kind is not tuple:
            return node, 1
        known = memo.get(id(node))
        if known is not None and known[0] is node:
            return known[1], known[2]
        size = 1
        if kind is dict:
            built: dict[object, object] = {}
            for key, item in dict.items(node):
                built[key], item_size = copy(item)
                size += item_size
            result: object = built
        else:
            parts = []
            for item in (list.__iter__(node) if kind is list else tuple.__iter__(node)):  # type: ignore[arg-type]
                part, item_size = copy(item)
                parts.append(part)
                size += item_size
            result = parts if kind is list else tuple(parts)
        if size > limit:
            raise ValueError(DISCOVERY_BUDGET_MESSAGE)
        memo[id(node)] = (node, result, size)
        return result, size

    return copy(value)[0]


def _pin_collected(inventory: _GraphInventory) -> None:
    """Pin, once each, the datetimes recorded while a capture was running (their zones are asked now)."""

    pending = list(inventory.collected)
    inventory.collected.clear()
    for leaf in pending:
        if id(leaf) not in inventory.pinned:
            inventory.pinned[id(leaf)] = (leaf, inventory.pin_leaf(leaf))  # type: ignore[misc]


def _capture_entry(value: object, inventory: _GraphInventory, depth: int) -> int:
    """Capture an entry met after the initial capture: freeze all of its native backing, then pin its datetimes.

    Asking a zone is caller code, so no datetime in the entry is pinned until every part of the entry has been
    frozen: a zone callback cannot repair a later sibling of what it sits in before that is captured.
    """

    if inventory.pin_leaf is None or inventory.collecting:
        return _capture_native_graph(value, inventory, depth)
    inventory.collecting = True
    try:
        height = _capture_native_graph(value, inventory, depth)
    finally:
        inventory.collecting = False
    _pin_collected(inventory)
    return height


def _unwrapped(annotation: object) -> object:
    for _ in range(4):  # Annotated[...] and Optional[...] wrappers
        args = typing.get_args(annotation)
        origin = typing.get_origin(annotation)
        if origin is Annotated or (origin is typing.Union and len(args) == 2 and type(None) in args):
            annotation = args[0] if origin is Annotated else next(a for a in args if a is not type(None))
            continue
        break
    return annotation


@functools.lru_cache(maxsize=256)  # bounded: classes a caller creates dynamically cannot accumulate
def _field_names(model_type: type[BaseModel]) -> dict[str, str]:
    """Each name a dictionary may use for a field of ``model_type`` (the field's name and its string aliases) -> the field."""

    names: dict[str, str] = {}
    for name, info in model_type.model_fields.items():
        names[name] = name
        for alias in (info.alias, info.validation_alias):
            if type(alias) is str:
                names[alias] = name
    return names


@functools.lru_cache(maxsize=256)  # bounded: classes a caller creates dynamically cannot accumulate
def _declared_models(model_type: type[BaseModel]) -> dict[str, type]:
    """The fields of ``model_type`` declared as exactly one model class, mapped to that class.

    A value held there is judged by the declared class, not by its own (possibly narrower) class, because the
    contract the receiving helper validates against is the declared one.
    """

    found: dict[str, type] = {}
    for name, info in model_type.model_fields.items():
        element = _unwrapped(info.annotation)
        if isinstance(element, type) and issubclass(element, BaseModel):
            found[name] = element
    return found


@functools.lru_cache(maxsize=256)  # bounded: classes a caller creates dynamically cannot accumulate
def _item_models(model_type: type[BaseModel]) -> dict[str, type]:
    """The fields of ``model_type`` that hold a homogeneous sequence of one model class, mapped to that class.

    Only ``tuple[T, ...]`` and ``list[T]`` count.  A fixed tuple (``tuple[A, B]``, ``tuple[A]``) has a type
    and a length per position, so no single class describes its entries: it is left to the model's own
    validation, which sees the whole view.
    """

    found: dict[str, type] = {}
    for name, info in model_type.model_fields.items():
        annotation = info.annotation
        for _ in range(4):  # Annotated[...] and Optional[...] wrappers
            args = typing.get_args(annotation)
            origin = typing.get_origin(annotation)
            if origin is Annotated or (origin is typing.Union and len(args) == 2 and type(None) in args):
                annotation = args[0] if origin is Annotated else next(a for a in args if a is not type(None))
                continue
            break
        origin = typing.get_origin(annotation)
        args = typing.get_args(annotation)
        homogeneous = (origin is tuple and len(args) == 2 and args[1] is Ellipsis) or (
            origin is list and len(args) == 1
        )
        if homogeneous:
            element = args[0]
            if isinstance(element, type) and issubclass(element, BaseModel):
                found[name] = element
    return found


def _exposed_items(
    value: object,
    inventory: _GraphInventory,
    depth: int,
    active_containers: set[int] | None = None,
    item_model: "type[BaseModel] | None" = None,
    checked: "list[tuple[object, object]] | None" = None,
) -> tuple[object, ...]:
    """Read a sequence subclass's public view and freeze every sibling.

    The native backing of all exposed items is captured before the first one is
    reconstructed, so a callback run while reconstructing an earlier item cannot
    change a later sibling's captured state.  Each item is captured the moment the
    iterator yields it, before the iterator is advanced: a generator resumed to
    produce the next item cannot repair what it just exposed.  The observation and each
    item it yields are charged as they are read, so an endless view is cut off by the budget.

    When the view is a field of a model whose items are a known model (``item_model``), each yielded entry
    is also snapshotted, passed through the receiving boundary's ``entry_guard`` and strictly validated right then, before the iterator is advanced; the first invalid
    entry refuses the whole input (a prefix of the view is never returned in its place) and the tail is
    never produced.  The snapshots are kept in ``checked`` for the caller to reuse, so no entry is
    reconstructed twice.
    """

    budget = inventory.budget
    inventory.impure_reads += 1
    budget.charge_edge()
    iterator = iter(value)  # type: ignore[call-overload]
    items: list[object] = []
    inventory.retained.append(items)  # kept alive from the first item, whatever the iterator does next
    while True:
        try:
            item = next(iterator)
        except StopIteration:
            break
        budget.charge_edge()
        _capture_entry(item, inventory, depth + 1)
        items.append(item)
        if item_model is not None and active_containers is not None and checked is not None:
            snapshot = _snapshot_captured_graph(item, active_containers, inventory, depth + 1, None, item_model)
            checked.append((item, snapshot))  # by position: the same object yielded twice is two entries
            if inventory.validated.get(id(snapshot)) is snapshot:
                continue  # the same shared snapshot, already guarded and validated (immutable, unreachable by callers)
            try:
                if inventory.entry_guard is not None:
                    inventory.entry_guard(snapshot)  # the receiving boundary's type check, before pydantic sees it
                # A private copy: the validators of the declared class may edit what they are given, and the
                # retained snapshot is the evidence the caller will judge.
                item_model.model_validate(_private_copy(snapshot, budget.tier.max_edges), strict=True)
                inventory.validated[id(snapshot)] = snapshot
            except Exception:  # noqa: BLE001 - the input is refused; a prefix is never returned as if it were the view
                raise ValueError("a contract entry exposed by a public view is not valid") from None
    return tuple(items)


def _snapshot_captured_graph(
    value: object,
    active_containers: set[int],
    inventory: _GraphInventory,
    depth: int = 0,
    item_model: "type[BaseModel] | None" = None,
    contract: "type[BaseModel] | None" = None,
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

    One nesting level is exactly one call of this function: children are rebuilt in
    plain loops that call it directly (no helper, closure or generator), so the stack
    a snapshot needs is bounded by the tier's depth, whatever the caller's own depth.
    """

    value_type = type(value)
    if not _is_subclass(value_type, (BaseModel, tuple, list, dict)):
        known = inventory.pinned.get(id(value))
        return known[1] if known is not None and known[0] is value else value

    budget = inventory.budget
    if depth > budget.tier.max_depth:
        budget.charge_edge(budget.tier.max_edges + 1)  # the stable budget error
    value_id = id(value)
    if value_id not in inventory.visited:
        _capture_entry(value, inventory, depth)
    completed = inventory.completed
    cached = completed.get(value_id)
    if cached is not None and cached[0] is value:
        # The edge to a reused subtree is the parent's entry, already charged.
        if depth + cached[2] > budget.tier.max_depth:
            budget.charge_edge(budget.tier.max_edges + 1)
        return cached[1]
    if value_id in active_containers:
        raise ValueError("cyclic contract input is not supported")

    reads_before = inventory.impure_reads
    below = depth + 1
    height = 0
    # Capturing an entry and rebuilding it once is one traversal, charged once (by capture);
    # a container rebuilt again charges every entry each time.  Public-view items are
    # charged when the view is read.
    again = value_id in inventory.rebuilt_containers
    inventory.rebuilt_containers.add(value_id)
    active_containers.add(value_id)
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
                extra_entries, extra_pure, _colliding = _plain_keys(
                    _mapping_entries(
                        captured.extra,  # type: ignore[arg-type]
                        type(captured.extra),
                        budget,
                    )
                )
                captured.extra_entries = extra_entries
                # Later rebuilds reuse the saved entries but still hash these keys: remember that a key of
                # a non-plain type ran code, so the model (and every ancestor) stays ineligible for reuse,
                # exactly as when the extras are a native dict.
                captured.pure_names = captured.pure_names and extra_pure
                # Extras first met through the public view: their height is carried to this model and every
                # recorded ancestor, each checked at the deepest level it was reached on any path (a hidden
                # occurrence may be far deeper than the path that found them).
                for _, item in extra_entries:
                    _adopt(inventory, value_id, item)
                    late_height = _capture_entry(item, inventory, below) + 1
                    _grow_height(inventory, value_id, late_height)

            fields: dict[object, object] = {}
            # Early validation of a view's entries uses the contract the receiving helper validates against (the
            # declared one, from the root down), never a narrower class the caller's own object happens to have.
            declared = contract if contract is not None else inventory.contracts.get(value_id)
            if declared is not None and _is_subclass(model_type, declared):
                item_models = _item_models(declared)
                declared_children = _declared_models(declared)
            else:
                # No trusted contract names this model: nothing is validated early, because the only class
                # to do it with would be the caller's own, whose validators are caller code.
                item_models = declared_children = {}
            for key, item in stored_entries:
                if again:
                    budget.charge_edge()
                fields[key] = _snapshot_captured_graph(
                    item, active_containers, inventory, below,
                    item_models.get(key) if item_models else None, declared_children.get(key),
                )
                height = _taller(height, item, completed)
            if captured.colliding_names:
                raise ValueError("contract input contains colliding keys")
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
                if again:
                    budget.charge_edge()
                fields[key] = _snapshot_captured_graph(item, active_containers, inventory, below)
                height = _taller(height, item, completed)
            result: object = fields
        elif _is_subclass(value_type, (tuple, list)):
            is_tuple = _is_subclass(value_type, tuple)
            exact = value_type is tuple if is_tuple else value_type is list
            checked: list[tuple[object, object]] = []
            items = (
                inventory.sequences[value_id]
                if exact
                else _exposed_items(value, inventory, depth, active_containers, item_model, checked)
            )
            copies: list[object] = []
            for position, item in enumerate(items):
                if exact and again:
                    budget.charge_edge()
                done = checked[position] if position < len(checked) else None
                if done is not None and done[0] is item:
                    copies.append(done[1])  # snapshotted and validated when it was yielded
                else:
                    copies.append(_snapshot_captured_graph(item, active_containers, inventory, below, None, item_model))
                height = _taller(height, item, completed)
            result = tuple(copies) if is_tuple else copies
        else:
            if value_id in inventory.colliding_dictionaries:
                raise ValueError("contract input contains colliding keys")
            if value_id not in inventory.pure_dictionaries:
                inventory.impure_reads += 1  # rebuilding it hashes keys that can run code
            rebuilt: dict[object, object] = {}
            # A dictionary that is to be validated as a model carries the same declared contract as that model's own
            # stored values do: its fields' entry models and declared child models are found by field name (or alias).
            declared = contract if contract is not None else inventory.contracts.get(value_id)
            if declared is not None and isinstance(declared, type) and _is_subclass(declared, BaseModel):
                item_models = _item_models(declared)
                declared_children = _declared_models(declared)
                lookup = _field_names(declared)
            else:
                item_models = declared_children = lookup = {}
            for key, item in inventory.dictionaries[value_id]:
                if again:
                    budget.charge_edge()
                field_name = lookup.get(key) if lookup and type(key) is str else None
                rebuilt[key] = _snapshot_captured_graph(
                    item, active_containers, inventory, below,
                    item_models.get(field_name) if field_name else None,
                    declared_children.get(field_name) if field_name else None,
                )
                height = _taller(height, item, completed)
            result = rebuilt
    finally:
        active_containers.remove(value_id)
    if inventory.impure_reads == reads_before:
        completed[value_id] = (value, result, height)
    return result


def _taller(height: int, item: object, completed: dict[int, tuple[object, object, int]]) -> int:
    """The parent's height after one child (a leaf call: it adds no nesting frame).

    A reused child knows its height; an impure child is never reused, and then neither is
    its parent, so the parent's height is not needed.
    """

    if not _is_subclass(type(item), (BaseModel, tuple, list, dict)):
        return height
    cached = completed.get(id(item))
    item_height = cached[2] if cached is not None and cached[0] is item else 0
    return max(height, item_height + 1)


def snapshot_contract_graph(
    value: object,
    active_containers: set[int] | None = None,
    *,
    pin_leaf: "Callable[[object], object] | None" = None,
    entry_guard: "Callable[[object], object] | None" = None,
    contract: "type[BaseModel] | tuple[type[BaseModel] | None, ...] | None" = None,
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

    ``pin_leaf``, when given, is applied once to every ``datetime`` leaf as its entry is captured (so before a
    public view that yielded it is advanced), and the snapshot holds the result instead of the caller's object.
    ``entry_guard`` is the receiving boundary's check of one snapshotted entry (supported types, allowed enums),
    run before an eager validation so that a value the boundary would refuse is never shown to pydantic.
    ``contract`` is the model class the root will be validated against (a tuple gives one per element of a tuple
    root): entries of a public view are validated early against the contract declared there, from the root down,
    never against a narrower class the caller's own object has.
    """

    if active_containers is None:
        active_containers = set()
    inventory = _GraphInventory(pin_leaf=pin_leaf, entry_guard=entry_guard)  # one discovery budget for the whole call
    if contract is not None:
        if type(contract) is tuple:  # one declared contract per element of an exact tuple root
            if type(value) is tuple and len(value) == len(contract):
                inventory.contracts.update({id(v): c for v, c in zip(value, contract) if c is not None})
        else:
            inventory.contracts[id(value)] = contract  # type: ignore[assignment]
    try:
        _capture_native_graph(value, inventory)
        inventory.collecting = False
        _pin_collected(inventory)  # all native backing is frozen: now the zones may be asked, once each
        return _snapshot_captured_graph(value, active_containers, inventory)
    except RecursionError:
        # One frame per level keeps every admitted depth far inside the interpreter's limit;
        # a caller already near that limit still gets the stable error, never a crash.
        raise ValueError(DISCOVERY_BUDGET_MESSAGE) from None


__all__ = [
    "ContractModel",
    "MAX_SAFE_INTEGER",
    "OpaqueId",
    "SafeCounter",
    "VersionedContract",
    "snapshot_contract_graph",
]
