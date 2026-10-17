"""Strict primitives shared by candidate product contracts.

Contract models reject unknown fields and coercion when validating Python
objects.  JSON validation accepts only the corresponding JSON scalar types.
``model_dump()`` is the native Python representation; ``model_dump_json()``
is the wire representation.  Neither representation grants identity or
authority merely because a value validates.
"""

from collections.abc import Iterable, Mapping
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
)
from pydantic.fields import FieldInfo


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


def _exact_entry(items: Iterable[tuple[object, object]], name: str) -> tuple[object, bool, bool]:
    """``(value, found, pure)`` for ``name`` among natively read ``items``, compared as exact text.

    No key's own hash or equality runs.  ``pure`` is False when a key that is not an exact ``str`` is
    present: a lookup through the mapping could have run its hooks, so the subtree is not reused.  An
    exact key wins over a ``str``-subclass key with the same text.
    """

    value: object = None
    found = exact = False
    pure = True
    for key, item in items:
        if type(key) is str:
            if key == name:
                value, found, exact = item, True, True
        else:
            pure = False
            if not exact and _is_subclass(type(key), str) and str.__str__(key) == name:
                value, found = item, True
    return value, found, pure


def _class_namespace_entry(model_type: type, name: str, budget: DiscoveryBudget) -> tuple[object, bool]:
    """``(value, pure)``: ``name`` from the first class on ``model_type``'s native MRO whose own
    namespace defines it.

    The namespaces are read directly (their native items, compared as exact text), so no metaclass
    ``__getattribute__``, descriptor or key hook runs.
    """

    pure = True
    for klass in _TYPE_MRO.__get__(model_type):
        namespace = _CLASS_NAMESPACE.__get__(klass)
        budget.charge_edge(MappingProxyType.__len__(namespace))  # every entry scanned is charged
        value, found, klass_pure = _exact_entry(MappingProxyType.items(namespace), name)
        pure = pure and klass_pure
        if found:
            return value, pure
    return None, pure


def _native_sequence(value: object, budget: DiscoveryBudget) -> tuple[tuple[object, ...], bool]:
    """``(items, pure)`` of a list or tuple read through its native storage (no iteration, length or
    indexing hook runs).  Its native length is charged before anything is copied, so an oversized one is
    refused without the copy.  A subclass is still read natively but reported impure; anything else is
    unreadable."""

    if _is_subclass(type(value), list):
        budget.charge_edge(list.__len__(value))  # type: ignore[arg-type]
        return tuple(list.__iter__(value)), type(value) is list  # type: ignore[arg-type]
    if _is_subclass(type(value), tuple):
        budget.charge_edge(tuple.__len__(value))  # type: ignore[arg-type]
        return (value if type(value) is tuple else tuple(tuple.__iter__(value))), type(value) is tuple  # type: ignore[return-value,arg-type]
    return (), False


def _text(value: object) -> object:
    """A ``str`` subclass as its exact text (no hook of it runs); anything else unchanged."""

    return str.__str__(value) if type(value) is not str and _is_subclass(type(value), str) else value


def _model_metadata(
    model_type: type[BaseModel], budget: DiscoveryBudget
) -> tuple[tuple[tuple[object, object], ...], object, bool]:
    """The model's declared field entries and configuration, read without caller hooks.

    pydantic keeps both as plain dicts in every model class's own namespace.  Anything else
    (a replaced or exotic store) is read through the class as before, and reported as impure:
    reading it may run caller code, so the subtree must not be reused.  The declared entries are
    charged before they are copied, and the copy is what every later scan reads: a hook that runs
    meanwhile cannot change them, and a large table costs its size wherever it is read.
    """

    fields, fields_pure = _class_namespace_entry(model_type, "__pydantic_fields__", budget)
    config, config_pure = _class_namespace_entry(model_type, "model_config", budget)
    pure = fields_pure and config_pure
    if type(fields) is not dict or (config is not None and type(config) is not dict):
        fields, config, pure = model_type.model_fields, model_type.model_config, False
        if not _is_subclass(type(fields), dict):
            if not _is_subclass(type(fields), Mapping):
                raise ValueError("contract model fields must be a mapping")
            return _mapping_entries(fields, type(fields), budget), config, pure  # each entry charged as read
    budget.charge_edge(dict.__len__(fields))  # type: ignore[arg-type]
    return tuple(dict.items(fields)), config, pure  # type: ignore[arg-type]


def _declared_metadata(
    model_type: type[BaseModel], inventory: "_GraphInventory"
) -> tuple[set[object], dict[str, set[str]]]:
    """``(declared names, alias owners)`` of ``model_type``: the call's one snapshot of the class.

    Taken the first time the call meets the class, during capture (which runs no caller code), charged as
    read, and kept with the class for the rest of the call.  Every reader (every instance, on every path)
    uses this one snapshot, so caller code that runs later in the call (a public view, a key hook, an extra
    mapping) cannot change how any instance is judged, and the class is read once however many instances
    it has.  A declared name that is not text is never hashed.
    """

    cached = inventory.metadata.get(id(model_type))
    if cached is not None and cached[0] is model_type:
        return cached[1], cached[2]
    budget = inventory.budget
    declared_fields, config, _pure = _model_metadata(model_type, budget)
    declared_names: set[object] = set()
    for name, _field in declared_fields:  # the charged copy, never the live table
        if _is_subclass(type(name), str):
            declared_names.add(_text(name))
    alias_owners, _aliases_pure = _validation_alias_owners(declared_fields, config, budget)
    inventory.metadata[id(model_type)] = (model_type, declared_names, alias_owners)
    return declared_names, alias_owners


def _validation_alias_owners(
    fields: tuple[tuple[object, object], ...], config: object, budget: DiscoveryBudget
) -> tuple[dict[str, set[str]], bool]:
    """Each validation alias (or alias-path root) as exact text, with the fields it belongs to.

    A ``str``-subclass alias is read as its text, so hashing it runs no hook; the configuration and
    the alias storage (``choices``, ``path``) are read natively and the copies charged.  The second
    value is False when reading may have run caller code (a subclass of a pydantic alias or field type,
    hook-capable storage or keys), which makes the subtree ineligible for reuse.
    """

    pure = type(config) is dict or config is None
    if type(config) is dict:
        budget.charge_edge(dict.__len__(config))
        validate_by_alias, _found, config_pure = _exact_entry(dict.items(config), "validate_by_alias")
        pure = pure and config_pure
    else:
        validate_by_alias = None if config is None else config.get("validate_by_alias")  # type: ignore[attr-defined]
    if validate_by_alias is False:
        return {}, pure

    owners: dict[str, set[str]] = {}
    budget.charge_edge(len(fields))  # this walk scans every declared field again: charged again
    for field_name, field in fields:  # the copy: a hook run by a field or alias cannot change what is walked
        if not _is_subclass(type(field_name), str):
            pure = False  # a non-text name can hash or compare through caller code: never hashed, never reused
            continue
        if type(field) is not FieldInfo:
            pure = False
        validation_alias = field.validation_alias  # type: ignore[attr-defined]
        if _is_subclass(type(validation_alias), AliasChoices):
            pure = pure and type(validation_alias) is AliasChoices
            choices, choices_pure = _native_sequence(validation_alias.choices, budget)
            pure = pure and choices_pure
        else:
            choices = (validation_alias,)
        name = _text(field_name)
        for choice in choices:
            if _is_subclass(type(choice), str):
                owners.setdefault(_text(choice), set()).add(name)  # type: ignore[arg-type]
            elif _is_subclass(type(choice), AliasPath):
                pure = pure and type(choice) is AliasPath
                path, path_pure = _native_sequence(choice.path, budget)
                pure = pure and path_pure
                root = path[0] if path else None
                if _is_subclass(type(root), str):
                    owners.setdefault(_text(root), set()).add(name)  # type: ignore[arg-type]
    return owners, pure


_TYPE_MRO = vars(type)["__mro__"]
_CLASS_NAMESPACE = vars(type)["__dict__"]
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
    # The call's one metadata snapshot per model class, taken at capture: ``id -> (class, declared names, alias owners)``.
    metadata: dict[int, tuple[type, set[object], dict[str, set[str]]]] = field(default_factory=dict)
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
        # The class's metadata is snapshotted at its first native encounter, before any caller code runs
        # (capture runs none); reconstruction reads only this snapshot.
        _declared_metadata(value_type, inventory)
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

    One nesting level is exactly one call of this function: children are rebuilt in
    plain loops that call it directly (no helper, closure or generator), so the stack
    a snapshot needs is bounded by the tier's depth, whatever the caller's own depth.
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
            # Declared fields, configuration and aliases are read without running caller code; where
            # that is not possible the subtree (and every ancestor) is not reused.
            # The class's snapshot for this call, taken at capture (before any caller code ran); reading
            # it runs nothing, so it never makes this subtree impure.
            declared_names, alias_owners = _declared_metadata(model_type, inventory)
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
                    late_height = _capture_native_graph(item, inventory, below) + 1
                    _grow_height(inventory, value_id, late_height)

            fields: dict[object, object] = {}
            for key, item in stored_entries:
                if again:
                    budget.charge_edge()
                fields[key] = _snapshot_captured_graph(item, active_containers, inventory, below)
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
            items = inventory.sequences[value_id] if exact else _exposed_items(value, inventory, depth)
            copies: list[object] = []
            for item in items:
                if exact and again:
                    budget.charge_edge()
                copies.append(_snapshot_captured_graph(item, active_containers, inventory, below))
                height = _taller(height, item, completed)
            result = tuple(copies) if is_tuple else copies
        else:
            if value_id in inventory.colliding_dictionaries:
                raise ValueError("contract input contains colliding keys")
            if value_id not in inventory.pure_dictionaries:
                inventory.impure_reads += 1  # rebuilding it hashes keys that can run code
            rebuilt: dict[object, object] = {}
            for key, item in inventory.dictionaries[value_id]:
                if again:
                    budget.charge_edge()
                rebuilt[key] = _snapshot_captured_graph(item, active_containers, inventory, below)
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
    try:
        _capture_native_graph(value, inventory)
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
