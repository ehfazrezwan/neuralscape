"""Portable bundle manifest contracts and pure structural validation.

These models describe what an exporter claims is present.  Successful
validation does not prove that an export is complete, authorized, decryptable,
or safe to import, and it does not inspect archive entries or file contents.
Importers must separately verify every declared byte length and checksum,
enforce archive expansion/symlink limits, authorize identity mappings, and
perform migration transaction and replay checks.
"""

from __future__ import annotations

from pathlib import PurePosixPath, PureWindowsPath
from collections.abc import Callable, Iterable, Mapping as _AbcMapping
from types import MappingProxyType
from typing import Any, Literal, Mapping

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from contracts_common import ContractModel, OpaqueId, SafeCounter, VersionedContract
from contracts_limits import PORTABILITY, DiscoveryBudget, is_exact_type


ChecksumAlgorithm = Literal["sha256"]
EncryptionMode = Literal["plaintext_authorized_export", "encrypted"]
ScopeKind = Literal["tenant", "projects"]
_WINDOWS_FORBIDDEN_COMPONENT_CHARACTERS = frozenset('<>:"|?*')
_PYDANTIC_DICT_DESCRIPTOR = BaseModel.__dict__["__dict__"]
_PYDANTIC_EXTRA_SLOT = BaseModel.__dict__["__pydantic_extra__"]
_PYDANTIC_FIELDS_SET_SLOT = BaseModel.__dict__["__pydantic_fields_set__"]
_ModelStorageEntries = tuple[tuple[Any, Any], ...]
_ModelStorageSnapshot = tuple[
    BaseModel,
    _ModelStorageEntries,
    _ModelStorageEntries,
]
_ModelStorageInventory = dict[int, _ModelStorageSnapshot]
_ModelExtraSnapshot = tuple[bool, Any]
_ModelDiscoverySnapshot = tuple[
    BaseModel,
    _ModelStorageEntries,
    Any,
    _ModelExtraSnapshot,
]
_ModelDiscoveryInventory = dict[int, _ModelDiscoverySnapshot]
_NativeContainerSnapshot = tuple[Any, type, tuple[Any, ...]]
_NativeContainerInventory = dict[int, _NativeContainerSnapshot]
# Public views: owner retained with the observed items (or key/value pairs).
_PublicViewInventory = dict[int, tuple[Any, tuple[Any, ...]]]
_RetainedStateInventory = tuple[
    _ModelStorageInventory,
    _NativeContainerInventory,
    _PublicViewInventory,
]


class _CallState:
    """The discovery state of one validation call, shared by every model root it rebuilds.

    A mapping input can hold many model instances.  Rebuilding each under its own budget and
    memo let K instances do K budgets of work (and copy shared substructure once per instance);
    with one state per call they share one discovery envelope, one set of inventories and one
    reconstruction memo, exactly like the roots of a single top-level model.
    """

    __slots__ = (
        "budget", "visited", "discovered", "discovery_inventory", "model_inventory",
        "native_container_inventory", "public_view_inventory", "completed", "first_expansion",
    )

    def __init__(self) -> None:
        self.budget = DiscoveryBudget(PORTABILITY)
        self.visited: set[int] = set()
        self.discovered: set[int] = set()
        self.discovery_inventory: _ModelDiscoveryInventory = {}
        self.model_inventory: _ModelStorageInventory = {}
        self.native_container_inventory: _NativeContainerInventory = {}
        self.public_view_inventory: _PublicViewInventory = {}
        # Completed reconstructions, keyed by identity with the owner retained.
        self.completed: dict[int, tuple[Any, Any]] = {}
        # What a node held when discovery first expanded it (owner retained): a shallower path that reaches
        # it again re-expands these saved edges, never the live storage, which a callback may have changed.
        self.first_expansion: dict[int, tuple[Any, tuple[Any, ...]]] = {}
_MISSING_FIELDS_SET = object()


def _validated_model_discovery_snapshot(
    value: BaseModel,
    inventory: _ModelDiscoveryInventory,
) -> tuple[_ModelStorageEntries, Any, _ModelExtraSnapshot]:
    """Return the discovery snapshot belonging to this model instance."""

    try:
        owner, stored_entries, fields_set_storage, extra_snapshot = inventory[
            id(value)
        ]
    except KeyError as exc:
        raise ValueError(
            "contract model was not present in the validated graph"
        ) from exc
    if owner is not value:
        raise ValueError(
            "contract model was not present in the validated graph"
        )
    return stored_entries, fields_set_storage, extra_snapshot


def _validated_model_storage(
    value: BaseModel,
    inventory: _ModelStorageInventory,
) -> tuple[_ModelStorageEntries, _ModelStorageEntries]:
    """Return the validated storage belonging to this model instance."""

    try:
        owner, stored_entries, extra_entries = inventory[id(value)]
    except KeyError as exc:
        raise ValueError(
            "contract model was not present in the validated graph"
        ) from exc
    if owner is not value:
        raise ValueError(
            "contract model was not present in the validated graph"
        )
    return stored_entries, extra_entries


def _validated_native_container_items(
    value: Any,
    inventory: _NativeContainerInventory,
) -> tuple[Any, ...]:
    """Return the snapshot belonging to this exact container instance."""

    try:
        owner, value_type, items = inventory[id(value)]
    except KeyError as exc:
        raise ValueError(
            "contract container was not present in the validated graph"
        ) from exc
    if owner is not value or value_type is not type(value):
        raise ValueError(
            "contract container was not present in the validated graph"
        )
    return items


def _snapshot_model_extra_storage(
    value: BaseModel, budget: DiscoveryBudget | None = None
) -> _ModelExtraSnapshot:
    """Capture native dict extras without invoking public mapping views.

    When a ``budget`` is given, the native entry count is charged before the
    entries are copied so an oversized extra mapping is never materialized.
    """

    try:
        extra = _PYDANTIC_EXTRA_SLOT.__get__(value, type(value))
    except AttributeError:
        extra = None
    if extra is None:
        return True, ()
    if issubclass(type(extra), dict):
        if budget is not None:
            budget.charge_edge(dict.__len__(extra))
        return True, tuple(dict.items(extra))
    return False, extra


_TYPE_MRO_DESCRIPTOR = type.__dict__["__mro__"]
_MAPPING_BASES = (_AbcMapping, dict, MappingProxyType)
_TYPE_DICT_DESCRIPTOR = type.__dict__["__dict__"]


def _inherits(value: object, *bases: type) -> bool:
    """Whether ``value``'s concrete class inherits from one of ``bases``, decided by identity.

    ``isinstance`` reads ``value.__class__``, which an object can override to report any
    class (a plain iterable reporting ``list``), and the ABC machinery behind ``BaseModel``
    honours registration.  The concrete type's native MRO compared with ``is`` cannot be
    spoofed and keeps genuine subclasses.
    """

    for base in _TYPE_MRO_DESCRIPTOR.__get__(type(value)):
        for candidate in bases:
            if base is candidate:
                return True
    return False


def _is_mapping(value: object) -> bool:
    """Whether ``value`` is a mapping by inheritance, decided by identity only.

    ``isinstance(value, Mapping)`` runs the ABC machinery, which hashes and compares
    the value's class and honours virtual registration, so a class whose metaclass
    defines ``__eq__``/``__hash__`` could run code or be matched by spoofing.  This
    walks the class's native MRO and compares with ``is``: it recognises
    ``collections.abc.Mapping`` subclasses, ``dict`` subclasses and ``mappingproxy``;
    a class that is only *registered* as a Mapping is not treated as one.
    """

    for base in _TYPE_MRO_DESCRIPTOR.__get__(type(value)):
        for candidate in _MAPPING_BASES:
            if base is candidate:
                return True
    return False


def _native_declared_field_names(
    model_type: type[BaseModel],
) -> tuple[str, ...] | None:
    """Return ordered declared field names from raw class storage, or ``None``.

    ``model_fields`` is an ordinary class attribute that a subclass can replace
    with a mapping whose membership or iteration protocols run user code.  The
    names are instead read from the class ``__dict__`` entry Pydantic itself
    creates, and only when that entry is a real ``dict`` whose keys are read
    with ``dict.keys``.  Anything else returns ``None`` so callers stay
    conservative rather than trusting a callback-bearing mapping.
    """

    for klass in _TYPE_MRO_DESCRIPTOR.__get__(model_type):
        raw = _TYPE_DICT_DESCRIPTOR.__get__(klass)
        if "__pydantic_fields__" not in raw:
            continue
        fields = raw["__pydantic_fields__"]
        if not issubclass(type(fields), dict):
            return None
        names = tuple(dict.keys(fields))
        if any(type(name) is not str for name in names):
            return None
        return names
    return None


def _model_discovery_values(
    model_type: type[BaseModel],
    stored_entries: _ModelStorageEntries,
) -> tuple[Any, ...]:
    """Select stored values needing callback-free graph discovery."""

    declared = _native_declared_field_names(model_type)
    if declared is None:
        return tuple(item for _, item in reversed(stored_entries))
    declared_names = frozenset(declared)
    return tuple(
        item
        for name, item in reversed(stored_entries)
        if type(name) is not str or name in declared_names
    )


def _validated_model_extra_storage(
    extra_snapshot: _ModelExtraSnapshot,
    stored_names: set[Any],
    declared_names: set[Any],
    budget: DiscoveryBudget,
) -> tuple[dict[Any, Any], set[Any]]:
    """Capture stable Pydantic extra entries and all observed names.

    A custom (non-dict) extra mapping is observed through its public views; each
    yielded name or entry is charged to ``budget`` before it is retained, so an
    endless view stops at the budget instead of exhausting memory.
    """

    captured_natively, extra = extra_snapshot
    if captured_natively:
        captured = dict(extra)
        observed_names = set(captured)
    else:
        if not _is_mapping(extra):
            raise ValueError(
                "contract model extra storage must be None or a mapping"
            )
        iterated_names = []
        for name in extra:
            budget.charge_edge()
            iterated_names.append(name)
        entries = []
        for entry in extra.items():
            budget.charge_edge(2)
            # Unpack while collecting: an iterator posing as a pair is read for at most
            # three elements (then refused), and only a native two-tuple is retained.
            try:
                key, value = entry
            except (TypeError, ValueError) as exc:
                raise ValueError("contract model extra storage has a malformed entry") from exc
            entries.append((key, value))
        captured = dict(entries)
        observed_names = set(iterated_names) | set(captured)
    declared_or_stored = declared_names | stored_names
    overlap = declared_or_stored.intersection(observed_names)
    if overlap:
        names = ", ".join(sorted((repr(name) for name in overlap)))
        raise ValueError(
            f"contract model has conflicting stored and extra fields: {names}"
        )
    return captured, observed_names


def _reject_retained_unknown_fields(
    root: BaseModel,
    state: _CallState | None = None,
) -> _RetainedStateInventory:
    """Reject undeclared state retained by unchecked Pydantic copies.

    ``model_copy(update=...)`` deliberately does not validate its update.  For
    models configured with ``extra="forbid"``, an undeclared update can remain
    in ``__dict__``/``model_fields_set`` while ``model_dump()`` silently omits
    it.  First discover model-owned state through native container backing so
    every reachable model inventory and exact built-in container edge is
    frozen before callback-bearing legacy protocols run.  Container subclasses
    retain their public traversal authority.  Then validate and walk those
    snapshots so the receiving boundary cannot accept a later sanitized
    subset.  Object identity, rather than value equality, makes shared and
    cyclic containers safe to inspect.
    """

    errors: list[dict[str, Any]] = []
    # One native-discovery envelope for this validation call, shared by every
    # root it visits (every model instance of a mapping input passes the same
    # ``state``).  Accounting reads only native lengths and object identities,
    # so it never runs a caller-defined protocol.  Public views (container
    # subclasses and custom mappings) are observed once, here, and replayed by
    # reconstruction; they are never iterated a second time.
    state = state if state is not None else _CallState()
    budget = state.budget
    visited = state.visited
    discovered = state.discovered
    discovery_inventory = state.discovery_inventory
    model_inventory = state.model_inventory
    native_container_inventory = state.native_container_inventory
    public_view_inventory = state.public_view_inventory
    first_expansion = state.first_expansion

    def discover(value: Any, depth: int = 0) -> None:
        pending = [(value, depth)]
        while pending:
            current, current_depth = pending.pop()
            value_type = type(current)
            if not issubclass(
                value_type,
                (BaseModel, dict, list, tuple, set, frozenset),
            ):
                continue

            identity = id(current)
            # Unique nodes are charged once; a shallower path re-expands a node
            # so the depth outcome does not depend on traversal order.
            if not budget.visit(current, current_depth):
                continue
            discovered.add(identity)
            child_depth = current_depth + 1

            first = first_expansion.get(identity)
            if first is not None and first[0] is not current:
                first = None
            if issubclass(value_type, BaseModel):
                if first is not None:
                    # A shallower revisit: the first snapshot stands (its edges are charged again, as a
                    # re-expansion is), and nothing is read from the live model.
                    stored_entries = first[1][0]
                    budget.charge_edge(len(stored_entries))
                else:
                    native_stored = _PYDANTIC_DICT_DESCRIPTOR.__get__(
                        current, BaseModel
                    )
                    budget.charge_edge(dict.__len__(native_stored))
                    stored_entries = tuple(dict.items(native_stored))
                    try:
                        fields_set_storage = _PYDANTIC_FIELDS_SET_SLOT.__get__(
                            current, value_type
                        )
                    except AttributeError:
                        fields_set_storage = _MISSING_FIELDS_SET
                    if issubclass(type(fields_set_storage), set):
                        budget.charge_edge(set.__len__(fields_set_storage))
                        fields_set_storage = set.copy(fields_set_storage)
                    extra_snapshot = _snapshot_model_extra_storage(current, budget)
                    discovery_inventory[identity] = (
                        current,
                        stored_entries,
                        fields_set_storage,
                        extra_snapshot,
                    )
                    first_expansion[identity] = (current, (stored_entries,))
                # Exact strings can be classified from the class field mapping
                # without invoking a user-defined name protocol.  Other names
                # remain conservative candidates for later hash authority.
                pending.extend(
                    (item, child_depth)
                    for item in _model_discovery_values(value_type, stored_entries)
                )
                continue

            if issubclass(value_type, dict):
                if first is not None:
                    items = first[1]
                    budget.charge_edge(2 * len(items))
                else:
                    budget.charge_edge(2 * dict.__len__(current))
                    items = tuple(dict.items(current))
                    first_expansion[identity] = (current, items)
                if value_type is dict and first is None:
                    native_container_inventory[identity] = (
                        current,
                        dict,
                        items,
                    )
                for key, item in reversed(items):
                    pending.append((item, child_depth))
                    pending.append((key, child_depth))
                continue
            if first is not None:
                items = first[1]
                budget.charge_edge(len(items))
            else:
                if issubclass(value_type, list):
                    budget.charge_edge(list.__len__(current))
                    items = tuple(list.__iter__(current))
                elif issubclass(value_type, tuple):
                    budget.charge_edge(tuple.__len__(current))
                    items = tuple(tuple.__iter__(current))
                elif issubclass(value_type, set):
                    budget.charge_edge(set.__len__(current))
                    items = tuple(set.__iter__(current))
                else:
                    budget.charge_edge(frozenset.__len__(current))
                    items = tuple(frozenset.__iter__(current))
                first_expansion[identity] = (current, items)
            if first is None and is_exact_type(current, list, tuple, set, frozenset):
                native_container_inventory[identity] = (
                    current,
                    value_type,
                    items,
                )
            pending.extend((item, child_depth) for item in reversed(items))

    def walk(value: Any, location: tuple[str | int, ...]) -> None:
        if not (_inherits(value, BaseModel, list, tuple, set, frozenset) or _is_mapping(value)):
            return

        identity = id(value)
        if identity in visited:
            return
        visited.add(identity)
        if identity not in discovered:
            discover(value, len(location))

        if _inherits(value, BaseModel):
            declared = _native_declared_field_names(type(value))
            if declared is None:
                raise ValueError("contract model field storage is unavailable")
            stored_entries, fields_set_storage, extra_snapshot = (
                _validated_model_discovery_snapshot(
                    value, discovery_inventory
                )
            )
            stored = dict(stored_entries)
            if fields_set_storage is _MISSING_FIELDS_SET:
                raise ValueError(
                    "contract model fields-set storage is missing"
                )
            if is_exact_type(fields_set_storage, set):
                # Native set storage was copied, and charged by length, at discovery.
                fields_set = fields_set_storage
            else:
                # Any other storage is observed through its public iterator; each
                # yielded name is charged before it is retained.
                fields_set = set()
                for name in fields_set_storage:
                    budget.charge_edge()
                    fields_set.add(name)
            pydantic_extra, observed_extra_names = (
                _validated_model_extra_storage(
                    extra_snapshot,
                    set(stored),
                    set(declared),
                    budget,
                )
            )
            model_inventory[identity] = (
                value,
                stored_entries,
                tuple(dict.items(pydantic_extra)),
            )
            retained_names = (
                set(stored)
                | fields_set
                | observed_extra_names
            )

            for name in sorted(retained_names - set(declared), key=repr):
                retained_value = (
                    stored[name] if name in stored else pydantic_extra.get(name)
                )
                error_location = name if _inherits(name, str, int) else repr(name)
                errors.append(
                    {
                        "type": "extra_forbidden",
                        "loc": (*location, error_location),
                        "input": retained_value,
                    }
                )

            for name in declared:
                if name in stored:
                    walk(stored[name], (*location, name))
            return

        if type(value) is dict:
            items = _validated_native_container_items(
                value, native_container_inventory
            )
            for index, (key, item) in enumerate(items):
                walk(key, (*location, "<key>", index))
                item_location = key if _inherits(key, str, int) else index
                walk(item, (*location, item_location))
            return

        if _is_mapping(value):
            observed_pairs: list[tuple[Any, Any]] = []
            for index, (key, item) in enumerate(value.items()):
                budget.charge_edge(2)  # public view charged as it is observed
                observed_pairs.append((key, item))
                walk(key, (*location, "<key>", index))
                item_location = key if _inherits(key, str, int) else index
                walk(item, (*location, item_location))
            public_view_inventory[identity] = (value, tuple(observed_pairs))
            return

        exact_native = is_exact_type(value, list, tuple, set, frozenset)
        items = (
            _validated_native_container_items(value, native_container_inventory)
            if exact_native
            else value
        )
        observed_items: list[Any] = []
        for index, item in enumerate(items):
            if not exact_native:
                # A public view is charged as it is observed; an exact native
                # container was already charged by its length in discovery.
                budget.charge_edge()
                observed_items.append(item)
            walk(item, (*location, index))
        if not exact_native:
            public_view_inventory[identity] = (value, tuple(observed_items))

    discover(root)
    walk(root, ())
    if errors:
        raise ValidationError.from_exception_data("PortableManifest", errors)
    return model_inventory, native_container_inventory, public_view_inventory


def _reconstruct_retained_state(
    root: BaseModel,
    retained_state_inventory: _RetainedStateInventory,
    state: _CallState | None = None,
) -> dict[str, Any]:
    """Copy a model graph without invoking lossy Pydantic serialization.

    Pydantic serializes nested models according to their annotated field type
    by default, and even duck-typed serialization honors field exclusions.
    Reconstructing from stored state keeps every runtime subclass field visible
    to the closed destination schemas.  Container kinds are retained so this
    receiving boundary cannot turn an invalid tuple or set into a valid list.
    """

    model_inventory, native_container_inventory, public_view_inventory = (
        retained_state_inventory
    )
    active: set[int] = set()
    # Completed reconstructions, keyed by identity with the owner retained so the
    # key cannot be reused: shared substructure is rebuilt once, not once per path
    # (nor once per root, when the call's ``state`` is shared).
    completed: dict[int, tuple[Any, Any]] = state.completed if state is not None else {}

    def observed(value: Any) -> tuple[Any, ...]:
        try:
            owner, items = public_view_inventory[id(value)]
        except KeyError as exc:
            raise ValueError(
                "contract container was not present in the validated graph"
            ) from exc
        if owner is not value:
            raise ValueError(
                "contract container was not present in the validated graph"
            )
        return items

    def rebuild(value: Any) -> Any:
        if not (_inherits(value, BaseModel, list, tuple, set, frozenset) or _is_mapping(value)):
            return value

        identity = id(value)
        done = completed.get(identity)
        if done is not None and done[0] is value:
            return done[1]
        if identity in active:
            raise ValueError("contract input graph must be acyclic")
        active.add(identity)
        try:
            result = build(value)
        finally:
            active.remove(identity)
        completed[identity] = (value, result)
        return result

    def entries(items: Any) -> dict[Any, Any]:
        """A rebuilt mapping: a ``str``-subclass key is its plain text (its hash and equality never run), and
        two keys with one text are refused rather than one silently shadowing the other."""

        result: dict[Any, Any] = {}
        for key, item in items:
            key = rebuild(key)
            if type(key) is not str and _inherits(key, str):
                key = str.__str__(key)
            if type(key) is not str:
                # An object that is no text could hash and compare equal to a key already present and
                # replace its value while the first key is kept: the schema would never see the value
                # that was overwritten.  Mapping keys of a contract are text.
                raise ValueError("a mapping key must be text")
            if key in result:
                raise ValueError("two keys of one mapping share a text")
            result[key] = rebuild(item)
        return result

    def build(value: Any) -> Any:
        if _inherits(value, BaseModel):
            stored_entries, extra_entries = _validated_model_storage(
                value, model_inventory
            )
            stored = dict(stored_entries)
            stored.update(extra_entries)
            return {name: rebuild(item) for name, item in stored.items()}

        if type(value) is dict:
            items = _validated_native_container_items(
                value, native_container_inventory
            )
            return entries(items)
        if _is_mapping(value):
            return entries(observed(value))
        if is_exact_type(value, list, tuple, set, frozenset):
            items = _validated_native_container_items(
                value, native_container_inventory
            )
        else:
            items = observed(value)  # the first observation, never a second iteration
        if _inherits(value, list):
            return [rebuild(item) for item in items]
        if _inherits(value, tuple):
            return tuple(rebuild(item) for item in items)
        if _inherits(value, set):
            return {rebuild(item) for item in items}
        return frozenset(rebuild(item) for item in items)

    reconstructed = rebuild(root)
    if not isinstance(reconstructed, dict):
        raise ValueError("contract model reconstruction must produce a mapping")
    return reconstructed


def _normalized_bundle_path(value: str) -> str:
    """Return a canonical relative POSIX bundle path or raise ``ValueError``.

    Backslashes are rejected rather than treated as ordinary POSIX filename
    characters because a later Windows consumer could reinterpret them as path
    separators.  Windows device names are rejected in every path component so
    a consumer cannot resolve an apparent file to a device.  ``.`` and repeated
    separators are normalized so aliases can be detected across the complete
    file inventory.
    """

    if issubclass(type(value), str):
        value = str.__str__(value)  # the text itself: a subclass's own string methods never decide
    if not value or "\x00" in value:
        raise ValueError("bundle path must be nonempty and contain no NUL bytes")
    if "\\" in value:
        raise ValueError("bundle path must use POSIX separators")

    raw_components = value.split("/")
    path = PurePosixPath(value)
    if path.is_absolute() or value.startswith("/"):
        raise ValueError("bundle path must be relative")
    if any(part == ".." for part in path.parts):
        raise ValueError("bundle path must not traverse outside the bundle")

    normalized = path.as_posix()
    if normalized in {"", "."}:
        raise ValueError("bundle path must identify a file")
    if PureWindowsPath(normalized).drive:
        raise ValueError("bundle path must not be Windows drive-qualified")
    for component in raw_components:
        if component in {"", ".", ".."}:
            continue
        if component.endswith((".", " ")):
            raise ValueError(
                "bundle path components must not end with a dot or space"
            )
        if any(
            ord(character) < 32
            or character in _WINDOWS_FORBIDDEN_COMPONENT_CHARACTERS
            for character in component
        ):
            raise ValueError(
                "bundle path components must not contain Windows-forbidden characters"
            )
    if any(PureWindowsPath(part).is_reserved() for part in path.parts):
        raise ValueError(
            "bundle path must not contain a Windows reserved device name"
        )
    return normalized


def normalize_bundle_path(value: str) -> str:
    """Public canonical relative POSIX bundle-path normalizer.

    Raises ``ValueError`` for any path the manifest contract would reject.
    """

    return _normalized_bundle_path(value)


def names_below_a_file(
    names: Iterable[str], file_names: Iterable[str], fold: Callable[[str], str] = str
) -> list[str]:
    """The names that have a proper ancestor which is itself a file name, compared component by component.

    A regular file cannot also be the directory of another entry.  ``file_names`` are the regular files; every
    name in ``names`` (files and directories) is walked through a trie of them, so the work is linear in the
    total length of the names whatever their depth, and ``fold`` (identity, ``str.lower``, a Unicode
    casefold) is applied to each component, so folded aliases and implicit parents are caught too.
    """

    trie: dict[str, dict[str, Any]] = {}
    for name in file_names:
        node = trie
        for component in name.split("/"):
            node = node.setdefault(fold(component), {})
        node[""] = {}  # a file ends here (no real component is empty)
    below: list[str] = []
    for name in names:
        node = trie
        parts = name.split("/")
        for depth, component in enumerate(parts):
            node = node.get(fold(component))  # type: ignore[assignment]
            if node is None:
                break
            if depth < len(parts) - 1 and "" in node:
                below.append(name)
                break
    return below


class CanonicalObjectId(ContractModel):
    """An exact canonical object identity carried by the bundle inventory."""

    kind: Literal[
        "record",
        "revision",
        "derivation",
        "source",
        "passage",
        "graph_node",
        "graph_edge",
        "episode",
        "artifact",
        "intent",
        "policy",
        "group",
        "tombstone",
        "hold",
        "projection",
    ]
    id: OpaqueId


class ExportScope(ContractModel):
    """The tenant boundary and optional project subset claimed by an export."""

    kind: ScopeKind
    tenant_id: OpaqueId
    project_ids: list[OpaqueId]

    @model_validator(mode="after")
    def _validate_scope(self) -> "ExportScope":
        if len(set(self.project_ids)) != len(self.project_ids):
            raise ValueError("scope contains duplicate project IDs")
        if self.kind == "tenant" and self.project_ids:
            raise ValueError("tenant scope must not include project IDs")
        if self.kind == "projects" and not self.project_ids:
            raise ValueError("projects scope requires at least one project ID")
        return self


class ProducerReference(ContractModel):
    """Identifies the software contract that produced the manifest."""

    implementation: OpaqueId
    version: OpaqueId


class PositionReference(ContractModel):
    """An opaque position in one explicitly identified ordered stream."""

    stream_id: OpaqueId
    position: OpaqueId


class SnapshotMetadata(ContractModel):
    """Complete snapshot, barrier, and replay coordinates for migration."""

    snapshot_id: OpaqueId
    snapshot_position: PositionReference
    completion_barrier: PositionReference
    replay_position: PositionReference


class ChecksumDescriptor(ContractModel):
    """A supported checksum algorithm and lowercase canonical digest."""

    algorithm: ChecksumAlgorithm
    digest: str = Field(pattern=r"^[a-f0-9]{64}$")


class ManifestFile(ContractModel):
    """A claimed regular file and the values an offline reader must verify."""

    path: str = Field(min_length=1, max_length=4096)
    size_bytes: SafeCounter
    checksum: ChecksumDescriptor
    entry_type: Literal["file"]

    @field_validator("path")
    @classmethod
    def _canonicalize_path(cls, value: str) -> str:
        return _normalized_bundle_path(value)


class DeclaredOmission(ContractModel):
    """A specific capability or state class the producer says is absent."""

    code: OpaqueId
    reason: str = Field(min_length=1, max_length=1024)


class IdentityMapping(ContractModel):
    """An explicit one-to-one source-to-destination principal mapping.

    ``authority_reference`` identifies the separate approval evidence an
    importer must resolve.  Its presence is not itself proof of authority.
    """

    source_identity_id: OpaqueId
    destination_identity_id: OpaqueId
    authority_reference: OpaqueId


class EncryptionReference(ContractModel):
    """References encryption semantics without embedding recovery material."""

    mode: EncryptionMode
    protocol_reference: OpaqueId | None
    key_envelope_paths: list[str]

    @field_validator("key_envelope_paths")
    @classmethod
    def _canonicalize_envelope_paths(cls, values: list[str]) -> list[str]:
        normalized = [_normalized_bundle_path(value) for value in values]
        if len(set(normalized)) != len(normalized):
            raise ValueError("encryption reference contains duplicate envelope paths")
        return normalized

    @model_validator(mode="after")
    def _validate_mode(self) -> "EncryptionReference":
        if self.mode == "encrypted":
            if self.protocol_reference is None:
                raise ValueError("encrypted mode requires a protocol reference")
            if not self.key_envelope_paths:
                raise ValueError("encrypted mode requires at least one key envelope path")
        if self.mode == "plaintext_authorized_export":
            if self.protocol_reference is not None or self.key_envelope_paths:
                raise ValueError(
                    "plaintext export must not declare encryption protocol or key envelopes"
                )
        return self


class PortableManifest(VersionedContract):
    """Strict portable-bundle inventory suitable for offline preflight.

    Lists are required even when empty so omitted declarations cannot be
    mistaken for known-empty state.  Optional projections, embeddings,
    history, and recovery policies have no defaults here; producers must state
    their included capabilities and omissions explicitly.
    """

    format_version: Literal["portable-candidate-v1"]
    manifest_id: OpaqueId
    producer: ProducerReference
    scope: ExportScope
    snapshot: SnapshotMetadata
    files: list[ManifestFile]
    canonical_ids: list[CanonicalObjectId]
    included_capabilities: list[OpaqueId]
    omissions: list[DeclaredOmission]
    identity_mappings: list[IdentityMapping]
    encryption: EncryptionReference

    @model_validator(mode="after")
    def _validate_inventory(self) -> "PortableManifest":
        file_paths = [item.path for item in self.files]
        if len(set(file_paths)) != len(file_paths):
            raise ValueError("manifest contains duplicate normalized file paths")
        windows_file_paths = [PureWindowsPath(path) for path in file_paths]
        if len(set(windows_file_paths)) != len(windows_file_paths):
            raise ValueError("manifest contains colliding Windows file paths")
        # A declared file cannot also be the directory of another declared file (Windows folding covers the
        # exact case too).
        if names_below_a_file(file_paths, file_paths, str.lower):
            raise ValueError("manifest declares a file that is also a directory of another file")

        object_keys = [(item.kind, item.id) for item in self.canonical_ids]
        if len(set(object_keys)) != len(object_keys):
            raise ValueError("manifest contains duplicate canonical object IDs")

        if len(set(self.included_capabilities)) != len(self.included_capabilities):
            raise ValueError("manifest contains duplicate included capabilities")

        omission_codes = [item.code for item in self.omissions]
        if len(set(omission_codes)) != len(omission_codes):
            raise ValueError("manifest contains duplicate omission codes")
        overlap = set(self.included_capabilities) & set(omission_codes)
        if overlap:
            raise ValueError("a capability cannot be both included and omitted")

        mapping_sources = [item.source_identity_id for item in self.identity_mappings]
        if len(set(mapping_sources)) != len(mapping_sources):
            raise ValueError("manifest maps a source identity more than once")
        mapping_destinations = [
            item.destination_identity_id for item in self.identity_mappings
        ]
        if len(set(mapping_destinations)) != len(mapping_destinations):
            raise ValueError("manifest maps more than one source identity to a destination")

        missing_envelopes = set(self.encryption.key_envelope_paths) - set(file_paths)
        if missing_envelopes:
            raise ValueError("encryption references files absent from the manifest")
        return self


def _plain_text_keys(document: Any) -> Any:
    """A copy of a mapping input, read by the same two phases and rules as a model graph.

    The whole graph (every nested dict, list, tuple and model, however shared) is discovered, with its
    native storage frozen and every public view of a container subclass or custom mapping observed once
    and charged, under one budget, before anything is rebuilt: a callback met while one part is read
    cannot repair another part that has not been snapshotted yet.  The copy is then rebuilt from those
    snapshots only.  A key that is a ``str`` subclass is read as its text without running its hooks, and
    two keys with one text are refused (see ``_reconstruct_retained_state``).  Anything that is not a
    container is left to the schema.
    """

    if not _inherits(document, dict):
        return document  # not a mapping: the schema refuses it
    state = _CallState()
    inventory = _reject_retained_unknown_fields(document, state)  # type: ignore[arg-type]
    return _reconstruct_retained_state(document, inventory, state)  # type: ignore[arg-type]


_OMITTED_INPUT = "<omitted: caller input is never rendered>"
_MAX_REFUSAL_TEXT = 1_000


def _value_free(error: ValidationError) -> ValidationError:
    """The same refusal with no caller input in it, so rendering it reads nothing of the caller.

    Pydantic keeps the rejected object as each error's input, and rendering the error
    (``str``, ``repr``, ``json``, logging) then runs that object's ``repr``: for a
    model carrying a custom extra view, an unbounded read.  Locations, error types and
    context are kept; an input is kept only when it is a short plain scalar.
    """

    details = []
    for item in error.errors(include_url=False):
        detail = {"type": item["type"], "loc": item["loc"], "input": _renderable_input(item.get("input"))}
        if "ctx" in item:
            detail["ctx"] = item["ctx"]
        details.append(detail)
    return ValidationError.from_exception_data(error.title, details)


_MAX_RENDERED_INPUT = 256


def _renderable_input(value: Any) -> Any:
    """Keep a plain, short scalar input (a useful diagnostic); anything else is omitted.

    Exact ``str`` (bounded length), ``int``, ``float``, ``bool`` and ``None`` render
    without running caller code; models, mappings, containers and other objects may
    not, so they become the placeholder.
    """

    if value is None or is_exact_type(value, bool, int, float):
        return value
    if is_exact_type(value, str) and len(value) <= _MAX_RENDERED_INPUT:
        return value
    return _OMITTED_INPUT


def _refusal(cause: BaseException) -> ValidationError:
    text = str(cause)
    bounded = ValueError(text if len(text) <= _MAX_REFUSAL_TEXT else text[:_MAX_REFUSAL_TEXT] + "...")
    return ValidationError.from_exception_data(
        "PortableManifest",
        [{"type": "value_error", "loc": (), "input": _OMITTED_INPUT, "ctx": {"error": bounded}}],
    )


def validate_portable_manifest(
    document: PortableManifest | Mapping[str, Any],
) -> PortableManifest:
    """Purely validate and return a portable manifest.

    This function performs no filesystem, archive, network, authorization, or
    cryptographic work.  Callers must compare the returned file descriptors to
    actual bytes before any mutation.  A refusal never carries the caller's input
    (see ``_value_free``) and is raised outside the handler that caught the
    original error, so no rendering re-reads caller objects.
    """

    failure: ValidationError | None = None
    cause: BaseException | None = None
    if isinstance(document, PortableManifest):
        try:
            model_inventory = _reject_retained_unknown_fields(document)
            candidate = _reconstruct_retained_state(document, model_inventory)
        except ValidationError as exc:
            failure = _value_free(exc)
        except (TypeError, ValueError, RecursionError) as exc:
            failure, cause = _refusal(exc), exc
    else:
        try:
            candidate = _plain_text_keys(document)
        except ValidationError as exc:
            failure = _value_free(exc)
        except (TypeError, ValueError, RecursionError) as exc:
            failure, cause = _refusal(exc), exc
    if failure is None:
        try:
            return PortableManifest.model_validate(candidate)
        except ValidationError as exc:
            failure = _value_free(exc)
    if cause is not None:
        raise failure from cause  # the traversal's own bounded error, never caller objects
    raise failure


__all__ = [
    "CanonicalObjectId",
    "ChecksumDescriptor",
    "DeclaredOmission",
    "EncryptionReference",
    "ExportScope",
    "IdentityMapping",
    "ManifestFile",
    "PortableManifest",
    "PositionReference",
    "ProducerReference",
    "SnapshotMetadata",
    "normalize_bundle_path",
    "validate_portable_manifest",
]
