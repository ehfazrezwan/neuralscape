"""Candidate canonical lifecycle types and pure consistency checks.

The models are inert contracts.  They select no storage system, accept no
command durably, authorize no caller, mutate no record, and publish no
projection.  Callers must establish those runtime properties separately.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from enum import Enum
from collections.abc import Iterator
from typing import Annotated, Any, Literal, TypeVar

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    Strict,
    ValidationError,
    WrapValidator,
    field_validator,
    model_validator,
)

from pydantic_core import PydanticCustomError

from contracts_bodies import MemoryBody, OpaqueEnvelopeBody, PlaintextBody
from contracts_common import ContractModel, OpaqueId, SafeCounter, VersionedContract
from contracts_limits import DISCOVERY_BUDGET_MESSAGE, PORTABILITY, DiscoveryBudget, is_exact_type
from contracts_references import ReferenceHandle, SourceVersion


_ModelT = TypeVar("_ModelT", bound=BaseModel)

# A stage may report one output reference per stored passage or memory, and one claim reads the
# intent with every receipt it holds.  Measured on the shared accounting (edge-bound, about twelve
# edges per reference handle): the standard tier admits about 10,900 handles per call in total,
# the portability tier about 65,500, within a few seconds in the worst case.  The standard tier
# therefore strands a large ingest, so lifecycle claims use the portability tier.
LIFECYCLE_TIER = PORTABILITY
_BASE_MODEL_DICT_DESCRIPTOR = vars(BaseModel)["__dict__"]
_BASE_MODEL_EXTRA_DESCRIPTOR = vars(BaseModel)["__pydantic_extra__"]


def _key_text(key: object) -> object:
    """A key's own text for comparison; a str subclass is read without running its hooks."""

    if type(key) is str:
        return key
    return str.__str__(key) if _derives_from(type(key), (str,)) else key


def _require_distinct_key_text(
    entries: tuple[tuple[object, object], ...],
) -> tuple[tuple[object, object], ...]:
    """Refuse keys that spell one name, however their own hash and equality disagree."""

    seen: set[object] = set()
    for key, _ in entries:
        text = _key_text(key)
        if text in seen:
            raise ValueError("contract mapping has colliding keys")
        seen.add(text)
    return entries


class _CapturedGraph:
    """Opaque native graph captured before reconstruction callbacks."""


@dataclass(frozen=True, slots=True)
class _CapturedLeaf(_CapturedGraph):
    value: object


@dataclass(frozen=True, slots=True)
class _CapturedBackReference(_CapturedGraph):
    """A native edge back to a container still being captured.

    Whether it is a cycle depends on the path that reaches it at reconstruction, not on which
    traversal happened to capture the container first, so it is resolved (and a real cycle
    refused) when the graph is rebuilt.
    """

    identity: int
    source: object


@dataclass(frozen=True, slots=True)
class _CapturedModel(_CapturedGraph):
    identity: int
    source: BaseModel
    model_type: type[BaseModel]
    stored_entries: tuple[tuple[object, _CapturedGraph], ...]
    extra_entries: tuple[tuple[object, _CapturedGraph], ...]
    malformed_extra: bool


@dataclass(frozen=True, slots=True)
class _CapturedDict(_CapturedGraph):
    identity: int
    source: dict[object, object]
    entries: tuple[tuple[object, _CapturedGraph], ...]


@dataclass(frozen=True, slots=True)
class _CapturedList(_CapturedGraph):
    identity: int
    source: list[object]
    items: tuple[_CapturedGraph, ...]


@dataclass(frozen=True, slots=True)
class _CapturedTuple(_CapturedGraph):
    identity: int
    source: tuple[object, ...]
    items: tuple[_CapturedGraph, ...]


_CapturedGraphSnapshot = tuple[
    _CapturedGraph,
    dict[Any, Any],
]


def _reserve_entries(budget: DiscoveryBudget, count: int) -> None:
    """Refuse a native container before copying it when its entries cannot all be charged.

    Every entry is charged one edge when it is captured, so a container with more entries than the
    remaining allowance would only be rejected after its whole snapshot was built.  ``count`` comes
    from the base type's own length, never from a caller hook.
    """

    if count > budget.tier.max_edges - budget.edges:
        budget.charge_edge(count)  # raises the stable budget error and leaves the budget exhausted


_UTC_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)

_DATE_FIELDS = tuple(date.__dict__[name] for name in ("year", "month", "day"))
_TIME_FIELDS = tuple(datetime.__dict__[name] for name in ("hour", "minute", "second", "microsecond"))


def _pinned_offset(value: object) -> object:
    """Read a timestamp's UTC offset exactly once and return a fixed-offset copy of it.

    A caller's ``tzinfo`` is asked for its offset when the value is validated and again whenever
    it is compared, and a zone with state can answer differently each time, so ordering could
    see another instant than validation did.  The copy carries the one answer as a fixed offset
    (an exact ``datetime``, fields read through the base descriptors), so every later read of it
    agrees.  A value with no offset becomes naive and is refused as such; anything that is not a
    ``datetime`` (JSON text, for one) is left to the field's own validation.
    """

    if not isinstance(value, datetime):
        return value
    try:
        offset = datetime.utcoffset(value)
        fields = [descriptor.__get__(value) for descriptor in _DATE_FIELDS + _TIME_FIELDS]
        return datetime(*fields, tzinfo=None if offset is None else timezone(offset))
    except Exception as error:  # a zone that cannot answer, or answers out of range
        raise ValueError("timestamp offset could not be read as a valid UTC offset") from error


def _pin_timestamp(value: object, handler: Any, info: Any) -> object:
    """Validate a timestamp from its one offset read (``_pinned_offset``).

    Python input must be a ``datetime`` (as in strict mode); JSON input must be text, parsed by
    the field's own validation.  The field is not strict itself only because a wrapped strict
    field would refuse JSON text.
    """

    if isinstance(value, datetime):
        return handler(_pinned_offset(value))
    if info.mode == "json" and isinstance(value, str):
        return handler(value)
    raise PydanticCustomError("datetime_type", "Input should be a valid datetime")


# An aware timestamp whose offset is read exactly once: validation and every comparison use it.
PinnedAwareDatetime = Annotated[AwareDatetime, Strict(False), WrapValidator(_pin_timestamp)]


def _utc_instant_before(earlier: datetime, later: datetime) -> bool:
    """Whether ``later`` is an earlier instant than ``earlier``.

    Aware datetimes that share one ``tzinfo`` compare by wall time, which ignores ``fold``: in a
    repeated hour the second 01:15 sorts before the first 01:45 though it is 30 minutes later.
    Subtracting a UTC epoch (a different ``tzinfo``) uses each value's own offset; the base-class
    methods keep a datetime subclass from answering for itself.
    """

    return datetime.__sub__(later, _UTC_EPOCH) < datetime.__sub__(earlier, _UTC_EPOCH)


_TYPE_MRO = type.__dict__["__mro__"]  # the real MRO, never a metaclass's answer
_TYPE_NAMESPACE = type.__dict__["__dict__"]  # a class's own namespace, read without metaclass dispatch
_NATIVE_LEAF_TYPES = (str, int, float, bytes, type(None), date, timedelta)


def _derives_from(kind: type, bases: tuple[type, ...]) -> bool:
    """Whether ``kind``'s real MRO holds one of ``bases``, by identity.

    ``issubclass`` dispatches to the base's metaclass (``__subclasscheck__``), which an ABC answers from
    registrations and caches keyed by the caller's class; the MRO read through ``type`` runs no hook.
    """

    return any(base is wanted for base in _TYPE_MRO.__get__(kind) for wanted in bases)


_EXACT_NATIVE_KINDS = (str, int, float, bytes, bool, type(None), date, datetime, timedelta)


def _is_exact_native_kind(kind: type) -> bool:
    """Whether ``kind`` is itself one of the native scalar types, by identity (never ``==``, which a
    metaclass can answer)."""

    return any(kind is native for native in _EXACT_NATIVE_KINDS)


_NATIVE_BASES = (str, bytes, int, float, datetime, date, timedelta, type(None))  # most derived first


def _namespace_value(namespace: Any, name: str, budget: DiscoveryBudget) -> object:
    """The entry named ``name`` in a class namespace, found by scanning keys as exact text (charged).

    A namespace can keep a ``str``-subclass key whose hash collides with ``name``: looking the name up
    would call that key's ``__eq__``.  ``None`` when absent."""

    budget.charge_edge(len(namespace))
    for key, item in namespace.items():
        # Only an exact-str key: the type machinery writes these names as exact strings, so a subclass key
        # spelling the name (planted through a metaclass's __prepare__) is never the real entry.
        if type(key) is str and key == name:
            return item
    return None


_FIELDS = "fields"  # memo key prefix: one model class's declared field names, read natively once per call


def _declared_field_names(model_type: type, captured_by_identity: dict[Any, Any], budget: DiscoveryBudget) -> frozenset:
    """The declared field names of ``model_type``, read from the real MRO's namespaces, never through its
    metaclass (``model_fields`` is an attribute read a caller's model metaclass would answer)."""

    generation = _generation(captured_by_identity)
    memo = captured_by_identity.get((_FIELDS, id(model_type)))
    if memo is not None and memo[0] is model_type and memo[2] == generation:
        return memo[1]
    names: frozenset = frozenset()
    for base in _TYPE_MRO.__get__(model_type):
        fields = _namespace_value(_TYPE_NAMESPACE.__get__(base), "__pydantic_fields__", budget)
        if type(fields) is dict:
            names = frozenset(str.__str__(key) for key in dict.keys(fields) if _derives_from(type(key), (str,)))
            break
    captured_by_identity[(_FIELDS, id(model_type))] = (model_type, names, generation)
    return names


_GRAPH_KIND = "graph-kind"  # memo key prefix: whether a class is a model, dict, list, tuple or a leaf


def _graph_kind(kind: type, captured_by_identity: dict[Any, Any]) -> type | None:
    """``BaseModel``, ``dict``, ``list`` or ``tuple`` when ``kind`` derives from it (in that order), else
    ``None``; decided from the real MRO once per class and call.  The answer cannot change within a call:
    CPython refuses a ``__bases__`` assignment that moves a class between these instance layouts, so the
    memo needs no generation (a view stepping item by item would otherwise re-walk the MRO per item)."""

    if kind is dict or kind is list or kind is tuple:
        return kind
    memo = captured_by_identity.get((_GRAPH_KIND, id(kind)))
    if memo is not None and memo[0] is kind:
        return memo[1]
    mro = _TYPE_MRO.__get__(kind)
    found = next((wanted for wanted in (BaseModel, dict, list, tuple) if any(base is wanted for base in mro)), None)
    captured_by_identity[(_GRAPH_KIND, id(kind))] = (kind, found)
    return found


def _exact_native(value: object, base: type) -> object:
    """An exact copy of a native scalar of a subclass, read through its native ``base``, so nothing of the
    subclass (its methods, or a hook installed on it later) travels on from capture."""

    if base is str:
        return str.__str__(value)
    if base is bytes:
        return bytes.__bytes__(value)
    if base is int:
        return int.__int__(value)
    if base is float:
        return float.__float__(value)
    if base is datetime:
        fields = [descriptor.__get__(value) for descriptor in _DATE_FIELDS + _TIME_FIELDS]
        return datetime(*fields, tzinfo=datetime.__dict__["tzinfo"].__get__(value),
                        fold=datetime.__dict__["fold"].__get__(value))
    if base is date:
        return date(*(descriptor.__get__(value) for descriptor in _DATE_FIELDS))
    if base is timedelta:
        return timedelta(days=timedelta.__dict__["days"].__get__(value),
                         seconds=timedelta.__dict__["seconds"].__get__(value),
                         microseconds=timedelta.__dict__["microseconds"].__get__(value))
    return value


_MEMBERS = "members"  # memo key prefix: one enum class's member identities, indexed once per call
_GENERATION = "generation"  # memo key: advanced whenever a public view has run caller code


def _generation(captured_by_identity: dict[Any, Any]) -> int:
    return captured_by_identity.get(_GENERATION, 0)


def _is_declared_member(
    value: object,
    captured_by_identity: dict[Any, Any],
    budget: DiscoveryBudget,
) -> bool:
    """Whether an enum instance is one of its class's declared members, read without any hook.

    ``type(value).__members__`` is an attribute lookup on the enum's class, so a caller's enum metaclass
    (its ``__getattribute__``, or a ``__members__`` it overrides) would run during capture and could change
    a later value before that value is captured.  The member map is read from the class's own namespace
    instead, and iterated with the base ``dict`` method.  It is indexed by identity once per class and
    call, each member charged one edge before the index is built, so repeated leaves of a large enum do
    not rescan it.  The index keeps the class and its members alive, so no indexed ``id()`` is recycled,
    and holds only until caller code runs (a public view can change the class).
    """

    kind = type(value)
    generation = _generation(captured_by_identity)
    memo = captured_by_identity.get((_MEMBERS, id(kind)))
    if memo is None or memo[0] is not kind or memo[3] != generation:
        members = _namespace_value(_TYPE_NAMESPACE.__get__(kind), "_member_map_", budget)
        if type(members) is dict:
            budget.charge_edge(dict.__len__(members))
            declared = tuple(dict.values(members))
        else:
            declared = ()
        memo = (kind, declared, frozenset(id(member) for member in declared), generation)
        captured_by_identity[(_MEMBERS, id(kind))] = memo
    return id(value) in memo[2]


_IMPERSONATION_HOOKS = ("__class__", "__getattribute__")
_TYPE_FLAGS = type.__dict__["__flags__"]
_HEAP_TYPE = 1 << 9  # Py_TPFLAGS_HEAPTYPE: a class created by a class statement (or ``type(...)``)


def _claims_another_class(kind: type, budget: DiscoveryBudget) -> bool:
    """Whether a class written in Python in ``kind``'s MRO defines ``__class__`` or ``__getattribute__``.

    Those are the two ways an object answers ``isinstance`` with a class it is not, or redirects the
    attribute reads validation makes (``__dict__``, the field set): a ``str`` subclass claiming a model
    class would otherwise reach a model-typed position as a "scalar" and hand validation storage that
    capture never traversed or charged.  Only heap types are inspected (the built-in types, whose own
    slots define these names, are fixed); flags and namespaces are read natively, so no hook runs.
    A namespace can keep a ``str``-subclass key whose hash collides with a hook name, so the names are not
    looked up (that would call the stored key's ``__eq__``): every key is copied to exact text and
    compared, and the scan is charged one edge per key.
    """

    for base in _TYPE_MRO.__get__(kind):
        if not _TYPE_FLAGS.__get__(base) & _HEAP_TYPE:
            continue
        namespace = _TYPE_NAMESPACE.__get__(base)
        budget.charge_edge(len(namespace))
        for key in namespace:
            if _derives_from(type(key), (str,)) and str.__str__(key) in _IMPERSONATION_HOOKS:
                return True
    return False


_HANDED_ENUMS = "handed-enums"  # memo key: enum members handed out while the current root materializes
_LEAF_CLASS = "leaf-class"  # memo key prefix: one leaf class's verdict, decided once per call


def _leaf_class_verdict(
    kind: type,
    captured_by_identity: dict[Any, Any],
    budget: DiscoveryBudget,
) -> object:
    """Classify a leaf's class once per call: ``"enum"``, ``"refused"``, or its native scalar base type.

    Every check walks the class's MRO, which a caller can make arbitrarily deep; the walk is charged one
    edge per class in it and its verdict kept (with the class, so its ``id()`` is not recycled), so
    repeated leaves of one class do not walk it again.  A verdict holds only until caller code runs:
    a public view can give an admitted class a ``__class__`` claim before its next instance is captured.
    """

    generation = _generation(captured_by_identity)
    memo = captured_by_identity.get((_LEAF_CLASS, id(kind)))
    if memo is not None and memo[0] is kind and memo[2] == generation:
        return memo[1]
    mro = _TYPE_MRO.__get__(kind)
    budget.charge_edge(tuple.__len__(mro))
    if _claims_another_class(kind, budget):
        verdict = "refused"
    elif any(base is Enum for base in mro):
        verdict = "enum"
    else:
        verdict = next((native for native in _NATIVE_BASES if any(base is native for base in mro)), "refused")
    captured_by_identity[(_LEAF_CLASS, id(kind))] = (kind, verdict, generation)
    return verdict


def _native_leaf(
    value: object,
    captured_by_identity: dict[Any, Any],
    budget: DiscoveryBudget,
) -> object:
    """Admit a leaf only when its real type is a native scalar, or an enum member by identity.

    Validation reads any other object through its own attribute protocol: a proxy claiming a contract
    class (``__class__``) would run caller code inside the validator, outside the discovery budget.
    An enum instance built with ``str.__new__(Enum, value)`` is no member: every ``is`` test would
    misjudge it.  Both are refused here, before validation sees them; no hook runs to decide.
    """

    verdict = _leaf_class_verdict(type(value), captured_by_identity, budget)
    if verdict == "enum":
        if not _is_declared_member(value, captured_by_identity, budget):
            raise ValueError("an enum value must be a declared member of its enum")
        return value
    if verdict != "refused":
        if _is_exact_native_kind(type(value)):
            return value
        return _exact_native(value, verdict)  # detached: a hook installed on the subclass later never travels
    raise ValueError("a contract value must be a native scalar, an enum member, a model or a container")


def _cached_identity_capture(
    value: object,
    captured_by_identity: dict[Any, Any],
) -> _CapturedGraph | None:
    """Return a cache hit only when it owns the exact source object."""

    captured = captured_by_identity.get(id(value))
    if isinstance(
        captured,
        (_CapturedModel, _CapturedDict, _CapturedList, _CapturedTuple),
    ) and captured.source is value:
        return captured
    return None


def _capture_native_contract_graph(
    value: object,
    active_containers: set[int] | None = None,
    captured_by_identity: dict[Any, Any] | None = None,
    budget: DiscoveryBudget | None = None,
) -> _CapturedGraph:
    """Freeze supported native backing without invoking public protocols."""

    value_type = type(value)
    if active_containers is None:
        active_containers = set()
    if captured_by_identity is None:
        captured_by_identity = {}
    if budget is None:
        budget = DiscoveryBudget()
    budget.charge_edge()

    graph_kind = _graph_kind(value_type, captured_by_identity)
    if graph_kind is not None:
        identity = id(value)
        if identity in active_containers:
            return _CapturedBackReference(identity=identity, source=value)
        cached = _cached_identity_capture(value, captured_by_identity)
        if cached is not None:
            return cached
        budget.visit(value, len(active_containers))
        active_containers.add(identity)
        try:
            if graph_kind is BaseModel:
                stored = _BASE_MODEL_DICT_DESCRIPTOR.__get__(value, BaseModel)
                _reserve_entries(budget, dict.__len__(stored))
                stored_entries = tuple(dict.items(stored))
                try:
                    extra = _BASE_MODEL_EXTRA_DESCRIPTOR.__get__(value, BaseModel)
                except AttributeError:
                    extra = None
                if extra is None:
                    extra_entries: tuple[tuple[object, object], ...] = ()
                    malformed_extra = False
                else:
                    extra_type = type(extra)
                    malformed_extra = not _derives_from(extra_type, (dict,))
                    if not malformed_extra:
                        _reserve_entries(budget, len(stored_entries) + dict.__len__(extra))
                    extra_entries = (
                        () if malformed_extra else tuple(dict.items(extra))
                    )
                captured: _CapturedGraph = _CapturedModel(
                    identity=identity,
                    source=value,
                    model_type=value_type,
                    stored_entries=tuple(
                        (
                            key,
                            _capture_native_contract_graph(
                                item,
                                active_containers,
                                captured_by_identity,
                                budget,
                            ),
                        )
                        for key, item in _require_distinct_key_text(stored_entries)
                    ),
                    extra_entries=tuple(
                        (
                            key,
                            _capture_native_contract_graph(
                                item,
                                active_containers,
                                captured_by_identity,
                                budget,
                            ),
                        )
                        for key, item in _require_distinct_key_text(extra_entries)
                    ),
                    malformed_extra=malformed_extra,
                )
            elif graph_kind is dict:
                _reserve_entries(budget, dict.__len__(value))
                entries = tuple(dict.items(value))
                captured = _CapturedDict(
                    identity=identity,
                    source=value,
                    entries=tuple(
                        (
                            key,
                            _capture_native_contract_graph(
                                item,
                                active_containers,
                                captured_by_identity,
                                budget,
                            ),
                        )
                        for key, item in _require_distinct_key_text(entries)
                    ),
                )
            elif graph_kind is list:
                _reserve_entries(budget, list.__len__(value))
                native_items = tuple(list.__iter__(value))
                captured = _CapturedList(
                    identity=identity,
                    source=value,
                    items=tuple(
                        _capture_native_contract_graph(
                            item,
                            active_containers,
                            captured_by_identity,
                            budget,
                        )
                        for item in native_items
                    ),
                )
            else:
                _reserve_entries(budget, tuple.__len__(value))
                native_items = tuple(tuple.__iter__(value))
                captured = _CapturedTuple(
                    identity=identity,
                    source=value,
                    items=tuple(
                        _capture_native_contract_graph(
                            item,
                            active_containers,
                            captured_by_identity,
                            budget,
                        )
                        for item in native_items
                    ),
                )
            captured_by_identity[identity] = captured
            return captured
        finally:
            active_containers.remove(identity)
    return _CapturedLeaf(_native_leaf(value, captured_by_identity, budget))


def _observed_public_view(
    source: object,
    captured_by_identity: dict[Any, Any],
    budget: DiscoveryBudget,
) -> Iterator[_CapturedGraph]:
    """Observe a public view once per call: later edges and revalidations replay it."""

    def public_item(item: object) -> _CapturedGraph:
        item_type = type(item)
        if _graph_kind(item_type, captured_by_identity) is not None:
            frozen = _cached_identity_capture(item, captured_by_identity)
            if frozen is not None:
                # A cache hit is still one public item read: charged, so a view that keeps
                # yielding inventoried items ends at the budget instead of growing without bound.
                budget.charge_edge()
                return frozen
            return _capture_native_contract_graph(
                item,
                captured_by_identity=captured_by_identity,
                budget=budget,
            )
        budget.charge_edge()
        return _CapturedLeaf(_native_leaf(item, captured_by_identity, budget))

    memo = captured_by_identity.get(("view", id(source)))
    if memo is not None and memo[0] is source:
        yield from memo[1]
        return
    seen: list[_CapturedGraph] = []
    for item in source:  # type: ignore[attr-defined]
        captured_by_identity[_GENERATION] = _generation(captured_by_identity) + 1  # caller code ran
        frozen = public_item(item)
        seen.append(frozen)
        yield frozen
    captured_by_identity[_GENERATION] = _generation(captured_by_identity) + 1
    captured_by_identity[("view", id(source))] = (source, tuple(seen))


_MATERIALIZED = "materialized"  # memo key prefix: the finished copy of a pure native container
_REUSED = "reused"  # memo key: set once a copy nesting containers was shared between paths


def _remember_pure_copy(
    value: _CapturedGraph,
    children: tuple[_CapturedGraph, ...],
    result: object,
    captured_by_identity: dict[Any, Any],
) -> None:
    """Keep ``result`` for reuse when ``value`` is a pure native container.

    Pure: an exact ``tuple``, ``list`` or ``dict`` (plain ``str`` keys) whose children are leaves or
    pure containers themselves, so rebuilding it on another path runs no caller code and gives
    the same copy.  The subtree's height is kept so a later path is still depth-checked, and its
    expanded size (every entry on every path through it) so a later path is charged for the work
    strict validation does there.
    """

    height = 0
    size = len(children)
    for child in children:
        if isinstance(child, _CapturedLeaf):
            continue
        if not isinstance(child, (_CapturedDict, _CapturedList, _CapturedTuple)):
            return
        entry = captured_by_identity.get((_MATERIALIZED, child.identity))
        if entry is None or entry[0] is not child.source:
            return
        height = max(height, entry[2] + 1)
        size += entry[3]
    captured_by_identity[(_MATERIALIZED, value.identity)] = (value.source, result, height, size)


def _materialize_captured_graph(
    value: _CapturedGraph,
    captured_by_identity: dict[Any, Any],
    active_containers: set[int] | None = None,
    budget: DiscoveryBudget | None = None,
) -> object:
    """Reconstruct one frozen graph with established diagnostic ordering."""

    if active_containers is None:
        active_containers = set()
    if budget is None:
        budget = DiscoveryBudget()
    budget.charge_edge()
    if isinstance(value, _CapturedBackReference):
        resolved = _cached_identity_capture(value.source, captured_by_identity)
        if resolved is None or value.identity in active_containers:
            raise ValueError("cyclic contract graph is not valid input")
        return _materialize_captured_graph(
            resolved, captured_by_identity, active_containers, budget
        )
    if isinstance(value, _CapturedLeaf):
        leaf = value.value
        kind = type(leaf)
        if not (_is_exact_native_kind(kind)):
            # Only an enum member is kept as itself (its identity is the value).  Caller code may have run
            # since it was captured and changed its class, so the class and membership are judged again.
            if _leaf_class_verdict(kind, captured_by_identity, budget) != "enum" or not _is_declared_member(
                leaf, captured_by_identity, budget
            ):
                raise ValueError(
                    "a contract value must be a native scalar, an enum member, a model or a container"
                )
            handed = captured_by_identity.get(_HANDED_ENUMS)
            if handed is not None:
                handed.append(leaf)
        return leaf
    identity = value.identity
    if identity in active_containers:
        raise ValueError("cyclic contract graph is not valid input")
    reused = captured_by_identity.get((_MATERIALIZED, identity))
    if reused is not None and reused[0] is value.source:
        # A pure native container already built on another path: reuse the copy (its subtree was
        # charged once), checking this path's depth against the subtree's height.
        budget.visit(value.source, len(active_containers) + reused[2])
        # Strict validation walks the copy again on this path (one copy, validated per occurrence),
        # so the occurrence is charged its expanded size: a shared raw container cannot multiply
        # validation work past the budget.  The size is arithmetic over distinct objects.
        budget.charge_edge(reused[3])
        # A shared copy that nests containers, or a flat one that holds entries: the input repeats it per
        # path, so an error must not echo it.  An empty container is one interned object that ordinary
        # models share everywhere; its rendering does not grow, so it must not suppress a diagnostic.
        if reused[2] or reused[1]:
            captured_by_identity[_REUSED] = True
        return reused[1]
    budget.visit(value.source, len(active_containers))
    active_containers.add(identity)
    try:
        if isinstance(value, _CapturedModel):
            declared_fields = _declared_field_names(value.model_type, captured_by_identity, budget)
            # Keys reach validation as their own text: a str subclass's hash and equality would
            # otherwise decide which field a value fills (the text was checked distinct at capture).
            fields = {
                _key_text(key): _materialize_captured_graph(
                    item,
                    captured_by_identity,
                    active_containers,
                    budget,
                )
                for key, item in value.stored_entries
            }
            if value.malformed_extra:
                raise ValueError(
                    "malformed stored contract extras are not valid input"
                )
            stored_texts = {_key_text(key) for key, _ in value.stored_entries}
            for key, item in value.extra_entries:
                # Compared as exact text only: a stored key's own hash or equality never runs.
                if _key_text(key) in stored_texts or _key_text(key) in declared_fields:
                    raise ValueError(
                        "conflicting stored contract field is not valid input"
                    )
                fields[_key_text(key)] = _materialize_captured_graph(
                    item,
                    captured_by_identity,
                    active_containers,
                    budget,
                )
            return fields
        if isinstance(value, _CapturedDict):
            built = {
                _key_text(key): _materialize_captured_graph(
                    item,
                    captured_by_identity,
                    active_containers,
                    budget,
                )
                for key, item in value.entries
            }
            if is_exact_type(value.source, dict) and all(is_exact_type(key, str) for key, _ in value.entries):
                _remember_pure_copy(value, tuple(item for _, item in value.entries), built, captured_by_identity)
            return built

        def observed_view(source: object) -> Iterator[_CapturedGraph]:
            return _observed_public_view(source, captured_by_identity, budget)

        if isinstance(value, _CapturedList):
            exact = is_exact_type(value.source, list)
            items = value.items if exact else observed_view(value.source)
            built_list = [
                _materialize_captured_graph(
                    item,
                    captured_by_identity,
                    active_containers,
                    budget,
                )
                for item in items
            ]
            if exact:
                _remember_pure_copy(value, value.items, built_list, captured_by_identity)
            return built_list
        exact = is_exact_type(value.source, tuple)
        items = value.items if exact else observed_view(value.source)
        built_tuple = tuple(
            _materialize_captured_graph(
                item,
                captured_by_identity,
                active_containers,
                budget,
            )
            for item in items
        )
        if exact:
            _remember_pure_copy(value, value.items, built_tuple, captured_by_identity)
        return built_tuple
    finally:
        active_containers.remove(identity)


def _native_contract_graph(
    value: object,
    active_containers: set[int] | None = None,
) -> object:
    """Materialize stored fields without dropping extras or hiding cycles."""

    captured = _capture_contract_graph(value, active_containers)
    return _materialize_contract_graph(captured)


def _capture_contract_graph(
    value: object,
    active_containers: set[int] | None = None,
    budget: DiscoveryBudget | None = None,
    captured_by_identity: dict[Any, Any] | None = None,
) -> _CapturedGraphSnapshot:
    """Capture one graph without running its public reconstruction views."""

    if captured_by_identity is None:
        captured_by_identity = {}
    captured = _capture_native_contract_graph(
        value,
        active_containers,
        captured_by_identity,
        budget,
    )
    return captured, captured_by_identity


def _materialize_contract_graph(
    captured: _CapturedGraphSnapshot,
    budget: DiscoveryBudget | None = None,
) -> object:
    """Materialize one previously captured graph through established views."""

    root, captured_by_identity = captured
    return _materialize_captured_graph(root, captured_by_identity, None, budget)


def _article(name: str) -> str:
    return "an" if name[:1] in "AEIOU" else "a"


def _is_instance_by_type(value: object, model_type: type) -> bool:
    """``isinstance`` by the object's real type: a ``__class__`` claim is never consulted."""

    return _derives_from(type(value), (model_type,))


def _captured_model(
    model_type: type[_ModelT],
    value: object,
) -> _CapturedGraphSnapshot:
    if not _is_instance_by_type(value, model_type):
        raise TypeError(f"value must be {_article(model_type.__name__)} {model_type.__name__}")
    return _capture_contract_graph(value)


def _revalidated_captured_model(
    model_type: type[_ModelT],
    captured: _CapturedGraphSnapshot,
    budget: DiscoveryBudget | None = None,
) -> _ModelT:
    """Strictly validate a materialized graph; if it shares subtrees, never echo it in an error.

    A validation error normally repeats the offending input.  When the copy shares subtrees
    between paths, the input's text is exponentially larger than the graph, so the error is
    rebuilt without the input: the same types, locations and messages, nothing embedded.
    Ordinary inputs keep the full diagnostic.
    """

    memo = captured[1]
    started = _generation(memo)
    memo[_HANDED_ENUMS] = handed = []
    try:
        native = _materialize_contract_graph(captured, budget)
    finally:
        memo.pop(_HANDED_ENUMS, None)
    if _generation(memo) != started:
        # A view materialized after an enum leaf may have changed its class once that leaf was judged:
        # judge every member handed out for this root again, before validation reads any of them.
        check_budget = DiscoveryBudget() if budget is None else budget
        for leaf in handed:
            if _leaf_class_verdict(type(leaf), memo, check_budget) != "enum" or not _is_declared_member(
                leaf, memo, check_budget
            ):
                raise ValueError(
                    "a contract value must be a native scalar, an enum member, a model or a container"
                )
    try:
        return model_type.model_validate(native, strict=True)
    except ValidationError as error:
        if not captured[1].get(_REUSED):
            raise
        lines = [
            {"type": e["type"], "loc": e["loc"], "input": "<omitted: input shares subtrees>",
             **({"ctx": e["ctx"]} if "ctx" in e else {})}
            for e in error.errors(include_input=False, include_url=False)
        ]
        try:
            bounded: Exception = ValidationError.from_exception_data(error.title, lines)  # type: ignore[arg-type]
        except Exception:  # noqa: BLE001 - a type that cannot be rebuilt still gets a bounded, value-free error
            bounded = ValueError(f"{error.title}: {error.error_count()} validation error(s)")
        raise bounded from None


def _revalidated_inventory_model(
    model_type: type[_ModelT],
    value: object,
    captured_by_identity: dict[Any, Any],
    budget: DiscoveryBudget | None = None,
) -> _ModelT:
    """Validate a public-view item, reusing entry-time state when available."""

    if not _is_instance_by_type(value, model_type):
        raise TypeError(f"value must be {_article(model_type.__name__)} {model_type.__name__}")
    captured = _cached_identity_capture(value, captured_by_identity)
    if captured is None:
        captured = _capture_native_contract_graph(
            value,
            captured_by_identity=captured_by_identity,
            budget=budget,
        )
    return _revalidated_captured_model(
        model_type,
        (captured, captured_by_identity),
        budget,
    )


def _revalidated_model(model_type: type[_ModelT], value: object) -> _ModelT:
    """Strictly reconstruct one expected model from its complete stored graph."""

    return _revalidated_captured_model(
        model_type,
        _captured_model(model_type, value),
    )


class MemoryRecordKind(str, Enum):
    ORIGINAL = "original"
    FACT = "fact"
    EPISODE = "episode"
    DERIVED_VIEW = "derived_view"


class MemoryLifecycle(str, Enum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    TOMBSTONED = "tombstoned"
    ERASED = "erased"


class ApplicabilityScope(str, Enum):
    GLOBAL = "global"
    PROJECT = "project"
    WORKSPACE = "workspace"


PlaintextMemoryBody = PlaintextBody
OpaqueEnvelopeMemoryBody = OpaqueEnvelopeBody


class MemoryRecord(VersionedContract):
    """Frozen canonical memory identity, distinct from every projection."""

    model_config = ConfigDict(frozen=True)

    id: OpaqueId
    tenant_id: OpaqueId
    owner_id: OpaqueId
    record_kind: MemoryRecordKind
    applicability: ApplicabilityScope
    project_id: OpaqueId | None
    workspace_id: OpaqueId | None
    category: OpaqueId
    author_kind: OpaqueId
    source_kind: OpaqueId
    content_revision: SafeCounter
    policy_epoch: SafeCounter
    lifecycle: MemoryLifecycle
    body: MemoryBody | None
    source_refs: tuple[ReferenceHandle, ...]
    required_parents: tuple[SourceVersion, ...]
    successor_id: OpaqueId | None
    retention_hold_ids: tuple[OpaqueId, ...]
    occurred_at: PinnedAwareDatetime | None
    recorded_at: PinnedAwareDatetime

    @model_validator(mode="after")
    def validate_record_semantics(self) -> MemoryRecord:
        if self.applicability is ApplicabilityScope.GLOBAL:
            if self.project_id is not None or self.workspace_id is not None:
                raise ValueError("global applicability cannot carry project_id or workspace_id")
        elif self.applicability is ApplicabilityScope.PROJECT:
            if self.project_id is None:
                raise ValueError("project applicability requires project_id")
            if self.workspace_id is not None:
                raise ValueError("project applicability cannot carry workspace_id")
        elif self.workspace_id is None:
            raise ValueError("workspace applicability requires workspace_id")

        if self.lifecycle is MemoryLifecycle.ERASED:
            if self.body is not None:
                raise ValueError("erased records cannot retain a body")
        elif self.body is None:
            raise ValueError("non-erased records require a body")
        _require_unique_source_ids(self.required_parents, "required_parents")
        for handle in self.source_refs:
            # Lineage may name another tenant's source (owner direction ``lineage = authorized_cross``)
            # but only as marked data: resolution always needs a current authorization check.
            if handle.cross_tenant != (handle.tenant_id != self.tenant_id):
                raise ValueError(
                    "source_refs must set cross_tenant exactly when the handle names another tenant"
                )
        return self


class ProcessingStage(str, Enum):
    CANONICAL = "canonical"
    EXTRACT = "extract"
    VECTOR = "vector"
    GRAPH = "graph"
    MARKDOWN = "markdown"


class StageEvidenceRequirement(ContractModel):
    """Declared evidence required for one processing-stage effect."""

    model_config = ConfigDict(frozen=True)

    stage: ProcessingStage
    effect_id: OpaqueId
    precondition_source_ids: tuple[OpaqueId, ...]
    upstream_stages: tuple[ProcessingStage, ...]
    committed_sources: tuple[SourceVersion, ...]
    minimum_output_refs: Literal[0, 1]
    # Empty source equality is not an application witness.  A stage that genuinely consumes no
    # source must say so here; otherwise an APPLIED receipt needs a nonempty applied source set.
    no_source_witness: bool = False

    @field_validator("minimum_output_refs", mode="before")
    @classmethod
    def validate_native_output_minimum(cls, value: object) -> object:
        if not is_exact_type(value, int) or value not in {0, 1}:
            raise ValueError("minimum_output_refs must be the native integer 0 or 1")
        return value

    @model_validator(mode="after")
    def validate_requirement_semantics(self) -> StageEvidenceRequirement:
        if len(set(self.precondition_source_ids)) != len(
            self.precondition_source_ids
        ):
            raise ValueError("precondition_source_ids must not contain duplicates")
        if len(set(self.upstream_stages)) != len(self.upstream_stages):
            raise ValueError("upstream_stages must not contain duplicates")
        if self.stage in self.upstream_stages:
            raise ValueError("a stage cannot depend on itself")
        _require_unique_source_ids(self.committed_sources, "committed_sources")
        return self


class Intent(VersionedContract):
    """Frozen validated command data; construction is not durable acceptance.

    Callers must revalidate data entering from unchecked construction or copy
    operations; Pydantic's low-level bypasses are outside this guarantee.
    """

    model_config = ConfigDict(frozen=True)

    id: OpaqueId
    tenant_id: OpaqueId
    actor_id: OpaqueId
    credential_id: OpaqueId
    operation: OpaqueId
    target_refs: tuple[ReferenceHandle, ...]
    request_digest: Annotated[str, Field(min_length=1, max_length=512)]
    idempotency_key: OpaqueId
    source_preconditions: tuple[SourceVersion, ...]
    stage_requirements: tuple[StageEvidenceRequirement, ...]
    correlation_id: OpaqueId
    accepted_at: PinnedAwareDatetime

    @model_validator(mode="after")
    def validate_stage_requirements(self) -> Intent:
        if any(item.tenant_id != self.tenant_id or item.cross_tenant for item in self.target_refs):
            # A write never targets another tenant's record; the cross-tenant marker is for lineage.
            raise ValueError("target_refs must match intent tenant_id and cannot be cross_tenant")
        _require_unique_source_ids(self.source_preconditions, "source_preconditions")
        if not self.stage_requirements:
            raise ValueError("stage_requirements must not be empty")

        requirements = {item.stage: item for item in self.stage_requirements}
        if len(requirements) != len(self.stage_requirements):
            raise ValueError("stage_requirements must not contain duplicate stages")
        if len({item.effect_id for item in self.stage_requirements}) != len(
            self.stage_requirements
        ):
            raise ValueError(
                "stage_requirements must not contain duplicate effect_id values"
            )

        preconditions = {item.record_id: item for item in self.source_preconditions}
        for requirement in self.stage_requirements:
            if set(requirement.precondition_source_ids) - set(preconditions):
                raise ValueError(
                    "precondition_source_ids must select declared source_preconditions"
                )
            if set(requirement.upstream_stages) - set(requirements):
                raise ValueError(
                    "upstream_stages must select declared stage_requirements"
                )

            calculated_ids = list(requirement.precondition_source_ids)
            for upstream_stage in requirement.upstream_stages:
                calculated_ids.extend(
                    item.record_id
                    for item in requirements[upstream_stage].committed_sources
                )
            if len(calculated_ids) != len(set(calculated_ids)):
                raise ValueError(
                    "calculated stage inputs must not contain record_id collisions"
                )

        _topological_stages(requirements)
        return self


class StageStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    APPLIED = "applied"
    SKIPPED = "skipped"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SUPERSEDED = "superseded"


class StageError(ContractModel):
    model_config = ConfigDict(frozen=True)

    code: OpaqueId
    retryable: bool
    safe_message: Annotated[str, Field(min_length=1, max_length=1024)]


class UpstreamApplication(ContractModel):
    """Immutable reference to the upstream stage application a downstream stage consumed.

    It names the identity of one APPLIED upstream receipt (intent, stage, attempt, effect id), so
    causal order between stages is carried by an explicit dependency, never by comparing worker
    clocks (owner direction ``causality = application_refs``).
    """

    model_config = ConfigDict(frozen=True)

    intent_id: OpaqueId
    stage: ProcessingStage
    attempt: SafeCounter
    effect_id: OpaqueId

    @model_validator(mode="after")
    def validate_attempt(self) -> UpstreamApplication:
        if self.attempt < 1:
            raise ValueError("attempt must be at least 1")
        return self


class StageReceipt(VersionedContract):
    """Frozen validated observation for one attempt, not aggregate completion.

    Callers must revalidate data entering from unchecked construction or copy
    operations; Pydantic's low-level bypasses are outside this guarantee.
    """

    model_config = ConfigDict(frozen=True)

    intent_id: OpaqueId
    attempt: SafeCounter
    stage: ProcessingStage
    effect_id: OpaqueId
    status: StageStatus
    applied_sources: tuple[SourceVersion, ...]
    committed_sources: tuple[SourceVersion, ...]
    output_refs: tuple[ReferenceHandle, ...]
    error: StageError | None
    started_at: PinnedAwareDatetime | None
    finished_at: PinnedAwareDatetime | None
    # Additive (default empty, so receipts written before this field existed still parse): the
    # upstream applications this stage consumed.  Only APPLIED receipts may carry them.
    upstream_applications: tuple[UpstreamApplication, ...] = ()

    @model_validator(mode="after")
    def validate_receipt_semantics(self) -> StageReceipt:
        if self.attempt < 1:
            raise ValueError("attempt must be at least 1")
        if self.upstream_applications:
            if self.status is not StageStatus.APPLIED:
                raise ValueError("only applied stage receipts may carry upstream_applications")
            if len({item.stage for item in self.upstream_applications}) != len(self.upstream_applications):
                raise ValueError("upstream_applications must cite each upstream stage once")
        _require_unique_source_ids(self.applied_sources, "applied_sources")
        _require_unique_source_ids(self.committed_sources, "committed_sources")
        _require_unique_reference_handles(self.output_refs, "output_refs")

        terminal = self.status in {
            StageStatus.APPLIED,
            StageStatus.SKIPPED,
            StageStatus.FAILED,
            StageStatus.CANCELLED,
            StageStatus.SUPERSEDED,
        }
        if terminal and self.finished_at is None:
            raise ValueError("terminal stage receipts require finished_at")
        if not terminal and self.finished_at is not None:
            raise ValueError("non-terminal stage receipts cannot have finished_at")
        if self.status is StageStatus.PENDING:
            if self.started_at is not None:
                raise ValueError("pending stage receipts cannot have started_at")
            if self.applied_sources or self.committed_sources or self.output_refs:
                raise ValueError("pending stage receipts cannot carry result fields")
        if self.status is not StageStatus.APPLIED and self.committed_sources:
            raise ValueError(
                "only applied stage receipts may carry committed_sources"
            )
        if self.status is StageStatus.PROCESSING and self.started_at is None:
            raise ValueError("processing stage receipts require started_at")
        if self.status in {StageStatus.APPLIED, StageStatus.FAILED}:
            if self.started_at is None:
                raise ValueError(
                    f"{self.status.value} stage receipts require started_at"
                )
        if self.started_at is not None and self.finished_at is not None:
            # the base comparison, so a datetime subclass cannot answer for the ordering
            if _utc_instant_before(self.started_at, self.finished_at):
                raise ValueError("finished_at cannot precede started_at")
        if self.status is StageStatus.FAILED:
            if self.error is None:
                raise ValueError("failed stage receipts require an error")
        elif self.error is not None:
            raise ValueError("only failed stage receipts may carry an error")
        return self


class IntentStatus(str, Enum):
    ACCEPTED = "accepted"
    PROCESSING = "processing"
    APPLIED = "applied"
    PARTIAL = "partial"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SUPERSEDED = "superseded"


_LEGAL_INTENT_TRANSITIONS: dict[IntentStatus, frozenset[IntentStatus]] = {
    IntentStatus.ACCEPTED: frozenset(
        {
            IntentStatus.PROCESSING,
            IntentStatus.PARTIAL,
            IntentStatus.FAILED,
            IntentStatus.CANCELLED,
            IntentStatus.SUPERSEDED,
        }
    ),
    IntentStatus.PROCESSING: frozenset(
        {
            IntentStatus.APPLIED,
            IntentStatus.PARTIAL,
            IntentStatus.FAILED,
            IntentStatus.CANCELLED,
            IntentStatus.SUPERSEDED,
        }
    ),
    IntentStatus.APPLIED: frozenset(),
    IntentStatus.PARTIAL: frozenset(),
    IntentStatus.FAILED: frozenset(),
    IntentStatus.CANCELLED: frozenset(),
    IntentStatus.SUPERSEDED: frozenset(),
}


def is_legal_intent_transition(current: IntentStatus, target: IntentStatus) -> bool:
    """Return whether one persisted intent state may advance to another."""

    if not is_exact_type(current, IntentStatus) or not any(
        current is member for member in IntentStatus
    ):
        raise TypeError("current must be an IntentStatus")
    if not is_exact_type(target, IntentStatus) or not any(
        target is member for member in IntentStatus
    ):
        raise TypeError("target must be an IntentStatus")
    return target in _LEGAL_INTENT_TRANSITIONS[current]


def source_versions_match(
    expected: SourceVersion,
    observed: SourceVersion,
) -> bool:
    """Require exact equality after deep validation of both version witnesses."""

    if not _is_instance_by_type(expected, SourceVersion):
        raise TypeError("value must be a SourceVersion")
    budget = DiscoveryBudget()  # one envelope for both witnesses
    shared: dict[Any, Any] = {}  # and one observation of any view they share
    try:
        captured_expected = _capture_contract_graph(expected, None, budget, shared)
        expected_capture_error: Exception | None = None
    except Exception as exc:
        captured_expected = None
        expected_capture_error = exc

    if _is_instance_by_type(observed, SourceVersion):
        try:
            captured_observed = _capture_contract_graph(observed, None, budget, shared)
            observed_capture_error: Exception | None = None
        except Exception as exc:
            captured_observed = None
            observed_capture_error = exc
    else:
        captured_observed = None
        observed_capture_error = TypeError("value must be a SourceVersion")

    if expected_capture_error is not None:
        raise expected_capture_error
    assert captured_expected is not None
    expected = _revalidated_captured_model(SourceVersion, captured_expected, budget)

    if observed_capture_error is not None:
        raise observed_capture_error
    assert captured_observed is not None
    observed = _revalidated_captured_model(SourceVersion, captured_observed, budget)

    return (
        expected.record_id == observed.record_id
        and expected.content_revision == observed.content_revision
        and expected.policy_epoch == observed.policy_epoch
    )


def _revalidated_claim_inputs(
    intent: object,
    receipts: object,
) -> tuple[Intent, tuple[StageReceipt, ...]]:
    """Capture and strictly revalidate an intent and its receipts under one budget.

    Shared by the aggregate claim and by record-time validation, so a receipt set the ledger
    stores is always one the claim can read.  Public views are observed once per call.
    """

    if not _is_instance_by_type(receipts, tuple):
        raise TypeError("receipts must be a tuple")

    if not _is_instance_by_type(intent, Intent):
        raise TypeError("value must be an Intent")
    budget = DiscoveryBudget(LIFECYCLE_TIER)  # one envelope for the intent and every receipt
    receipt_inventory: dict[Any, Any] = {}  # and one observation of any view they share
    try:
        captured_intent = _capture_contract_graph(intent, None, budget, receipt_inventory)
        intent_capture_error: Exception | None = None
    except Exception as exc:
        captured_intent = None
        intent_capture_error = exc

    receipt_capture_errors: dict[int, Exception] = {}
    # The intent's error is raised first whatever the receipts hold, so after it nothing is captured.
    # Each receipt reference is charged before it is filtered or deduplicated, and the inventory stops
    # at exhaustion or at the first capture error (only one error is ever raised, so no later one is
    # kept): receipts not inventoried are captured on demand, in order, so an earlier receipt's own
    # error still comes first and later ones report their own error or the exhausted budget.
    for receipt in () if intent_capture_error is not None else tuple.__iter__(receipts):
        try:
            budget.charge_edge()
        except ValueError:
            break
        if not _derives_from(type(receipt), (StageReceipt,)):
            continue
        identity = id(receipt)
        if identity in receipt_inventory or identity in receipt_capture_errors:
            continue
        try:
            _capture_native_contract_graph(
                receipt,
                captured_by_identity=receipt_inventory,
                budget=budget,
            )
        except Exception as exc:
            receipt_capture_errors[identity] = exc
            break

    if intent_capture_error is not None:
        raise intent_capture_error
    assert captured_intent is not None
    checked_intent = _revalidated_captured_model(Intent, captured_intent, budget)

    def revalidate_receipt(receipt: object) -> StageReceipt:
        capture_error = receipt_capture_errors.get(id(receipt))
        if capture_error is not None and _is_instance_by_type(receipt, StageReceipt):
            raise capture_error
        return _revalidated_inventory_model(
            StageReceipt,
            receipt,
            receipt_inventory,
            budget,
        )

    if is_exact_type(receipts, tuple):
        # An exact tuple is immutable and runs no caller code: read it in place, so a tail the
        # inventory never charged (it stops at the first capture error) is never copied.
        public_receipts: tuple[object, ...] = receipts
    else:
        # A tuple subclass's public view is read through the same first-observation cache as
        # every nested view: if the intent already observed this object (one tuple shared
        # between a field and the root), its first view is replayed, never a second one.
        public_receipts = tuple(
            item.value if isinstance(item, _CapturedLeaf) else item.source
            for item in _observed_public_view(receipts, receipt_inventory, budget)
        )
    checked = tuple(revalidate_receipt(receipt) for receipt in public_receipts)

    requirements = {item.stage: item for item in checked_intent.stage_requirements}
    for receipt in checked:
        if receipt.intent_id != checked_intent.id:
            raise ValueError("stage receipt belongs to a different intent")
        if any(
            output_ref.tenant_id != checked_intent.tenant_id or output_ref.cross_tenant
            for output_ref in receipt.output_refs
        ):
            raise ValueError("stage receipt output_refs must match intent tenant_id")
        requirement = requirements.get(receipt.stage)
        if requirement is not None and receipt.effect_id != requirement.effect_id:
            raise ValueError("required-stage receipt must match its declared effect_id")
    return checked_intent, checked


def validate_stage_receipts(
    intent: Intent,
    receipts: tuple[StageReceipt, ...],
) -> tuple[Intent, tuple[StageReceipt, ...]]:
    """Strictly revalidate an intent and its receipts exactly as the aggregate claim reads them.

    This is the record-time seam: a store passes the intent and every receipt it holds for it,
    including the one being recorded.  It applies the same budget (the lifecycle tier, one
    envelope for the whole call) and the same per-receipt identity checks as
    ``validate_required_stage_claim``, and returns the revalidated copies.  A set that is refused
    here (for example ``contract graph exceeds the native discovery budget``) is one the claim
    could never settle, so it can be refused up front instead of stranding the intent.

    The returned copies are what a store persists and what the claim later reads, and validation
    fills schema defaults (every handle gains ``cross_tenant``), so a copy can cost more to read
    than its input.  The copies are therefore read once more under a fresh envelope of the same
    tier, exactly as the claim will read them: a set whose copies exceed it is refused here.
    Revalidating copies adds nothing, so what this returns is a fixed point the claim can read.
    """

    checked_intent, checked = _revalidated_claim_inputs(intent, receipts)
    return _revalidated_claim_inputs(checked_intent, checked)


def validate_required_stage_claim(
    *,
    claimed_status: IntentStatus,
    intent: Intent,
    receipts: tuple[StageReceipt, ...],
) -> None:
    """Reject aggregate claims unsupported by fully revalidated inputs.

    Existing Pydantic instances are not proof of validity because unchecked
    construction and copy updates can bypass validators.  This public decision
    boundary therefore materializes every stored field and revalidates each
    complete nested graph before inspecting it.

    Every applied receipt for a required stage must match its declared effect,
    calculated inputs, committed generations, output minimum, and upstream
    support.  Once matching applied evidence exists, that stage remains
    satisfied; greatest-attempt state describes only stages that remain
    unsatisfied.

    A non-null ``started_at`` on any historical required-stage receipt is
    monotonic evidence that work started.  Greatest-attempt state still decides
    whether unsatisfied work remains unfinished.

    ``PROCESSING`` stays valid while any required stage is unfinished, even if
    other required stages already failed, were skipped, or were fenced; with no
    started evidence the only legal aggregate is ``ACCEPTED``.  Once every
    required stage has settled exactly one terminal aggregate is legal:

    * every stage applied: ``APPLIED``;
    * some applied, some not: ``PARTIAL`` (receipts keep the per-stage detail,
      including cancelled or superseded stages);
    * none applied: ``FAILED`` if any stage failed, else ``CANCELLED`` if any
      was cancelled, else ``SUPERSEDED`` if any was superseded, else (all
      skipped) ``PARTIAL``.

    Application witnesses: an APPLIED receipt needs a nonempty ``applied_sources`` unless its
    requirement declares ``no_source_witness`` (empty equality is not evidence), and must cite,
    in ``upstream_applications``, exactly one APPLIED upstream receipt (intent, stage, attempt,
    effect id) per upstream stage it consumed.

    Clock rule (pinned by test): receipt timestamps are *not* compared with
    ``Intent.accepted_at`` or with each other.  Wall-clock times from different workers are not
    causal evidence; sequence is proven by the ledger's per-aggregate versions and, between
    stages, by the upstream application witnesses above.

    Tenancy: ``target_refs`` and ``output_refs`` stay tenant-scoped (a write never targets or
    produces another tenant's record); only lineage handles (``MemoryRecord.source_refs``) may be
    marked ``cross_tenant``, as data whose resolution needs a fresh authorization check.

    Supported envelope: a container's *public* view is authoritative and is captured once at first
    observation.  A tuple subclass whose native backing differs from its public view (shared between
    fields or not) is outside that envelope: the backing is never read as the answer, so a claim that
    only the backing supports is rejected, and anything that cannot be reconciled with the public view
    is refused rather than accepted.  Hostile in-process callbacks are out of scope.

    This checks consistency of caller-supplied reports.  It does not prove an
    external effect exists, choose authoritative output references, establish
    retry exhaustion, or select a runtime retry policy.
    """

    if not is_exact_type(claimed_status, IntentStatus) or not any(
        claimed_status is member for member in IntentStatus
    ):
        raise TypeError("claimed_status must be an IntentStatus member")
    intent, receipts = _revalidated_claim_inputs(intent, receipts)

    requirements = {item.stage: item for item in intent.stage_requirements}
    required = set(requirements)
    latest_attempts: dict[ProcessingStage, int] = {}
    for receipt in receipts:
        previous_attempt = latest_attempts.get(receipt.stage)
        if previous_attempt is None or receipt.attempt > previous_attempt:
            latest_attempts[receipt.stage] = receipt.attempt

    latest: dict[ProcessingStage, StageReceipt] = {}
    attempt_receipts: dict[tuple[ProcessingStage, int], StageReceipt] = {}
    for receipt in receipts:
        if receipt.stage in required:
            attempt_key = (receipt.stage, receipt.attempt)
            previous_attempt_receipt = attempt_receipts.get(attempt_key)
            if previous_attempt_receipt is None:
                attempt_receipts[attempt_key] = receipt
            elif receipt != previous_attempt_receipt and (
                receipt.status is StageStatus.APPLIED
                or previous_attempt_receipt.status is StageStatus.APPLIED
            ):
                raise ValueError("conflicting receipts for the same stage attempt")

        if receipt.attempt != latest_attempts[receipt.stage]:
            continue
        previous = latest.get(receipt.stage)
        if previous is None:
            latest[receipt.stage] = receipt
        elif receipt != previous:
            raise ValueError("conflicting receipts for the same stage attempt")

    required_receipts = {stage: latest.get(stage) for stage in required}
    satisfied: set[ProcessingStage] = set()
    applied_by_stage: dict[ProcessingStage, list[StageReceipt]] = {
        stage: [] for stage in required
    }
    for receipt in receipts:
        if receipt.stage in required and receipt.status is StageStatus.APPLIED:
            applied_by_stage[receipt.stage].append(receipt)

    for stage in _topological_stages(requirements):
        requirement = requirements[stage]
        stage_applications = applied_by_stage[stage]
        expected_inputs = _expected_stage_inputs(intent, requirement, requirements)
        for receipt in stage_applications:
            if not _source_sets_match(expected_inputs, receipt.applied_sources):
                raise ValueError(
                    "applied required-stage receipt must match calculated inputs"
                )
            if not _source_sets_match(
                requirement.committed_sources, receipt.committed_sources
            ):
                raise ValueError(
                    "applied required-stage receipt must match committed_sources"
                )
            if len(receipt.output_refs) < requirement.minimum_output_refs:
                raise ValueError(
                    "applied required-stage receipt does not meet minimum_output_refs"
                )
            if not set(requirement.upstream_stages) <= satisfied:
                raise ValueError(
                    "applied required-stage receipt requires upstream applied support"
                )
            if not receipt.applied_sources and not requirement.no_source_witness:
                raise ValueError(
                    "applied required-stage receipt needs a nonempty source witness "
                    "unless the requirement declares no_source_witness"
                )
            cited = {item.stage: item for item in receipt.upstream_applications}
            if set(cited) != set(requirement.upstream_stages):
                raise ValueError(
                    "applied required-stage receipt must cite exactly the upstream "
                    "applications it consumed"
                )
            for upstream, witness in cited.items():
                # One indexed lookup per witness (a scan of the upstream history would make
                # many applications citing one upstream attempt quadratic).  An attempt with an
                # APPLIED receipt is indexed by that receipt: any other receipt for the same
                # attempt is either equal to it or refused as a conflict above.
                upstream_receipt = attempt_receipts.get((upstream, witness.attempt))
                if (
                    witness.intent_id != intent.id
                    or upstream_receipt is None
                    or upstream_receipt.status is not StageStatus.APPLIED
                    or upstream_receipt.effect_id != witness.effect_id
                ):
                    raise ValueError(
                        "upstream application witness does not name an applied upstream receipt"
                    )
        if stage_applications:
            first = stage_applications[0]
            for receipt in stage_applications[1:]:
                if not (
                    _source_sets_match(first.applied_sources, receipt.applied_sources)
                    and _source_sets_match(
                        first.committed_sources, receipt.committed_sources
                    )
                    and _reference_sets_match(first.output_refs, receipt.output_refs)
                ):
                    raise ValueError(
                        "applied receipts for a stage must report consistent "
                        "result sets"
                    )
            satisfied.add(stage)

    started_stages = {
        receipt.stage
        for receipt in receipts
        if receipt.stage in required and receipt.started_at is not None
    }
    unsatisfied = required - satisfied
    unsatisfied_receipts = {
        stage: required_receipts[stage] for stage in unsatisfied
    }
    # A stage whose current attempt is already settled (skipped, failed,
    # cancelled or superseded) counts as started work even if it never
    # recorded ``started_at``; a superseded pre-start fence followed by a
    # pending retry does not.
    started_stages |= {
        stage
        for stage, receipt in unsatisfied_receipts.items()
        if receipt is not None and receipt.status is not StageStatus.PENDING
    }
    unfinished = any(
        receipt is None
        or receipt.status in {StageStatus.PENDING, StageStatus.PROCESSING}
        for receipt in unsatisfied_receipts.values()
    )

    if claimed_status is IntentStatus.ACCEPTED:
        if started_stages or any(
            receipt is not None and receipt.status is not StageStatus.PENDING
            for receipt in unsatisfied_receipts.values()
        ):
            raise ValueError("accepted cannot claim started or terminal required stages")
    elif claimed_status is IntentStatus.PROCESSING:
        # Confirmed finality direction: a job with any required stage still
        # unfinished stays nonterminal PROCESSING even when other required
        # stages already failed, were skipped, or were fenced; per-stage
        # outcomes remain visible in the receipts.
        if not started_stages or not unfinished:
            raise ValueError(
                "processing requires started and unfinished required work"
            )
    elif claimed_status in _TERMINAL_CLAIMS:
        if unfinished:
            raise ValueError(
                f"{claimed_status.value} requires every required stage to have settled"
            )
        expected = _settled_aggregate(
            satisfied=satisfied == required,
            any_applied=bool(satisfied),
            statuses={
                receipt.status
                for receipt in unsatisfied_receipts.values()
                if receipt is not None
            },
        )
        if claimed_status is not expected:
            raise ValueError(
                f"{claimed_status.value} does not match the settled aggregate "
                f"({expected.value}) for the required-stage receipts"
            )
    else:
        raise ValueError("unsupported IntentStatus")


_TERMINAL_CLAIMS = frozenset(
    {
        IntentStatus.APPLIED,
        IntentStatus.PARTIAL,
        IntentStatus.FAILED,
        IntentStatus.CANCELLED,
        IntentStatus.SUPERSEDED,
    }
)


def _settled_aggregate(
    *, satisfied: bool, any_applied: bool, statuses: set[StageStatus]
) -> IntentStatus:
    """The single terminal aggregate for fully settled required stages.

    Every required stage is applied or settled (skipped, failed, cancelled,
    superseded):

    * all applied -> ``APPLIED``;
    * some applied, some not -> ``PARTIAL`` (per-stage receipts keep the
      failed/skipped/fenced detail);
    * none applied: any failed -> ``FAILED``; else any cancelled ->
      ``CANCELLED``; else any superseded -> ``SUPERSEDED``; else every stage
      was skipped -> ``PARTIAL``.
    """

    if satisfied:
        return IntentStatus.APPLIED
    if any_applied:
        return IntentStatus.PARTIAL
    if StageStatus.FAILED in statuses:
        return IntentStatus.FAILED
    if StageStatus.CANCELLED in statuses:
        return IntentStatus.CANCELLED
    if StageStatus.SUPERSEDED in statuses:
        return IntentStatus.SUPERSEDED
    return IntentStatus.PARTIAL


def _source_sets_match(
    expected: tuple[SourceVersion, ...],
    observed: tuple[SourceVersion, ...],
) -> bool:
    if len(expected) != len(observed):
        return False
    observed_by_id = {item.record_id: item for item in observed}
    return len(observed_by_id) == len(observed) and all(
        source.record_id in observed_by_id
        and source_versions_match(source, observed_by_id[source.record_id])
        for source in expected
    )


def _reference_sets_match(
    expected: tuple[ReferenceHandle, ...],
    observed: tuple[ReferenceHandle, ...],
) -> bool:
    if len(expected) != len(observed):
        return False
    expected_keys = {_reference_identity_key(item) for item in expected}
    observed_keys = {_reference_identity_key(item) for item in observed}
    return (
        len(expected_keys) == len(expected)
        and len(observed_keys) == len(observed)
        and expected_keys == observed_keys
    )


def _require_unique_reference_handles(
    references: tuple[ReferenceHandle, ...], field: str
) -> None:
    seen: set[tuple[str, str, str, str]] = set()
    for reference in references:
        key = _reference_identity_key(reference)
        if key in seen:
            raise ValueError(f"{field} must not contain duplicate handles")
        seen.add(key)


def _reference_identity_key(
    reference: ReferenceHandle,
) -> tuple[str, str, str, str]:
    return (
        reference.kind,
        reference.id,
        reference.tenant_id,
        reference.resolver,
    )


def _topological_stages(
    requirements: dict[ProcessingStage, StageEvidenceRequirement],
) -> tuple[ProcessingStage, ...]:
    visiting: set[ProcessingStage] = set()
    visited: set[ProcessingStage] = set()
    ordered: list[ProcessingStage] = []

    def visit(stage: ProcessingStage) -> None:
        if stage in visiting:
            raise ValueError("stage_requirements upstream graph must be acyclic")
        if stage in visited:
            return
        visiting.add(stage)
        for upstream in sorted(
            requirements[stage].upstream_stages, key=lambda item: item.value
        ):
            visit(upstream)
        visiting.remove(stage)
        visited.add(stage)
        ordered.append(stage)

    for stage in sorted(requirements, key=lambda item: item.value):
        visit(stage)
    return tuple(ordered)


def _expected_stage_inputs(
    intent: Intent,
    requirement: StageEvidenceRequirement,
    requirements: dict[ProcessingStage, StageEvidenceRequirement],
) -> tuple[SourceVersion, ...]:
    preconditions = {item.record_id: item for item in intent.source_preconditions}
    selected = [preconditions[item] for item in requirement.precondition_source_ids]
    for upstream in requirement.upstream_stages:
        selected.extend(requirements[upstream].committed_sources)
    return tuple(selected)


def _require_unique_source_ids(sources: tuple[SourceVersion, ...], field: str) -> None:
    ids = [source.record_id for source in sources]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{field} must not contain duplicate record_id values")


__all__ = [
    "ApplicabilityScope",
    "Intent",
    "IntentStatus",
    "MemoryLifecycle",
    "MemoryRecord",
    "MemoryRecordKind",
    "OpaqueEnvelopeMemoryBody",
    "PlaintextMemoryBody",
    "ProcessingStage",
    "StageEvidenceRequirement",
    "StageError",
    "StageReceipt",
    "StageStatus",
    "LIFECYCLE_TIER",
    "is_legal_intent_transition",
    "source_versions_match",
    "validate_required_stage_claim",
    "validate_stage_receipts",
]
