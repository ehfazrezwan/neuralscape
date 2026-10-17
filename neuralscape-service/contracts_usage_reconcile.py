"""Pure, order-independent reconciliation for :mod:`contracts_usage` events."""

from __future__ import annotations

import threading
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, GetCoreSchemaHandler, model_validator
from pydantic_core import core_schema

from contracts_common import ContractModel, MAX_SAFE_INTEGER, OpaqueId, SafeCounter
from contracts_limits import (
    DISCOVERY_BUDGET_MESSAGE,
    STANDARD,
    DiscoveryBudget,
    inherits_native,
    is_native_instance,
)
from contracts_tenancy import unpack_pairs
from contracts_usage import (
    AttributionSnapshot,
    AttemptOutcome,
    MissingReason,
    TokenUsage,
    UsageEvent,
    UsageLedger,
    UsageStatus,
)


_MAX_SAFE_COUNTER = MAX_SAFE_INTEGER
_MODEL_DICT_DESCRIPTOR = BaseModel.__dict__["__dict__"]
_MODEL_EXTRAS_DESCRIPTOR = BaseModel.__dict__["__pydantic_extra__"]
_FrozenModelStorage = tuple[
    BaseModel,
    tuple[tuple[object, object], ...],
    object,
    tuple[tuple[object, object], ...] | None,
]
_FrozenDictStorage = tuple[
    dict[object, object],
    tuple[tuple[object, object], ...],
]
_FrozenTupleStorage = tuple[
    tuple[object, ...],
    tuple[object, ...],
]
_LEDGER_ORDER: tuple[UsageLedger, ...] = (
    "service",
    "consuming_agent",
    "evaluation",
)

ReconciliationErrorCode = Literal[
    "empty_reconciliation",
    "invalid_event",
    "conflicting_event_id",
    "mixed_tenants",
    "unknown_predecessor",
    "correction_cycle",
    "branched_correction",
    "cross_stream_correction",
    "multiple_stream_roots",
    "invalid_transition",
    "counter_overflow",
]
LedgerCoverage = Literal["reported", "unreported"]
LedgerMissingReason = Literal["no_events"]


class _FailedNativeDictTraversal(dict[object, object]):
    """Let Pydantic locate one failed native-dict mapping traversal."""

    def __init__(self, error: Exception) -> None:
        super().__init__()
        self.error = error

    def items(self):
        raise self.error


@dataclass(frozen=True, slots=True)
class _ValidatedPublicMappingEntries:
    entries: Iterable[object]
    iterator: Iterator[object]
    retained_keys: dict[object, None]
    retained_entries: tuple[tuple[object, object, object], ...]


_BUDGET_KEY = ("discovery-budget",)
_TEXT_KEY = "text-copy"  # memo entries for text copies are keyed (_TEXT_KEY, id), apart from node ids
# The envelope and memo of the snapshot that is validating a public mapping view right now (per thread).
# They are passed this way because the validator's signature is part of what the existing tests replace.
_view_state = threading.local()


def _session_budget(memo: dict) -> DiscoveryBudget:
    """The one discovery envelope of this snapshot call (it lives in the call's memo)."""

    budget = memo.get(_BUDGET_KEY)
    if budget is None:
        budget = memo[_BUDGET_KEY] = DiscoveryBudget(STANDARD)
    return budget


def _precheck(budget: DiscoveryBudget, count: int) -> None:
    """Refuse to copy native storage that cannot fit the remaining edge allowance (before copying it)."""

    if count > budget.tier.max_edges - budget.edges:
        budget.charge_edge(count)  # raises the stable budget error


def _sequence_items(
    value: object,
    exact: type,
    captured: dict[int, tuple[object, tuple[object, ...]]],
    budget: DiscoveryBudget,
) -> Iterable[object]:
    """The items a list or set is rebuilt from.

    An instance of exactly ``exact`` is rebuilt from what its capture copied (and charged), not read
    again: a callback that ran since then cannot grow it past the envelope.  A subclass keeps its
    public view, which is a further read and is charged as it is read.
    """

    if type(value) is exact:
        held = captured.get(id(value))
        if held is not None and held[0] is value:
            return held[1]
    return _charged(value, budget)


def _charged(entries: Iterable[object], budget: DiscoveryBudget) -> Iterator[object]:
    """Yield each entry of a public view after charging it, so an endless or huge view is cut off."""

    for entry in entries:
        budget.charge_edge()
        yield entry


def _validate_public_mapping_entries(
    entries: Iterable[object],
) -> _FailedNativeDictTraversal | _ValidatedPublicMappingEntries:
    """Validate public mapping entry shape and key insertion only."""

    try:
        source_iterator = iter(entries)
    except Exception as exc:
        return _FailedNativeDictTraversal(exc)
    public_entries: dict[object, None] = {}
    retained_entries: list[tuple[object, object, object]] = []
    while True:
        try:
            pair = next(source_iterator)
        except StopIteration:
            break
        except Exception as exc:
            return _FailedNativeDictTraversal(exc)
        shared = getattr(_view_state, "budget", None)
        memo = getattr(_view_state, "memo", None)
        if shared is not None:
            shared.charge_edge()  # every pair read is charged to the snapshot's one envelope
        elif len(retained_entries) >= STANDARD.max_edges:
            raise ValueError(DISCOVERY_BUDGET_MESSAGE)  # no envelope given: this one view is still bounded
        try:
            key, public_value = pair
        except Exception as exc:
            return _FailedNativeDictTraversal(exc)
        try:
            public_entries[_hook_free_key(key, memo, shared)] = None  # the shape check hashes no key of the caller's
        except Exception as exc:
            return _FailedNativeDictTraversal(exc)
        retained_entries.append((pair, key, public_value))
    return _ValidatedPublicMappingEntries(
        entries=entries,
        iterator=source_iterator,
        retained_keys=public_entries,
        retained_entries=tuple(retained_entries),
    )


class UsageReconciliationError(ValueError):
    """A deterministic validation failure in a supplied event set."""

    def __init__(self, code: ReconciliationErrorCode, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


def _snapshot_python_input(value: Any) -> Any:
    return _native_snapshot(value)


class ReconciledUsageStream(ContractModel):
    """The selected head of one attempt/ledger correction chain."""

    model_config = ConfigDict(revalidate_instances="always")

    tenant_id: OpaqueId
    task_id: OpaqueId
    session_id: OpaqueId | None
    intent_id: OpaqueId | None
    attempt_id: OpaqueId
    ledger: UsageLedger
    operation: OpaqueId
    head_event_id: OpaqueId
    status: UsageStatus
    attempt_outcome: AttemptOutcome
    late_after_cancellation: bool
    attribution: AttributionSnapshot
    usage: TokenUsage | None
    usage_missing_reason: MissingReason | None
    known_token_subtotal: SafeCounter
    total_tokens: SafeCounter | None
    incomplete_categories: tuple[str, ...]

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source_type: Any,
        handler: GetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:
        schema = handler(source_type)
        return core_schema.json_or_python_schema(
            json_schema=schema,
            python_schema=core_schema.no_info_before_validator_function(
                _snapshot_python_input,
                schema,
            ),
        )

    @model_validator(mode="after")
    def validate_usage_summary(self) -> "ReconciledUsageStream":
        if self.status == "unavailable":
            if self.usage is not None or self.usage_missing_reason is None:
                raise ValueError(
                    "unavailable stream requires no token usage and a missing reason"
                )
        else:
            if self.usage_missing_reason is not None:
                raise ValueError(
                    "usage_missing_reason is only valid for unavailable streams"
                )
            if self.status == "final" and self.usage is None:
                raise ValueError("final stream requires token usage")
        if self.late_after_cancellation:
            if self.status != "final" or self.attempt_outcome != "cancelled":
                raise ValueError(
                    "late stream must be final usage for a cancelled attempt"
                )

        known, missing = _token_total(self.usage)
        if self.status != "final" and "usage_status" not in missing:
            missing = (*missing, "usage_status")
        if self.known_token_subtotal != known:
            raise ValueError("stream subtotal does not match token usage")
        if self.incomplete_categories != missing:
            raise ValueError("stream incomplete categories do not match token usage")
        expected_total = known if not missing else None
        if self.total_tokens != expected_total:
            raise ValueError("stream total does not match token usage completeness")
        return self


class ReconciledLedger(ContractModel):
    """One ledger total with explicit evidence coverage.

    ``known_token_subtotal`` may be zero without proving a complete zero total.
    A ledger with no events is unreported and therefore has no ``total_tokens``.
    """

    model_config = ConfigDict(revalidate_instances="always")

    ledger: UsageLedger
    coverage: LedgerCoverage
    missing_reason: LedgerMissingReason | None
    known_token_subtotal: SafeCounter
    total_tokens: SafeCounter | None
    incomplete_attempt_ids: tuple[OpaqueId, ...]

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source_type: Any,
        handler: GetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:
        schema = handler(source_type)
        return core_schema.json_or_python_schema(
            json_schema=schema,
            python_schema=core_schema.no_info_before_validator_function(
                _snapshot_python_input,
                schema,
            ),
        )

    @model_validator(mode="after")
    def validate_coverage(self) -> "ReconciledLedger":
        if self.coverage == "unreported":
            if self.missing_reason != "no_events":
                raise ValueError("unreported ledger requires a no_events reason")
            if self.known_token_subtotal != 0 or self.total_tokens is not None:
                raise ValueError("unreported ledger cannot claim measured token totals")
            if self.incomplete_attempt_ids:
                raise ValueError("unreported ledger cannot name observed attempts")
            return self

        if self.missing_reason is not None:
            raise ValueError("reported ledger cannot have a coverage missing reason")
        if self.incomplete_attempt_ids:
            if self.total_tokens is not None:
                raise ValueError("incomplete reported ledger cannot have a complete total")
        elif self.total_tokens != self.known_token_subtotal:
            raise ValueError("complete reported total must equal its known subtotal")
        return self


class UsageReconciliation(ContractModel):
    """Reconciled streams and three totals for exactly one tenant."""

    model_config = ConfigDict(revalidate_instances="always")

    tenant_id: OpaqueId
    streams: tuple[ReconciledUsageStream, ...]
    ledgers: tuple[ReconciledLedger, ...]

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source_type: Any,
        handler: GetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:
        schema = handler(source_type)
        return core_schema.json_or_python_schema(
            json_schema=schema,
            python_schema=core_schema.no_info_before_validator_function(
                _snapshot_python_input,
                schema,
            ),
        )

    @model_validator(mode="after")
    def validate_result_graph(self) -> "UsageReconciliation":
        if not self.streams:
            raise ValueError("reconciliation result requires at least one stream")
        if any(stream.tenant_id != self.tenant_id for stream in self.streams):
            raise ValueError("every stream tenant must match the reconciliation tenant")
        if tuple(ledger.ledger for ledger in self.ledgers) != _LEDGER_ORDER:
            raise ValueError("result requires one canonical summary for each ledger")

        stream_keys = {
            (
                stream.tenant_id,
                stream.task_id,
                stream.session_id,
                stream.intent_id,
                stream.attempt_id,
                stream.ledger,
                stream.operation,
            )
            for stream in self.streams
        }
        if len(stream_keys) != len(self.streams):
            raise ValueError("reconciliation result contains duplicate streams")
        if len({stream.head_event_id for stream in self.streams}) != len(self.streams):
            raise ValueError("reconciliation result contains duplicate head event ids")

        for ledger_summary in self.ledgers:
            ledger_streams = tuple(
                stream
                for stream in self.streams
                if stream.ledger == ledger_summary.ledger
            )
            incomplete_ids = tuple(
                stream.attempt_id
                for stream in ledger_streams
                if stream.total_tokens is None
            )
            known = _checked_sum(
                stream.known_token_subtotal for stream in ledger_streams
            )
            expected = {
                "coverage": "reported" if ledger_streams else "unreported",
                "missing_reason": None if ledger_streams else "no_events",
                "known_token_subtotal": known,
                "total_tokens": (
                    known if ledger_streams and not incomplete_ids else None
                ),
                "incomplete_attempt_ids": incomplete_ids,
            }
            for field_name, expected_value in expected.items():
                if getattr(ledger_summary, field_name) != expected_value:
                    raise ValueError(
                        f"{ledger_summary.ledger} ledger {field_name} does not match streams"
                    )
        return self


_TYPE_MRO = type.__dict__["__mro__"]


def _is_real_mapping(value: object) -> bool:
    """A mapping subclass or the read-only view; a class that only registers with the ABC is not one."""

    candidate = type(value)
    return candidate is MappingProxyType or any(
        entry is Mapping or entry is dict for entry in _TYPE_MRO.__get__(candidate)
    )


def _inherits_str(candidate: type) -> bool:
    return any(entry is str for entry in _TYPE_MRO.__get__(candidate))


_CLASS_NAMESPACE = type.__dict__["__dict__"]


def _declared_field_names(model_type: type, budget: DiscoveryBudget) -> set[object]:
    """The model's declared field names, from the first namespace on its native MRO that defines them.

    pydantic keeps them as a plain dict in every model class's own namespace.  Namespaces and that dict are
    read natively (only exact-text keys are compared, and the entries scanned are charged), so no metaclass
    ``__getattribute__``, descriptor or key hook runs.  Anything else is refused.
    """

    for klass in _TYPE_MRO.__get__(model_type):
        namespace = _CLASS_NAMESPACE.__get__(klass)
        budget.charge_edge(MappingProxyType.__len__(namespace))
        for key, fields in MappingProxyType.items(namespace):
            if type(key) is not str:
                raise ValueError("contract model metadata must be read natively")
            if key == "__pydantic_fields__":
                if type(fields) is not dict:
                    raise ValueError("contract model metadata must be read natively")
                budget.charge_edge(dict.__len__(fields))
                names = tuple(dict.__iter__(fields))
                if any(type(name) is not str for name in names):
                    raise ValueError("contract model metadata must be read natively")
                return set(names)
    raise ValueError("contract model metadata must be read natively")


def _plain_name(name: object) -> object:
    """A stored field name as plain text (read without running a text subclass's own hooks).

    Anything that is not text is returned unchanged for the caller's text checks to refuse.
    """

    if type(name) is str or not _inherits_str(type(name)):
        return name
    return str.__str__(name)  # type: ignore[arg-type]


def _memo_text(name: object, memo: dict) -> object:
    """``_plain_name`` once per call per text-subclass object: the copy is kept in the call's memo with the
    object (so its id cannot be reused), and every later reference (a leaf, a key, a field name) shares it.

    Its entries have their own key space in the memo, so a text subclass that is also projected as a
    container keeps the two results apart.
    """

    if type(name) is str or not _inherits_str(type(name)):
        return name
    slot = (_TEXT_KEY, id(name))
    cached = memo.get(slot)
    if cached is not None and cached[0] is name:
        return cached[1]
    text = _plain_name(name)
    memo[slot] = (name, text)
    return text


def _hashes_natively(kind: type, budget: DiscoveryBudget | None) -> bool:
    """Whether hashing an instance of ``kind`` runs no caller code: its ``__hash__`` and ``__eq__`` are
    ``object``'s own, or it is unhashable (hashing raises natively before anything is compared).

    The native MRO's namespaces are scanned directly, comparing only exact-text names (charged when the call
    has an envelope), so no metaclass or descriptor runs; a namespace with any other name is not trusted.
    """

    hash_found = eq_found = False
    for klass in _TYPE_MRO.__get__(kind):
        namespace = _CLASS_NAMESPACE.__get__(klass)
        if budget is not None:
            budget.charge_edge(MappingProxyType.__len__(namespace))
        for name, entry in MappingProxyType.items(namespace):
            if type(name) is not str:
                return False
            if name == "__hash__" and not hash_found:
                if entry is None:
                    return True
                if klass is not object:
                    return False
                hash_found = True
            elif name == "__eq__" and not eq_found:
                if klass is not object:
                    return False
                eq_found = True
        if hash_found and eq_found:
            return True
    return False


def _hook_free_key(key: object, memo: dict | None = None, budget: DiscoveryBudget | None = None) -> object:
    """A mapping key as something that hashes and compares without caller code: a text subclass as its
    text, an exact native scalar as itself, and an object hashed by ``object``'s own slots (or unhashable,
    which fails natively) unchanged.  Any other key is refused before it is hashed or compared."""

    kind = type(key)
    if kind is str or kind is int or kind is bool or kind is float or kind is bytes or key is None:
        return key
    if _inherits_str(kind):
        return _plain_name(key) if memo is None else _memo_text(key, memo)
    if _hashes_natively(kind, budget):
        return key
    raise ValueError("contract input keys must be text")


def _is_plain_leaf(value: object) -> bool:
    """An exact ``str``, ``int``, ``bool``, ``float`` or ``None``: captured and rebuilt as itself.

    Compared by identity (never by hashing the type, which could run a metaclass hook), so the common leaf
    skips the container checks of every capture and rebuild pass.
    """

    kind = type(value)
    return kind is str or kind is int or kind is bool or kind is float or value is None


def _is_node(value: object) -> bool:
    return is_native_instance(value, BaseModel, list, tuple, set, frozenset) or _is_real_mapping(value)


_freeze_state = threading.local()  # the current depth of the recursive freeze, per thread


def _freeze_model_storage(
    value: object,
    frozen_models: dict[int, _FrozenModelStorage],
    frozen_dicts: dict[int, _FrozenDictStorage],
    frozen_tuples: dict[int, _FrozenTupleStorage],
    visited: set[int],
    budget: DiscoveryBudget,
) -> None:
    """Capture reachable native model stores without public traversal hooks.

    Every capture is charged to the call's one envelope: each object once as a node, and each
    container's native entries as edges, checked against what is left before they are copied.
    """

    if _is_plain_leaf(value):
        return  # nothing to capture
    depth = getattr(_freeze_state, "depth", 0)
    if depth > STANDARD.max_depth:
        raise ValueError(DISCOVERY_BUDGET_MESSAGE)  # nesting beyond the envelope is refused, not recursed into
    _freeze_state.depth = depth + 1
    try:
        _freeze_model_storage_body(value, frozen_models, frozen_dicts, frozen_tuples, visited, budget)
    finally:
        _freeze_state.depth = depth


def _freeze_model_storage_body(
    value: object,
    frozen_models: dict[int, _FrozenModelStorage],
    frozen_dicts: dict[int, _FrozenDictStorage],
    frozen_tuples: dict[int, _FrozenTupleStorage],
    visited: set[int],
    budget: DiscoveryBudget,
) -> None:

    value_type = type(value)
    is_native_container = any(
        inherits_native(value_type, base) for base in (BaseModel, dict, list, tuple, set, frozenset)
    )
    if not is_native_container:
        return

    identity = id(value)
    if identity in visited:
        return
    visited.add(identity)
    budget.charge_node(value)  # by identity: an object already charged (wherever) costs nothing more
    if _already_frozen(value, frozen_models, frozen_dicts, frozen_tuples):
        return  # already captured in this call (a later freeze starts with a fresh ``visited``)

    if inherits_native(value_type, BaseModel):
        frozen_model = frozen_models.get(identity)
        if frozen_model is not None and frozen_model[0] is value:
            return
        stored = _MODEL_DICT_DESCRIPTOR.__get__(value, BaseModel)
        _precheck(budget, dict.__len__(stored))
        stored_items = tuple(dict.items(stored))
        budget.charge_edge(len(stored_items))
        try:
            extras = _MODEL_EXTRAS_DESCRIPTOR.__get__(value, BaseModel)
        except AttributeError:
            extras = None
        native_extra_items = None
        if extras is not None and inherits_native(type(extras), dict):
            _precheck(budget, dict.__len__(extras))
            native_extra_items = tuple(dict.items(extras))
            budget.charge_edge(len(native_extra_items))
        # The names are checked when the model is projected (``_checked_native_names``), not here: a model
        # captured from a list subclass's native backing is never projected when its public view hides it.
        frozen_models[identity] = (
            value,
            stored_items,
            extras,
            native_extra_items,
        )
        for _name, field_value in stored_items:
            _freeze_model_storage(
                field_value,
                frozen_models,
                frozen_dicts,
                frozen_tuples,
                visited,
                budget,
            )
        if native_extra_items is not None:
            for _name, field_value in native_extra_items:
                _freeze_model_storage(
                    field_value,
                    frozen_models,
                    frozen_dicts,
                    frozen_tuples,
                    visited,
                    budget,
                )
        return

    if inherits_native(value_type, dict):
        frozen_dict = frozen_dicts.get(identity)
        if frozen_dict is not None and frozen_dict[0] is value:
            return
        _precheck(budget, dict.__len__(value))
        native_items = tuple(dict.items(value))
        budget.charge_edge(len(native_items))
        frozen_dicts[identity] = (value, native_items)
        items = (item for _key, item in native_items)
    elif inherits_native(value_type, list):
        _precheck(budget, list.__len__(value))
        items = tuple(list.__iter__(value))
        budget.charge_edge(len(items))
        # rebuilt from this capture (owner retained): a callback that grows the list later is not read
        frozen_tuples[identity] = (value, items)
    elif inherits_native(value_type, tuple):
        frozen_tuple = frozen_tuples.get(identity)
        if frozen_tuple is not None and frozen_tuple[0] is value:
            return
        _precheck(budget, tuple.__len__(value))
        native_items = tuple(tuple.__iter__(value))
        budget.charge_edge(len(native_items))
        frozen_tuples[identity] = (value, native_items)
        items = iter(native_items)
    elif inherits_native(value_type, set):
        _precheck(budget, set.__len__(value))
        items = tuple(set.__iter__(value))
        budget.charge_edge(len(items))
        frozen_tuples[identity] = (value, items)
    else:
        _precheck(budget, frozenset.__len__(value))
        items = tuple(frozenset.__iter__(value))
        budget.charge_edge(len(items))
        frozen_tuples[identity] = (value, items)
    for item in items:
        _freeze_model_storage(
            item,
            frozen_models,
            frozen_dicts,
            frozen_tuples,
            visited,
            budget,
        )


def _checked_native_names(
    stored_items: tuple[tuple[object, object], ...],
    native_extra_items: tuple[tuple[object, object], ...] | None,
    memo: dict,
) -> None:
    """Refuse a projected model whose native field names are not unique text in one namespace."""

    group_texts: list[set[object]] = []
    for group in (stored_items, native_extra_items or ()):
        if any(not _inherits_str(type(name)) for name, _item in group):
            raise ValueError("contract input field names must be text")
        texts = [_memo_text(name, memo) for name, _item in group]
        if len(set(texts)) != len(texts):
            raise ValueError("contract input field names must be unique text")
        group_texts.append(set(texts))
    # The two native stores are one namespace: compare their names as plain text, never through a
    # key's own hash and equality (a text subclass could otherwise sit in both under one text).
    if group_texts[0] & group_texts[1]:
        raise ValueError("contract extra storage overlaps stored fields")


def _already_frozen(
    value: object,
    frozen_models: dict[int, _FrozenModelStorage],
    frozen_dicts: dict[int, _FrozenDictStorage],
    frozen_tuples: dict[int, _FrozenTupleStorage],
) -> bool:
    """Whether this exact object's storage was already captured in this call (owner checked by identity)."""

    identity = id(value)
    for store in (frozen_models, frozen_dicts, frozen_tuples):
        held = store.get(identity)
        if held is not None and held[0] is value:
            return True
    return False


def _native_snapshot(
    value: object,
    active: set[int] | None = None,
    *,
    _frozen_models: dict[int, _FrozenModelStorage] | None = None,
    _frozen_dicts: dict[int, _FrozenDictStorage] | None = None,
    _frozen_tuples: dict[int, _FrozenTupleStorage] | None = None,
    _memo: dict[int, tuple[object, object]] | None = None,
) -> object:
    """Project a nested native/model graph once per distinct object.

    Each container is projected at its first observation and the result is
    replayed for every later reference to the same object, so a graph that
    shares descendants costs time proportional to its distinct objects rather
    than to the number of paths, and a public traversal hook runs once per
    object per call.
    """

    if _is_plain_leaf(value):
        return value  # rebuilt as itself, whatever the pass
    if _memo is None:
        _memo = {}
    if _is_node(value):
        cached = _memo.get(id(value))
        if cached is not None and cached[0] is value:
            return cached[1]
    result = _snapshot_uncached(
        value,
        active,
        _frozen_models=_frozen_models,
        _frozen_dicts=_frozen_dicts,
        _frozen_tuples=_frozen_tuples,
        _memo=_memo,
    )
    if _is_node(value):
        _memo[id(value)] = (value, result)
    return result


def _snapshot_uncached(
    value: object,
    active: set[int] | None = None,
    *,
    _frozen_models: dict[int, _FrozenModelStorage] | None = None,
    _frozen_dicts: dict[int, _FrozenDictStorage] | None = None,
    _frozen_tuples: dict[int, _FrozenTupleStorage] | None = None,
    _memo: dict[int, tuple[object, object]],
) -> object:
    """Copy a nested native/model graph without trusting model construction.

    Reading ``__dict__`` deliberately retains unknown fields injected by
    unchecked model copies so the destination contract can reject them.
    """

    budget = _session_budget(_memo)  # the one envelope of this call, shared by capture and rebuild
    if (
        _frozen_models is None
        or _frozen_dicts is None
        or _frozen_tuples is None
    ):
        if _frozen_models is None:
            _frozen_models = {}
        if _frozen_dicts is None:
            _frozen_dicts = {}
        if _frozen_tuples is None:
            _frozen_tuples = {}
        _freeze_model_storage(
            value,
            _frozen_models,
            _frozen_dicts,
            _frozen_tuples,
            set(),
            budget,
        )
    if active is None:
        active = set()
    if not _is_node(value):
        # A text subclass reaches validation as its exact text: its own hash and equality never run there,
        # so it cannot match a literal it does not spell.
        return _memo_text(value, _memo)

    # The one charging point for every container kind, before any branch is chosen: a container first met
    # here (exposed only by a public view) costs a node like any natively reachable one, whichever branch then
    # rebuilds it (a tuple that is also a mapping takes the mapping branch).  Charging is by identity, so an
    # already captured container costs nothing more; capture decides what it has frozen by its own stores.
    budget.charge_node(value)
    identity = id(value)
    if identity in active:
        raise ValueError("cyclic input graph")
    active.add(identity)
    try:
        if len(active) > STANDARD.max_depth:
            raise ValueError(DISCOVERY_BUDGET_MESSAGE)
        if is_native_instance(value, BaseModel):
            frozen = _frozen_models.get(identity)
            if frozen is None or frozen[0] is not value:
                _freeze_model_storage(
                    value,
                    _frozen_models,
                    _frozen_dicts,
                    _frozen_tuples,
                    set(),
                    budget,
                )
                frozen = _frozen_models[identity]
            _model, stored_items, extras, native_extra_items = frozen
            _checked_native_names(stored_items, native_extra_items, _memo)
            extra_items: tuple[tuple[object, object], ...] = ()
            if extras is not None:
                if native_extra_items is not None:
                    extra_items = native_extra_items
                    extra_names = {_memo_text(name, _memo) for name, _ in extra_items}
                else:
                    if not _is_real_mapping(extras):
                        raise ValueError("contract extra storage must be a mapping")
                    iterated_extra_names = tuple(_charged(extras, budget))
                    # One unpacking per pair: the name check and the field build share these tuples.
                    extra_items = unpack_pairs(_charged(extras.items(), budget))
                    extra_names = {
                        _memo_text(name, _memo)
                        for name in (*iterated_extra_names, *(name for name, _ in extra_items))
                    }
                declared_fields = _declared_field_names(type(value), budget)
                stored_names = {_memo_text(name, _memo) for name, _ in stored_items}
                if (stored_names | set(declared_fields)).intersection(extra_names):
                    raise ValueError("contract extra storage overlaps stored fields")
            fields = {
                _memo_text(name, _memo): _native_snapshot(
                    field_value,
                    active,
                    _frozen_models=_frozen_models,
                    _frozen_dicts=_frozen_dicts,
                    _frozen_tuples=_frozen_tuples,
                    _memo=_memo,
                )
                for name, field_value in stored_items
            }
            if extra_items:
                fields.update(
                {
                    _memo_text(name, _memo): _native_snapshot(
                        field_value,
                        active,
                        _frozen_models=_frozen_models,
                        _frozen_dicts=_frozen_dicts,
                        _frozen_tuples=_frozen_tuples,
                        _memo=_memo,
                    )
                    for name, field_value in extra_items
                }
            )
            return fields
        if _is_real_mapping(value):
            frozen_dict = _frozen_dicts.get(identity)
            if frozen_dict is not None and frozen_dict[0] is value:
                # Preserve the public traversal callback, but retain the native
                # dict edges captured before that callback could replace them.
                try:
                    public_items = value.items()
                except Exception as exc:
                    return _FailedNativeDictTraversal(exc)
                outer_budget = getattr(_view_state, "budget", None)
                outer_memo = getattr(_view_state, "memo", None)
                _view_state.budget = budget
                _view_state.memo = _memo  # the shape check shares the call's text copies
                try:
                    validated_entries = _validate_public_mapping_entries(public_items)
                finally:
                    _view_state.budget = outer_budget
                    _view_state.memo = outer_memo
                if isinstance(validated_entries, _FailedNativeDictTraversal):
                    return validated_entries
                # Keep the iterator, retained keys, and all entry components
                # in caller scope until authoritative projection is complete.
                source_items = frozen_dict[1]
            else:
                try:
                    source_iterator = iter(value.items())
                except Exception as exc:
                    return _FailedNativeDictTraversal(exc)
                projected: dict[object, object] = {}
                subclass_keys: set[object] = set()  # texts that arrived as a text subclass
                while True:
                    try:
                        pair = next(source_iterator)
                    except StopIteration:
                        break
                    except Exception as exc:
                        return _FailedNativeDictTraversal(exc)
                    budget.charge_edge()  # a view is read only as far as the envelope allows
                    try:
                        key, field_value = pair
                    except Exception as exc:
                        return _FailedNativeDictTraversal(exc)
                    projected_value = _native_snapshot(
                        field_value,
                        active,
                        _frozen_models=_frozen_models,
                        _frozen_dicts=_frozen_dicts,
                        _frozen_tuples=_frozen_tuples,
                        _memo=_memo,
                    )
                    try:
                        # A text-subclass key is reduced to its text; a repeated text is ambiguous input.
                        plain = _hook_free_key(key, _memo, budget)
                        if plain in projected and (type(key) is not str or plain in subclass_keys):
                            raise ValueError("contract input field names must be unique text")
                        if type(key) is not str:
                            subclass_keys.add(plain)
                        projected[plain] = projected_value
                    except Exception as exc:
                        return _FailedNativeDictTraversal(exc)
                return projected
            projected = {}
            subclass_keys = set()
            for key, field_value in source_items:
                projected_value = _native_snapshot(
                    field_value,
                    active,
                    _frozen_models=_frozen_models,
                    _frozen_dicts=_frozen_dicts,
                    _frozen_tuples=_frozen_tuples,
                    _memo=_memo,
                )
                try:
                    plain = _hook_free_key(key, _memo, budget)
                    if plain in projected and (type(key) is not str or plain in subclass_keys):
                        raise ValueError("contract input field names must be unique text")
                    if type(key) is not str:
                        subclass_keys.add(plain)
                    projected[plain] = projected_value
                except Exception as exc:
                    return _FailedNativeDictTraversal(exc)
            del validated_entries
            return projected
        if is_native_instance(value, list):
            return [
                _native_snapshot(
                    item,
                    active,
                    _frozen_models=_frozen_models,
                    _frozen_dicts=_frozen_dicts,
                    _frozen_tuples=_frozen_tuples,
                    _memo=_memo,
                )
                for item in _sequence_items(value, list, _frozen_tuples, budget)
            ]
        if is_native_instance(value, tuple):
            frozen_tuple = _frozen_tuples.get(identity)
            if frozen_tuple is None or frozen_tuple[0] is not value:
                _freeze_model_storage(
                    value,
                    _frozen_models,
                    _frozen_dicts,
                    _frozen_tuples,
                    set(),
                    budget,
                )
                frozen_tuple = _frozen_tuples[identity]
            return tuple(
                _native_snapshot(
                    item,
                    active,
                    _frozen_models=_frozen_models,
                    _frozen_dicts=_frozen_dicts,
                    _frozen_tuples=_frozen_tuples,
                    _memo=_memo,
                )
                for item in frozen_tuple[1]
            )
        members = [
            _native_snapshot(
                item,
                active,
                _frozen_models=_frozen_models,
                _frozen_dicts=_frozen_dicts,
                _frozen_tuples=_frozen_tuples,
                _memo=_memo,
            )
            for item in _sequence_items(
                value, set if is_native_instance(value, set) else frozenset, _frozen_tuples, budget
            )
        ]
        try:
            return set(members) if is_native_instance(value, set) else frozenset(members)
        except TypeError:
            # The snapshot rebuilds a mapping as a dict, which a set cannot hold: refuse it as bad input,
            # not as an unhashable-type leak.
            raise ValueError("contract input set holds an unhashable member") from None
    finally:
        active.remove(identity)


def _validated_event_snapshot(value: object) -> UsageEvent:
    """Return a fresh, deeply validated event or one deterministic error."""

    try:
        return UsageEvent.model_validate(_native_snapshot(value))
    except Exception as exc:
        raise UsageReconciliationError(
            "invalid_event", "an input event failed contract validation"
        ) from exc


def _stream_key(event: UsageEvent) -> tuple[object, ...]:
    return (
        event.tenant_id,
        event.task_id,
        event.session_id,
        event.intent_id,
        event.attempt_id,
        event.ledger,
        event.operation,
    )


def _event_fingerprint(event: UsageEvent) -> object:
    return event.model_dump(mode="json")


def _checked_sum(values: Iterable[int]) -> int:
    total = sum(values)
    if total > _MAX_SAFE_COUNTER:
        raise UsageReconciliationError(
            "counter_overflow", "a reconciled subtotal exceeds SafeCounter"
        )
    return total


def _token_total(usage: TokenUsage | None) -> tuple[int, tuple[str, ...]]:
    if usage is None:
        return 0, ("usage",)

    included = [
        ("input_tokens", usage.input_tokens),
        ("output_tokens", usage.output_tokens),
    ]
    if usage.cache_read_inclusion == "additional_to_input":
        included.append(("cache_read_tokens", usage.cache_read_tokens))
    if usage.cache_write_inclusion == "additional_to_input":
        included.append(("cache_write_tokens", usage.cache_write_tokens))
    if usage.reasoning_inclusion == "additional_to_output":
        included.append(("reasoning_tokens", usage.reasoning_tokens))

    known = _checked_sum(
        quantity.value for _, quantity in included if quantity.value is not None
    )
    missing = tuple(name for name, quantity in included if quantity.value is None)
    return known, missing


def _validate_transition(predecessor: UsageEvent, successor: UsageEvent) -> None:
    if _stream_key(predecessor) != _stream_key(successor):
        raise UsageReconciliationError(
            "cross_stream_correction",
            f"{successor.event_id} does not share its predecessor's stream identity",
        )
    if predecessor.status in {"final", "unavailable"} and successor.status == "pending":
        raise UsageReconciliationError(
            "invalid_transition", "a terminal observation cannot return to pending"
        )
    if predecessor.attempt_outcome != "in_progress":
        if successor.attempt_outcome != predecessor.attempt_outcome:
            raise UsageReconciliationError(
                "invalid_transition", "a terminal attempt outcome cannot change"
            )
    if (
        successor.late_after_cancellation
        and predecessor.attempt_outcome != "cancelled"
    ):
        raise UsageReconciliationError(
            "invalid_transition",
            "late usage requires a predecessor recording cancellation",
        )
    if predecessor.late_after_cancellation and not successor.late_after_cancellation:
        raise UsageReconciliationError(
            "invalid_transition", "late-after-cancellation attribution cannot be removed"
        )
    if (
        predecessor.attempt_outcome == "cancelled"
        and successor.status == "final"
        and not successor.late_after_cancellation
    ):
        raise UsageReconciliationError(
            "invalid_transition", "late usage after cancellation must remain explicit"
        )


def _validate_no_correction_cycles(
    by_id: Mapping[str, UsageEvent],
) -> None:
    """Reject cycles while resolving every event at most once."""

    resolved: set[str] = set()
    for start_id in sorted(by_id):
        if start_id in resolved:
            continue

        path: set[str] = set()
        current_id: str | None = start_id
        while current_id is not None and current_id not in resolved:
            if current_id in path:
                raise UsageReconciliationError(
                    "correction_cycle",
                    f"correction chain containing {current_id} is cyclic",
                )
            path.add(current_id)
            current_id = by_id[current_id].predecessor_event_id
        resolved.update(path)


def reconcile_usage_events(events: Iterable[UsageEvent]) -> UsageReconciliation:
    """Validate and reconcile one tenant's usage events.

    Exact repeats of an event ID are idempotent.  A repeated ID with different
    content, an unknown predecessor, a correction cycle, a branch, or a
    correction across correlation fields is invalid.  A late-usage marker must
    follow an event recording cancellation and cannot originate on a stream
    root.  The function has no I/O and produces the same result regardless of
    input ordering.
    """

    by_id: dict[str, UsageEvent] = {}
    conflicting_ids: set[str] = set()
    for supplied_event in events:
        event = _validated_event_snapshot(supplied_event)
        existing = by_id.get(event.event_id)
        if existing is None:
            by_id[event.event_id] = event
        elif _event_fingerprint(existing) != _event_fingerprint(event):
            conflicting_ids.add(event.event_id)
    if conflicting_ids:
        # Report the smallest conflicting ID so the error does not depend on
        # which conflict the input order happened to reach first.
        conflict_id = min(conflicting_ids)
        raise UsageReconciliationError(
            "conflicting_event_id", f"event ID {conflict_id} has two payloads"
        )

    if not by_id:
        raise UsageReconciliationError(
            "empty_reconciliation",
            "at least one event is required to establish tenant and ledger evidence",
        )

    tenant_ids = {event.tenant_id for event in by_id.values()}
    if len(tenant_ids) > 1:
        raise UsageReconciliationError(
            "mixed_tenants", "one reconciliation cannot combine tenant ledgers"
        )

    ordered_events = tuple(by_id[event_id] for event_id in sorted(by_id))

    if any(
        event.predecessor_event_id is None and event.late_after_cancellation
        for event in ordered_events
    ):
        raise UsageReconciliationError(
            "invalid_transition",
            "late usage requires a predecessor recording cancellation",
        )

    successor_by_id: dict[str, str] = {}
    for event in ordered_events:
        predecessor_id = event.predecessor_event_id
        if predecessor_id is None:
            continue
        predecessor = by_id.get(predecessor_id)
        if predecessor is None:
            raise UsageReconciliationError(
                "unknown_predecessor",
                f"event {event.event_id} names missing event {predecessor_id}",
            )
        existing_successor = successor_by_id.get(predecessor_id)
        if existing_successor is not None and existing_successor != event.event_id:
            raise UsageReconciliationError(
                "branched_correction", f"event {predecessor_id} has multiple corrections"
            )
        successor_by_id[predecessor_id] = event.event_id
        _validate_transition(predecessor, event)

    # Check cycles independently of input order, including multi-event cycles.
    _validate_no_correction_cycles(by_id)

    roots_by_stream: dict[tuple[object, ...], list[UsageEvent]] = {}
    for event in ordered_events:
        if event.predecessor_event_id is None:
            roots_by_stream.setdefault(_stream_key(event), []).append(event)
    for roots in roots_by_stream.values():
        if len(roots) > 1:
            raise UsageReconciliationError(
                "multiple_stream_roots",
                "one attempt/ledger/operation stream has multiple independent roots",
            )

    heads = [
        event for event in ordered_events if event.event_id not in successor_by_id
    ]
    streams: list[ReconciledUsageStream] = []
    for event in heads:
        known, missing = _token_total(event.usage)
        if event.status != "final" and "usage_status" not in missing:
            missing = (*missing, "usage_status")
        streams.append(
            ReconciledUsageStream(
                tenant_id=event.tenant_id,
                task_id=event.task_id,
                session_id=event.session_id,
                intent_id=event.intent_id,
                attempt_id=event.attempt_id,
                ledger=event.ledger,
                operation=event.operation,
                head_event_id=event.event_id,
                status=event.status,
                attempt_outcome=event.attempt_outcome,
                late_after_cancellation=event.late_after_cancellation,
                attribution=event.attribution,
                usage=event.usage,
                usage_missing_reason=event.usage_missing_reason,
                known_token_subtotal=known,
                total_tokens=known if not missing else None,
                incomplete_categories=missing,
            )
        )
    streams.sort(
        key=lambda item: (
            _LEDGER_ORDER.index(item.ledger),
            item.task_id,
            item.attempt_id,
            item.operation,
            item.head_event_id,
        )
    )

    ledgers: list[ReconciledLedger] = []
    for ledger in _LEDGER_ORDER:
        ledger_streams = [stream for stream in streams if stream.ledger == ledger]
        incomplete_ids = tuple(
            stream.attempt_id for stream in ledger_streams if stream.total_tokens is None
        )
        known = _checked_sum(stream.known_token_subtotal for stream in ledger_streams)
        ledgers.append(
            ReconciledLedger(
                ledger=ledger,
                coverage="reported" if ledger_streams else "unreported",
                missing_reason=None if ledger_streams else "no_events",
                known_token_subtotal=known,
                total_tokens=known if ledger_streams and not incomplete_ids else None,
                incomplete_attempt_ids=incomplete_ids,
            )
        )

    tenant_id = next(iter(tenant_ids))
    return UsageReconciliation(
        tenant_id=tenant_id,
        streams=tuple(streams),
        ledgers=tuple(ledgers),
    )
