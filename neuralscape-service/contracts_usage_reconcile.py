"""Pure, order-independent reconciliation for :mod:`contracts_usage` events."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, GetCoreSchemaHandler, model_validator
from pydantic_core import core_schema

from contracts_common import ContractModel, OpaqueId, SafeCounter
from contracts_usage import (
    AttributionSnapshot,
    AttemptOutcome,
    MissingReason,
    TokenUsage,
    UsageEvent,
    UsageLedger,
    UsageStatus,
)


_MAX_SAFE_COUNTER = 9_007_199_254_740_991
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
        try:
            self.attribution = AttributionSnapshot.model_validate(
                _native_snapshot(self.attribution)
            )
        except Exception as exc:
            raise ValueError("stream attribution is invalid") from exc
        if self.usage is not None:
            try:
                self.usage = TokenUsage.model_validate(_native_snapshot(self.usage))
            except Exception as exc:
                raise ValueError("stream usage is invalid") from exc
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
        try:
            self.streams = tuple(
                ReconciledUsageStream.model_validate(_native_snapshot(stream))
                for stream in self.streams
            )
            self.ledgers = tuple(
                ReconciledLedger.model_validate(_native_snapshot(ledger))
                for ledger in self.ledgers
            )
        except Exception as exc:
            raise ValueError(
                "reconciliation result contains an invalid nested value"
            ) from exc

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


def _freeze_model_storage(
    value: object,
    frozen_models: dict[int, _FrozenModelStorage],
    frozen_dicts: dict[int, _FrozenDictStorage],
    frozen_tuples: dict[int, _FrozenTupleStorage],
    visited: set[int],
) -> None:
    """Capture reachable native model stores without public traversal hooks."""

    value_type = type(value)
    is_native_container = issubclass(
        value_type,
        (BaseModel, dict, list, tuple, set, frozenset),
    )
    if not is_native_container:
        return

    identity = id(value)
    if identity in visited:
        return
    visited.add(identity)

    if issubclass(value_type, BaseModel):
        frozen_model = frozen_models.get(identity)
        if frozen_model is not None and frozen_model[0] is value:
            return
        stored = _MODEL_DICT_DESCRIPTOR.__get__(value, BaseModel)
        stored_items = tuple(dict.items(stored))
        try:
            extras = _MODEL_EXTRAS_DESCRIPTOR.__get__(value, BaseModel)
        except AttributeError:
            extras = None
        native_extra_items = (
            tuple(dict.items(extras))
            if extras is not None and issubclass(type(extras), dict)
            else None
        )
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
            )
        if native_extra_items is not None:
            for _name, field_value in native_extra_items:
                _freeze_model_storage(
                    field_value,
                    frozen_models,
                    frozen_dicts,
                    frozen_tuples,
                    visited,
                )
        return

    if issubclass(value_type, dict):
        frozen_dict = frozen_dicts.get(identity)
        if frozen_dict is not None and frozen_dict[0] is value:
            return
        native_items = tuple(dict.items(value))
        frozen_dicts[identity] = (value, native_items)
        items = (item for _key, item in native_items)
    elif issubclass(value_type, list):
        items = list.__iter__(value)
    elif issubclass(value_type, tuple):
        frozen_tuple = frozen_tuples.get(identity)
        if frozen_tuple is not None and frozen_tuple[0] is value:
            return
        native_items = tuple(tuple.__iter__(value))
        frozen_tuples[identity] = (value, native_items)
        items = iter(native_items)
    elif issubclass(value_type, set):
        items = set.__iter__(value)
    else:
        items = frozenset.__iter__(value)
    for item in items:
        _freeze_model_storage(
            item,
            frozen_models,
            frozen_dicts,
            frozen_tuples,
            visited,
        )


def _native_snapshot(
    value: object,
    active: set[int] | None = None,
    *,
    _frozen_models: dict[int, _FrozenModelStorage] | None = None,
    _frozen_dicts: dict[int, _FrozenDictStorage] | None = None,
    _frozen_tuples: dict[int, _FrozenTupleStorage] | None = None,
) -> object:
    """Copy a nested native/model graph without trusting model construction.

    Reading ``__dict__`` deliberately retains unknown fields injected by
    unchecked model copies so the destination contract can reject them.
    """

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
        )
    if active is None:
        active = set()
    if not isinstance(
        value,
        (BaseModel, Mapping, list, tuple, set, frozenset),
    ):
        return value

    identity = id(value)
    if identity in active:
        raise ValueError("cyclic input graph")
    active.add(identity)
    try:
        if isinstance(value, BaseModel):
            frozen = _frozen_models.get(identity)
            if frozen is None or frozen[0] is not value:
                _freeze_model_storage(
                    value,
                    _frozen_models,
                    _frozen_dicts,
                    _frozen_tuples,
                    set(),
                )
                frozen = _frozen_models[identity]
            _model, stored_items, extras, native_extra_items = frozen
            extra_items: tuple[tuple[object, object], ...] = ()
            if extras is not None:
                if native_extra_items is not None:
                    extra_items = native_extra_items
                    extra_names = {name for name, _ in extra_items}
                else:
                    if not isinstance(extras, Mapping):
                        raise ValueError("contract extra storage must be a mapping")
                    iterated_extra_names = tuple(extras)
                    extra_items = tuple(extras.items())
                    extra_names = set(iterated_extra_names) | {
                        name for name, _ in extra_items
                    }
                declared_fields = type(value).model_fields
                stored_names = {name for name, _ in stored_items}
                if (stored_names | set(declared_fields)).intersection(extra_names):
                    raise ValueError("contract extra storage overlaps stored fields")
            fields = {
                name: _native_snapshot(
                    field_value,
                    active,
                    _frozen_models=_frozen_models,
                    _frozen_dicts=_frozen_dicts,
                    _frozen_tuples=_frozen_tuples,
                )
                for name, field_value in stored_items
            }
            if extra_items:
                fields.update(
                {
                    name: _native_snapshot(
                        field_value,
                        active,
                        _frozen_models=_frozen_models,
                        _frozen_dicts=_frozen_dicts,
                        _frozen_tuples=_frozen_tuples,
                    )
                    for name, field_value in extra_items
                }
            )
            return fields
        if isinstance(value, Mapping):
            frozen_dict = _frozen_dicts.get(identity)
            if frozen_dict is not None and frozen_dict[0] is value:
                # Preserve the public traversal callback, but retain the native
                # dict edges captured before that callback could replace them.
                try:
                    source_iterator = iter(value.items())
                except Exception as exc:
                    return _FailedNativeDictTraversal(exc)
                public_entries: dict[object, None] = {}
                while True:
                    try:
                        pair = next(source_iterator)
                    except StopIteration:
                        break
                    except Exception as exc:
                        return _FailedNativeDictTraversal(exc)
                    try:
                        key, _public_value = pair
                    except Exception as exc:
                        return _FailedNativeDictTraversal(exc)
                    try:
                        public_entries[key] = None
                    except Exception as exc:
                        return _FailedNativeDictTraversal(exc)
                source_items = frozen_dict[1]
            else:
                try:
                    source_iterator = iter(value.items())
                except Exception as exc:
                    return _FailedNativeDictTraversal(exc)
                projected: dict[object, object] = {}
                while True:
                    try:
                        pair = next(source_iterator)
                    except StopIteration:
                        break
                    except Exception as exc:
                        return _FailedNativeDictTraversal(exc)
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
                    )
                    try:
                        projected[key] = projected_value
                    except Exception as exc:
                        return _FailedNativeDictTraversal(exc)
                return projected
            projected = {}
            for key, field_value in source_items:
                projected_value = _native_snapshot(
                    field_value,
                    active,
                    _frozen_models=_frozen_models,
                    _frozen_dicts=_frozen_dicts,
                    _frozen_tuples=_frozen_tuples,
                )
                try:
                    projected[key] = projected_value
                except Exception as exc:
                    return _FailedNativeDictTraversal(exc)
            return projected
        if isinstance(value, list):
            return [
                _native_snapshot(
                    item,
                    active,
                    _frozen_models=_frozen_models,
                    _frozen_dicts=_frozen_dicts,
                    _frozen_tuples=_frozen_tuples,
                )
                for item in value
            ]
        if isinstance(value, tuple):
            frozen_tuple = _frozen_tuples.get(identity)
            if frozen_tuple is None or frozen_tuple[0] is not value:
                _freeze_model_storage(
                    value,
                    _frozen_models,
                    _frozen_dicts,
                    _frozen_tuples,
                    set(),
                )
                frozen_tuple = _frozen_tuples[identity]
            return tuple(
                _native_snapshot(
                    item,
                    active,
                    _frozen_models=_frozen_models,
                    _frozen_dicts=_frozen_dicts,
                    _frozen_tuples=_frozen_tuples,
                )
                for item in frozen_tuple[1]
            )
        if isinstance(value, set):
            return {
                _native_snapshot(
                    item,
                    active,
                    _frozen_models=_frozen_models,
                    _frozen_dicts=_frozen_dicts,
                    _frozen_tuples=_frozen_tuples,
                )
                for item in value
            }
        return frozenset(
            _native_snapshot(
                item,
                active,
                _frozen_models=_frozen_models,
                _frozen_dicts=_frozen_dicts,
                _frozen_tuples=_frozen_tuples,
            )
            for item in value
        )
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
    for start_id in by_id:
        if start_id in resolved:
            continue

        path: set[str] = set()
        current_id: str | None = start_id
        while current_id is not None and current_id not in resolved:
            if current_id in path:
                raise UsageReconciliationError(
                    "correction_cycle",
                    f"correction chain containing {current_id} cycles",
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
    for supplied_event in events:
        event = _validated_event_snapshot(supplied_event)
        existing = by_id.get(event.event_id)
        if existing is None:
            by_id[event.event_id] = event
        elif _event_fingerprint(existing) != _event_fingerprint(event):
            raise UsageReconciliationError(
                "conflicting_event_id", f"event ID {event.event_id} has two payloads"
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

    if any(
        event.predecessor_event_id is None and event.late_after_cancellation
        for event in by_id.values()
    ):
        raise UsageReconciliationError(
            "invalid_transition",
            "late usage requires a predecessor recording cancellation",
        )

    successor_by_id: dict[str, str] = {}
    for event in by_id.values():
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
    for event in by_id.values():
        if event.predecessor_event_id is None:
            roots_by_stream.setdefault(_stream_key(event), []).append(event)
    for roots in roots_by_stream.values():
        if len(roots) > 1:
            raise UsageReconciliationError(
                "multiple_stream_roots",
                "one attempt/ledger/operation stream has multiple independent roots",
            )

    heads = [event for event in by_id.values() if event.event_id not in successor_by_id]
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
