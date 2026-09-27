"""Pure, order-independent reconciliation for :mod:`contracts_usage` events."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

from contracts_common import ContractModel, OpaqueId, SafeCounter
from contracts_usage import (
    AttributionSnapshot,
    AttemptOutcome,
    TokenUsage,
    UsageEvent,
    UsageLedger,
    UsageStatus,
)


_MAX_SAFE_COUNTER = 9_007_199_254_740_991
_LEDGER_ORDER: tuple[UsageLedger, ...] = (
    "service",
    "consuming_agent",
    "evaluation",
)

ReconciliationErrorCode = Literal[
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


class UsageReconciliationError(ValueError):
    """A deterministic validation failure in a supplied event set."""

    def __init__(self, code: ReconciliationErrorCode, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


class ReconciledUsageStream(ContractModel):
    """The selected head of one attempt/ledger correction chain."""

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
    known_token_subtotal: SafeCounter
    total_tokens: SafeCounter | None
    incomplete_categories: tuple[str, ...]


class ReconciledLedger(ContractModel):
    """One ledger total; ``total_tokens`` is absent if any stream is incomplete."""

    ledger: UsageLedger
    known_token_subtotal: SafeCounter
    total_tokens: SafeCounter | None
    incomplete_attempt_ids: tuple[OpaqueId, ...]


class UsageReconciliation(ContractModel):
    """Reconciled streams and three totals for exactly one tenant."""

    tenant_id: OpaqueId | None
    streams: tuple[ReconciledUsageStream, ...]
    ledgers: tuple[ReconciledLedger, ...]


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
    if predecessor.late_after_cancellation and not successor.late_after_cancellation:
        raise UsageReconciliationError(
            "invalid_transition", "late-after-cancellation attribution cannot be removed"
        )
    if (
        predecessor.status == "unavailable"
        and predecessor.attempt_outcome == "cancelled"
        and successor.status == "final"
        and not successor.late_after_cancellation
    ):
        raise UsageReconciliationError(
            "invalid_transition", "late usage after cancellation must remain explicit"
        )


def reconcile_usage_events(events: Iterable[UsageEvent]) -> UsageReconciliation:
    """Validate and reconcile one tenant's usage events.

    Exact repeats of an event ID are idempotent.  A repeated ID with different
    content, an unknown predecessor, a correction cycle, a branch, or a
    correction across correlation fields is invalid.  The function has no I/O
    and produces the same result regardless of input ordering.
    """

    by_id: dict[str, UsageEvent] = {}
    for event in events:
        existing = by_id.get(event.event_id)
        if existing is None:
            by_id[event.event_id] = event
        elif _event_fingerprint(existing) != _event_fingerprint(event):
            raise UsageReconciliationError(
                "conflicting_event_id", f"event ID {event.event_id} has two payloads"
            )

    tenant_ids = {event.tenant_id for event in by_id.values()}
    if len(tenant_ids) > 1:
        raise UsageReconciliationError(
            "mixed_tenants", "one reconciliation cannot combine tenant ledgers"
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
    for start_id in by_id:
        seen: set[str] = set()
        current_id: str | None = start_id
        while current_id is not None:
            if current_id in seen:
                raise UsageReconciliationError(
                    "correction_cycle", f"correction chain containing {current_id} cycles"
                )
            seen.add(current_id)
            current_id = by_id[current_id].predecessor_event_id

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
                known_token_subtotal=known,
                total_tokens=known if not incomplete_ids else None,
                incomplete_attempt_ids=incomplete_ids,
            )
        )

    tenant_id = next(iter(tenant_ids)) if tenant_ids else None
    return UsageReconciliation(
        tenant_id=tenant_id,
        streams=tuple(streams),
        ledgers=tuple(ledgers),
    )
