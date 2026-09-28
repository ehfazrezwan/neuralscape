"""Adversarial correction and aggregation tests for usage reconciliation."""

from itertools import permutations

import pytest
from pydantic import ValidationError

from contracts_usage import AttributionSnapshot, TokenQuantity, TokenUsage, UsageEvent
from contracts_usage_reconcile import (
    ReconciledLedger,
    UsageReconciliationError,
    reconcile_usage_events,
)


def _quantity(value: int | None, reason: str | None = None) -> TokenQuantity:
    return TokenQuantity(value=value, missing_reason=reason)


def _usage(
    *,
    input_tokens: int | None = 10,
    output_tokens: int | None = 5,
    cache_read_tokens: int | None = 4,
    reasoning_tokens: int | None = 2,
    cache_read_inclusion: str = "included_in_input",
    reasoning_inclusion: str = "included_in_output",
) -> TokenUsage:
    return TokenUsage(
        measurement="actual",
        input_tokens=_quantity(
            input_tokens, None if input_tokens is not None else "provider_did_not_report"
        ),
        output_tokens=_quantity(
            output_tokens, None if output_tokens is not None else "provider_did_not_report"
        ),
        cache_read_tokens=_quantity(
            cache_read_tokens,
            None if cache_read_tokens is not None else "provider_did_not_report",
        ),
        cache_write_tokens=_quantity(None, "not_applicable"),
        reasoning_tokens=_quantity(
            reasoning_tokens,
            None if reasoning_tokens is not None else "provider_did_not_report",
        ),
        cache_read_inclusion=cache_read_inclusion,
        cache_write_inclusion="not_applicable",
        reasoning_inclusion=reasoning_inclusion,
    )


def _attribution(**updates: object) -> AttributionSnapshot:
    values = {
        "producer": "producer-1",
        "bill_owner": "billing-account-1",
        "bill_owner_missing_reason": None,
        "execution_location": "worker-1",
        "provider": "provider-1",
        "provider_missing_reason": None,
        "model": "model-revision-1",
        "model_missing_reason": None,
        "recipients": ("provider-1",),
        "recipients_missing_reason": None,
    }
    values.update(updates)
    return AttributionSnapshot(**values)


def _event(**updates: object) -> UsageEvent:
    values = {
        "schema_version": "candidate-v1",
        "event_id": "event-1",
        "tenant_id": "tenant-1",
        "task_id": "task-1",
        "session_id": "session-1",
        "intent_id": "intent-1",
        "attempt_id": "attempt-1",
        "ledger": "service",
        "operation": "extract",
        "predecessor_event_id": None,
        "status": "final",
        "attempt_outcome": "completed",
        "late_after_cancellation": False,
        "attribution": _attribution(),
        "usage": _usage(),
        "usage_missing_reason": None,
    }
    values.update(updates)
    return UsageEvent(**values)


def _error_code(events: list[UsageEvent]) -> str:
    with pytest.raises(UsageReconciliationError) as caught:
        reconcile_usage_events(events)
    return caught.value.code


def test_reconciles_three_ledgers_without_cross_ledger_relabelling() -> None:
    service = _event()
    agent = _event(
        event_id="event-agent",
        attempt_id="attempt-agent",
        ledger="consuming_agent",
        operation="agent-turn",
        usage=_usage(
            input_tokens=7,
            output_tokens=3,
            cache_read_tokens=2,
            reasoning_tokens=1,
            cache_read_inclusion="additional_to_input",
            reasoning_inclusion="additional_to_output",
        ),
    )
    evaluation = _event(
        event_id="event-eval",
        attempt_id="attempt-eval",
        ledger="evaluation",
        operation="judge",
        status="unavailable",
        attempt_outcome="failed",
        usage=None,
        usage_missing_reason="dependency_unavailable",
    )

    result = reconcile_usage_events([evaluation, agent, service])
    ledgers = {ledger.ledger: ledger for ledger in result.ledgers}

    assert ledgers["service"].total_tokens == 15
    assert ledgers["consuming_agent"].total_tokens == 13
    assert {ledger.coverage for ledger in ledgers.values()} == {"reported"}
    assert ledgers["evaluation"].known_token_subtotal == 0
    assert ledgers["evaluation"].total_tokens is None
    assert ledgers["evaluation"].incomplete_attempt_ids == ("attempt-eval",)


def test_empty_input_cannot_fabricate_tenant_or_ledger_completeness() -> None:
    assert _error_code([]) == "empty_reconciliation"


def test_absent_ledgers_are_unreported_not_complete_zeroes() -> None:
    result = reconcile_usage_events([_event()])
    ledgers = {ledger.ledger: ledger for ledger in result.ledgers}

    assert result.tenant_id == "tenant-1"
    assert ledgers["service"].coverage == "reported"
    assert ledgers["service"].missing_reason is None
    assert ledgers["service"].total_tokens == 15
    for ledger_name in ("consuming_agent", "evaluation"):
        ledger = ledgers[ledger_name]
        assert ledger.coverage == "unreported"
        assert ledger.missing_reason == "no_events"
        assert ledger.known_token_subtotal == 0
        assert ledger.total_tokens is None
        assert ledger.incomplete_attempt_ids == ()


def test_explicit_measured_zero_is_a_complete_reported_zero() -> None:
    zero = _event(
        usage=_usage(
            input_tokens=0,
            output_tokens=0,
            cache_read_tokens=0,
            reasoning_tokens=0,
        )
    )
    result = reconcile_usage_events([zero])
    service = result.ledgers[0]

    assert service.coverage == "reported"
    assert service.missing_reason is None
    assert service.known_token_subtotal == 0
    assert service.total_tokens == 0


@pytest.mark.parametrize(
    "changes",
    [
        {"missing_reason": None},
        {"known_token_subtotal": 1},
        {"total_tokens": 0},
        {"incomplete_attempt_ids": ("attempt-1",)},
    ],
)
def test_unreported_ledger_cannot_claim_observed_or_complete_usage(
    changes: dict[str, object],
) -> None:
    values: dict[str, object] = {
        "ledger": "consuming_agent",
        "coverage": "unreported",
        "missing_reason": "no_events",
        "known_token_subtotal": 0,
        "total_tokens": None,
        "incomplete_attempt_ids": (),
    }
    values.update(changes)

    with pytest.raises(ValidationError):
        ReconciledLedger(**values)


def test_correction_is_order_independent_and_exact_duplicate_is_idempotent() -> None:
    pending = _event(
        event_id="pending",
        status="pending",
        attempt_outcome="in_progress",
        usage=None,
    )
    final = _event(event_id="final", predecessor_event_id="pending")

    forward = reconcile_usage_events([pending, final, final])
    reverse = reconcile_usage_events([final, pending])

    assert forward == reverse
    assert len(forward.streams) == 1
    assert forward.streams[0].head_event_id == "final"
    assert forward.streams[0].total_tokens == 15


def test_three_event_correction_chain_is_permutation_invariant() -> None:
    root = _event(
        event_id="root",
        status="pending",
        attempt_outcome="in_progress",
        usage=None,
    )
    update = _event(
        event_id="update",
        predecessor_event_id="root",
        status="pending",
        attempt_outcome="in_progress",
        usage=_usage(input_tokens=8),
    )
    final = _event(event_id="final", predecessor_event_id="update")

    results = [
        reconcile_usage_events(order)
        for order in permutations((root, update, final))
    ]

    assert all(result == results[0] for result in results)
    assert results[0].streams[0].head_event_id == "final"


def test_same_event_id_with_different_payload_is_a_conflict() -> None:
    first = _event()
    conflicting = _event(attempt_id="attempt-elsewhere")
    assert _error_code([first, conflicting]) == "conflicting_event_id"


def test_unknown_predecessor_is_invalid() -> None:
    correction = _event(event_id="correction", predecessor_event_id="not-present")
    assert _error_code([correction]) == "unknown_predecessor"


def test_correction_cycle_is_invalid() -> None:
    first = _event(event_id="a", predecessor_event_id="b")
    second = _event(event_id="b", predecessor_event_id="a")
    assert _error_code([first, second]) == "correction_cycle"


def test_branched_correction_is_invalid() -> None:
    root = _event(event_id="root", status="pending", attempt_outcome="in_progress", usage=None)
    first = _event(event_id="first", predecessor_event_id="root")
    second = _event(event_id="second", predecessor_event_id="root")
    assert _error_code([root, first, second]) == "branched_correction"


def test_mixed_tenants_cannot_be_aggregated() -> None:
    first = _event()
    second = _event(event_id="other", tenant_id="tenant-2", attempt_id="attempt-2")
    assert _error_code([first, second]) == "mixed_tenants"


@pytest.mark.parametrize("changed", ["tenant_id", "task_id", "attempt_id", "ledger"])
def test_correction_cannot_move_between_tenants_tasks_attempts_or_ledgers(
    changed: str,
) -> None:
    root = _event(event_id="root", status="pending", attempt_outcome="in_progress", usage=None)
    updates: dict[str, object] = {
        "event_id": "correction",
        "predecessor_event_id": "root",
    }
    updates[changed] = {
        "tenant_id": "tenant-2",
        "task_id": "task-2",
        "attempt_id": "attempt-2",
        "ledger": "consuming_agent",
    }[changed]
    correction = _event(**updates)

    expected = "mixed_tenants" if changed == "tenant_id" else "cross_stream_correction"
    assert _error_code([root, correction]) == expected


def test_distinct_attempts_remain_distinct_even_with_same_attribution() -> None:
    first = _event(event_id="one", attempt_id="attempt-1")
    second = _event(event_id="two", attempt_id="attempt-2")
    result = reconcile_usage_events([second, first])

    assert [stream.attempt_id for stream in result.streams] == [
        "attempt-1",
        "attempt-2",
    ]
    assert result.ledgers[0].total_tokens == 30


def test_unknown_additional_category_makes_total_unknown_not_zero() -> None:
    event = _event(
        usage=_usage(
            cache_read_tokens=None,
            cache_read_inclusion="additional_to_input",
        )
    )
    result = reconcile_usage_events([event])
    stream = result.streams[0]

    assert stream.known_token_subtotal == 15
    assert stream.total_tokens is None
    assert stream.incomplete_categories == ("cache_read_tokens",)
    assert result.ledgers[0].known_token_subtotal == 15
    assert result.ledgers[0].total_tokens is None


def test_unknown_included_subcategory_does_not_double_count_or_hide_base_total() -> None:
    event = _event(usage=_usage(cache_read_tokens=None))
    result = reconcile_usage_events([event])

    assert result.streams[0].total_tokens == 15
    assert result.streams[0].incomplete_categories == ()


def test_late_usage_after_cancellation_replaces_unavailable_observation() -> None:
    unavailable = _event(
        event_id="cancelled",
        status="unavailable",
        attempt_outcome="cancelled",
        usage=None,
        usage_missing_reason="not_yet_reported",
    )
    late = _event(
        event_id="late",
        predecessor_event_id="cancelled",
        attempt_outcome="cancelled",
        late_after_cancellation=True,
    )
    result = reconcile_usage_events([late, unavailable])

    assert result.streams[0].head_event_id == "late"
    assert result.streams[0].late_after_cancellation is True
    assert result.streams[0].total_tokens == 15


def test_late_usage_after_cancellation_must_remain_explicit() -> None:
    unavailable = _event(
        event_id="cancelled",
        status="unavailable",
        attempt_outcome="cancelled",
        usage=None,
        usage_missing_reason="not_yet_reported",
    )
    incorrectly_final = _event(
        event_id="late",
        predecessor_event_id="cancelled",
        attempt_outcome="cancelled",
    )
    assert _error_code([unavailable, incorrectly_final]) == "invalid_transition"
