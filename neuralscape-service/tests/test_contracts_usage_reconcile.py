"""Adversarial correction and aggregation tests for usage reconciliation."""

import copy
import json
from itertools import permutations

import pytest
from pydantic import ValidationError

import contracts_usage_reconcile as usage_reconcile_contracts
from contracts_usage import AttributionSnapshot, TokenQuantity, TokenUsage, UsageEvent
from contracts_usage_reconcile import (
    ReconciledLedger,
    ReconciledUsageStream,
    UsageReconciliation,
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


def _assert_result_payload_rejected(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        UsageReconciliation.model_validate(payload)
    with pytest.raises(ValidationError):
        UsageReconciliation.model_validate_json(json.dumps(payload))


def _assert_attribution_authority_rejected(value: object) -> None:
    with pytest.raises(ValidationError) as caught:
        ReconciledUsageStream.model_validate(value)

    assert [
        (error["type"], error["loc"])
        for error in caught.value.errors(include_url=False)
    ] == [("extra_forbidden", ("attribution", "authority"))]


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


def test_result_rejects_top_tenant_mismatching_stream() -> None:
    payload = copy.deepcopy(
        reconcile_usage_events([_event()]).model_dump(mode="python")
    )
    payload["tenant_id"] = "tenant-other"

    _assert_result_payload_rejected(payload)


def test_result_requires_exactly_one_canonical_summary_per_ledger() -> None:
    payload = copy.deepcopy(
        reconcile_usage_events([_event()]).model_dump(mode="python")
    )
    service = payload["ledgers"][0]
    payload["ledgers"] = (service, copy.deepcopy(service), copy.deepcopy(service))

    _assert_result_payload_rejected(payload)


def test_result_rejects_ledger_total_mismatching_streams() -> None:
    payload = copy.deepcopy(
        reconcile_usage_events([_event()]).model_dump(mode="python")
    )
    payload["ledgers"][0]["known_token_subtotal"] = 999
    payload["ledgers"][0]["total_tokens"] = 999

    _assert_result_payload_rejected(payload)


def test_result_rejects_reported_ledger_without_streams() -> None:
    payload = copy.deepcopy(
        reconcile_usage_events([_event()]).model_dump(mode="python")
    )
    payload["streams"] = ()

    _assert_result_payload_rejected(payload)


def test_result_rejects_unavailable_stream_retaining_usage_and_total() -> None:
    payload = copy.deepcopy(
        reconcile_usage_events([_event()]).model_dump(mode="python")
    )
    payload["streams"][0]["status"] = "unavailable"

    _assert_result_payload_rejected(payload)


def test_valid_result_round_trips_through_public_json_contract() -> None:
    result = reconcile_usage_events([_event()])

    assert UsageReconciliation.model_validate_json(result.model_dump_json()) == result


def test_result_revalidation_rejects_hidden_extra_in_nested_copy() -> None:
    result = reconcile_usage_events([_event()])
    invalid_stream = result.streams[0].model_copy(update={"authority": True})
    invalid_result = result.model_copy(update={"streams": (invalid_stream,)})

    with pytest.raises(ValidationError):
        UsageReconciliation.model_validate(invalid_result)


def test_stream_rejects_existing_attribution_with_hidden_extra() -> None:
    stream = reconcile_usage_events([_event()]).streams[0]
    invalid_attribution = stream.attribution.model_copy(
        update={"authority": "admin"}
    )
    payload = stream.model_dump(mode="python")
    payload["attribution"] = invalid_attribution

    _assert_attribution_authority_rejected(payload)


def test_existing_stream_revalidation_rejects_attribution_hidden_extra() -> None:
    stream = reconcile_usage_events([_event()]).streams[0]
    invalid_attribution = stream.attribution.model_copy(
        update={"authority": "admin"}
    )
    invalid_stream = stream.model_copy(
        update={"attribution": invalid_attribution}
    )

    _assert_attribution_authority_rejected(invalid_stream)


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


def test_reconciliation_rejects_unchecked_negative_nested_counter() -> None:
    event = _event()
    assert event.usage is not None
    negative_input = event.usage.input_tokens.model_copy(update={"value": -1})
    invalid_usage = event.usage.model_copy(update={"input_tokens": negative_input})
    invalid_event = event.model_copy(update={"usage": invalid_usage})

    assert _error_code([invalid_event]) == "invalid_event"


def test_reconciliation_rejects_unchecked_contradictory_event_state() -> None:
    invalid_event = _event().model_copy(
        update={
            "status": "unavailable",
            "usage_missing_reason": "provider_did_not_report",
        }
    )

    assert invalid_event.usage is not None
    assert _error_code([invalid_event]) == "invalid_event"


def test_reconciliation_rejects_direct_nested_mutation() -> None:
    invalid_event = _event()
    assert invalid_event.usage is not None
    invalid_event.usage.input_tokens.value = -1

    assert _error_code([invalid_event]) == "invalid_event"


def test_reconciliation_rejects_hidden_extra_from_unchecked_copy() -> None:
    event = _event()
    assert event.usage is not None
    invalid_input = event.usage.input_tokens.model_copy(update={"authority": True})
    invalid_usage = event.usage.model_copy(update={"input_tokens": invalid_input})
    invalid_event = event.model_copy(update={"usage": invalid_usage})

    assert _error_code([invalid_event]) == "invalid_event"


def test_reconciliation_rejects_cyclic_mutated_input_graph() -> None:
    invalid_event = _event()
    invalid_event.usage = invalid_event  # type: ignore[assignment]

    assert _error_code([invalid_event]) == "invalid_event"


def test_reconciliation_result_is_isolated_from_later_input_mutation() -> None:
    event = _event()
    result = reconcile_usage_events([event])
    assert event.usage is not None
    assert result.streams[0].usage is not None

    event.usage.input_tokens.value = 0

    assert result.streams[0].usage.input_tokens.value == 10
    assert result.streams[0].total_tokens == 15


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


def test_cycle_validation_visits_each_event_once_for_long_valid_chain() -> None:
    class CountingEventMap(dict[str, UsageEvent]):
        reads = 0

        def __getitem__(self, event_id: str) -> UsageEvent:
            self.reads += 1
            return super().__getitem__(event_id)

    chain_length = 128
    events = CountingEventMap()
    predecessor_id = None
    for index in range(chain_length):
        event_id = f"event-{index}"
        events[event_id] = _event(
            event_id=event_id,
            predecessor_event_id=predecessor_id,
            status="pending",
            attempt_outcome="in_progress",
            usage=None,
        )
        predecessor_id = event_id

    usage_reconcile_contracts._validate_no_correction_cycles(events)

    assert events.reads == chain_length


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


def test_final_usage_after_pending_cancellation_must_be_explicit() -> None:
    cancelled = _event(
        event_id="cancelled",
        status="pending",
        attempt_outcome="cancelled",
        usage=None,
    )
    incorrectly_final = _event(
        event_id="late",
        predecessor_event_id="cancelled",
        attempt_outcome="cancelled",
    )

    assert _error_code([incorrectly_final, cancelled]) == "invalid_transition"


def test_explicit_final_usage_after_pending_cancellation_is_retained() -> None:
    cancelled = _event(
        event_id="cancelled",
        status="pending",
        attempt_outcome="cancelled",
        usage=None,
    )
    late = _event(
        event_id="late",
        predecessor_event_id="cancelled",
        attempt_outcome="cancelled",
        late_after_cancellation=True,
    )

    result = reconcile_usage_events([late, cancelled])

    assert result.streams[0].head_event_id == "late"
    assert result.streams[0].late_after_cancellation is True


def test_final_cancellation_correction_requires_explicit_late_marker() -> None:
    cancelled = _event(
        event_id="cancelled",
        attempt_outcome="cancelled",
    )
    nonexplicit_correction = _event(
        event_id="nonexplicit",
        predecessor_event_id="cancelled",
        attempt_outcome="cancelled",
    )
    explicit_correction = _event(
        event_id="explicit",
        predecessor_event_id="cancelled",
        attempt_outcome="cancelled",
        late_after_cancellation=True,
    )

    assert _error_code([cancelled, nonexplicit_correction]) == "invalid_transition"
    result = reconcile_usage_events([cancelled, explicit_correction])
    assert result.streams[0].head_event_id == "explicit"
    assert result.streams[0].late_after_cancellation is True


def test_independent_final_cancelled_root_does_not_infer_late_usage() -> None:
    independent = _event(
        attempt_outcome="cancelled",
        late_after_cancellation=False,
    )

    result = reconcile_usage_events([independent])

    assert result.streams[0].late_after_cancellation is False
