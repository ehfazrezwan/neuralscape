"""Field and state semantics for the candidate usage contract."""

import pytest
from pydantic import ValidationError

from contracts_usage import AttributionSnapshot, TokenQuantity, TokenUsage, UsageEvent


def _known(value: int) -> TokenQuantity:
    return TokenQuantity(value=value, missing_reason=None)


def _missing(reason: str = "provider_did_not_report") -> TokenQuantity:
    return TokenQuantity(value=None, missing_reason=reason)


def _attribution(**updates: object) -> AttributionSnapshot:
    values = {
        "producer": "extractor-v3",
        "bill_owner": "account-7",
        "bill_owner_missing_reason": None,
        "execution_location": "customer-worker",
        "provider": "provider-a",
        "provider_missing_reason": None,
        "model": "model-a-2026-09",
        "model_missing_reason": None,
        "recipients": ("provider-a",),
        "recipients_missing_reason": None,
    }
    values.update(updates)
    return AttributionSnapshot(**values)


def _usage() -> TokenUsage:
    return TokenUsage(
        measurement="actual",
        input_tokens=_known(100),
        output_tokens=_known(30),
        cache_read_tokens=_known(40),
        cache_write_tokens=_missing("not_applicable"),
        reasoning_tokens=_known(10),
        cache_read_inclusion="included_in_input",
        cache_write_inclusion="not_applicable",
        reasoning_inclusion="included_in_output",
    )


def _event(**updates: object) -> UsageEvent:
    values = {
        "schema_version": "candidate-v1",
        "event_id": "event-1",
        "tenant_id": "tenant-a",
        "task_id": "task-1",
        "session_id": "session-1",
        "intent_id": "intent-1",
        "attempt_id": "attempt-1",
        "ledger": "service",
        "operation": "memory.extract",
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


def test_usage_event_requires_explicit_version_and_rejects_unknown_fields() -> None:
    payload = _event().model_dump()
    payload.pop("schema_version")
    with pytest.raises(ValidationError):
        UsageEvent(**payload)

    payload["schema_version"] = "candidate-v1"
    payload["authority"] = "admin"
    with pytest.raises(ValidationError):
        UsageEvent(**payload)


@pytest.mark.parametrize(
    ("field", "reason_field"),
    [
        ("bill_owner", "bill_owner_missing_reason"),
        ("provider", "provider_missing_reason"),
        ("model", "model_missing_reason"),
        ("recipients", "recipients_missing_reason"),
    ],
)
def test_attribution_requires_exactly_one_value_or_missing_reason(
    field: str, reason_field: str
) -> None:
    with pytest.raises(ValidationError, match="requires a missing reason"):
        _attribution(**{field: None, reason_field: None})

    with pytest.raises(ValidationError, match="cannot have both"):
        _attribution(**{reason_field: "redacted_by_policy"})

    snapshot = _attribution(**{field: None, reason_field: "redacted_by_policy"})
    assert getattr(snapshot, field) is None
    assert getattr(snapshot, reason_field) == "redacted_by_policy"


def test_token_quantity_never_collapses_unknown_to_zero() -> None:
    unknown = _missing()
    zero = _known(0)

    assert unknown.value is None
    assert zero.value == 0
    assert unknown != zero

    with pytest.raises(ValidationError):
        TokenQuantity(value=None, missing_reason=None)
    with pytest.raises(ValidationError):
        TokenQuantity(value=0, missing_reason="provider_did_not_report")


def test_cache_and_reasoning_inclusion_conventions_are_explicit() -> None:
    usage = _usage()
    assert usage.cache_read_inclusion == "included_in_input"
    assert usage.reasoning_inclusion == "included_in_output"

    payload = usage.model_dump()
    payload["cache_write_inclusion"] = "additional_to_input"
    with pytest.raises(ValidationError, match="must agree on not_applicable"):
        TokenUsage(**payload)

    payload = usage.model_dump()
    payload["reasoning_tokens"] = _missing("not_applicable")
    with pytest.raises(ValidationError, match="must agree on not_applicable"):
        TokenUsage(**payload)


def test_pending_final_and_unavailable_have_distinct_shapes() -> None:
    assert _event(status="pending", attempt_outcome="in_progress", usage=None).usage is None
    unavailable = _event(
        status="unavailable",
        attempt_outcome="failed",
        usage=None,
        usage_missing_reason="provider_did_not_report",
    )
    assert unavailable.usage_missing_reason == "provider_did_not_report"

    with pytest.raises(ValidationError, match="final usage requires"):
        _event(usage=None)
    with pytest.raises(ValidationError, match="unavailable usage requires"):
        _event(status="unavailable", usage=None)


def test_late_usage_is_only_final_usage_for_a_cancelled_attempt() -> None:
    late = _event(attempt_outcome="cancelled", late_after_cancellation=True)
    assert late.status == "final"

    with pytest.raises(ValidationError, match="late usage must be final"):
        _event(late_after_cancellation=True)
    with pytest.raises(ValidationError, match="late usage must be final"):
        _event(
            status="pending",
            attempt_outcome="cancelled",
            late_after_cancellation=True,
            usage=None,
        )


def test_ledgers_are_closed_and_cannot_be_relabelled() -> None:
    assert _event(ledger="service").ledger == "service"
    assert _event(ledger="consuming_agent").ledger == "consuming_agent"
    assert _event(ledger="evaluation").ledger == "evaluation"
    with pytest.raises(ValidationError):
        _event(ledger="customer")
