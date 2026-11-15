"""Content-free usage events for separate operational accounting ledgers.

The models in this module describe observations.  They do not authorize work,
assign a price, or imply that an absent measurement was zero.  Provider,
model, and plaintext-recipient values are snapshots for the recorded attempt;
callers must use an explicit reason whenever one of those values is missing.
"""

from __future__ import annotations

from typing import Literal

from pydantic import model_validator

from contracts_common import ContractModel, OpaqueId, SafeCounter, VersionedContract


UsageLedger = Literal["service", "consuming_agent", "evaluation"]
UsageStatus = Literal["pending", "final", "unavailable"]
AttemptOutcome = Literal["in_progress", "completed", "failed", "cancelled"]
MeasurementKind = Literal["actual", "estimated"]
MissingReason = Literal[
    "not_yet_reported",
    "provider_did_not_report",
    "client_not_integrated",
    "dependency_unavailable",
    "redacted_by_policy",
    "not_applicable",
]
CacheInclusion = Literal["included_in_input", "additional_to_input", "not_applicable"]
ReasoningInclusion = Literal[
    "included_in_output",
    "additional_to_output",
    "not_applicable",
]


class AttributionSnapshot(ContractModel):
    """Attempt-time attribution; it is evidence, not identity or authority."""

    producer: OpaqueId
    bill_owner: OpaqueId | None
    bill_owner_missing_reason: MissingReason | None
    execution_location: OpaqueId
    provider: OpaqueId | None
    provider_missing_reason: MissingReason | None
    model: OpaqueId | None
    model_missing_reason: MissingReason | None
    recipients: tuple[OpaqueId, ...] | None
    recipients_missing_reason: MissingReason | None

    @model_validator(mode="after")
    def require_values_or_reasons(self) -> "AttributionSnapshot":
        pairs = (
            ("bill_owner", self.bill_owner, self.bill_owner_missing_reason),
            ("provider", self.provider, self.provider_missing_reason),
            ("model", self.model, self.model_missing_reason),
            ("recipients", self.recipients, self.recipients_missing_reason),
        )
        for name, value, reason in pairs:
            if value is None and reason is None:
                raise ValueError(f"{name} requires a missing reason")
            if value is not None and reason is not None:
                raise ValueError(f"{name} cannot have both a value and a missing reason")
        if self.recipients is not None and not self.recipients:
            raise ValueError("recipients must be nonempty when present")
        return self


class TokenQuantity(ContractModel):
    """One provider category, retaining unknown and not-applicable distinctly."""

    value: SafeCounter | None
    missing_reason: MissingReason | None

    @model_validator(mode="after")
    def require_value_or_reason(self) -> "TokenQuantity":
        if self.value is None and self.missing_reason is None:
            raise ValueError("a token quantity requires a value or missing reason")
        if self.value is not None and self.missing_reason is not None:
            raise ValueError("a token quantity cannot have both a value and missing reason")
        return self


class TokenUsage(ContractModel):
    """Provider token categories plus the conventions needed to total them.

    Input and output are the base categories.  Cache reads/writes and reasoning
    are added only when their convention says they are additional; otherwise
    their counts are informational subcategories already included in a base.
    """

    measurement: MeasurementKind
    input_tokens: TokenQuantity
    output_tokens: TokenQuantity
    cache_read_tokens: TokenQuantity
    cache_write_tokens: TokenQuantity
    reasoning_tokens: TokenQuantity
    cache_read_inclusion: CacheInclusion
    cache_write_inclusion: CacheInclusion
    reasoning_inclusion: ReasoningInclusion

    @model_validator(mode="after")
    def match_not_applicable_conventions(self) -> "TokenUsage":
        pairs = (
            (self.cache_read_tokens, self.cache_read_inclusion, "cache_read"),
            (self.cache_write_tokens, self.cache_write_inclusion, "cache_write"),
            (self.reasoning_tokens, self.reasoning_inclusion, "reasoning"),
        )
        for quantity, convention, name in pairs:
            quantity_is_na = quantity.missing_reason == "not_applicable"
            convention_is_na = convention == "not_applicable"
            if quantity_is_na != convention_is_na:
                raise ValueError(
                    f"{name} count and inclusion convention must agree on not_applicable"
                )
        return self


class UsageEvent(VersionedContract):
    """A replaceable observation for exactly one attempt and one ledger.

    A correction names the event it replaces.  Reconciliation validates the
    full chain and chooses its leaf, so ingestion order never changes results.
    A cancelled attempt may still receive final provider usage; that event is
    retained with ``late_after_cancellation=True`` rather than discarded.
    """

    event_id: OpaqueId
    tenant_id: OpaqueId
    task_id: OpaqueId
    session_id: OpaqueId | None
    intent_id: OpaqueId | None
    attempt_id: OpaqueId
    ledger: UsageLedger
    operation: OpaqueId
    predecessor_event_id: OpaqueId | None
    status: UsageStatus
    attempt_outcome: AttemptOutcome
    late_after_cancellation: bool
    attribution: AttributionSnapshot
    usage: TokenUsage | None
    usage_missing_reason: MissingReason | None

    @model_validator(mode="after")
    def validate_state(self) -> "UsageEvent":
        if self.predecessor_event_id == self.event_id:
            raise ValueError("an event cannot correct itself")

        if self.status == "unavailable":
            if self.usage is not None or self.usage_missing_reason is None:
                raise ValueError(
                    "unavailable usage requires no usage value and a missing reason"
                )
        else:
            if self.usage_missing_reason is not None:
                raise ValueError(
                    "usage_missing_reason is only valid for unavailable events"
                )
            if self.status == "final" and self.usage is None:
                raise ValueError("final usage requires token usage")

        if self.late_after_cancellation:
            if self.status != "final" or self.attempt_outcome != "cancelled":
                raise ValueError(
                    "late usage must be final usage for a cancelled attempt"
                )
        return self
