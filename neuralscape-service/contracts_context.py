"""Inert task-aware context delivery and privacy-aware receipt contracts.

These types describe messages only.  They neither authenticate a principal nor
prove authorization, source existence, transactionality, or token counting.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import AwareDatetime, Field, model_validator

from contracts_common import ContractModel, OpaqueId, SafeCounter, VersionedContract
from contracts_freshness import (
    FreshnessWitness,
    ProjectionStatus,
    UpstreamVerificationStatus,
    projection_status_for,
)
from contracts_references import ReferenceHandle, SourceVersion


class TimePerspective(str, Enum):
    CURRENT = "current"
    HISTORICAL = "historical"


class FreshnessRequirement(str, Enum):
    """Requested content freshness; policy checks always remain current."""

    CURRENT_VERIFIED = "current_verified"
    CURRENT_KNOWN = "current_known"
    LABELLED_STALE_ACCEPTABLE = "labelled_stale_acceptable"


class ContextOutcome(str, Enum):
    COMPLETE = "complete"
    INCOMPLETE = "incomplete"
    NO_RELEVANT_EVIDENCE = "no_relevant_evidence"
    UNAVAILABLE_DEPENDENCY = "unavailable_dependency"
    BUDGET_EXHAUSTED = "budget_exhausted"


class CapabilityStatus(str, Enum):
    AVAILABLE = "available"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"


class TemporalStatus(str, Enum):
    CURRENT = "current"
    HISTORICAL = "historical"


class EvidenceForm(str, Enum):
    EXACT = "exact"
    EXTRACT = "extract"
    GENERATED_SYNTHESIS = "generated_synthesis"


class Uncertainty(str, Enum):
    NONE_REPORTED = "none_reported"
    SOURCE_UNCERTAIN = "source_uncertain"
    CONFLICTING_EVIDENCE = "conflicting_evidence"
    INCOMPLETE_SUPPORT = "incomplete_support"


class SafeOmissionCode(str, Enum):
    """Content-free omission reasons safe for the response and receipt."""

    BUDGET_LIMIT = "budget_limit"
    DEADLINE_LIMIT = "deadline_limit"
    DEPENDENCY_UNAVAILABLE = "dependency_unavailable"
    CAPABILITY_UNAVAILABLE = "capability_unavailable"
    POLICY_RESTRICTED = "policy_restricted"
    FRESHNESS_REQUIREMENT = "freshness_requirement"


class SafeDegradationCode(str, Enum):
    UPSTREAM_VERIFICATION_STALE = "upstream_verification_stale"
    UPSTREAM_VERIFICATION_UNAVAILABLE = "upstream_verification_unavailable"
    PROJECTION_STALE = "projection_stale"
    OPTIONAL_CAPABILITY_UNAVAILABLE = "optional_capability_unavailable"
    COMPACT_FORM_USED = "compact_form_used"


class ContextApplicability(ContractModel):
    """Optional narrowing filters; these identifiers never grant authority."""

    project_id: OpaqueId | None = None
    workspace_id: OpaqueId | None = None

    @model_validator(mode="after")
    def require_filter(self) -> "ContextApplicability":
        if self.project_id is None and self.workspace_id is None:
            raise ValueError("at least one applicability filter is required")
        return self


class TokenizerBasis(ContractModel):
    """Declared tokenizer identity for a request or measured response."""

    tokenizer_id: OpaqueId
    tokenizer_version: OpaqueId


class SerializedResponseBudget(ContractModel):
    """Maximum token count for the entire serialized response envelope."""

    max_tokens: SafeCounter = Field(gt=0)
    scope: Literal["entire_serialized_response"]
    tokenizer: TokenizerBasis


class SerializedResponseUsage(ContractModel):
    """Measured tokens for framing, metadata, evidence, and all other fields."""

    tokens: SafeCounter
    scope: Literal["entire_serialized_response"]
    tokenizer: TokenizerBasis


class ContextRequest(VersionedContract):
    """A task-scoped request that can only narrow independently verified access."""

    request_id: OpaqueId
    task_intent: str = Field(min_length=1, max_length=8192)
    requested_applicability: ContextApplicability | None
    time_perspective: TimePerspective
    as_of: AwareDatetime | None
    response_budget: SerializedResponseBudget
    deadline: AwareDatetime
    freshness_requirement: FreshnessRequirement
    prerequisite_receipt: ReferenceHandle | None

    @model_validator(mode="after")
    def validate_time_perspective(self) -> "ContextRequest":
        if self.time_perspective is TimePerspective.CURRENT and self.as_of is not None:
            raise ValueError("a current request cannot specify as_of")
        if self.time_perspective is TimePerspective.HISTORICAL and self.as_of is None:
            raise ValueError("a historical request requires as_of")
        if (
            self.prerequisite_receipt is not None
            and self.prerequisite_receipt.kind != "artifact"
        ):
            raise ValueError("prerequisite_receipt requires an artifact reference")
        return self


class SelectedContextItem(ContractModel):
    """One selected item with exact expansion, lineage, and freshness evidence."""

    reference: ReferenceHandle
    source_versions: tuple[SourceVersion, ...] = Field(min_length=1)
    expansion_handle: ReferenceHandle
    temporal_status: TemporalStatus
    evidence_form: EvidenceForm
    evidence_text: str | None = Field(default=None, max_length=65536)
    uncertainty: Uncertainty
    freshness: FreshnessWitness

    @model_validator(mode="after")
    def require_unique_source_versions(self) -> "SelectedContextItem":
        record_ids = [str(version.record_id) for version in self.source_versions]
        if len(record_ids) != len(set(record_ids)):
            raise ValueError("source_versions must identify unique records")
        if (
            self.freshness.projection.status is ProjectionStatus.CURRENT
            and projection_status_for(
                self.freshness.projection.applied_sources, self.source_versions
            )
            is not ProjectionStatus.CURRENT
        ):
            raise ValueError(
                "a current projection must match the complete selected source set"
            )
        return self


class ContextBundle(VersionedContract):
    """A bounded context response with explicit empty, partial, and failure states."""

    bundle_id: OpaqueId
    request_id: OpaqueId
    outcome: ContextOutcome
    capability_status: CapabilityStatus
    selected_items: tuple[SelectedContextItem, ...]
    omissions: tuple[SafeOmissionCode, ...]
    response_usage: SerializedResponseUsage
    assembly_receipt: ReferenceHandle

    @model_validator(mode="after")
    def validate_outcome_shape(self) -> "ContextBundle":
        if self.assembly_receipt.kind != "artifact":
            raise ValueError("assembly_receipt requires an artifact reference")
        if len(self.omissions) != len(set(self.omissions)):
            raise ValueError("omission codes must be unique")

        empty_outcomes = {
            ContextOutcome.NO_RELEVANT_EVIDENCE,
            ContextOutcome.UNAVAILABLE_DEPENDENCY,
            ContextOutcome.BUDGET_EXHAUSTED,
        }
        if self.outcome in empty_outcomes and self.selected_items:
            raise ValueError(f"{self.outcome.value} cannot contain selected items")
        if self.outcome is ContextOutcome.COMPLETE:
            if not self.selected_items:
                raise ValueError("a complete bundle requires selected evidence")
            if self.omissions:
                raise ValueError("a complete bundle cannot report omissions")
            if self.capability_status is not CapabilityStatus.AVAILABLE:
                raise ValueError("a complete bundle requires available capabilities")
        if self.outcome is ContextOutcome.NO_RELEVANT_EVIDENCE:
            if self.omissions:
                raise ValueError("no_relevant_evidence cannot report omissions")
            if self.capability_status is not CapabilityStatus.AVAILABLE:
                raise ValueError(
                    "no_relevant_evidence requires available capabilities"
                )
        if self.outcome is ContextOutcome.INCOMPLETE and not self.omissions:
            raise ValueError("an incomplete bundle requires a safe omission reason")
        if self.outcome is ContextOutcome.UNAVAILABLE_DEPENDENCY:
            if SafeOmissionCode.DEPENDENCY_UNAVAILABLE not in self.omissions:
                raise ValueError("unavailable_dependency requires its omission code")
        if self.outcome is ContextOutcome.BUDGET_EXHAUSTED:
            if SafeOmissionCode.BUDGET_LIMIT not in self.omissions:
                raise ValueError("budget_exhausted requires its omission code")
        return self


class AssemblyStageTiming(ContractModel):
    """Content-free elapsed timing for an observable assembly stage."""

    stage: OpaqueId
    elapsed_ms: SafeCounter


class ExpansionLineage(ContractModel):
    """A delivered reference produced by expanding another authorized reference."""

    input_reference: ReferenceHandle
    output_reference: ReferenceHandle


class ContextAssemblyReceipt(VersionedContract):
    """Privacy-aware evidence of observable context assembly decisions.

    The receipt intentionally has no raw query, source name, excluded-object
    identifier/count, model reasoning, or plaintext evidence fields.
    """

    receipt_id: OpaqueId
    request_id: OpaqueId
    bundle_id: OpaqueId
    selector_version: OpaqueId
    router_version: OpaqueId
    source_versions: tuple[SourceVersion, ...]
    policy_decision_witnesses: tuple[OpaqueId, ...]
    selected_references: tuple[ReferenceHandle, ...]
    exclusions: tuple[SafeOmissionCode, ...]
    degradations: tuple[SafeDegradationCode, ...]
    stage_timings: tuple[AssemblyStageTiming, ...]
    response_usage: SerializedResponseUsage
    optimization_lineage: tuple[ReferenceHandle, ...]
    expansion_lineage: tuple[ExpansionLineage, ...]

    @model_validator(mode="after")
    def validate_receipt_sets(self) -> "ContextAssemblyReceipt":
        source_ids = [str(version.record_id) for version in self.source_versions]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("source_versions must identify unique records")
        if len(self.policy_decision_witnesses) != len(
            set(self.policy_decision_witnesses)
        ):
            raise ValueError("policy decision witnesses must be unique")
        if len(self.exclusions) != len(set(self.exclusions)):
            raise ValueError("exclusion codes must be unique")
        if len(self.degradations) != len(set(self.degradations)):
            raise ValueError("degradation codes must be unique")

        stages = [str(timing.stage) for timing in self.stage_timings]
        if len(stages) != len(set(stages)):
            raise ValueError("stage timings must identify unique stages")

        if (
            SafeDegradationCode.UPSTREAM_VERIFICATION_STALE in self.degradations
            and SafeDegradationCode.UPSTREAM_VERIFICATION_UNAVAILABLE
            in self.degradations
        ):
            raise ValueError("upstream verification cannot be stale and unavailable")
        return self


def upstream_degradation_for(
    status: UpstreamVerificationStatus,
) -> SafeDegradationCode | None:
    """Map non-current upstream knowledge to a content-free receipt code."""

    if status is UpstreamVerificationStatus.STALE:
        return SafeDegradationCode.UPSTREAM_VERIFICATION_STALE
    if status in {
        UpstreamVerificationStatus.UNVERIFIED,
        UpstreamVerificationStatus.UNAVAILABLE,
    }:
        return SafeDegradationCode.UPSTREAM_VERIFICATION_UNAVAILABLE
    return None


def bundle_fits_request(request: ContextRequest, bundle: ContextBundle) -> bool:
    """Check request correlation and whole-response budget/basis agreement."""

    return (
        bundle.request_id == request.request_id
        and bundle.response_usage.scope == request.response_budget.scope
        and bundle.response_usage.tokenizer == request.response_budget.tokenizer
        and bundle.response_usage.tokens <= request.response_budget.max_tokens
    )


def receipt_matches_bundle(
    receipt: ContextAssemblyReceipt, bundle: ContextBundle
) -> bool:
    """Check the public receipt witnesses against the delivered bundle."""

    selected = tuple(item.reference for item in bundle.selected_items)
    source_versions = {
        (
            str(version.record_id),
            version.content_revision,
            version.policy_epoch,
        )
        for item in bundle.selected_items
        for version in item.source_versions
    }
    receipt_sources = {
        (
            str(version.record_id),
            version.content_revision,
            version.policy_epoch,
        )
        for version in receipt.source_versions
    }
    return (
        receipt.request_id == bundle.request_id
        and receipt.bundle_id == bundle.bundle_id
        and receipt.response_usage == bundle.response_usage
        and receipt.selected_references == selected
        and receipt_sources == source_versions
        and receipt.exclusions == bundle.omissions
        and receipt.receipt_id == bundle.assembly_receipt.id
        and bundle.assembly_receipt.kind == "artifact"
    )


__all__ = [
    "AssemblyStageTiming",
    "CapabilityStatus",
    "ContextApplicability",
    "ContextAssemblyReceipt",
    "ContextBundle",
    "ContextOutcome",
    "ContextRequest",
    "EvidenceForm",
    "ExpansionLineage",
    "FreshnessRequirement",
    "SafeDegradationCode",
    "SafeOmissionCode",
    "SelectedContextItem",
    "SerializedResponseBudget",
    "SerializedResponseUsage",
    "TemporalStatus",
    "TimePerspective",
    "TokenizerBasis",
    "Uncertainty",
    "bundle_fits_request",
    "receipt_matches_bundle",
    "upstream_degradation_for",
]
