"""Inert task-aware context delivery and privacy-aware receipt contracts.

These types describe messages only.  They neither authenticate a principal nor
prove authorization, source existence, transactionality, or token counting.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
from typing import Literal, TypeVar

from pydantic import AwareDatetime, BaseModel, Field, TypeAdapter, model_validator

from contracts_common import ContractModel, OpaqueId, SafeCounter, VersionedContract
from contracts_freshness import (
    FreshnessWitness,
    ProjectionStatus,
    UpstreamVerificationStatus,
    projection_status_for,
)
from contracts_references import ReferenceHandle, SourceVersion


_ContractT = TypeVar("_ContractT", bound=ContractModel)


def _native_contract_graph(
    value: object, active_containers: set[int] | None = None
) -> object:
    """Materialize complete stored fields and reject cyclic input graphs."""

    if not isinstance(value, (BaseModel, tuple, list, dict)):
        return value

    if active_containers is None:
        active_containers = set()
    value_id = id(value)
    if value_id in active_containers:
        raise ValueError("cyclic contract input is not supported")
    active_containers.add(value_id)
    try:
        if isinstance(value, BaseModel):
            fields = {
                key: _native_contract_graph(item, active_containers)
                for key, item in vars(value).items()
            }
            extra = getattr(value, "__pydantic_extra__", None)
            if extra is not None:
                if not isinstance(extra, Mapping):
                    raise ValueError("contract extra storage must be a mapping")
                if fields.keys() & extra.keys():
                    raise ValueError(
                        "contract input contains conflicting declared and extra fields"
                    )
                fields.update(
                    {
                        key: _native_contract_graph(item, active_containers)
                        for key, item in extra.items()
                    }
                )
            return fields
        if isinstance(value, tuple):
            return tuple(
                _native_contract_graph(item, active_containers) for item in value
            )
        if isinstance(value, list):
            return [
                _native_contract_graph(item, active_containers) for item in value
            ]
        return {
            key: _native_contract_graph(item, active_containers)
            for key, item in value.items()
        }
    finally:
        active_containers.remove(value_id)


def _revalidated_snapshot(
    model_type: type[_ContractT], value: object
) -> _ContractT | None:
    """Strictly rebuild a complete contract graph, failing closed on any defect."""

    try:
        return model_type.model_validate(
            _native_contract_graph(value),
            strict=True,
        )
    except Exception:
        return None


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
    PROJECTION_UNKNOWN = "projection_unknown"
    PROJECTION_UNAVAILABLE = "projection_unavailable"
    OPTIONAL_CAPABILITY_UNAVAILABLE = "optional_capability_unavailable"
    COMPACT_FORM_USED = "compact_form_used"


_UPSTREAM_STATUS_ADAPTER = TypeAdapter(UpstreamVerificationStatus)
_FRESHNESS_DEGRADATION_CODES = frozenset(
    {
        SafeDegradationCode.UPSTREAM_VERIFICATION_STALE,
        SafeDegradationCode.UPSTREAM_VERIFICATION_UNAVAILABLE,
        SafeDegradationCode.PROJECTION_STALE,
        SafeDegradationCode.PROJECTION_UNKNOWN,
        SafeDegradationCode.PROJECTION_UNAVAILABLE,
    }
)


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
        if self.reference.tenant_id != self.expansion_handle.tenant_id:
            raise ValueError(
                "reference and expansion_handle must have the same tenant scope"
            )
        if any(
            checkpoint.source.tenant_id != self.reference.tenant_id
            for checkpoint in self.freshness.source_checkpoints
        ):
            raise ValueError(
                "source checkpoints must have the selected reference tenant scope"
            )

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
        if any(
            item.reference.tenant_id != self.assembly_receipt.tenant_id
            for item in self.selected_items
        ):
            raise ValueError(
                "selected items and assembly_receipt must share one tenant scope"
            )
        if len(self.omissions) != len(set(self.omissions)):
            raise ValueError("omission codes must be unique")

        source_versions: dict[str, tuple[int, int]] = {}
        for item in self.selected_items:
            for version in item.source_versions:
                record_id = str(version.record_id)
                value = (version.content_revision, version.policy_epoch)
                previous = source_versions.setdefault(record_id, value)
                if previous != value:
                    raise ValueError(
                        "selected items contain conflicting versions for source "
                        f"{record_id!r}"
                    )

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
            if self.capability_status is not CapabilityStatus.UNAVAILABLE:
                raise ValueError(
                    "unavailable_dependency requires unavailable capabilities"
                )
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

    @model_validator(mode="after")
    def require_one_tenant_scope(self) -> "ExpansionLineage":
        if self.input_reference.tenant_id != self.output_reference.tenant_id:
            raise ValueError("expansion lineage endpoints must share one tenant scope")
        return self


class ContextAssemblyReceipt(VersionedContract):
    """Privacy-aware evidence of observable context assembly decisions.

    The receipt intentionally has no raw query, source name, excluded-object
    identifier/count, model reasoning, or plaintext evidence fields.
    """

    receipt_id: OpaqueId
    tenant_id: OpaqueId
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

        scoped_references = [
            *self.selected_references,
            *self.optimization_lineage,
            *(
                reference
                for lineage in self.expansion_lineage
                for reference in (
                    lineage.input_reference,
                    lineage.output_reference,
                )
            ),
        ]
        if any(
            reference.tenant_id != self.tenant_id
            for reference in scoped_references
        ):
            raise ValueError(
                "receipt references and lineage must share the receipt tenant scope"
            )
        return self


def upstream_degradation_for(
    status: UpstreamVerificationStatus,
) -> SafeDegradationCode | None:
    """Map non-current upstream knowledge to a content-free receipt code."""

    status = _UPSTREAM_STATUS_ADAPTER.validate_python(status, strict=True)
    if status is UpstreamVerificationStatus.STALE:
        return SafeDegradationCode.UPSTREAM_VERIFICATION_STALE
    if status in {
        UpstreamVerificationStatus.UNVERIFIED,
        UpstreamVerificationStatus.UNAVAILABLE,
    }:
        return SafeDegradationCode.UPSTREAM_VERIFICATION_UNAVAILABLE
    return None


def bundle_matches_reported_evidence(
    request: ContextRequest, bundle: ContextBundle
) -> bool:
    """Compare only request correlation and evidence reported by the bundle.

    This checks request correlation, the reported whole-response token count,
    scope and tokenizer basis, a satisfying reported outcome with selected
    evidence, current temporal labels, and the requested freshness level.  It
    does not establish deadline compliance, requested applicability,
    prerequisite fulfillment, authorization, task relevance, or reference and
    evidence resolution.

    Historical items currently lack an applicability interval or point-in-time
    witness, so this comparison never matches a historical ``as_of`` request.
    It also treats partial and failed outcomes as non-matching even though those
    bundles remain valid, informative responses.
    """

    request_snapshot = _revalidated_snapshot(ContextRequest, request)
    bundle_snapshot = _revalidated_snapshot(ContextBundle, bundle)
    if request_snapshot is None or bundle_snapshot is None:
        return False
    request = request_snapshot
    bundle = bundle_snapshot

    if (
        bundle.request_id != request.request_id
        or bundle.response_usage.scope != request.response_budget.scope
        or bundle.response_usage.tokenizer != request.response_budget.tokenizer
        or bundle.response_usage.tokens > request.response_budget.max_tokens
        or bundle.outcome
        not in {ContextOutcome.COMPLETE, ContextOutcome.NO_RELEVANT_EVIDENCE}
    ):
        return False

    if not bundle.selected_items:
        return False

    if request.time_perspective is TimePerspective.HISTORICAL:
        return False
    if any(
        item.temporal_status is not TemporalStatus.CURRENT
        for item in bundle.selected_items
    ):
        return False

    if request.freshness_requirement is FreshnessRequirement.CURRENT_VERIFIED:
        return all(
            item.freshness.upstream_status is UpstreamVerificationStatus.CURRENT
            and item.freshness.projection.status is ProjectionStatus.CURRENT
            for item in bundle.selected_items
        )
    if request.freshness_requirement is FreshnessRequirement.CURRENT_KNOWN:
        return all(
            item.freshness.upstream_status
            in {UpstreamVerificationStatus.CURRENT, UpstreamVerificationStatus.STALE}
            and item.freshness.projection.status is ProjectionStatus.CURRENT
            for item in bundle.selected_items
        )
    return all(
        item.freshness.upstream_status
        in {UpstreamVerificationStatus.CURRENT, UpstreamVerificationStatus.STALE}
        and item.freshness.projection.status
        in {ProjectionStatus.CURRENT, ProjectionStatus.STALE}
        for item in bundle.selected_items
    )


def _freshness_degradations_for(
    items: tuple[SelectedContextItem, ...],
) -> set[SafeDegradationCode]:
    result: set[SafeDegradationCode] = set()
    for item in items:
        upstream = upstream_degradation_for(item.freshness.upstream_status)
        if upstream is not None:
            result.add(upstream)
        if item.freshness.projection.status is ProjectionStatus.STALE:
            result.add(SafeDegradationCode.PROJECTION_STALE)
        elif item.freshness.projection.status is ProjectionStatus.UNKNOWN:
            result.add(SafeDegradationCode.PROJECTION_UNKNOWN)
        elif item.freshness.projection.status is ProjectionStatus.UNAVAILABLE:
            result.add(SafeDegradationCode.PROJECTION_UNAVAILABLE)
    return result


def _reference_key(reference: ReferenceHandle) -> tuple[str, str, str, str]:
    return (
        reference.kind,
        str(reference.id),
        str(reference.tenant_id),
        str(reference.resolver),
    )


def receipt_matches_bundle(
    receipt: ContextAssemblyReceipt, bundle: ContextBundle
) -> bool:
    """Check the public receipt witnesses against the delivered bundle."""

    receipt_snapshot = _revalidated_snapshot(ContextAssemblyReceipt, receipt)
    bundle_snapshot = _revalidated_snapshot(ContextBundle, bundle)
    if receipt_snapshot is None or bundle_snapshot is None:
        return False
    receipt = receipt_snapshot
    bundle = bundle_snapshot

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
    expected_degradations = _freshness_degradations_for(bundle.selected_items)
    receipt_freshness_degradations = (
        set(receipt.degradations) & _FRESHNESS_DEGRADATION_CODES
    )
    expected_expansions = {
        (_reference_key(item.expansion_handle), _reference_key(item.reference))
        for item in bundle.selected_items
        if item.expansion_handle != item.reference
    }
    receipt_expansions = {
        (
            _reference_key(lineage.input_reference),
            _reference_key(lineage.output_reference),
        )
        for lineage in receipt.expansion_lineage
    }
    bundle_tenant = bundle.assembly_receipt.tenant_id
    receipt_lineage_references = [
        *receipt.optimization_lineage,
        *(
            reference
            for lineage in receipt.expansion_lineage
            for reference in (lineage.input_reference, lineage.output_reference)
        ),
    ]
    return (
        receipt.request_id == bundle.request_id
        and receipt.bundle_id == bundle.bundle_id
        and receipt.tenant_id == bundle_tenant
        and receipt.response_usage == bundle.response_usage
        and receipt.selected_references == selected
        and receipt_sources == source_versions
        and receipt.exclusions == bundle.omissions
        and receipt_freshness_degradations == expected_degradations
        and receipt_expansions == expected_expansions
        and len(receipt_expansions) == len(receipt.expansion_lineage)
        and all(
            reference.tenant_id == bundle_tenant
            for reference in receipt_lineage_references
        )
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
    "bundle_matches_reported_evidence",
    "receipt_matches_bundle",
    "upstream_degradation_for",
]
