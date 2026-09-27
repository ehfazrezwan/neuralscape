"""Validated boundaries for bounded decisions and generative proposals.

Decision operations may only select supplied candidate identities (or produce a
distinct empty, abstained, or error outcome).  Generation operations return
machine-authored proposals with explicit source support.  Neither result grants
authority or publishes content by itself.
"""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field, JsonValue, model_validator

from contracts_common import ContractModel, OpaqueId, SafeCounter, VersionedContract
from contracts_references import ReferenceHandle, SourceVersion

FiniteNumber = Annotated[float, Field(allow_inf_nan=False)]


class FallbackGranularity(StrEnum):
    WHOLE_BATCH = "whole_batch"
    PER_ITEM = "per_item"


class ScoreKind(StrEnum):
    """The meaning of scores, without treating confidence as calibration."""

    UNKNOWN = "unknown"
    RAW_SCORE = "raw_score"
    UNCALIBRATED_PROBABILITY = "uncalibrated_probability"
    CALIBRATED_PROBABILITY = "calibrated_probability"


class DecisionOutcome(StrEnum):
    SELECTED = "selected"
    EMPTY = "empty"
    ABSTAINED = "abstained"
    ERROR = "error"


class GenerationOutcome(StrEnum):
    PROPOSED = "proposed"
    EMPTY = "empty"
    ABSTAINED = "abstained"
    ERROR = "error"


class SourceSpan(ContractModel):
    """An exact half-open character span in one immutable source version."""

    span_id: OpaqueId
    source_version: SourceVersion
    start: SafeCounter
    end: SafeCounter

    @model_validator(mode="after")
    def validate_bounds(self) -> SourceSpan:
        if self.end <= self.start:
            raise ValueError("source span end must be greater than start")
        return self


class DecisionCandidate(ContractModel):
    """A closed-set candidate whose opaque identity must be preserved."""

    candidate_id: OpaqueId


class DecisionHead(ContractModel):
    """One independently identifiable decision over source spans and candidates."""

    head_id: OpaqueId
    span_ids: tuple[OpaqueId, ...]
    candidate_ids: tuple[OpaqueId, ...]
    required: bool

    @model_validator(mode="after")
    def validate_identifiers(self) -> DecisionHead:
        if not self.span_ids:
            raise ValueError("a decision head requires at least one source span")
        if len(set(self.span_ids)) != len(self.span_ids):
            raise ValueError("decision head span IDs must be unique")
        if not self.candidate_ids:
            raise ValueError("a decision head requires at least one candidate")
        if len(set(self.candidate_ids)) != len(self.candidate_ids):
            raise ValueError("decision head candidate IDs must be unique")
        return self


class InferenceResourceLimits(ContractModel):
    """Caller-supplied limits; this contract does not invent product budgets."""

    max_input_units: SafeCounter
    max_output_units: SafeCounter
    max_attempts: SafeCounter

    @model_validator(mode="after")
    def validate_positive_limits(self) -> InferenceResourceLimits:
        if self.max_input_units == 0 or self.max_output_units == 0:
            raise ValueError("input and output limits must be positive")
        if self.max_attempts == 0:
            raise ValueError("max_attempts must be positive")
        return self


class AttemptUsage(ContractModel):
    """Usage retained per attempt, including work observed after cancellation."""

    attempt_id: OpaqueId
    model_version: OpaqueId
    policy_version: OpaqueId
    input_units: SafeCounter
    output_units: SafeCounter
    late: bool


def _validate_source_snapshot(source_versions: tuple[SourceVersion, ...]) -> None:
    if not source_versions:
        raise ValueError("at least one source version is required")
    record_ids = [source.record_id for source in source_versions]
    if len(set(record_ids)) != len(record_ids):
        raise ValueError("a source snapshot cannot contain two versions of one record")


class _DecisionInputs(VersionedContract):
    request_id: OpaqueId
    operation: OpaqueId
    source_versions: tuple[SourceVersion, ...]
    spans: tuple[SourceSpan, ...]
    candidates: tuple[DecisionCandidate, ...]
    processing_policy: ReferenceHandle
    resource_limits: InferenceResourceLimits
    deadline_ms: SafeCounter

    def _validate_input_identities(self) -> None:
        _validate_source_snapshot(self.source_versions)
        if len(self.spans) == 0:
            raise ValueError("at least one source span is required")
        span_ids = [span.span_id for span in self.spans]
        if len(set(span_ids)) != len(span_ids):
            raise ValueError("source span IDs must be unique")
        for span in self.spans:
            if span.source_version not in self.source_versions:
                raise ValueError("every span must reference a declared source version")
        candidate_ids = [candidate.candidate_id for candidate in self.candidates]
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("candidate IDs must be unique")
        if self.deadline_ms == 0:
            raise ValueError("deadline_ms must be positive")

    def _validate_heads(self, heads: tuple[DecisionHead, ...]) -> None:
        span_ids = {span.span_id for span in self.spans}
        candidate_ids = {candidate.candidate_id for candidate in self.candidates}
        head_ids = [head.head_id for head in heads]
        if len(set(head_ids)) != len(head_ids):
            raise ValueError("decision head IDs must be unique")
        for head in heads:
            if not set(head.span_ids).issubset(span_ids):
                raise ValueError("decision head references an unknown span ID")
            if not set(head.candidate_ids).issubset(candidate_ids):
                raise ValueError("decision head references an unknown candidate ID")


class DecisionRequest(_DecisionInputs):
    """One atomic bounded decision."""

    head: DecisionHead

    @model_validator(mode="after")
    def validate_request(self) -> DecisionRequest:
        self._validate_input_identities()
        self._validate_heads((self.head,))
        return self


class DecisionBatchRequest(_DecisionInputs):
    """Composite decisions sharing one source snapshot and fallback boundary."""

    heads: tuple[DecisionHead, ...]
    fallback_granularity: FallbackGranularity

    @model_validator(mode="after")
    def validate_request(self) -> DecisionBatchRequest:
        self._validate_input_identities()
        if not self.heads:
            raise ValueError("a decision batch requires at least one head")
        self._validate_heads(self.heads)
        return self


class CandidateScore(ContractModel):
    candidate_id: OpaqueId
    value: FiniteNumber


class ScoreDistribution(ContractModel):
    """Scores for one head, preserving their meaning and distribution identity."""

    distribution_id: OpaqueId
    head_id: OpaqueId
    kind: ScoreKind
    complete: bool
    scores: tuple[CandidateScore, ...]

    @model_validator(mode="after")
    def validate_distribution(self) -> ScoreDistribution:
        candidate_ids = [score.candidate_id for score in self.scores]
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("a distribution cannot score a candidate twice")
        if self.kind is ScoreKind.UNKNOWN:
            if self.complete or self.scores:
                raise ValueError("unknown confidence has no scores and is incomplete")
            return self
        if not self.scores:
            raise ValueError("known score kinds require candidate scores")
        if self.kind in {
            ScoreKind.UNCALIBRATED_PROBABILITY,
            ScoreKind.CALIBRATED_PROBABILITY,
        }:
            if not self.complete:
                raise ValueError("a probability distribution must be complete")
            if any(score.value < 0 or score.value > 1 for score in self.scores):
                raise ValueError("probabilities must be between zero and one")
            if not math.isclose(
                sum(score.value for score in self.scores),
                1.0,
                rel_tol=0.0,
                abs_tol=1e-6,
            ):
                raise ValueError("probability distribution must sum to one")
        return self


class DecisionHeadResult(ContractModel):
    """One head outcome; empty is deliberately distinct from inability to decide."""

    head_id: OpaqueId
    outcome: DecisionOutcome
    selected_candidate_ids: tuple[OpaqueId, ...]
    distribution: ScoreDistribution | None
    reason_code: OpaqueId | None

    @model_validator(mode="after")
    def validate_outcome(self) -> DecisionHeadResult:
        if len(set(self.selected_candidate_ids)) != len(self.selected_candidate_ids):
            raise ValueError("selected candidate IDs must be unique")
        if self.outcome is DecisionOutcome.SELECTED:
            if not self.selected_candidate_ids:
                raise ValueError("selected outcome requires at least one candidate")
            if self.reason_code is not None:
                raise ValueError("selected outcome cannot have a failure reason")
        elif self.selected_candidate_ids:
            raise ValueError("only a selected outcome may contain selected candidates")
        if self.outcome in {DecisionOutcome.ABSTAINED, DecisionOutcome.ERROR}:
            if self.reason_code is None:
                raise ValueError("abstained and error outcomes require a reason code")
        elif self.reason_code is not None:
            raise ValueError("selected and empty outcomes cannot have a failure reason")
        if self.distribution is not None and self.distribution.head_id != self.head_id:
            raise ValueError("score distribution must identify the same head")
        return self


class DecisionResult(VersionedContract):
    request_id: OpaqueId
    head_result: DecisionHeadResult
    attempts: tuple[AttemptUsage, ...]


class DecisionBatchResult(VersionedContract):
    """Batch diagnostics plus whether the batch may cross publication boundary."""

    request_id: OpaqueId
    fallback_granularity: FallbackGranularity
    publishable: bool
    head_results: tuple[DecisionHeadResult, ...]
    attempts: tuple[AttemptUsage, ...]

    @model_validator(mode="after")
    def validate_unique_heads(self) -> DecisionBatchResult:
        head_ids = [result.head_id for result in self.head_results]
        if len(set(head_ids)) != len(head_ids):
            raise ValueError("batch head result IDs must be unique")
        return self


def _validate_head_result(head: DecisionHead, result: DecisionHeadResult) -> None:
    if result.head_id != head.head_id:
        raise ValueError("result head ID does not match request head ID")
    allowed = set(head.candidate_ids)
    if not set(result.selected_candidate_ids).issubset(allowed):
        raise ValueError("result selected an unknown candidate ID")
    if result.distribution is not None:
        scored = {score.candidate_id for score in result.distribution.scores}
        if not scored.issubset(allowed):
            raise ValueError("result scored an unknown candidate ID")
        if result.distribution.complete and scored != allowed:
            raise ValueError("complete distribution must score every head candidate")


def validate_decision_result(
    request: DecisionRequest,
    result: DecisionResult,
) -> None:
    """Validate an atomic result against the exact identities in its request."""

    if result.request_id != request.request_id:
        raise ValueError("result request ID does not match request")
    _validate_head_result(request.head, result.head_result)


def validate_decision_batch_result(
    request: DecisionBatchRequest,
    result: DecisionBatchResult,
) -> None:
    """Validate closed identities and enforce the declared fallback boundary."""

    if result.request_id != request.request_id:
        raise ValueError("result request ID does not match request")
    if result.fallback_granularity is not request.fallback_granularity:
        raise ValueError("result fallback granularity does not match request")
    request_heads = {head.head_id: head for head in request.heads}
    if {item.head_id for item in result.head_results} != set(request_heads):
        raise ValueError("batch result must contain exactly the requested heads")
    unusable_required_head = False
    for head_result in result.head_results:
        head = request_heads[head_result.head_id]
        _validate_head_result(head, head_result)
        if head.required and head_result.outcome in {
            DecisionOutcome.ABSTAINED,
            DecisionOutcome.ERROR,
        }:
            unusable_required_head = True
    if (
        request.fallback_granularity is FallbackGranularity.WHOLE_BATCH
        and unusable_required_head
        and result.publishable
    ):
        raise ValueError("whole-batch fallback forbids partial publication")


class GenerationRequest(VersionedContract):
    """A request for proposals, separate from closed-set decisions."""

    request_id: OpaqueId
    operation: OpaqueId
    source_versions: tuple[SourceVersion, ...]
    spans: tuple[SourceSpan, ...]
    processing_policy: ReferenceHandle
    output_schema: dict[str, JsonValue]
    resource_limits: InferenceResourceLimits
    deadline_ms: SafeCounter

    @model_validator(mode="after")
    def validate_request(self) -> GenerationRequest:
        _validate_source_snapshot(self.source_versions)
        if not self.spans:
            raise ValueError("generation requires source spans")
        span_ids = [span.span_id for span in self.spans]
        if len(set(span_ids)) != len(span_ids):
            raise ValueError("source span IDs must be unique")
        if any(span.source_version not in self.source_versions for span in self.spans):
            raise ValueError("every span must reference a declared source version")
        if not self.output_schema:
            raise ValueError("generation requires a declared output schema")
        if self.deadline_ms == 0:
            raise ValueError("deadline_ms must be positive")
        return self


class GenerationProposal(ContractModel):
    proposal_id: OpaqueId
    output: JsonValue
    support_span_ids: tuple[OpaqueId, ...]
    machine_authored: Literal[True]

    @model_validator(mode="after")
    def validate_support(self) -> GenerationProposal:
        if not self.support_span_ids:
            raise ValueError("a generation proposal requires source support")
        if len(set(self.support_span_ids)) != len(self.support_span_ids):
            raise ValueError("proposal support span IDs must be unique")
        return self


class GenerationResult(VersionedContract):
    request_id: OpaqueId
    outcome: GenerationOutcome
    proposals: tuple[GenerationProposal, ...]
    reason_code: OpaqueId | None
    attempts: tuple[AttemptUsage, ...]

    @model_validator(mode="after")
    def validate_outcome(self) -> GenerationResult:
        if self.outcome is GenerationOutcome.PROPOSED:
            if not self.proposals:
                raise ValueError("proposed outcome requires at least one proposal")
            if self.reason_code is not None:
                raise ValueError("proposed outcome cannot have a failure reason")
        elif self.proposals:
            raise ValueError("only a proposed outcome may contain proposals")
        if self.outcome in {GenerationOutcome.ABSTAINED, GenerationOutcome.ERROR}:
            if self.reason_code is None:
                raise ValueError("abstained and error outcomes require a reason code")
        elif self.reason_code is not None:
            raise ValueError("proposed and empty outcomes cannot have a failure reason")
        proposal_ids = [proposal.proposal_id for proposal in self.proposals]
        if len(set(proposal_ids)) != len(proposal_ids):
            raise ValueError("generation proposal IDs must be unique")
        return self


def validate_generation_result(
    request: GenerationRequest,
    result: GenerationResult,
) -> None:
    """Validate proposal support against the request's exact span identities."""

    if result.request_id != request.request_id:
        raise ValueError("result request ID does not match request")
    allowed_span_ids = {span.span_id for span in request.spans}
    for proposal in result.proposals:
        if not set(proposal.support_span_ids).issubset(allowed_span_ids):
            raise ValueError("proposal references an unknown support span ID")
