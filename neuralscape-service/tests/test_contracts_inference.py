"""Identity, fallback, and score semantics for inference contracts."""

import pytest
from pydantic import ValidationError

from contracts_inference import (
    CandidateScore,
    DecisionBatchRequest,
    DecisionBatchResult,
    DecisionCandidate,
    DecisionHead,
    DecisionHeadResult,
    DecisionOutcome,
    DecisionRequest,
    DecisionResult,
    FallbackGranularity,
    GenerationOutcome,
    GenerationProposal,
    GenerationRequest,
    GenerationResult,
    InferenceResourceLimits,
    ScoreDistribution,
    ScoreKind,
    SourceSpan,
    validate_decision_batch_result,
    validate_decision_result,
    validate_generation_result,
)
from contracts_references import ReferenceHandle, SourceVersion


VERSION = "candidate-v1"


def source_version(source_id="source-1", revision=3):
    """Construct the dependency as an already-validated product value.

    SourceVersion's exact field-level behavior is owned and tested by the
    references contract.  These tests exercise preservation and comparison of
    that value without duplicating its schema.
    """

    return SourceVersion(
        record_id=source_id,
        content_revision=revision,
        policy_epoch=7,
    )


def policy_reference():
    return ReferenceHandle(
        kind="artifact",
        id="processing-policy-1",
        tenant_id="tenant-1",
        resolver="policy",
    )


def limits():
    return InferenceResourceLimits(
        max_input_units=1000,
        max_output_units=100,
        max_attempts=2,
    )


def decision_request(
    *,
    batch=False,
    fallback=FallbackGranularity.WHOLE_BATCH,
    additional_source_versions=(),
):
    source = source_version()
    values = {
        "schema_version": VERSION,
        "request_id": "request-1",
        "operation": "category_suggestion",
        "source_versions": (source, *additional_source_versions),
        "spans": (
            SourceSpan(span_id="span-1", source_version=source, start=0, end=8),
            SourceSpan(span_id="span-2", source_version=source, start=9, end=16),
        ),
        "candidates": (
            DecisionCandidate(candidate_id="candidate-a"),
            DecisionCandidate(candidate_id="candidate-b"),
        ),
        "processing_policy": policy_reference(),
        "resource_limits": limits(),
        "deadline_ms": 500,
    }
    heads = (
        DecisionHead(
            head_id="retain",
            span_ids=("span-1",),
            candidate_ids=("candidate-a", "candidate-b"),
            required=True,
        ),
        DecisionHead(
            head_id="category",
            span_ids=("span-2",),
            candidate_ids=("candidate-a", "candidate-b"),
            required=True,
        ),
    )
    if batch:
        return DecisionBatchRequest(
            **values,
            heads=heads,
            fallback_granularity=fallback,
        )
    return DecisionRequest(**values, head=heads[0])


def selected(head_id="retain", candidate_id="candidate-a", distribution=None):
    return DecisionHeadResult(
        head_id=head_id,
        outcome=DecisionOutcome.SELECTED,
        selected_candidate_ids=(candidate_id,),
        distribution=distribution,
        reason_code=None,
    )


def abstained(head_id):
    return DecisionHeadResult(
        head_id=head_id,
        outcome=DecisionOutcome.ABSTAINED,
        selected_candidate_ids=(),
        distribution=None,
        reason_code="below-threshold",
    )


def test_source_span_is_half_open_and_nonempty():
    source = source_version()
    with pytest.raises(ValidationError, match="greater than start"):
        SourceSpan(span_id="empty", source_version=source, start=2, end=2)
    with pytest.raises(ValidationError):
        SourceSpan(span_id="reverse", source_version=source, start=3, end=2)


def test_shared_snapshot_rejects_conflicting_versions_of_one_source():
    with pytest.raises(ValidationError, match="two versions of one record"):
        decision_request(additional_source_versions=(source_version(revision=4),))


@pytest.mark.parametrize(
    "mutation, message",
    [
        ("unknown_span", "unknown span"),
        ("unknown_candidate", "unknown candidate"),
        ("duplicate_head", "head IDs must be unique"),
    ],
)
def test_batch_request_rejects_broken_identity_links(mutation, message):
    request = decision_request(batch=True)
    heads = list(request.heads)
    if mutation == "unknown_span":
        heads[0] = heads[0].model_copy(update={"span_ids": ("not-present",)})
    elif mutation == "unknown_candidate":
        heads[0] = heads[0].model_copy(update={"candidate_ids": ("not-present",)})
    else:
        heads[1] = heads[1].model_copy(update={"head_id": heads[0].head_id})

    with pytest.raises(ValidationError, match=message):
        DecisionBatchRequest(**request.model_dump(exclude={"heads"}), heads=tuple(heads))


def test_atomic_result_cannot_select_or_score_candidates_outside_closed_set():
    request = decision_request()
    unknown_selection = DecisionResult(
        schema_version=VERSION,
        request_id=request.request_id,
        head_result=selected(candidate_id="invented"),
        attempts=(),
    )
    with pytest.raises(ValueError, match="selected an unknown"):
        validate_decision_result(request, unknown_selection)

    distribution = ScoreDistribution(
        distribution_id="dist-1",
        head_id="retain",
        kind=ScoreKind.RAW_SCORE,
        complete=False,
        scores=(CandidateScore(candidate_id="invented", value=4.2),),
    )
    scored_unknown = DecisionResult(
        schema_version=VERSION,
        request_id=request.request_id,
        head_result=selected(distribution=distribution),
        attempts=(),
    )
    with pytest.raises(ValueError, match="scored an unknown"):
        validate_decision_result(request, scored_unknown)


def test_complete_distribution_must_cover_every_candidate_for_the_head():
    request = decision_request()
    incomplete_identity_set = ScoreDistribution(
        distribution_id="dist-1",
        head_id="retain",
        kind=ScoreKind.RAW_SCORE,
        complete=True,
        scores=(CandidateScore(candidate_id="candidate-a", value=4.2),),
    )
    result = DecisionResult(
        schema_version=VERSION,
        request_id=request.request_id,
        head_result=selected(distribution=incomplete_identity_set),
        attempts=(),
    )
    with pytest.raises(ValueError, match="every head candidate"):
        validate_decision_result(request, result)


def test_unknown_confidence_has_no_numeric_value_or_completeness_claim():
    distribution = ScoreDistribution(
        distribution_id="dist-unknown",
        head_id="retain",
        kind=ScoreKind.UNKNOWN,
        complete=False,
        scores=(),
    )
    assert distribution.kind is ScoreKind.UNKNOWN

    with pytest.raises(ValidationError, match="unknown confidence"):
        ScoreDistribution(
            distribution_id="dist-bad",
            head_id="retain",
            kind=ScoreKind.UNKNOWN,
            complete=False,
            scores=(CandidateScore(candidate_id="candidate-a", value=0.9),),
        )


@pytest.mark.parametrize(
    "kind",
    [ScoreKind.UNCALIBRATED_PROBABILITY, ScoreKind.CALIBRATED_PROBABILITY],
)
def test_probabilities_require_complete_normalized_distributions(kind):
    valid = ScoreDistribution(
        distribution_id="dist-1",
        head_id="retain",
        kind=kind,
        complete=True,
        scores=(
            CandidateScore(candidate_id="candidate-a", value=0.7),
            CandidateScore(candidate_id="candidate-b", value=0.3),
        ),
    )
    assert sum(item.value for item in valid.scores) == pytest.approx(1)

    with pytest.raises(ValidationError, match="sum to one"):
        ScoreDistribution(
            distribution_id="dist-2",
            head_id="retain",
            kind=kind,
            complete=True,
            scores=(
                CandidateScore(candidate_id="candidate-a", value=0.7),
                CandidateScore(candidate_id="candidate-b", value=0.4),
            ),
        )
    with pytest.raises(ValidationError, match="must be complete"):
        ScoreDistribution(
            distribution_id="dist-3",
            head_id="retain",
            kind=kind,
            complete=False,
            scores=(CandidateScore(candidate_id="candidate-a", value=1.0),),
        )


def test_raw_score_is_not_reinterpreted_as_probability():
    distribution = ScoreDistribution(
        distribution_id="dist-raw",
        head_id="retain",
        kind=ScoreKind.RAW_SCORE,
        complete=False,
        scores=(CandidateScore(candidate_id="candidate-a", value=12.5),),
    )
    assert distribution.scores[0].value == 12.5


def test_valid_empty_is_distinct_from_abstention_and_error():
    empty = DecisionHeadResult(
        head_id="retain",
        outcome=DecisionOutcome.EMPTY,
        selected_candidate_ids=(),
        distribution=None,
        reason_code=None,
    )
    assert empty.outcome is DecisionOutcome.EMPTY

    with pytest.raises(ValidationError, match="require a reason"):
        DecisionHeadResult(
            head_id="retain",
            outcome=DecisionOutcome.ABSTAINED,
            selected_candidate_ids=(),
            distribution=None,
            reason_code=None,
        )


def test_whole_batch_fallback_preserves_diagnostics_but_blocks_partial_publication():
    request = decision_request(batch=True)
    result = DecisionBatchResult(
        schema_version=VERSION,
        request_id=request.request_id,
        fallback_granularity=FallbackGranularity.WHOLE_BATCH,
        publishable=False,
        head_results=(selected(), abstained("category")),
        attempts=(),
    )
    validate_decision_batch_result(request, result)
    assert result.head_results[0].outcome is DecisionOutcome.SELECTED

    unsafe = result.model_copy(update={"publishable": True})
    with pytest.raises(ValueError, match="forbids partial publication"):
        validate_decision_batch_result(request, unsafe)


def test_batch_result_must_match_requested_heads_and_fallback_granularity():
    request = decision_request(batch=True)
    missing_head = DecisionBatchResult(
        schema_version=VERSION,
        request_id=request.request_id,
        fallback_granularity=FallbackGranularity.WHOLE_BATCH,
        publishable=False,
        head_results=(selected(),),
        attempts=(),
    )
    with pytest.raises(ValueError, match="exactly the requested heads"):
        validate_decision_batch_result(request, missing_head)

    wrong_fallback = missing_head.model_copy(
        update={"fallback_granularity": FallbackGranularity.PER_ITEM}
    )
    with pytest.raises(ValueError, match="does not match request"):
        validate_decision_batch_result(request, wrong_fallback)


def generation_request():
    source = source_version()
    return GenerationRequest(
        schema_version=VERSION,
        request_id="generation-1",
        operation="summarize",
        source_versions=(source,),
        spans=(SourceSpan(span_id="support-1", source_version=source, start=0, end=8),),
        processing_policy=policy_reference(),
        output_schema={"type": "object"},
        resource_limits=limits(),
        deadline_ms=500,
    )


def test_generation_is_machine_authored_and_requires_exact_source_support():
    request = generation_request()
    result = GenerationResult(
        schema_version=VERSION,
        request_id=request.request_id,
        outcome=GenerationOutcome.PROPOSED,
        proposals=(
            GenerationProposal(
                proposal_id="proposal-1",
                output={"summary": "proposal, not authority"},
                support_span_ids=("support-1",),
                machine_authored=True,
            ),
        ),
        reason_code=None,
        attempts=(),
    )
    validate_generation_result(request, result)

    unsupported = result.model_copy(
        update={
            "proposals": (
                result.proposals[0].model_copy(update={"support_span_ids": ("invented",)}),
            )
        }
    )
    with pytest.raises(ValueError, match="unknown support span"):
        validate_generation_result(request, unsupported)


def test_generation_result_cannot_mix_failure_with_proposals():
    proposal = GenerationProposal(
        proposal_id="proposal-1",
        output="text",
        support_span_ids=("support-1",),
        machine_authored=True,
    )
    with pytest.raises(ValidationError, match="only a proposed outcome"):
        GenerationResult(
            schema_version=VERSION,
            request_id="generation-1",
            outcome=GenerationOutcome.ERROR,
            proposals=(proposal,),
            reason_code="provider-error",
            attempts=(),
        )


def test_resource_limits_and_deadlines_cannot_be_zero():
    with pytest.raises(ValidationError, match="limits must be positive"):
        InferenceResourceLimits(
            max_input_units=0,
            max_output_units=100,
            max_attempts=1,
        )
    request = decision_request().model_dump()
    request["deadline_ms"] = 0
    with pytest.raises(ValidationError, match="deadline_ms must be positive"):
        DecisionRequest(**request)
