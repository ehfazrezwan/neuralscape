"""Identity, fallback, schema, score, and resource inference invariants."""

import math

import pytest
from pydantic import ValidationError

import contracts_inference
from contracts_engines import QualificationClaim
from contracts_inference import (
    JSON_SCHEMA_DIALECT,
    AttemptUsage,
    CalibrationClaim,
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
    ResourceViolation,
    ScoreDistribution,
    ScoreKind,
    SourceEvidence,
    SourceSpan,
    validate_decision_batch_result,
    validate_decision_result,
    validate_generation_result,
    validate_source_span,
)
from contracts_references import ReferenceHandle, SourceVersion


VERSION = "candidate-v1"


def reference(identifier: str, resolver: str = "evidence") -> ReferenceHandle:
    return ReferenceHandle(
        kind="artifact",
        id=identifier,
        tenant_id="tenant-1",
        resolver=resolver,
    )


def source_version(source_id="source-1", revision=3) -> SourceVersion:
    return SourceVersion(
        record_id=source_id,
        content_revision=revision,
        policy_epoch=7,
    )


def source_evidence(text="Aé🙂Z", source=None) -> SourceEvidence:
    return SourceEvidence(source_version=source or source_version(), text=text)


def limits(**overrides) -> InferenceResourceLimits:
    values = {
        "max_total_input_utf8_bytes": 1000,
        "max_total_output_utf8_bytes": 100,
        "max_attempts": 2,
    }
    values.update(overrides)
    return InferenceResourceLimits(**values)


def qualification(operation="category_suggestion") -> QualificationClaim:
    return QualificationClaim(
        operation=operation,
        profile_reference=reference("per-item-profile"),
        profile_version="profile-v2",
        evidence_reference=reference("per-item-evidence"),
        evidence_version="evidence-v4",
    )


def calibration() -> CalibrationClaim:
    return CalibrationClaim(
        profile_reference=reference("calibration-profile"),
        profile_version="profile-v1",
        evidence_reference=reference("calibration-evidence"),
        evidence_version="evidence-v5",
    )


def usage(
    attempt_id="attempt-1",
    input_bytes=10,
    output_bytes=5,
    late=False,
) -> AttemptUsage:
    return AttemptUsage(
        attempt_id=attempt_id,
        model_version="model-v1",
        policy_version="policy-v1",
        input_utf8_bytes=input_bytes,
        output_utf8_bytes=output_bytes,
        late=late,
    )


def decision_request(
    *,
    batch=False,
    fallback=FallbackGranularity.WHOLE_BATCH,
    per_item_qualification=None,
    resource_limits=None,
    sources=None,
    spans=None,
):
    evidence = source_evidence()
    source_values = sources or (evidence,)
    span_values = spans or (
        SourceSpan(
            span_id="span-1",
            source_version=evidence.source_version,
            start_utf8_byte=0,
            end_utf8_byte=3,
        ),
        SourceSpan(
            span_id="span-2",
            source_version=evidence.source_version,
            start_utf8_byte=3,
            end_utf8_byte=8,
        ),
    )
    values = {
        "schema_version": VERSION,
        "request_id": "request-1",
        "operation": "category_suggestion",
        "sources": source_values,
        "spans": span_values,
        "candidates": (
            DecisionCandidate(candidate_id="candidate-a"),
            DecisionCandidate(candidate_id="candidate-b"),
        ),
        "processing_policy": reference("processing-policy", "policy"),
        "resource_limits": resource_limits or limits(),
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
            per_item_qualification=per_item_qualification,
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


def probability_distribution(
    head_id="retain",
    distribution_id="dist-1",
    kind=ScoreKind.UNCALIBRATED_PROBABILITY,
):
    return ScoreDistribution(
        distribution_id=distribution_id,
        head_id=head_id,
        kind=kind,
        complete=True,
        scores=(
            CandidateScore(candidate_id="candidate-a", value=0.7),
            CandidateScore(candidate_id="candidate-b", value=0.3),
        ),
        calibration=calibration() if kind is ScoreKind.CALIBRATED_PROBABILITY else None,
    )


def output_schema(**overrides):
    schema = {
        "$schema": JSON_SCHEMA_DIALECT,
        "type": "object",
        "properties": {"summary": {"type": "string"}},
        "required": ["summary"],
        "additionalProperties": False,
    }
    schema.update(overrides)
    return schema


def generation_request(*, schema=None, resource_limits=None):
    evidence = source_evidence()
    return GenerationRequest(
        schema_version=VERSION,
        request_id="generation-1",
        operation="summarize",
        sources=(evidence,),
        spans=(
            SourceSpan(
                span_id="support-1",
                source_version=evidence.source_version,
                start_utf8_byte=0,
                end_utf8_byte=8,
            ),
        ),
        processing_policy=reference("processing-policy", "policy"),
        output_schema_dialect=JSON_SCHEMA_DIALECT,
        output_schema=schema or output_schema(),
        resource_limits=resource_limits or limits(),
        deadline_ms=500,
    )


def generation_result(output=None, attempts=()):
    return GenerationResult(
        schema_version=VERSION,
        request_id="generation-1",
        outcome=GenerationOutcome.PROPOSED,
        proposals=(
            GenerationProposal(
                proposal_id="proposal-1",
                output=output if output is not None else {"summary": "supported"},
                support_span_ids=("support-1",),
                machine_authored=True,
            ),
        ),
        reason_code=None,
        attempts=attempts,
    )


def test_utf8_spans_require_bounds_and_code_point_boundaries():
    evidence = source_evidence()  # byte boundaries: 0, 1, 3, 7, 8
    out_of_bounds = (
        SourceSpan(
            span_id="span-1",
            source_version=evidence.source_version,
            start_utf8_byte=0,
            end_utf8_byte=9,
        ),
    )
    with pytest.raises(ValidationError, match="exceeds"):
        decision_request(sources=(evidence,), spans=out_of_bounds)

    split_code_point = (
        SourceSpan(
            span_id="span-1",
            source_version=evidence.source_version,
            start_utf8_byte=1,
            end_utf8_byte=2,
        ),
    )
    with pytest.raises(ValidationError, match="split a UTF-8 code point"):
        decision_request(sources=(evidence,), spans=split_code_point)

    with pytest.raises(ValueError, match="exceeds"):
        validate_source_span(evidence, out_of_bounds[0])


def test_source_evidence_rejects_non_utf8_surrogate_text():
    with pytest.raises(ValidationError, match="valid UTF-8"):
        source_evidence(text="bad\ud800text")


def test_span_must_reference_exact_declared_source_version():
    evidence = source_evidence()
    other_version = source_version(revision=4)
    span = SourceSpan(
        span_id="span-1",
        source_version=other_version,
        start_utf8_byte=0,
        end_utf8_byte=1,
    )
    with pytest.raises(ValidationError, match="declared source evidence"):
        decision_request(sources=(evidence,), spans=(span,))


def test_request_computes_span_layout_once_per_referenced_source(monkeypatch):
    first = source_evidence(source=source_version("source-1", 3))
    second = source_evidence(text="BC", source=source_version("source-2", 4))
    spans = (
        SourceSpan(
            span_id="span-1",
            source_version=first.source_version,
            start_utf8_byte=0,
            end_utf8_byte=1,
        ),
        SourceSpan(
            span_id="span-2",
            source_version=first.source_version,
            start_utf8_byte=1,
            end_utf8_byte=3,
        ),
        SourceSpan(
            span_id="span-3",
            source_version=second.source_version,
            start_utf8_byte=0,
            end_utf8_byte=2,
        ),
    )
    calls: dict[str, int] = {}
    extent_calls = 0
    original = SourceEvidence.utf8_boundaries

    def counted_boundaries(self):
        record_id = self.source_version.record_id
        calls[record_id] = calls.get(record_id, 0) + 1
        return original(self)

    def counted_max(values):
        nonlocal extent_calls
        extent_calls += 1
        return max(values)

    monkeypatch.setattr(SourceEvidence, "utf8_boundaries", counted_boundaries)
    monkeypatch.setattr(contracts_inference, "max", counted_max, raising=False)

    request = decision_request(sources=(first, second), spans=spans)
    assert tuple(span.span_id for span in request.spans) == (
        "span-1",
        "span-2",
        "span-3",
    )
    assert calls == {"source-1": 1, "source-2": 1}
    assert extent_calls == 2

    calls.clear()
    extent_calls = 0
    validate_source_span(first, spans[0])
    assert calls == {"source-1": 1}
    assert extent_calls == 1


def test_per_item_fallback_requires_matching_qualification_claim():
    with pytest.raises(ValidationError, match="requires a qualification claim"):
        decision_request(batch=True, fallback=FallbackGranularity.PER_ITEM)
    with pytest.raises(ValidationError, match="operation must match"):
        decision_request(
            batch=True,
            fallback=FallbackGranularity.PER_ITEM,
            per_item_qualification=qualification("other-operation"),
        )
    request = decision_request(
        batch=True,
        fallback=FallbackGranularity.PER_ITEM,
        per_item_qualification=qualification(),
    )
    result = DecisionBatchResult(
        schema_version=VERSION,
        request_id=request.request_id,
        fallback_granularity=FallbackGranularity.PER_ITEM,
        publishable=True,
        head_results=(selected(), abstained("category")),
        attempts=(),
    )
    assert validate_decision_batch_result(request, result).compliant


def test_whole_batch_fallback_blocks_partial_publication():
    request = decision_request(batch=True)
    result = DecisionBatchResult(
        schema_version=VERSION,
        request_id=request.request_id,
        fallback_granularity=FallbackGranularity.WHOLE_BATCH,
        publishable=True,
        head_results=(selected(), abstained("category")),
        attempts=(),
    )
    with pytest.raises(ValueError, match="forbids partial publication"):
        validate_decision_batch_result(request, result)


def test_unknown_raw_and_probability_score_semantics_remain_distinct():
    unknown = ScoreDistribution(
        distribution_id="unknown",
        head_id="retain",
        kind=ScoreKind.UNKNOWN,
        complete=False,
        scores=(),
        calibration=None,
    )
    raw = ScoreDistribution(
        distribution_id="raw",
        head_id="retain",
        kind=ScoreKind.RAW_SCORE,
        complete=False,
        scores=(CandidateScore(candidate_id="candidate-a", value=12.5),),
        calibration=None,
    )
    assert unknown.scores == ()
    assert raw.scores[0].value == 12.5
    assert probability_distribution().kind is ScoreKind.UNCALIBRATED_PROBABILITY


def test_probability_distribution_must_be_complete_bounded_and_normalized():
    with pytest.raises(ValidationError, match="sum to one"):
        ScoreDistribution(
            distribution_id="bad",
            head_id="retain",
            kind=ScoreKind.UNCALIBRATED_PROBABILITY,
            complete=True,
            scores=(
                CandidateScore(candidate_id="candidate-a", value=0.8),
                CandidateScore(candidate_id="candidate-b", value=0.3),
            ),
            calibration=None,
        )


def test_calibrated_probability_requires_versioned_claim_and_other_scores_forbid_it():
    with pytest.raises(ValidationError, match="requires a calibration claim"):
        ScoreDistribution(
            distribution_id="calibrated",
            head_id="retain",
            kind=ScoreKind.CALIBRATED_PROBABILITY,
            complete=True,
            scores=(CandidateScore(candidate_id="candidate-a", value=1.0),),
            calibration=None,
        )
    with pytest.raises(ValidationError, match="only calibrated"):
        ScoreDistribution(
            distribution_id="raw",
            head_id="retain",
            kind=ScoreKind.RAW_SCORE,
            complete=False,
            scores=(CandidateScore(candidate_id="candidate-a", value=1.0),),
            calibration=calibration(),
        )
    assert probability_distribution(kind=ScoreKind.CALIBRATED_PROBABILITY).calibration


def test_batch_rejects_duplicate_distribution_ids():
    with pytest.raises(ValidationError, match="distribution IDs must be unique"):
        DecisionBatchResult(
            schema_version=VERSION,
            request_id="request-1",
            fallback_granularity=FallbackGranularity.WHOLE_BATCH,
            publishable=True,
            head_results=(
                selected(distribution=probability_distribution()),
                selected(
                    head_id="category",
                    distribution=probability_distribution(
                        head_id="category", distribution_id="dist-1"
                    ),
                ),
            ),
            attempts=(),
        )


def test_results_reject_duplicate_attempt_ids():
    duplicates = (usage(), usage())
    with pytest.raises(ValidationError, match="attempt IDs must be unique"):
        DecisionResult(
            schema_version=VERSION,
            request_id="request-1",
            head_result=selected(),
            attempts=duplicates,
        )
    with pytest.raises(ValidationError, match="attempt IDs must be unique"):
        generation_result(attempts=duplicates)


def test_resource_overrun_is_reported_without_deleting_late_attempt_evidence():
    request = decision_request(
        resource_limits=limits(
            max_attempts=1,
            max_total_input_utf8_bytes=5,
            max_total_output_utf8_bytes=2,
        )
    )
    attempts = (
        usage(input_bytes=4, output_bytes=2),
        usage("attempt-2", input_bytes=7, output_bytes=3, late=True),
    )
    result = DecisionResult(
        schema_version=VERSION,
        request_id=request.request_id,
        head_result=selected(),
        attempts=attempts,
    )
    compliance = validate_decision_result(request, result)
    assert not compliance.compliant
    assert compliance.violations == (
        ResourceViolation.MAX_ATTEMPTS,
        ResourceViolation.MAX_TOTAL_INPUT_UTF8_BYTES,
        ResourceViolation.MAX_TOTAL_OUTPUT_UTF8_BYTES,
    )
    assert compliance.total_input_utf8_bytes == 11
    assert result.attempts[1].late is True


def test_generation_output_matches_declared_schema_and_source_support():
    request = generation_request()
    result = generation_result()
    assert validate_generation_result(request, result).compliant

    wrong_type = generation_result(output="not-an-object")
    with pytest.raises(ValueError, match="does not match declared schema"):
        validate_generation_result(request, wrong_type)
    missing_required = generation_result(output={})
    with pytest.raises(ValueError, match="required property"):
        validate_generation_result(request, missing_required)


def test_generation_schema_validation_stops_after_first_error(monkeypatch):
    request = generation_request()
    result = generation_result(output={})

    class LazyValidator:
        consumed = 0

        def iter_errors(self, output):
            self.consumed += 1
            yield type("SchemaError", (), {"message": "first schema failure"})()
            self.consumed += 1
            raise AssertionError("schema validation consumed a later error")

    validator = LazyValidator()
    monkeypatch.setattr(
        contracts_inference,
        "_build_output_validator",
        lambda schema: validator,
    )

    with pytest.raises(ValueError, match="first schema failure"):
        validate_generation_result(request, result)
    assert validator.consumed == 1


def test_generation_schema_rejects_unknown_keywords_versions_formats_and_references():
    cases = (
        (output_schema(unknownKeyword=True), "unsupported output schema keyword"),
        (output_schema(**{"$schema": "https://example.invalid/schema"}), "supported Draft"),
        (output_schema(format="private-format"), "unsupported output schema format"),
        (output_schema(**{"$ref": "https://example.invalid/schema"}), "reference semantics"),
        (output_schema(**{"$defs": {"value": {"type": "string"}}}), "reference semantics"),
    )
    for schema, message in cases:
        with pytest.raises((ValidationError, ValueError), match=message):
            generation_request(schema=schema)


@pytest.mark.parametrize(
    "nested_schema",
    [
        output_schema(
            properties={
                "summary": {
                    "$schema": "https://example.invalid/nested",
                    "type": "string",
                }
            }
        ),
        output_schema(
            properties={"summary": {"$schema": None, "type": "string"}}
        ),
        output_schema(
            **{
                "if": {
                    "$schema": "https://example.invalid/nested",
                    "type": "object",
                }
            }
        ),
        output_schema(allOf=[{"$schema": "https://example.invalid/nested"}]),
    ],
)
def test_generation_rejects_unsupported_dialect_in_nested_schema_nodes(
    nested_schema,
):
    with pytest.raises((ValidationError, ValueError), match="unsupported dialect"):
        generation_request(schema=nested_schema)


def test_generation_does_not_execute_schema_like_annotation_payloads():
    schema = output_schema(
        default={
            "$schema": "https://example.invalid/annotation-data",
            "unknownKeyword": True,
        },
        examples=[
            {
                "$schema": "https://example.invalid/annotation-data",
                "$ref": "https://example.invalid/not-a-schema-reference",
            }
        ],
    )

    request = generation_request(schema=schema)

    assert request.output_schema["default"]["unknownKeyword"] is True
    assert request.output_schema["examples"][0]["$ref"].endswith(
        "not-a-schema-reference"
    )


@pytest.mark.parametrize("nonfinite", [math.nan, math.inf, -math.inf])
def test_generation_rejects_nonfinite_numbers_before_json_serialization(nonfinite):
    with pytest.raises(ValidationError, match="non-finite JSON number"):
        GenerationProposal(
            proposal_id="bad-number",
            output={"nested": [nonfinite]},
            support_span_ids=("support-1",),
            machine_authored=True,
        )
    schema = output_schema(default=nonfinite)
    with pytest.raises((ValidationError, ValueError), match="non-finite JSON number"):
        generation_request(schema=schema)


def test_generation_rejects_unknown_support_span_and_duplicate_proposals():
    request = generation_request()
    result = generation_result()
    unsupported = result.model_copy(
        update={
            "proposals": (
                result.proposals[0].model_copy(update={"support_span_ids": ("invented",)}),
            )
        }
    )
    with pytest.raises(ValueError, match="unknown support span"):
        validate_generation_result(request, unsupported)

    with pytest.raises(ValidationError, match="proposal IDs must be unique"):
        GenerationResult(
            schema_version=VERSION,
            request_id="generation-1",
            outcome=GenerationOutcome.PROPOSED,
            proposals=(result.proposals[0], result.proposals[0]),
            reason_code=None,
            attempts=(),
        )


def test_valid_empty_remains_distinct_from_abstention():
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


def test_decision_boundaries_revalidate_requests_results_and_nested_models():
    request = decision_request()
    result = DecisionResult(
        schema_version=VERSION,
        request_id=request.request_id,
        head_result=selected(distribution=probability_distribution()),
        attempts=(),
    )
    assert validate_decision_result(request, result).compliant

    request.sources[0].text = "A"
    with pytest.raises(ValidationError, match="exceeds"):
        validate_decision_result(request, result)

    request = decision_request()
    result.head_result.distribution.scores[0].value = 9.0
    with pytest.raises(ValidationError, match="between zero and one|sum to one"):
        validate_decision_result(request, result)


def test_batch_boundary_rejects_unqualified_copy_and_duplicate_diagnostics():
    request = decision_request(
        batch=True,
        fallback=FallbackGranularity.PER_ITEM,
        per_item_qualification=qualification(),
    )
    result = DecisionBatchResult(
        schema_version=VERSION,
        request_id=request.request_id,
        fallback_granularity=FallbackGranularity.PER_ITEM,
        publishable=True,
        head_results=(
            selected(distribution=probability_distribution()),
            selected(
                head_id="category",
                distribution=probability_distribution(
                    head_id="category", distribution_id="dist-2"
                ),
            ),
        ),
        attempts=(),
    )

    unqualified = request.model_copy(update={"per_item_qualification": None})
    with pytest.raises(ValidationError, match="requires a qualification claim"):
        validate_decision_batch_result(unqualified, result)

    result.head_results[1].distribution.distribution_id = "dist-1"
    with pytest.raises(ValidationError, match="distribution IDs must be unique"):
        validate_decision_batch_result(request, result)

    copied_duplicate = result.model_copy(update={"head_results": result.head_results})
    with pytest.raises(ValidationError, match="distribution IDs must be unique"):
        validate_decision_batch_result(request, copied_duplicate)


def test_generation_boundary_revalidates_outcome_and_nonfinite_output():
    request = generation_request()
    valid = generation_result()
    reparsed = GenerationResult.model_validate_json(valid.model_dump_json())
    assert validate_generation_result(request, reparsed).compliant

    result = generation_result()
    result.outcome = GenerationOutcome.ERROR
    result.reason_code = "provider-error"
    with pytest.raises(ValidationError, match="only a proposed outcome"):
        validate_generation_result(request, result)

    mutated = generation_result()
    mutated.proposals[0].output = {"summary": math.nan}
    with pytest.raises(ValidationError, match="non-finite JSON number"):
        validate_generation_result(request, mutated)

    for nonfinite in (math.nan, math.inf, -math.inf):
        copied_proposal = generation_result().proposals[0].model_copy(
            update={"output": {"summary": [nonfinite]}}
        )
        copied_result = generation_result().model_copy(
            update={"proposals": (copied_proposal,)}
        )
        with pytest.raises(ValidationError, match="non-finite JSON number"):
            validate_generation_result(request, copied_result)


def test_receiving_boundaries_preserve_container_and_mapping_types():
    request = decision_request()
    list_candidates = selected().model_copy(
        update={"selected_candidate_ids": ["candidate-a"]}
    )
    list_result = DecisionResult(
        schema_version=VERSION,
        request_id=request.request_id,
        head_result=selected(),
        attempts=(),
    ).model_copy(update={"head_result": list_candidates})
    with pytest.raises(ValidationError):
        validate_decision_result(request, list_result)

    generation = generation_result()
    numeric_key_proposal = generation.proposals[0].model_copy(
        update={"output": {1: "not-a-JSON-object-key"}}
    )
    numeric_key_result = generation.model_copy(
        update={"proposals": (numeric_key_proposal,)}
    )
    with pytest.raises(ValidationError):
        validate_generation_result(generation_request(), numeric_key_result)


def test_receiving_boundaries_reject_unknown_fields_constructed_values_and_cycles():
    request = decision_request()
    unknown_result = DecisionResult(
        schema_version=VERSION,
        request_id=request.request_id,
        head_result=selected(),
        attempts=(),
    ).model_copy(update={"future_semantics": "deny"})
    with pytest.raises(ValueError, match="undeclared contract field"):
        validate_decision_result(request, unknown_result)

    constructed = DecisionResult.model_construct(
        schema_version=VERSION,
        request_id=request.request_id,
        head_result=selected().model_copy(update={"outcome": "invented"}),
        attempts=(),
    )
    with pytest.raises(ValidationError):
        validate_decision_result(request, constructed)

    cycle = []
    cycle.append(cycle)
    cyclic = generation_result().model_copy(update={"proposals": cycle})
    with pytest.raises(ValueError, match="cyclic contract graph"):
        validate_generation_result(generation_request(), cyclic)


def test_source_span_boundary_revalidates_mutated_inputs():
    evidence = source_evidence()
    span = SourceSpan(
        span_id="span-1",
        source_version=evidence.source_version,
        start_utf8_byte=0,
        end_utf8_byte=8,
    )
    evidence.text = "A"
    with pytest.raises(ValueError, match="exceeds"):
        validate_source_span(evidence, span)
