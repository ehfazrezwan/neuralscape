"""Identity, fallback, schema, score, and resource inference invariants."""

import math

import pytest
from pydantic import BaseModel, ValidationError

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


class FalseyExtraDict(dict):
    def __bool__(self):
        return False


class HiddenExtraKeysDict(dict):
    def __iter__(self):
        return iter(())


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


def test_request_encodes_each_source_once_without_boundary_set(monkeypatch):
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
    original = contracts_inference._source_utf8_bytes

    def counted_encoding(evidence):
        record_id = evidence.source_version.record_id
        calls[record_id] = calls.get(record_id, 0) + 1
        return original(evidence)

    def forbidden_boundaries(self):
        raise AssertionError("span validation allocated a boundary set")

    monkeypatch.setattr(contracts_inference, "_source_utf8_bytes", counted_encoding)
    monkeypatch.setattr(SourceEvidence, "utf8_boundaries", forbidden_boundaries)

    request = decision_request(sources=(first, second), spans=spans)
    assert tuple(span.span_id for span in request.spans) == (
        "span-1",
        "span-2",
        "span-3",
    )
    assert calls == {"source-1": 1, "source-2": 1}

    calls.clear()
    validate_source_span(first, spans[0])
    assert calls == {"source-1": 1}


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


def test_generation_boundary_rejects_models_inside_json_values():
    class SchemaShapedModel(BaseModel):
        summary: str

    model_value = SchemaShapedModel(summary="must-not-normalize")
    base_result = generation_result()
    declared_proposals = (
        base_result.proposals[0].model_copy(update={"output": model_value}),
        base_result.proposals[0].model_copy(
            update={"output": {"summary": [model_value]}}
        ),
    )
    native_proposals = (
        {
            "proposal_id": "proposal-1",
            "output": model_value,
            "support_span_ids": ("support-1",),
            "machine_authored": True,
        },
        {
            "proposal_id": "proposal-1",
            "output": {"summary": [model_value]},
            "support_span_ids": ("support-1",),
            "machine_authored": True,
        },
    )
    for proposal in (*declared_proposals, *native_proposals):
        result = base_result.model_copy(update={"proposals": (proposal,)})
        with pytest.raises(ValueError, match="JSON value contains a model object"):
            validate_generation_result(generation_request(), result)

    request = generation_request()
    request.output_schema["properties"]["summary"]["default"] = {
        "nested": [model_value]
    }
    with pytest.raises(ValueError, match="JSON value contains a model object"):
        validate_generation_result(request, generation_result())

    ordinary_mapping = {
        "proposal_id": "proposal-1",
        "output": {"summary": "ordinary JSON"},
        "support_span_ids": ("support-1",),
        "machine_authored": True,
    }
    mapping_result = base_result.model_copy(
        update={"proposals": (ordinary_mapping,)}
    )
    assert validate_generation_result(generation_request(), mapping_result).compliant

    unknown_mapping = {**ordinary_mapping, "future_semantics": "deny"}
    unknown_result = base_result.model_copy(
        update={"proposals": (unknown_mapping,)}
    )
    with pytest.raises(ValidationError, match="extra"):
        validate_generation_result(generation_request(), unknown_result)

    cycle = []
    cycle.append(cycle)
    cyclic_mapping = {**ordinary_mapping, "output": cycle}
    cyclic_result = base_result.model_copy(
        update={"proposals": (cyclic_mapping,)}
    )
    with pytest.raises(ValueError, match="cyclic contract graph"):
        validate_generation_result(generation_request(), cyclic_result)

    list_result = base_result.model_copy(update={"proposals": [ordinary_mapping]})
    with pytest.raises(ValidationError, match="tuple"):
        validate_generation_result(generation_request(), list_result)


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


@pytest.mark.parametrize("regex_keyword", ["pattern", "patternProperties"])
@pytest.mark.parametrize(
    ("schema_position", "parent_keyword"),
    [
        ("root", None),
        ("container", "properties"),
        ("container", "dependentSchemas"),
        ("single", "additionalProperties"),
        ("single", "contains"),
        ("single", "if"),
        ("single", "items"),
        ("single", "not"),
        ("single", "propertyNames"),
        ("single", "unevaluatedItems"),
        ("single", "unevaluatedProperties"),
        ("array", "allOf"),
        ("array", "anyOf"),
        ("array", "oneOf"),
        ("array", "prefixItems"),
    ],
)
def test_generation_rejects_executable_regex_at_schema_positions(
    regex_keyword,
    schema_position,
    parent_keyword,
):
    regex_schema = (
        {"pattern": "^bounded$"}
        if regex_keyword == "pattern"
        else {"patternProperties": {"^bounded$": {"type": "string"}}}
    )
    schema = output_schema()
    if schema_position == "root":
        schema.update(regex_schema)
    elif schema_position == "container":
        schema[parent_keyword] = {"nested": regex_schema}
    elif schema_position == "single":
        schema[parent_keyword] = regex_schema
    else:
        schema[parent_keyword] = [regex_schema]

    with pytest.raises(
        (ValidationError, ValueError),
        match=f"unsupported executable schema regex.*{regex_keyword}",
    ):
        generation_request(schema=schema)


def test_generation_allows_incidental_regex_keyword_names_in_json_data():
    schema = output_schema(
        properties={
            "summary": {"type": "string"},
            "pattern": {"type": "string"},
            "patternProperties": {"type": "string"},
        },
        default={
            "pattern": "annotation data",
            "nested": {"patternProperties": "annotation data"},
        },
        examples=[
            {
                "pattern": "example data",
                "patternProperties": {"not": "a schema here"},
            }
        ],
    )
    request = generation_request(schema=schema)
    result = generation_result(
        output={
            "summary": "supported",
            "pattern": "ordinary property",
            "patternProperties": "ordinary property",
        }
    )

    assert validate_generation_result(request, result).compliant


@pytest.mark.parametrize("regex_keyword", ["pattern", "patternProperties"])
def test_generation_mutated_request_rejects_regex_before_schema_execution(
    monkeypatch,
    regex_keyword,
):
    request = generation_request()
    regex_value = (
        "^bounded$"
        if regex_keyword == "pattern"
        else {"^bounded$": {"type": "string"}}
    )
    request.output_schema["properties"]["summary"][regex_keyword] = regex_value
    result = generation_result()
    validator_class, format_checker_class = contracts_inference._jsonschema_types()

    class ExecutionSentinel:
        VALIDATORS = validator_class.VALIDATORS

        @classmethod
        def check_schema(cls, schema):
            raise AssertionError("unsafe schema reached check_schema")

        def __init__(self, schema, format_checker):
            raise AssertionError("unsafe schema reached validator construction")

    monkeypatch.setattr(
        contracts_inference,
        "_jsonschema_types",
        lambda: (ExecutionSentinel, format_checker_class),
    )

    with pytest.raises(
        (ValidationError, ValueError),
        match=f"unsupported executable schema regex.*{regex_keyword}",
    ):
        validate_generation_result(request, result)


@pytest.mark.parametrize(
    ("format_name", "valid_value", "invalid_value"),
    [
        ("date", "2024-02-29", "2023-02-29"),
        ("email", "@", "missing-at"),
        ("idn-email", "δοκιμή@παράδειγμα", "missing-at"),
        ("idn-hostname", "bücher.example", "bad host"),
        ("ipv4", "192.0.2.1", "999.0.2.1"),
        ("ipv6", "2001:db8::1", "fe80::1%eth0"),
        ("regex", "^[a-z]+$", "["),
        ("time", "01:02:03Z", "1:02:03"),
        (
            "uuid",
            "123e4567-e89b-12d3-a456-426614174000",
            "123e4567e89b12d3a456426614174000",
        ),
    ],
)
def test_generation_uses_fixed_contract_owned_format_checkers(
    monkeypatch,
    format_name,
    valid_value,
    invalid_value,
):
    assert contracts_inference._SUPPORTED_OUTPUT_SCHEMA_FORMATS == frozenset(
        {
            "date",
            "email",
            "idn-email",
            "idn-hostname",
            "ipv4",
            "ipv6",
            "regex",
            "time",
            "uuid",
        }
    )
    _, format_checker_class = contracts_inference._jsonschema_types()
    monkeypatch.setitem(
        format_checker_class.checkers,
        format_name,
        (lambda value: True, ()),
    )
    monkeypatch.setitem(
        format_checker_class.checkers,
        "private-format",
        (lambda value: True, ()),
    )

    schema = output_schema(
        properties={"address": {"type": "string", "format": format_name}},
        required=["address"],
    )
    request = generation_request(schema=schema)
    assert validate_generation_result(
        request,
        generation_result(output={"address": valid_value}),
    ).compliant
    with pytest.raises(ValueError, match="does not match declared schema"):
        validate_generation_result(
            request,
            generation_result(output={"address": invalid_value}),
        )

    with pytest.raises(ValueError, match="unsupported output schema format"):
        generation_request(schema=output_schema(format="private-format"))


@pytest.mark.parametrize("compiler_exception", [RecursionError, OverflowError])
def test_generation_regex_format_maps_compiler_limits_to_schema_mismatch(
    monkeypatch,
    compiler_exception,
):
    def compiler_limited(value):
        raise compiler_exception("synthetic compiler limit")

    monkeypatch.setattr(contracts_inference, "_is_regex", compiler_limited)
    schema = output_schema(
        properties={"expression": {"type": "string", "format": "regex"}},
        required=["expression"],
    )
    request = generation_request(schema=schema)

    with pytest.raises(ValueError, match="does not match declared schema"):
        validate_generation_result(
            request,
            generation_result(output={"expression": "bounded sentinel"}),
        )


@pytest.mark.parametrize(
    ("value", "valid"),
    [
        ("08:30:06Z", True),
        ("08:30:06z", True),
        ("08:30:06.123456Z", True),
        ("01:02:03+05:30", True),
        ("12:34:56-00:00", True),
        ("23:59:60Z", True),
        ("01:29:60+01:30", True),
        ("15:59:60-08:00", True),
        ("23:29:60+23:30", True),
        ("1:02:03Z", False),
        ("01:02:03", False),
        ("008:030:006Z", False),
        ("24:00:00Z", False),
        ("00:60:00Z", False),
        ("00:00:61Z", False),
        ("22:59:60Z", False),
        ("23:58:60Z", False),
        ("23:59:60+01:00", False),
        ("00:29:60+01:30", False),
        ("01:02:03+24:00", False),
        ("01:02:03+00:60", False),
        ("01:02:03Z+00:30", False),
        ("01:02:03,1Z", False),
        ("01:02:03Z\n", False),
        ("1২:00:00Z", False),
    ],
)
def test_generation_time_format_uses_rfc3339_full_time(value, valid):
    schema = output_schema(
        properties={"time": {"type": "string", "format": "time"}},
        required=["time"],
    )
    request = generation_request(schema=schema)
    result = generation_result(output={"time": value})

    if valid:
        assert validate_generation_result(request, result).compliant
    else:
        with pytest.raises(ValueError, match="does not match declared schema"):
            validate_generation_result(request, result)


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


@pytest.mark.parametrize(
    "json_case",
    ["schema-value", "schema-key", "output-value", "output-key"],
)
def test_generation_rejects_surrogate_json_at_construction(json_case):
    surrogate = "\ud800"
    with pytest.raises(ValidationError, match="invalid Unicode JSON string"):
        if json_case == "schema-value":
            generation_request(schema=output_schema(description=surrogate))
        elif json_case == "schema-key":
            generation_request(schema=output_schema(default={surrogate: "value"}))
        elif json_case == "output-value":
            generation_result(output={"summary": surrogate})
        else:
            generation_result(output={"summary": "supported", surrogate: "value"})


@pytest.mark.parametrize("json_location", ["value", "key"])
@pytest.mark.parametrize(
    "boundary",
    ["request-copy", "request-construct", "proposal-copy", "proposal-construct"],
)
def test_generation_receiving_boundary_rejects_surrogate_json(
    boundary,
    json_location,
):
    surrogate = "\ud800"
    request = generation_request()
    result = generation_result()

    if boundary.startswith("request"):
        schema = (
            output_schema(description=surrogate)
            if json_location == "value"
            else output_schema(default={surrogate: "value"})
        )
        if boundary == "request-copy":
            request = request.model_copy(update={"output_schema": schema})
        else:
            fields = {
                name: getattr(request, name)
                for name in GenerationRequest.model_fields
            }
            fields["output_schema"] = schema
            request = GenerationRequest.model_construct(**fields)
    else:
        output = (
            {"summary": surrogate}
            if json_location == "value"
            else {"summary": "supported", surrogate: "value"}
        )
        proposal = result.proposals[0]
        if boundary == "proposal-copy":
            proposal = proposal.model_copy(update={"output": output})
        else:
            fields = {
                name: getattr(proposal, name)
                for name in GenerationProposal.model_fields
            }
            fields["output"] = output
            proposal = GenerationProposal.model_construct(**fields)
        result = result.model_copy(update={"proposals": (proposal,)})

    with pytest.raises(ValidationError, match="invalid Unicode JSON string"):
        validate_generation_result(request, result)


def test_generation_receiving_boundary_retains_shared_opaque_id_rejection():
    request = generation_request().model_copy(update={"request_id": "\ud800"})

    with pytest.raises(ValidationError, match="string_unicode"):
        validate_generation_result(request, generation_result())


def test_generation_preserves_valid_unicode_json_values_and_keys():
    schema = output_schema(
        description="বাংলা 😀",
        properties={
            "summary": {"type": "string"},
            "ключ": {"type": "string"},
        },
        default={"鍵": "值"},
    )
    request = generation_request(schema=schema)
    result = generation_result(output={"summary": "مرحبا 😀", "ключ": "值"})

    assert validate_generation_result(request, result).compliant
    assert GenerationRequest.model_validate_json(request.model_dump_json()) == request
    assert GenerationResult.model_validate_json(result.model_dump_json()) == result


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


@pytest.mark.parametrize("extra_value", ["span-1", "shadow-value"])
@pytest.mark.parametrize("target_name", ["span", "nested_source_version"])
def test_inference_boundary_rejects_declared_fields_duplicated_in_extras(
    target_name,
    extra_value,
):
    evidence = source_evidence()
    span = SourceSpan(
        span_id="span-1",
        source_version=evidence.source_version,
        start_utf8_byte=0,
        end_utf8_byte=3,
    )
    if target_name == "span":
        target = span
        field_name = "span_id"
    else:
        target = span.source_version
        field_name = "record_id"
    object.__setattr__(target, "__pydantic_extra__", {field_name: extra_value})

    with pytest.raises(
        ValueError,
        match=rf"malformed contract extras.*duplicate field\(s\): {field_name}",
    ):
        validate_source_span(evidence, span)


@pytest.mark.parametrize(
    "extras_type",
    [FalseyExtraDict, HiddenExtraKeysDict],
    ids=["falsey-dict-subclass", "hidden-keys-dict-subclass"],
)
@pytest.mark.parametrize(
    "payload",
    [
        {"record_id": "shadow-record"},
        {"future_semantics": "deny"},
    ],
    ids=["declared-field", "unknown-field"],
)
def test_inference_boundary_rejects_nested_dict_subclass_extra_storage(
    extras_type,
    payload,
):
    evidence = source_evidence()
    span = SourceSpan(
        span_id="span-1",
        source_version=evidence.source_version,
        start_utf8_byte=0,
        end_utf8_byte=3,
    )
    object.__setattr__(
        span.source_version,
        "__pydantic_extra__",
        extras_type(payload),
    )

    with pytest.raises(ValueError, match="malformed contract extras"):
        validate_source_span(evidence, span)


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


# ---- one capture of every root before any reconstruction (request AND result) ----

BUDGET = "contract graph exceeds the native discovery budget"
_NATIVE = BaseModel.__dict__["__dict__"]


def _native_set(model, field, value):
    dict.__setitem__(_NATIVE.__get__(model, BaseModel), field, value)


class _Firing(tuple):
    """Public iteration first runs a callback (a hostile or buggy view), then yields its items."""

    def __new__(cls, items, fire):
        instance = super().__new__(cls, items)
        instance.fire, instance.fired = fire, 0
        return instance

    def __iter__(self):
        if not self.fired:
            self.fired = 1
            self.fire()
        return tuple.__iter__(self)


class _FiringList(list):
    def __init__(self, items, fire):
        super().__init__(items)
        self.fire, self.fired = fire, 0

    def __iter__(self):
        if not self.fired:
            self.fired = 1
            self.fire()
        return list.__iter__(self)


def test_a_candidates_view_cannot_repair_an_invalid_result_before_it_is_captured():
    request = decision_request()
    result = DecisionResult(schema_version=VERSION, request_id=request.request_id, head_result=selected(candidate_id="candidate-z"), attempts=())
    valid = selected(candidate_id="candidate-a")
    _native_set(request, "candidates", _Firing(request.candidates, lambda: _native_set(result, "head_result", valid)))
    with pytest.raises(ValueError, match="unknown candidate"):
        validate_decision_result(request, result)  # used to accept the repaired selection


def test_a_heads_view_cannot_change_batch_publishability_before_it_is_captured():
    request = decision_request(batch=True)
    result = DecisionBatchResult(
        schema_version=VERSION, request_id=request.request_id, fallback_granularity=FallbackGranularity.WHOLE_BATCH,
        publishable=True, head_results=(selected(), abstained("category")), attempts=(),
    )
    _native_set(request, "heads", _Firing(request.heads, lambda: _native_set(result, "publishable", False)))
    with pytest.raises(ValueError, match="forbids partial publication"):
        validate_decision_batch_result(request, result)


def test_the_decision_boundaries_use_one_budget_for_request_and_result(monkeypatch):
    import contracts_limits

    created = []

    def factory(*args, **kwargs):
        budget = contracts_limits.DiscoveryBudget(*args, **kwargs)
        created.append(budget)
        return budget

    import contracts_engines

    monkeypatch.setattr(contracts_engines, "DiscoveryBudget", factory)
    request = decision_request()
    validate_decision_result(request, DecisionResult(schema_version=VERSION, request_id=request.request_id, head_result=selected(), attempts=()))
    batch = decision_request(batch=True)
    validate_decision_batch_result(batch, DecisionBatchResult(
        schema_version=VERSION, request_id=batch.request_id, fallback_granularity=FallbackGranularity.WHOLE_BATCH,
        publishable=True, head_results=(selected(), selected("category")), attempts=()))
    assert len(created) == 2  # one shared session per validation call (was two budgets for each)


# ---- generation: freeze and account for both roots before any JSON scan ----


def test_a_schema_view_cannot_repair_an_invalid_result_before_it_is_captured():
    request = generation_request()
    result = generation_result()
    bad = result.proposals[0].model_copy(update={"support_span_ids": ("unknown-span",)})
    _native_set(result, "proposals", (bad,))
    good = (result.proposals[0].model_copy(update={"support_span_ids": ("support-1",)}),)
    _native_set(request, "output_schema",
                {**request.output_schema, "examples": _FiringList([1], lambda: _native_set(result, "proposals", good))})
    with pytest.raises(ValueError, match="unknown support span"):
        validate_generation_result(request, result)  # used to accept: the pre-scan ran the callback before the result was frozen


class _Bulk(list):
    yielded = 0

    def __init__(self, count):
        super().__init__()
        self.count = count

    def __iter__(self):
        for index in range(self.count):
            type(self).yielded += 1
            yield index


def test_an_oversized_view_in_the_schema_is_cut_off_by_the_budget_not_expanded_by_a_pre_scan():
    _Bulk.yielded = 0
    request = generation_request()
    _native_set(request, "output_schema", {**request.output_schema, "examples": _Bulk(400_000)})
    with pytest.raises(ValueError, match=BUDGET):
        validate_generation_result(request, generation_result())
    assert _Bulk.yielded <= 140_000  # the edge tier is 131,072; it used to be expanded completely first


def test_an_oversized_model_copy_list_is_rejected_by_the_budget():
    request = generation_request()
    huge = request.model_copy(update={"output_schema": {**request.output_schema, "examples": list(range(200_000))}})
    with pytest.raises(ValueError, match=BUDGET):
        validate_generation_result(huge, generation_result())


def test_model_objects_inside_json_values_are_still_rejected_in_the_same_replay():
    class Marker(BaseModel):
        value: int = 1

    request = generation_request()
    bad_schema = {**request.output_schema, "examples": [{"nested": [Marker()]}]}
    with pytest.raises(ValueError, match="JSON value contains a model object"):
        validate_generation_result(request.model_copy(update={"output_schema": bad_schema}), generation_result())
    result = generation_result()
    for proposal in (
        {"proposal_id": "proposal-1", "output": {"summary": [Marker()]}, "support_span_ids": ("support-1",), "machine_authored": True},
    ):
        with pytest.raises(ValueError, match="JSON value contains a model object"):
            validate_generation_result(request, result.model_copy(update={"proposals": (proposal,)}))


# ── every root is captured before any root's public view can run ────────────


class _RepairingSourceVersion(dict):
    """A dict-subclass ``source_version`` whose ``items()`` runs a callback (a public-view callback)."""

    def __init__(self, backing, repair):
        super().__init__(backing)
        self._repair = repair

    def items(self):  # noqa: D102
        self._repair()
        return dict.items(self)


def test_a_span_is_captured_before_the_evidence_public_view_can_repair_it():
    evidence = source_evidence()
    out_of_bounds = SourceSpan(
        span_id="span-1", source_version=evidence.source_version, start_utf8_byte=0, end_utf8_byte=2
    ).model_copy(update={"end_utf8_byte": 9999})  # beyond the 8 UTF-8 bytes of the text

    def repair() -> None:
        out_of_bounds.__dict__["end_utf8_byte"] = 8  # a callback run while evidence is reconstructed

    hostile = evidence.model_copy(
        update={"source_version": _RepairingSourceVersion(evidence.source_version.model_dump(), repair)}
    )
    with pytest.raises(ValueError, match="exceeds"):
        validate_source_span(hostile, out_of_bounds)
    assert out_of_bounds.end_utf8_byte == 8  # the callback did run: the old order would have accepted the repair


def test_the_type_check_of_both_roots_precedes_any_capture():
    calls = []
    evidence = source_evidence()
    hostile = evidence.model_copy(
        update={"source_version": _RepairingSourceVersion(evidence.source_version.model_dump(), lambda: calls.append(1))}
    )
    with pytest.raises(TypeError, match="source span must be a SourceSpan"):
        validate_source_span(hostile, "not-a-span")
    assert calls == []  # no public view ran before the strict top-level type checks


def test_an_honest_span_still_validates_after_the_single_capture():
    evidence = source_evidence()
    span = SourceSpan(span_id="span-1", source_version=evidence.source_version, start_utf8_byte=1, end_utf8_byte=3)
    assert validate_source_span(evidence, span) is None


# ── a str-subclass key is the same key to a dict lookup, so it is the same field to the guard ──


class _OutputKey(str):
    """Equal to, and hashed like, ``"output"`` — an ordinary ``str`` subclass."""


def test_a_str_subclass_output_key_does_not_hide_a_model_from_the_json_guard():
    class SchemaShapedModel(BaseModel):
        summary: str

    model_value = SchemaShapedModel(summary="must-not-normalize")
    base_result = generation_result()
    for key in (_OutputKey("output"), "output"):
        proposal = {
            "proposal_id": "proposal-1",
            key: model_value,
            "support_span_ids": ("support-1",),
            "machine_authored": True,
        }
        result = base_result.model_copy(update={"proposals": (proposal,)})
        with pytest.raises(ValueError, match="JSON value contains a model object"):
            validate_generation_result(generation_request(), result)


def test_a_str_subclass_key_in_ordinary_json_is_normalized_not_rejected():
    result = generation_result()
    proposal = {
        _OutputKey("proposal_id"): "proposal-1",
        _OutputKey("output"): {"summary": "ordinary JSON"},
        "support_span_ids": ("support-1",),
        "machine_authored": True,
    }
    updated = result.model_copy(update={"proposals": (proposal,)})
    assert validate_generation_result(generation_request(), updated).compliant


def test_colliding_str_subclass_keys_are_rejected():
    class Colliding(str):
        def __hash__(self):  # distinct hash: both keys can live in one dict
            return hash("x" + str(self))

        def __eq__(self, other):
            return False

    proposal = {
        "proposal_id": "proposal-1",
        "output": {"summary": "plain"},
        Colliding("output"): {"summary": "second"},
        "support_span_ids": ("support-1",),
        "machine_authored": True,
    }
    result = generation_result().model_copy(update={"proposals": (proposal,)})
    with pytest.raises(ValueError):
        validate_generation_result(generation_request(), result)


# ── a class that only registers itself as a Mapping is not one ──────────────


class _RegisteredOnly:
    """Registered with the Mapping ABC but with none of its methods."""


def test_a_registered_only_mapping_in_a_json_value_is_a_value_error_not_an_attribute_error():
    from collections.abc import Mapping

    Mapping.register(_RegisteredOnly)
    request = generation_request()
    proposal = {
        "proposal_id": "proposal-1", "output": _RegisteredOnly(),
        "support_span_ids": ("support-1",), "machine_authored": True,
    }
    result = generation_result().model_copy(update={"proposals": (proposal,)})
    with pytest.raises(ValueError):  # a stable class: ValidationError or ValueError, never AttributeError
        validate_generation_result(request, result)
    schema = {**request.output_schema, "examples": [_RegisteredOnly()]}
    with pytest.raises(ValueError):
        validate_generation_result(request.model_copy(update={"output_schema": schema}), generation_result())


def test_a_real_non_dict_mapping_inside_a_json_value_is_still_scanned_for_models():
    from types import MappingProxyType

    class SchemaShaped(BaseModel):
        summary: str

    proposal = {
        "proposal_id": "proposal-1", "output": MappingProxyType({"summary": SchemaShaped(summary="m")}),
        "support_span_ids": ("support-1",), "machine_authored": True,
    }
    result = generation_result().model_copy(update={"proposals": (proposal,)})
    with pytest.raises(ValueError, match="JSON value contains a model object"):
        validate_generation_result(generation_request(), result)


def test_the_engine_and_inference_contracts_never_read_the_clock():
    import ast
    from pathlib import Path

    import contracts_engines

    clock_calls = {"now", "today", "utcnow", "time", "monotonic", "perf_counter", "time_ns"}
    clock_modules = {"time", "calendar", "zoneinfo"}
    for module in (contracts_engines, contracts_inference):
        tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert not {a.name.split(".")[0] for a in node.names} & clock_modules, module.__name__
            if isinstance(node, ast.ImportFrom):
                assert (node.module or "").split(".")[0] not in clock_modules, module.__name__
            if isinstance(node, ast.Attribute):
                assert node.attr not in clock_calls, (module.__name__, node.attr)


# ── pinned classes: probes found these already closed ───────────────────────


class _SchemaShapedModel(BaseModel):
    summary: str


def test_a_dict_subclass_view_cannot_repair_a_child_after_exposing_it():
    request = generation_request()
    props = {"summary": {"type": "string", "default": _SchemaShapedModel(summary="m")}}

    class RepairingSchema(dict):
        def items(self):
            yield from list(dict.items(self))
            props["summary"]["default"] = "ok"  # after every entry, and the child, was exposed

    schema = RepairingSchema(
        {"$schema": JSON_SCHEMA_DIALECT, "type": "object", "properties": props,
         "required": ["summary"], "additionalProperties": False}
    )
    with pytest.raises(ValueError, match="JSON value contains a model object"):
        validate_generation_result(request.model_copy(update={"output_schema": schema}), generation_result())


class _AlwaysEqual(str):
    def __eq__(self, other):
        return True

    def __ne__(self, other):
        return False

    __hash__ = str.__hash__


def test_a_str_subclass_value_cannot_satisfy_an_enum_it_does_not_match():
    schema = {
        "$schema": JSON_SCHEMA_DIALECT, "type": "object",
        "properties": {"summary": {"type": "string", "enum": ["allowed-value"]}},
        "required": ["summary"], "additionalProperties": False,
    }
    request = generation_request(schema=schema)
    assert validate_generation_result(request, generation_result(output={"summary": "allowed-value"})).compliant
    with pytest.raises(ValueError):
        validate_generation_result(request, generation_result(output={"summary": _AlwaysEqual("forbidden-value")}))


def test_a_pair_that_is_a_tuple_subclass_with_a_changing_iterator_is_read_once_in_a_dict_view():
    reads = []

    class _Pair(tuple):
        def __iter__(self):
            reads.append(1)
            return iter(("summary", f"value-{len(reads)}"))

    class _PairsView(dict):
        def items(self):
            return iter((_Pair(("ignored", "ignored")),))

    request = generation_request()
    result = generation_result()
    proposal = {
        "proposal_id": "proposal-1", "output": _PairsView(), "support_span_ids": ("support-1",),
        "machine_authored": True,
    }
    updated = result.model_copy(update={"proposals": (proposal,)})
    validate_generation_result(request, updated)  # accepted from the one read, or refused: never two reads
    assert len(reads) <= 1


def test_requirements_and_manifest_are_each_captured_before_the_other_view_runs():
    from contracts_engines import (
        CapabilityHealth,
        CapabilityManifest,
        CapabilityOperationState,
        CapabilityRequirement,
        EngineAdapterKind,
        validate_capability_requirements,
    )
    import contracts_engines

    descriptor = contracts_engines._MODEL_DICT_DESCRIPTOR

    def state(health):
        return CapabilityOperationState(
            operation="search", supported=True, configured=True, health=health, qualification=None
        )

    def requirement():
        return CapabilityRequirement(
            operation="search", contract_schema_version="candidate-v1", require_configured=True,
            require_healthy=True, qualification_profile_reference=None, qualification_profile_version=None,
        )

    class Repairing(tuple):
        def __new__(cls, items, repair):
            instance = super().__new__(cls, items)
            instance.repair = repair
            return instance

        def __iter__(self):
            self.repair()
            return iter(tuple.__iter__(self))

    def manifest_of(operations):
        return CapabilityManifest(
            schema_version="candidate-v1", implementation_id="impl", implementation_version="1",
            adapter_kind=EngineAdapterKind.VECTOR_STORE, contract_schema_versions=("candidate-v1",),
            operations=tuple(operations),
        )

    degraded = manifest_of([state(CapabilityHealth.DEGRADED)])
    stored = descriptor.__get__(degraded.operations[0], BaseModel)
    repairing = Repairing(
        (requirement(),), lambda: dict.__setitem__(stored, "health", CapabilityHealth.HEALTHY)
    )
    assert [v.fact for v in validate_capability_requirements(degraded, repairing)] == ["healthy"]

    strict = requirement()
    stored_requirement = descriptor.__get__(strict, BaseModel)
    manifest = manifest_of([state(CapabilityHealth.DEGRADED)])
    object.__setattr__(
        manifest, "operations",
        Repairing(manifest.operations, lambda: dict.__setitem__(stored_requirement, "require_healthy", False)),
    )
    assert [v.fact for v in validate_capability_requirements(manifest, (strict,))] == ["healthy"]


def test_objects_the_engine_snapshot_is_keyed_by_stay_alive_for_the_call():
    import gc
    import weakref

    import contracts_engines
    from contracts_limits import DiscoveryBudget

    class _Weak(dict):
        pass

    child = _Weak({"k": 1})
    alive = weakref.ref(child)
    holder = [child]
    frozen_models, frozen_containers, discovered = {}, {}, {}
    contracts_engines._freeze_native_model_graph(holder, frozen_models, frozen_containers, discovered, DiscoveryBudget())
    del child, holder
    gc.collect()
    assert alive() is not None  # an id() cannot be recycled while the call's tables hold the object


# ── a reused subtree is shared by the snapshot but walked by path afterwards ──


def _aliased(levels: int):
    value = "x"
    for _ in range(levels):
        value = [value, value]  # two references to one list at every level: 2**levels paths
    return value


def _array_schema():
    return {"$schema": JSON_SCHEMA_DIALECT, "type": ["array", "string"]}


def test_a_proposal_output_that_reuses_a_subtree_exponentially_is_refused_quickly():
    import time

    request = generation_request(schema=_array_schema())
    proposal = {
        "proposal_id": "proposal-1", "output": _aliased(22), "support_span_ids": ("support-1",),
        "machine_authored": True,
    }
    result = generation_result().model_copy(update={"proposals": (proposal,)})
    started = time.monotonic()
    with pytest.raises(ValueError, match="native discovery budget"):
        validate_generation_result(request, result)
    assert time.monotonic() - started < 2.0  # 2**22 paths took about 14 s to walk before


def test_a_declared_output_schema_that_reuses_a_subtree_exponentially_is_refused_quickly():
    import time

    request = generation_request()
    schema = {**request.output_schema, "examples": [_aliased(22)]}
    started = time.monotonic()
    with pytest.raises(ValueError, match="native discovery budget"):
        validate_generation_result(request.model_copy(update={"output_schema": schema}), generation_result())
    assert time.monotonic() - started < 2.0


def test_modest_reuse_and_large_ordinary_json_are_still_accepted():
    request = generation_request(schema=_array_schema())
    for output in (_aliased(8), list(range(40_000))):
        proposal = {
            "proposal_id": "proposal-1", "output": output, "support_span_ids": ("support-1",),
            "machine_authored": True,
        }
        result = generation_result().model_copy(update={"proposals": (proposal,)})
        assert validate_generation_result(request, result).compliant


def _proposals(output, count, *, same_object=False):
    one = {"proposal_id": "proposal-0", "output": output, "support_span_ids": ("support-1",),
           "machine_authored": True}
    if same_object:
        return (one,) * count
    return tuple({**one, "proposal_id": f"proposal-{i}"} for i in range(count))


def test_many_proposals_sharing_one_admitted_expansion_are_refused_by_one_call_allowance():
    import time

    from contracts_limits import DISCOVERY_BUDGET_MESSAGE

    request = generation_request(schema=_array_schema())
    shared = _aliased(15)  # about 65k paths: admitted for one field
    validate_generation_result(request, generation_result().model_copy(update={"proposals": _proposals(shared, 1)}))
    for count, same in ((3, False), (60, False), (5_000, False), (100_000, True)):
        result = generation_result().model_copy(update={"proposals": _proposals(shared, count, same_object=same)})
        started = time.monotonic()
        with pytest.raises(ValueError, match=DISCOVERY_BUDGET_MESSAGE):
            validate_generation_result(request, result)
        assert time.monotonic() - started < 10, (count, same)


def test_large_flat_json_is_still_accepted_alone_and_the_allowance_spans_the_call():
    from contracts_limits import DISCOVERY_BUDGET_MESSAGE

    request = generation_request(schema=_array_schema())
    flat = list(range(100_000))
    validate_generation_result(request, generation_result().model_copy(update={"proposals": _proposals(flat, 1)}))
    two = generation_result().model_copy(update={"proposals": _proposals(flat, 2)})
    with pytest.raises(ValueError, match=DISCOVERY_BUDGET_MESSAGE):
        validate_generation_result(request, two)  # the same 100k list twice is 200k paths in one call
    small = generation_result().model_copy(update={"proposals": _proposals(list(range(1_000)), 50)})
    assert validate_generation_result(request, small).compliant


def test_the_output_schema_and_the_proposals_share_the_call_allowance():
    from contracts_limits import DISCOVERY_BUDGET_MESSAGE

    flat = list(range(80_000))  # one object: discovery charges it once, but both fields walk it
    schema = {**_array_schema(), "examples": [flat]}
    request = generation_request(schema=schema)
    validate_generation_result(request, generation_result().model_copy(update={"proposals": _proposals(["x"], 1)}))
    with pytest.raises(ValueError, match=DISCOVERY_BUDGET_MESSAGE):
        validate_generation_result(request, generation_result().model_copy(update={"proposals": _proposals(flat, 1)}))


def test_a_metaclass_hook_cannot_repair_the_result_while_the_request_is_captured(monkeypatch):
    """Classifying a caller's container type never runs a caller's metaclass hook.

    Under an ABC-based model subclass check (``ABCMeta.__subclasscheck__``, which pydantic releases
    other than the locked one use), ``issubclass(t, BaseModel)`` walks every model subclass; one whose
    metaclass defines ``__subclasscheck__`` would run while the request is captured, before the
    result's entry state is frozen, and could swap an invalid result for a valid one.  The capture
    classifies types by their own MRO, so no pydantic version can open that window.
    """

    import abc

    from pydantic._internal._model_construction import ModelMetaclass

    monkeypatch.setattr(ModelMetaclass, "__subclasscheck__", abc.ABCMeta.__subclasscheck__)
    request = decision_request()
    result = DecisionResult(
        schema_version=VERSION,
        request_id="request-1",
        head_result=selected(candidate_id="candidate-unknown"),  # invalid for this request
        attempts=(usage(),),
    )
    fired: list[str] = []

    class Sneaky(tuple):
        pass

    class HookMeta(type(BaseModel)):
        def __subclasscheck__(cls, subclass):
            if subclass is Sneaky and not fired:
                fired.append("hook ran during capture")
                result.__dict__["head_result"] = selected()  # the repair a hook would attempt
            return super().__subclasscheck__(subclass)

    class Hooked(BaseModel, metaclass=HookMeta):
        pass

    request.head.__dict__["candidate_ids"] = Sneaky(request.head.candidate_ids)
    with pytest.raises(ValueError, match="unknown candidate"):
        validate_decision_result(request, result)
    assert fired == []
    assert Hooked  # kept alive: it must stay among BaseModel's subclasses during the call


@pytest.mark.parametrize(
    "value",
    [
        "123e4567-e89b-12d3-a456-426614174000-",  # trailing hyphen
        "123e4567-e89b-12d3-a456-426614174000}",  # trailing brace
        "{123e4567-e89b-12d3-a456-426614174000}",
        "urn:uuid:123e4567-e89b-12d3-a456-426614174000",
        "123e4567e89b12d3a456426614174000",
        "123e4567-e89b-12d3-a456-42661417400٠",  # a non-ASCII digit
    ],
)
def test_generation_uuid_format_requires_the_canonical_form(value):
    schema = output_schema(
        properties={"id": {"type": "string", "format": "uuid"}},
        required=["id"],
    )
    request = generation_request(schema=schema)
    for canonical in ("123e4567-e89b-12d3-a456-426614174000", "123E4567-E89B-12D3-A456-426614174000"):
        assert validate_generation_result(request, generation_result(output={"id": canonical})).compliant
    with pytest.raises(ValueError, match="does not match declared schema"):
        validate_generation_result(request, generation_result(output={"id": value}))


def _generation_with_output(output, **properties):
    request = generation_request(schema=output_schema(properties=properties, required=list(properties)))
    result = generation_result(output={"placeholder": "x"})
    result.proposals[0].__dict__["output"] = output  # unchecked storage, as a caller can supply it
    return request, result


def test_a_mapping_pair_cannot_repair_its_value_while_unpacking_checks_its_length():
    """Unpacking ``key, value = pair`` asks the pair for a third item before the value is captured."""

    exposed = ["a", "b", "c"]  # violates maxItems

    class Pair(tuple):
        def __iter__(self):
            yield "answer"
            yield exposed
            exposed[:] = ["a"]  # the exhaustion check would run this before capture

    class Output(dict):
        def items(self):
            return iter([Pair()])

    request, result = _generation_with_output(
        Output(answer=None), answer={"type": "array", "maxItems": 1, "items": {"type": "string"}}
    )
    with pytest.raises(ValueError, match="does not match declared schema"):
        validate_generation_result(request, result)


@pytest.mark.parametrize(
    "pair, message",
    [
        (("answer",), "not enough values to unpack"),
        (("answer", ["a"], "extra"), "too many values to unpack"),
    ],
)
def test_a_malformed_mapping_pair_is_refused_as_unpacking_would(pair, message):
    class Pair(tuple):
        def __iter__(self):
            return iter(tuple.__iter__(self))

    class Output(dict):
        def items(self):
            return iter([Pair(pair)])

    request, result = _generation_with_output(Output(answer=None), answer={"type": "array"})
    with pytest.raises(ValueError, match=message):
        validate_generation_result(request, result)


class _ImpostorKey:
    """A non-text key that hashes and compares like ``"score"`` and records its hooks."""

    hooks: list[str] = []

    def __hash__(self):
        _ImpostorKey.hooks.append("hash")
        return hash("score")

    def __eq__(self, other):
        _ImpostorKey.hooks.append("eq")
        return other == "score"


def test_a_non_text_key_in_a_public_view_cannot_overwrite_a_text_key_or_run_its_hooks():
    class Output(dict):
        def items(self):
            return iter([("score", 999), (_ImpostorKey(), 5)])  # 999 breaks the schema; the impostor would "fix" it

    request, result = _generation_with_output(Output(score=999), score={"type": "integer", "maximum": 10})
    _ImpostorKey.hooks = []
    with pytest.raises(ValueError, match="keys must be text"):
        validate_generation_result(request, result)
    assert _ImpostorKey.hooks == []


def test_a_non_text_key_in_native_storage_is_refused_without_running_its_hooks():
    output = {}
    output[_ImpostorKey()] = 5  # stored before the call; the call itself must not hash or compare it
    request, result = _generation_with_output(output, score={"type": "integer", "maximum": 10})
    _ImpostorKey.hooks = []
    with pytest.raises(ValueError, match="keys must be text"):
        validate_generation_result(request, result)
    assert _ImpostorKey.hooks == []


def test_a_shared_typed_collection_is_charged_for_every_occurrence_revalidation_walks():
    """3,000 heads sharing one tuple of 3,000 candidate ids: one tuple to discovery, 9M entries to revalidation."""

    import time

    from contracts_limits import DISCOVERY_BUDGET_MESSAGE

    count = 3_000
    request = decision_request(batch=True)
    shared_ids = tuple(f"candidate-{index}" for index in range(count))
    request.__dict__["candidates"] = tuple(DecisionCandidate(candidate_id=c) for c in shared_ids)
    request.__dict__["heads"] = tuple(
        DecisionHead.model_construct(head_id=f"head-{index}", span_ids=("span-1",), candidate_ids=shared_ids, required=True)
        for index in range(count)
    )
    result = DecisionBatchResult(
        schema_version=VERSION, request_id="request-1", fallback_granularity=FallbackGranularity.WHOLE_BATCH,
        publishable=False, head_results=(abstained("head-0"),), attempts=(),
    )
    started = time.monotonic()
    with pytest.raises(ValueError, match=DISCOVERY_BUDGET_MESSAGE):
        validate_decision_batch_result(request, result)
    assert time.monotonic() - started < 2


def test_a_shared_typed_collection_within_the_allowance_still_validates():
    request = decision_request(batch=True)
    shared_ids = request.heads[0].candidate_ids
    request.__dict__["heads"] = tuple(
        DecisionHead.model_construct(head_id=head.head_id, span_ids=head.span_ids, candidate_ids=shared_ids, required=True)
        for head in request.heads
    )
    result = DecisionBatchResult(
        schema_version=VERSION, request_id="request-1", fallback_granularity=FallbackGranularity.WHOLE_BATCH,
        publishable=True, head_results=(selected("retain"), selected("category")), attempts=(usage(),),
    )
    assert validate_decision_batch_result(request, result).compliant


def test_many_references_to_one_wide_list_are_sized_once():
    """40,000 references to one 40,000-element list: the size memo is consulted before the list is copied."""

    import time

    wide = list(range(40_000))
    request, result = _generation_with_output({"v": [wide] * 40_000}, v={"type": "array"})
    started = time.monotonic()
    with pytest.raises(ValueError):
        validate_generation_result(request, result)
    assert time.monotonic() - started < 1.5


def test_a_shared_source_text_costs_little_per_occurrence():
    """One 2 MB source text shared by 100 source entries: each occurrence is revalidated."""

    import time

    request = decision_request()
    request.__dict__["sources"] = (source_evidence(text="a" * 2_000_000),) * 100
    result = DecisionResult(schema_version=VERSION, request_id="request-1", head_result=selected(), attempts=(usage(),))
    started = time.monotonic()
    with pytest.raises(ValueError):  # two versions of one record: refused after revalidation
        validate_decision_result(request, result)
    assert time.monotonic() - started < 2


def test_a_long_string_repeated_in_a_json_value_is_charged_for_every_occurrence():
    """1,000 references to one 1 MB string: few entries, but every occurrence is scanned."""

    from contracts_limits import DISCOVERY_BUDGET_MESSAGE

    request, result = _generation_with_output({"v": ["x" * 1_000_000] * 1_000}, v={"type": "array"})
    with pytest.raises(ValueError, match=DISCOVERY_BUDGET_MESSAGE):
        validate_generation_result(request, result)


def test_one_long_distinct_string_is_still_accepted():
    request, result = _generation_with_output({"v": "x" * 5_000_000}, v={"type": "string"})
    assert validate_generation_result(request, result).compliant


def test_a_long_scalar_output_shared_by_many_proposals_is_charged_for_every_occurrence():
    """1,000 proposals sharing one 1 MiB scalar output: no containers, but every occurrence is scanned."""

    import time

    from contracts_limits import DISCOVERY_BUDGET_MESSAGE

    request, result = _generation_with_output("x" * (1 << 20), v={"type": "string"})
    result.__dict__["proposals"] = result.proposals * 1_000
    started = time.monotonic()
    with pytest.raises(ValueError, match=DISCOVERY_BUDGET_MESSAGE):
        validate_generation_result(request, result)
    assert time.monotonic() - started < 2


def test_a_long_scalar_output_used_once_is_charged_once_and_admitted_to_validation():
    from contracts_limits import DISCOVERY_BUDGET_MESSAGE

    request, result = _generation_with_output("x" * (1 << 20), v={"type": "string"})
    with pytest.raises(ValueError) as refused:  # not an object: the schema refuses it, the allowance does not
        validate_generation_result(request, result)
    assert DISCOVERY_BUDGET_MESSAGE not in str(refused.value)


class _TextSubclass(str):
    pass


def test_repeated_string_subclass_leaves_are_charged_like_strings():
    """A str subclass is kept through reconstruction; it must cost what a plain string of its length costs."""

    from contracts_limits import DISCOVERY_BUDGET_MESSAGE

    leaf = _TextSubclass("x" * 200_000)
    request, result = _generation_with_output({"v": [leaf] * 1_000}, v={"type": "array"})
    with pytest.raises(ValueError, match=DISCOVERY_BUDGET_MESSAGE):
        validate_generation_result(request, result)


# ---- schema execution is bounded, not only the inputs it runs on


def test_unique_items_over_many_distinct_objects_is_checked_in_linear_time():
    """8,000 distinct objects under uniqueItems: jsonschema's object fallback compares every pair."""

    import time

    request, result = _generation_with_output(
        {"v": [{"id": index} for index in range(8_000)]}, v={"type": "array", "uniqueItems": True}
    )
    started = time.monotonic()
    assert validate_generation_result(request, result).compliant
    assert time.monotonic() - started < 2


@pytest.mark.parametrize(
    "items, unique",
    [
        ([1, 1.0], False),  # one number, two spellings
        ([1, True], True),  # a boolean is not a number
        ([0, False], True),
        ([[True], [1]], True),  # also inside arrays
        ([{"a": 1}, {"a": 1.0}], False),
        ([{"a": 1, "b": 2}, {"b": 2, "a": 1}], False),  # key order does not matter
        ([[1, 2], [2, 1]], True),  # item order does
        (["1", 1], True),
        ([None, None], False),
    ],
)
def test_unique_items_keeps_json_schema_equality(items, unique):
    request, result = _generation_with_output({"v": items}, v={"type": "array", "uniqueItems": True})
    if unique:
        assert validate_generation_result(request, result).compliant
    else:
        with pytest.raises(ValueError, match="does not match declared schema"):
            validate_generation_result(request, result)


def test_a_schema_whose_work_multiplies_with_the_output_is_refused_in_bounded_time():
    """300 subschemas applied to each of 3,000 items: both fit the input budgets, the product does not."""

    import time

    request, result = _generation_with_output(
        {"v": list(range(3_000))},
        v={"type": "array", "items": {"allOf": [{"type": "number"}] * 300}},
    )
    started = time.monotonic()
    with pytest.raises(ValueError, match="schema execution"):
        validate_generation_result(request, result)
    assert time.monotonic() - started < 3  # about 10 s when execution was unbounded


def test_a_large_enum_applied_to_many_items_is_refused_in_bounded_time():
    import time

    allowed = [{"id": index} for index in range(3_000)]
    request, result = _generation_with_output(
        {"v": [{"id": index} for index in range(3_000)]},
        v={"type": "array", "items": {"enum": allowed}},
    )
    started = time.monotonic()
    with pytest.raises(ValueError, match="schema execution"):
        validate_generation_result(request, result)
    assert time.monotonic() - started < 2


def test_checking_a_very_large_schema_against_the_meta_schema_is_bounded():
    """The meta-schema applies dozens of keywords to every schema node: a schema near the input allowance
    took tens of seconds to check, before any output was looked at."""

    import time

    started = time.monotonic()
    with pytest.raises((ValidationError, ValueError), match="schema execution"):
        generation_request(
            schema=output_schema(properties={"v": {"allOf": [{"type": "number"}] * 20_000}}, required=["v"])
        )
    assert time.monotonic() - started < 3


def test_a_nested_dialect_declaration_does_not_escape_the_execution_budget():
    """A subschema declaring $schema ran on jsonschema's own validator, outside the meter."""

    import time

    nested = {"$schema": contracts_inference.JSON_SCHEMA_DIALECT, "type": "number"}
    request, result = _generation_with_output(
        {"v": list(range(3_000))}, v={"type": "array", "items": {"allOf": [nested] * 300}}
    )
    started = time.monotonic()
    with pytest.raises(ValueError, match="schema execution"):
        validate_generation_result(request, result)
    assert time.monotonic() - started < 3


def test_data_that_looks_like_a_dialect_declaration_is_kept():
    value = {"$schema": "not-a-dialect", "x": 1}
    request, result = _generation_with_output({"v": value}, v={"enum": [value]})
    assert validate_generation_result(request, result).compliant


# ---- enum equality recurses: the work is bounded by the nodes of the entries and the instance

def _enum_shape(entries: int, items: int, width: int):
    """Entries and items that keep their size one level down (``[0] * width + [j]``): the top-level sizes are
    small, but the equality jsonschema runs for every (item, entry) pair walks ``width`` nodes each."""

    allowed = [[0] * width + [j] for j in range(entries)]
    values = [[0] * width + [entries - 1] for _ in range(items)]  # every item matches the last entry
    return _generation_with_output({"v": values}, v={"type": "array", "items": {"enum": allowed}})


def _enum_and_input_only_seconds(entries: int, items: int, width: int):
    """Seconds to validate the shape with the list as an enum, and with the same list as ``examples`` (the input
    budgets and contract checks alone, which are not what is bounded here), and the enum run's outcome."""

    import time

    allowed = [[0] * width + [j] for j in range(entries)]
    values = [[0] * width + [entries - 1] for _ in range(items)]
    request, result = _generation_with_output({"v": values}, v={"type": "array", "items": {"examples": allowed}})
    started = time.monotonic()
    validate_generation_result(request, result)
    baseline = time.monotonic() - started
    request, result = _enum_shape(entries, items, width)
    started = time.monotonic()
    try:
        validate_generation_result(request, result)
        outcome = None
    except ValueError as error:
        outcome = str(error)
    return time.monotonic() - started, baseline, outcome


def test_an_enum_whose_entries_hide_their_size_is_refused_in_bounded_time():
    elapsed, baseline, outcome = _enum_and_input_only_seconds(entries=1_000, items=1_000, width=60)
    assert outcome is not None and "schema execution" in outcome
    # The execution share (elapsed minus the input-only cost) was about 5-7 s before the nodes were charged;
    # the refusal is the deterministic assertion, the bound is relative to the same machine's baseline.
    assert elapsed - baseline < 2.5


def test_a_modest_enum_over_nested_entries_still_validates():
    request, result = _enum_shape(entries=50, items=50, width=20)
    assert validate_generation_result(request, result).compliant


@pytest.mark.parametrize(
    "allowed, value, member",
    [
        ([1, 2], 1.0, True),  # one number, two spellings
        ([1, 2], True, False),  # a boolean is not a number
        ([[True]], [1], False),  # also inside arrays
        ([{"a": 1, "b": [1.0]}], {"b": [1], "a": 1.0}, True),  # key order does not matter, depth does not matter
        ([[1, 2]], [2, 1], False),  # item order does
        (["1"], 1, False),
        ([None], None, True),
        ([{"a": [{"b": 1}]}], {"a": [{"b": 2}]}, False),
    ],
)
def test_enum_keeps_json_schema_equality(allowed, value, member):
    request, result = _generation_with_output({"v": value}, v={"enum": allowed})
    if member:
        assert validate_generation_result(request, result).compliant
    else:
        with pytest.raises(ValueError, match="does not match declared schema"):
            validate_generation_result(request, result)


# ---- the documented schema-execution refusal

def test_the_schema_execution_refusal_is_the_documented_message_on_both_paths():
    """The module documents one refusal text for both bounds (the meter and the subschema cap); a change to
    either the text or the documentation must be deliberate."""

    documented = "output schema execution exceeds its budget"
    assert contracts_inference.SCHEMA_EXECUTION_MESSAGE == documented
    assert documented in " ".join(contracts_inference.__doc__.split())
    # the meter
    request, result = _generation_with_output(
        {"v": list(range(3_000))}, v={"type": "array", "items": {"allOf": [{"type": "number"}] * 300}}
    )
    with pytest.raises(ValueError) as meter:
        validate_generation_result(request, result)
    assert type(meter.value) is ValueError and str(meter.value) == documented  # the whole message, exactly
    # the subschema cap
    with pytest.raises((ValidationError, ValueError)) as cap:
        generation_request(
            schema=output_schema(
                properties={"v": {"allOf": [{"type": "number"}] * (contracts_inference.MAX_OUTPUT_SCHEMA_SUBSCHEMAS + 1)}},
                required=["v"],
            )
        )
    assert isinstance(cap.value, ValueError) and documented in str(cap.value)  # inside pydantic's ValidationError


# ---- the enum mechanism is pinned: lists and instances are hashed once per call, every node is charged

def _count_identity_nodes(monkeypatch):
    calls = [0]
    original = contracts_inference._json_identity

    def counting(value, meter):
        calls[0] += 1
        return original(value, meter)

    monkeypatch.setattr(contracts_inference, "_json_identity", counting)
    return calls


def test_each_enum_list_and_instance_is_hashed_once_per_call_whatever_the_applications(monkeypatch):
    """40 nested entries of about 500 nodes applied to 200 items: hashing the list per application would be
    4,000,000 nodes (refused by the meter); once per call it is about 20,000, plus the items."""

    calls = _count_identity_nodes(monkeypatch)
    allowed = [[0] * 499 + [j] for j in range(40)]
    values = [[0] * 499 + [39] for _ in range(200)]
    request, result = _generation_with_output({"v": values}, v={"type": "array", "items": {"enum": allowed}})
    assert validate_generation_result(request, result).compliant
    assert calls[0] < 150_000, calls[0]


def test_repeated_enum_applications_over_one_instance_hash_it_once(monkeypatch):
    """200 single-entry enums under anyOf, all applied to the same 50,005-node value (five arrays of 10,000: a
    small top-level size, so only the nodes can make it expensive)."""

    calls = _count_identity_nodes(monkeypatch)
    request, result = _generation_with_output(
        {"v": [list(range(10_000)) for _ in range(5)]},
        v={"anyOf": [{"enum": [[k]]} for k in range(200)]},
    )
    with pytest.raises(ValueError, match="does not match declared schema"):  # nothing matches; not the meter
        validate_generation_result(request, result)
    assert calls[0] < 200_000, calls[0]  # per application it was 200 x 50,005 = 10 million


def test_many_small_enums_over_one_large_instance_are_not_refused_by_the_meter():
    request, result = _generation_with_output(
        {"v": [list(range(20_000)) for _ in range(5)]},
        v={"allOf": [{"not": {"enum": [[k]]}} for k in range(150)]},
    )
    assert validate_generation_result(request, result).compliant  # the parent answered this; the meter must too


def test_every_hashed_node_is_charged_to_the_meter():
    meter = contracts_inference._SchemaExecutionMeter(limit=100)
    contracts_inference._json_identity([[1, 2], [3], {"a": True}], meter)  # 1 + 2 + 2 + 1 + 1 + 1 + ... nodes
    nodes = 100 - meter.left
    assert nodes == 8  # outer list, two lists and a dict, then 2 + 1 + 1 leaves
    with pytest.raises(ValueError) as refused:
        contracts_inference._json_identity([[1, 2], [3], {"a": True}], contracts_inference._SchemaExecutionMeter(limit=nodes - 1))
    assert str(refused.value) == contracts_inference.SCHEMA_EXECUTION_MESSAGE


# ---- identities keep JSON Schema equality but not Python's predictable numeric hash; const is metered

import sys as _sys


def _identity_hashes(values):
    meter = contracts_inference._SchemaExecutionMeter()
    return [hash(contracts_inference._json_identity(value, meter)) for value in values]


def test_integers_that_collide_in_pythons_numeric_hash_do_not_collide_as_identities():
    """hash(i * modulus) is 0 for every i: an attacker-chosen set of distinct integers would put every item of a
    uniqueItems or cached enum set in one bucket (quadratic, uncharged).  Counted as collisions, not timed."""

    modulus = _sys.hash_info.modulus
    values = [i * modulus for i in range(2_000)]
    assert len(set(hash(value) for value in values)) == 1  # the premise: Python's own hash collides
    assert len(set(_identity_hashes(values))) == len(values)
    floats = [float(i * modulus) for i in range(1, 500)]
    assert len(set(_identity_hashes(floats))) == len(floats)


@pytest.mark.parametrize(
    "left, right, equal",
    [
        (1, 1.0, True), (0, -0.0, True), (True, 1, False), (False, 0, False), (10**400, 10**400, True),
        (10**400, 1e308, False), (2**53 + 1, 2.0**53, False), (2**53, 2.0**53, True), (0.1, 0.1, True),
        (0.1, 0.10000000000000002, False), (-1, 1, False), ("1", 1, False), (None, None, True), (None, 0, False),
        ([1, [2.0]], [1.0, [2]], True), ([True], [1], False), ({"a": [1]}, {"a": [1.0]}, True),
        ({"a": 1, "b": 2}, {"b": 2, "a": 1}, True), ([1, 2], [2, 1], False), ({"a": 1}, {"a": 1, "b": 2}, False),
    ],
)
def test_identities_keep_json_schema_equality(left, right, equal):
    meter = contracts_inference._SchemaExecutionMeter()
    one = contracts_inference._json_identity(left, meter)
    two = contracts_inference._json_identity(right, meter)
    assert (one == two) is equal
    if equal:
        assert hash(one) == hash(two)
    # and through the three keywords that use them
    for keyword_schema, outcome in (
        ({"enum": [left]}, equal),
        ({"const": left}, equal),
    ):
        request, result = _generation_with_output({"v": right}, v=keyword_schema)
        if outcome:
            assert validate_generation_result(request, result).compliant
        else:
            with pytest.raises(ValueError, match="does not match declared schema"):
                validate_generation_result(request, result)
    request, result = _generation_with_output({"v": [left, right]}, v={"type": "array", "uniqueItems": True})
    if equal:
        with pytest.raises(ValueError, match="does not match declared schema"):
            validate_generation_result(request, result)
    else:
        assert validate_generation_result(request, result).compliant


def test_a_const_repeated_as_separate_copies_is_answered_by_lookup_not_comparison(monkeypatch):
    """100 allOf const entries (separate copies of 200 numbers) over 100 items that equal it: the parent compared
    2,000,000 node pairs; here every node is hashed once (about 40,000) and each application is a lookup."""

    nodes = [0]
    original = contracts_inference._json_identity

    def counting(value, meter):
        nodes[0] += 1
        return original(value, meter)

    monkeypatch.setattr(contracts_inference, "_json_identity", counting)
    request, result = _generation_with_output(
        {"v": [{"k": list(range(200))} for _ in range(100)]},
        v={"type": "array", "items": {"allOf": [{"const": {"k": list(range(200))}} for _ in range(100)]}},
    )
    assert validate_generation_result(request, result).compliant
    assert nodes[0] <= 2 * 100 * 202 + 100, nodes[0]  # each distinct object once, not once per application


def test_const_hashes_its_value_once_per_call_and_each_instance_once(monkeypatch):
    """Five const entries applied to four items: every distinct object (the const, each item copy the call
    validates) is hashed exactly once, not once per application."""

    import collections

    hashed = collections.Counter()
    original = contracts_inference._json_identity

    def counting(value, meter):
        if isinstance(value, dict) and "k" in value:
            hashed[id(value)] += 1
        return original(value, meter)

    monkeypatch.setattr(contracts_inference, "_json_identity", counting)
    const = {"k": list(range(50))}
    request, result = _generation_with_output(
        {"v": [{"k": list(range(50))} for _ in range(4)]},
        v={"type": "array", "items": {"allOf": [{"const": const}] * 5}},
    )
    assert validate_generation_result(request, result).compliant
    assert hashed and set(hashed.values()) == {1}, hashed  # per application it would be 5 for each item
    assert id(const) not in hashed or hashed[id(const)] == 1


def test_a_modest_nested_const_still_validates_and_a_mismatch_is_still_reported():
    request, result = _generation_with_output({"v": {"k": [1, 2.0]}}, v={"const": {"k": [1.0, 2]}})
    assert validate_generation_result(request, result).compliant
    request, result = _generation_with_output({"v": {"k": [1, 3]}}, v={"const": {"k": [1.0, 2]}})
    with pytest.raises(ValueError, match="does not match declared schema"):
        validate_generation_result(request, result)


# ---- every enum-typed value is a declared member by identity (``str.__new__(Enum, "value")`` is refused)

def _every_enum():
    import enum as _enum

    import contracts_engines
    import contracts_errors

    return [
        value
        for module in (contracts_inference, contracts_engines, contracts_errors)
        for value in vars(module).values()
        if isinstance(value, type) and issubclass(value, _enum.Enum) and value.__module__ == module.__name__
    ]


def _spoof(member):
    return str.__new__(type(member), member.value)  # right class and value, but not a declared member


def test_the_enum_sweep_covers_every_enum_of_the_inference_engine_and_error_contracts() -> None:
    assert {kind.__name__ for kind in _every_enum()} >= {
        "FallbackGranularity", "ScoreKind", "DecisionOutcome", "GenerationOutcome", "ResourceViolation",
        "EngineAdapterKind", "CapabilityHealth", "InternalErrorCode", "PublicErrorCode",
    }


@pytest.mark.parametrize("kind", _every_enum(), ids=lambda kind: kind.__name__)
def test_a_contract_field_refuses_a_noncanonical_enum_instance_and_keeps_every_member(kind) -> None:
    from pydantic import create_model

    from contracts_common import ContractModel

    holder = create_model("Holder", __base__=ContractModel, value=(kind, ...), many=(tuple[kind, ...], ()))
    for member in kind:
        assert holder.model_validate({"value": member}).value is member
        with pytest.raises(ValidationError):
            holder.model_validate({"value": _spoof(member)})
        with pytest.raises(ValidationError):
            holder.model_validate({"value": member, "many": (member, _spoof(member))})


def test_a_noncanonical_selected_outcome_cannot_skip_the_selection_rules() -> None:
    """Identity tests on ``outcome`` skipped every rule for a look-alike SELECTED (and PROPOSED) and accepted a
    selected head with no candidate, and a proposed result with no proposal."""

    from contracts_inference import DecisionHeadResult, DecisionOutcome, GenerationOutcome, GenerationResult

    head = {"head_id": "h", "selected_candidate_ids": (), "distribution": None, "reason_code": None}
    with pytest.raises(ValidationError):
        DecisionHeadResult.model_validate({**head, "outcome": _spoof(DecisionOutcome.SELECTED)}, strict=True)
    result = {"schema_version": "candidate-v1", "request_id": "r", "proposals": (), "reason_code": None, "attempts": ()}
    with pytest.raises(ValidationError):
        GenerationResult.model_validate({**result, "outcome": _spoof(GenerationOutcome.PROPOSED)}, strict=True)
    # controls: real members still follow the rules
    assert DecisionHeadResult.model_validate({**head, "outcome": DecisionOutcome.EMPTY}, strict=True).outcome is DecisionOutcome.EMPTY


# ---- const: one comparison per (instance, const) pair; a mismatch stays cheap

def test_one_matching_pair_is_compared_once_per_call():
    """The same const object applied 100 times to each of 300 equal items is 300 comparisons (about 60,000
    units), not 30,000; it used to be charged per application and refused."""

    const = {"k": list(range(200))}
    request, result = _generation_with_output(
        {"v": [{"k": list(range(200))} for _ in range(300)]},
        v={"type": "array", "items": {"allOf": [{"const": const}] * 100}},
    )
    assert validate_generation_result(request, result).compliant


def test_a_deeply_nested_equal_const_never_leaks_a_recursion_error():
    depth = 200
    value: object = 1
    for _ in range(depth):
        value = [value]
    request, result = _generation_with_output({"v": value}, v={"const": value})
    try:
        assert validate_generation_result(request, result).compliant
    except ValueError as refused:  # a refusal is fine; a RecursionError is not
        assert not isinstance(refused, RecursionError)


def test_many_mismatching_consts_over_large_items_stay_cheap():
    """100 negated consts that differ from 300 items of 201 nodes: the cached hashes reject each pair at once.
    Without that short-circuit every pair would be compared and charged (6 million units, refused)."""

    request, result = _generation_with_output(
        {"v": [{"k": list(range(200))} for _ in range(300)]},
        v={"type": "array", "items": {"allOf": [{"not": {"const": {"k": [index]}}} for index in range(100)]}},
    )
    assert validate_generation_result(request, result).compliant


def test_const_comparison_work_is_per_distinct_value_not_per_application(monkeypatch):
    """100 applications of a const to each of 300 equal items: equality walks are counted per node (a Python-level
    ``__eq__`` per node).  Per application it was 100 x 300 x 201 = 6 million; interned, each distinct object is
    compared with its pooled twin once (about 60,000)."""

    walks = [0]
    original = contracts_inference._Identity.__eq__

    def counting(self, other):
        walks[0] += 1
        return original(self, other)

    monkeypatch.setattr(contracts_inference._Identity, "__eq__", counting)
    const = {"k": list(range(200))}
    request, result = _generation_with_output(
        {"v": [{"k": list(range(200))} for _ in range(300)]},
        v={"type": "array", "items": {"allOf": [{"const": const}] * 100}},
    )
    assert validate_generation_result(request, result).compliant
    assert walks[0] < 300_000, walks[0]


def test_the_enum_check_visits_a_shared_container_once_not_once_per_path(monkeypatch):
    """20 tuple layers each holding the previous twice are 2**20 paths to one leaf; the check keeps what it saw."""

    import contracts_common

    checked = [0]
    original = contracts_common.is_canonical_enum_member

    def counting(value):
        checked[0] += 1
        return original(value)

    monkeypatch.setattr(contracts_common, "is_canonical_enum_member", counting)
    from contracts_inference import DecisionOutcome

    layer: object = DecisionOutcome.SELECTED
    for _ in range(20):
        layer = (layer, layer)
    contracts_common.require_canonical_enum_members(layer)
    assert checked[0] < 100, checked[0]  # per path it was 1,048,576
    bad: object = _spoof(DecisionOutcome.SELECTED)
    for _ in range(20):
        bad = (bad, bad)
    with pytest.raises(ValueError, match="declared member"):
        contracts_common.require_canonical_enum_members(bad)


# ---- a successful enum lookup does no deep equality per application

def test_enum_lookups_that_match_do_equality_work_per_distinct_value_not_per_application(monkeypatch):
    """200 singleton enum branches (separate copies of an object of 200 numbers) over 200 matching objects: every
    lookup hits, which used to walk the whole value per application (8 million node comparisons).  Counted as
    Python-level equality calls per node."""

    walks = [0]
    original = contracts_inference._Identity.__eq__

    def counting(self, other):
        walks[0] += 1
        return original(self, other)

    monkeypatch.setattr(contracts_inference._Identity, "__eq__", counting)
    request, result = _generation_with_output(
        {"v": [{"k": list(range(200))} for _ in range(200)]},
        v={"type": "array", "items": {"allOf": [{"enum": [{"k": list(range(200))}]} for _ in range(200)]}},
    )
    assert validate_generation_result(request, result).compliant
    assert walks[0] < 300_000, walks[0]  # about 200 + 200 distinct objects compared once with their pooled twin


def test_unique_items_over_equal_large_items_stops_at_the_first_duplicate(monkeypatch):
    walks = [0]
    original = contracts_inference._Identity.__eq__

    def counting(self, other):
        walks[0] += 1
        return original(self, other)

    monkeypatch.setattr(contracts_inference._Identity, "__eq__", counting)
    request, result = _generation_with_output(
        {"v": [{"k": list(range(200))} for _ in range(500)]}, v={"type": "array", "uniqueItems": True}
    )
    with pytest.raises(ValueError, match="does not match declared schema"):
        validate_generation_result(request, result)
    assert walks[0] < 2_000, walks[0]  # the second item equals the first; nothing after it is compared


# ---- execution work that scales with the length of a value is charged by that length

_SCHEMA = {"$schema": contracts_inference.JSON_SCHEMA_DIALECT}


def _errors(schema, instance):
    validator = contracts_inference._build_output_validator({**_SCHEMA, **schema})
    return list(validator.iter_errors(instance))


def test_an_integer_is_charged_by_its_length_before_its_key_is_built():
    small, large = contracts_inference._SchemaExecutionMeter(), contracts_inference._SchemaExecutionMeter()
    contracts_inference._json_identity(12345, small)
    contracts_inference._json_identity(1 << (8 * 262_144), large)  # 256 KiB
    assert small.left == contracts_inference.MAX_SCHEMA_EXECUTION_UNITS - 1
    assert contracts_inference.MAX_SCHEMA_EXECUTION_UNITS - large.left >= 262_144 // 64
    refusing = contracts_inference._SchemaExecutionMeter(limit=100)
    with pytest.raises(ValueError, match="exceeds its budget"):
        contracts_inference._json_identity(1 << (8 * 262_144), refusing)


def test_many_references_to_one_large_integer_are_refused_by_the_meter():
    """256 references to one 256 KiB integer converted 64 MiB while charging 256 units."""

    request, result = _generation_with_output({"v": 1}, v={"enum": [1 << (8 * 262_144)] * 256})
    with pytest.raises(ValueError, match="exceeds its budget"):
        validate_generation_result(request, result)


def test_repeated_format_checks_over_one_large_string_are_charged_by_its_length():
    text = "12:00:00." + "1" * (1 << 20) + "Z"  # a valid 1 MiB time
    request, result = _generation_with_output(
        {"v": text}, v={"allOf": [{"format": "time"} for _ in range(512)]}
    )
    with pytest.raises(ValueError, match="exceeds its budget"):
        validate_generation_result(request, result)
    short = "12:00:00Z"
    request, result = _generation_with_output(
        {"v": short}, v={"allOf": [{"format": "time"} for _ in range(512)]}
    )
    assert validate_generation_result(request, result).compliant


def test_one_format_check_over_a_large_string_is_still_allowed():
    text = "12:00:00." + "1" * (1 << 20) + "Z"
    request, result = _generation_with_output({"v": text}, v={"format": "time"})
    assert validate_generation_result(request, result).compliant


@pytest.mark.parametrize("keyword", ["enum", "const"])
def test_a_mismatch_message_is_bounded_whatever_the_size_of_the_values(keyword):
    big = "x" * (5 << 20)
    value = ["y"] if keyword == "enum" else "y"
    (error,) = _errors({keyword: value}, big)
    assert len(error.message) < 300, len(error.message)
    value = [big] if keyword == "enum" else big  # a large expected value as well
    (error,) = _errors({keyword: value}, "y")
    assert len(error.message) < 300, len(error.message)
    nested = {"k": [big]}
    (error,) = _errors({keyword: [nested] if keyword == "enum" else nested}, ["z"])
    assert len(error.message) < 300, len(error.message)
    (error,) = _errors({keyword: [1 << 100_000] if keyword == "enum" else 1 << 100_000}, 3)
    assert len(error.message) < 300, len(error.message)  # (repr of a huge integer would raise)


def test_a_mismatch_message_still_says_what_was_expected_for_small_values():
    (error,) = _errors({"enum": ["a", "b"]}, "c")
    assert error.message == "'c' is not one of 2 allowed values"
    (error,) = _errors({"const": "a"}, "c")
    assert "'a'" in error.message


def test_any_keywords_mismatch_message_is_charged_and_bounded():
    big = "x" * (5 << 20)
    for schema in ({"type": "integer"}, {"minLength": 10**7}, {"maxLength": 1}, {"format": "email"}):
        (error,) = _errors(schema, big)
        assert len(error.message) < 300, (schema, len(error.message))


def test_repeated_negated_keywords_over_a_large_string_stay_cheap_and_compliant():
    """Each negated mismatch used to build a message quoting the whole instance; the output is valid and the
    work is constant per application, so it is neither slow nor refused."""

    big = "x" * (5 << 20)
    for keyword in ({"type": "integer"}, {"maxLength": 1}, {"minLength": 10**7}):
        request, result = _generation_with_output(
            {"v": big}, v={"allOf": [{"not": keyword} for _ in range(1000)]}
        )
        assert validate_generation_result(request, result).compliant


def test_repeated_negated_enum_and_const_mismatches_over_a_large_string_stay_cheap():
    big = "x" * (5 << 20)
    for keyword in ({"enum": ["y"]}, {"const": "y"}):
        request, result = _generation_with_output(
            {"v": big}, v={"allOf": [{"not": keyword} for _ in range(1000)]}
        )
        assert validate_generation_result(request, result).compliant


# ---- a valid output that the base accepted is not refused for the messages of its discarded failures

def test_optional_fields_that_are_large_strings_are_accepted():
    """anyOf [null, string] fails its first branch for every string, building a message that quotes it."""

    optional = {"anyOf": [{"type": "null"}, {"type": "string"}]}
    for count, size in ((8, 5 << 20), (40, 1 << 20)):
        properties = {f"f{i}": optional for i in range(count)}
        request, result = _generation_with_output({f"f{i}": "x" * size for i in range(count)}, **properties)
        assert validate_generation_result(request, result).compliant


def test_a_negated_type_applied_repeatedly_to_one_large_string_is_accepted():
    big = "x" * (5 << 20)
    request, result = _generation_with_output({"v": big}, v={"allOf": [{"not": {"type": "integer"}}] * 7})
    assert validate_generation_result(request, result).compliant


def test_the_bounded_view_quotes_large_values_briefly_and_small_ones_in_full():
    view = contracts_inference._bounded_view
    for large in ("x" * (5 << 20), ["y"] * 100_000, {f"k{i}": i for i in range(100_000)}):
        assert len(repr(view(large))) < 100
        assert view(large) == large
    assert repr(view({"a": [1, "b"]})) == "{'a': [1, 'b']}"
    assert repr(view("x" * 256)) == repr("x" * 256)


def test_the_bounded_view_keeps_shared_containers_shared_and_does_not_expand_them():
    layer: object = ["x" * 300]
    for _ in range(40):
        layer = [layer, layer]  # 2**40 paths, 41 distinct containers
    copied = contracts_inference._bounded_view(layer)
    assert copied[0] is copied[1]  # type: ignore[index]
    assert copied[0][0] is copied[0][1]  # type: ignore[index]


def test_a_hostname_longer_than_any_hostname_is_refused_without_a_scan():
    label = "a" * 63
    longest = ".".join([label, label, label, "a" * 61])  # 253 characters
    assert len(longest) == 253
    assert _errors({"format": "idn-hostname"}, longest) == []
    assert _errors({"format": "idn-hostname"}, "a" * 254)
    request, result = _generation_with_output(
        {"v": "a" * (1 << 20)}, v={"allOf": [{"not": {"format": "idn-hostname"}} for _ in range(16)]}
    )
    assert validate_generation_result(request, result).compliant  # 16 refusals, none of them a megabyte scan


# ---- a huge integer never reaches a diagnostic's str(), and a hostname may carry its root dot

@pytest.fixture
def default_int_digit_limit():
    import sys

    previous = sys.get_int_max_str_digits()
    sys.set_int_max_str_digits(4300)  # Python's default: str() and repr() of a longer integer raise
    yield
    sys.set_int_max_str_digits(previous)


HUGE = 1 << 16_000  # 4,817 digits


def test_a_huge_integer_is_accepted_by_a_union_whose_first_branch_fails(default_int_digit_limit):
    union = {"anyOf": [{"type": "string"}, {"type": "integer"}]}
    request, result = _generation_with_output({"v": HUGE}, v=union)
    assert validate_generation_result(request, result).compliant
    request, result = _generation_with_output({"v": [HUGE], "w": {"k": HUGE}}, v={"anyOf": [{"type": "string"}, {"type": "array"}]},
                                              w={"anyOf": [{"type": "string"}, {"type": "object"}]})
    assert validate_generation_result(request, result).compliant


def test_a_negated_mismatch_over_a_huge_integer_is_accepted(default_int_digit_limit):
    request, result = _generation_with_output({"v": HUGE}, v={"not": {"type": "string"}})
    assert validate_generation_result(request, result).compliant
    request, result = _generation_with_output({"v": HUGE}, v={"not": {"enum": ["a"]}})
    assert validate_generation_result(request, result).compliant
    request, result = _generation_with_output({"v": HUGE}, v={"not": {"const": "a"}})
    assert validate_generation_result(request, result).compliant


def test_a_huge_integer_in_the_schema_does_not_break_a_discarded_branch(default_int_digit_limit):
    union = {"anyOf": [{"type": "integer", "minimum": HUGE}, {"type": "integer"}]}
    request, result = _generation_with_output({"v": 5}, v=union)
    assert validate_generation_result(request, result).compliant


def test_huge_integers_keep_their_numeric_semantics(default_int_digit_limit):
    cases = [
        ({"type": "integer"}, HUGE, True), ({"type": "number"}, HUGE, True), ({"type": "string"}, HUGE, False),
        ({"minimum": HUGE}, HUGE, True), ({"minimum": HUGE}, HUGE - 1, False), ({"maximum": HUGE}, HUGE + 1, False),
        ({"enum": [HUGE]}, HUGE, True), ({"enum": [HUGE]}, HUGE + 1, False), ({"const": HUGE}, HUGE, True),
        ({"const": HUGE}, HUGE + 1, False), ({"enum": [1, 2.0]}, 2, True),
        ({"multipleOf": 2}, HUGE, True), ({"multipleOf": 2}, HUGE + 1, False),
    ]
    for schema, instance, valid in cases:
        request, result = _generation_with_output({"v": instance}, v=schema)
        if valid:
            assert validate_generation_result(request, result).compliant, (schema, instance)
        else:
            with pytest.raises(ValueError, match="does not match declared schema"):
                validate_generation_result(request, result)
    request, result = _generation_with_output({"v": [HUGE, HUGE]}, v={"type": "array", "uniqueItems": True})
    with pytest.raises(ValueError, match="does not match declared schema"):
        validate_generation_result(request, result)


def test_a_message_about_a_huge_integer_is_short(default_int_digit_limit):
    for schema in ({"type": "string"}, {"enum": ["a"]}, {"const": "a"}):
        validator = contracts_inference._build_output_validator({**_SCHEMA, **schema})
        (error,) = validator.iter_errors(contracts_inference._bounded_view(HUGE))
        assert len(error.message) < 200, (schema, error.message)


def _hostname(length: int, dot: bool) -> str:
    label = "a" * 63
    core = ".".join([label, label, label, "a" * (length - 3 * 64 - (1 if dot else 0))])
    return core + ("." if dot else "")


@pytest.mark.parametrize(
    ("length", "dot", "valid"),
    [(253, False, True), (254, False, False), (255, False, False), (253, True, True), (254, True, True), (255, True, False)],
)
def test_the_hostname_length_limit_allows_one_more_character_for_the_root_dot(length, dot, valid):
    name = _hostname(length, dot)
    assert len(name) == length and name.endswith(".") == dot
    assert (_errors({"format": "idn-hostname"}, name) == []) is valid


def test_a_float_instance_against_a_divisor_past_the_float_range_is_judged_exactly(default_int_digit_limit):
    union = {"anyOf": [{"multipleOf": HUGE}, {"type": "number"}]}
    request, result = _generation_with_output({"v": 2.0**1000}, v=union)
    assert validate_generation_result(request, result).compliant  # not a multiple, so the second branch decides
    for instance, multiple in ((0.0, True), (2.0**1000, False), (3.5, False)):
        request, result = _generation_with_output({"v": instance}, v={"multipleOf": HUGE})
        if multiple:
            assert validate_generation_result(request, result).compliant
        else:
            with pytest.raises(ValueError, match="does not match declared schema"):
                validate_generation_result(request, result)


# ---- everything that copies, hashes or does arithmetic on a long string or a big integer is charged by size

def _units_spent(schema, instance):
    meter = contracts_inference._SchemaExecutionMeter()
    validator = contracts_inference._build_output_validator({**_SCHEMA, **schema}, meter)
    errors = list(validator.iter_errors(contracts_inference._bounded_view(instance)))
    return contracts_inference.MAX_SCHEMA_EXECUTION_UNITS - meter.left, errors


def test_a_long_string_is_charged_by_its_length_and_hashed_once_per_object(monkeypatch):
    built = [0]
    original = contracts_inference._Identity.__init__

    def counting(self, key):
        built[0] += 1
        original(self, key)

    monkeypatch.setattr(contracts_inference._Identity, "__init__", counting)
    text = "x" * (1 << 20)
    spent, errors = _units_spent({"allOf": [{"uniqueItems": True} for _ in range(512)]}, [text])
    assert errors == []
    assert spent >= (1 << 20) // 64  # charged by length
    assert built[0] < 10, built[0]  # not rebuilt (copied and hashed) for each of the 512 applications


def test_distinct_long_strings_and_keys_are_each_charged_by_length():
    spent, _ = _units_spent({"uniqueItems": True}, [f"{i}" + "y" * 262_144 for i in range(8)])
    assert spent >= 8 * 262_144 // 64
    spent, _ = _units_spent({"uniqueItems": True}, [{"k" * 262_144: 1}])
    assert spent >= 262_144 // 64


def test_multiple_of_is_charged_by_the_size_of_its_operands():
    big, divisor = (1 << 262_144) - 1, (1 << 131_072) + 1
    spent, errors = _units_spent({"multipleOf": divisor}, big)
    assert errors == [] and spent >= 3_000  # a valid multiple; the division is real work (it used to cost 8)
    small, _ = _units_spent({"multipleOf": 3}, 12)
    assert small < 50
    divisor_schema = {"not": {"multipleOf": (1 << 1_000_000) + 1}}  # each passes, so both run
    request, result = _generation_with_output({"v": 1 << 2_000_000}, v={"allOf": [divisor_schema, divisor_schema]})
    with pytest.raises(ValueError, match="exceeds its budget"):
        validate_generation_result(request, result)  # refused before any arithmetic


def test_multiple_of_with_a_float_is_charged_by_its_exact_fraction_too():
    spent, _ = _units_spent({"multipleOf": 0.1}, 1 << 262_144)
    assert spent >= 50  # one fraction of up to ~1,100 bits against 8,739 limbs; it used to cost 8


def test_a_large_bound_shared_by_many_branches_is_copied_once():
    import tracemalloc

    bound = 1 << (8 * 65_536)  # 64 KiB
    schema = {**_SCHEMA, "allOf": [{"minimum": bound} for _ in range(256)]}
    tracemalloc.start()
    try:
        contracts_inference._executable_schema(schema)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 1 << 20, peak  # it was one 64 KiB copy per occurrence: 16 MiB


def test_a_long_string_or_integer_repeated_by_reference_is_copied_once_into_the_view():
    import tracemalloc

    text, number = "z" * (1 << 20), 1 << (8 * 65_536)
    tracemalloc.start()
    try:
        view = contracts_inference._bounded_view([text] * 300 + [number] * 300)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 4 << 20, peak  # it was a copy per reference: 300 MiB
    assert view[0] is view[1] and view[300] is view[301]  # type: ignore[index]
