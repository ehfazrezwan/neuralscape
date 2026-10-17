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
