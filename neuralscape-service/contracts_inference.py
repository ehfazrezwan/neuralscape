"""Validated boundaries for bounded decisions and generative proposals.

Decision operations may only select supplied candidate identities. Generation
operations return schema-checked, machine-authored proposals with explicit
source support. Neither kind of result grants authority or publishes content.
"""

from __future__ import annotations

import math
import re
from datetime import date
from enum import Enum
from ipaddress import AddressValueError, IPv4Address, IPv6Address
from typing import Annotated, Any, Literal, Mapping
from uuid import UUID

import idna
from pydantic import BaseModel, Field, JsonValue, model_validator

from contracts_common import ContractModel, OpaqueId, SafeCounter, VersionedContract
from contracts_engines import QualificationClaim, _validated_contract_snapshot
from contracts_references import ReferenceHandle, SourceVersion

FiniteNumber = Annotated[float, Field(allow_inf_nan=False)]
JSON_SCHEMA_DIALECT = "https://json-schema.org/draft/2020-12/schema"
MAX_SAFE_COUNTER = 9_007_199_254_740_991
_SUPPORTED_OUTPUT_SCHEMA_FORMATS = frozenset(
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
_DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$", re.ASCII)
_TIME_PATTERN = re.compile(
    r"(?P<hour>[01][0-9]|2[0-3]):"
    r"(?P<minute>[0-5][0-9]):"
    r"(?P<second>[0-5][0-9]|60)"
    r"(?:\.[0-9]+)?"
    r"(?:[Zz]|(?P<offset_sign>[+-])"
    r"(?P<offset_hour>[01][0-9]|2[0-3]):"
    r"(?P<offset_minute>[0-5][0-9]))",
    re.ASCII,
)


class FallbackGranularity(str, Enum):
    WHOLE_BATCH = "whole_batch"
    PER_ITEM = "per_item"


class ScoreKind(str, Enum):
    """The meaning of scores, without treating confidence as calibration."""

    UNKNOWN = "unknown"
    RAW_SCORE = "raw_score"
    UNCALIBRATED_PROBABILITY = "uncalibrated_probability"
    CALIBRATED_PROBABILITY = "calibrated_probability"


class DecisionOutcome(str, Enum):
    SELECTED = "selected"
    EMPTY = "empty"
    ABSTAINED = "abstained"
    ERROR = "error"


class GenerationOutcome(str, Enum):
    PROPOSED = "proposed"
    EMPTY = "empty"
    ABSTAINED = "abstained"
    ERROR = "error"


class ResourceViolation(str, Enum):
    MAX_ATTEMPTS = "max_attempts"
    MAX_TOTAL_INPUT_UTF8_BYTES = "max_total_input_utf8_bytes"
    MAX_TOTAL_OUTPUT_UTF8_BYTES = "max_total_output_utf8_bytes"
    AGGREGATE_COUNTER_OVERFLOW = "aggregate_counter_overflow"


class CalibrationClaim(ContractModel):
    """Versioned references behind a calibration claim, not proof of it."""

    profile_reference: ReferenceHandle
    profile_version: OpaqueId
    evidence_reference: ReferenceHandle
    evidence_version: OpaqueId


class SourceEvidence(ContractModel):
    """Versioned source text supplied to validate portable UTF-8 byte spans."""

    source_version: SourceVersion
    text: str

    @model_validator(mode="after")
    def validate_utf8(self) -> "SourceEvidence":
        if any(0xD800 <= ord(character) <= 0xDFFF for character in self.text):
            raise ValueError("source evidence must be valid UTF-8 text")
        return self

    def utf8_boundaries(self) -> frozenset[int]:
        """Return offsets at which a UTF-8 code point begins or ends."""

        offsets = {0}
        cursor = 0
        for character in self.text:
            cursor += len(character.encode("utf-8"))
            offsets.add(cursor)
        return frozenset(offsets)


class SourceSpan(ContractModel):
    """Half-open UTF-8 byte coordinates to bind to explicit source evidence."""

    span_id: OpaqueId
    source_version: SourceVersion
    start_utf8_byte: SafeCounter
    end_utf8_byte: SafeCounter

    @model_validator(mode="after")
    def validate_order(self) -> "SourceSpan":
        if self.end_utf8_byte <= self.start_utf8_byte:
            raise ValueError("source span end must be greater than start")
        return self


def _validate_source_span_values(
    evidence: SourceEvidence,
    span: SourceSpan,
    source_utf8: bytes,
) -> None:
    if span.source_version != evidence.source_version:
        raise ValueError("source span does not match the source evidence version")
    if span.end_utf8_byte > len(source_utf8):
        raise ValueError("source span exceeds the source UTF-8 byte length")
    if not _is_utf8_boundary(source_utf8, span.start_utf8_byte) or not _is_utf8_boundary(
        source_utf8, span.end_utf8_byte
    ):
        raise ValueError("source span must not split a UTF-8 code point")


def _source_utf8_bytes(evidence: SourceEvidence) -> bytes:
    return evidence.text.encode("utf-8")


def _is_utf8_boundary(source_utf8: bytes, offset: int) -> bool:
    return offset == len(source_utf8) or (
        source_utf8[offset] & 0b1100_0000
    ) != 0b1000_0000


def validate_source_span(evidence: SourceEvidence, span: SourceSpan) -> None:
    """Validate exact source identity, bounds, and UTF-8 code-point boundaries."""

    evidence = _validated_contract_snapshot(
        evidence,
        SourceEvidence,
        label="source evidence",
    )
    span = _validated_contract_snapshot(span, SourceSpan, label="source span")
    _validate_source_span_values(evidence, span, _source_utf8_bytes(evidence))


class DecisionCandidate(ContractModel):
    """A closed-set candidate whose opaque identity must be preserved."""

    candidate_id: OpaqueId


class DecisionHead(ContractModel):
    """One independently identifiable decision over spans and candidates."""

    head_id: OpaqueId
    span_ids: tuple[OpaqueId, ...]
    candidate_ids: tuple[OpaqueId, ...]
    required: bool

    @model_validator(mode="after")
    def validate_identifiers(self) -> "DecisionHead":
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
    """Whole-operation limits in portable UTF-8 byte units."""

    max_total_input_utf8_bytes: SafeCounter
    max_total_output_utf8_bytes: SafeCounter
    max_attempts: SafeCounter

    @model_validator(mode="after")
    def validate_positive_limits(self) -> "InferenceResourceLimits":
        if self.max_total_input_utf8_bytes == 0 or self.max_total_output_utf8_bytes == 0:
            raise ValueError("input and output limits must be positive")
        if self.max_attempts == 0:
            raise ValueError("max_attempts must be positive")
        return self


class AttemptUsage(ContractModel):
    """Observed attempt usage, retained even when late or over a limit."""

    attempt_id: OpaqueId
    model_version: OpaqueId
    policy_version: OpaqueId
    input_utf8_bytes: SafeCounter
    output_utf8_bytes: SafeCounter
    late: bool


class ResourceCompliance(ContractModel):
    """Comparison of preserved attempt evidence with declared total limits."""

    compliant: bool
    violations: tuple[ResourceViolation, ...]
    observed_attempts: SafeCounter
    total_input_utf8_bytes: SafeCounter | None
    total_output_utf8_bytes: SafeCounter | None

    @model_validator(mode="after")
    def validate_consistency(self) -> "ResourceCompliance":
        if self.compliant != (not self.violations):
            raise ValueError("resource compliance must match its violations")
        return self


def _validate_source_evidence(
    sources: tuple[SourceEvidence, ...],
    spans: tuple[SourceSpan, ...],
) -> None:
    if not sources:
        raise ValueError("at least one source evidence value is required")
    record_ids = [source.source_version.record_id for source in sources]
    if len(set(record_ids)) != len(record_ids):
        raise ValueError("source evidence cannot contain two versions of one record")
    evidence_by_version = {
        (
            source.source_version.record_id,
            source.source_version.content_revision,
            source.source_version.policy_epoch,
        ): source
        for source in sources
    }
    span_ids = [span.span_id for span in spans]
    if not spans:
        raise ValueError("at least one source span is required")
    if len(set(span_ids)) != len(span_ids):
        raise ValueError("source span IDs must be unique")
    source_utf8_by_version: dict[tuple[str, int, int], bytes] = {}
    for span in spans:
        source_key = (
            span.source_version.record_id,
            span.source_version.content_revision,
            span.source_version.policy_epoch,
        )
        evidence = evidence_by_version.get(source_key)
        if evidence is None:
            raise ValueError("every span must reference declared source evidence")
        source_utf8 = source_utf8_by_version.get(source_key)
        if source_utf8 is None:
            source_utf8 = _source_utf8_bytes(evidence)
            source_utf8_by_version[source_key] = source_utf8
        _validate_source_span_values(evidence, span, source_utf8)


class _DecisionInputs(VersionedContract):
    request_id: OpaqueId
    operation: OpaqueId
    sources: tuple[SourceEvidence, ...]
    spans: tuple[SourceSpan, ...]
    candidates: tuple[DecisionCandidate, ...]
    processing_policy: ReferenceHandle
    resource_limits: InferenceResourceLimits
    deadline_ms: SafeCounter

    def _validate_input_identities(self) -> None:
        _validate_source_evidence(self.sources, self.spans)
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
    def validate_request(self) -> "DecisionRequest":
        self._validate_input_identities()
        self._validate_heads((self.head,))
        return self


class DecisionBatchRequest(_DecisionInputs):
    """Composite decisions sharing one source and fallback boundary."""

    heads: tuple[DecisionHead, ...]
    fallback_granularity: FallbackGranularity
    per_item_qualification: QualificationClaim | None

    @model_validator(mode="after")
    def validate_request(self) -> "DecisionBatchRequest":
        self._validate_input_identities()
        if not self.heads:
            raise ValueError("a decision batch requires at least one head")
        self._validate_heads(self.heads)
        if self.fallback_granularity is FallbackGranularity.PER_ITEM:
            if self.per_item_qualification is None:
                raise ValueError("per-item fallback requires a qualification claim")
            if self.per_item_qualification.operation != self.operation:
                raise ValueError("per-item qualification operation must match the request")
        elif self.per_item_qualification is not None:
            raise ValueError("whole-batch fallback must not carry per-item qualification")
        return self


class CandidateScore(ContractModel):
    candidate_id: OpaqueId
    value: FiniteNumber


class ScoreDistribution(ContractModel):
    """Scores with explicit semantics and reviewable calibration claims."""

    distribution_id: OpaqueId
    head_id: OpaqueId
    kind: ScoreKind
    complete: bool
    scores: tuple[CandidateScore, ...]
    calibration: CalibrationClaim | None

    @model_validator(mode="after")
    def validate_distribution(self) -> "ScoreDistribution":
        candidate_ids = [score.candidate_id for score in self.scores]
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("a distribution cannot score a candidate twice")
        if self.kind is ScoreKind.CALIBRATED_PROBABILITY:
            if self.calibration is None:
                raise ValueError("calibrated probability requires a calibration claim")
        elif self.calibration is not None:
            raise ValueError("only calibrated probability may carry calibration evidence")
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
    """One head outcome; empty differs from inability to decide."""

    head_id: OpaqueId
    outcome: DecisionOutcome
    selected_candidate_ids: tuple[OpaqueId, ...]
    distribution: ScoreDistribution | None
    reason_code: OpaqueId | None

    @model_validator(mode="after")
    def validate_outcome(self) -> "DecisionHeadResult":
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


def _validate_unique_attempts(attempts: tuple[AttemptUsage, ...]) -> None:
    attempt_ids = [attempt.attempt_id for attempt in attempts]
    if len(set(attempt_ids)) != len(attempt_ids):
        raise ValueError("attempt IDs must be unique")


class DecisionResult(VersionedContract):
    request_id: OpaqueId
    head_result: DecisionHeadResult
    attempts: tuple[AttemptUsage, ...]

    @model_validator(mode="after")
    def validate_attempts(self) -> "DecisionResult":
        _validate_unique_attempts(self.attempts)
        return self


class DecisionBatchResult(VersionedContract):
    """Batch diagnostics plus whether the batch may cross publication boundary."""

    request_id: OpaqueId
    fallback_granularity: FallbackGranularity
    publishable: bool
    head_results: tuple[DecisionHeadResult, ...]
    attempts: tuple[AttemptUsage, ...]

    @model_validator(mode="after")
    def validate_unique_ids(self) -> "DecisionBatchResult":
        head_ids = [result.head_id for result in self.head_results]
        if len(set(head_ids)) != len(head_ids):
            raise ValueError("batch head result IDs must be unique")
        distribution_ids = [
            result.distribution.distribution_id
            for result in self.head_results
            if result.distribution is not None
        ]
        if len(set(distribution_ids)) != len(distribution_ids):
            raise ValueError("batch distribution IDs must be unique")
        _validate_unique_attempts(self.attempts)
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


def _resource_compliance(
    limits: InferenceResourceLimits,
    attempts: tuple[AttemptUsage, ...],
) -> ResourceCompliance:
    """Compare all attempts, including late attempts, without dropping evidence."""

    _validate_unique_attempts(attempts)
    total_input = sum(attempt.input_utf8_bytes for attempt in attempts)
    total_output = sum(attempt.output_utf8_bytes for attempt in attempts)
    violations: list[ResourceViolation] = []
    if len(attempts) > limits.max_attempts:
        violations.append(ResourceViolation.MAX_ATTEMPTS)
    if total_input > limits.max_total_input_utf8_bytes:
        violations.append(ResourceViolation.MAX_TOTAL_INPUT_UTF8_BYTES)
    if total_output > limits.max_total_output_utf8_bytes:
        violations.append(ResourceViolation.MAX_TOTAL_OUTPUT_UTF8_BYTES)
    portable_totals = total_input <= MAX_SAFE_COUNTER and total_output <= MAX_SAFE_COUNTER
    if not portable_totals:
        violations.append(ResourceViolation.AGGREGATE_COUNTER_OVERFLOW)
    return ResourceCompliance(
        compliant=not violations,
        violations=tuple(violations),
        observed_attempts=len(attempts),
        total_input_utf8_bytes=total_input if total_input <= MAX_SAFE_COUNTER else None,
        total_output_utf8_bytes=total_output if total_output <= MAX_SAFE_COUNTER else None,
    )


def validate_decision_result(
    request: DecisionRequest,
    result: DecisionResult,
) -> ResourceCompliance:
    """Validate exact identities and report resource compliance separately."""

    request = _validated_contract_snapshot(
        request,
        DecisionRequest,
        label="decision request",
    )
    result = _validated_contract_snapshot(
        result,
        DecisionResult,
        label="decision result",
    )
    if result.request_id != request.request_id:
        raise ValueError("result request ID does not match request")
    _validate_head_result(request.head, result.head_result)
    return _resource_compliance(request.resource_limits, result.attempts)


def validate_decision_batch_result(
    request: DecisionBatchRequest,
    result: DecisionBatchResult,
) -> ResourceCompliance:
    """Validate identities/fallback and report resource compliance separately."""

    request = _validated_contract_snapshot(
        request,
        DecisionBatchRequest,
        label="decision batch request",
    )
    result = _validated_contract_snapshot(
        result,
        DecisionBatchResult,
        label="decision batch result",
    )
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
    return _resource_compliance(request.resource_limits, result.attempts)


_SCHEMA_CONTAINER_KEYWORDS = {
    "dependentSchemas",
    "properties",
}
_SCHEMA_SINGLE_KEYWORDS = {
    "additionalProperties",
    "contains",
    "contentSchema",
    "else",
    "if",
    "items",
    "not",
    "propertyNames",
    "then",
    "unevaluatedItems",
    "unevaluatedProperties",
}
_SCHEMA_ARRAY_KEYWORDS = {"allOf", "anyOf", "oneOf", "prefixItems"}
# Python's backtracking regex engine has no execution budget. Candidate-v1
# therefore keeps executable schema regex out of output validation until a
# safe engine or a hard execution boundary is qualified. The "regex" format
# remains a syntax-only instance check and does not match against caller data.
_UNSUPPORTED_EXECUTABLE_SCHEMA_KEYWORDS = {"pattern", "patternProperties"}
_SCHEMA_ANNOTATION_KEYWORDS = {
    "$comment",
    "$schema",
    "default",
    "deprecated",
    "description",
    "examples",
    "readOnly",
    "title",
    "writeOnly",
}


def _jsonschema_types() -> tuple[Any, Any]:
    try:
        from jsonschema import Draft202012Validator, FormatChecker
    except ImportError as exc:  # pragma: no cover - packaging gate, not unit environment
        raise RuntimeError(
            "generation schema validation requires the direct jsonschema runtime dependency"
        ) from exc
    return Draft202012Validator, FormatChecker


def _is_date(value: object) -> bool:
    if not isinstance(value, str):
        return True
    return bool(_DATE_PATTERN.fullmatch(value) and date.fromisoformat(value))


def _is_email(value: object) -> bool:
    if not isinstance(value, str):
        return True
    return "@" in value


def _is_idn_hostname(value: object) -> bool:
    if not isinstance(value, str):
        return True
    idna.encode(value)
    return True


def _is_ipv4(value: object) -> bool:
    if not isinstance(value, str):
        return True
    return bool(IPv4Address(value))


def _is_ipv6(value: object) -> bool:
    if not isinstance(value, str):
        return True
    address = IPv6Address(value)
    return not getattr(address, "scope_id", "")


def _is_regex(value: object) -> bool:
    if not isinstance(value, str):
        return True
    return bool(re.compile(value))


def _is_time(value: object) -> bool:
    if not isinstance(value, str):
        return True
    match = _TIME_PATTERN.fullmatch(value)
    if match is None:
        return False
    if match["second"] != "60":
        return True

    local_minutes = int(match["hour"]) * 60 + int(match["minute"])
    if match["offset_sign"] is None:
        offset_minutes = 0
    else:
        offset_minutes = (
            int(match["offset_hour"]) * 60 + int(match["offset_minute"])
        )
        if match["offset_sign"] == "-":
            offset_minutes = -offset_minutes
    return (local_minutes - offset_minutes) % (24 * 60) == 23 * 60 + 59


def _is_uuid(value: object) -> bool:
    if not isinstance(value, str):
        return True
    UUID(value)
    return all(value[position] == "-" for position in (8, 13, 18, 23))


def _contract_format_checker(format_checker_class: Any) -> Any:
    """Build candidate-v1's fixed best-effort format assertion subset.

    This is not the complete Draft 2020-12 Format-Assertion vocabulary.
    In particular, email and idn-email intentionally use jsonschema's minimal
    at-sign sanity check rather than claiming full RFC mailbox validation.
    """

    checker = format_checker_class(formats=())
    checker.checks("date", raises=ValueError)(_is_date)
    checker.checks("email")(_is_email)
    checker.checks("idn-email")(_is_email)
    checker.checks(
        "idn-hostname",
        raises=(idna.IDNAError, UnicodeError),
    )(_is_idn_hostname)
    checker.checks("ipv4", raises=AddressValueError)(_is_ipv4)
    checker.checks("ipv6", raises=AddressValueError)(_is_ipv6)
    checker.checks("regex", raises=re.error)(_is_regex)
    checker.checks("time", raises=ValueError)(_is_time)
    checker.checks("uuid", raises=ValueError)(_is_uuid)
    return checker


def _reject_unknown_schema_semantics(
    schema: bool | Mapping[str, Any],
    validator_class: Any,
    location: str = "$",
) -> None:
    if isinstance(schema, bool):
        return
    declared_dialect = schema.get("$schema")
    if "$schema" in schema and declared_dialect != JSON_SCHEMA_DIALECT:
        raise ValueError(
            f"output schema declares an unsupported dialect at {location}: "
            f"{declared_dialect}"
        )
    known = set(validator_class.VALIDATORS) | _SCHEMA_ANNOTATION_KEYWORDS
    for keyword, value in schema.items():
        if keyword in _UNSUPPORTED_EXECUTABLE_SCHEMA_KEYWORDS:
            raise ValueError(
                f"unsupported executable schema regex at {location}: {keyword}"
            )
        if keyword in {
            "$anchor",
            "$defs",
            "$dynamicAnchor",
            "$dynamicRef",
            "$id",
            "$ref",
        }:
            raise ValueError(
                f"output schema reference semantics are unsupported at {location}: {keyword}"
            )
        if keyword not in known:
            raise ValueError(f"unsupported output schema keyword at {location}: {keyword}")
        if keyword == "format" and (
            not isinstance(value, str)
            or value not in _SUPPORTED_OUTPUT_SCHEMA_FORMATS
        ):
            raise ValueError(f"unsupported output schema format at {location}: {value}")
        if keyword in _SCHEMA_CONTAINER_KEYWORDS and isinstance(value, Mapping):
            for name, nested in value.items():
                if isinstance(nested, (bool, Mapping)):
                    _reject_unknown_schema_semantics(
                        nested,
                        validator_class,
                        f"{location}/{keyword}/{name}",
                    )
        elif keyword in _SCHEMA_SINGLE_KEYWORDS and isinstance(value, (bool, Mapping)):
            _reject_unknown_schema_semantics(
                value,
                validator_class,
                f"{location}/{keyword}",
            )
        elif keyword in _SCHEMA_ARRAY_KEYWORDS and isinstance(value, list):
            for index, nested in enumerate(value):
                if isinstance(nested, (bool, Mapping)):
                    _reject_unknown_schema_semantics(
                        nested,
                        validator_class,
                        f"{location}/{keyword}/{index}",
                    )


def _validate_finite_json(value: JsonValue, location: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite JSON number at {location}")
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_finite_json(item, f"{location}/{index}")
    elif isinstance(value, dict):
        for key, item in value.items():
            _validate_finite_json(item, f"{location}/{key}")


def _reject_model_json_values(value: object, location: str) -> None:
    pending = [(value, location)]
    seen_containers: set[int] = set()
    while pending:
        item, item_location = pending.pop()
        if isinstance(item, BaseModel):
            raise ValueError(f"JSON value contains a model object at {item_location}")
        if not isinstance(item, (Mapping, list)):
            continue
        identity = id(item)
        if identity in seen_containers:
            continue
        seen_containers.add(identity)
        if isinstance(item, Mapping):
            pending.extend(
                (nested, f"{item_location}/value/{index}")
                for index, nested in enumerate(item.values())
            )
        else:
            pending.extend(
                (nested, f"{item_location}/{index}")
                for index, nested in enumerate(item)
            )


def _build_output_validator(schema: Mapping[str, JsonValue]) -> Any:
    validator_class, format_checker_class = _jsonschema_types()
    format_checker = _contract_format_checker(format_checker_class)
    if schema.get("$schema") != JSON_SCHEMA_DIALECT:
        raise ValueError("output schema must declare the supported Draft 2020-12 dialect")
    _validate_finite_json(dict(schema))
    _reject_unknown_schema_semantics(schema, validator_class)
    try:
        validator_class.check_schema(schema)
    except Exception as exc:
        raise ValueError(f"invalid output schema: {exc}") from exc
    return validator_class(schema, format_checker=format_checker)


class GenerationRequest(VersionedContract):
    """A request for schema-constrained proposals, separate from decisions."""

    request_id: OpaqueId
    operation: OpaqueId
    sources: tuple[SourceEvidence, ...]
    spans: tuple[SourceSpan, ...]
    processing_policy: ReferenceHandle
    output_schema_dialect: Literal[JSON_SCHEMA_DIALECT]
    output_schema: dict[str, JsonValue]
    resource_limits: InferenceResourceLimits
    deadline_ms: SafeCounter

    @model_validator(mode="after")
    def validate_request(self) -> "GenerationRequest":
        _validate_source_evidence(self.sources, self.spans)
        if not self.output_schema:
            raise ValueError("generation requires a declared output schema")
        _build_output_validator(self.output_schema)
        if self.deadline_ms == 0:
            raise ValueError("deadline_ms must be positive")
        return self


class GenerationProposal(ContractModel):
    proposal_id: OpaqueId
    output: JsonValue
    support_span_ids: tuple[OpaqueId, ...]
    machine_authored: Literal[True]

    @model_validator(mode="after")
    def validate_proposal(self) -> "GenerationProposal":
        _validate_finite_json(self.output)
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
    def validate_outcome(self) -> "GenerationResult":
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
        _validate_unique_attempts(self.attempts)
        return self


def validate_generation_result(
    request: GenerationRequest,
    result: GenerationResult,
) -> ResourceCompliance:
    """Validate schema/source support and report resource compliance separately."""

    if isinstance(request, GenerationRequest):
        _reject_model_json_values(
            request.output_schema,
            "generation request/output_schema",
        )
    if isinstance(result, GenerationResult) and isinstance(result.proposals, tuple):
        for index, proposal in enumerate(result.proposals):
            if isinstance(proposal, GenerationProposal):
                _reject_model_json_values(
                    proposal.output,
                    f"generation result/proposals/{index}/output",
                )
            elif isinstance(proposal, Mapping) and "output" in proposal:
                _reject_model_json_values(
                    proposal["output"],
                    f"generation result/proposals/{index}/output",
                )
    request = _validated_contract_snapshot(
        request,
        GenerationRequest,
        label="generation request",
    )
    result = _validated_contract_snapshot(
        result,
        GenerationResult,
        label="generation result",
    )
    if result.request_id != request.request_id:
        raise ValueError("result request ID does not match request")
    allowed_span_ids = {span.span_id for span in request.spans}
    output_validator = _build_output_validator(request.output_schema)
    for proposal in result.proposals:
        _validate_finite_json(proposal.output)
        if not set(proposal.support_span_ids).issubset(allowed_span_ids):
            raise ValueError("proposal references an unknown support span ID")
        first_error = next(output_validator.iter_errors(proposal.output), None)
        if first_error is not None:
            raise ValueError(
                "proposal output does not match declared schema: "
                f"{first_error.message}"
            )
    return _resource_compliance(request.resource_limits, result.attempts)


__all__ = [
    "AttemptUsage",
    "CalibrationClaim",
    "CandidateScore",
    "DecisionBatchRequest",
    "DecisionBatchResult",
    "DecisionCandidate",
    "DecisionHead",
    "DecisionHeadResult",
    "DecisionOutcome",
    "DecisionRequest",
    "DecisionResult",
    "FallbackGranularity",
    "GenerationOutcome",
    "GenerationProposal",
    "GenerationRequest",
    "GenerationResult",
    "InferenceResourceLimits",
    "JSON_SCHEMA_DIALECT",
    "ResourceCompliance",
    "ResourceViolation",
    "ScoreDistribution",
    "ScoreKind",
    "SourceEvidence",
    "SourceSpan",
    "validate_decision_batch_result",
    "validate_decision_result",
    "validate_generation_result",
    "validate_source_span",
]
