"""Validated boundaries for bounded decisions and generative proposals.

Decision operations may only select supplied candidate identities. Generation
operations return schema-checked, machine-authored proposals with explicit
source support. Neither kind of result grants authority or publishes content.

Output-schema execution has its own bound, separate from the input budgets
(``MAX_SCHEMA_EXECUTION_UNITS``, ``_KEYWORD_UNITS`` and
``MAX_OUTPUT_SCHEMA_SUBSCHEMAS`` below): every keyword application and every
node hashed for ``uniqueItems`` or ``enum`` is charged to a per-call meter.  An
output schema or result that fits every input budget can still be refused when
applying the one to the other costs more than the meter allows (a strict schema
over very many small objects, or one with more than
``MAX_OUTPUT_SCHEMA_SUBSCHEMAS`` subschemas).  The refusal is a ``ValueError``
carrying exactly ``SCHEMA_EXECUTION_MESSAGE`` ("output schema execution exceeds
its budget"): it is the whole message when ``validate_generation_result`` refuses
an output, and it is inside pydantic's ``ValidationError`` (a ``ValueError``) when
the subschema cap refuses a ``GenerationRequest`` at construction.  It is raised
before any output is accepted, never a partial validation; shrink the output or
simplify the schema.

One value-size bound is part of that contract: ``MAX_REGEX_CHARACTERS`` (8,192).
The ``regex`` format check does not compile a string longer than that, because
Python's regex compiler is superlinear in the pattern (alternatives with long
shared prefixes take quadratic time).  Such a string is not judged a valid or an
invalid regex: it is refused with the same ``SCHEMA_EXECUTION_MESSAGE``
``ValueError``, and the refusal is of the whole output, wherever the ``format``
keyword sits, including inside ``anyOf`` or ``not`` where another branch could
have accepted the value.  Ordinary patterns are far below the bound.  Its
length is also charged to the meter when the pattern is first checked.

The constants, ``MAX_REGEX_CHARACTERS`` included, are implementation limits of
this candidate, not a service-level commitment.
"""

from __future__ import annotations

import math
from fractions import Fraction
import struct
import re
from collections.abc import Callable, Mapping
from datetime import date
from enum import Enum
from ipaddress import AddressValueError, IPv4Address, IPv6Address
from typing import Annotated, Any, Literal

import idna
from pydantic import BaseModel, Field, JsonValue, model_validator

from contracts_common import ContractModel, OpaqueId, SafeCounter, VersionedContract
from contracts_engines import QualificationClaim, _validated_contract_snapshots
from contracts_limits import STANDARD_MAX_EDGES
from contracts_native import is_subclass
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
_UUID_PATTERN = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)
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
        # The strict UTF-8 codec refuses exactly the surrogate code points, in C: this runs for every
        # occurrence of the evidence a request holds, so a per-character Python loop would not do.
        try:
            str.encode(self.text, "utf-8")
        except UnicodeEncodeError:
            raise ValueError("source evidence must be valid UTF-8 text") from None
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

    # Both roots are typed, captured and accounted before either is reconstructed, so a public
    # view reached while one is rebuilt (a dict-subclass source version, for example) cannot
    # repair the other's native state first.
    evidence, span = _validated_contract_snapshots(
        (evidence, SourceEvidence, "source evidence"),
        (span, SourceSpan, "source span"),
    )
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

    request, result = _validated_contract_snapshots(
        (request, DecisionRequest, "decision request"),
        (result, DecisionResult, "decision result"),
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

    request, result = _validated_contract_snapshots(
        (request, DecisionBatchRequest, "decision batch request"),
        (result, DecisionBatchResult, "decision batch result"),
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
    if not is_subclass(type(value), str):
        return True
    return bool(_DATE_PATTERN.fullmatch(value) and date.fromisoformat(value))


def _is_email(value: object) -> bool:
    if not is_subclass(type(value), str):
        return True
    return "@" in value


def _is_idn_hostname(value: object) -> bool:
    if not is_subclass(type(value), str):
        return True
    # No hostname is longer than 253 characters, or 254 with its root dot (its ASCII form is never shorter
    # than its text); idna enforces the exact encoded limits, but would scan all of a longer string first.
    if str.__len__(value) > (254 if str.endswith(value, ".") else 253):  # type: ignore[arg-type]
        return False
    idna.encode(value)
    return True


def _is_ipv4(value: object) -> bool:
    if not is_subclass(type(value), str):
        return True
    return bool(IPv4Address(value))


def _is_ipv6(value: object) -> bool:
    if not is_subclass(type(value), str):
        return True
    address = IPv6Address(value)
    return not getattr(address, "scope_id", "")


# The longest regex the format check will compile.  Python's compiler is superlinear in the length (alternatives
# with long shared prefixes take quadratic time: about 12 ms at this length, seconds at 320,000 characters), and a
# schema-sized regex is a few hundred characters, so a longer one is refused as over budget rather than judged.
MAX_REGEX_CHARACTERS = 8_192


def _is_regex(value: object) -> bool:
    if not is_subclass(type(value), str):
        return True
    if str.__len__(value) > MAX_REGEX_CHARACTERS:  # type: ignore[arg-type]
        raise ValueError(SCHEMA_EXECUTION_MESSAGE)
    return bool(re.compile(value))


def _is_time(value: object) -> bool:
    if not is_subclass(type(value), str):
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
    # Only the canonical 36-character hex-and-hyphen form: ``UUID()`` alone also accepts braces, a
    # ``urn:uuid:`` prefix, missing or trailing hyphens and non-ASCII digits.
    if not is_subclass(type(value), str):
        return True
    return _UUID_PATTERN.fullmatch(value) is not None


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
    checker.checks(
        "regex",
        raises=(re.error, RecursionError, OverflowError),
    )(_is_regex)
    checker.checks("time", raises=ValueError)(_is_time)
    checker.checks("uuid")(_is_uuid)
    return checker


def _reject_unknown_schema_semantics(
    schema: bool | Mapping[str, Any],
    validator_class: Any,
    location: str = "$",
    seen: list[int] | None = None,
) -> None:
    # Checking a schema against the dialect's meta-schema applies dozens of keywords to every subschema,
    # and jsonschema runs that check on its own (unmetered) validator, so the subschemas are counted here.
    seen = [0] if seen is None else seen
    seen[0] += 1
    if seen[0] > MAX_OUTPUT_SCHEMA_SUBSCHEMAS:
        raise ValueError(SCHEMA_EXECUTION_MESSAGE)
    if is_subclass(type(schema), bool):
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
            not is_subclass(type(value), str)
            or value not in _SUPPORTED_OUTPUT_SCHEMA_FORMATS
        ):
            raise ValueError(f"unsupported output schema format at {location}: {value}")
        if keyword in _SCHEMA_CONTAINER_KEYWORDS and is_subclass(type(value), Mapping):
            for name, nested in value.items():
                if is_subclass(type(nested), (bool, Mapping)):
                    _reject_unknown_schema_semantics(
                        nested,
                        validator_class,
                        f"{location}/{keyword}/{name}",
                        seen,
                    )
        elif keyword in _SCHEMA_SINGLE_KEYWORDS and is_subclass(type(value), (bool, Mapping)):
            _reject_unknown_schema_semantics(
                value,
                validator_class,
                f"{location}/{keyword}",
                seen,
            )
        elif keyword in _SCHEMA_ARRAY_KEYWORDS and is_subclass(type(value), list):
            for index, nested in enumerate(value):
                if is_subclass(type(nested), (bool, Mapping)):
                    _reject_unknown_schema_semantics(
                        nested,
                        validator_class,
                        f"{location}/{keyword}/{index}",
                        seen,
                    )


def _validate_json_value(value: JsonValue, location: str = "$") -> None:
    """Reject JSON values that cannot cross the finite UTF-8 wire boundary."""

    if is_subclass(type(value), str):
        try:
            value.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ValueError(f"invalid Unicode JSON string at {location}") from exc
    if is_subclass(type(value), float) and not math.isfinite(value):
        raise ValueError(f"non-finite JSON number at {location}")
    if is_subclass(type(value), list):
        for index, item in enumerate(value):
            _validate_json_value(item, f"{location}/{index}")
    elif is_subclass(type(value), dict):
        for index, (key, item) in enumerate(value.items()):
            _validate_json_value(key, f"{location}/key/{index}")
            _validate_json_value(item, f"{location}/value/{index}")


# Schema execution is bounded on its own: the discovery and expansion budgets bound the schema and the
# outputs, not the work of applying one to the other, which multiplies (every subschema of an applicator
# runs against every item it reaches) and, for a few keywords, grows faster than either input.
SCHEMA_EXECUTION_MESSAGE = "output schema execution exceeds its budget"
MAX_SCHEMA_EXECUTION_UNITS = 8 * STANDARD_MAX_EDGES
# One keyword application weighs as much as this many entries: about 130,000 applications per call.
_KEYWORD_UNITS = 8
# The most subschemas an output schema may have (checking it against the meta-schema costs per subschema).
MAX_OUTPUT_SCHEMA_SUBSCHEMAS = 2_048


class _SchemaExecutionMeter:
    """Work of applying an output schema during one validation call; refuses past the limit."""

    __slots__ = ("left",)

    def __init__(self, limit: int = MAX_SCHEMA_EXECUTION_UNITS) -> None:
        self.left = limit

    def charge(self, units: int) -> None:
        self.left -= units
        if self.left < 0:
            raise ValueError(SCHEMA_EXECUTION_MESSAGE)


# Work that scales with the length of one value (an integer's bytes, a string a format check scans) is
# charged by that length: one unit per this many bytes or characters, on top of the per-node unit.
_BYTES_PER_UNIT = 32
# Longest text of a value a mismatch message quotes.
_MESSAGE_TEXT = 40


def _container_size(value: object) -> int:
    return len(value) if is_subclass(type(value), (list, dict)) else 0  # type: ignore[arg-type]


def _string_units(value: object) -> int:
    """The length of a string instance in units (0 for anything else), read without the value's own hooks."""

    return str.__len__(value) // _BYTES_PER_UNIT if is_subclass(type(value), str) else 0  # type: ignore[arg-type]


def _brief(value: object) -> str:
    """A bounded description of a value for a message: never longer than a few dozen characters, however
    large the value is, and built without walking it (repr of a large value is linear, of a huge integer an error)."""

    kind = type(value)
    if is_subclass(kind, str):
        text = repr(str.__getitem__(value, slice(0, _MESSAGE_TEXT)))  # type: ignore[arg-type]
        length = str.__len__(value)  # type: ignore[arg-type]
        return text if length <= _MESSAGE_TEXT else f"{text}... ({length} characters)"
    if kind is int or kind is _LargeInt:
        bits = int.bit_length(value)  # type: ignore[arg-type]
        return repr(value) if bits <= _LARGE_INT_BITS else f"<integer of {bits} bits>"
    if kind is bool or kind is float or value is None:
        return repr(value)
    if is_subclass(kind, list):
        return f"<array of {len(value)} items>"  # type: ignore[arg-type]
    if is_subclass(kind, dict):
        return f"<object of {len(value)} members>"  # type: ignore[arg-type]
    return f"<{kind.__name__}>"


# A string longer than this, or a container with more nodes than _SMALL_NODES, is quoted by a short description
# in the messages jsonschema builds from the instance.
_LONG_TEXT = 256
_SMALL_NODES = 16
_LARGE_INT_BITS = 64


class _LongText(str):
    """A long string whose repr is bounded; equal to, hashed and measured as the string it copies."""

    __slots__ = ()

    def __repr__(self) -> str:
        return _brief(self)


class _LargeInt(int):
    """An integer too large for its repr to be cheap or safe (str() of a very large int raises); the
    value is unchanged: equal, hashed and measured as the integer it copies."""

    __slots__ = ()

    def __repr__(self) -> str:
        return _brief(self)

    __str__ = __repr__


class _LargeList(list):  # type: ignore[type-arg]
    __slots__ = ()

    def __repr__(self) -> str:
        return _brief(self)


class _LargeDict(dict):  # type: ignore[type-arg]
    __slots__ = ()

    def __repr__(self) -> str:
        return _brief(self)


def _bounded_view(value: object) -> object:
    """A copy of an output with every long string and large container replaced by an equal one whose repr
    is short.  jsonschema builds a message quoting the instance whenever a keyword fails, including inside
    anyOf and not where the failure is discarded, so a large instance would otherwise cost its full length
    per application; with this view no message is long and nothing needs refusing for having built one.
    Shared containers are copied once (a value repeated by reference stays repeated), and every original is
    kept alive by the memo for the copy's lifetime, so its id is its own."""

    memo: dict[int, tuple[object, object, int]] = {}

    def walk(item: object) -> tuple[object, int]:
        kind = type(item)
        scalar = is_subclass(kind, str) or (kind is int and int.bit_length(item) > _LARGE_INT_BITS)  # type: ignore[arg-type]
        if not scalar and not is_subclass(kind, (list, dict)):
            return item, 1
        known = memo.get(id(item))  # a value repeated by reference is copied once, whatever its size
        if known is not None and known[0] is item:
            return known[1], known[2]
        if scalar:
            if kind is int:
                copied_scalar: object = int.__new__(_LargeInt, item)  # type: ignore[call-overload]
            else:
                text = str.__str__(item)  # type: ignore[arg-type]
                copied_scalar = str.__new__(_LongText, text) if len(text) > _LONG_TEXT else text
            memo[id(item)] = (item, copied_scalar, 1)
            return copied_scalar, 1
        nodes = 1
        if is_subclass(kind, list):
            elements = []
            for element in item:  # type: ignore[attr-defined]
                copied, count = walk(element)
                elements.append(copied)
                nodes += count
            copy: object = _LargeList(elements) if nodes > _SMALL_NODES else elements
        else:
            members = {}
            for key, element in item.items():  # type: ignore[attr-defined]
                copied_key, key_count = walk(key)
                copied, count = walk(element)
                members[copied_key] = copied
                nodes += key_count + count
            copy = _LargeDict(members) if nodes > _SMALL_NODES else members
        memo[id(item)] = (item, copy, nodes)
        return copy, nodes

    try:
        return walk(value)[0]
    except RecursionError:
        raise ValueError(SCHEMA_EXECUTION_MESSAGE) from None


class _Identity:
    """A node of a JSON value with JSON Schema equality, hashed once when it is built.

    ``key`` is a tag with the node's content; its hash is cached, so a set lookup or a comparison costs
    the node, not the subtree, and tuples are never re-hashed.  Numbers are keyed by their canonical
    bytes (below), whose hash is Python's randomized byte hash: an attacker cannot choose integers that
    collide in ``hash(int)`` (every multiple of ``sys.hash_info.modulus`` hashes to 0) to turn a set into
    a list.  Equality compares the cached hashes first, then the content.
    """

    __slots__ = ("hash", "key")

    def __init__(self, key: tuple[object, ...]) -> None:
        self.key = key
        self.hash = hash(key)

    def __hash__(self) -> int:
        return self.hash

    def __eq__(self, other: object) -> bool:
        return type(other) is _Identity and self.hash == other.hash and self.key == other.key  # type: ignore[attr-defined]


def _number_key(value: int | float) -> tuple[object, ...]:
    """The canonical key of a JSON number: equal numbers (1 and 1.0, 0 and -0.0) share it, distinct numbers
    (2**53 + 1 and 2.0**53, big integers, nearby floats) never do, and its hash is a bytes hash."""

    if type(value) is float:
        if value != value:
            return ("nan",)
        if value.is_integer():
            value = int(value)
        else:
            return ("f", struct.pack(">d", value))
    size = (int.bit_length(value) + 8) // 8  # one bit for the sign
    return ("n", int.to_bytes(value, size, "big", signed=True))


def _text_of(value: object, meter: "_SchemaExecutionMeter") -> str:
    """A string's exact text for an identity, charged by its length first (the copy and the hash scan it).
    Our own long-text class is used as it is, so it is neither copied nor, once hashed, hashed again."""

    meter.charge(_string_units(value))
    return value if type(value) is str or type(value) is _LongText else str.__str__(value)  # type: ignore[arg-type,return-value]


# Operand bit length of a float in the exact fallback: its fraction has at most this many bits.
_FLOAT_BITS = 1_100
_LIMB_BITS = 30
_LIMB_PRODUCTS_PER_UNIT = 4_096


# Comparing two integers whose high limbs match scans the shorter one: about 0.35 ns per limb, a unit being about
# a microsecond.
_LIMBS_PER_COMPARE_UNIT = 2_048
_BOUND_KEYWORDS = frozenset({"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"})


def _comparison_units(bound: object, instance: object) -> int:
    def limbs(value: object) -> int:
        return int.bit_length(value) // _LIMB_BITS + 1 if is_subclass(type(value), int) and type(value) is not bool else 0  # type: ignore[arg-type]

    return min(limbs(bound), limbs(instance)) // _LIMBS_PER_COMPARE_UNIT


def _division_units(dividend: object, divisor: object) -> int:
    """Units for the arbitrary-precision division behind multipleOf: schoolbook cost, limbs times limbs."""

    def limbs(value: object) -> int:
        bits = int.bit_length(value) if is_subclass(type(value), int) and type(value) is not bool else _FLOAT_BITS  # type: ignore[arg-type]
        return bits // _LIMB_BITS + 1

    return limbs(dividend) * limbs(divisor) // _LIMB_PRODUCTS_PER_UNIT


def _json_identity(value: object, meter: _SchemaExecutionMeter) -> _Identity:
    """The identity of a JSON value with JSON Schema equality: a boolean is not a number, 1 equals 1.0,
    arrays compare in order and objects by their key set, at any depth.  One unit per node."""

    meter.charge(1)
    kind = type(value)
    if kind is bool:
        return _Identity(("bool", value))
    if is_subclass(kind, str):
        return _Identity(("str", _text_of(value, meter)))
    if is_subclass(kind, list):
        return _Identity(("array", tuple(_json_identity(item, meter) for item in value)))  # type: ignore[union-attr]
    if is_subclass(kind, dict):
        return _Identity(
            ("object", frozenset((_text_of(key, meter), _json_identity(item, meter)) for key, item in value.items()))  # type: ignore[union-attr]
        )
    if kind is int or kind is float or kind is _LargeInt:
        if kind is int or kind is _LargeInt:  # the key is the integer's whole byte string: charged by length before it is built
            meter.charge((int.bit_length(value) + 8) // 8 // _BYTES_PER_UNIT)  # type: ignore[arg-type]
        return _Identity(_number_key(value))  # type: ignore[arg-type]
    return _Identity(("value", value))  # null, or anything else a schema may carry


def _metered_validator_class(validator_class: Any, meter: _SchemaExecutionMeter) -> Any:
    """The same validator with every keyword charged to ``meter`` and uniqueItems checked in linear time.

    A keyword application costs ``_KEYWORD_UNITS`` (descending into a subschema is the expensive step)
    plus one unit per entry of the instance it inspects and of its own
    value (an enum list, required names, the subschemas of an applicator), the work it does itself; work
    it delegates to subschemas is charged when they run.  jsonschema's uniqueItems falls back to comparing
    every pair of items that cannot be sorted (objects), so it is replaced by a hashed check.
    """

    from jsonschema.exceptions import ValidationError as SchemaError
    from jsonschema.validators import extend

    def metered(
        keyword: Callable[..., Any], scans_strings: bool = False, compares_integers: bool = False
    ) -> Callable[..., Any]:
        def apply(validator: Any, value: Any, instance: Any, schema: Any) -> Any:
            meter.charge(_KEYWORD_UNITS + _container_size(instance) + _container_size(value))
            if scans_strings:
                meter.charge(_string_units(instance))
            if compares_integers:
                meter.charge(_comparison_units(value, instance))
            for error in keyword(validator, value, instance, schema):
                # jsonschema's own messages quote the whole instance (a type, length or range mismatch) and are
                # built even when an enclosing not or anyOf discards them: charge the construction by its
                # length, and keep only a bounded prefix of what a caller can see.
                message = error.message
                if len(message) > _MESSAGE_TEXT * 4:
                    meter.charge(len(message) // _BYTES_PER_UNIT)
                    error.message = message[: _MESSAGE_TEXT * 4] + "..."
                yield error

        return apply

    def unique_items(validator: Any, unique: Any, instance: Any, schema: Any) -> Any:
        if unique and validator.is_type(instance, "array"):
            seen: set[object] = set()
            for item in instance:
                identity = identity_of(item)  # hashed and charged once per object, however many applications
                if identity in seen:
                    yield SchemaError("array has non-unique elements")
                    return
                seen.add(identity)

    # jsonschema's enum compares the instance with every entry, and its equality recurses through nested
    # arrays and objects, so entries and items that keep their size one level down cost entries x items x
    # depth while the top-level sizes charged above stay small.  Each enum list is hashed once per call
    # (one unit per node, cached by the list object), the instance is hashed per application, and
    # membership is a set lookup.
    enum_sets: dict[int, tuple[object, frozenset[object]]] = {}
    instance_identities: dict[int, tuple[object, _Identity]] = {}
    const_identities: dict[int, tuple[object, _Identity]] = {}
    pool: dict[_Identity, _Identity] = {}

    def canonical(identity: _Identity) -> _Identity:
        # Equal JSON values share one identity object per call, so "is it the same value" is an identity test
        # however many separately built copies of it the schema and the output hold (a schema's repeated const
        # is a distinct copy per use).  Interning compares an identity with at most the one already pooled, a
        # walk bounded by the nodes just hashed and charged.
        try:
            return pool.setdefault(identity, identity)
        except RecursionError:
            raise ValueError(SCHEMA_EXECUTION_MESSAGE) from None

    def identity_of(instance: Any) -> _Identity:
        # An instance is hashed once per call, however many enums or consts are applied to it (jsonschema
        # compares cheaply when the type or length differs, so many small keywords over one large instance
        # must not pay for the whole instance each time); the object is kept alive so its id is its own.
        hashed = instance_identities.get(id(instance))
        if hashed is None or hashed[0] is not instance:
            hashed = instance_identities[id(instance)] = (instance, canonical(_json_identity(instance, meter)))
        return hashed[1]

    # jsonschema's enum compares the instance with every entry, and its equality recurses through nested
    # arrays and objects, so entries and items that keep their size one level down cost entries x items x
    # depth while the top-level sizes charged above stay small.  Each enum list is hashed once per call
    # (one unit per node, cached by the list object), the instance is hashed per distinct object, and
    # membership is a set lookup on canonical identities.
    def enum(validator: Any, enums: Any, instance: Any, schema: Any) -> Any:
        known = enum_sets.get(id(enums))
        if known is None or known[0] is not enums:  # the list is kept alive, so its id is its own
            known = enum_sets[id(enums)] = (
                enums,
                frozenset(canonical(_json_identity(entry, meter)) for entry in enums),
            )
        if identity_of(instance) not in known[1]:
            yield SchemaError(f"{_brief(instance)} is not one of {_container_size(enums)} allowed values")

    def const(validator: Any, expected: Any, instance: Any, schema: Any) -> Any:
        # jsonschema's const equality recurses through the whole value, which the generic charge (top-level
        # sizes) never saw.  The value and each instance are hashed once (every node charged) and made
        # canonical, so a pair is equal exactly when the two canonical identities are one object: an
        # application costs a lookup, however many copies of the const the schema repeats.
        known = const_identities.get(id(expected))
        if known is None or known[0] is not expected:
            known = const_identities[id(expected)] = (expected, canonical(_json_identity(expected, meter)))
        if identity_of(instance) is not known[1]:
            yield SchemaError(f"{_brief(expected)} was expected")

    jsonschema_multiple_of = validator_class.VALIDATORS["multipleOf"]

    def multiple_of(validator: Any, divisor: Any, instance: Any, schema: Any) -> Any:
        # jsonschema converts an integer divisor to a float for a float instance, which raises OverflowError
        # for a divisor past the float range; the exact answer is the rational quotient being whole.
        meter.charge(_division_units(instance, divisor))
        try:
            yield from jsonschema_multiple_of(validator, divisor, instance, schema)
        except OverflowError:
            if validator.is_type(instance, "number") and (Fraction(instance) / Fraction(divisor)).denominator != 1:
                yield SchemaError(f"{_brief(instance)} is not a multiple of {_brief(divisor)}")

    jsonschema_format = validator_class.VALIDATORS["format"]
    format_results: dict[tuple[str, str], tuple[str, str | None]] = {}

    def format_(validator: Any, format_name: Any, instance: Any, schema: Any) -> Any:
        # A format result is computed once per distinct value and format in a call.  A failed regex compile is
        # not cached by ``re`` (a successful one is), so under not or anyOf every application would recompile;
        # a regex is also weighted by its length, since compiling costs far more per character than a scan.
        if not is_subclass(type(format_name), str) or not (type(instance) is str or type(instance) is _LongText):
            yield from jsonschema_format(validator, format_name, instance, schema)
            return
        key = (str.__str__(format_name), instance)  # type: ignore[arg-type]
        known = format_results.get(key)
        if known is None:
            if key[0] == "regex":
                meter.charge(str.__len__(instance))
            errors = list(jsonschema_format(validator, format_name, instance, schema))
            known = format_results[key] = (str.__str__(instance), errors[0].message if errors else None)
        if known[1] is not None:
            yield SchemaError(known[1])

    keywords = {
        name: metered(keyword, scans_strings=name == "format", compares_integers=name in _BOUND_KEYWORDS)
        for name, keyword in validator_class.VALIDATORS.items()
    }
    keywords["multipleOf"] = metered(multiple_of)
    keywords["format"] = metered(format_, scans_strings=True)
    keywords["uniqueItems"] = metered(unique_items)
    keywords["enum"] = metered(enum)
    keywords["const"] = metered(const)
    return extend(validator_class, keywords)


def _executable_schema(schema: Any, root: bool = True, shared: "dict[int, tuple[object, object]] | None" = None) -> Any:
    """A copy of the schema without nested ``$schema`` keys, for execution.

    jsonschema runs a subschema that declares ``$schema`` on the validator registered for that dialect,
    not on the metered one, so a nested declaration would escape the execution budget.  Nested
    declarations must equal the root dialect (checked earlier), so removing them changes nothing.  Only
    schema positions are rewritten; data positions (enum, const, default, examples) are kept as they are.
    A large integer is wrapped once per object (``shared``), however many occurrences of it the schema has.
    """

    if not is_subclass(type(schema), dict):
        return schema
    if shared is None:
        shared = {}
    copied: dict[str, Any] = {}
    for keyword, value in schema.items():
        if keyword == "$schema" and not root:
            continue
        if keyword in _SCHEMA_CONTAINER_KEYWORDS and is_subclass(type(value), dict):
            value = {name: _executable_schema(nested, False, shared) for name, nested in value.items()}
        elif keyword in _SCHEMA_SINGLE_KEYWORDS:
            value = _executable_schema(value, False, shared)
        elif keyword in _SCHEMA_ARRAY_KEYWORDS and is_subclass(type(value), list):
            value = [_executable_schema(nested, False, shared) for nested in value]
        elif type(value) is int and int.bit_length(value) > _LARGE_INT_BITS:
            known = shared.get(id(value))  # a bound or divisor that a discarded branch would quote
            if known is None or known[0] is not value:
                known = shared[id(value)] = (value, int.__new__(_LargeInt, value))
            value = known[1]
        copied[keyword] = value
    return copied


def _build_output_validator(
    schema: Mapping[str, JsonValue], meter: _SchemaExecutionMeter | None = None
) -> Any:
    validator_class, format_checker_class = _jsonschema_types()
    format_checker = _contract_format_checker(format_checker_class)
    if schema.get("$schema") != JSON_SCHEMA_DIALECT:
        raise ValueError("output schema must declare the supported Draft 2020-12 dialect")
    _validate_json_value(dict(schema))
    _reject_unknown_schema_semantics(schema, validator_class)
    try:
        validator_class.check_schema(schema)  # bounded by the subschema limit checked above
    except Exception as exc:
        message = getattr(exc, "message", type(exc).__name__)
        raise ValueError(f"invalid output schema: {message}") from exc  # never the whole schema
    metered_class = _metered_validator_class(validator_class, meter or _SchemaExecutionMeter())
    return metered_class(_executable_schema(schema), format_checker=format_checker)


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
        _validate_json_value(self.output)
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


_PROPOSAL_LOCATION = re.compile(r"^generation result\.proposals\[\d+\]$")


def _is_json_valued_field(parent_location: str, field_name: str) -> bool:
    """The fields of a generation request and result whose values are plain JSON."""

    if field_name == "output_schema" and parent_location == "generation request":
        return True
    return field_name == "output" and _PROPOSAL_LOCATION.match(parent_location) is not None


def validate_generation_result(
    request: GenerationRequest,
    result: GenerationResult,
) -> ResourceCompliance:
    """Validate schema/source support and report resource compliance separately."""

    # Both roots are captured and accounted before anything is reconstructed or scanned; model objects
    # inside the JSON-valued fields are rejected during that same reconstruction, not in a separate
    # pre-scan that would expand (and run public views of) uncaptured, unbudgeted state.
    request, result = _validated_contract_snapshots(
        (request, GenerationRequest, "generation request"),
        (result, GenerationResult, "generation result"),
        json_guard=_is_json_valued_field,
    )
    if result.request_id != request.request_id:
        raise ValueError("result request ID does not match request")
    allowed_span_ids = {span.span_id for span in request.spans}
    output_validator = _build_output_validator(request.output_schema)
    for proposal in result.proposals:
        _validate_json_value(proposal.output)
        if not set(proposal.support_span_ids).issubset(allowed_span_ids):
            raise ValueError("proposal references an unknown support span ID")
        first_error = next(output_validator.iter_errors(_bounded_view(proposal.output)), None)
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
