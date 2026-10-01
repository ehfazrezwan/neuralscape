"""Candidate canonical lifecycle types and pure consistency checks.

The models are inert contracts.  They select no storage system, accept no
command durably, authorize no caller, mutate no record, and publish no
projection.  Callers must establish those runtime properties separately.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Annotated, Literal, TypeVar

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from contracts_bodies import MemoryBody, OpaqueEnvelopeBody, PlaintextBody
from contracts_common import ContractModel, OpaqueId, SafeCounter, VersionedContract
from contracts_references import ReferenceHandle, SourceVersion


_ModelT = TypeVar("_ModelT", bound=BaseModel)
_BASE_MODEL_DICT_DESCRIPTOR = vars(BaseModel)["__dict__"]
_BASE_MODEL_EXTRA_DESCRIPTOR = vars(BaseModel)["__pydantic_extra__"]


class _CapturedGraph:
    """Opaque native graph captured before reconstruction callbacks."""


@dataclass(frozen=True, slots=True)
class _CapturedLeaf(_CapturedGraph):
    value: object


@dataclass(frozen=True, slots=True)
class _CapturedCycle(_CapturedGraph):
    pass


@dataclass(frozen=True, slots=True)
class _CapturedModel(_CapturedGraph):
    identity: int
    model_type: type[BaseModel]
    stored_entries: tuple[tuple[object, _CapturedGraph], ...]
    extra_entries: tuple[tuple[object, _CapturedGraph], ...]
    malformed_extra: bool


@dataclass(frozen=True, slots=True)
class _CapturedDict(_CapturedGraph):
    identity: int
    entries: tuple[tuple[object, _CapturedGraph], ...]


@dataclass(frozen=True, slots=True)
class _CapturedList(_CapturedGraph):
    identity: int
    source: list[object]
    items: tuple[_CapturedGraph, ...]


@dataclass(frozen=True, slots=True)
class _CapturedTuple(_CapturedGraph):
    identity: int
    source: tuple[object, ...]
    items: tuple[_CapturedGraph, ...]


_CapturedGraphSnapshot = tuple[
    _CapturedGraph,
    dict[int, _CapturedGraph],
]


def _capture_native_contract_graph(
    value: object,
    active_containers: set[int] | None = None,
    captured_by_identity: dict[int, _CapturedGraph] | None = None,
) -> _CapturedGraph:
    """Freeze supported native backing without invoking public protocols."""

    value_type = type(value)
    if active_containers is None:
        active_containers = set()
    if captured_by_identity is None:
        captured_by_identity = {}

    if issubclass(value_type, (BaseModel, dict, list, tuple)):
        identity = id(value)
        if identity in active_containers:
            return _CapturedCycle()
        if identity in captured_by_identity:
            return captured_by_identity[identity]
        active_containers.add(identity)
        try:
            if issubclass(value_type, BaseModel):
                stored = _BASE_MODEL_DICT_DESCRIPTOR.__get__(value, BaseModel)
                stored_entries = tuple(dict.items(stored))
                try:
                    extra = _BASE_MODEL_EXTRA_DESCRIPTOR.__get__(value, BaseModel)
                except AttributeError:
                    extra = None
                if extra is None:
                    extra_entries: tuple[tuple[object, object], ...] = ()
                    malformed_extra = False
                else:
                    extra_type = type(extra)
                    malformed_extra = not issubclass(extra_type, dict)
                    extra_entries = (
                        () if malformed_extra else tuple(dict.items(extra))
                    )
                captured: _CapturedGraph = _CapturedModel(
                    identity=identity,
                    model_type=value_type,
                    stored_entries=tuple(
                        (
                            key,
                            _capture_native_contract_graph(
                                item,
                                active_containers,
                                captured_by_identity,
                            ),
                        )
                        for key, item in stored_entries
                    ),
                    extra_entries=tuple(
                        (
                            key,
                            _capture_native_contract_graph(
                                item,
                                active_containers,
                                captured_by_identity,
                            ),
                        )
                        for key, item in extra_entries
                    ),
                    malformed_extra=malformed_extra,
                )
            elif issubclass(value_type, dict):
                entries = tuple(dict.items(value))
                captured = _CapturedDict(
                    identity=identity,
                    entries=tuple(
                        (
                            key,
                            _capture_native_contract_graph(
                                item,
                                active_containers,
                                captured_by_identity,
                            ),
                        )
                        for key, item in entries
                    ),
                )
            elif issubclass(value_type, list):
                native_items = tuple(list.__iter__(value))
                captured = _CapturedList(
                    identity=identity,
                    source=value,
                    items=tuple(
                        _capture_native_contract_graph(
                            item,
                            active_containers,
                            captured_by_identity,
                        )
                        for item in native_items
                    ),
                )
            else:
                native_items = tuple(tuple.__iter__(value))
                captured = _CapturedTuple(
                    identity=identity,
                    source=value,
                    items=tuple(
                        _capture_native_contract_graph(
                            item,
                            active_containers,
                            captured_by_identity,
                        )
                        for item in native_items
                    ),
                )
            captured_by_identity[identity] = captured
            return captured
        finally:
            active_containers.remove(identity)
    return _CapturedLeaf(value)


def _materialize_captured_graph(
    value: _CapturedGraph,
    captured_by_identity: dict[int, _CapturedGraph],
    active_containers: set[int] | None = None,
) -> object:
    """Reconstruct one frozen graph with established diagnostic ordering."""

    if active_containers is None:
        active_containers = set()
    if isinstance(value, _CapturedCycle):
        raise ValueError("cyclic contract graph is not valid input")
    if isinstance(value, _CapturedLeaf):
        return value.value
    identity = value.identity
    if identity in active_containers:
        raise ValueError("cyclic contract graph is not valid input")
    active_containers.add(identity)
    try:
        if isinstance(value, _CapturedModel):
            stored_names = {key for key, _ in value.stored_entries}
            declared_fields = value.model_type.model_fields
            fields = {
                key: _materialize_captured_graph(
                    item,
                    captured_by_identity,
                    active_containers,
                )
                for key, item in value.stored_entries
            }
            if value.malformed_extra:
                raise ValueError(
                    "malformed stored contract extras are not valid input"
                )
            for key, item in value.extra_entries:
                if key in stored_names or key in declared_fields:
                    raise ValueError(
                        "conflicting stored contract field is not valid input"
                    )
                fields[key] = _materialize_captured_graph(
                    item,
                    captured_by_identity,
                    active_containers,
                )
            return fields
        if isinstance(value, _CapturedDict):
            return {
                key: _materialize_captured_graph(
                    item,
                    captured_by_identity,
                    active_containers,
                )
                for key, item in value.entries
            }

        def public_item(item: object) -> _CapturedGraph:
            item_type = type(item)
            if issubclass(item_type, (BaseModel, dict, list, tuple)):
                frozen = captured_by_identity.get(id(item))
                if frozen is not None:
                    return frozen
                return _capture_native_contract_graph(
                    item,
                    captured_by_identity=captured_by_identity,
                )
            return _CapturedLeaf(item)

        if isinstance(value, _CapturedList):
            items = value.items if type(value.source) is list else (
                public_item(item) for item in value.source
            )
            return [
                _materialize_captured_graph(
                    item,
                    captured_by_identity,
                    active_containers,
                )
                for item in items
            ]
        items = value.items if type(value.source) is tuple else (
            public_item(item) for item in value.source
        )
        return tuple(
            _materialize_captured_graph(
                item,
                captured_by_identity,
                active_containers,
            )
            for item in items
        )
    finally:
        active_containers.remove(identity)


def _native_contract_graph(
    value: object,
    active_containers: set[int] | None = None,
) -> object:
    """Materialize stored fields without dropping extras or hiding cycles."""

    captured = _capture_contract_graph(value, active_containers)
    return _materialize_contract_graph(captured)


def _capture_contract_graph(
    value: object,
    active_containers: set[int] | None = None,
) -> _CapturedGraphSnapshot:
    """Capture one graph without running its public reconstruction views."""

    captured_by_identity: dict[int, _CapturedGraph] = {}
    captured = _capture_native_contract_graph(
        value,
        active_containers,
        captured_by_identity,
    )
    return captured, captured_by_identity


def _materialize_contract_graph(
    captured: _CapturedGraphSnapshot,
) -> object:
    """Materialize one previously captured graph through established views."""

    root, captured_by_identity = captured
    return _materialize_captured_graph(root, captured_by_identity)


def _captured_model(
    model_type: type[_ModelT],
    value: object,
) -> _CapturedGraphSnapshot:
    if not isinstance(value, model_type):
        raise TypeError(f"value must be a {model_type.__name__}")
    return _capture_contract_graph(value)


def _revalidated_captured_model(
    model_type: type[_ModelT],
    captured: _CapturedGraphSnapshot,
) -> _ModelT:
    return model_type.model_validate(
        _materialize_contract_graph(captured),
        strict=True,
    )


def _revalidated_inventory_model(
    model_type: type[_ModelT],
    value: object,
    captured_by_identity: dict[int, _CapturedGraph],
) -> _ModelT:
    """Validate a public-view item, reusing entry-time state when available."""

    if not isinstance(value, model_type):
        raise TypeError(f"value must be a {model_type.__name__}")
    captured = captured_by_identity.get(id(value))
    if captured is None:
        captured = _capture_native_contract_graph(
            value,
            captured_by_identity=captured_by_identity,
        )
    return _revalidated_captured_model(
        model_type,
        (captured, captured_by_identity),
    )


def _revalidated_model(model_type: type[_ModelT], value: object) -> _ModelT:
    """Strictly reconstruct one expected model from its complete stored graph."""

    return _revalidated_captured_model(
        model_type,
        _captured_model(model_type, value),
    )


class MemoryRecordKind(str, Enum):
    ORIGINAL = "original"
    FACT = "fact"
    EPISODE = "episode"
    DERIVED_VIEW = "derived_view"


class MemoryLifecycle(str, Enum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    TOMBSTONED = "tombstoned"
    ERASED = "erased"


class ApplicabilityScope(str, Enum):
    GLOBAL = "global"
    PROJECT = "project"
    WORKSPACE = "workspace"


PlaintextMemoryBody = PlaintextBody
OpaqueEnvelopeMemoryBody = OpaqueEnvelopeBody


class MemoryRecord(VersionedContract):
    """Frozen canonical memory identity, distinct from every projection."""

    model_config = ConfigDict(frozen=True)

    id: OpaqueId
    tenant_id: OpaqueId
    owner_id: OpaqueId
    record_kind: MemoryRecordKind
    applicability: ApplicabilityScope
    project_id: OpaqueId | None
    workspace_id: OpaqueId | None
    category: OpaqueId
    author_kind: OpaqueId
    source_kind: OpaqueId
    content_revision: SafeCounter
    policy_epoch: SafeCounter
    lifecycle: MemoryLifecycle
    body: MemoryBody | None
    source_refs: tuple[ReferenceHandle, ...]
    required_parents: tuple[SourceVersion, ...]
    successor_id: OpaqueId | None
    retention_hold_ids: tuple[OpaqueId, ...]
    occurred_at: AwareDatetime | None
    recorded_at: AwareDatetime

    @model_validator(mode="after")
    def validate_record_semantics(self) -> MemoryRecord:
        if self.applicability is ApplicabilityScope.GLOBAL:
            if self.project_id is not None or self.workspace_id is not None:
                raise ValueError("global applicability cannot carry project_id or workspace_id")
        elif self.applicability is ApplicabilityScope.PROJECT:
            if self.project_id is None:
                raise ValueError("project applicability requires project_id")
            if self.workspace_id is not None:
                raise ValueError("project applicability cannot carry workspace_id")
        elif self.workspace_id is None:
            raise ValueError("workspace applicability requires workspace_id")

        if self.lifecycle is MemoryLifecycle.ERASED:
            if self.body is not None:
                raise ValueError("erased records cannot retain a body")
        elif self.body is None:
            raise ValueError("non-erased records require a body")
        _require_unique_source_ids(self.required_parents, "required_parents")
        return self


class ProcessingStage(str, Enum):
    CANONICAL = "canonical"
    EXTRACT = "extract"
    VECTOR = "vector"
    GRAPH = "graph"
    MARKDOWN = "markdown"


class StageEvidenceRequirement(ContractModel):
    """Declared evidence required for one processing-stage effect."""

    model_config = ConfigDict(frozen=True)

    stage: ProcessingStage
    effect_id: OpaqueId
    precondition_source_ids: tuple[OpaqueId, ...]
    upstream_stages: tuple[ProcessingStage, ...]
    committed_sources: tuple[SourceVersion, ...]
    minimum_output_refs: Literal[0, 1]

    @field_validator("minimum_output_refs", mode="before")
    @classmethod
    def validate_native_output_minimum(cls, value: object) -> object:
        if type(value) is not int or value not in {0, 1}:
            raise ValueError("minimum_output_refs must be the native integer 0 or 1")
        return value

    @model_validator(mode="after")
    def validate_requirement_semantics(self) -> StageEvidenceRequirement:
        if len(set(self.precondition_source_ids)) != len(
            self.precondition_source_ids
        ):
            raise ValueError("precondition_source_ids must not contain duplicates")
        if len(set(self.upstream_stages)) != len(self.upstream_stages):
            raise ValueError("upstream_stages must not contain duplicates")
        if self.stage in self.upstream_stages:
            raise ValueError("a stage cannot depend on itself")
        _require_unique_source_ids(self.committed_sources, "committed_sources")
        return self


class Intent(VersionedContract):
    """Frozen validated command data; construction is not durable acceptance.

    Callers must revalidate data entering from unchecked construction or copy
    operations; Pydantic's low-level bypasses are outside this guarantee.
    """

    model_config = ConfigDict(frozen=True)

    id: OpaqueId
    tenant_id: OpaqueId
    actor_id: OpaqueId
    credential_id: OpaqueId
    operation: OpaqueId
    target_refs: tuple[ReferenceHandle, ...]
    request_digest: Annotated[str, Field(min_length=1, max_length=512)]
    idempotency_key: OpaqueId
    source_preconditions: tuple[SourceVersion, ...]
    stage_requirements: tuple[StageEvidenceRequirement, ...]
    correlation_id: OpaqueId
    accepted_at: AwareDatetime

    @model_validator(mode="after")
    def validate_stage_requirements(self) -> Intent:
        if any(item.tenant_id != self.tenant_id for item in self.target_refs):
            raise ValueError("target_refs must match intent tenant_id")
        _require_unique_source_ids(self.source_preconditions, "source_preconditions")
        if not self.stage_requirements:
            raise ValueError("stage_requirements must not be empty")

        requirements = {item.stage: item for item in self.stage_requirements}
        if len(requirements) != len(self.stage_requirements):
            raise ValueError("stage_requirements must not contain duplicate stages")
        if len({item.effect_id for item in self.stage_requirements}) != len(
            self.stage_requirements
        ):
            raise ValueError(
                "stage_requirements must not contain duplicate effect_id values"
            )

        preconditions = {item.record_id: item for item in self.source_preconditions}
        for requirement in self.stage_requirements:
            if set(requirement.precondition_source_ids) - set(preconditions):
                raise ValueError(
                    "precondition_source_ids must select declared source_preconditions"
                )
            if set(requirement.upstream_stages) - set(requirements):
                raise ValueError(
                    "upstream_stages must select declared stage_requirements"
                )

            calculated_ids = list(requirement.precondition_source_ids)
            for upstream_stage in requirement.upstream_stages:
                calculated_ids.extend(
                    item.record_id
                    for item in requirements[upstream_stage].committed_sources
                )
            if len(calculated_ids) != len(set(calculated_ids)):
                raise ValueError(
                    "calculated stage inputs must not contain record_id collisions"
                )

        _topological_stages(requirements)
        return self


class StageStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    APPLIED = "applied"
    SKIPPED = "skipped"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SUPERSEDED = "superseded"


class StageError(ContractModel):
    model_config = ConfigDict(frozen=True)

    code: OpaqueId
    retryable: bool
    safe_message: Annotated[str, Field(min_length=1, max_length=1024)]


class StageReceipt(VersionedContract):
    """Frozen validated observation for one attempt, not aggregate completion.

    Callers must revalidate data entering from unchecked construction or copy
    operations; Pydantic's low-level bypasses are outside this guarantee.
    """

    model_config = ConfigDict(frozen=True)

    intent_id: OpaqueId
    attempt: SafeCounter
    stage: ProcessingStage
    effect_id: OpaqueId
    status: StageStatus
    applied_sources: tuple[SourceVersion, ...]
    committed_sources: tuple[SourceVersion, ...]
    output_refs: tuple[ReferenceHandle, ...]
    error: StageError | None
    started_at: AwareDatetime | None
    finished_at: AwareDatetime | None

    @model_validator(mode="after")
    def validate_receipt_semantics(self) -> StageReceipt:
        if self.attempt < 1:
            raise ValueError("attempt must be at least 1")
        _require_unique_source_ids(self.applied_sources, "applied_sources")
        _require_unique_source_ids(self.committed_sources, "committed_sources")
        _require_unique_reference_handles(self.output_refs, "output_refs")

        terminal = self.status in {
            StageStatus.APPLIED,
            StageStatus.SKIPPED,
            StageStatus.FAILED,
            StageStatus.CANCELLED,
            StageStatus.SUPERSEDED,
        }
        if terminal and self.finished_at is None:
            raise ValueError("terminal stage receipts require finished_at")
        if not terminal and self.finished_at is not None:
            raise ValueError("non-terminal stage receipts cannot have finished_at")
        if self.status is StageStatus.PENDING:
            if self.started_at is not None:
                raise ValueError("pending stage receipts cannot have started_at")
            if self.applied_sources or self.committed_sources or self.output_refs:
                raise ValueError("pending stage receipts cannot carry result fields")
        if self.status is not StageStatus.APPLIED and self.committed_sources:
            raise ValueError(
                "only applied stage receipts may carry committed_sources"
            )
        if self.status is StageStatus.PROCESSING and self.started_at is None:
            raise ValueError("processing stage receipts require started_at")
        if self.status in {StageStatus.APPLIED, StageStatus.FAILED}:
            if self.started_at is None:
                raise ValueError(
                    f"{self.status.value} stage receipts require started_at"
                )
        if self.started_at is not None and self.finished_at is not None:
            if self.finished_at < self.started_at:
                raise ValueError("finished_at cannot precede started_at")
        if self.status is StageStatus.FAILED:
            if self.error is None:
                raise ValueError("failed stage receipts require an error")
        elif self.error is not None:
            raise ValueError("only failed stage receipts may carry an error")
        return self


class IntentStatus(str, Enum):
    ACCEPTED = "accepted"
    PROCESSING = "processing"
    APPLIED = "applied"
    PARTIAL = "partial"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SUPERSEDED = "superseded"


_LEGAL_INTENT_TRANSITIONS: dict[IntentStatus, frozenset[IntentStatus]] = {
    IntentStatus.ACCEPTED: frozenset(
        {
            IntentStatus.PROCESSING,
            IntentStatus.PARTIAL,
            IntentStatus.FAILED,
            IntentStatus.CANCELLED,
            IntentStatus.SUPERSEDED,
        }
    ),
    IntentStatus.PROCESSING: frozenset(
        {
            IntentStatus.APPLIED,
            IntentStatus.PARTIAL,
            IntentStatus.FAILED,
            IntentStatus.CANCELLED,
            IntentStatus.SUPERSEDED,
        }
    ),
    IntentStatus.APPLIED: frozenset(),
    IntentStatus.PARTIAL: frozenset(),
    IntentStatus.FAILED: frozenset(),
    IntentStatus.CANCELLED: frozenset(),
    IntentStatus.SUPERSEDED: frozenset(),
}


def is_legal_intent_transition(current: IntentStatus, target: IntentStatus) -> bool:
    """Return whether one persisted intent state may advance to another."""

    if type(current) is not IntentStatus or not any(
        current is member for member in IntentStatus
    ):
        raise TypeError("current must be an IntentStatus")
    if type(target) is not IntentStatus or not any(
        target is member for member in IntentStatus
    ):
        raise TypeError("target must be an IntentStatus")
    return target in _LEGAL_INTENT_TRANSITIONS[current]


def source_versions_match(
    expected: SourceVersion,
    observed: SourceVersion,
) -> bool:
    """Require exact equality after deep validation of both version witnesses."""

    if not isinstance(expected, SourceVersion):
        raise TypeError("value must be a SourceVersion")
    try:
        captured_expected = _capture_contract_graph(expected)
        expected_capture_error: Exception | None = None
    except Exception as exc:
        captured_expected = None
        expected_capture_error = exc

    if isinstance(observed, SourceVersion):
        try:
            captured_observed = _capture_contract_graph(observed)
            observed_capture_error: Exception | None = None
        except Exception as exc:
            captured_observed = None
            observed_capture_error = exc
    else:
        captured_observed = None
        observed_capture_error = TypeError("value must be a SourceVersion")

    if expected_capture_error is not None:
        raise expected_capture_error
    assert captured_expected is not None
    expected = _revalidated_captured_model(SourceVersion, captured_expected)

    if observed_capture_error is not None:
        raise observed_capture_error
    assert captured_observed is not None
    observed = _revalidated_captured_model(SourceVersion, captured_observed)

    return (
        expected.record_id == observed.record_id
        and expected.content_revision == observed.content_revision
        and expected.policy_epoch == observed.policy_epoch
    )


def validate_required_stage_claim(
    *,
    claimed_status: IntentStatus,
    intent: Intent,
    receipts: tuple[StageReceipt, ...],
) -> None:
    """Reject aggregate claims unsupported by fully revalidated inputs.

    Existing Pydantic instances are not proof of validity because unchecked
    construction and copy updates can bypass validators.  This public decision
    boundary therefore materializes every stored field and revalidates each
    complete nested graph before inspecting it.

    Every applied receipt for a required stage must match its declared effect,
    calculated inputs, committed generations, output minimum, and upstream
    support.  Once matching applied evidence exists, that stage remains
    satisfied; greatest-attempt state describes only stages that remain
    unsatisfied.

    A non-null ``started_at`` on any historical required-stage receipt is
    monotonic evidence that work started.  Greatest-attempt state still decides
    whether unsatisfied work remains unfinished.

    This checks consistency of caller-supplied reports.  It does not prove an
    external effect exists, choose authoritative output references, establish
    retry exhaustion, or select a runtime retry policy.
    """

    if not isinstance(claimed_status, IntentStatus):
        raise TypeError("claimed_status must be an IntentStatus")
    if not isinstance(receipts, tuple):
        raise TypeError("receipts must be a tuple")

    if not isinstance(intent, Intent):
        raise TypeError("value must be a Intent")
    try:
        captured_intent = _capture_contract_graph(intent)
        intent_capture_error: Exception | None = None
    except Exception as exc:
        captured_intent = None
        intent_capture_error = exc

    receipt_inventory: dict[int, _CapturedGraph] = {}
    receipt_capture_errors: dict[int, Exception] = {}
    for receipt in tuple.__iter__(receipts):
        if not issubclass(type(receipt), StageReceipt):
            continue
        identity = id(receipt)
        if identity in receipt_inventory or identity in receipt_capture_errors:
            continue
        try:
            _capture_native_contract_graph(
                receipt,
                captured_by_identity=receipt_inventory,
            )
        except Exception as exc:
            receipt_capture_errors[identity] = exc

    if intent_capture_error is not None:
        raise intent_capture_error
    assert captured_intent is not None
    intent = _revalidated_captured_model(Intent, captured_intent)

    def revalidate_receipt(receipt: object) -> StageReceipt:
        capture_error = receipt_capture_errors.get(id(receipt))
        if capture_error is not None and isinstance(receipt, StageReceipt):
            raise capture_error
        return _revalidated_inventory_model(
            StageReceipt,
            receipt,
            receipt_inventory,
        )

    receipts = tuple(
        revalidate_receipt(receipt)
        for receipt in receipts
    )

    requirements = {item.stage: item for item in intent.stage_requirements}
    required = set(requirements)
    latest_attempts: dict[ProcessingStage, int] = {}
    for receipt in receipts:
        if receipt.intent_id != intent.id:
            raise ValueError("stage receipt belongs to a different intent")
        if any(
            output_ref.tenant_id != intent.tenant_id
            for output_ref in receipt.output_refs
        ):
            raise ValueError("stage receipt output_refs must match intent tenant_id")
        requirement = requirements.get(receipt.stage)
        if requirement is not None and receipt.effect_id != requirement.effect_id:
            raise ValueError("required-stage receipt must match its declared effect_id")
        previous_attempt = latest_attempts.get(receipt.stage)
        if previous_attempt is None or receipt.attempt > previous_attempt:
            latest_attempts[receipt.stage] = receipt.attempt

    latest: dict[ProcessingStage, StageReceipt] = {}
    attempt_receipts: dict[tuple[ProcessingStage, int], StageReceipt] = {}
    for receipt in receipts:
        if receipt.stage in required:
            attempt_key = (receipt.stage, receipt.attempt)
            previous_attempt_receipt = attempt_receipts.get(attempt_key)
            if previous_attempt_receipt is None:
                attempt_receipts[attempt_key] = receipt
            elif receipt != previous_attempt_receipt and (
                receipt.status is StageStatus.APPLIED
                or previous_attempt_receipt.status is StageStatus.APPLIED
            ):
                raise ValueError("conflicting receipts for the same stage attempt")

        if receipt.attempt != latest_attempts[receipt.stage]:
            continue
        previous = latest.get(receipt.stage)
        if previous is None:
            latest[receipt.stage] = receipt
        elif receipt != previous:
            raise ValueError("conflicting receipts for the same stage attempt")

    required_receipts = {stage: latest.get(stage) for stage in required}
    satisfied: set[ProcessingStage] = set()
    applied_by_stage: dict[ProcessingStage, list[StageReceipt]] = {
        stage: [] for stage in required
    }
    for receipt in receipts:
        if receipt.stage in required and receipt.status is StageStatus.APPLIED:
            applied_by_stage[receipt.stage].append(receipt)

    for stage in _topological_stages(requirements):
        requirement = requirements[stage]
        stage_applications = applied_by_stage[stage]
        expected_inputs = _expected_stage_inputs(intent, requirement, requirements)
        for receipt in stage_applications:
            if not _source_sets_match(expected_inputs, receipt.applied_sources):
                raise ValueError(
                    "applied required-stage receipt must match calculated inputs"
                )
            if not _source_sets_match(
                requirement.committed_sources, receipt.committed_sources
            ):
                raise ValueError(
                    "applied required-stage receipt must match committed_sources"
                )
            if len(receipt.output_refs) < requirement.minimum_output_refs:
                raise ValueError(
                    "applied required-stage receipt does not meet minimum_output_refs"
                )
            if not set(requirement.upstream_stages) <= satisfied:
                raise ValueError(
                    "applied required-stage receipt requires upstream applied support"
                )
        if stage_applications:
            first = stage_applications[0]
            for receipt in stage_applications[1:]:
                if not (
                    _source_sets_match(first.applied_sources, receipt.applied_sources)
                    and _source_sets_match(
                        first.committed_sources, receipt.committed_sources
                    )
                    and _reference_sets_match(first.output_refs, receipt.output_refs)
                ):
                    raise ValueError(
                        "applied receipts for a stage must report consistent "
                        "result sets"
                    )
            satisfied.add(stage)

    started_stages = {
        receipt.stage
        for receipt in receipts
        if receipt.stage in required and receipt.started_at is not None
    }
    unsatisfied = required - satisfied
    unsatisfied_receipts = {
        stage: required_receipts[stage] for stage in unsatisfied
    }
    partial_failures = {
        stage
        for stage, receipt in unsatisfied_receipts.items()
        if receipt is not None
        and receipt.status
        in {
            StageStatus.SKIPPED,
            StageStatus.FAILED,
        }
    }

    if claimed_status is IntentStatus.ACCEPTED:
        if started_stages or any(
            receipt is not None and receipt.status is not StageStatus.PENDING
            for receipt in unsatisfied_receipts.values()
        ):
            raise ValueError("accepted cannot claim started or terminal required stages")
    elif claimed_status is IntentStatus.PROCESSING:
        terminal_unsatisfied = any(
            receipt is not None
            and receipt.status not in {StageStatus.PENDING, StageStatus.PROCESSING}
            for receipt in unsatisfied_receipts.values()
        )
        if not started_stages or not unsatisfied or terminal_unsatisfied:
            raise ValueError(
                "processing requires started and unfinished required work "
                "with no terminal failure"
            )
    elif claimed_status is IntentStatus.APPLIED:
        if satisfied != required:
            raise ValueError("applied requires matching applied receipts for every required stage")
    elif claimed_status is IntentStatus.PARTIAL:
        mixed_terminal = (
            bool(satisfied)
            and bool(unsatisfied)
            and partial_failures == unsatisfied
        )
        all_skipped = (
            not satisfied
            and bool(unsatisfied)
            and all(
                receipt is not None and receipt.status is StageStatus.SKIPPED
                for receipt in unsatisfied_receipts.values()
            )
        )
        if not (mixed_terminal or all_skipped):
            raise ValueError(
                "partial requires applied with terminal non-applied stages "
                "or all required stages skipped"
            )
    elif claimed_status is IntentStatus.FAILED:
        statuses = {
            stage: receipt.status
            for stage, receipt in unsatisfied_receipts.items()
            if receipt is not None
        }
        if (
            satisfied
            or set(statuses) != required
            or StageStatus.FAILED not in statuses.values()
            or any(
                status not in {StageStatus.FAILED, StageStatus.SKIPPED}
                for status in statuses.values()
            )
        ):
            raise ValueError("failed requires only failed or skipped terminal required stages")
    elif claimed_status is IntentStatus.CANCELLED:
        _validate_fenced_terminal_claim(
            required=required,
            required_receipts=required_receipts,
            satisfied=satisfied,
            fence_status=StageStatus.CANCELLED,
        )
    elif claimed_status is IntentStatus.SUPERSEDED:
        _validate_fenced_terminal_claim(
            required=required,
            required_receipts=required_receipts,
            satisfied=satisfied,
            fence_status=StageStatus.SUPERSEDED,
        )
    else:
        raise ValueError("unsupported IntentStatus")


def _validate_fenced_terminal_claim(
    *,
    required: set[ProcessingStage],
    required_receipts: dict[ProcessingStage, StageReceipt | None],
    satisfied: set[ProcessingStage],
    fence_status: StageStatus,
) -> None:
    fenced = {
        stage
        for stage, receipt in required_receipts.items()
        if stage not in satisfied
        and receipt is not None
        and receipt.status is fence_status
    }
    if not fenced or satisfied | fenced != required:
        raise ValueError(
            f"{fence_status.value} requires every required stage to be applied "
            f"or {fence_status.value}"
        )


def _source_sets_match(
    expected: tuple[SourceVersion, ...],
    observed: tuple[SourceVersion, ...],
) -> bool:
    if len(expected) != len(observed):
        return False
    observed_by_id = {item.record_id: item for item in observed}
    return len(observed_by_id) == len(observed) and all(
        source.record_id in observed_by_id
        and source_versions_match(source, observed_by_id[source.record_id])
        for source in expected
    )


def _reference_sets_match(
    expected: tuple[ReferenceHandle, ...],
    observed: tuple[ReferenceHandle, ...],
) -> bool:
    if len(expected) != len(observed):
        return False
    expected_keys = {_reference_identity_key(item) for item in expected}
    observed_keys = {_reference_identity_key(item) for item in observed}
    return (
        len(expected_keys) == len(expected)
        and len(observed_keys) == len(observed)
        and expected_keys == observed_keys
    )


def _require_unique_reference_handles(
    references: tuple[ReferenceHandle, ...], field: str
) -> None:
    seen: set[tuple[str, str, str, str]] = set()
    for reference in references:
        key = _reference_identity_key(reference)
        if key in seen:
            raise ValueError(f"{field} must not contain duplicate handles")
        seen.add(key)


def _reference_identity_key(
    reference: ReferenceHandle,
) -> tuple[str, str, str, str]:
    return (
        reference.kind,
        reference.id,
        reference.tenant_id,
        reference.resolver,
    )


def _topological_stages(
    requirements: dict[ProcessingStage, StageEvidenceRequirement],
) -> tuple[ProcessingStage, ...]:
    visiting: set[ProcessingStage] = set()
    visited: set[ProcessingStage] = set()
    ordered: list[ProcessingStage] = []

    def visit(stage: ProcessingStage) -> None:
        if stage in visiting:
            raise ValueError("stage_requirements upstream graph must be acyclic")
        if stage in visited:
            return
        visiting.add(stage)
        for upstream in sorted(
            requirements[stage].upstream_stages, key=lambda item: item.value
        ):
            visit(upstream)
        visiting.remove(stage)
        visited.add(stage)
        ordered.append(stage)

    for stage in sorted(requirements, key=lambda item: item.value):
        visit(stage)
    return tuple(ordered)


def _expected_stage_inputs(
    intent: Intent,
    requirement: StageEvidenceRequirement,
    requirements: dict[ProcessingStage, StageEvidenceRequirement],
) -> tuple[SourceVersion, ...]:
    preconditions = {item.record_id: item for item in intent.source_preconditions}
    selected = [preconditions[item] for item in requirement.precondition_source_ids]
    for upstream in requirement.upstream_stages:
        selected.extend(requirements[upstream].committed_sources)
    return tuple(selected)


def _require_unique_source_ids(sources: tuple[SourceVersion, ...], field: str) -> None:
    ids = [source.record_id for source in sources]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{field} must not contain duplicate record_id values")


__all__ = [
    "ApplicabilityScope",
    "Intent",
    "IntentStatus",
    "MemoryLifecycle",
    "MemoryRecord",
    "MemoryRecordKind",
    "OpaqueEnvelopeMemoryBody",
    "PlaintextMemoryBody",
    "ProcessingStage",
    "StageEvidenceRequirement",
    "StageError",
    "StageReceipt",
    "StageStatus",
    "is_legal_intent_transition",
    "source_versions_match",
    "validate_required_stage_claim",
]
