"""Candidate canonical lifecycle types and pure consistency checks.

The models are inert contracts.  They select no storage system, accept no
command durably, authorize no caller, mutate no record, and publish no
projection.  Callers must establish those runtime properties separately.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, TypeVar

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from contracts_bodies import MemoryBody, OpaqueEnvelopeBody, PlaintextBody
from contracts_common import ContractModel, OpaqueId, SafeCounter, VersionedContract
from contracts_references import ReferenceHandle, SourceVersion


_ModelT = TypeVar("_ModelT", bound=BaseModel)


def _native_contract_graph(
    value: object,
    active_containers: set[int] | None = None,
) -> object:
    """Materialize stored fields without dropping extras or hiding cycles."""

    if active_containers is None:
        active_containers = set()

    if isinstance(value, (BaseModel, dict, list, tuple)):
        identity = id(value)
        if identity in active_containers:
            raise ValueError("cyclic contract graph is not valid input")
        active_containers.add(identity)
        try:
            if isinstance(value, BaseModel):
                fields = {
                    key: _native_contract_graph(item, active_containers)
                    for key, item in vars(value).items()
                }
                extra = getattr(value, "__pydantic_extra__", None)
                if extra:
                    if not isinstance(extra, dict):
                        raise ValueError(
                            "malformed stored contract extras are not valid input"
                        )
                    for key, item in extra.items():
                        if key in fields:
                            raise ValueError(
                                "conflicting stored contract field is not valid input"
                            )
                        fields[key] = _native_contract_graph(item, active_containers)
                return fields
            if isinstance(value, dict):
                return {
                    key: _native_contract_graph(item, active_containers)
                    for key, item in value.items()
                }
            if isinstance(value, list):
                return [
                    _native_contract_graph(item, active_containers)
                    for item in value
                ]
            return tuple(
                _native_contract_graph(item, active_containers) for item in value
            )
        finally:
            active_containers.remove(identity)
    return value


def _revalidated_model(model_type: type[_ModelT], value: object) -> _ModelT:
    """Strictly reconstruct one expected model from its complete stored graph."""

    if not isinstance(value, model_type):
        raise TypeError(f"value must be a {model_type.__name__}")
    return model_type.model_validate(
        _native_contract_graph(value),
        strict=True,
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
    expected_sources: tuple[SourceVersion, ...]
    required_stages: tuple[ProcessingStage, ...]
    correlation_id: OpaqueId
    accepted_at: AwareDatetime

    @model_validator(mode="after")
    def validate_required_stages(self) -> Intent:
        if not self.required_stages:
            raise ValueError("required_stages must not be empty")
        if len(set(self.required_stages)) != len(self.required_stages):
            raise ValueError("required_stages must not contain duplicates")
        _require_unique_source_ids(self.expected_sources, "expected_sources")
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
    status: StageStatus
    applied_sources: tuple[SourceVersion, ...]
    output_refs: tuple[ReferenceHandle, ...]
    error: StageError | None
    started_at: AwareDatetime | None
    finished_at: AwareDatetime | None

    @model_validator(mode="after")
    def validate_receipt_semantics(self) -> StageReceipt:
        if self.attempt < 1:
            raise ValueError("attempt must be at least 1")
        _require_unique_source_ids(self.applied_sources, "applied_sources")

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
        if self.status is StageStatus.PENDING and self.started_at is not None:
            raise ValueError("pending stage receipts cannot have started_at")
        if self.status is StageStatus.PROCESSING and self.started_at is None:
            raise ValueError("processing stage receipts require started_at")
        if self.finished_at is not None and self.started_at is None:
            raise ValueError("finished_at requires started_at")
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

    expected = _revalidated_model(SourceVersion, expected)
    observed = _revalidated_model(SourceVersion, observed)

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

    Only the greatest attempt number for each required stage is considered.
    An applied receipt satisfies a stage only when its complete per-source
    revision/epoch set exactly matches the intent's expected sources.
    """

    if not isinstance(claimed_status, IntentStatus):
        raise TypeError("claimed_status must be an IntentStatus")
    if not isinstance(receipts, tuple):
        raise TypeError("receipts must be a tuple")

    intent = _revalidated_model(Intent, intent)
    receipts = tuple(_revalidated_model(StageReceipt, receipt) for receipt in receipts)

    latest_attempts: dict[ProcessingStage, int] = {}
    for receipt in receipts:
        if receipt.intent_id != intent.id:
            raise ValueError("stage receipt belongs to a different intent")
        previous_attempt = latest_attempts.get(receipt.stage)
        if previous_attempt is None or receipt.attempt > previous_attempt:
            latest_attempts[receipt.stage] = receipt.attempt

    latest: dict[ProcessingStage, StageReceipt] = {}
    for receipt in receipts:
        if receipt.attempt != latest_attempts[receipt.stage]:
            continue
        previous = latest.get(receipt.stage)
        if previous is None:
            latest[receipt.stage] = receipt
        elif receipt != previous:
            raise ValueError("conflicting receipts for the same stage attempt")

    required = set(intent.required_stages)
    required_receipts = {stage: latest.get(stage) for stage in required}
    stale_applied = {
        stage
        for stage, receipt in required_receipts.items()
        if receipt is not None
        and receipt.status is StageStatus.APPLIED
        and not _source_sets_match(intent.expected_sources, receipt.applied_sources)
    }
    if stale_applied:
        raise ValueError("applied required-stage receipts must match expected sources")
    applied = {
        stage
        for stage, receipt in required_receipts.items()
        if receipt is not None
        and receipt.status is StageStatus.APPLIED
        and _source_sets_match(intent.expected_sources, receipt.applied_sources)
    }
    partial_failures = {
        stage
        for stage, receipt in required_receipts.items()
        if receipt is not None
        and receipt.status
        in {
            StageStatus.SKIPPED,
            StageStatus.FAILED,
        }
    }

    if claimed_status is IntentStatus.ACCEPTED:
        if any(
            receipt is not None and receipt.status is not StageStatus.PENDING
            for receipt in required_receipts.values()
        ):
            raise ValueError("accepted cannot claim started or terminal required stages")
    elif claimed_status is IntentStatus.PROCESSING:
        if not any(
            receipt is not None
            and receipt.status in {StageStatus.PROCESSING, StageStatus.APPLIED}
            for receipt in required_receipts.values()
        ) or not any(
            receipt is None
            or receipt.status in {StageStatus.PENDING, StageStatus.PROCESSING}
            for receipt in required_receipts.values()
        ) or any(
            receipt is not None
            and receipt.status
            in {
                StageStatus.SKIPPED,
                StageStatus.FAILED,
                StageStatus.CANCELLED,
                StageStatus.SUPERSEDED,
            }
            for receipt in required_receipts.values()
        ):
            raise ValueError(
                "processing requires started and unfinished required work "
                "with no terminal failure"
            )
    elif claimed_status is IntentStatus.APPLIED:
        if applied != required:
            raise ValueError("applied requires matching applied receipts for every required stage")
    elif claimed_status is IntentStatus.PARTIAL:
        if not applied or not partial_failures or applied | partial_failures != required:
            raise ValueError("partial requires applied and terminal non-applied required stages")
    elif claimed_status is IntentStatus.FAILED:
        statuses = {
            stage: receipt.status
            for stage, receipt in required_receipts.items()
            if receipt is not None
        }
        if (
            set(statuses) != required
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
            applied=applied,
            fence_status=StageStatus.CANCELLED,
        )
    elif claimed_status is IntentStatus.SUPERSEDED:
        _validate_fenced_terminal_claim(
            required=required,
            required_receipts=required_receipts,
            applied=applied,
            fence_status=StageStatus.SUPERSEDED,
        )
    else:
        raise ValueError("unsupported IntentStatus")


def _validate_fenced_terminal_claim(
    *,
    required: set[ProcessingStage],
    required_receipts: dict[ProcessingStage, StageReceipt | None],
    applied: set[ProcessingStage],
    fence_status: StageStatus,
) -> None:
    fenced = {
        stage
        for stage, receipt in required_receipts.items()
        if receipt is not None and receipt.status is fence_status
    }
    if not fenced or applied | fenced != required:
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
    "StageError",
    "StageReceipt",
    "StageStatus",
    "is_legal_intent_transition",
    "source_versions_match",
    "validate_required_stage_claim",
]
