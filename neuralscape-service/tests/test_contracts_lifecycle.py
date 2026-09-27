"""Positive and adversarial tests for inert lifecycle contracts."""

from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from contracts_lifecycle import (
    ApplicabilityScope,
    Intent,
    IntentStatus,
    MemoryLifecycle,
    MemoryRecord,
    MemoryRecordKind,
    PlaintextMemoryBody,
    ProcessingStage,
    StageError,
    StageReceipt,
    StageStatus,
    is_legal_intent_transition,
    source_versions_match,
    validate_required_stage_claim,
)
from contracts_references import SourceVersion


NOW = datetime(2026, 9, 28, 12, 0, tzinfo=timezone.utc)


def source(record_id: str, revision: int = 3, epoch: int = 7) -> SourceVersion:
    return SourceVersion(
        record_id=record_id,
        content_revision=revision,
        policy_epoch=epoch,
    )


def intent(*stages: ProcessingStage) -> Intent:
    return Intent(
        schema_version="candidate-v1",
        id="intent-1",
        tenant_id="tenant-1",
        actor_id="actor-1",
        credential_id="credential-1",
        operation="write",
        target_refs=(),
        request_digest="sha256:abc",
        idempotency_key="client-key-1",
        expected_sources=(source("memory-1"),),
        required_stages=stages,
        correlation_id="correlation-1",
        accepted_at=NOW,
    )


def receipt(
    stage: ProcessingStage,
    status: StageStatus,
    *,
    sources: tuple[SourceVersion, ...] | None = None,
    attempt: int = 1,
) -> StageReceipt:
    error = (
        StageError(code="projection_failed", retryable=False, safe_message="Projection failed")
        if status is StageStatus.FAILED
        else None
    )
    terminal = status not in {StageStatus.PENDING, StageStatus.PROCESSING}
    return StageReceipt(
        schema_version="candidate-v1",
        intent_id="intent-1",
        attempt=attempt,
        stage=stage,
        status=status,
        applied_sources=sources if sources is not None else (source("memory-1"),),
        output_refs=(),
        error=error,
        started_at=NOW,
        finished_at=NOW + timedelta(seconds=1) if terminal else None,
    )


def test_memory_record_keeps_content_revision_and_policy_epoch_independent() -> None:
    record = MemoryRecord(
        schema_version="candidate-v1",
        id="memory-1",
        tenant_id="tenant-1",
        owner_id="actor-1",
        record_kind=MemoryRecordKind.FACT,
        applicability=ApplicabilityScope.PROJECT,
        project_id="project-1",
        workspace_id=None,
        category="preference",
        author_kind="user",
        source_kind="direct",
        content_revision=3,
        policy_epoch=19,
        lifecycle=MemoryLifecycle.ACTIVE,
        body=PlaintextMemoryBody(kind="plaintext", text="Use concise answers"),
        source_refs=(),
        required_parents=(source("parent-a", 8, 2), source("parent-b", 2, 11)),
        successor_id=None,
        retention_hold_ids=(),
        occurred_at=None,
        recorded_at=NOW,
    )

    assert record.content_revision == 3
    assert record.policy_epoch == 19
    assert {item.record_id: item.content_revision for item in record.required_parents} == {
        "parent-a": 8,
        "parent-b": 2,
    }


def test_record_rejects_scope_mismatch_and_body_after_erasure() -> None:
    common = dict(
        schema_version="candidate-v1",
        id="memory-1",
        tenant_id="tenant-1",
        owner_id="actor-1",
        record_kind=MemoryRecordKind.FACT,
        category="preference",
        author_kind="user",
        source_kind="direct",
        content_revision=3,
        policy_epoch=7,
        source_refs=(),
        required_parents=(),
        successor_id=None,
        retention_hold_ids=(),
        occurred_at=None,
        recorded_at=NOW,
    )
    with pytest.raises(ValidationError, match="project applicability requires project_id"):
        MemoryRecord(
            **common,
            applicability=ApplicabilityScope.PROJECT,
            project_id=None,
            workspace_id=None,
            lifecycle=MemoryLifecycle.ACTIVE,
            body=PlaintextMemoryBody(kind="plaintext", text="body"),
        )
    with pytest.raises(ValidationError, match="erased records cannot retain a body"):
        MemoryRecord(
            **common,
            applicability=ApplicabilityScope.GLOBAL,
            project_id=None,
            workspace_id=None,
            lifecycle=MemoryLifecycle.ERASED,
            body=PlaintextMemoryBody(kind="plaintext", text="must be absent"),
        )


def test_intent_rejects_empty_or_duplicate_required_stages() -> None:
    with pytest.raises(ValidationError, match="required_stages must not be empty"):
        intent()
    with pytest.raises(ValidationError, match="must not contain duplicates"):
        intent(ProcessingStage.CANONICAL, ProcessingStage.CANONICAL)


def test_intent_is_an_immutable_command_snapshot() -> None:
    command = intent(ProcessingStage.CANONICAL)
    with pytest.raises(ValidationError, match="frozen"):
        command.operation = "delete"


def test_legal_transitions_are_forward_only_and_terminal_states_stay_terminal() -> None:
    assert is_legal_intent_transition(IntentStatus.ACCEPTED, IntentStatus.PROCESSING)
    assert is_legal_intent_transition(IntentStatus.PROCESSING, IntentStatus.PARTIAL)
    assert is_legal_intent_transition(IntentStatus.PROCESSING, IntentStatus.CANCELLED)
    assert is_legal_intent_transition(IntentStatus.PROCESSING, IntentStatus.SUPERSEDED)
    assert not is_legal_intent_transition(IntentStatus.ACCEPTED, IntentStatus.APPLIED)
    assert not is_legal_intent_transition(IntentStatus.CANCELLED, IntentStatus.PROCESSING)
    assert not is_legal_intent_transition(IntentStatus.SUPERSEDED, IntentStatus.APPLIED)


def test_stale_content_revision_or_policy_epoch_never_matches() -> None:
    expected = source("memory-1", revision=4, epoch=9)

    assert source_versions_match(expected, source("memory-1", revision=4, epoch=9))
    assert not source_versions_match(expected, source("memory-1", revision=3, epoch=9))
    assert not source_versions_match(expected, source("memory-1", revision=4, epoch=8))
    assert not source_versions_match(expected, source("another-memory", revision=4, epoch=9))


def test_applied_claim_requires_every_required_stage_at_expected_versions() -> None:
    command = intent(ProcessingStage.CANONICAL, ProcessingStage.GRAPH)
    validate_required_stage_claim(
        claimed_status=IntentStatus.APPLIED,
        intent=command,
        receipts=(
            receipt(ProcessingStage.CANONICAL, StageStatus.APPLIED),
            receipt(ProcessingStage.GRAPH, StageStatus.APPLIED),
        ),
    )

    stale_graph = receipt(
        ProcessingStage.GRAPH,
        StageStatus.APPLIED,
        sources=(source("memory-1", revision=2, epoch=7),),
    )
    with pytest.raises(ValueError, match="every required stage"):
        validate_required_stage_claim(
            claimed_status=IntentStatus.APPLIED,
            intent=command,
            receipts=(receipt(ProcessingStage.CANONICAL, StageStatus.APPLIED), stale_graph),
        )


def test_partial_requires_both_applied_and_terminal_non_applied_stages() -> None:
    command = intent(ProcessingStage.CANONICAL, ProcessingStage.GRAPH)
    validate_required_stage_claim(
        claimed_status=IntentStatus.PARTIAL,
        intent=command,
        receipts=(
            receipt(ProcessingStage.CANONICAL, StageStatus.APPLIED),
            receipt(ProcessingStage.GRAPH, StageStatus.SKIPPED),
        ),
    )

    with pytest.raises(ValueError, match="partial requires"):
        validate_required_stage_claim(
            claimed_status=IntentStatus.PARTIAL,
            intent=command,
            receipts=(
                receipt(ProcessingStage.CANONICAL, StageStatus.APPLIED),
                receipt(ProcessingStage.GRAPH, StageStatus.PROCESSING),
            ),
        )


def test_failed_claim_rejects_applied_or_in_flight_required_stages() -> None:
    command = intent(ProcessingStage.CANONICAL, ProcessingStage.GRAPH)
    validate_required_stage_claim(
        claimed_status=IntentStatus.FAILED,
        intent=command,
        receipts=(
            receipt(ProcessingStage.CANONICAL, StageStatus.SKIPPED),
            receipt(ProcessingStage.GRAPH, StageStatus.FAILED),
        ),
    )
    with pytest.raises(ValueError, match="failed requires"):
        validate_required_stage_claim(
            claimed_status=IntentStatus.FAILED,
            intent=command,
            receipts=(
                receipt(ProcessingStage.CANONICAL, StageStatus.APPLIED),
                receipt(ProcessingStage.GRAPH, StageStatus.FAILED),
            ),
        )


@pytest.mark.parametrize(
    ("claim", "stage_status"),
    [
        (IntentStatus.CANCELLED, StageStatus.CANCELLED),
        (IntentStatus.SUPERSEDED, StageStatus.SUPERSEDED),
    ],
)
def test_cancelled_and_superseded_do_not_masquerade_as_applied(
    claim: IntentStatus,
    stage_status: StageStatus,
) -> None:
    command = intent(ProcessingStage.CANONICAL, ProcessingStage.VECTOR)
    receipts = (
        receipt(ProcessingStage.CANONICAL, stage_status),
        receipt(ProcessingStage.VECTOR, stage_status),
    )
    validate_required_stage_claim(claimed_status=claim, intent=command, receipts=receipts)
    with pytest.raises(ValueError, match="every required stage"):
        validate_required_stage_claim(
            claimed_status=IntentStatus.APPLIED,
            intent=command,
            receipts=receipts,
        )


def test_cancelled_claim_can_describe_effects_committed_before_fence() -> None:
    command = intent(ProcessingStage.CANONICAL, ProcessingStage.GRAPH)
    validate_required_stage_claim(
        claimed_status=IntentStatus.CANCELLED,
        intent=command,
        receipts=(
            receipt(ProcessingStage.CANONICAL, StageStatus.APPLIED),
            receipt(ProcessingStage.GRAPH, StageStatus.CANCELLED),
        ),
    )


def test_latest_attempt_controls_stage_claim_and_conflicts_are_rejected() -> None:
    command = intent(ProcessingStage.GRAPH)
    validate_required_stage_claim(
        claimed_status=IntentStatus.APPLIED,
        intent=command,
        receipts=(
            receipt(ProcessingStage.GRAPH, StageStatus.FAILED, attempt=1),
            receipt(ProcessingStage.GRAPH, StageStatus.APPLIED, attempt=2),
        ),
    )

    with pytest.raises(ValueError, match="conflicting receipts"):
        validate_required_stage_claim(
            claimed_status=IntentStatus.APPLIED,
            intent=command,
            receipts=(
                receipt(ProcessingStage.GRAPH, StageStatus.APPLIED, attempt=2),
                receipt(
                    ProcessingStage.GRAPH,
                    StageStatus.APPLIED,
                    sources=(source("memory-1", revision=2, epoch=7),),
                    attempt=2,
                ),
            ),
        )


def test_failed_receipt_requires_error_and_terminal_timing() -> None:
    with pytest.raises(ValidationError, match="failed stage receipts require an error"):
        StageReceipt(
            schema_version="candidate-v1",
            intent_id="intent-1",
            attempt=1,
            stage=ProcessingStage.GRAPH,
            status=StageStatus.FAILED,
            applied_sources=(),
            output_refs=(),
            error=None,
            started_at=NOW,
            finished_at=NOW + timedelta(seconds=1),
        )
    with pytest.raises(ValidationError, match="terminal stage receipts require finished_at"):
        StageReceipt(
            schema_version="candidate-v1",
            intent_id="intent-1",
            attempt=1,
            stage=ProcessingStage.GRAPH,
            status=StageStatus.APPLIED,
            applied_sources=(source("memory-1"),),
            output_refs=(),
            error=None,
            started_at=NOW,
            finished_at=None,
        )
    with pytest.raises(ValidationError, match="processing stage receipts require started_at"):
        StageReceipt(
            schema_version="candidate-v1",
            intent_id="intent-1",
            attempt=1,
            stage=ProcessingStage.GRAPH,
            status=StageStatus.PROCESSING,
            applied_sources=(),
            output_refs=(),
            error=None,
            started_at=None,
            finished_at=None,
        )
