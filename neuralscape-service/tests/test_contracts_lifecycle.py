"""Positive and adversarial tests for inert lifecycle contracts."""

import json
from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from contracts_bodies import MEMORY_BODY_ADAPTER, OpaqueEnvelopeBody, PlaintextBody
from contracts_lifecycle import (
    ApplicabilityScope,
    Intent,
    IntentStatus,
    MemoryLifecycle,
    MemoryRecord,
    MemoryRecordKind,
    OpaqueEnvelopeMemoryBody,
    PlaintextMemoryBody,
    ProcessingStage,
    StageError,
    StageReceipt,
    StageStatus,
    is_legal_intent_transition,
    source_versions_match,
    validate_required_stage_claim,
)
from contracts_references import ReferenceHandle, SourceVersion


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
        started_at=None if status is StageStatus.PENDING else NOW,
        finished_at=NOW + timedelta(seconds=1) if terminal else None,
    )


def memory_record(**overrides: object) -> MemoryRecord:
    values: dict[str, object] = {
        "schema_version": "candidate-v1",
        "id": "memory-1",
        "tenant_id": "tenant-1",
        "owner_id": "actor-1",
        "record_kind": MemoryRecordKind.FACT,
        "applicability": ApplicabilityScope.GLOBAL,
        "project_id": None,
        "workspace_id": None,
        "category": "preference",
        "author_kind": "user",
        "source_kind": "direct",
        "content_revision": 3,
        "policy_epoch": 7,
        "lifecycle": MemoryLifecycle.ACTIVE,
        "body": PlaintextMemoryBody(kind="plaintext", text="Use concise answers"),
        "source_refs": (),
        "required_parents": (),
        "successor_id": None,
        "retention_hold_ids": (),
        "occurred_at": None,
        "recorded_at": NOW,
    }
    values.update(overrides)
    return MemoryRecord(**values)


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


def test_memory_record_accepts_canonical_body_variants_in_python() -> None:
    plaintext = memory_record(
        body=PlaintextMemoryBody(kind="plaintext", text="exact text")
    )
    opaque = memory_record(
        body=OpaqueEnvelopeMemoryBody(
            kind="opaque_envelope",
            envelope_id="envelope-1",
        )
    )

    assert isinstance(plaintext.body, PlaintextBody)
    assert plaintext.body.text == "exact text"
    assert isinstance(opaque.body, OpaqueEnvelopeBody)
    assert opaque.body.envelope_id == "envelope-1"


@pytest.mark.parametrize(
    ("body", "body_type"),
    [
        ({"kind": "plaintext", "text": "wire text"}, PlaintextBody),
        (
            {"kind": "opaque_envelope", "envelope_id": "envelope-1"},
            OpaqueEnvelopeBody,
        ),
    ],
)
def test_memory_record_accepts_canonical_body_variants_in_json(
    body: dict[str, object],
    body_type: type[PlaintextBody] | type[OpaqueEnvelopeBody],
) -> None:
    payload = memory_record().model_dump(mode="json")
    payload["body"] = body

    record = MemoryRecord.model_validate_json(json.dumps(payload))

    assert isinstance(record.body, body_type)


@pytest.mark.parametrize(
    "body",
    [
        {"kind": "plaintext", "text": "readable", "envelope_id": "env-1"},
        {"kind": "opaque_envelope", "envelope_id": "env-1", "text": "leak"},
        {"kind": "encrypted", "envelope_id": "env-1"},
    ],
)
def test_memory_record_rejects_mixed_or_legacy_body_shapes(
    body: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        memory_record(body=body)

    payload = memory_record().model_dump(mode="json")
    payload["body"] = body
    with pytest.raises(ValidationError):
        MemoryRecord.model_validate_json(json.dumps(payload))


def test_memory_record_revalidates_unchecked_body_copy() -> None:
    valid = PlaintextBody(kind="plaintext", text="body")
    unchecked = valid.model_copy(update={"text": ""})

    with pytest.raises(ValidationError, match="at least 1 character"):
        memory_record(body=unchecked)


def test_lifecycle_body_schema_is_the_canonical_producer_schema() -> None:
    assert PlaintextMemoryBody is PlaintextBody
    assert OpaqueEnvelopeMemoryBody is OpaqueEnvelopeBody

    producer_schema = MEMORY_BODY_ADAPTER.json_schema()
    record_schema = MemoryRecord.model_json_schema()
    record_body_union = record_schema["properties"]["body"]["anyOf"][0]
    assert record_body_union == {
        "discriminator": producer_schema["discriminator"],
        "oneOf": producer_schema["oneOf"],
    }
    for definition in ("PlaintextBody", "OpaqueEnvelopeBody"):
        assert record_schema["$defs"][definition] == producer_schema["$defs"][
            definition
        ]
    assert "PlaintextMemoryBody" not in record_schema["$defs"]
    assert "EncryptedMemoryBody" not in record_schema["$defs"]


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


def test_record_project_scope_rejects_workspace_id_at_model_boundary() -> None:
    payload = memory_record(
        applicability=ApplicabilityScope.PROJECT,
        project_id="project-1",
    ).model_dump()
    payload["workspace_id"] = "workspace-1"

    with pytest.raises(
        ValidationError,
        match="project applicability cannot carry workspace_id",
    ):
        MemoryRecord.model_validate(payload)


@pytest.mark.parametrize(
    ("applicability", "project_id", "workspace_id"),
    [
        (ApplicabilityScope.GLOBAL, None, None),
        (ApplicabilityScope.PROJECT, "project-1", None),
        (ApplicabilityScope.WORKSPACE, None, "workspace-1"),
        (ApplicabilityScope.WORKSPACE, "project-1", "workspace-1"),
    ],
)
def test_record_accepts_valid_scope_identifiers(
    applicability: ApplicabilityScope,
    project_id: str | None,
    workspace_id: str | None,
) -> None:
    record = memory_record(
        applicability=applicability,
        project_id=project_id,
        workspace_id=workspace_id,
    )

    assert record.applicability is applicability
    assert record.project_id == project_id
    assert record.workspace_id == workspace_id


@pytest.mark.parametrize(
    "parents",
    [
        (source("parent-a", 3, 7), source("parent-a", 3, 7)),
        (source("parent-a", 3, 7), source("parent-a", 4, 8)),
    ],
)
def test_record_rejects_duplicate_required_parent_witnesses(
    parents: tuple[SourceVersion, ...],
) -> None:
    with pytest.raises(ValidationError, match="required_parents.*duplicate"):
        memory_record(required_parents=parents)


@pytest.mark.parametrize(
    ("field_name", "replacement"),
    [
        ("lifecycle", MemoryLifecycle.ERASED),
        ("applicability", ApplicabilityScope.PROJECT),
        ("project_id", "project-1"),
        ("workspace_id", "workspace-1"),
    ],
)
def test_memory_record_is_an_immutable_canonical_snapshot(
    field_name: str,
    replacement: object,
) -> None:
    record = memory_record()

    with pytest.raises(ValidationError, match="frozen"):
        setattr(record, field_name, replacement)

    assert record.lifecycle is MemoryLifecycle.ACTIVE
    assert record.applicability is ApplicabilityScope.GLOBAL
    assert record.project_id is None
    assert record.workspace_id is None


def test_intent_rejects_empty_or_duplicate_required_stages() -> None:
    with pytest.raises(ValidationError, match="required_stages must not be empty"):
        intent()
    with pytest.raises(ValidationError, match="must not contain duplicates"):
        intent(ProcessingStage.CANONICAL, ProcessingStage.CANONICAL)


def test_intent_is_an_immutable_command_snapshot() -> None:
    command = intent(ProcessingStage.CANONICAL)
    with pytest.raises(ValidationError, match="frozen"):
        command.operation = "delete"


def test_locally_owned_nested_stage_error_is_frozen() -> None:
    stage_error = StageError(
        code="projection_failed",
        retryable=False,
        safe_message="Projection failed",
    )
    failed = StageReceipt(
        schema_version="candidate-v1",
        intent_id="intent-1",
        attempt=1,
        stage=ProcessingStage.GRAPH,
        status=StageStatus.FAILED,
        applied_sources=(),
        output_refs=(),
        error=stage_error,
        started_at=NOW,
        finished_at=NOW + timedelta(seconds=1),
    )

    with pytest.raises(ValidationError, match="frozen"):
        failed.error.retryable = True


def test_shared_nested_reference_and_version_values_are_frozen() -> None:
    reference = ReferenceHandle(
        kind="memory",
        id="memory-1",
        tenant_id="tenant-1",
        resolver="resolve_memory",
    )
    command = Intent(
        schema_version="candidate-v1",
        id="intent-1",
        tenant_id="tenant-1",
        actor_id="actor-1",
        credential_id="credential-1",
        operation="write",
        target_refs=(reference,),
        request_digest="sha256:abc",
        idempotency_key="client-key-1",
        expected_sources=(source("memory-1"),),
        required_stages=(ProcessingStage.CANONICAL,),
        correlation_id="correlation-1",
        accepted_at=NOW,
    )
    applied = StageReceipt(
        schema_version="candidate-v1",
        intent_id="intent-1",
        attempt=1,
        stage=ProcessingStage.CANONICAL,
        status=StageStatus.APPLIED,
        applied_sources=(source("memory-1"),),
        output_refs=(reference,),
        error=None,
        started_at=NOW,
        finished_at=NOW + timedelta(seconds=1),
    )

    nested_mutations = [
        (command.expected_sources[0], "content_revision", 99),
        (command.target_refs[0], "id", "other-memory"),
        (applied.applied_sources[0], "policy_epoch", 99),
        (applied.output_refs[0], "resolver", "other_resolver"),
    ]
    for value, field_name, replacement in nested_mutations:
        with pytest.raises(ValidationError, match="frozen"):
            setattr(value, field_name, replacement)


@pytest.mark.parametrize(
    "required_stages",
    [
        (),
        (ProcessingStage.CANONICAL, ProcessingStage.CANONICAL),
    ],
)
def test_aggregate_boundary_revalidates_copied_intent_stages(
    required_stages: tuple[ProcessingStage, ...],
) -> None:
    command = intent(ProcessingStage.CANONICAL)
    unchecked = command.model_copy(update={"required_stages": required_stages})

    with pytest.raises(ValidationError, match="required_stages"):
        validate_required_stage_claim(
            claimed_status=IntentStatus.APPLIED,
            intent=unchecked,
            receipts=(receipt(ProcessingStage.CANONICAL, StageStatus.APPLIED),),
        )


def test_aggregate_boundary_revalidates_constructed_intent() -> None:
    command = intent(ProcessingStage.CANONICAL)
    values = dict(command.__dict__)
    values["required_stages"] = ()
    unchecked = Intent.model_construct(**values)

    with pytest.raises(ValidationError, match="required_stages"):
        validate_required_stage_claim(
            claimed_status=IntentStatus.APPLIED,
            intent=unchecked,
            receipts=(),
        )


def test_aggregate_boundary_rejects_copy_injected_extra_fields() -> None:
    command = intent(ProcessingStage.CANONICAL)
    valid_receipt = receipt(ProcessingStage.CANONICAL, StageStatus.APPLIED)

    extra_intent = command.model_copy(update={"undeclared_mode": True})
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        validate_required_stage_claim(
            claimed_status=IntentStatus.APPLIED,
            intent=extra_intent,
            receipts=(valid_receipt,),
        )

    extra_reference = ReferenceHandle(
        kind="memory",
        id="memory-1",
        tenant_id="tenant-1",
        resolver="resolve_memory",
    ).model_copy(update={"undeclared_scope": "other"})
    extra_nested_receipt = valid_receipt.model_copy(
        update={"output_refs": (extra_reference,)}
    )
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        validate_required_stage_claim(
            claimed_status=IntentStatus.APPLIED,
            intent=command,
            receipts=(extra_nested_receipt,),
        )


def test_aggregate_boundary_preserves_sequence_types_during_revalidation() -> None:
    command = intent(ProcessingStage.CANONICAL)
    unchecked = command.model_copy(
        update={"required_stages": [ProcessingStage.CANONICAL]}
    )

    with pytest.raises(ValidationError, match="valid tuple"):
        validate_required_stage_claim(
            claimed_status=IntentStatus.APPLIED,
            intent=unchecked,
            receipts=(receipt(ProcessingStage.CANONICAL, StageStatus.APPLIED),),
        )


def test_aggregate_boundary_rejects_cyclic_graph_predictably() -> None:
    command = intent(ProcessingStage.CANONICAL).model_copy(
        update={"undeclared_cycle": None}
    )
    command.__dict__["undeclared_cycle"] = command

    with pytest.raises(ValueError, match="cyclic contract graph"):
        validate_required_stage_claim(
            claimed_status=IntentStatus.APPLIED,
            intent=command,
            receipts=(),
        )


def test_aggregate_boundary_rejects_non_tuple_receipt_collection() -> None:
    with pytest.raises(TypeError, match="receipts must be a tuple"):
        validate_required_stage_claim(
            claimed_status=IntentStatus.APPLIED,
            intent=intent(ProcessingStage.CANONICAL),
            receipts=[  # type: ignore[arg-type]
                receipt(ProcessingStage.CANONICAL, StageStatus.APPLIED)
            ],
        )


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"attempt": 0}, "attempt"),
        ({"finished_at": None}, "finished_at"),
        ({"started_at": datetime(2026, 9, 28, 12, 0)}, "timezone"),
        (
            {
                "started_at": NOW + timedelta(seconds=2),
                "finished_at": NOW + timedelta(seconds=1),
            },
            "cannot precede",
        ),
    ],
)
def test_aggregate_boundary_revalidates_copied_receipt_invariants(
    updates: dict[str, object],
    message: str,
) -> None:
    command = intent(ProcessingStage.CANONICAL)
    valid_receipt = receipt(ProcessingStage.CANONICAL, StageStatus.APPLIED)
    unchecked = valid_receipt.model_copy(update=updates)

    with pytest.raises(ValidationError, match=message):
        validate_required_stage_claim(
            claimed_status=IntentStatus.APPLIED,
            intent=command,
            receipts=(unchecked,),
        )


def test_aggregate_boundary_revalidates_copied_nested_values() -> None:
    command = intent(ProcessingStage.CANONICAL)
    valid_receipt = receipt(ProcessingStage.CANONICAL, StageStatus.APPLIED)

    invalid_source = command.expected_sources[0].model_copy(
        update={"content_revision": -1}
    )
    invalid_source_command = command.model_copy(
        update={"expected_sources": (invalid_source,)}
    )
    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        Intent.model_validate(invalid_source_command)
    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        validate_required_stage_claim(
            claimed_status=IntentStatus.APPLIED,
            intent=invalid_source_command,
            receipts=(valid_receipt,),
        )

    reference = ReferenceHandle(
        kind="memory",
        id="memory-1",
        tenant_id="tenant-1",
        resolver="resolve_memory",
    )
    invalid_reference = reference.model_copy(update={"id": ""})
    invalid_reference_receipt = valid_receipt.model_copy(
        update={"output_refs": (invalid_reference,)}
    )
    with pytest.raises(ValidationError, match="at least 1 character"):
        StageReceipt.model_validate(invalid_reference_receipt)
    with pytest.raises(ValidationError, match="at least 1 character"):
        validate_required_stage_claim(
            claimed_status=IntentStatus.APPLIED,
            intent=command,
            receipts=(invalid_reference_receipt,),
        )


@pytest.mark.parametrize("claimed_status", ["applied", "unknown", object()])
def test_aggregate_boundary_rejects_non_native_claimed_status(
    claimed_status: object,
) -> None:
    with pytest.raises(TypeError, match="claimed_status must be an IntentStatus"):
        validate_required_stage_claim(
            claimed_status=claimed_status,  # type: ignore[arg-type]
            intent=intent(ProcessingStage.CANONICAL),
            receipts=(),
        )


def test_aggregate_boundary_dispatches_every_native_status() -> None:
    single_stage = intent(ProcessingStage.CANONICAL)
    cases = {
        IntentStatus.ACCEPTED: (
            receipt(ProcessingStage.CANONICAL, StageStatus.PENDING),
        ),
        IntentStatus.PROCESSING: (
            receipt(ProcessingStage.CANONICAL, StageStatus.PROCESSING),
        ),
        IntentStatus.APPLIED: (
            receipt(ProcessingStage.CANONICAL, StageStatus.APPLIED),
        ),
        IntentStatus.FAILED: (
            receipt(ProcessingStage.CANONICAL, StageStatus.FAILED),
        ),
        IntentStatus.CANCELLED: (
            receipt(ProcessingStage.CANONICAL, StageStatus.CANCELLED),
        ),
        IntentStatus.SUPERSEDED: (
            receipt(ProcessingStage.CANONICAL, StageStatus.SUPERSEDED),
        ),
    }
    for claimed_status, receipts in cases.items():
        validate_required_stage_claim(
            claimed_status=claimed_status,
            intent=single_stage,
            receipts=receipts,
        )

    two_stage = intent(ProcessingStage.CANONICAL, ProcessingStage.GRAPH)
    validate_required_stage_claim(
        claimed_status=IntentStatus.PARTIAL,
        intent=two_stage,
        receipts=(
            receipt(ProcessingStage.CANONICAL, StageStatus.APPLIED),
            receipt(ProcessingStage.GRAPH, StageStatus.SKIPPED),
        ),
    )


def test_legal_transitions_are_forward_only_and_terminal_states_stay_terminal() -> None:
    assert is_legal_intent_transition(IntentStatus.ACCEPTED, IntentStatus.PROCESSING)
    assert is_legal_intent_transition(IntentStatus.PROCESSING, IntentStatus.PARTIAL)
    assert is_legal_intent_transition(IntentStatus.PROCESSING, IntentStatus.CANCELLED)
    assert is_legal_intent_transition(IntentStatus.PROCESSING, IntentStatus.SUPERSEDED)
    assert not is_legal_intent_transition(IntentStatus.ACCEPTED, IntentStatus.APPLIED)
    assert not is_legal_intent_transition(IntentStatus.CANCELLED, IntentStatus.PROCESSING)
    assert not is_legal_intent_transition(IntentStatus.SUPERSEDED, IntentStatus.APPLIED)


def test_transition_helper_accepts_canonical_native_and_json_statuses() -> None:
    assert is_legal_intent_transition(
        IntentStatus.ACCEPTED,
        IntentStatus.PROCESSING,
    )

    current = IntentStatus(json.loads('"processing"'))
    target = IntentStatus(json.loads('"applied"'))
    assert current is IntentStatus.PROCESSING
    assert target is IntentStatus.APPLIED
    assert is_legal_intent_transition(current, target)


@pytest.mark.parametrize(
    ("current", "target"),
    [
        ("accepted", IntentStatus.PROCESSING),
        (IntentStatus.ACCEPTED, "processing"),
        (object(), IntentStatus.PROCESSING),
    ],
)
def test_transition_helper_rejects_non_native_statuses(
    current: object,
    target: object,
) -> None:
    with pytest.raises(TypeError, match="must be an IntentStatus"):
        is_legal_intent_transition(  # type: ignore[arg-type]
            current,
            target,
        )


@pytest.mark.parametrize(
    ("position", "raw"),
    [
        ("current", "accepted"),
        ("current", "unknown"),
        ("target", "processing"),
        ("target", "unknown"),
    ],
)
def test_transition_helper_rejects_forged_exact_type_statuses(
    position: str,
    raw: str,
) -> None:
    forged = str.__new__(IntentStatus, raw)
    forged._name_ = "FORGED"
    forged._value_ = raw

    assert type(forged) is IntentStatus
    assert not any(forged is member for member in IntentStatus)

    current = forged if position == "current" else IntentStatus.ACCEPTED
    target = forged if position == "target" else IntentStatus.PROCESSING
    with pytest.raises(TypeError, match=f"{position} must be an IntentStatus"):
        is_legal_intent_transition(current, target)


def test_stale_content_revision_or_policy_epoch_never_matches() -> None:
    expected = source("memory-1", revision=4, epoch=9)

    assert source_versions_match(expected, source("memory-1", revision=4, epoch=9))
    assert not source_versions_match(expected, source("memory-1", revision=3, epoch=9))
    assert not source_versions_match(expected, source("memory-1", revision=4, epoch=8))
    assert not source_versions_match(expected, source("another-memory", revision=4, epoch=9))


def test_source_version_decision_boundary_revalidates_copied_values() -> None:
    valid = source("memory-1", revision=4, epoch=9)
    invalid = valid.model_copy(update={"content_revision": -1})

    revalidated = SourceVersion.model_validate(valid)
    assert revalidated == valid
    assert revalidated is not valid
    assert source_versions_match(valid, revalidated)

    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        SourceVersion.model_validate(invalid)
    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        source_versions_match(invalid, invalid)


def test_source_version_boundary_rejects_copy_injected_extra_fields() -> None:
    valid = source("memory-1", revision=4, epoch=9)
    unchecked = valid.model_copy(update={"undeclared_witness": True})

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        source_versions_match(unchecked, valid)

    non_string_key = valid.model_copy(update={1: "undeclared"})  # type: ignore[dict-item]
    with pytest.raises(ValidationError, match="Keys should be strings"):
        source_versions_match(non_string_key, valid)


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
    with pytest.raises(ValueError, match="must match expected sources"):
        validate_required_stage_claim(
            claimed_status=IntentStatus.APPLIED,
            intent=command,
            receipts=(receipt(ProcessingStage.CANONICAL, StageStatus.APPLIED), stale_graph),
        )


@pytest.mark.parametrize(
    "stale_source",
    [
        source("memory-1", revision=2, epoch=7),
        source("memory-1", revision=3, epoch=6),
    ],
)
def test_processing_rejects_latest_applied_receipt_with_stale_witness(
    stale_source: SourceVersion,
) -> None:
    command = intent(ProcessingStage.CANONICAL, ProcessingStage.GRAPH)
    with pytest.raises(ValueError, match="must match expected sources"):
        validate_required_stage_claim(
            claimed_status=IntentStatus.PROCESSING,
            intent=command,
            receipts=(
                receipt(
                    ProcessingStage.CANONICAL,
                    StageStatus.APPLIED,
                    sources=(stale_source,),
                ),
                receipt(ProcessingStage.GRAPH, StageStatus.PROCESSING),
            ),
        )


def test_processing_requires_started_and_unfinished_work() -> None:
    command = intent(ProcessingStage.CANONICAL, ProcessingStage.GRAPH)

    validate_required_stage_claim(
        claimed_status=IntentStatus.PROCESSING,
        intent=command,
        receipts=(
            receipt(ProcessingStage.CANONICAL, StageStatus.APPLIED),
            receipt(ProcessingStage.GRAPH, StageStatus.PENDING),
        ),
    )
    validate_required_stage_claim(
        claimed_status=IntentStatus.PROCESSING,
        intent=command,
        receipts=(receipt(ProcessingStage.CANONICAL, StageStatus.APPLIED),),
    )
    validate_required_stage_claim(
        claimed_status=IntentStatus.PROCESSING,
        intent=command,
        receipts=(receipt(ProcessingStage.CANONICAL, StageStatus.PROCESSING),),
    )


@pytest.mark.parametrize(
    "receipts",
    [
        (),
        (
            receipt(ProcessingStage.CANONICAL, StageStatus.PENDING),
            receipt(ProcessingStage.GRAPH, StageStatus.PENDING),
        ),
        (
            receipt(ProcessingStage.CANONICAL, StageStatus.APPLIED),
            receipt(ProcessingStage.GRAPH, StageStatus.APPLIED),
        ),
    ],
)
def test_processing_rejects_no_started_or_no_unfinished_work(
    receipts: tuple[StageReceipt, ...],
) -> None:
    with pytest.raises(ValueError, match="started and unfinished"):
        validate_required_stage_claim(
            claimed_status=IntentStatus.PROCESSING,
            intent=intent(ProcessingStage.CANONICAL, ProcessingStage.GRAPH),
            receipts=receipts,
        )


def test_accepted_still_allows_empty_or_pending_required_work() -> None:
    command = intent(ProcessingStage.CANONICAL, ProcessingStage.GRAPH)

    validate_required_stage_claim(
        claimed_status=IntentStatus.ACCEPTED,
        intent=command,
        receipts=(),
    )
    validate_required_stage_claim(
        claimed_status=IntentStatus.ACCEPTED,
        intent=command,
        receipts=(
            receipt(ProcessingStage.CANONICAL, StageStatus.PENDING),
            receipt(ProcessingStage.GRAPH, StageStatus.PENDING),
        ),
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


def test_pending_receipt_cannot_claim_started_work() -> None:
    with pytest.raises(ValidationError, match="pending stage receipts cannot have started_at"):
        StageReceipt(
            schema_version="candidate-v1",
            intent_id="intent-1",
            attempt=1,
            stage=ProcessingStage.GRAPH,
            status=StageStatus.PENDING,
            applied_sources=(),
            output_refs=(),
            error=None,
            started_at=NOW,
            finished_at=None,
        )

    command = intent(ProcessingStage.GRAPH)
    validate_required_stage_claim(
        claimed_status=IntentStatus.ACCEPTED,
        intent=command,
        receipts=(receipt(ProcessingStage.GRAPH, StageStatus.PENDING, sources=()),),
    )


def test_all_lifecycle_timestamps_reject_naive_native_datetimes() -> None:
    naive = datetime(2026, 9, 28, 12, 0)

    with pytest.raises(ValidationError, match="timezone"):
        memory_record(recorded_at=naive)
    with pytest.raises(ValidationError, match="timezone"):
        memory_record(occurred_at=naive)

    command_values = intent(ProcessingStage.CANONICAL).model_dump()
    command_values["accepted_at"] = naive
    with pytest.raises(ValidationError, match="timezone"):
        Intent.model_validate(command_values)

    with pytest.raises(ValidationError, match="timezone"):
        StageReceipt(
            schema_version="candidate-v1",
            intent_id="intent-1",
            attempt=1,
            stage=ProcessingStage.GRAPH,
            status=StageStatus.APPLIED,
            applied_sources=(source("memory-1"),),
            output_refs=(),
            error=None,
            started_at=naive,
            finished_at=naive + timedelta(seconds=1),
        )


def test_receipt_json_rejects_naive_mixed_and_reversed_timestamps() -> None:
    payload = {
        "schema_version": "candidate-v1",
        "intent_id": "intent-1",
        "attempt": 1,
        "stage": "graph",
        "status": "applied",
        "applied_sources": [
            {"record_id": "memory-1", "content_revision": 3, "policy_epoch": 7}
        ],
        "output_refs": [],
        "error": None,
        "started_at": "2026-09-28T12:00:00Z",
        "finished_at": "2026-09-28T12:00:01Z",
    }
    assert StageReceipt.model_validate_json(json.dumps(payload)).finished_at is not None

    naive = dict(payload, started_at="2026-09-28T12:00:00")
    with pytest.raises(ValidationError, match="timezone"):
        StageReceipt.model_validate_json(json.dumps(naive))

    mixed = dict(payload, finished_at="2026-09-28T12:00:01")
    with pytest.raises(ValidationError, match="timezone"):
        StageReceipt.model_validate_json(json.dumps(mixed))

    reversed_times = dict(
        payload,
        started_at="2026-09-28T12:00:02Z",
        finished_at="2026-09-28T12:00:01Z",
    )
    with pytest.raises(ValidationError, match="cannot precede"):
        StageReceipt.model_validate_json(json.dumps(reversed_times))
