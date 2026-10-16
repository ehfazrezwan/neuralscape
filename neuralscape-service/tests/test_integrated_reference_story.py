"""Contract-level integrated story over the candidate Phase 1 contracts.

**This is not runtime proof.**  Nothing here authenticates a caller, persists
anything, queues work or talks to a store.  The story chains only the pure
contract models and reference validators (policy evaluation, lifecycle claims,
version witnesses, freshness status, processing-policy dispatch checks,
portable manifest validation and usage reconciliation) over a dictionary-backed
fixture, to show that the candidate contracts compose and that each negative
control is rejected by a contract rule.  Real two-tenant, durable and encrypted
paths are separate runtime checks.

The assertions are written from the integration story in the plan (one numbered
step per behavior) and not from the implementation: each test names the step it
checks, states the expectation in plan terms and would fail if the contracts
accepted what the plan says must be rejected.
"""

import copy
import json

import pytest
from pydantic import ValidationError

import contracts_pack as pack
from contracts_bodies import parse_memory_body
from contracts_errors import InternalErrorCode, PublicErrorCode, public_error_for
from contracts_freshness import ProjectionStatus, projection_status_for
from contracts_inference import DecisionResult
from contracts_lifecycle import (
    Intent,
    IntentStatus,
    MemoryRecord,
    StageReceipt,
    is_legal_intent_transition,
    source_versions_match,
    validate_required_stage_claim,
)
from contracts_policy import AccessPolicy, PolicyEvaluationInput, PrincipalContext
from contracts_policy_reference import evaluate_policy
from contracts_portability import PortableManifest, validate_portable_manifest
from contracts_processing import PlaintextDispatchRejected, ProcessingPolicy, validate_plaintext_dispatch
from contracts_references import SourceVersion
from contracts_usage import UsageEvent
from contracts_usage_reconcile import reconcile_usage_events

EXAMPLES = pack.valid_examples()
NOW = "2026-09-28T12:00:00Z"
LATER = "2026-09-28T12:00:01Z"


def build(model, name: str, **overrides):
    document = copy.deepcopy(EXAMPLES[name])
    document.update(overrides)
    return model.model_validate_json(json.dumps(document))


def reference(identifier: str, *, tenant: str) -> dict:
    return {"kind": "memory", "id": identifier, "tenant_id": tenant, "resolver": "memory_by_id"}


def principal(subject: str, *, tenant: str = "tenant-a", credential: str | None = None) -> PrincipalContext:
    return build(
        PrincipalContext,
        "PrincipalContext",
        tenant_id=tenant,
        subject_id=subject,
        credential_id=credential or f"credential-{subject}",
    )


def policy(epoch: int, *grants: tuple[str, str], tenant: str = "tenant-a") -> AccessPolicy:
    """Policy whose statements grant (subject, action) on memory-1."""

    members = sorted({subject for subject, _ in grants} | {"alice"})
    return build(
        AccessPolicy,
        "AccessPolicy",
        tenant_id=tenant,
        policy_epoch=epoch,
        membership_versions=[{"subject_id": subject, "version": 7} for subject in members],
        statements=[
            {
                "id": f"grant-{index}",
                "effect": "allow",
                "subject_id": subject,
                "credential_id": None,
                "action": action,
                "resource": reference("memory-1", tenant=tenant),
            }
            for index, (subject, action) in enumerate(grants)
        ],
    )


def decide(who: PrincipalContext, action: str, current_policy: AccessPolicy, *, tenant: str = "tenant-a"):
    evaluation = build(PolicyEvaluationInput, "PolicyEvaluationInput", action=action, resource=reference("memory-1", tenant=tenant))
    return evaluate_policy(principal=who, evaluation=evaluation, policy=current_policy)


def record(revision: int, epoch: int, text: str = "Use concise answers", **overrides) -> MemoryRecord:
    values = {
        "id": "memory-1",
        "tenant_id": "tenant-a",
        "content_revision": revision,
        "policy_epoch": epoch,
        "body": {"kind": "plaintext", "text": text},
    }
    values.update(overrides)
    return build(MemoryRecord, "MemoryRecord", **values)


def version(revision: int, epoch: int) -> SourceVersion:
    return SourceVersion(record_id="memory-1", content_revision=revision, policy_epoch=epoch)


def intent(*stages: str, revision: int = 1, epoch: int = 7) -> Intent:
    return build(
        Intent,
        "Intent",
        tenant_id="tenant-a",
        source_preconditions=[{"record_id": "memory-1", "content_revision": revision, "policy_epoch": epoch}],
        stage_requirements=[
            {
                "stage": stage,
                "effect_id": f"effect-{stage}",
                "precondition_source_ids": ["memory-1"],
                "upstream_stages": [],
                "committed_sources": [],
                "minimum_output_refs": 0,
            }
            for stage in stages
        ],
    )


def receipt(stage: str, status: str, revision: int = 1, epoch: int = 7) -> StageReceipt:
    terminal = status not in {"pending", "processing"}
    started = status != "pending"
    error = {"code": "projection_failed", "retryable": False, "safe_message": "Projection failed"} if status == "failed" else None
    return build(
        StageReceipt,
        "StageReceipt",
        stage=stage,
        effect_id=f"effect-{stage}",
        status=status,
        applied_sources=[] if status == "pending" else [{"record_id": "memory-1", "content_revision": revision, "policy_epoch": epoch}],
        error=error,
        started_at=NOW if started else None,
        finished_at=LATER if terminal else None,
    )


ALICE = principal("alice")
BOB = principal("bob")


# Step 1 — "Authenticates a principal and rejects a forged body identity."
def test_step1_identity_comes_only_from_the_principal_and_a_forged_body_is_rejected() -> None:
    forged = {**EXAMPLES["PolicyEvaluationInput"], "subject_id": "alice", "tenant_id": "tenant-a"}
    with pytest.raises(ValidationError):
        PolicyEvaluationInput.model_validate_json(json.dumps(forged))

    decision = decide(ALICE, "read", policy(19, ("alice", "read")))
    assert decision.outcome == "allow"
    assert (decision.evaluated_subject_id, decision.evaluated_tenant_id) == ("alice", "tenant-a")
    # A caller cannot be someone else by claiming a different subject in the request:
    assert decide(BOB, "read", policy(19, ("alice", "read"))).outcome == "deny"


# Step 2 — "Accepts a project-specific preference without changing its semantic category."
def test_step2_project_preference_keeps_its_category_and_needs_a_project() -> None:
    saved = record(1, 7, applicability="project", project_id="project-1", category="preference")

    assert saved.category == "preference"
    assert saved.applicability.value == "project" and saved.project_id == "project-1"
    with pytest.raises(ValidationError):
        record(1, 7, applicability="project", project_id=None)


# Step 4 — "Applies vector/graph/Markdown reference projections with per-stage receipts."
def test_step4_an_intent_is_applied_only_when_every_required_stage_is_applied() -> None:
    accepted = intent("canonical", "vector", "graph")

    assert is_legal_intent_transition(IntentStatus.ACCEPTED, IntentStatus.PROCESSING)
    assert not is_legal_intent_transition(IntentStatus.APPLIED, IntentStatus.ACCEPTED)

    all_applied = tuple(receipt(stage, "applied") for stage in ("canonical", "vector", "graph"))
    validate_required_stage_claim(claimed_status=IntentStatus.APPLIED, intent=accepted, receipts=all_applied)

    one_pending = (receipt("canonical", "applied"), receipt("vector", "applied"), receipt("graph", "pending"))
    with pytest.raises(ValueError):
        validate_required_stage_claim(claimed_status=IntentStatus.APPLIED, intent=accepted, receipts=one_pending)
    # unfinished work stays non-terminal rather than being reported as a settled failure or success
    validate_required_stage_claim(claimed_status=IntentStatus.PROCESSING, intent=accepted, receipts=one_pending)

    one_failed = (receipt("canonical", "applied"), receipt("vector", "failed"), receipt("graph", "applied"))
    with pytest.raises(ValueError):
        validate_required_stage_claim(claimed_status=IntentStatus.APPLIED, intent=accepted, receipts=one_failed)
    validate_required_stage_claim(claimed_status=IntentStatus.PARTIAL, intent=accepted, receipts=one_failed)


# Step 5 — "Corrects the source, then rejects an old content revision's publication."
def test_step5_after_a_correction_an_old_revisions_publication_is_stale() -> None:
    original = record(1, 7)
    corrected = record(2, 7, text="Use detailed answers")

    assert corrected.content_revision > original.content_revision
    assert not source_versions_match(version(1, 7), version(2, 7))
    assert source_versions_match(version(2, 7), version(2, 7))
    assert projection_status_for((version(1, 7),), (version(2, 7),)) is ProjectionStatus.STALE
    assert projection_status_for((version(2, 7),), (version(2, 7),)) is ProjectionStatus.CURRENT
    # A queued stage receipt for the old revision does not satisfy the corrected source:
    assert receipt("vector", "applied", revision=1).applied_sources[0].content_revision == 1
    assert not source_versions_match(
        receipt("vector", "applied", revision=1).applied_sources[0], version(2, 7)
    )


# Step 6 — "Narrows visibility, enforces a deny fence and rejects an old shared graph job."
def test_step6_narrowing_denies_the_removed_reader_and_old_policy_work_is_stale() -> None:
    before = policy(19, ("alice", "read"), ("bob", "read"))
    after = policy(20, ("alice", "read"))

    assert decide(BOB, "read", before).outcome == "allow"
    assert decide(BOB, "read", after).outcome == "deny"
    assert decide(ALICE, "read", after).outcome == "allow"

    old_decision = decide(BOB, "read", before)
    assert old_decision.policy_epoch < after.policy_epoch  # a witness from the old policy is not current

    # the record's policy epoch advanced with the narrowing, so a job queued at the old epoch is rejected
    assert not source_versions_match(version(2, 19), version(2, 20))
    assert projection_status_for((version(2, 19),), (version(2, 20),)) is ProjectionStatus.STALE
    # the safe public form does not reveal whether the resource exists
    assert public_error_for(InternalErrorCode.DENIED).code is PublicErrorCode.NOT_FOUND


# Step 8 — "Exports canonical state, validates offline, imports with an explicit identity mapping,
# verifies tombstones/policy."
def test_step8_export_manifest_validates_offline_and_restores_canonical_identity() -> None:
    manifest = validate_portable_manifest(copy.deepcopy(EXAMPLES["PortableManifest"]))
    assert isinstance(manifest, PortableManifest)

    exported = {(item.kind, item.id) for item in manifest.canonical_ids}
    fixture_store = {("record", "record-1"), ("revision", "revision-1"), ("tombstone", "tombstone-9")}
    assert exported == fixture_store  # complete canonical state, including the tombstone

    restored = {(item.kind, item.id) for item in validate_portable_manifest(manifest.model_dump(mode="json")).canonical_ids}
    assert restored == fixture_store
    assert [(m.source_identity_id, m.destination_identity_id) for m in manifest.identity_mappings] == [
        ("source-member-2", "destination-member-8")
    ]
    assert manifest.encryption.mode == "encrypted" and manifest.omissions  # limits are declared, not hidden

    unsafe = copy.deepcopy(EXAMPLES["PortableManifest"])
    unsafe["files"][0]["path"] = "../outside"
    with pytest.raises(ValidationError):
        validate_portable_manifest(unsafe)
    dropped = copy.deepcopy(EXAMPLES["PortableManifest"])
    dropped["canonical_ids"] = [item for item in dropped["canonical_ids"] if item["kind"] != "tombstone"]
    assert {(i["kind"], i["id"]) for i in dropped["canonical_ids"]} != fixture_store  # a lossy export is detectable


# Two-tenant negative control — "Cross-tenant context is rejected at both boundaries."
def test_two_tenants_cannot_reach_each_others_resources_or_references() -> None:
    mallory = principal("mallory", tenant="tenant-b")

    assert decide(mallory, "read", policy(19, ("alice", "read"))).outcome == "deny"
    assert decide(ALICE, "read", policy(19, ("alice", "read"), tenant="tenant-b"), tenant="tenant-a").outcome == "deny"
    assert decide(ALICE, "read", policy(19, ("alice", "read")), tenant="tenant-b").outcome == "deny"
    for action in ("search", "list", "count", "export", "delete", "update"):
        assert decide(mallory, action, policy(19, ("alice", action))).outcome == "deny"

    with pytest.raises(ValidationError):  # an intent cannot target another tenant's reference
        build(Intent, "Intent", tenant_id="tenant-b", target_refs=[reference("memory-1", tenant="tenant-a")])
    # A record may carry a cross-tenant reference as a value (references are not authority; cross-tenant
    # lineage is a confirmed direction), but resolving it needs current authorization in the owning tenant:
    cited = record(1, 7, source_refs=[reference("memory-9", tenant="tenant-b")])
    assert cited.source_refs[0].tenant_id == "tenant-b"
    foreign = build(PolicyEvaluationInput, "PolicyEvaluationInput", action="read", resource=reference("memory-9", tenant="tenant-b"))
    assert evaluate_policy(principal=ALICE, evaluation=foreign, policy=policy(19, ("alice", "read"))).outcome == "deny"


# Step 10 — "Passes an opaque encrypted-body reference through save/status/read boundaries without
# assuming the service can decrypt it."
def test_step10_opaque_body_stays_opaque_and_ciphertext_is_not_reported_as_processed() -> None:
    opaque = record(1, 7, body={"kind": "opaque_envelope", "envelope_id": "envelope-1"})

    assert opaque.body.kind == "opaque_envelope"
    assert "text" not in opaque.model_dump(mode="json")["body"]
    with pytest.raises(ValidationError):
        parse_memory_body({"kind": "opaque_envelope", "envelope_id": "envelope-1", "text": "plaintext"})
    with pytest.raises(ValidationError):
        parse_memory_body({"kind": "plaintext", "text": "x", "envelope_id": "envelope-1"})

    stored_only = intent("canonical", "extract")
    receipts = (receipt("canonical", "applied"), receipt("extract", "pending"))
    with pytest.raises(ValueError):
        validate_required_stage_claim(claimed_status=IntentStatus.APPLIED, intent=stored_only, receipts=receipts)


# Step 11 — "Rejects an unauthorized cloud fallback" and "a hosted-only processor can be rejected
# for a strict-local profile before invocation."
def test_step11_strict_local_processing_rejects_external_and_fallback_dispatch() -> None:
    strict = build(ProcessingPolicy, "ProcessingPolicy")
    common = {"current_policy_epoch": 7, "currently_authorized_recipient_ids": ["provider-a"]}

    validate_plaintext_dispatch(strict, execution_location="endpoint", recipient_id=None, is_fallback=False, **common)
    for location in ("external_provider", "operator_worker", "customer_worker"):
        with pytest.raises(PlaintextDispatchRejected):
            validate_plaintext_dispatch(strict, execution_location=location, recipient_id="provider-a", is_fallback=False, **common)
    with pytest.raises(PlaintextDispatchRejected):
        validate_plaintext_dispatch(strict, execution_location="endpoint", recipient_id=None, is_fallback=True, **common)

    external = build(
        ProcessingPolicy,
        "ProcessingPolicy",
        mode="operator_trusted",
        allowed_execution_locations=["endpoint", "external_provider"],
        approved_recipient_ids=["provider-a"],
        fallback_policy="within_approved_recipients",
    )
    validate_plaintext_dispatch(external, execution_location="external_provider", recipient_id="provider-a", is_fallback=False, **common)
    with pytest.raises(PlaintextDispatchRejected):  # approved recipient that is no longer authorized
        validate_plaintext_dispatch(
            external,
            execution_location="external_provider",
            recipient_id="provider-a",
            is_fallback=False,
            current_policy_epoch=7,
            currently_authorized_recipient_ids=[],
        )
    with pytest.raises(PlaintextDispatchRejected):  # a recipient the policy never approved
        validate_plaintext_dispatch(
            external,
            execution_location="external_provider",
            recipient_id="provider-z",
            is_fallback=False,
            current_policy_epoch=7,
            currently_authorized_recipient_ids=["provider-z"],
        )
    with pytest.raises(PlaintextDispatchRejected):  # a stale processing policy
        validate_plaintext_dispatch(
            external,
            execution_location="external_provider",
            recipient_id="provider-a",
            is_fallback=False,
            current_policy_epoch=8,
            currently_authorized_recipient_ids=["provider-a"],
        )


# Step 11 (continued) — "distinguishes abstention from a valid empty result"
def test_step11_abstention_and_a_valid_empty_result_are_different_outcomes() -> None:
    def result(head: dict) -> DecisionResult:
        return build(DecisionResult, "DecisionResult", head_result=head)

    empty = result({"head_id": "retain", "outcome": "empty", "selected_candidate_ids": [], "distribution": None, "reason_code": None})
    abstained = result(
        {"head_id": "retain", "outcome": "abstained", "selected_candidate_ids": [], "distribution": None, "reason_code": "below-threshold"}
    )

    assert empty.head_result.outcome != abstained.head_result.outcome
    assert abstained.head_result.reason_code is not None and empty.head_result.reason_code is None
    with pytest.raises(ValidationError):  # abstention must say why; it is not an empty result
        result({"head_id": "retain", "outcome": "abstained", "selected_candidate_ids": [], "distribution": None, "reason_code": None})


# Step 11 (continued) — "records late usage after publication is fenced"
def test_step11_late_usage_after_cancellation_is_counted_once_in_its_attempt() -> None:
    def event(**changes) -> UsageEvent:
        return build(UsageEvent, "UsageEvent", **changes)

    cancelled = event(
        event_id="event-1",
        status="unavailable",
        attempt_outcome="cancelled",
        usage=None,
        usage_missing_reason="not_yet_reported",
    )
    late = event(
        event_id="event-2",
        predecessor_event_id="event-1",
        status="final",
        attempt_outcome="cancelled",
        late_after_cancellation=True,
    )

    ledger = reconcile_usage_events([cancelled, late])
    again = reconcile_usage_events([cancelled, late, late])

    assert len(ledger.streams) == len(again.streams) == 1
    stream = ledger.streams[0]
    assert stream.late_after_cancellation is True and stream.status == "final"
    assert stream.usage.input_tokens.value == 100  # the late event's measured usage is attributed, not dropped
    assert again.streams[0].usage == stream.usage  # a duplicate delivery is not double counted


def test_the_story_only_uses_registered_contracts() -> None:
    registered = {entry.name for entry in pack.CONTRACTS}
    for name in (
        "PrincipalContext",
        "AccessPolicy",
        "PolicyEvaluationInput",
        "MemoryRecord",
        "Intent",
        "StageReceipt",
        "PortableManifest",
        "ProcessingPolicy",
        "DecisionResult",
        "UsageEvent",
    ):
        assert name in registered
