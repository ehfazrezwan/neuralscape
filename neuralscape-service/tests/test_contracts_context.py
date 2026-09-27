"""Contract tests for bounded context delivery and safe assembly receipts."""

import json

import pytest
from pydantic import ValidationError

from contracts_context import (
    ContextAssemblyReceipt,
    ContextBundle,
    ContextOutcome,
    ContextRequest,
    SafeOmissionCode,
    bundle_fits_request,
    receipt_matches_bundle,
)


NOW = "2026-09-28T12:00:00Z"
DEADLINE = "2026-09-28T12:00:05Z"


def _reference(
    kind: str = "memory", identifier: str = "memory-1"
) -> dict[str, object]:
    return {
        "kind": kind,
        "id": identifier,
        "tenant_id": "tenant-1",
        "resolver": f"resolve_{kind}",
    }


def _version(record_id: str = "source-a", revision: int = 4) -> dict[str, object]:
    return {"record_id": record_id, "content_revision": revision, "policy_epoch": 2}


def _tokenizer() -> dict[str, str]:
    return {"tokenizer_id": "example-tokenizer", "tokenizer_version": "2026-09"}


def _usage(tokens: int = 80) -> dict[str, object]:
    return {
        "tokens": tokens,
        "scope": "entire_serialized_response",
        "tokenizer": _tokenizer(),
    }


def _request(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": "candidate-v1",
        "request_id": "request-1",
        "task_intent": "Continue the synthetic queue migration",
        "requested_applicability": {"project_id": "project-1", "workspace_id": None},
        "time_perspective": "current",
        "as_of": None,
        "response_budget": {
            "max_tokens": 100,
            "scope": "entire_serialized_response",
            "tokenizer": _tokenizer(),
        },
        "deadline": DEADLINE,
        "freshness_requirement": "current_verified",
        "prerequisite_receipt": None,
    }
    value.update(overrides)
    return value


def _freshness() -> dict[str, object]:
    return {
        "upstream_status": "stale",
        "source_checkpoints": [
            {
                "schema_version": "candidate-v1",
                "source": _reference("source", "source-a"),
                "verification_status": "stale",
                "verification_method": "provider_revision",
                "observed_at": NOW,
                "last_successful_verification_at": "2026-09-28T11:00:00Z",
                "provider_revision": "revision-4",
                "opaque_cursor": None,
                "processed_through": "revision-4",
                "reconciliation_state": "complete",
                "freshness_policy": "standard",
            }
        ],
        "projection": {
            "status": "current",
            "applied_sources": [_version()],
            "verified_at": NOW,
        },
    }


def _item() -> dict[str, object]:
    return {
        "reference": _reference(),
        "source_versions": [_version()],
        "expansion_handle": _reference(),
        "temporal_status": "current",
        "evidence_form": "exact",
        "evidence_text": "Synthetic current decision.",
        "uncertainty": "none_reported",
        "freshness": _freshness(),
    }


def _bundle(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": "candidate-v1",
        "bundle_id": "bundle-1",
        "request_id": "request-1",
        "outcome": "complete",
        "capability_status": "available",
        "selected_items": [_item()],
        "omissions": [],
        "response_usage": _usage(),
        "assembly_receipt": _reference("artifact", "receipt-1"),
    }
    value.update(overrides)
    return value


def _receipt(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": "candidate-v1",
        "receipt_id": "receipt-1",
        "request_id": "request-1",
        "bundle_id": "bundle-1",
        "selector_version": "selector-1",
        "router_version": "router-1",
        "source_versions": [_version()],
        "policy_decision_witnesses": ["policy-decision-1"],
        "selected_references": [_reference()],
        "exclusions": [],
        "degradations": ["upstream_verification_stale"],
        "stage_timings": [
            {"stage": "authorize", "elapsed_ms": 2},
            {"stage": "assemble", "elapsed_ms": 5},
        ],
        "response_usage": _usage(),
        "optimization_lineage": [],
        "expansion_lineage": [],
    }
    value.update(overrides)
    return value


def _validate(model: type, payload: dict[str, object]):
    return model.model_validate_json(json.dumps(payload))


def test_current_and_historical_requests_are_not_interchangeable() -> None:
    current = _validate(ContextRequest, _request())
    historical = _validate(
        ContextRequest,
        _request(time_perspective="historical", as_of="2026-09-01T00:00:00Z"),
    )

    assert current.as_of is None
    assert historical.as_of is not None

    with pytest.raises(ValidationError, match="historical request requires as_of"):
        _validate(ContextRequest, _request(time_perspective="historical"))
    with pytest.raises(ValidationError, match="current request cannot specify as_of"):
        _validate(ContextRequest, _request(as_of="2026-09-01T00:00:00Z"))


def test_request_has_no_identity_or_authority_override() -> None:
    payload = _request(subject_id="attacker", allowed=True)

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        _validate(ContextRequest, payload)


def test_receipt_fields_require_typed_artifact_references() -> None:
    with pytest.raises(ValidationError, match="prerequisite_receipt requires"):
        _validate(
            ContextRequest,
            _request(prerequisite_receipt=_reference("memory", "not-a-receipt")),
        )

    with pytest.raises(ValidationError, match="assembly_receipt requires"):
        _validate(
            ContextBundle,
            _bundle(assembly_receipt=_reference("memory", "not-a-receipt")),
        )


def test_budget_is_whole_response_and_uses_the_declared_tokenizer() -> None:
    request = _validate(ContextRequest, _request())
    bundle = _validate(ContextBundle, _bundle())
    assert bundle_fits_request(request, bundle)

    over_budget = _validate(ContextBundle, _bundle(response_usage=_usage(101)))
    wrong_basis = _validate(
        ContextBundle,
        _bundle(
            response_usage={
                **_usage(),
                "tokenizer": {
                    "tokenizer_id": "other-tokenizer",
                    "tokenizer_version": "1",
                },
            }
        ),
    )
    assert not bundle_fits_request(request, over_budget)
    assert not bundle_fits_request(request, wrong_basis)


@pytest.mark.parametrize(
    ("outcome", "omissions"),
    [
        ("no_relevant_evidence", []),
        ("unavailable_dependency", ["dependency_unavailable"]),
        ("budget_exhausted", ["budget_limit"]),
    ],
)
def test_empty_outcomes_remain_distinct(
    outcome: str, omissions: list[str]
) -> None:
    bundle = _validate(
        ContextBundle,
        _bundle(
            outcome=outcome,
            capability_status="unavailable" if omissions else "available",
            selected_items=[],
            omissions=omissions,
        ),
    )
    assert bundle.outcome.value == outcome


def test_incomplete_bundle_requires_safe_reason_without_hidden_counts() -> None:
    bundle = _validate(
        ContextBundle,
        _bundle(
            outcome="incomplete",
            capability_status="degraded",
            omissions=["policy_restricted"],
        ),
    )
    dumped = bundle.model_dump(mode="json")

    assert dumped["omissions"] == ["policy_restricted"]
    assert "omitted_count" not in dumped
    assert "excluded_ids" not in dumped

    with pytest.raises(ValidationError, match="requires a safe omission reason"):
        _validate(ContextBundle, _bundle(outcome="incomplete", omissions=[]))


def test_empty_failure_outcome_cannot_smuggle_selected_evidence() -> None:
    with pytest.raises(ValidationError, match="cannot contain selected items"):
        _validate(
            ContextBundle,
            _bundle(
                outcome="unavailable_dependency",
                omissions=["dependency_unavailable"],
            ),
        )


def test_current_projection_must_match_full_selected_source_set() -> None:
    item = _item()
    item["source_versions"] = [_version(revision=4), _version("source-b", 9)]
    freshness = item["freshness"]
    assert isinstance(freshness, dict)
    projection = freshness["projection"]
    assert isinstance(projection, dict)
    projection["applied_sources"] = [
        _version(revision=3),
        _version("source-b", 9),
    ]

    with pytest.raises(ValidationError, match="complete selected source set"):
        _validate(ContextBundle, _bundle(selected_items=[item]))


def test_receipt_corresponds_to_delivered_bundle_and_revision_set() -> None:
    bundle = _validate(ContextBundle, _bundle())
    receipt = _validate(ContextAssemblyReceipt, _receipt())
    assert receipt_matches_bundle(receipt, bundle)

    stale_receipt = _validate(
        ContextAssemblyReceipt,
        _receipt(source_versions=[_version(revision=3)]),
    )
    assert not receipt_matches_bundle(stale_receipt, bundle)


def test_receipt_rejects_sensitive_diagnostics_and_hidden_counts() -> None:
    payload = _receipt(
        raw_query="private task text",
        excluded_object_ids=["secret-1"],
        denied_object_count=7,
        chain_of_thought="not observable evidence",
    )

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        _validate(ContextAssemblyReceipt, payload)


def test_budget_shortfall_is_explicit_and_requires_matching_safe_code() -> None:
    bundle = _validate(
        ContextBundle,
        _bundle(
            outcome="budget_exhausted",
            capability_status="degraded",
            selected_items=[],
            omissions=["budget_limit"],
            response_usage=_usage(35),
        ),
    )
    assert bundle.outcome is ContextOutcome.BUDGET_EXHAUSTED
    assert bundle.omissions == (SafeOmissionCode.BUDGET_LIMIT,)

    with pytest.raises(ValidationError, match="requires its omission code"):
        _validate(
            ContextBundle,
            _bundle(
                outcome="budget_exhausted",
                selected_items=[],
                omissions=["deadline_limit"],
            ),
        )
