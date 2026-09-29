"""Contract tests for bounded context delivery and safe assembly receipts."""

import json
from copy import deepcopy

import pytest
from pydantic import ValidationError

from contracts_context import (
    ContextAssemblyReceipt,
    ContextBundle,
    ContextOutcome,
    ContextRequest,
    ExpansionLineage,
    SafeDegradationCode,
    SafeOmissionCode,
    bundle_fits_request,
    receipt_matches_bundle,
    upstream_degradation_for,
)
from contracts_freshness import UpstreamVerificationStatus


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
    request = _validate(
        ContextRequest,
        _request(freshness_requirement="labelled_stale_acceptable"),
    )
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


def test_request_fit_rejects_stale_upstream_for_current_verified() -> None:
    request = _validate(ContextRequest, _request())
    stale_bundle = _validate(ContextBundle, _bundle())

    assert not bundle_fits_request(request, stale_bundle)

    current_item = _item()
    freshness = current_item["freshness"]
    assert isinstance(freshness, dict)
    freshness["upstream_status"] = "current"
    checkpoints = freshness["source_checkpoints"]
    assert isinstance(checkpoints, list)
    checkpoint = checkpoints[0]
    assert isinstance(checkpoint, dict)
    checkpoint["verification_status"] = "current"
    current_bundle = _validate(ContextBundle, _bundle(selected_items=[current_item]))

    assert bundle_fits_request(request, current_bundle)


@pytest.mark.parametrize(
    ("upstream_status", "projection_status", "expected"),
    [
        (
            upstream,
            projection,
            upstream in {"current", "stale"}
            and projection in {"current", "stale"},
        )
        for upstream in ("current", "stale", "unverified", "unavailable")
        for projection in ("current", "stale", "unknown", "unavailable")
    ],
)
def test_labelled_stale_accepts_only_known_upstream_and_projection_states(
    upstream_status: str,
    projection_status: str,
    expected: bool,
) -> None:
    item = _item()
    freshness = item["freshness"]
    assert isinstance(freshness, dict)
    freshness["upstream_status"] = upstream_status
    checkpoints = freshness["source_checkpoints"]
    assert isinstance(checkpoints, list)
    if upstream_status == "current":
        checkpoint = checkpoints[0]
        assert isinstance(checkpoint, dict)
        checkpoint["verification_status"] = "current"
    elif upstream_status in {"unverified", "unavailable"}:
        freshness["source_checkpoints"] = []
    projection = freshness["projection"]
    assert isinstance(projection, dict)
    projection["status"] = projection_status
    request = _validate(
        ContextRequest,
        _request(freshness_requirement="labelled_stale_acceptable"),
    )
    bundle = _validate(ContextBundle, _bundle(selected_items=[item]))

    assert bundle_fits_request(request, bundle) is expected


def test_request_fit_rejects_historical_item_for_current_request() -> None:
    item = _item()
    item["temporal_status"] = "historical"
    request = _validate(
        ContextRequest,
        _request(freshness_requirement="labelled_stale_acceptable"),
    )
    bundle = _validate(ContextBundle, _bundle(selected_items=[item]))

    assert not bundle_fits_request(request, bundle)


def test_request_fit_does_not_claim_historical_as_of_without_witness() -> None:
    item = _item()
    item["temporal_status"] = "historical"
    request = _validate(
        ContextRequest,
        _request(
            time_perspective="historical",
            as_of="2026-09-01T00:00:00Z",
            freshness_requirement="labelled_stale_acceptable",
        ),
    )
    bundle = _validate(ContextBundle, _bundle(selected_items=[item]))

    assert not bundle_fits_request(request, bundle)


@pytest.mark.parametrize(
    "freshness_requirement",
    ["current_verified", "current_known", "labelled_stale_acceptable"],
)
def test_empty_no_evidence_bundle_never_claims_request_fit(
    freshness_requirement: str,
) -> None:
    request = _validate(
        ContextRequest,
        _request(freshness_requirement=freshness_requirement),
    )
    bundle = _validate(
        ContextBundle,
        _bundle(
            outcome="no_relevant_evidence",
            selected_items=[],
        ),
    )

    assert not bundle_fits_request(request, bundle)


def test_request_fit_revalidates_copied_and_constructed_requests() -> None:
    request = _validate(
        ContextRequest,
        _request(freshness_requirement="labelled_stale_acceptable"),
    )
    bundle = _validate(ContextBundle, _bundle())

    copied = request.model_copy(
        update={"freshness_requirement": "unsupported-future-mode"}
    )
    constructed = ContextRequest.model_construct(
        **{
            **request.model_dump(),
            "freshness_requirement": "unsupported-future-mode",
        }
    )
    copied_with_extra = request.model_copy(update={"unreviewed_mode": True})

    assert not bundle_fits_request(copied, bundle)
    assert not bundle_fits_request(constructed, bundle)
    assert not bundle_fits_request(copied_with_extra, bundle)


def test_request_fit_rejects_nested_copied_extra_and_cyclic_graph() -> None:
    request = _validate(
        ContextRequest,
        _request(freshness_requirement="labelled_stale_acceptable"),
    )
    bundle = _validate(ContextBundle, _bundle())
    copied_item = bundle.selected_items[0].model_copy(
        update={"unreviewed_selection": True}
    )
    copied_bundle = bundle.model_copy(update={"selected_items": (copied_item,)})
    cycle: list[object] = []
    cycle.append(cycle)
    cyclic_request = request.model_copy(update={"unreviewed_cycle": cycle})

    assert not bundle_fits_request(request, copied_bundle)
    assert not bundle_fits_request(cyclic_request, bundle)


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


def test_selected_item_rejects_cross_tenant_freshness_scope() -> None:
    item = _item()
    freshness = item["freshness"]
    assert isinstance(freshness, dict)
    checkpoints = freshness["source_checkpoints"]
    assert isinstance(checkpoints, list)
    checkpoint = checkpoints[0]
    assert isinstance(checkpoint, dict)
    source = checkpoint["source"]
    assert isinstance(source, dict)
    source["tenant_id"] = "tenant-2"

    with pytest.raises(ValidationError, match="selected reference tenant scope"):
        _validate(ContextBundle, _bundle(selected_items=[item]))


def test_bundle_rejects_selected_item_outside_receipt_tenant_scope() -> None:
    item = _item()
    reference = item["reference"]
    expansion = item["expansion_handle"]
    assert isinstance(reference, dict)
    assert isinstance(expansion, dict)
    reference["tenant_id"] = "tenant-2"
    expansion["tenant_id"] = "tenant-2"
    freshness = item["freshness"]
    assert isinstance(freshness, dict)
    checkpoints = freshness["source_checkpoints"]
    assert isinstance(checkpoints, list)
    checkpoint = checkpoints[0]
    assert isinstance(checkpoint, dict)
    source = checkpoint["source"]
    assert isinstance(source, dict)
    source["tenant_id"] = "tenant-2"

    with pytest.raises(ValidationError, match="share one tenant scope"):
        _validate(ContextBundle, _bundle(selected_items=[item]))


def test_bundle_rejects_conflicting_versions_across_selected_items() -> None:
    first = _item()
    second = deepcopy(_item())
    second["reference"] = _reference("memory", "memory-2")
    second["expansion_handle"] = _reference("memory", "memory-2")
    second["source_versions"] = [_version(revision=5)]
    freshness = second["freshness"]
    assert isinstance(freshness, dict)
    projection = freshness["projection"]
    assert isinstance(projection, dict)
    projection["applied_sources"] = [_version(revision=5)]

    with pytest.raises(ValidationError, match="conflicting versions"):
        _validate(ContextBundle, _bundle(selected_items=[first, second]))


def test_same_source_version_across_items_has_representable_receipt() -> None:
    first = _item()
    second = deepcopy(_item())
    second["reference"] = _reference("memory", "memory-2")
    second["expansion_handle"] = _reference("memory", "memory-2")
    bundle = _validate(ContextBundle, _bundle(selected_items=[first, second]))
    receipt = _validate(
        ContextAssemblyReceipt,
        _receipt(selected_references=[first["reference"], second["reference"]]),
    )

    assert receipt_matches_bundle(receipt, bundle)


def test_receipt_match_revalidates_cross_tenant_model_copies() -> None:
    bundle = _validate(ContextBundle, _bundle())
    receipt = _validate(ContextAssemblyReceipt, _receipt())
    item = bundle.selected_items[0]
    copied_reference = item.reference.model_copy(update={"tenant_id": "tenant-2"})
    copied_expansion = item.expansion_handle.model_copy(
        update={"tenant_id": "tenant-2"}
    )
    copied_item = item.model_copy(
        update={
            "reference": copied_reference,
            "expansion_handle": copied_expansion,
        }
    )
    copied_bundle = bundle.model_copy(update={"selected_items": (copied_item,)})
    copied_receipt = receipt.model_copy(
        update={"selected_references": (copied_reference,)}
    )

    assert not receipt_matches_bundle(copied_receipt, copied_bundle)


def test_expansion_lineage_requires_one_tenant_scope() -> None:
    with pytest.raises(ValidationError, match="endpoints must share one tenant"):
        _validate(
            ContextAssemblyReceipt,
            _receipt(
                expansion_lineage=[
                    {
                        "input_reference": _reference(),
                        "output_reference": {
                            **_reference(),
                            "tenant_id": "tenant-2",
                        },
                    }
                ]
            ),
        )


@pytest.mark.parametrize(
    "lineage_field", ["optimization_lineage", "expansion_lineage"]
)
def test_receipt_lineage_uses_selected_reference_tenant(
    lineage_field: str,
) -> None:
    other_tenant = {**_reference(), "tenant_id": "tenant-2"}
    lineage: object
    if lineage_field == "optimization_lineage":
        lineage = [other_tenant]
    else:
        lineage = [
            {
                "input_reference": other_tenant,
                "output_reference": other_tenant,
            }
        ]

    with pytest.raises(ValidationError, match="selected reference tenant scope"):
        _validate(ContextAssemblyReceipt, _receipt(**{lineage_field: lineage}))


def test_receipt_correspondence_requires_exact_expansion_lineage() -> None:
    item = _item()
    item["expansion_handle"] = _reference("artifact", "expansion-1")
    bundle = _validate(ContextBundle, _bundle(selected_items=[item]))
    expected_lineage = {
        "input_reference": item["expansion_handle"],
        "output_reference": item["reference"],
    }
    receipt = _validate(
        ContextAssemblyReceipt,
        _receipt(expansion_lineage=[expected_lineage]),
    )

    assert receipt_matches_bundle(receipt, bundle)
    assert not receipt_matches_bundle(
        _validate(ContextAssemblyReceipt, _receipt(expansion_lineage=[])),
        bundle,
    )
    assert not receipt_matches_bundle(
        _validate(
            ContextAssemblyReceipt,
            _receipt(
                expansion_lineage=[
                    {
                        "input_reference": expected_lineage["output_reference"],
                        "output_reference": expected_lineage["input_reference"],
                    }
                ]
            ),
        ),
        bundle,
    )


def test_receipt_match_revalidates_copied_and_constructed_lineage() -> None:
    bundle = _validate(ContextBundle, _bundle())
    receipt = _validate(ContextAssemblyReceipt, _receipt())
    valid_reference = bundle.selected_items[0].reference
    cross_tenant = valid_reference.model_copy(update={"tenant_id": "tenant-2"})

    copied_lineage = ExpansionLineage(
        input_reference=valid_reference,
        output_reference=valid_reference,
    ).model_copy(update={"output_reference": cross_tenant})
    constructed_lineage = ExpansionLineage.model_construct(
        input_reference=valid_reference,
        output_reference=cross_tenant,
    )

    for lineage in (copied_lineage, constructed_lineage):
        copied_receipt = receipt.model_copy(update={"expansion_lineage": (lineage,)})
        assert not receipt_matches_bundle(copied_receipt, bundle)


def test_receipt_match_revalidates_invalid_nested_source_version_copy() -> None:
    bundle = _validate(ContextBundle, _bundle())
    receipt = _validate(ContextAssemblyReceipt, _receipt())
    item = bundle.selected_items[0]
    copied_version = item.source_versions[0].model_copy(
        update={"content_revision": -1}
    )
    copied_projection = item.freshness.projection.model_copy(
        update={"applied_sources": (copied_version,)}
    )
    copied_freshness = item.freshness.model_copy(
        update={"projection": copied_projection}
    )
    copied_item = item.model_copy(
        update={
            "source_versions": (copied_version,),
            "freshness": copied_freshness,
        }
    )
    copied_bundle = bundle.model_copy(update={"selected_items": (copied_item,)})
    copied_receipt = receipt.model_copy(
        update={"source_versions": (copied_version,)}
    )

    assert not receipt_matches_bundle(copied_receipt, copied_bundle)


def test_helpers_revalidate_copied_cross_item_version_conflict() -> None:
    first = _item()
    second = deepcopy(_item())
    second["reference"] = _reference("memory", "memory-2")
    second["expansion_handle"] = _reference("memory", "memory-2")
    bundle = _validate(ContextBundle, _bundle(selected_items=[first, second]))
    receipt = _validate(
        ContextAssemblyReceipt,
        _receipt(selected_references=[first["reference"], second["reference"]]),
    )
    request = _validate(
        ContextRequest,
        _request(freshness_requirement="labelled_stale_acceptable"),
    )

    second_item = bundle.selected_items[1]
    conflicting_version = second_item.source_versions[0].model_copy(
        update={"content_revision": 5}
    )
    conflicting_projection = second_item.freshness.projection.model_copy(
        update={"applied_sources": (conflicting_version,)}
    )
    conflicting_freshness = second_item.freshness.model_copy(
        update={"projection": conflicting_projection}
    )
    conflicting_item = second_item.model_copy(
        update={
            "source_versions": (conflicting_version,),
            "freshness": conflicting_freshness,
        }
    )
    conflicting_bundle = bundle.model_copy(
        update={"selected_items": (bundle.selected_items[0], conflicting_item)}
    )

    assert not bundle_fits_request(request, conflicting_bundle)
    assert not receipt_matches_bundle(receipt, conflicting_bundle)


def test_helpers_fail_closed_for_constructed_invalid_bundles_and_receipts() -> None:
    request = _validate(
        ContextRequest,
        _request(freshness_requirement="labelled_stale_acceptable"),
    )
    bundle = _validate(ContextBundle, _bundle())
    receipt = _validate(ContextAssemblyReceipt, _receipt())
    constructed_bundle = ContextBundle.model_construct(
        **{
            **bundle.model_dump(),
            "selected_items": (),
        }
    )
    constructed_receipt = ContextAssemblyReceipt.model_construct(
        **{
            **receipt.model_dump(exclude={"router_version"}),
        }
    )

    assert not bundle_fits_request(request, constructed_bundle)
    assert not receipt_matches_bundle(constructed_receipt, bundle)


def test_receipt_corresponds_to_delivered_bundle_and_revision_set() -> None:
    bundle = _validate(ContextBundle, _bundle())
    receipt = _validate(ContextAssemblyReceipt, _receipt())
    assert receipt_matches_bundle(receipt, bundle)

    stale_receipt = _validate(
        ContextAssemblyReceipt,
        _receipt(source_versions=[_version(revision=3)]),
    )
    assert not receipt_matches_bundle(stale_receipt, bundle)


def test_receipt_requires_exact_derived_freshness_degradations() -> None:
    bundle = _validate(ContextBundle, _bundle())

    assert receipt_matches_bundle(
        _validate(
            ContextAssemblyReceipt,
            _receipt(
                degradations=[
                    "upstream_verification_stale",
                    "compact_form_used",
                ]
            ),
        ),
        bundle,
    )
    assert not receipt_matches_bundle(
        _validate(ContextAssemblyReceipt, _receipt(degradations=[])),
        bundle,
    )
    assert not receipt_matches_bundle(
        _validate(
            ContextAssemblyReceipt,
            _receipt(degradations=["upstream_verification_unavailable"]),
        ),
        bundle,
    )


def test_receipt_aggregates_mixed_item_freshness_degradations() -> None:
    first = _item()
    second = deepcopy(_item())
    second["reference"] = _reference("memory", "memory-2")
    second["expansion_handle"] = _reference("memory", "memory-2")
    second["source_versions"] = [_version("source-b", 9)]
    freshness = second["freshness"]
    assert isinstance(freshness, dict)
    freshness["upstream_status"] = "unavailable"
    freshness["source_checkpoints"] = []
    projection = freshness["projection"]
    assert isinstance(projection, dict)
    projection["status"] = "stale"
    projection["applied_sources"] = [_version("source-b", 9)]
    bundle = _validate(ContextBundle, _bundle(selected_items=[first, second]))
    receipt_data = _receipt(
        source_versions=[_version(), _version("source-b", 9)],
        selected_references=[first["reference"], second["reference"]],
        degradations=[
            "upstream_verification_stale",
            "upstream_verification_unavailable",
            "projection_stale",
        ],
    )
    receipt = _validate(ContextAssemblyReceipt, receipt_data)

    assert receipt_matches_bundle(receipt, bundle)
    for omitted in receipt_data["degradations"]:
        degradations = [
            code for code in receipt_data["degradations"] if code != omitted
        ]
        assert not receipt_matches_bundle(
            _validate(
                ContextAssemblyReceipt,
                {**receipt_data, "degradations": degradations},
            ),
            bundle,
        )


def test_upstream_degradation_helper_strictly_validates_scalar_input() -> None:
    assert (
        upstream_degradation_for(UpstreamVerificationStatus.STALE)
        is SafeDegradationCode.UPSTREAM_VERIFICATION_STALE
    )
    with pytest.raises(ValidationError):
        upstream_degradation_for("stale")  # type: ignore[arg-type]


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
