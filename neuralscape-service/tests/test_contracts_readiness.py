"""Contract tests for capability-specific readiness evidence."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from contracts_readiness import (
    CapabilityReadiness,
    TenantReadinessPublication,
    capability_is_ready,
    validate_readiness_publication,
)


VERSION = "candidate-v1"


def observation(
    capability: str = "exact_reads",
    status: str = "healthy",
    age: int | None = 3,
    **updates: object,
) -> CapabilityReadiness:
    document: dict[str, object] = {
        "schema_version": VERSION,
        "capability": capability,
        "status": status,
        "probe_age_seconds": age,
    }
    document.update(updates)
    return CapabilityReadiness.model_validate_json(json.dumps(document))


def publication(**updates: object) -> TenantReadinessPublication:
    document: dict[str, object] = {
        "schema_version": VERSION,
        "tenant_id": "tenant-a",
        "placement_generation": 12,
        "capabilities": [
            observation("api_acceptance", "healthy", 1).model_dump(mode="json"),
            observation("exact_reads", "healthy", 3).model_dump(mode="json"),
            observation(
                "graph_reads", "unavailable", 4, detail_code="store-timeout"
            ).model_dump(mode="json"),
            observation("worker_processing", "unknown", None).model_dump(mode="json"),
        ],
    }
    document.update(updates)
    return TenantReadinessPublication.model_validate_json(json.dumps(document))


@pytest.mark.parametrize("status", ["healthy", "degraded", "unavailable"])
def test_measured_statuses_require_explicit_probe_age(status: str) -> None:
    with pytest.raises(ValidationError, match="requires probe_age_seconds"):
        observation(status=status, age=None)


@pytest.mark.parametrize("status", ["disabled", "unconfigured"])
def test_non_probed_configuration_statuses_reject_probe_age(status: str) -> None:
    with pytest.raises(ValidationError, match="cannot claim a probe age"):
        observation(status=status, age=0)


def test_unavailable_and_unknown_are_preserved_as_distinct_states() -> None:
    unavailable = observation(status="unavailable", age=5)
    unknown = observation(status="unknown", age=None)

    assert unavailable.status == "unavailable"
    assert unavailable.probe_age_seconds == 5
    assert unknown.status == "unknown"
    assert unknown.probe_age_seconds is None


def test_readiness_is_evaluated_per_capability_and_age_limit() -> None:
    exact_reads = observation("exact_reads", "healthy", 9)
    graph_reads = observation("graph_reads", "unavailable", 1)

    assert capability_is_ready(exact_reads, max_probe_age_seconds=9)
    assert not capability_is_ready(exact_reads, max_probe_age_seconds=8)
    assert not capability_is_ready(graph_reads, max_probe_age_seconds=9)


@pytest.mark.parametrize(
    ("status", "age"),
    [
        ("degraded", 0),
        ("unavailable", 0),
        ("unknown", None),
        ("disabled", None),
        ("unconfigured", None),
    ],
)
def test_nonhealthy_status_never_becomes_ready_from_recency(
    status: str, age: int | None
) -> None:
    assert not capability_is_ready(
        observation(status=status, age=age), max_probe_age_seconds=100
    )


def test_publication_keeps_partial_outage_visible() -> None:
    report = publication()
    by_capability = {item.capability: item for item in report.capabilities}

    assert capability_is_ready(
        by_capability["exact_reads"], max_probe_age_seconds=10
    )
    assert not capability_is_ready(
        by_capability["graph_reads"], max_probe_age_seconds=10
    )
    assert not capability_is_ready(
        by_capability["worker_processing"], max_probe_age_seconds=10
    )


def test_publication_rejects_duplicate_capability_observations() -> None:
    with pytest.raises(ValidationError, match="must be unique"):
        publication(
            capabilities=[
                observation("exact_reads", "healthy", 1).model_dump(mode="json"),
                observation("exact_reads", "unavailable", 2).model_dump(mode="json"),
            ]
        )


def test_publication_requires_at_least_one_capability() -> None:
    with pytest.raises(ValidationError, match="at least one"):
        publication(capabilities=[])


def test_publication_rejects_stale_and_future_generations() -> None:
    report = publication(placement_generation=12)

    for current_generation in (11, 13):
        with pytest.raises(ValueError, match="stale or unexpected"):
            validate_readiness_publication(
                report,
                expected_tenant_id="tenant-a",
                current_placement_generation=current_generation,
            )


def test_publication_accepts_only_exact_tenant_and_generation() -> None:
    report = publication()

    assert (
        validate_readiness_publication(
            report,
            expected_tenant_id="tenant-a",
            current_placement_generation=12,
        )
        is report
    )
    with pytest.raises(ValueError, match="tenant_id does not match"):
        validate_readiness_publication(
            report,
            expected_tenant_id="tenant-b",
            current_placement_generation=12,
        )


def test_generic_process_health_cannot_stand_in_for_capability_evidence() -> None:
    with pytest.raises(ValidationError):
        TenantReadinessPublication.model_validate_json(
            json.dumps(
                {
                    "schema_version": VERSION,
                    "tenant_id": "tenant-a",
                    "placement_generation": 12,
                    "capabilities": [],
                    "process_healthy": True,
                }
            )
        )


@pytest.mark.parametrize("invalid_age", [True, -1, 9_007_199_254_740_992])
def test_probe_age_uses_safe_counter_bounds(invalid_age: object) -> None:
    with pytest.raises(ValidationError):
        observation(age=invalid_age)


def test_version_and_unknown_status_are_rejected() -> None:
    with pytest.raises(ValidationError):
        CapabilityReadiness.model_validate_json(
            json.dumps(
                {
                    "capability": "exact_reads",
                    "status": "healthy",
                    "probe_age_seconds": 1,
                }
            )
        )
    with pytest.raises(ValidationError):
        observation(status="process_running", age=1)
