"""Contract tests for capability-specific readiness evidence."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from contracts_readiness import (
    CapabilityReadiness,
    TenantReadinessPublication,
    capability_is_ready,
    validate_readiness_publication,
)


VERSION = "candidate-v1"
OBSERVED_AT = datetime(2026, 9, 28, 0, 0, tzinfo=timezone.utc)


def observation(
    capability: str = "exact_reads",
    status: str = "healthy",
    observed_at: str = "2026-09-28T00:00:00Z",
    **updates: object,
) -> CapabilityReadiness:
    document: dict[str, object] = {
        "schema_version": VERSION,
        "capability": capability,
        "status": status,
        "observed_at": observed_at,
    }
    document.update(updates)
    return CapabilityReadiness.model_validate_json(json.dumps(document))


def publication(**updates: object) -> TenantReadinessPublication:
    document: dict[str, object] = {
        "schema_version": VERSION,
        "tenant_id": "tenant-a",
        "placement_generation": 12,
        "capabilities": [
            observation("api_acceptance", "healthy").model_dump(mode="json"),
            observation("exact_reads", "healthy").model_dump(mode="json"),
            observation(
                "graph_reads", "unavailable", detail_code="store-timeout"
            ).model_dump(mode="json"),
            observation("worker_processing", "unknown").model_dump(mode="json"),
        ],
    }
    document.update(updates)
    return TenantReadinessPublication.model_validate_json(json.dumps(document))


def evaluate(
    item: CapabilityReadiness,
    *,
    elapsed_seconds: int = 3,
    max_age_seconds: object = 10,
) -> bool:
    return capability_is_ready(
        item,
        evaluated_at=OBSERVED_AT + timedelta(seconds=elapsed_seconds),
        max_probe_age_seconds=max_age_seconds,
    )


def test_observation_requires_an_aware_timestamp() -> None:
    assert observation().observed_at == OBSERVED_AT

    with pytest.raises(ValidationError):
        observation(observed_at="2026-09-28T00:00:00")
    with pytest.raises(ValidationError):
        CapabilityReadiness.model_validate(
            {
                "schema_version": VERSION,
                "capability": "exact_reads",
                "status": "healthy",
                "observed_at": datetime(2026, 9, 28, 0, 0),
            }
        )


def test_unavailable_and_unknown_are_preserved_as_distinct_states() -> None:
    unavailable = observation(status="unavailable")
    unknown = observation(status="unknown")

    assert unavailable.status == "unavailable"
    assert unknown.status == "unknown"
    assert not evaluate(unavailable)
    assert not evaluate(unknown)


def test_readiness_is_evaluated_per_capability_and_explicit_time() -> None:
    exact_reads = observation("exact_reads", "healthy")
    graph_reads = observation("graph_reads", "unavailable")

    assert evaluate(exact_reads, elapsed_seconds=9, max_age_seconds=9)
    assert not evaluate(exact_reads, elapsed_seconds=10, max_age_seconds=9)
    assert not evaluate(graph_reads, elapsed_seconds=1, max_age_seconds=9)


def test_replayed_observation_ages_out_without_a_generation_change() -> None:
    exact_reads = observation("exact_reads", "healthy")

    assert evaluate(exact_reads, elapsed_seconds=1, max_age_seconds=5)
    assert not evaluate(exact_reads, elapsed_seconds=6, max_age_seconds=5)


@pytest.mark.parametrize(
    "status", ["degraded", "unavailable", "unknown", "disabled", "unconfigured"]
)
def test_nonhealthy_status_never_becomes_ready_from_recency(status: str) -> None:
    assert not evaluate(observation(status=status), elapsed_seconds=0)


def test_evaluation_time_cannot_precede_observation() -> None:
    with pytest.raises(ValueError, match="cannot precede"):
        evaluate(observation(), elapsed_seconds=-1)


def test_evaluation_time_must_be_explicit_and_aware() -> None:
    item = observation()

    with pytest.raises(ValidationError):
        capability_is_ready(
            item,
            evaluated_at=datetime(2026, 9, 28, 0, 0),
            max_probe_age_seconds=10,
        )


@pytest.mark.parametrize(
    "invalid_max_age",
    [True, 1.0, -1, 9_007_199_254_740_992],
)
def test_readiness_helper_strictly_validates_max_age(
    invalid_max_age: object,
) -> None:
    with pytest.raises(ValidationError):
        evaluate(observation(), max_age_seconds=invalid_max_age)


def test_publication_keeps_partial_outage_visible() -> None:
    report = publication()
    by_capability = {item.capability: item for item in report.capabilities}

    assert evaluate(by_capability["exact_reads"])
    assert not evaluate(by_capability["graph_reads"])
    assert not evaluate(by_capability["worker_processing"])


def test_publication_rejects_duplicate_capability_observations() -> None:
    with pytest.raises(ValidationError, match="must be unique"):
        publication(
            capabilities=[
                observation("exact_reads", "healthy").model_dump(mode="json"),
                observation("exact_reads", "unavailable").model_dump(mode="json"),
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

    result = validate_readiness_publication(
        report,
        expected_tenant_id="tenant-a",
        current_placement_generation=12,
    )
    assert result == report
    assert result is not report
    with pytest.raises(ValueError, match="tenant_id does not match"):
        validate_readiness_publication(
            report,
            expected_tenant_id="tenant-b",
            current_placement_generation=12,
        )


def test_publication_revalidates_copied_empty_capability_set() -> None:
    corrupted = publication().model_copy(update={"capabilities": ()})

    with pytest.raises(ValidationError, match="at least one"):
        validate_readiness_publication(
            corrupted,
            expected_tenant_id="tenant-a",
            current_placement_generation=12,
        )


def test_publication_revalidates_copied_duplicate_capabilities() -> None:
    exact_reads = observation("exact_reads")
    corrupted = publication().model_copy(
        update={"capabilities": (exact_reads, exact_reads.model_copy())}
    )

    with pytest.raises(ValidationError, match="must be unique"):
        validate_readiness_publication(
            corrupted,
            expected_tenant_id="tenant-a",
            current_placement_generation=12,
        )


def test_publication_revalidates_nested_capability_graph() -> None:
    invalid = observation("exact_reads").model_copy(
        update={"status": "process_running"}
    )
    corrupted = publication().model_copy(update={"capabilities": (invalid,)})

    with pytest.raises(ValidationError):
        validate_readiness_publication(
            corrupted,
            expected_tenant_id="tenant-a",
            current_placement_generation=12,
        )


def test_readiness_evaluation_revalidates_copied_observation_graph() -> None:
    invalid_status = observation(status="unavailable").model_copy(
        update={"status": "process_running"}
    )
    naive_timestamp = observation().model_copy(
        update={"observed_at": datetime(2026, 9, 28, 0, 0)}
    )

    with pytest.raises(ValidationError):
        evaluate(invalid_status)
    with pytest.raises(ValidationError):
        evaluate(naive_timestamp)


def test_valid_explicit_healthy_copy_has_same_semantics_as_fresh_input() -> None:
    copied = observation(status="unavailable").model_copy(
        update={"status": "healthy"}
    )
    fresh = observation(status="healthy")

    assert copied.model_dump() == fresh.model_dump()
    assert evaluate(copied) == evaluate(fresh) is True


def test_readiness_evaluation_revalidates_post_construction_mutation() -> None:
    item = observation()
    item.observed_at = datetime(2026, 9, 28, 0, 0)

    with pytest.raises(ValidationError):
        evaluate(item)


@pytest.mark.parametrize(
    "invalid_generation",
    [True, 12.0, -1, 9_007_199_254_740_992],
)
def test_publication_helper_strictly_validates_generation(
    invalid_generation: object,
) -> None:
    with pytest.raises(ValidationError):
        validate_readiness_publication(
            publication(),
            expected_tenant_id="tenant-a",
            current_placement_generation=invalid_generation,
        )


@pytest.mark.parametrize("invalid_tenant_id", [1, b"tenant-a", ""])
def test_publication_helper_strictly_validates_tenant_id(
    invalid_tenant_id: object,
) -> None:
    with pytest.raises(ValidationError):
        validate_readiness_publication(
            publication(),
            expected_tenant_id=invalid_tenant_id,
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


def test_version_and_unknown_status_are_rejected() -> None:
    with pytest.raises(ValidationError):
        CapabilityReadiness.model_validate_json(
            json.dumps(
                {
                    "capability": "exact_reads",
                    "status": "healthy",
                    "observed_at": "2026-09-28T00:00:00Z",
                }
            )
        )
    with pytest.raises(ValidationError):
        observation(status="process_running")
