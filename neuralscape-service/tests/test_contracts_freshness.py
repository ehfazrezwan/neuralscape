"""Adversarial tests for independent upstream and projection freshness."""

import json

import pytest
from pydantic import ValidationError

from contracts_freshness import (
    FreshnessWitness,
    ProjectionFreshness,
    ProjectionStatus,
    SourceCheckpoint,
    UpstreamVerificationStatus,
    projection_status_for,
)
from contracts_references import SourceVersion


NOW = "2026-09-28T12:00:00Z"
EARLIER = "2026-09-28T11:00:00Z"


def _reference(identifier: str = "source-a") -> dict[str, object]:
    return {
        "kind": "source",
        "id": identifier,
        "tenant_id": "tenant-1",
        "resolver": "resolve_source",
    }


def _version(record_id: str, revision: int, policy_epoch: int = 2) -> dict[str, object]:
    return {
        "record_id": record_id,
        "content_revision": revision,
        "policy_epoch": policy_epoch,
    }


def _source_version(
    record_id: str, revision: int, policy_epoch: int = 2
) -> SourceVersion:
    return SourceVersion.model_validate_json(
        json.dumps(_version(record_id, revision, policy_epoch))
    )


def _checkpoint(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": "candidate-v1",
        "source": _reference(),
        "verification_status": "current",
        "verification_method": "provider_revision",
        "observed_at": NOW,
        "last_successful_verification_at": EARLIER,
        "provider_revision": "commit-abc",
        "opaque_cursor": None,
        "processed_through": "commit-abc",
        "reconciliation_state": "complete",
        "freshness_policy": "verify-before-consequential-use",
    }
    value.update(overrides)
    return value


def test_revision_change_is_stale_despite_unchanged_higher_revision() -> None:
    applied = [_source_version("source-a", 3), _source_version("source-b", 9)]
    required = [_source_version("source-a", 4), _source_version("source-b", 9)]

    assert max(version.content_revision for version in applied) == 9
    assert max(version.content_revision for version in required) == 9
    assert projection_status_for(applied, required) is ProjectionStatus.STALE
    assert projection_status_for(required, required) is ProjectionStatus.CURRENT


def test_policy_epoch_change_also_invalidates_projection() -> None:
    applied = [_source_version("source-a", 4, policy_epoch=2)]
    required = [_source_version("source-a", 4, policy_epoch=3)]

    assert projection_status_for(applied, required) is ProjectionStatus.STALE


def test_projection_comparison_revalidates_copied_source_versions() -> None:
    valid = _source_version("source-a", 4)
    copied = valid.model_copy(update={"content_revision": -1})
    constructed = SourceVersion.model_construct(
        record_id="source-a",
        content_revision=-1,
        policy_epoch=2,
    )

    with pytest.raises(ValidationError):
        projection_status_for([copied], [copied])
    with pytest.raises(ValidationError):
        projection_status_for([constructed], [constructed])


def test_projection_comparison_rejects_copied_extras_and_cycles() -> None:
    valid = _source_version("source-a", 4)
    copied_with_extra = valid.model_copy(update={"unreviewed_revision": 4})
    cycle: list[object] = []
    cycle.append(cycle)
    copied_with_cycle = valid.model_copy(update={"unreviewed_cycle": cycle})

    with pytest.raises(ValidationError):
        projection_status_for([copied_with_extra], [valid])
    with pytest.raises(ValueError, match="cyclic contract input"):
        projection_status_for([copied_with_cycle], [valid])


def test_projection_comparison_preserves_container_types_for_strict_validation() -> None:
    valid = _source_version("source-a", 4)
    malformed_mapping = {
        "record_id": "source-a",
        "content_revision": 4,
        "policy_epoch": 2,
        7: "unknown-key",
    }

    with pytest.raises(ValidationError):
        projection_status_for([malformed_mapping], [valid])  # type: ignore[list-item]


def test_stale_upstream_can_coexist_with_healthy_local_projection() -> None:
    payload = {
        "upstream_status": "stale",
        "source_checkpoints": [
            _checkpoint(
                verification_status="stale",
                last_successful_verification_at=EARLIER,
            )
        ],
        "projection": {
            "status": "current",
            "applied_sources": [_version("source-a", 4), _version("source-b", 9)],
            "verified_at": NOW,
        },
    }

    witness = FreshnessWitness.model_validate_json(json.dumps(payload))

    assert witness.upstream_status is UpstreamVerificationStatus.STALE
    assert witness.projection.status is ProjectionStatus.CURRENT


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"provider_revision": None}, "requires its corresponding marker"),
        (
            {
                "verification_method": "observation_only",
                "provider_revision": "must-not-be-present",
            },
            "cannot claim a provider revision",
        ),
        (
            {"reconciliation_state": "known_gap"},
            "current verification cannot contain a known source gap",
        ),
        (
            {"last_successful_verification_at": "2026-09-28T13:00:00Z"},
            "cannot be after observed_at",
        ),
    ],
)
def test_checkpoint_rejects_false_freshness_claims(
    changes: dict[str, object], message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        SourceCheckpoint.model_validate_json(json.dumps(_checkpoint(**changes)))


def test_projection_rejects_duplicate_source_versions() -> None:
    payload = {
        "status": "current",
        "applied_sources": [_version("source-a", 3), _version("source-a", 4)],
        "verified_at": NOW,
    }

    with pytest.raises(ValidationError, match="duplicate source version"):
        ProjectionFreshness.model_validate_json(json.dumps(payload))


def test_checkpoint_rejects_non_source_reference_kind() -> None:
    with pytest.raises(ValidationError, match="requires a source reference"):
        SourceCheckpoint.model_validate_json(
            json.dumps(_checkpoint(source=_reference() | {"kind": "memory"}))
        )


def test_current_aggregate_cannot_hide_stale_checkpoint() -> None:
    payload = {
        "upstream_status": "current",
        "source_checkpoints": [_checkpoint(verification_status="stale")],
        "projection": {
            "status": "current",
            "applied_sources": [_version("source-a", 4)],
            "verified_at": NOW,
        },
    }

    with pytest.raises(ValidationError, match="requires current checkpoints"):
        FreshnessWitness.model_validate_json(json.dumps(payload))


def test_current_upstream_rejects_checkpoint_for_unrelated_source() -> None:
    payload = {
        "upstream_status": "current",
        "source_checkpoints": [
            _checkpoint(source=_reference("source-x")),
        ],
        "projection": {
            "status": "current",
            "applied_sources": [_version("source-a", 4)],
            "verified_at": NOW,
        },
    }

    with pytest.raises(ValidationError, match="map to applied source record IDs"):
        FreshnessWitness.model_validate_json(json.dumps(payload))


def test_current_upstream_requires_checkpoint_for_every_applied_source() -> None:
    payload = {
        "upstream_status": "current",
        "source_checkpoints": [_checkpoint()],
        "projection": {
            "status": "current",
            "applied_sources": [
                _version("source-a", 4),
                _version("source-b", 9),
            ],
            "verified_at": NOW,
        },
    }

    with pytest.raises(ValidationError, match="every applied source checkpoint"):
        FreshnessWitness.model_validate_json(json.dumps(payload))


def test_freshness_witness_rejects_mixed_checkpoint_tenant_scope() -> None:
    other_tenant_source = _reference("source-b")
    other_tenant_source["tenant_id"] = "tenant-2"
    payload = {
        "upstream_status": "stale",
        "source_checkpoints": [
            _checkpoint(),
            _checkpoint(
                source=other_tenant_source,
                verification_status="stale",
            ),
        ],
        "projection": {
            "status": "current",
            "applied_sources": [
                _version("source-a", 4),
                _version("source-b", 9),
            ],
            "verified_at": NOW,
        },
    }

    with pytest.raises(ValidationError, match="one tenant scope"):
        FreshnessWitness.model_validate_json(json.dumps(payload))


def test_checkpoint_datetimes_must_be_timezone_aware() -> None:
    payload = _checkpoint()
    checkpoint = SourceCheckpoint.model_validate_json(json.dumps(payload))
    assert checkpoint.observed_at.utcoffset() is not None

    payload["observed_at"] = "2026-09-28T12:00:00"
    with pytest.raises(ValidationError):
        SourceCheckpoint.model_validate_json(json.dumps(payload))
