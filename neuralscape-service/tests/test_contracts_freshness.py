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
    VerificationMethod,
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


def test_projection_comparison_rejects_conflicting_extra_storage() -> None:
    valid = _source_version("source-a", 4)
    corrupted = valid.model_copy(update={"content_revision": -1})
    object.__setattr__(
        corrupted,
        "__pydantic_extra__",
        {"content_revision": valid.content_revision},
    )

    with pytest.raises(ValueError, match="conflicting declared and extra fields"):
        projection_status_for([corrupted], [valid])


def test_projection_comparison_rejects_malformed_extra_storage() -> None:
    valid = _source_version("source-a", 4)
    corrupted = valid.model_copy()
    object.__setattr__(corrupted, "__pydantic_extra__", [])

    with pytest.raises(ValueError, match="extra storage must be a mapping"):
        projection_status_for([corrupted], [valid])


def test_projection_comparison_preserves_nonconflicting_extra_storage() -> None:
    valid = _source_version("source-a", 4)
    corrupted = valid.model_copy()
    object.__setattr__(
        corrupted,
        "__pydantic_extra__",
        {"unreviewed_revision": 4},
    )

    with pytest.raises(ValidationError):
        projection_status_for([corrupted], [valid])


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
            ),
            _checkpoint(source=_reference("source-b")),
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


@pytest.mark.parametrize("verification_status", ["current", "stale"])
def test_known_checkpoint_requires_successful_time_from_json(
    verification_status: str,
) -> None:
    with pytest.raises(
        ValidationError,
        match=(
            f"{verification_status} verification requires a successful check time"
        ),
    ):
        SourceCheckpoint.model_validate_json(
            json.dumps(
                _checkpoint(
                    verification_status=verification_status,
                    last_successful_verification_at=None,
                )
            )
        )


def test_stale_checkpoint_requires_successful_time_from_native_mapping() -> None:
    checkpoint = SourceCheckpoint.model_validate_json(
        json.dumps(_checkpoint(verification_status="stale"))
    )
    payload = checkpoint.model_dump(mode="python")
    payload["last_successful_verification_at"] = None

    with pytest.raises(
        ValidationError,
        match="stale verification requires a successful check time",
    ):
        SourceCheckpoint.model_validate(payload)


def test_stale_checkpoint_copy_is_revalidated_at_witness_boundary() -> None:
    payload = {
        "upstream_status": "stale",
        "source_checkpoints": [
            _checkpoint(verification_status="stale"),
        ],
        "projection": {
            "status": "current",
            "applied_sources": [_version("source-a", 4)],
            "verified_at": NOW,
        },
    }
    witness = FreshnessWitness.model_validate_json(json.dumps(payload))
    invalid_checkpoint = witness.source_checkpoints[0].model_copy(
        update={"last_successful_verification_at": None}
    )
    invalid_witness = witness.model_copy(
        update={"source_checkpoints": (invalid_checkpoint,)}
    )

    with pytest.raises(
        ValidationError,
        match="stale verification requires a successful check time",
    ):
        FreshnessWitness.model_validate(invalid_witness)


@pytest.mark.parametrize("verification_status", ["unverified", "unavailable"])
def test_weaker_checkpoint_statuses_do_not_require_successful_time(
    verification_status: str,
) -> None:
    checkpoint = SourceCheckpoint.model_validate_json(
        json.dumps(
            _checkpoint(
                verification_status=verification_status,
                last_successful_verification_at=None,
            )
        )
    )

    assert checkpoint.verification_status.value == verification_status
    assert checkpoint.last_successful_verification_at is None


@pytest.mark.parametrize(
    "verification_method",
    ["content_digest", "conditional_read"],
)
def test_nonrevision_methods_remain_representable_as_unverified(
    verification_method: str,
) -> None:
    checkpoint = SourceCheckpoint.model_validate_json(
        json.dumps(
            _checkpoint(
                verification_status="unverified",
                verification_method=verification_method,
                last_successful_verification_at=None,
                provider_revision=None,
            )
        )
    )

    assert checkpoint.verification_method.value == verification_method
    assert checkpoint.verification_status is UpstreamVerificationStatus.UNVERIFIED


@pytest.mark.parametrize(
    "verification_method",
    [
        "provider_revision",
        "opaque_cursor",
        "observation_only",
        "content_digest",
        "conditional_read",
    ],
)
@pytest.mark.parametrize(
    "verification_status",
    ["current", "stale", "unverified", "unavailable"],
)
def test_checkpoint_method_status_support_matrix(
    verification_method: str,
    verification_status: str,
) -> None:
    markers: dict[str, object] = {
        "provider_revision": None,
        "opaque_cursor": None,
    }
    if verification_method == "provider_revision":
        markers["provider_revision"] = "commit-abc"
    elif verification_method == "opaque_cursor":
        markers["opaque_cursor"] = "cursor-abc"

    payload = _checkpoint(
        verification_method=verification_method,
        verification_status=verification_status,
        last_successful_verification_at=(
            EARLIER
            if verification_status in {"current", "stale"}
            else None
        ),
        **markers,
    )
    known_support_is_deferred = (
        verification_method in {"content_digest", "conditional_read"}
        and verification_status in {"current", "stale"}
    )

    if known_support_is_deferred:
        with pytest.raises(
            ValidationError,
            match="cannot claim current or stale until its evidence protocol",
        ):
            SourceCheckpoint.model_validate_json(json.dumps(payload))
        return

    checkpoint = SourceCheckpoint.model_validate_json(json.dumps(payload))
    assert checkpoint.verification_method.value == verification_method
    assert checkpoint.verification_status.value == verification_status


@pytest.mark.parametrize(
    "verification_method",
    [VerificationMethod.CONTENT_DIGEST, VerificationMethod.CONDITIONAL_READ],
)
def test_witness_boundary_rejects_copied_unsupported_known_checkpoint(
    verification_method: VerificationMethod,
) -> None:
    witness = FreshnessWitness.model_validate_json(
        json.dumps(
            {
                "upstream_status": "current",
                "source_checkpoints": [_checkpoint()],
                "projection": {
                    "status": "current",
                    "applied_sources": [_version("source-a", 4)],
                    "verified_at": NOW,
                },
            }
        )
    )
    invalid_checkpoint = witness.source_checkpoints[0].model_copy(
        update={"verification_method": verification_method}
    )
    invalid_witness = witness.model_copy(
        update={"source_checkpoints": (invalid_checkpoint,)}
    )

    with pytest.raises(
        ValidationError,
        match="cannot claim current or stale until its evidence protocol",
    ):
        FreshnessWitness.model_validate(invalid_witness)


@pytest.mark.parametrize(
    ("checkpoint_status", "message"),
    [
        ("unverified", "only current or stale checkpoints"),
        ("unavailable", "only current or stale checkpoints"),
    ],
)
def test_stale_aggregate_cannot_hide_weaker_checkpoint_evidence(
    checkpoint_status: str,
    message: str,
) -> None:
    payload = {
        "upstream_status": "stale",
        "source_checkpoints": [
            _checkpoint(verification_status=checkpoint_status),
        ],
        "projection": {
            "status": "current",
            "applied_sources": [_version("source-a", 4)],
            "verified_at": NOW,
        },
    }

    with pytest.raises(ValidationError, match=message):
        FreshnessWitness.model_validate_json(json.dumps(payload))


def test_stale_aggregate_requires_complete_applied_source_coverage() -> None:
    payload = {
        "upstream_status": "stale",
        "source_checkpoints": [
            _checkpoint(verification_status="stale"),
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

    with pytest.raises(ValidationError, match="every applied source checkpoint"):
        FreshnessWitness.model_validate_json(json.dumps(payload))


@pytest.mark.parametrize(
    ("upstream_status", "checkpoint_status"),
    [
        ("unverified", "unverified"),
        ("unavailable", "unavailable"),
    ],
)
def test_weak_aggregates_accept_matching_complete_evidence(
    upstream_status: str,
    checkpoint_status: str,
) -> None:
    payload = {
        "upstream_status": upstream_status,
        "source_checkpoints": [
            _checkpoint(verification_status=checkpoint_status),
        ],
        "projection": {
            "status": "current",
            "applied_sources": [_version("source-a", 4)],
            "verified_at": NOW,
        },
    }

    witness = FreshnessWitness.model_validate_json(json.dumps(payload))
    assert witness.upstream_status.value == upstream_status


def test_weak_aggregates_accept_documented_partial_coverage() -> None:
    for upstream_status in ("unverified", "unavailable"):
        payload = {
            "upstream_status": upstream_status,
            "source_checkpoints": [],
            "projection": {
                "status": "current",
                "applied_sources": [_version("source-a", 4)],
                "verified_at": NOW,
            },
        }

        witness = FreshnessWitness.model_validate_json(json.dumps(payload))
        assert witness.upstream_status.value == upstream_status


@pytest.mark.parametrize(
    ("upstream_status", "checkpoint_status", "message"),
    [
        (
            "stale",
            "current",
            "requires at least one stale checkpoint",
        ),
        (
            "unverified",
            "current",
            "requires unverified evidence",
        ),
        (
            "unavailable",
            "current",
            "requires unavailable evidence",
        ),
        (
            "unverified",
            "unavailable",
            "cannot hide unavailable checkpoints",
        ),
    ],
)
def test_aggregate_rejects_contradictory_complete_evidence(
    upstream_status: str,
    checkpoint_status: str,
    message: str,
) -> None:
    payload = {
        "upstream_status": upstream_status,
        "source_checkpoints": [
            _checkpoint(verification_status=checkpoint_status),
        ],
        "projection": {
            "status": "current",
            "applied_sources": [_version("source-a", 4)],
            "verified_at": NOW,
        },
    }

    with pytest.raises(ValidationError, match=message):
        FreshnessWitness.model_validate_json(json.dumps(payload))


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
        (
            {
                "verification_status": "stale",
                "last_successful_verification_at": "2026-09-28T13:00:00Z",
            },
            "cannot be after observed_at",
        ),
        (
            {
                "verification_status": "unverified",
                "last_successful_verification_at": "2026-09-28T13:00:00Z",
            },
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


# ── the verification time is ordered by instant, not by wall clock ───────────

from datetime import datetime as _datetime  # noqa: E402
from zoneinfo import ZoneInfo as _ZoneInfo  # noqa: E402

_NEW_YORK = _ZoneInfo("America/New_York")
# 2026-11-01 01:30 happens twice: first at -04:00 (fold=0, 05:30 UTC), then at -05:00 (fold=1, 06:30 UTC).


def _timestamps(**overrides: object) -> dict[str, object]:
    return _checkpoint(**overrides)


def test_a_successful_check_after_the_observation_is_refused_across_a_clock_rollback() -> None:
    # Same zone object: Python compares wall times and ignores fold, so 01:30 looks earlier than 01:45.
    observed = _datetime(2026, 11, 1, 1, 45, tzinfo=_NEW_YORK)  # 05:45 UTC
    later_check = _datetime(2026, 11, 1, 1, 30, fold=1, tzinfo=_NEW_YORK)  # 06:30 UTC, after the observation
    assert later_check < observed  # the trap: wall-clock comparison says the opposite of the instants
    with pytest.raises(ValidationError, match="cannot be after observed_at"):
        SourceCheckpoint.model_validate(
            SourceCheckpoint.model_validate_json(json.dumps(_checkpoint())).model_dump(mode="python")
            | {"observed_at": observed, "last_successful_verification_at": later_check}
        )


def test_a_check_before_the_observation_is_accepted_across_a_clock_rollback() -> None:
    observed = _datetime(2026, 11, 1, 1, 15, fold=1, tzinfo=_NEW_YORK)  # 06:15 UTC
    earlier_check = _datetime(2026, 11, 1, 1, 45, tzinfo=_NEW_YORK)  # 05:45 UTC, 30 minutes before
    assert earlier_check > observed  # wall-clock comparison would reject a valid pair
    checkpoint = SourceCheckpoint.model_validate(
        SourceCheckpoint.model_validate_json(json.dumps(_checkpoint())).model_dump(mode="python")
        | {"observed_at": observed, "last_successful_verification_at": earlier_check}
    )
    assert checkpoint.last_successful_verification_at is not None


def test_a_check_equal_to_the_observation_in_instants_is_accepted_across_zones() -> None:
    utc = _datetime(2026, 11, 1, 6, 30, tzinfo=__import__("datetime").timezone.utc)
    local = _datetime(2026, 11, 1, 1, 30, fold=1, tzinfo=_NEW_YORK)  # the same instant
    base = SourceCheckpoint.model_validate_json(json.dumps(_checkpoint())).model_dump(mode="python")
    assert SourceCheckpoint.model_validate(base | {"observed_at": utc, "last_successful_verification_at": local})
    assert SourceCheckpoint.model_validate(base | {"observed_at": local, "last_successful_verification_at": utc})


# ── both dependency sequences are captured together before any map is built ──

from datetime import timezone as _timezone  # noqa: E402

from contracts_freshness import projection_status_for  # noqa: E402


def _sv(record: str, revision: int) -> SourceVersion:
    return SourceVersion(record_id=record, content_revision=revision, policy_epoch=1)


def test_one_object_shared_by_both_arguments_is_read_at_one_instant() -> None:
    shared = _sv("rec-1", 1)

    def required():
        object.__setattr__(shared, "content_revision", 2)  # caller code runs while the second argument is read
        yield shared

    class Lazy:
        def __iter__(self):
            return required()

        def __len__(self):
            return 1

    # at one instant both arguments name the same object, so they cannot differ
    assert projection_status_for([shared], Lazy()).value == "current"


def _invalid(record: str = "rec-2") -> SourceVersion:
    return _sv(record, 1).model_copy(update={"content_revision": -1})  # an unchecked copy: invalid on its face


def test_an_entry_exposed_by_an_iterator_is_captured_before_the_iterator_advances() -> None:
    """First observation wins: an invalid entry repaired after it was exposed is still invalid."""

    bad = _invalid()

    class Repairing:
        def __iter__(self):
            yield _sv("rec-1", 1)
            yield bad
            object.__setattr__(bad, "content_revision", 1)  # repaired once the iterator resumes

    with pytest.raises(Exception):
        projection_status_for(Repairing(), [_sv("rec-1", 1), _sv("rec-2", 1)])


def test_an_exposed_entry_edited_after_exposure_is_judged_as_it_was_first_seen() -> None:
    first, second = _sv("rec-1", 1), _sv("rec-2", 1)

    class Editing:
        def __iter__(self):
            yield first
            object.__setattr__(first, "content_revision", 5)  # edits an entry that was already yielded
            yield second

    assert projection_status_for(Editing(), [_sv("rec-1", 5), _sv("rec-2", 1)]).value == "stale"


def test_the_first_arguments_iterator_cannot_repair_an_invalid_model_held_in_the_second() -> None:
    for hold in (tuple, list):
        bad = _invalid()

        class RepairOnIter:
            def __iter__(self):
                object.__setattr__(bad, "content_revision", 1)  # runs before the first entry is exposed
                yield _sv("rec-1", 1)

        with pytest.raises(Exception):
            projection_status_for(RepairOnIter(), hold([bad]))


def test_a_model_shared_by_both_arguments_is_captured_once_at_its_first_exposure() -> None:
    shared = _invalid("rec-1")

    class RepairAfterExposure:
        def __iter__(self):
            yield shared
            object.__setattr__(shared, "content_revision", 1)

    # the second argument holds the model natively, so it is fenced (invalid) before anything runs
    with pytest.raises(Exception):
        projection_status_for(RepairAfterExposure(), (shared,))


def test_an_endless_sequence_is_cut_off_by_the_budget_while_it_is_read() -> None:
    produced: list[int] = []

    class Endless:
        def __iter__(self):
            for index in range(5_000_000):
                produced.append(index)
                yield _sv(f"rec-{index}", 1)

    with pytest.raises(ValueError, match="budget"):
        projection_status_for(Endless(), [_sv("rec-1", 1)])
    assert len(produced) <= 131_072 + 1


def test_an_exact_argument_is_unchanged_by_the_pre_capture() -> None:
    assert projection_status_for([_sv("a", 1), _sv("b", 2)], (_sv("b", 2), _sv("a", 1))).value == "current"
    assert projection_status_for([_sv("a", 1)], [_sv("a", 2)]).value == "stale"
    with pytest.raises(ValueError, match="duplicate"):
        projection_status_for([_sv("a", 1), _sv("a", 1)], [_sv("a", 1)])


def test_the_extreme_datetimes_are_a_validation_error_not_an_overflow() -> None:
    from datetime import datetime, timedelta

    class Fixed(__import__("datetime").tzinfo):
        def __init__(self, hours):
            self._o = timedelta(hours=hours)

        def utcoffset(self, value):
            return self._o

        def dst(self, value):
            return None

        def tzname(self, value):
            return "x"

    base = SourceCheckpoint.model_validate_json(json.dumps(_checkpoint())).model_dump(mode="python")
    for observed, checked in (
        (datetime(2026, 1, 1, tzinfo=_timezone.utc), datetime(9999, 12, 31, 23, 0, tzinfo=Fixed(-5))),
        (datetime(1, 1, 1, 1, 0, tzinfo=Fixed(3)), datetime(1, 1, 1, 0, 30, tzinfo=_timezone.utc)),
    ):
        with pytest.raises(ValidationError):
            SourceCheckpoint.model_validate(base | {"observed_at": observed, "last_successful_verification_at": checked})


def test_the_ordering_of_extreme_instants_is_decided_without_leaving_the_range() -> None:
    from datetime import datetime, timedelta, timezone

    base = SourceCheckpoint.model_validate_json(json.dumps(_checkpoint())).model_dump(mode="python")
    late = datetime(9999, 12, 31, 23, 0, tzinfo=timezone(-timedelta(hours=5)))  # 2000-01-01 04:00 UTC in year 10000
    early = datetime(9999, 12, 31, 20, 0, tzinfo=timezone(-timedelta(hours=5)))
    assert SourceCheckpoint.model_validate(base | {"observed_at": late, "last_successful_verification_at": early})
    with pytest.raises(ValidationError, match="cannot be after observed_at"):
        SourceCheckpoint.model_validate(base | {"observed_at": early, "last_successful_verification_at": late})


def test_a_non_dict_extra_mapping_cannot_repair_a_later_source_before_it_is_captured() -> None:
    """An earlier source's extras callback edits a later source: both are captured before it can run."""

    events: list[str] = []
    later = _sv("rec-2", 1).model_copy(update={"content_revision": -1})  # unchecked copy: invalid on its face

    class Repairing:
        def items(self):
            events.append("items")
            object.__setattr__(later, "content_revision", 1)
            return iter(())

        def __iter__(self):
            return iter(())

        def __len__(self):
            return 0

    earlier = _sv("rec-1", 1)
    object.__setattr__(earlier, "__pydantic_extra__", Repairing())
    with pytest.raises(Exception):
        projection_status_for([earlier, later], [_sv("rec-1", 1), _sv("rec-2", 1)])
    assert events == []  # the callback never ran, so the invalid copy was judged as it was captured

    # across the two arguments: the same shape, the repairing source in one argument, the invalid one in the other
    later2 = _sv("rec-2", 1).model_copy(update={"content_revision": -1})
    earlier2 = _sv("rec-1", 1)

    class Repairing2(Repairing):
        def items(self):
            events.append("items")
            object.__setattr__(later2, "content_revision", 1)
            return iter(())

    object.__setattr__(earlier2, "__pydantic_extra__", Repairing2())
    with pytest.raises(Exception):
        projection_status_for([earlier2], [later2])
    assert events == []


def test_an_invalid_entry_is_rejected_as_it_is_captured_and_later_entries_are_never_read() -> None:
    """Total work on invalid input is bounded by the first bad entry, not by the number of entries."""

    import time

    shared = tuple(range(100_000))
    produced: list[int] = []

    class Hostile:
        def __iter__(self):
            for index in range(400):
                produced.append(index)
                # every entry is invalid (record_id is a tuple) and holds the same large tuple
                yield SourceVersion.model_construct(record_id=shared, content_revision=1, policy_epoch=1)

    started = time.monotonic()
    with pytest.raises(Exception):
        projection_status_for(Hostile(), [_sv("rec-1", 1)])
    assert time.monotonic() - started < 3.0
    assert len(produced) <= 2, "entries after the first invalid one were read and snapshotted"


def test_a_valid_public_view_is_still_read_in_full_and_a_late_invalid_entry_is_still_refused() -> None:
    class View:
        def __init__(self, bad_at):
            self._bad_at = bad_at

        def __iter__(self):
            for index in range(50):
                yield _invalid(f"rec-{index}") if index == self._bad_at else _sv(f"rec-{index}", 1)

    assert projection_status_for(View(-1), [_sv(f"rec-{i}", 1) for i in range(50)]).value == "current"
    with pytest.raises(Exception):
        projection_status_for(View(37), [_sv(f"rec-{i}", 1) for i in range(50)])
    with pytest.raises(ValueError, match="duplicate"):
        projection_status_for(View(-1), [_sv("a", 1), _sv("a", 2)])
