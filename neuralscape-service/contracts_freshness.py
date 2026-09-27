"""Product contracts for upstream verification and local projection freshness.

An upstream checkpoint reports what was observed at a source authority.  It is
deliberately independent of projection freshness: a local index can be healthy
and internally current while its last successful upstream verification is old.
"""

from __future__ import annotations

from enum import Enum
from typing import Sequence

from pydantic import AwareDatetime, Field, model_validator

from contracts_common import ContractModel, OpaqueId, VersionedContract
from contracts_references import ReferenceHandle, SourceVersion


class UpstreamVerificationStatus(str, Enum):
    """Knowledge available about the source authority, not an index status."""

    CURRENT = "current"
    STALE = "stale"
    UNVERIFIED = "unverified"
    UNAVAILABLE = "unavailable"


class VerificationMethod(str, Enum):
    """Basis used to check an upstream source."""

    PROVIDER_REVISION = "provider_revision"
    OPAQUE_CURSOR = "opaque_cursor"
    CONDITIONAL_READ = "conditional_read"
    CONTENT_DIGEST = "content_digest"
    OBSERVATION_ONLY = "observation_only"


class ReconciliationState(str, Enum):
    """Whether the connector knows of unprocessed or uncertain source ranges."""

    COMPLETE = "complete"
    KNOWN_GAP = "known_gap"
    RECONCILING = "reconciling"
    UNKNOWN = "unknown"


class ProjectionStatus(str, Enum):
    """Relationship between a projection and its declared canonical inputs."""

    CURRENT = "current"
    STALE = "stale"
    UNAVAILABLE = "unavailable"
    UNKNOWN = "unknown"


class SourceCheckpoint(VersionedContract):
    """A versioned observation of one upstream source authority.

    Provider revisions and cursors are opaque within a source.  They must not be
    ordered or compared across sources.  ``observation_only`` is the explicit
    weaker basis for a source that exposes neither kind of version marker.
    """

    source: ReferenceHandle
    verification_status: UpstreamVerificationStatus
    verification_method: VerificationMethod
    observed_at: AwareDatetime
    last_successful_verification_at: AwareDatetime | None
    provider_revision: OpaqueId | None
    opaque_cursor: OpaqueId | None
    processed_through: OpaqueId | None
    reconciliation_state: ReconciliationState
    freshness_policy: OpaqueId

    @model_validator(mode="after")
    def validate_verification_basis(self) -> "SourceCheckpoint":
        if self.source.kind != "source":
            raise ValueError("a source checkpoint requires a source reference")

        if self.last_successful_verification_at is not None:
            if self.last_successful_verification_at > self.observed_at:
                raise ValueError(
                    "last_successful_verification_at cannot be after observed_at"
                )

        marker_by_method = {
            VerificationMethod.PROVIDER_REVISION: self.provider_revision,
            VerificationMethod.OPAQUE_CURSOR: self.opaque_cursor,
        }
        required_marker = marker_by_method.get(self.verification_method)
        if self.verification_method in marker_by_method and required_marker is None:
            raise ValueError(
                f"{self.verification_method.value} requires its corresponding marker"
            )

        if self.verification_method is VerificationMethod.OBSERVATION_ONLY:
            if self.provider_revision is not None or self.opaque_cursor is not None:
                raise ValueError(
                    "observation_only cannot claim a provider revision or cursor"
                )

        if self.verification_status is UpstreamVerificationStatus.CURRENT:
            if self.last_successful_verification_at is None:
                raise ValueError(
                    "current verification requires a successful check time"
                )
            if self.reconciliation_state is not ReconciliationState.COMPLETE:
                raise ValueError(
                    "current verification cannot contain a known source gap"
                )

        return self


class ProjectionFreshness(ContractModel):
    """Local projection state and the complete source-version set it consumed."""

    status: ProjectionStatus
    applied_sources: tuple[SourceVersion, ...] = Field(min_length=1)
    verified_at: AwareDatetime

    @model_validator(mode="after")
    def require_unique_sources(self) -> "ProjectionFreshness":
        _version_map(self.applied_sources)
        return self


class FreshnessWitness(ContractModel):
    """Independent upstream and local-projection freshness evidence."""

    upstream_status: UpstreamVerificationStatus
    source_checkpoints: tuple[SourceCheckpoint, ...]
    projection: ProjectionFreshness

    @model_validator(mode="after")
    def require_unique_checkpoints(self) -> "FreshnessWitness":
        source_keys = [
            (checkpoint.source.kind, checkpoint.source.id)
            for checkpoint in self.source_checkpoints
        ]
        if len(source_keys) != len(set(source_keys)):
            raise ValueError("source_checkpoints must identify unique sources")
        if self.upstream_status is UpstreamVerificationStatus.CURRENT:
            if not self.source_checkpoints:
                raise ValueError("current upstream status requires a checkpoint")
            if any(
                checkpoint.verification_status
                is not UpstreamVerificationStatus.CURRENT
                for checkpoint in self.source_checkpoints
            ):
                raise ValueError("current upstream status requires current checkpoints")
        return self


def projection_status_for(
    applied_sources: Sequence[SourceVersion],
    required_sources: Sequence[SourceVersion],
) -> ProjectionStatus:
    """Compare complete dependency sets without using a global max revision.

    A source's content revision and policy epoch are separate requirements.  A
    missing, extra, duplicated, or changed source therefore makes the applied
    projection stale.
    """

    applied = _version_map(applied_sources)
    required = _version_map(required_sources)
    return (
        ProjectionStatus.CURRENT
        if applied == required
        else ProjectionStatus.STALE
    )


def _version_map(versions: Sequence[SourceVersion]) -> dict[str, tuple[int, int]]:
    result: dict[str, tuple[int, int]] = {}
    for version in versions:
        record_id = str(version.record_id)
        if record_id in result:
            raise ValueError(f"duplicate source version for record_id {record_id!r}")
        result[record_id] = (version.content_revision, version.policy_epoch)
    return result


__all__ = [
    "FreshnessWitness",
    "ProjectionFreshness",
    "ProjectionStatus",
    "ReconciliationState",
    "SourceCheckpoint",
    "UpstreamVerificationStatus",
    "VerificationMethod",
    "projection_status_for",
]
