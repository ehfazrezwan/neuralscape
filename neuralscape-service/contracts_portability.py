"""Portable bundle manifest contracts and pure structural validation.

These models describe what an exporter claims is present.  Successful
validation does not prove that an export is complete, authorized, decryptable,
or safe to import, and it does not inspect archive entries or file contents.
Importers must separately verify every declared byte length and checksum,
enforce archive expansion/symlink limits, authorize identity mappings, and
perform migration transaction and replay checks.
"""

from __future__ import annotations

from pathlib import PurePosixPath, PureWindowsPath
from typing import Any, Literal, Mapping

from pydantic import Field, field_validator, model_validator

from contracts_common import ContractModel, OpaqueId, SafeCounter, VersionedContract


ChecksumAlgorithm = Literal["sha256"]
EncryptionMode = Literal["plaintext_authorized_export", "encrypted"]
ScopeKind = Literal["tenant", "projects"]


def _normalized_bundle_path(value: str) -> str:
    """Return a canonical relative POSIX bundle path or raise ``ValueError``.

    Backslashes are rejected rather than treated as ordinary POSIX filename
    characters because a later Windows consumer could reinterpret them as path
    separators.  ``.`` and repeated separators are normalized so aliases can
    be detected across the complete file inventory.
    """

    if not value or "\x00" in value:
        raise ValueError("bundle path must be nonempty and contain no NUL bytes")
    if "\\" in value:
        raise ValueError("bundle path must use POSIX separators")
    if PureWindowsPath(value).drive:
        raise ValueError("bundle path must not be Windows drive-qualified")

    path = PurePosixPath(value)
    if path.is_absolute() or value.startswith("/"):
        raise ValueError("bundle path must be relative")
    if any(part == ".." for part in path.parts):
        raise ValueError("bundle path must not traverse outside the bundle")

    normalized = path.as_posix()
    if normalized in {"", "."}:
        raise ValueError("bundle path must identify a file")
    return normalized


class CanonicalObjectId(ContractModel):
    """An exact canonical object identity carried by the bundle inventory."""

    kind: Literal[
        "record",
        "revision",
        "derivation",
        "source",
        "passage",
        "graph_node",
        "graph_edge",
        "episode",
        "artifact",
        "intent",
        "policy",
        "group",
        "tombstone",
        "hold",
        "projection",
    ]
    id: OpaqueId


class ExportScope(ContractModel):
    """The tenant boundary and optional project subset claimed by an export."""

    kind: ScopeKind
    tenant_id: OpaqueId
    project_ids: list[OpaqueId]

    @model_validator(mode="after")
    def _validate_scope(self) -> "ExportScope":
        if len(set(self.project_ids)) != len(self.project_ids):
            raise ValueError("scope contains duplicate project IDs")
        if self.kind == "tenant" and self.project_ids:
            raise ValueError("tenant scope must not include project IDs")
        if self.kind == "projects" and not self.project_ids:
            raise ValueError("projects scope requires at least one project ID")
        return self


class ProducerReference(ContractModel):
    """Identifies the software contract that produced the manifest."""

    implementation: OpaqueId
    version: OpaqueId


class PositionReference(ContractModel):
    """An opaque position in one explicitly identified ordered stream."""

    stream_id: OpaqueId
    position: OpaqueId


class SnapshotMetadata(ContractModel):
    """Complete snapshot, barrier, and replay coordinates for migration."""

    snapshot_id: OpaqueId
    snapshot_position: PositionReference
    completion_barrier: PositionReference
    replay_position: PositionReference


class ChecksumDescriptor(ContractModel):
    """A supported checksum algorithm and lowercase canonical digest."""

    algorithm: ChecksumAlgorithm
    digest: str = Field(pattern=r"^[a-f0-9]{64}$")


class ManifestFile(ContractModel):
    """A claimed regular file and the values an offline reader must verify."""

    path: str = Field(min_length=1, max_length=4096)
    size_bytes: SafeCounter
    checksum: ChecksumDescriptor
    entry_type: Literal["file"]

    @field_validator("path")
    @classmethod
    def _canonicalize_path(cls, value: str) -> str:
        return _normalized_bundle_path(value)


class DeclaredOmission(ContractModel):
    """A specific capability or state class the producer says is absent."""

    code: OpaqueId
    reason: str = Field(min_length=1, max_length=1024)


class IdentityMapping(ContractModel):
    """An explicit one-to-one source-to-destination principal mapping.

    ``authority_reference`` identifies the separate approval evidence an
    importer must resolve.  Its presence is not itself proof of authority.
    """

    source_identity_id: OpaqueId
    destination_identity_id: OpaqueId
    authority_reference: OpaqueId


class EncryptionReference(ContractModel):
    """References encryption semantics without embedding recovery material."""

    mode: EncryptionMode
    protocol_reference: OpaqueId | None
    key_envelope_paths: list[str]

    @field_validator("key_envelope_paths")
    @classmethod
    def _canonicalize_envelope_paths(cls, values: list[str]) -> list[str]:
        normalized = [_normalized_bundle_path(value) for value in values]
        if len(set(normalized)) != len(normalized):
            raise ValueError("encryption reference contains duplicate envelope paths")
        return normalized

    @model_validator(mode="after")
    def _validate_mode(self) -> "EncryptionReference":
        if self.mode == "encrypted":
            if self.protocol_reference is None:
                raise ValueError("encrypted mode requires a protocol reference")
            if not self.key_envelope_paths:
                raise ValueError("encrypted mode requires at least one key envelope path")
        if self.mode == "plaintext_authorized_export":
            if self.protocol_reference is not None or self.key_envelope_paths:
                raise ValueError(
                    "plaintext export must not declare encryption protocol or key envelopes"
                )
        return self


class PortableManifest(VersionedContract):
    """Strict portable-bundle inventory suitable for offline preflight.

    Lists are required even when empty so omitted declarations cannot be
    mistaken for known-empty state.  Optional projections, embeddings,
    history, and recovery policies have no defaults here; producers must state
    their included capabilities and omissions explicitly.
    """

    format_version: Literal["portable-candidate-v1"]
    manifest_id: OpaqueId
    producer: ProducerReference
    scope: ExportScope
    snapshot: SnapshotMetadata
    files: list[ManifestFile]
    canonical_ids: list[CanonicalObjectId]
    included_capabilities: list[OpaqueId]
    omissions: list[DeclaredOmission]
    identity_mappings: list[IdentityMapping]
    encryption: EncryptionReference

    @model_validator(mode="after")
    def _validate_inventory(self) -> "PortableManifest":
        file_paths = [item.path for item in self.files]
        if len(set(file_paths)) != len(file_paths):
            raise ValueError("manifest contains duplicate normalized file paths")

        object_keys = [(item.kind, item.id) for item in self.canonical_ids]
        if len(set(object_keys)) != len(object_keys):
            raise ValueError("manifest contains duplicate canonical object IDs")

        if len(set(self.included_capabilities)) != len(self.included_capabilities):
            raise ValueError("manifest contains duplicate included capabilities")

        omission_codes = [item.code for item in self.omissions]
        if len(set(omission_codes)) != len(omission_codes):
            raise ValueError("manifest contains duplicate omission codes")
        overlap = set(self.included_capabilities) & set(omission_codes)
        if overlap:
            raise ValueError("a capability cannot be both included and omitted")

        mapping_sources = [item.source_identity_id for item in self.identity_mappings]
        if len(set(mapping_sources)) != len(mapping_sources):
            raise ValueError("manifest maps a source identity more than once")
        mapping_destinations = [
            item.destination_identity_id for item in self.identity_mappings
        ]
        if len(set(mapping_destinations)) != len(mapping_destinations):
            raise ValueError("manifest maps more than one source identity to a destination")

        missing_envelopes = set(self.encryption.key_envelope_paths) - set(file_paths)
        if missing_envelopes:
            raise ValueError("encryption references files absent from the manifest")
        return self


def validate_portable_manifest(
    document: PortableManifest | Mapping[str, Any],
) -> PortableManifest:
    """Purely validate and return a portable manifest.

    This function performs no filesystem, archive, network, authorization, or
    cryptographic work.  Callers must compare the returned file descriptors to
    actual bytes before any mutation.
    """

    candidate = document.model_dump() if isinstance(document, PortableManifest) else document
    return PortableManifest.model_validate(candidate)


__all__ = [
    "CanonicalObjectId",
    "ChecksumDescriptor",
    "DeclaredOmission",
    "EncryptionReference",
    "ExportScope",
    "IdentityMapping",
    "ManifestFile",
    "PortableManifest",
    "PositionReference",
    "ProducerReference",
    "SnapshotMetadata",
    "validate_portable_manifest",
]
