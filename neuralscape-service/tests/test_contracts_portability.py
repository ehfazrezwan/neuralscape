"""Behavioral and adversarial tests for portable manifest validation."""

from copy import deepcopy

import pytest
from pydantic import ValidationError

from contracts_portability import PortableManifest, validate_portable_manifest


def valid_manifest() -> dict:
    return {
        "schema_version": "candidate-v1",
        "format_version": "portable-candidate-v1",
        "manifest_id": "manifest-7",
        "producer": {"implementation": "reference-exporter", "version": "1.4.2"},
        "scope": {
            "kind": "projects",
            "tenant_id": "tenant-a",
            "project_ids": ["project-red", "project-blue"],
        },
        "snapshot": {
            "snapshot_id": "snapshot-15",
            "snapshot_position": {"stream_id": "canonical-ledger", "position": "1042"},
            "completion_barrier": {"stream_id": "canonical-ledger", "position": "1042"},
            "replay_position": {"stream_id": "accepted-intents", "position": "733"},
        },
        "files": [
            {
                "path": "canonical/records.jsonl",
                "size_bytes": 8192,
                "checksum": {"algorithm": "sha256", "digest": "a" * 64},
                "entry_type": "file",
            },
            {
                "path": "crypto/envelope.bin",
                "size_bytes": 256,
                "checksum": {"algorithm": "sha256", "digest": "b" * 64},
                "entry_type": "file",
            },
        ],
        "canonical_ids": [
            {"kind": "record", "id": "record-1"},
            {"kind": "revision", "id": "revision-1"},
            {"kind": "tombstone", "id": "tombstone-9"},
        ],
        "included_capabilities": ["canonical-records", "tombstones"],
        "omissions": [
            {"code": "model-specific-projections", "reason": "Not selected for this export."}
        ],
        "identity_mappings": [
            {
                "source_identity_id": "source-member-2",
                "destination_identity_id": "destination-member-8",
                "authority_reference": "mapping-approval-31",
            }
        ],
        "encryption": {
            "mode": "encrypted",
            "protocol_reference": "customer-protocol-3",
            "key_envelope_paths": ["crypto/envelope.bin"],
        },
    }


def test_valid_manifest_preserves_opaque_ids_and_serializes() -> None:
    document = valid_manifest()
    document["canonical_ids"][0]["id"] = " Record-1 "

    manifest = validate_portable_manifest(document)

    assert isinstance(manifest, PortableManifest)
    assert manifest.canonical_ids[0].id == " Record-1 "
    assert manifest.model_dump(mode="json")["snapshot"]["replay_position"]["position"] == "733"
    revalidated = validate_portable_manifest(manifest)
    assert revalidated == manifest
    assert revalidated is not manifest


@pytest.mark.parametrize(
    "unsafe_path",
    [
        "/etc/passwd",
        "../records.jsonl",
        "canonical/../../outside",
        r"..\outside",
        r"C:\outside\records.jsonl",
        "C:/outside/records.jsonl",
        "C:outside/records.jsonl",
        "\x00records.jsonl",
        ".",
    ],
)
def test_rejects_unsafe_file_paths(unsafe_path: str) -> None:
    document = valid_manifest()
    document["files"][0]["path"] = unsafe_path

    with pytest.raises(ValidationError, match="bundle path"):
        validate_portable_manifest(document)


def test_rejects_duplicate_normalized_paths() -> None:
    document = valid_manifest()
    duplicate = deepcopy(document["files"][0])
    duplicate["path"] = "canonical/./records.jsonl"
    document["files"].append(duplicate)

    with pytest.raises(ValidationError, match="duplicate normalized file paths"):
        validate_portable_manifest(document)


@pytest.mark.parametrize(
    ("checksum", "error"),
    [
        ({"algorithm": "md5", "digest": "a" * 64}, "algorithm"),
        ({"algorithm": "sha256", "digest": "A" * 64}, "digest"),
        ({"algorithm": "sha256", "digest": "a" * 63}, "digest"),
        ({"digest": "a" * 64}, "algorithm"),
    ],
)
def test_rejects_invalid_checksum_descriptors(checksum: dict, error: str) -> None:
    document = valid_manifest()
    document["files"][0]["checksum"] = checksum

    with pytest.raises(ValidationError, match=error):
        validate_portable_manifest(document)


def test_rejects_boolean_and_out_of_range_file_lengths() -> None:
    for invalid in (True, -1, 9_007_199_254_740_992):
        document = valid_manifest()
        document["files"][0]["size_bytes"] = invalid
        with pytest.raises(ValidationError):
            validate_portable_manifest(document)


@pytest.mark.parametrize(
    "missing_field",
    ["snapshot_id", "snapshot_position", "completion_barrier", "replay_position"],
)
def test_rejects_incomplete_snapshot_metadata(missing_field: str) -> None:
    document = valid_manifest()
    del document["snapshot"][missing_field]

    with pytest.raises(ValidationError, match=missing_field):
        validate_portable_manifest(document)


def test_rejects_duplicate_canonical_ids_within_a_kind() -> None:
    document = valid_manifest()
    document["canonical_ids"].append({"kind": "record", "id": "record-1"})

    with pytest.raises(ValidationError, match="duplicate canonical object IDs"):
        validate_portable_manifest(document)


def test_same_opaque_id_in_distinct_canonical_kinds_is_not_a_collision() -> None:
    document = valid_manifest()
    document["canonical_ids"].append({"kind": "artifact", "id": "record-1"})

    manifest = validate_portable_manifest(document)

    assert manifest.canonical_ids[-1].kind == "artifact"


def test_scope_requires_consistent_explicit_project_selection() -> None:
    tenant_document = valid_manifest()
    tenant_document["scope"] = {
        "kind": "tenant",
        "tenant_id": "tenant-a",
        "project_ids": [],
    }
    assert validate_portable_manifest(tenant_document).scope.kind == "tenant"

    for scope in (
        {"kind": "projects", "tenant_id": "tenant-a", "project_ids": []},
        {"kind": "tenant", "tenant_id": "tenant-a", "project_ids": ["project-red"]},
        {
            "kind": "projects",
            "tenant_id": "tenant-a",
            "project_ids": ["project-red", "project-red"],
        },
    ):
        document = valid_manifest()
        document["scope"] = scope
        with pytest.raises(ValidationError, match="scope|project"):
            validate_portable_manifest(document)


def test_requires_explicit_omissions_and_does_not_default_optional_state() -> None:
    document = valid_manifest()
    del document["omissions"]

    with pytest.raises(ValidationError, match="omissions"):
        validate_portable_manifest(document)


def test_capability_cannot_be_declared_included_and_omitted() -> None:
    document = valid_manifest()
    document["omissions"].append(
        {"code": "canonical-records", "reason": "Contradictory declaration."}
    )

    with pytest.raises(ValidationError, match="both included and omitted"):
        validate_portable_manifest(document)


def test_rejects_duplicate_source_identity_mapping() -> None:
    document = valid_manifest()
    duplicate = deepcopy(document["identity_mappings"][0])
    duplicate["destination_identity_id"] = "destination-member-9"
    document["identity_mappings"].append(duplicate)

    with pytest.raises(ValidationError, match="source identity more than once"):
        validate_portable_manifest(document)


def test_rejects_many_to_one_destination_identity_mapping() -> None:
    document = valid_manifest()
    document["identity_mappings"].append(
        {
            "source_identity_id": "source-member-3",
            "destination_identity_id": "destination-member-8",
            "authority_reference": "mapping-approval-32",
        }
    )

    with pytest.raises(ValidationError, match="more than one source identity"):
        validate_portable_manifest(document)


def test_identity_mapping_requires_separate_authority_reference() -> None:
    document = valid_manifest()
    del document["identity_mappings"][0]["authority_reference"]

    with pytest.raises(ValidationError, match="authority_reference"):
        validate_portable_manifest(document)


def test_encrypted_mode_requires_protocol_and_declared_envelope_files() -> None:
    without_protocol = valid_manifest()
    without_protocol["encryption"]["protocol_reference"] = None
    with pytest.raises(ValidationError, match="protocol reference"):
        validate_portable_manifest(without_protocol)

    without_envelope = valid_manifest()
    without_envelope["encryption"]["key_envelope_paths"] = []
    with pytest.raises(ValidationError, match="at least one key envelope"):
        validate_portable_manifest(without_envelope)

    missing_envelope = valid_manifest()
    missing_envelope["encryption"]["key_envelope_paths"] = ["crypto/missing.bin"]
    with pytest.raises(ValidationError, match="files absent"):
        validate_portable_manifest(missing_envelope)


def test_plaintext_mode_forbids_crypto_references() -> None:
    document = valid_manifest()
    document["encryption"]["mode"] = "plaintext_authorized_export"

    with pytest.raises(ValidationError, match="plaintext export"):
        validate_portable_manifest(document)

    document["encryption"] = {
        "mode": "plaintext_authorized_export",
        "protocol_reference": None,
        "key_envelope_paths": [],
    }
    assert validate_portable_manifest(document).encryption.mode == "plaintext_authorized_export"


def test_revalidating_instance_rejects_mutated_unsafe_path() -> None:
    manifest = validate_portable_manifest(valid_manifest())
    manifest.files[0].path = "../escaped.jsonl"

    with pytest.raises(ValidationError, match="bundle path"):
        validate_portable_manifest(manifest)


def test_revalidating_instance_rejects_mutated_inconsistent_scope() -> None:
    manifest = validate_portable_manifest(valid_manifest())
    manifest.scope.kind = "tenant"

    with pytest.raises(ValidationError, match="tenant scope"):
        validate_portable_manifest(manifest)


def test_revalidating_instance_rejects_mutated_duplicate_inventory() -> None:
    manifest = validate_portable_manifest(valid_manifest())
    manifest.files.append(manifest.files[0])

    with pytest.raises(ValidationError, match="duplicate normalized file paths"):
        validate_portable_manifest(manifest)


def test_revalidating_instance_rejects_mutated_missing_envelope_file() -> None:
    manifest = validate_portable_manifest(valid_manifest())
    manifest.encryption.key_envelope_paths.append("crypto/missing.bin")

    with pytest.raises(ValidationError, match="files absent"):
        validate_portable_manifest(manifest)


def test_unknown_fields_and_missing_schema_version_are_rejected() -> None:
    unknown = valid_manifest()
    unknown["authorization_proven"] = True
    with pytest.raises(ValidationError, match="authorization_proven"):
        validate_portable_manifest(unknown)

    unversioned = valid_manifest()
    del unversioned["schema_version"]
    with pytest.raises(ValidationError, match="schema_version"):
        validate_portable_manifest(unversioned)
