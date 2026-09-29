"""Behavioral and adversarial tests for portable manifest validation."""

import sys
from copy import deepcopy
from pathlib import PureWindowsPath

import pytest
from pydantic import Field, ValidationError

from contracts_portability import (
    ChecksumDescriptor,
    EncryptionReference,
    ExportScope,
    ManifestFile,
    PortableManifest,
    ProducerReference,
    validate_portable_manifest,
)


class FutureManifestFile(ManifestFile):
    future_contract_field: str


class FutureProducerReference(ProducerReference):
    future_contract_field: str


class FutureChecksumDescriptor(ChecksumDescriptor):
    future_contract_field: str


class FutureExportScope(ExportScope):
    future_contract_field: str


class FutureEncryptionReference(EncryptionReference):
    future_contract_field: str


class CompatibleManifestFile(ManifestFile):
    pass


class ExcludingFutureManifestFile(ManifestFile):
    future_contract_field: str = Field(exclude=True)


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
        "./C:/outside/records.jsonl",
        "./C:outside/records.jsonl",
        "\x00records.jsonl",
        ".",
    ],
)
def test_rejects_unsafe_file_paths(unsafe_path: str) -> None:
    document = valid_manifest()
    document["files"][0]["path"] = unsafe_path

    with pytest.raises(ValidationError, match="bundle path"):
        validate_portable_manifest(document)


@pytest.mark.parametrize(
    "reserved_path",
    [
        "NUL",
        "files/CON.txt",
        "CON/files.json",
        "nested/AuX",
        "nested/PrN.json",
        "nested/COM1.bin",
        "nested/lpt9.backup",
        "nested/CONIN$",
        "nested/conout$.log",
        "nested/NUL .txt",
    ],
)
def test_rejects_windows_reserved_device_paths(reserved_path: str) -> None:
    document = valid_manifest()
    document["files"][0]["path"] = reserved_path

    with pytest.raises(ValidationError, match="Windows reserved device name"):
        validate_portable_manifest(document)


@pytest.mark.parametrize(
    "safe_path",
    [
        "devices/null.bin",
        "files/console.txt",
        "devices/auxiliary.txt",
        "ports/com0.bin",
        "ports/com10.bin",
        "ports/lpt0.bin",
        "ports/lpt10.bin",
        "reports/prn-report.json",
        "reports/conifer.json",
    ],
)
def test_accepts_names_similar_to_windows_devices(safe_path: str) -> None:
    document = valid_manifest()
    document["files"][0]["path"] = safe_path

    manifest = validate_portable_manifest(document)

    assert manifest.files[0].path == safe_path


@pytest.mark.parametrize("reserved_path", ["keys/NUL.bin", "CON/envelope.bin"])
def test_rejects_windows_reserved_envelope_paths(reserved_path: str) -> None:
    document = valid_manifest()
    document["encryption"]["key_envelope_paths"] = [reserved_path]

    with pytest.raises(ValidationError, match="Windows reserved device name"):
        validate_portable_manifest(document)


@pytest.mark.parametrize("target", ["file", "envelope"])
@pytest.mark.parametrize(
    "invalid_path",
    [
        "records.jsonl:payload",
        "nested/records.jsonl:payload",
        "nested/<records>.jsonl",
        'nested/"records".jsonl',
        "nested/records|backup.jsonl",
        "nested/records?.jsonl",
        "nested/records*.jsonl",
        "nested/control-\x01.jsonl",
        "nested/control-\x1f.jsonl",
        "records.jsonl.",
        "nested/records.jsonl.",
        "canonical/records.jsonl./",
        "records.jsonl ",
        "nested/records.jsonl ",
        "canonical/archive /./records.jsonl",
    ],
)
def test_rejects_windows_invalid_file_and_envelope_components(
    target: str,
    invalid_path: str,
) -> None:
    document = valid_manifest()
    if target == "file":
        document["files"][0]["path"] = invalid_path
    else:
        document["encryption"]["key_envelope_paths"] = [invalid_path]

    with pytest.raises(ValidationError, match="bundle path components"):
        validate_portable_manifest(document)


@pytest.mark.parametrize(
    ("safe_path", "normalized"),
    [
        ("canonical/records.v1.jsonl", "canonical/records.v1.jsonl"),
        ("canonical/my records.jsonl", "canonical/my records.jsonl"),
        ("canonical/semi;colon.jsonl", "canonical/semi;colon.jsonl"),
        ("canonical/equal=name+copy.jsonl", "canonical/equal=name+copy.jsonl"),
        ("canonical/[records].jsonl", "canonical/[records].jsonl"),
        ("canonical/.metadata/records", "canonical/.metadata/records"),
        ("canonical/records：payload.jsonl", "canonical/records：payload.jsonl"),
        ("canonical/./safe-records.jsonl", "canonical/safe-records.jsonl"),
        ("canonical//safe-records.jsonl", "canonical/safe-records.jsonl"),
    ],
)
def test_accepts_windows_safe_near_misses_and_normalization(
    safe_path: str,
    normalized: str,
) -> None:
    document = valid_manifest()
    document["files"][1]["path"] = safe_path
    document["encryption"]["key_envelope_paths"] = [safe_path]

    manifest = validate_portable_manifest(document)

    assert manifest.files[1].path == normalized
    assert manifest.encryption.key_envelope_paths == [normalized]


def test_rejects_duplicate_normalized_paths() -> None:
    document = valid_manifest()
    duplicate = deepcopy(document["files"][0])
    duplicate["path"] = "canonical/./records.jsonl"
    document["files"].append(duplicate)

    with pytest.raises(ValidationError, match="duplicate normalized file paths"):
        validate_portable_manifest(document)


@pytest.mark.parametrize("entrypoint", ["helper", "model_validate", "constructor"])
@pytest.mark.parametrize(
    ("first_path", "colliding_path"),
    [
        ("canonical/Records.jsonl", "canonical/records.jsonl"),
        ("Canonical/records.jsonl", "canonical/records.jsonl"),
        (
            "canonical/Subdir/records.jsonl",
            "canonical/subdir/records.jsonl",
        ),
        ("canonical/Ä-records.jsonl", "canonical/ä-records.jsonl"),
    ],
)
def test_rejects_file_inventory_collisions_under_windows_path_semantics(
    entrypoint: str,
    first_path: str,
    colliding_path: str,
) -> None:
    assert first_path != colliding_path
    assert PureWindowsPath(first_path) == PureWindowsPath(colliding_path)
    document = valid_manifest()
    document["files"][0]["path"] = first_path
    duplicate = deepcopy(document["files"][0])
    duplicate["path"] = colliding_path
    document["files"].append(duplicate)

    with pytest.raises(ValidationError, match="colliding Windows file paths"):
        if entrypoint == "helper":
            validate_portable_manifest(document)
        elif entrypoint == "model_validate":
            PortableManifest.model_validate(document)
        else:
            PortableManifest(**document)


@pytest.mark.parametrize("entrypoint", ["helper", "model_validate"])
def test_rejects_windows_case_collision_in_existing_model_inventory(
    entrypoint: str,
) -> None:
    manifest = validate_portable_manifest(valid_manifest())
    colliding_file = manifest.files[0].model_copy(
        update={"path": "canonical/Records.jsonl"}
    )
    corrupted = manifest.model_copy(
        update={"files": [colliding_file, *manifest.files]}
    )

    with pytest.raises(ValidationError, match="colliding Windows file paths"):
        if entrypoint == "helper":
            validate_portable_manifest(corrupted)
        else:
            PortableManifest.model_validate(corrupted)


def test_unique_mixed_case_inventory_path_preserves_exact_spelling() -> None:
    document = valid_manifest()
    document["files"][0]["path"] = "Canonical/Records.JSONL"

    manifest = validate_portable_manifest(document)

    assert manifest.files[0].path == "Canonical/Records.JSONL"


@pytest.mark.parametrize(
    ("first_path", "distinct_path"),
    [
        ("canonical/straße.jsonl", "canonical/STRASSE.jsonl"),
        ("canonical/café.jsonl", "canonical/cafe\u0301.jsonl"),
        ("canonical/records：payload.jsonl", "canonical/records_payload.jsonl"),
    ],
)
def test_preserves_unicode_paths_distinct_under_windows_path_semantics(
    first_path: str,
    distinct_path: str,
) -> None:
    assert PureWindowsPath(first_path) != PureWindowsPath(distinct_path)
    document = valid_manifest()
    document["files"][0]["path"] = first_path
    distinct = deepcopy(document["files"][0])
    distinct["path"] = distinct_path
    document["files"].append(distinct)

    manifest = validate_portable_manifest(document)

    retained_paths = [
        item.path
        for item in manifest.files
        if item.path in {first_path, distinct_path}
    ]
    assert retained_paths == [
        first_path,
        distinct_path,
    ]


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


def test_rejects_drive_qualified_envelope_path_after_normalization() -> None:
    document = valid_manifest()
    document["encryption"]["key_envelope_paths"] = ["./C:/outside/envelope.bin"]

    with pytest.raises(ValidationError, match="Windows drive-qualified"):
        validate_portable_manifest(document)


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


def _manifest_with_nested_runtime_subclass(
    nested_kind: str,
) -> tuple[PortableManifest, object]:
    manifest = validate_portable_manifest(valid_manifest()).model_copy(deep=True)
    if nested_kind == "file":
        node = FutureManifestFile(
            **manifest.files[0].model_dump(),
            future_contract_field="deny",
        )
        manifest.files[0] = node
    elif nested_kind == "producer":
        node = FutureProducerReference(
            **manifest.producer.model_dump(),
            future_contract_field="deny",
        )
        manifest.producer = node
    elif nested_kind == "checksum":
        node = FutureChecksumDescriptor(
            **manifest.files[0].checksum.model_dump(),
            future_contract_field="deny",
        )
        manifest.files[0].checksum = node
    elif nested_kind == "scope":
        node = FutureExportScope(
            **manifest.scope.model_dump(),
            future_contract_field="deny",
        )
        manifest.scope = node
    else:
        node = FutureEncryptionReference(
            **manifest.encryption.model_dump(),
            future_contract_field="deny",
        )
        manifest.encryption = node
    return manifest, node


@pytest.mark.parametrize("entrypoint", ["helper", "model_validate"])
@pytest.mark.parametrize(
    "nested_kind",
    ["file", "producer", "checksum", "scope", "encryption"],
)
def test_rejects_declared_fields_from_nested_runtime_subclasses(
    entrypoint: str,
    nested_kind: str,
) -> None:
    corrupted, node = _manifest_with_nested_runtime_subclass(nested_kind)

    assert node.__dict__["future_contract_field"] == "deny"
    with pytest.raises(ValidationError, match="future_contract_field"):
        if entrypoint == "helper":
            validate_portable_manifest(corrupted)
        else:
            PortableManifest.model_validate(corrupted)


def test_rejects_undeclared_field_retained_on_nested_runtime_subclass() -> None:
    corrupted, node = _manifest_with_nested_runtime_subclass("file")
    unknown = node.model_copy(update={"undeclared_constraint": "deny"})
    corrupted.files[0] = unknown

    with pytest.raises(ValidationError, match="undeclared_constraint") as raised:
        validate_portable_manifest(corrupted)

    assert raised.value.errors()[0]["type"] == "extra_forbidden"


def test_rejects_nested_subclass_field_excluded_from_serialization() -> None:
    manifest = validate_portable_manifest(valid_manifest()).model_copy(deep=True)
    future_file = ExcludingFutureManifestFile(
        **manifest.files[0].model_dump(),
        future_contract_field="deny",
    )
    manifest.files[0] = future_file

    assert "future_contract_field" not in future_file.model_dump()
    with pytest.raises(ValidationError, match="future_contract_field"):
        validate_portable_manifest(manifest)


def _model_at_location(manifest: PortableManifest, location: str) -> object:
    if location == "manifest":
        return manifest
    if location == "file":
        return manifest.files[0]
    return manifest.files[0].checksum


@pytest.mark.parametrize("malformed_extra", [[], ["retained_unknown"]])
@pytest.mark.parametrize("location", ["manifest", "file", "checksum"])
def test_rejects_malformed_pydantic_extra_storage_as_controlled_validation(
    location: str,
    malformed_extra: list[str],
) -> None:
    manifest = validate_portable_manifest(valid_manifest()).model_copy(deep=True)
    node = _model_at_location(manifest, location)
    object.__setattr__(node, "__pydantic_extra__", malformed_extra)

    with pytest.raises(
        ValidationError,
        match="contract model extra storage must be None or a mapping",
    ):
        validate_portable_manifest(manifest)


@pytest.mark.parametrize(
    ("location", "field_name", "conflicting_value"),
    [
        ("manifest", "manifest_id", "different-manifest"),
        ("file", "path", "../escaped.jsonl"),
        ("checksum", "algorithm", "sha512"),
    ],
)
@pytest.mark.parametrize("use_same_value", [True, False])
def test_rejects_overlap_between_stored_and_extra_state(
    location: str,
    field_name: str,
    conflicting_value: str,
    use_same_value: bool,
) -> None:
    manifest = validate_portable_manifest(valid_manifest()).model_copy(deep=True)
    node = _model_at_location(manifest, location)
    extra_value = node.__dict__[field_name] if use_same_value else conflicting_value
    object.__setattr__(node, "__pydantic_extra__", {field_name: extra_value})

    with pytest.raises(
        ValidationError,
        match="contract model has conflicting stored and extra fields",
    ):
        validate_portable_manifest(manifest)


def test_empty_mapping_extra_storage_remains_valid() -> None:
    manifest = validate_portable_manifest(valid_manifest()).model_copy(deep=True)
    object.__setattr__(manifest.files[0], "__pydantic_extra__", {})

    revalidated = validate_portable_manifest(manifest)

    assert revalidated.files[0].path == manifest.files[0].path


def test_compatible_nested_subclass_without_new_fields_remains_valid() -> None:
    manifest = validate_portable_manifest(valid_manifest()).model_copy(deep=True)
    compatible = CompatibleManifestFile(**manifest.files[0].model_dump())
    manifest.files[0] = compatible

    revalidated = validate_portable_manifest(manifest)

    assert revalidated.files[0].model_dump() == compatible.model_dump()
    assert type(revalidated.files[0]) is ManifestFile
    assert revalidated.files[0] is not compatible


def test_shared_nested_model_is_not_mistaken_for_an_active_cycle() -> None:
    manifest = validate_portable_manifest(valid_manifest()).model_copy(deep=True)
    shared_checksum = manifest.files[0].checksum
    manifest.files[1].checksum = shared_checksum

    revalidated = validate_portable_manifest(manifest)

    assert revalidated.files[0].checksum == shared_checksum
    assert revalidated.files[1].checksum == shared_checksum


@pytest.mark.parametrize(
    "malformed_kind", ["tuple_files", "set_capabilities", "bool_size"]
)
def test_subclass_preserving_reconstruction_does_not_normalize_malformed_values(
    malformed_kind: str,
) -> None:
    manifest = validate_portable_manifest(valid_manifest()).model_copy(deep=True)
    if malformed_kind == "tuple_files":
        malformed_value = tuple(manifest.files)
        corrupted = manifest.model_copy(update={"files": malformed_value})
    elif malformed_kind == "set_capabilities":
        malformed_value = set(manifest.included_capabilities)
        corrupted = manifest.model_copy(
            update={"included_capabilities": malformed_value}
        )
    else:
        malformed_value = True
        invalid_file = manifest.files[0].model_copy(update={"size_bytes": True})
        corrupted = manifest.model_copy(
            update={"files": [invalid_file, manifest.files[1]]}
        )

    with pytest.raises(ValidationError):
        validate_portable_manifest(corrupted)
    if malformed_kind == "tuple_files":
        assert corrupted.files is malformed_value
    elif malformed_kind == "set_capabilities":
        assert corrupted.included_capabilities is malformed_value
    else:
        assert corrupted.files[0].size_bytes is malformed_value


def test_cyclic_declared_nested_subclass_field_is_a_controlled_failure() -> None:
    corrupted, node = _manifest_with_nested_runtime_subclass("file")
    cyclic_value = []
    cyclic_value.append(cyclic_value)
    corrupted.files[0] = node.model_copy(
        update={"future_contract_field": cyclic_value}
    )

    with pytest.raises(ValidationError):
        validate_portable_manifest(corrupted)


def test_mixed_mapping_rejects_nested_manifest_file_with_unsafe_path() -> None:
    manifest = validate_portable_manifest(valid_manifest())
    document = manifest.model_dump()
    invalid_file = manifest.files[0].model_copy(update={"path": "../escaped"})
    document["files"][0] = invalid_file

    assert invalid_file.path == "../escaped"
    with pytest.raises(ValidationError, match="bundle path"):
        validate_portable_manifest(document)


def test_mixed_mapping_rejects_nested_manifest_file_with_retained_extra() -> None:
    manifest = validate_portable_manifest(valid_manifest())
    document = manifest.model_dump()
    invalid_file = manifest.files[0].model_copy(
        update={"future_entry_semantics": "deny"}
    )
    document["files"][0] = invalid_file

    assert invalid_file.__dict__["future_entry_semantics"] == "deny"
    assert "future_entry_semantics" in invalid_file.model_fields_set
    with pytest.raises(ValidationError, match="future_entry_semantics"):
        validate_portable_manifest(document)


@pytest.mark.parametrize("nested_kind", ["producer", "checksum", "scope"])
def test_mixed_mapping_rejects_other_invalid_nested_models(
    nested_kind: str,
) -> None:
    manifest = validate_portable_manifest(valid_manifest())
    document = manifest.model_dump()

    if nested_kind == "producer":
        invalid_node = manifest.producer.model_copy(update={"implementation": ""})
        document["producer"] = invalid_node
        error = "implementation"
    elif nested_kind == "checksum":
        invalid_node = manifest.files[0].checksum.model_copy(update={"digest": "bad"})
        document["files"][0]["checksum"] = invalid_node
        error = "digest"
    else:
        invalid_node = manifest.scope.model_copy(update={"tenant_id": ""})
        document["scope"] = invalid_node
        error = "tenant_id"

    with pytest.raises(ValidationError, match=error):
        validate_portable_manifest(document)


@pytest.mark.parametrize(
    "location",
    [
        "manifest",
        "file",
        "checksum",
        "scope",
        "replay_position",
        "identity_mapping",
        "encryption",
    ],
)
def test_revalidating_instance_rejects_retained_unknown_fields(location: str) -> None:
    manifest = validate_portable_manifest(valid_manifest()).model_copy(deep=True)

    if location == "manifest":
        candidate = manifest.model_copy(update={"future_contract_field": "deny"})
        retained_node = candidate
    else:
        if location == "file":
            retained_node = manifest.files[0].model_copy(
                update={"future_contract_field": "deny"}
            )
            manifest.files[0] = retained_node
        elif location == "checksum":
            retained_node = manifest.files[0].checksum.model_copy(
                update={"future_contract_field": "deny"}
            )
            manifest.files[0].checksum = retained_node
        elif location == "scope":
            retained_node = manifest.scope.model_copy(
                update={"future_contract_field": "deny"}
            )
            manifest.scope = retained_node
        elif location == "replay_position":
            retained_node = manifest.snapshot.replay_position.model_copy(
                update={"future_contract_field": "deny"}
            )
            manifest.snapshot.replay_position = retained_node
        elif location == "identity_mapping":
            retained_node = manifest.identity_mappings[0].model_copy(
                update={"future_contract_field": "deny"}
            )
            manifest.identity_mappings[0] = retained_node
        else:
            retained_node = manifest.encryption.model_copy(
                update={"future_contract_field": "deny"}
            )
            manifest.encryption = retained_node
        candidate = manifest

    assert retained_node.__dict__["future_contract_field"] == "deny"
    assert "future_contract_field" in retained_node.model_fields_set
    with pytest.raises(ValidationError, match="future_contract_field"):
        validate_portable_manifest(candidate)


def test_valid_declared_field_copy_is_normalized_and_deeply_rebuilt() -> None:
    manifest = validate_portable_manifest(valid_manifest())
    copied_file = manifest.files[0].model_copy(
        update={"path": "canonical/./renamed-records.jsonl"}
    )
    candidate = manifest.model_copy(update={"files": [copied_file, manifest.files[1]]})

    revalidated = validate_portable_manifest(candidate)

    assert revalidated.files[0].path == "canonical/renamed-records.jsonl"
    assert revalidated is not candidate
    assert revalidated.files is not candidate.files
    assert revalidated.files[0] is not candidate.files[0]
    assert revalidated.files[0].checksum is not candidate.files[0].checksum


@pytest.mark.parametrize("container_kind", ["tuple", "mapping"])
def test_retained_unknown_field_in_native_container_is_not_sanitized(
    container_kind: str,
) -> None:
    manifest = validate_portable_manifest(valid_manifest())
    copied_file = manifest.files[0].model_copy(
        update={"future_contract_field": "deny"}
    )
    if container_kind == "tuple":
        native_container = (copied_file, manifest.files[1])
    else:
        native_container = {("opaque", 1): copied_file}
    candidate = manifest.model_copy(update={"files": native_container})

    with pytest.raises(ValidationError, match="future_contract_field"):
        validate_portable_manifest(candidate)


def test_revalidating_instance_controls_malformed_defined_field() -> None:
    manifest = validate_portable_manifest(valid_manifest()).model_copy(deep=True)
    manifest.files[0].__dict__["checksum"] = "not-a-checksum"

    with pytest.raises(ValidationError, match="checksum"):
        validate_portable_manifest(manifest)


@pytest.mark.parametrize("cycle_kind", ["files", "checksum", "replay_position"])
def test_revalidating_instance_controls_cyclic_defined_fields(cycle_kind: str) -> None:
    manifest = validate_portable_manifest(valid_manifest()).model_copy(deep=True)
    if cycle_kind == "files":
        cyclic_files = []
        cyclic_files.append(cyclic_files)
        manifest.__dict__["files"] = cyclic_files
    elif cycle_kind == "checksum":
        manifest.files[0].__dict__["checksum"] = manifest.files[0]
    else:
        manifest.snapshot.__dict__["replay_position"] = manifest.snapshot

    with pytest.raises(ValidationError):
        validate_portable_manifest(manifest)


def test_mapping_input_cycle_is_a_controlled_validation_failure() -> None:
    document = valid_manifest()
    document["files"] = document

    with pytest.raises(ValidationError):
        validate_portable_manifest(document)


def test_deep_unchecked_native_container_is_a_controlled_validation_failure() -> None:
    manifest = validate_portable_manifest(valid_manifest())
    deeply_nested: object = "leaf"
    for _ in range(sys.getrecursionlimit() + 100):
        deeply_nested = [deeply_nested]
    corrupted = manifest.model_copy(update={"files": deeply_nested})

    with pytest.raises(ValidationError) as raised:
        validate_portable_manifest(corrupted)

    assert isinstance(raised.value.__cause__, RecursionError)


def test_unknown_fields_and_missing_schema_version_are_rejected() -> None:
    unknown = valid_manifest()
    unknown["authorization_proven"] = True
    with pytest.raises(ValidationError, match="authorization_proven"):
        validate_portable_manifest(unknown)

    unversioned = valid_manifest()
    del unversioned["schema_version"]
    with pytest.raises(ValidationError, match="schema_version"):
        validate_portable_manifest(unversioned)
