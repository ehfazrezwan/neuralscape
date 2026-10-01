"""Behavioral and adversarial tests for portable manifest validation."""

import json
import sys
from collections.abc import ItemsView, Iterator, Mapping
from copy import deepcopy
from pathlib import PureWindowsPath
from types import MappingProxyType

import pytest
from pydantic import BaseModel, Field, ValidationError

from contracts_portability import (
    ChecksumDescriptor,
    EncryptionReference,
    ExportScope,
    ManifestFile,
    PortableManifest,
    ProducerReference,
    validate_portable_manifest,
)


_PYDANTIC_DICT_DESCRIPTOR = BaseModel.__dict__["__dict__"]
_PYDANTIC_EXTRA_SLOT = BaseModel.__dict__["__pydantic_extra__"]
_PYDANTIC_FIELDS_SET_SLOT = BaseModel.__dict__["__pydantic_fields_set__"]


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


class CompatibleDefaultManifest(PortableManifest):
    defaulted_contract_field: str = "default"


class CompatibleDefaultManifestFile(ManifestFile):
    defaulted_contract_field: str = "default"


class ExcludingFutureManifestFile(ManifestFile):
    future_contract_field: str = Field(exclude=True)


class SplitViewExtra(Mapping[str, str]):
    def __init__(
        self,
        entries: tuple[tuple[str, str], ...],
        *,
        iteration_keys: tuple[str, ...] = (),
        reported_keys: tuple[str, ...] = (),
        reported_length: int = 0,
    ) -> None:
        self._entries = entries
        self._iteration_keys = iteration_keys
        self._reported_keys = reported_keys
        self._reported_length = reported_length
        self._values = dict(entries)
        self.iteration_calls = 0
        self.item_calls = 0

    def __getitem__(self, key: str) -> str:
        return self._values[key]

    def __iter__(self) -> Iterator[str]:
        self.iteration_calls += 1
        return iter(self._iteration_keys)

    def __len__(self) -> int:
        return self._reported_length

    def keys(self) -> tuple[str, ...]:
        return self._reported_keys

    def items(self) -> tuple[tuple[str, str], ...]:
        self.item_calls += 1
        return self._entries


class InverseItemsExtra(Mapping[str, str]):
    def __init__(self, name: str, value: str) -> None:
        self.name = name
        self.value = value
        self.iteration_calls = 0
        self.item_calls = 0

    def __getitem__(self, key: str) -> str:
        if key == self.name:
            return self.value
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        self.iteration_calls += 1
        return iter((self.name,))

    def __len__(self) -> int:
        return 1

    def items(self) -> tuple[tuple[str, str], ...]:
        self.item_calls += 1
        return ()


class HiddenBackingExtra(dict[str, str]):
    def __init__(self, entries: dict[str, str]) -> None:
        dict.__init__(self, entries)
        self.length_calls = 0
        self.iteration_calls = 0
        self.key_calls = 0
        self.item_calls = 0
        self.value_calls = 0
        self.truth_calls = 0

    def __len__(self) -> int:
        self.length_calls += 1
        return 0

    def __iter__(self) -> Iterator[str]:
        self.iteration_calls += 1
        return iter(())

    def keys(self) -> tuple[str, ...]:
        self.key_calls += 1
        return ()

    def items(self) -> tuple[tuple[str, str], ...]:
        self.item_calls += 1
        return ()

    def values(self) -> tuple[str, ...]:
        self.value_calls += 1
        return ()

    def __bool__(self) -> bool:
        self.truth_calls += 1
        return False

    def overridden_view_calls(self) -> tuple[int, ...]:
        return (
            self.length_calls,
            self.iteration_calls,
            self.key_calls,
            self.item_calls,
            self.value_calls,
            self.truth_calls,
        )


class InheritedHiddenBackingExtra(HiddenBackingExtra):
    pass


class RaisingClassHiddenBackingExtra(HiddenBackingExtra):
    def __init__(self, entries: dict[str, str]) -> None:
        super().__init__(entries)
        self.class_calls = 0

    @property
    def __class__(self) -> type[dict]:
        self.class_calls += 1
        raise RuntimeError("__class__ must not be read")


class SpoofedDictClassMapping(Mapping[str, str]):
    def __init__(self, entries: dict[str, str]) -> None:
        self._entries = entries
        self.class_calls = 0
        self.iteration_calls = 0
        self.item_calls = 0

    @property
    def __class__(self) -> type[dict]:
        self.class_calls += 1
        return dict

    def __getitem__(self, key: str) -> str:
        return self._entries[key]

    def __iter__(self) -> Iterator[str]:
        self.iteration_calls += 1
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def items(self) -> tuple[tuple[str, str], ...]:
        self.item_calls += 1
        return tuple(self._entries.items())


class ChangingItemsExtra(Mapping[str, str]):
    def __init__(self) -> None:
        self.item_calls = 0
        self.iteration_calls = 0

    def __getitem__(self, key: str) -> str:
        if key == "manifest_id":
            return "shadow-manifest"
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        self.iteration_calls += 1
        return iter(())

    def __len__(self) -> int:
        return 0

    def keys(self) -> tuple[str, ...]:
        return ("manifest_id",)

    def items(self) -> tuple[tuple[str, str], ...]:
        self.item_calls += 1
        if self.item_calls == 1:
            return ()
        return (("manifest_id", "shadow-manifest"),)


class HiddenModelBacking(dict[str, object]):
    def __init__(
        self,
        entries: tuple[tuple[str, object], ...],
        *,
        hidden_name: str | None = None,
    ) -> None:
        dict.__init__(self, entries)
        self.hidden_name = hidden_name
        self.iteration_calls = 0
        self.key_calls = 0
        self.item_calls = 0
        self.value_calls = 0
        self.length_calls = 0

    def __iter__(self) -> Iterator[str]:
        self.iteration_calls += 1
        return (
            key for key in dict.__iter__(self) if key != self.hidden_name
        )

    def keys(self) -> tuple[str, ...]:
        self.key_calls += 1
        return tuple(self)

    def items(self) -> tuple[tuple[str, object], ...]:
        self.item_calls += 1
        return tuple(
            (key, dict.__getitem__(self, key)) for key in self
        )

    def values(self) -> tuple[object, ...]:
        self.value_calls += 1
        return tuple(dict.__getitem__(self, key) for key in self)

    def __len__(self) -> int:
        self.length_calls += 1
        hidden_count = int(
            self.hidden_name is not None
            and dict.__contains__(self, self.hidden_name)
        )
        return dict.__len__(self) - hidden_count

    def overridden_view_calls(self) -> tuple[int, ...]:
        return (
            self.iteration_calls,
            self.key_calls,
            self.item_calls,
            self.value_calls,
            self.length_calls,
        )


class RepairingProducer(dict[str, str]):
    def __init__(self, owner: PortableManifest) -> None:
        dict.__init__(
            self,
            implementation="reference-exporter",
            version="1.4.2",
        )
        self.owner = owner
        self.item_calls = 0

    def items(self) -> ItemsView[str, str]:
        self.item_calls += 1
        native_stored = object.__getattribute__(self.owner, "__dict__")
        dict.__setitem__(native_stored, "manifest_id", "manifest-7")
        return dict.items(self)


class RepairingFieldsSetIterable:
    def __init__(
        self,
        names: tuple[str, ...],
        target: BaseModel,
        field_name: str,
        replacement: str,
    ) -> None:
        self.names = names
        self.target = target
        self.field_name = field_name
        self.replacement = replacement
        self.iteration_calls = 0

    def __iter__(self) -> Iterator[str]:
        self.iteration_calls += 1
        native_stored = _PYDANTIC_DICT_DESCRIPTOR.__get__(
            self.target, BaseModel
        )
        dict.__setitem__(native_stored, self.field_name, self.replacement)
        return iter(self.names)


class RepairingProducerMapping(Mapping[str, str]):
    def __init__(
        self,
        target: BaseModel,
        field_name: str,
        replacement: str,
    ) -> None:
        self.entries = {
            "implementation": "reference-exporter",
            "version": "1.4.2",
        }
        self.target = target
        self.field_name = field_name
        self.replacement = replacement
        self.item_calls = 0

    def __getitem__(self, key: str) -> str:
        return self.entries[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.entries)

    def __len__(self) -> int:
        return len(self.entries)

    def items(self) -> tuple[tuple[str, str], ...]:
        self.item_calls += 1
        native_stored = _PYDANTIC_DICT_DESCRIPTOR.__get__(
            self.target, BaseModel
        )
        dict.__setitem__(native_stored, self.field_name, self.replacement)
        return tuple(self.entries.items())


class NoneReportingExtraManifest(PortableManifest):
    def __getattribute__(self, name: str) -> object:
        if name == "__pydantic_extra__":
            return None
        return super().__getattribute__(name)


class EmptyReportingExtraManifestFile(ManifestFile):
    def __getattribute__(self, name: str) -> object:
        if name == "__pydantic_extra__":
            return {}
        return super().__getattribute__(name)


class PopulatedReportingExtraManifest(PortableManifest):
    def __getattribute__(self, name: str) -> object:
        if name == "__pydantic_extra__":
            return {"reported_only": "deny"}
        return super().__getattribute__(name)


class HidingFieldsSetManifest(PortableManifest):
    def __getattribute__(self, name: str) -> object:
        if name == "model_fields_set":
            return set(type(self).model_fields)
        return super().__getattribute__(name)


class HidingFieldsSetManifestFile(ManifestFile):
    def __getattribute__(self, name: str) -> object:
        if name == "model_fields_set":
            return set(type(self).model_fields)
        return super().__getattribute__(name)


class HiddenFieldsSetBacking(set[str]):
    def __init__(self, values: set[str]) -> None:
        set.__init__(self, values)
        self.iteration_calls = 0
        self.length_calls = 0
        self.contains_calls = 0

    def __iter__(self) -> Iterator[str]:
        self.iteration_calls += 1
        return (
            name for name in set.__iter__(self) if name != "phantom_only"
        )

    def __len__(self) -> int:
        self.length_calls += 1
        hidden_count = int(set.__contains__(self, "phantom_only"))
        return set.__len__(self) - hidden_count

    def __contains__(self, name: object) -> bool:
        self.contains_calls += 1
        if name == "phantom_only":
            return False
        return set.__contains__(self, name)

    def overridden_view_calls(self) -> tuple[int, int, int]:
        return (
            self.iteration_calls,
            self.length_calls,
            self.contains_calls,
        )


class BenignFieldName(str):
    pass


class RepairingFieldName(str):
    def __new__(
        cls,
        value: str,
        target: BaseModel,
        field_name: str,
        replacement: str,
    ) -> "RepairingFieldName":
        instance = super().__new__(cls, value)
        instance.target = target
        instance.field_name = field_name
        instance.replacement = replacement
        instance.hash_calls = 0
        return instance

    def __hash__(self) -> int:
        self.hash_calls += 1
        native_stored = _PYDANTIC_DICT_DESCRIPTOR.__get__(
            self.target, BaseModel
        )
        dict.__setitem__(native_stored, self.field_name, self.replacement)
        return str.__hash__(self)


class DescriptorMaskedManifest(PortableManifest):
    @property
    def __dict__(self) -> dict[str, object]:
        native_stored = _PYDANTIC_DICT_DESCRIPTOR.__get__(self, BaseModel)
        return {
            name: value
            for name, value in dict.items(native_stored)
            if name != "future_entry_semantics"
        }

    @__dict__.setter
    def __dict__(self, value: dict[str, object]) -> None:
        _PYDANTIC_DICT_DESCRIPTOR.__set__(self, value)


class DescriptorMaskedManifestFile(ManifestFile):
    @property
    def __dict__(self) -> dict[str, object]:
        native_stored = _PYDANTIC_DICT_DESCRIPTOR.__get__(self, BaseModel)
        return {
            name: value
            for name, value in dict.items(native_stored)
            if name != "future_entry_semantics"
        }

    @__dict__.setter
    def __dict__(self, value: dict[str, object]) -> None:
        _PYDANTIC_DICT_DESCRIPTOR.__set__(self, value)


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
    if location == "replay_position":
        return manifest.snapshot.replay_position
    return manifest.files[0].checksum


def _manifest_from_origin(origin: str) -> PortableManifest:
    if origin == "native":
        return PortableManifest.model_validate(valid_manifest())
    return PortableManifest.model_validate_json(json.dumps(valid_manifest()))


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


@pytest.mark.parametrize("origin", ["native", "json"])
@pytest.mark.parametrize(
    ("location", "field_name", "conflicting_value"),
    [
        ("manifest", "manifest_id", "different-manifest"),
        ("file", "path", "canonical/different-records.jsonl"),
        ("replay_position", "position", "734"),
    ],
)
@pytest.mark.parametrize("use_original_value", [True, False])
def test_rejects_declared_field_moved_to_extra_storage(
    origin: str,
    location: str,
    field_name: str,
    conflicting_value: str,
    use_original_value: bool,
) -> None:
    manifest = _manifest_from_origin(origin)
    node = _model_at_location(manifest, location)
    original_value = node.__dict__.pop(field_name)
    extra_value = original_value if use_original_value else conflicting_value
    object.__setattr__(node, "__pydantic_extra__", {field_name: extra_value})

    with pytest.raises(
        ValidationError,
        match="contract model has conflicting stored and extra fields",
    ):
        validate_portable_manifest(manifest)


@pytest.mark.parametrize(
    ("location", "field_name"),
    [
        ("manifest", "manifest_id"),
        ("file", "path"),
        ("replay_position", "position"),
    ],
)
def test_declared_required_field_absent_from_all_storage_remains_missing(
    location: str,
    field_name: str,
) -> None:
    manifest = validate_portable_manifest(valid_manifest()).model_copy(deep=True)
    node = _model_at_location(manifest, location)
    node.__dict__.pop(field_name)
    object.__setattr__(node, "__pydantic_extra__", None)

    with pytest.raises(ValidationError, match=field_name):
        validate_portable_manifest(manifest)


def test_absent_defaulted_subclass_declaration_remains_valid() -> None:
    manifest = CompatibleDefaultManifest.model_validate(valid_manifest())
    manifest.__dict__.pop("defaulted_contract_field")

    revalidated = validate_portable_manifest(manifest)

    assert type(revalidated) is PortableManifest
    assert revalidated.manifest_id == manifest.manifest_id


def test_rejects_subclass_declared_field_moved_to_extra_storage() -> None:
    manifest = CompatibleDefaultManifest.model_validate(valid_manifest())
    default_value = manifest.__dict__.pop("defaulted_contract_field")
    object.__setattr__(
        manifest,
        "__pydantic_extra__",
        {"defaulted_contract_field": default_value},
    )

    with pytest.raises(
        ValidationError,
        match="contract model has conflicting stored and extra fields",
    ):
        validate_portable_manifest(manifest)


@pytest.mark.parametrize(
    "extra_storage",
    [None, {}, MappingProxyType({})],
    ids=["none", "empty-dict", "empty-custom-mapping"],
)
def test_none_and_empty_mapping_extra_storage_remain_valid(
    extra_storage: object,
) -> None:
    manifest = validate_portable_manifest(valid_manifest()).model_copy(deep=True)
    object.__setattr__(manifest.files[0], "__pydantic_extra__", extra_storage)

    revalidated = validate_portable_manifest(manifest)

    assert revalidated.files[0].path == manifest.files[0].path


def test_root_model_extra_override_cannot_hide_native_unknown() -> None:
    manifest = NoneReportingExtraManifest.model_validate(valid_manifest())
    native_extra = {"future_entry_semantics": "deny"}
    _PYDANTIC_EXTRA_SLOT.__set__(manifest, native_extra)

    assert manifest.__pydantic_extra__ is None
    assert _PYDANTIC_EXTRA_SLOT.__get__(manifest, type(manifest)) is native_extra
    with pytest.raises(ValidationError) as raised:
        validate_portable_manifest(manifest)

    errors = raised.value.errors()
    assert len(errors) == 1
    assert errors[0]["type"] == "extra_forbidden"
    assert errors[0]["loc"] == ("future_entry_semantics",)
    assert errors[0]["input"] == "deny"


def test_nested_model_extra_override_cannot_hide_native_mapping_unknown() -> None:
    manifest = validate_portable_manifest(valid_manifest()).model_copy(deep=True)
    nested = EmptyReportingExtraManifestFile.model_validate(
        manifest.files[0].model_dump()
    )
    native_extra = MappingProxyType({"future_entry_semantics": "deny"})
    _PYDANTIC_EXTRA_SLOT.__set__(nested, native_extra)
    manifest.files[0] = nested

    assert nested.__pydantic_extra__ == {}
    assert _PYDANTIC_EXTRA_SLOT.__get__(nested, type(nested)) is native_extra
    with pytest.raises(ValidationError) as raised:
        validate_portable_manifest(manifest)

    errors = raised.value.errors()
    assert len(errors) == 1
    assert errors[0]["type"] == "extra_forbidden"
    assert errors[0]["loc"] == ("files", 0, "future_entry_semantics")
    assert errors[0]["input"] == "deny"


@pytest.mark.parametrize("native_state", ["none", "absent"])
def test_reported_extra_does_not_replace_empty_native_storage(
    native_state: str,
) -> None:
    manifest = PopulatedReportingExtraManifest.model_validate(valid_manifest())
    if native_state == "none":
        _PYDANTIC_EXTRA_SLOT.__set__(manifest, None)
    else:
        _PYDANTIC_EXTRA_SLOT.__delete__(manifest)

    assert manifest.__pydantic_extra__ == {"reported_only": "deny"}
    revalidated = validate_portable_manifest(manifest)

    assert type(revalidated) is PortableManifest
    assert revalidated.manifest_id == "manifest-7"


def test_custom_mapping_extra_storage_preserves_unknown_for_rejection() -> None:
    manifest = validate_portable_manifest(valid_manifest()).model_copy(deep=True)
    object.__setattr__(
        manifest.files[0],
        "__pydantic_extra__",
        MappingProxyType({"future_entry_semantics": "deny"}),
    )

    with pytest.raises(ValidationError, match="future_entry_semantics") as raised:
        validate_portable_manifest(manifest)

    assert raised.value.errors()[0]["type"] == "extra_forbidden"


def test_root_fields_set_only_unknown_is_controlled_extra_error() -> None:
    manifest = validate_portable_manifest(valid_manifest()).model_copy(deep=True)
    manifest.__pydantic_fields_set__.add("phantom_only")

    assert "phantom_only" not in manifest.__dict__
    assert manifest.__pydantic_extra__ is None
    with pytest.raises(ValidationError) as raised:
        validate_portable_manifest(manifest)

    errors = raised.value.errors()
    assert len(errors) == 1
    assert errors[0]["type"] == "extra_forbidden"
    assert errors[0]["loc"] == ("phantom_only",)
    assert errors[0]["input"] is None


def test_nested_fields_set_only_unknown_is_controlled_extra_error() -> None:
    manifest = validate_portable_manifest(valid_manifest()).model_copy(deep=True)
    nested = manifest.files[0]
    nested.__pydantic_fields_set__.add("phantom_only")

    assert "phantom_only" not in nested.__dict__
    assert nested.__pydantic_extra__ is None
    with pytest.raises(ValidationError) as raised:
        validate_portable_manifest(manifest)

    errors = raised.value.errors()
    assert len(errors) == 1
    assert errors[0]["type"] == "extra_forbidden"
    assert errors[0]["loc"] == ("files", 0, "phantom_only")
    assert errors[0]["input"] is None


def test_model_fields_set_override_cannot_hide_root_unknown() -> None:
    ordinary = validate_portable_manifest(valid_manifest()).model_copy(deep=True)
    ordinary_fields_set = _PYDANTIC_FIELDS_SET_SLOT.__get__(
        ordinary, type(ordinary)
    )
    ordinary_fields_set.add("phantom_only")

    hidden = HidingFieldsSetManifest.model_validate(valid_manifest())
    hidden_fields_set = HiddenFieldsSetBacking(
        set.copy(_PYDANTIC_FIELDS_SET_SLOT.__get__(hidden, type(hidden)))
    )
    set.add(hidden_fields_set, "phantom_only")
    _PYDANTIC_FIELDS_SET_SLOT.__set__(hidden, hidden_fields_set)

    assert "phantom_only" in ordinary.model_fields_set
    assert set.__contains__(hidden_fields_set, "phantom_only")
    assert "phantom_only" not in hidden.model_fields_set

    observed_errors = []
    for model in (ordinary, hidden):
        with pytest.raises(ValidationError) as raised:
            validate_portable_manifest(model)
        observed_errors.append(raised.value.errors())

    assert observed_errors[0] == observed_errors[1]
    assert len(observed_errors[0]) == 1
    assert observed_errors[0][0]["type"] == "extra_forbidden"
    assert observed_errors[0][0]["loc"] == ("phantom_only",)
    assert observed_errors[0][0]["input"] is None
    assert hidden_fields_set.overridden_view_calls() == (0, 0, 0)


def test_model_fields_set_override_cannot_hide_nested_unknown() -> None:
    manifest = validate_portable_manifest(valid_manifest()).model_copy(deep=True)
    nested = HidingFieldsSetManifestFile.model_validate(
        manifest.files[0].model_dump()
    )
    native_fields_set = _PYDANTIC_FIELDS_SET_SLOT.__get__(nested, type(nested))
    native_fields_set.add("phantom_only")
    manifest.files[0] = nested

    assert "phantom_only" not in nested.model_fields_set
    assert set.__contains__(native_fields_set, "phantom_only")
    with pytest.raises(ValidationError) as raised:
        validate_portable_manifest(manifest)

    errors = raised.value.errors()
    assert len(errors) == 1
    assert errors[0]["type"] == "extra_forbidden"
    assert errors[0]["loc"] == ("files", 0, "phantom_only")
    assert errors[0]["input"] is None


@pytest.mark.parametrize("representation", ["frozenset", "tuple", "list"])
@pytest.mark.parametrize("with_unknown", [False, True])
def test_fields_set_iterable_storage_preserves_prior_behavior(
    representation: str,
    with_unknown: bool,
) -> None:
    manifest = validate_portable_manifest(valid_manifest()).model_copy(deep=True)
    native_fields_set = set.copy(
        _PYDANTIC_FIELDS_SET_SLOT.__get__(manifest, type(manifest))
    )
    if with_unknown:
        native_fields_set.add("phantom_only")
    if representation == "frozenset":
        replacement = frozenset(native_fields_set)
    elif representation == "tuple":
        replacement = tuple(native_fields_set)
    else:
        replacement = list(native_fields_set)
    _PYDANTIC_FIELDS_SET_SLOT.__set__(manifest, replacement)

    if with_unknown:
        with pytest.raises(ValidationError) as raised:
            validate_portable_manifest(manifest)
        errors = raised.value.errors()
        assert len(errors) == 1
        assert errors[0]["type"] == "extra_forbidden"
        assert errors[0]["loc"] == ("phantom_only",)
        assert errors[0]["input"] is None
    else:
        revalidated = validate_portable_manifest(manifest)
        assert revalidated.manifest_id == "manifest-7"


@pytest.mark.parametrize("location", ["manifest", "file"])
def test_fields_set_storage_states_use_controlled_public_results(
    location: str,
) -> None:
    intact = validate_portable_manifest(valid_manifest()).model_copy(deep=True)
    assert validate_portable_manifest(intact).manifest_id == "manifest-7"

    missing = validate_portable_manifest(valid_manifest()).model_copy(deep=True)
    missing_target = _model_at_location(missing, location)
    _PYDANTIC_FIELDS_SET_SLOT.__delete__(missing_target)
    with pytest.raises(ValidationError) as missing_error:
        validate_portable_manifest(missing)
    missing_errors = missing_error.value.errors(include_url=False)
    assert len(missing_errors) == 1
    assert missing_errors[0]["type"] == "value_error"
    assert missing_errors[0]["loc"] == ()
    assert "contract model fields-set storage is missing" in missing_errors[0][
        "msg"
    ]

    malformed = validate_portable_manifest(valid_manifest()).model_copy(deep=True)
    malformed_target = _model_at_location(malformed, location)
    _PYDANTIC_FIELDS_SET_SLOT.__set__(malformed_target, 0)
    with pytest.raises(ValidationError) as malformed_error:
        validate_portable_manifest(malformed)
    malformed_errors = malformed_error.value.errors(include_url=False)
    assert len(malformed_errors) == 1
    assert malformed_errors[0]["type"] == "value_error"
    assert malformed_errors[0]["loc"] == ()
    assert "'int' object is not iterable" in malformed_errors[0]["msg"]


@pytest.mark.parametrize("location", ["manifest", "file"])
def test_missing_fields_set_precedes_malformed_extra_storage(
    location: str,
) -> None:
    manifest = validate_portable_manifest(valid_manifest()).model_copy(deep=True)
    target = _model_at_location(manifest, location)
    _PYDANTIC_FIELDS_SET_SLOT.__delete__(target)
    _PYDANTIC_EXTRA_SLOT.__set__(target, [])

    with pytest.raises(
        ValidationError,
        match="contract model fields-set storage is missing",
    ):
        validate_portable_manifest(manifest)


@pytest.mark.parametrize(
    ("location", "field_name"),
    [("manifest", "manifest_id"), ("file", "path")],
)
def test_benign_string_subclass_field_names_remain_valid(
    location: str,
    field_name: str,
) -> None:
    manifest = validate_portable_manifest(valid_manifest()).model_copy(deep=True)
    target = _model_at_location(manifest, location)
    native_fields_set = set.copy(
        _PYDANTIC_FIELDS_SET_SLOT.__get__(target, type(target))
    )
    native_fields_set.remove(field_name)
    native_fields_set.add(BenignFieldName(field_name))
    _PYDANTIC_FIELDS_SET_SLOT.__set__(target, native_fields_set)

    revalidated = validate_portable_manifest(manifest)

    assert revalidated.manifest_id == "manifest-7"


def test_model_dict_descriptor_cannot_hide_root_unknown() -> None:
    ordinary = validate_portable_manifest(valid_manifest()).model_copy(deep=True)
    masked = DescriptorMaskedManifest.model_validate(valid_manifest())

    observed_errors = []
    for model in (ordinary, masked):
        native_stored = _PYDANTIC_DICT_DESCRIPTOR.__get__(model, BaseModel)
        dict.__setitem__(native_stored, "future_entry_semantics", "deny")
        assert dict.__getitem__(native_stored, "future_entry_semantics") == "deny"
        if model is masked:
            assert "future_entry_semantics" not in model.__dict__
        with pytest.raises(ValidationError) as raised:
            validate_portable_manifest(model)
        observed_errors.append(raised.value.errors())

    assert observed_errors[0] == observed_errors[1]
    assert len(observed_errors[0]) == 1
    assert observed_errors[0][0]["type"] == "extra_forbidden"
    assert observed_errors[0][0]["loc"] == ("future_entry_semantics",)
    assert observed_errors[0][0]["input"] == "deny"


def test_model_dict_descriptor_cannot_hide_nested_unknown() -> None:
    ordinary_manifest = validate_portable_manifest(valid_manifest()).model_copy(
        deep=True
    )
    masked_manifest = validate_portable_manifest(valid_manifest()).model_copy(
        deep=True
    )
    nested = DescriptorMaskedManifestFile.model_validate(
        masked_manifest.files[0].model_dump()
    )
    masked_manifest.files[0] = nested

    observed_errors = []
    for manifest in (ordinary_manifest, masked_manifest):
        node = manifest.files[0]
        native_stored = _PYDANTIC_DICT_DESCRIPTOR.__get__(node, BaseModel)
        dict.__setitem__(native_stored, "future_entry_semantics", "deny")
        assert dict.__getitem__(native_stored, "future_entry_semantics") == "deny"
        if node is nested:
            assert "future_entry_semantics" not in node.__dict__
        with pytest.raises(ValidationError) as raised:
            validate_portable_manifest(manifest)
        observed_errors.append(raised.value.errors())

    assert observed_errors[0] == observed_errors[1]
    assert len(observed_errors[0]) == 1
    assert observed_errors[0][0]["type"] == "extra_forbidden"
    assert observed_errors[0][0]["loc"] == (
        "files",
        0,
        "future_entry_semantics",
    )
    assert observed_errors[0][0]["input"] == "deny"


@pytest.mark.parametrize("location", ["manifest", "file"])
def test_model_dict_descriptor_without_unknown_remains_valid(
    location: str,
) -> None:
    if location == "manifest":
        manifest = DescriptorMaskedManifest.model_validate(valid_manifest())
    else:
        manifest = validate_portable_manifest(valid_manifest()).model_copy(deep=True)
        manifest.files[0] = DescriptorMaskedManifestFile.model_validate(
            manifest.files[0].model_dump()
        )

    revalidated = validate_portable_manifest(manifest)

    assert type(revalidated) is PortableManifest
    assert revalidated.files[0].path == "canonical/records.jsonl"


@pytest.mark.parametrize(
    ("location", "expected_location"),
    [
        ("manifest", ("hidden_unknown",)),
        ("file", ("files", 0, "hidden_unknown")),
    ],
)
def test_hidden_native_model_backing_unknown_is_not_discarded(
    location: str,
    expected_location: tuple[object, ...],
) -> None:
    manifest = validate_portable_manifest(valid_manifest()).model_copy(deep=True)
    node = _model_at_location(manifest, location)
    native_stored = object.__getattribute__(node, "__dict__")
    hidden = HiddenModelBacking(
        tuple(dict.items(native_stored)),
        hidden_name="hidden_unknown",
    )
    dict.__setitem__(hidden, "hidden_unknown", "deny")
    object.__setattr__(node, "__dict__", hidden)

    with pytest.raises(ValidationError) as raised:
        validate_portable_manifest(manifest)

    assert hidden.overridden_view_calls() == (0, 0, 0, 0, 0)
    errors = raised.value.errors()
    assert len(errors) == 1
    assert errors[0]["type"] == "extra_forbidden"
    assert errors[0]["loc"] == expected_location
    assert errors[0]["input"] == "deny"


def test_native_model_backing_subclass_without_hidden_state_remains_valid() -> None:
    manifest = validate_portable_manifest(valid_manifest()).model_copy(deep=True)
    native_stored = object.__getattribute__(manifest.files[0], "__dict__")
    backing = HiddenModelBacking(tuple(dict.items(native_stored)))
    object.__setattr__(manifest.files[0], "__dict__", backing)

    revalidated = validate_portable_manifest(manifest)

    assert revalidated.files[0].path == manifest.files[0].path
    assert backing.overridden_view_calls() == (0, 0, 0, 0, 0)


def test_nested_mapping_cannot_repair_initially_invalid_model_storage() -> None:
    manifest = validate_portable_manifest(valid_manifest()).model_copy(deep=True)
    native_stored = object.__getattribute__(manifest, "__dict__")
    dict.__setitem__(native_stored, "manifest_id", "")
    repairing_producer = RepairingProducer(manifest)
    dict.__setitem__(native_stored, "producer", repairing_producer)

    with pytest.raises(ValidationError) as raised:
        validate_portable_manifest(manifest)

    assert raised.value.errors()[0]["loc"] == ("manifest_id",)
    assert repairing_producer.item_calls == 2


def test_nested_mapping_with_initially_valid_model_storage_remains_valid() -> None:
    manifest = validate_portable_manifest(valid_manifest()).model_copy(deep=True)
    repairing_producer = RepairingProducer(manifest)
    native_stored = object.__getattribute__(manifest, "__dict__")
    dict.__setitem__(native_stored, "producer", repairing_producer)

    revalidated = validate_portable_manifest(manifest)

    assert revalidated.manifest_id == "manifest-7"
    assert revalidated.producer.implementation == "reference-exporter"
    assert repairing_producer.item_calls == 2


@pytest.mark.parametrize(
    ("location", "field_name", "expected_location"),
    [
        ("manifest", "manifest_id", ("manifest_id",)),
        ("file", "path", ("files", 0, "path")),
    ],
)
def test_ordinary_initially_invalid_model_storage_remains_rejected(
    location: str,
    field_name: str,
    expected_location: tuple[object, ...],
) -> None:
    manifest = validate_portable_manifest(valid_manifest()).model_copy(deep=True)
    target = _model_at_location(manifest, location)
    native_stored = _PYDANTIC_DICT_DESCRIPTOR.__get__(target, BaseModel)
    dict.__setitem__(native_stored, field_name, "")

    with pytest.raises(ValidationError) as raised:
        validate_portable_manifest(manifest)

    assert raised.value.errors()[0]["loc"] == expected_location


@pytest.mark.parametrize("hook_kind", ["fields-set", "mapping"])
@pytest.mark.parametrize(
    ("location", "field_name", "valid_value", "expected_location"),
    [
        ("manifest", "manifest_id", "manifest-7", ("manifest_id",)),
        ("file", "path", "canonical/records.jsonl", ("files", 0, "path")),
    ],
)
@pytest.mark.parametrize("initially_valid", [False, True])
def test_supported_hook_cannot_repair_frozen_model_storage(
    hook_kind: str,
    location: str,
    field_name: str,
    valid_value: str,
    expected_location: tuple[object, ...],
    initially_valid: bool,
) -> None:
    manifest = validate_portable_manifest(valid_manifest()).model_copy(deep=True)
    target = _model_at_location(manifest, location)
    native_target = _PYDANTIC_DICT_DESCRIPTOR.__get__(target, BaseModel)
    if not initially_valid:
        dict.__setitem__(native_target, field_name, "")

    if hook_kind == "fields-set":
        native_fields_set = _PYDANTIC_FIELDS_SET_SLOT.__get__(
            manifest, type(manifest)
        )
        hook = RepairingFieldsSetIterable(
            tuple(set.__iter__(native_fields_set)),
            target,
            field_name,
            valid_value,
        )
        _PYDANTIC_FIELDS_SET_SLOT.__set__(manifest, hook)
    else:
        hook = RepairingProducerMapping(target, field_name, valid_value)
        native_manifest = _PYDANTIC_DICT_DESCRIPTOR.__get__(manifest, BaseModel)
        dict.__setitem__(native_manifest, "producer", hook)

    if initially_valid:
        revalidated = validate_portable_manifest(manifest)
        assert revalidated.manifest_id == "manifest-7"
    else:
        with pytest.raises(ValidationError) as raised:
            validate_portable_manifest(manifest)
        assert raised.value.errors()[0]["loc"] == expected_location

    assert dict.__getitem__(native_target, field_name) == valid_value
    if hook_kind == "fields-set":
        assert hook.iteration_calls == 1
    else:
        assert hook.item_calls == 2


@pytest.mark.parametrize(
    ("location", "field_name", "valid_value", "expected_location"),
    [
        ("manifest", "manifest_id", "manifest-7", ("manifest_id",)),
        ("file", "path", "canonical/records.jsonl", ("files", 0, "path")),
    ],
)
@pytest.mark.parametrize("initially_valid", [False, True])
def test_benign_field_name_hash_cannot_repair_frozen_model_storage(
    location: str,
    field_name: str,
    valid_value: str,
    expected_location: tuple[object, ...],
    initially_valid: bool,
) -> None:
    manifest = validate_portable_manifest(valid_manifest()).model_copy(deep=True)
    target = _model_at_location(manifest, location)
    native_target = _PYDANTIC_DICT_DESCRIPTOR.__get__(target, BaseModel)
    if not initially_valid:
        dict.__setitem__(native_target, field_name, "")

    native_fields_set = _PYDANTIC_FIELDS_SET_SLOT.__get__(
        manifest, type(manifest)
    )
    field_names = [
        name for name in set.__iter__(native_fields_set) if name != "manifest_id"
    ]
    repairing_name = RepairingFieldName(
        "manifest_id",
        target,
        field_name,
        valid_value,
    )
    field_names.append(repairing_name)
    _PYDANTIC_FIELDS_SET_SLOT.__set__(manifest, field_names)

    if initially_valid:
        revalidated = validate_portable_manifest(manifest)
        assert revalidated.manifest_id == "manifest-7"
    else:
        with pytest.raises(ValidationError) as raised:
            validate_portable_manifest(manifest)
        assert raised.value.errors()[0]["loc"] == expected_location

    assert repairing_name.hash_calls >= 1
    assert dict.__getitem__(native_target, field_name) == valid_value


@pytest.mark.parametrize("competing_error", ["overlap", "unknown", "cycle"])
def test_frozen_nested_state_preserves_existing_error_priority(
    competing_error: str,
) -> None:
    manifest = validate_portable_manifest(valid_manifest()).model_copy(deep=True)
    nested = manifest.files[0]
    native_nested = _PYDANTIC_DICT_DESCRIPTOR.__get__(nested, BaseModel)
    dict.__setitem__(native_nested, "path", "")
    native_fields_set = _PYDANTIC_FIELDS_SET_SLOT.__get__(
        manifest, type(manifest)
    )
    repair = RepairingFieldsSetIterable(
        tuple(set.__iter__(native_fields_set)),
        nested,
        "path",
        "canonical/records.jsonl",
    )
    _PYDANTIC_FIELDS_SET_SLOT.__set__(manifest, repair)

    if competing_error == "overlap":
        _PYDANTIC_EXTRA_SLOT.__set__(manifest, {"manifest_id": "shadow"})
        expected = "contract model has conflicting stored and extra fields"
    elif competing_error == "unknown":
        native_manifest = _PYDANTIC_DICT_DESCRIPTOR.__get__(manifest, BaseModel)
        dict.__setitem__(native_manifest, "future_entry_semantics", "deny")
        expected = "future_entry_semantics"
    else:
        cyclic_capabilities: list[object] = []
        cyclic_capabilities.append(cyclic_capabilities)
        native_manifest = _PYDANTIC_DICT_DESCRIPTOR.__get__(manifest, BaseModel)
        dict.__setitem__(
            native_manifest,
            "included_capabilities",
            cyclic_capabilities,
        )
        expected = "contract input graph must be acyclic"

    with pytest.raises(ValidationError, match=expected):
        validate_portable_manifest(manifest)

    assert repair.iteration_calls == 1
    assert dict.__getitem__(native_nested, "path") == "canonical/records.jsonl"


@pytest.mark.parametrize(
    ("location", "field_name", "shadow_value"),
    [
        ("manifest", "manifest_id", "shadow-manifest"),
        ("file", "path", "canonical/shadow-records.jsonl"),
        ("replay_position", "position", "shadow-position"),
    ],
)
def test_split_extra_views_cannot_overwrite_declared_fields(
    location: str,
    field_name: str,
    shadow_value: str,
) -> None:
    manifest = validate_portable_manifest(valid_manifest()).model_copy(deep=True)
    node = _model_at_location(manifest, location)
    split_view = SplitViewExtra(
        ((field_name, shadow_value),),
        reported_keys=(field_name,),
    )
    object.__setattr__(node, "__pydantic_extra__", split_view)

    with pytest.raises(ValidationError) as raised:
        validate_portable_manifest(manifest)
    assert split_view.iteration_calls == 1
    assert split_view.item_calls == 1
    assert "contract model has conflicting stored and extra fields" in str(
        raised.value
    )


def test_items_only_unknown_extra_is_not_silently_discarded() -> None:
    manifest = validate_portable_manifest(valid_manifest()).model_copy(deep=True)
    split_view = SplitViewExtra((("future_entry_semantics", "deny"),))
    object.__setattr__(manifest, "__pydantic_extra__", split_view)

    with pytest.raises(ValidationError, match="future_entry_semantics") as raised:
        validate_portable_manifest(manifest)

    assert split_view.iteration_calls == 1
    assert split_view.item_calls == 1
    assert raised.value.errors()[0]["type"] == "extra_forbidden"


def test_populated_falsey_extra_storage_is_not_silently_discarded() -> None:
    manifest = validate_portable_manifest(valid_manifest()).model_copy(deep=True)
    falsey_populated = SplitViewExtra((("future_entry_semantics", "deny"),))
    assert not falsey_populated
    object.__setattr__(manifest.files[0], "__pydantic_extra__", falsey_populated)

    with pytest.raises(ValidationError, match="future_entry_semantics"):
        validate_portable_manifest(manifest)
    assert falsey_populated.iteration_calls == 1
    assert falsey_populated.item_calls == 1


def test_extra_items_are_captured_once_for_validation_and_reconstruction() -> None:
    manifest = validate_portable_manifest(valid_manifest()).model_copy(deep=True)
    changing = ChangingItemsExtra()
    object.__setattr__(manifest, "__pydantic_extra__", changing)

    revalidated = validate_portable_manifest(manifest)

    assert revalidated.manifest_id == "manifest-7"
    assert changing.iteration_calls == 1
    assert changing.item_calls == 1


def test_stable_empty_custom_extra_snapshot_remains_valid() -> None:
    manifest = validate_portable_manifest(valid_manifest()).model_copy(deep=True)
    stable_empty = SplitViewExtra(())
    object.__setattr__(manifest.files[0], "__pydantic_extra__", stable_empty)

    revalidated = validate_portable_manifest(manifest)

    assert revalidated.files[0].path == manifest.files[0].path
    assert stable_empty.iteration_calls == 1
    assert stable_empty.item_calls == 1


@pytest.mark.parametrize(
    ("location", "extra_type", "expected_location"),
    [
        ("manifest", HiddenBackingExtra, ("future_entry_semantics",)),
        (
            "file",
            InheritedHiddenBackingExtra,
            ("files", 0, "future_entry_semantics"),
        ),
    ],
)
def test_hidden_native_dict_backing_unknown_is_not_discarded(
    location: str,
    extra_type: type[HiddenBackingExtra],
    expected_location: tuple[object, ...],
) -> None:
    manifest = validate_portable_manifest(valid_manifest()).model_copy(deep=True)
    node = _model_at_location(manifest, location)
    hidden = extra_type({"future_entry_semantics": "deny"})
    object.__setattr__(node, "__pydantic_extra__", hidden)

    with pytest.raises(ValidationError) as raised:
        validate_portable_manifest(manifest)

    assert hidden.overridden_view_calls() == (0, 0, 0, 0, 0, 0)
    errors = raised.value.errors()
    assert len(errors) == 1
    assert errors[0]["type"] == "extra_forbidden"
    assert errors[0]["loc"] == expected_location
    assert errors[0]["input"] == "deny"


@pytest.mark.parametrize(
    ("location", "extra_type", "field_name", "shadow_value"),
    [
        (
            "manifest",
            HiddenBackingExtra,
            "manifest_id",
            "shadow-manifest",
        ),
        (
            "file",
            InheritedHiddenBackingExtra,
            "path",
            "canonical/shadow-records.jsonl",
        ),
    ],
)
def test_hidden_native_dict_backing_declared_overlap_is_rejected(
    location: str,
    extra_type: type[HiddenBackingExtra],
    field_name: str,
    shadow_value: str,
) -> None:
    manifest = validate_portable_manifest(valid_manifest()).model_copy(deep=True)
    node = _model_at_location(manifest, location)
    hidden = extra_type({field_name: shadow_value})
    object.__setattr__(node, "__pydantic_extra__", hidden)

    with pytest.raises(ValidationError) as raised:
        validate_portable_manifest(manifest)

    assert hidden.overridden_view_calls() == (0, 0, 0, 0, 0, 0)
    assert "contract model has conflicting stored and extra fields" in str(
        raised.value
    )


def test_empty_hidden_native_dict_subclass_remains_valid() -> None:
    manifest = validate_portable_manifest(valid_manifest()).model_copy(deep=True)
    hidden = InheritedHiddenBackingExtra({})
    object.__setattr__(manifest.files[0], "__pydantic_extra__", hidden)

    revalidated = validate_portable_manifest(manifest)

    assert revalidated.files[0].path == manifest.files[0].path
    assert hidden.overridden_view_calls() == (0, 0, 0, 0, 0, 0)


def _assert_concrete_extra_dispatch_counts(
    extra: RaisingClassHiddenBackingExtra | SpoofedDictClassMapping,
) -> None:
    assert extra.class_calls == 0
    if type(extra) is RaisingClassHiddenBackingExtra:
        assert extra.overridden_view_calls() == (0, 0, 0, 0, 0, 0)
    else:
        assert extra.iteration_calls == 1
        assert extra.item_calls == 1


@pytest.mark.parametrize(
    "extra_type",
    [RaisingClassHiddenBackingExtra, SpoofedDictClassMapping],
)
def test_concrete_extra_dispatch_preserves_empty_storage(
    extra_type: type[RaisingClassHiddenBackingExtra]
    | type[SpoofedDictClassMapping],
) -> None:
    manifest = validate_portable_manifest(valid_manifest()).model_copy(deep=True)
    extra = extra_type({})
    object.__setattr__(manifest, "__pydantic_extra__", extra)

    revalidated = validate_portable_manifest(manifest)

    assert revalidated.manifest_id == manifest.manifest_id
    _assert_concrete_extra_dispatch_counts(extra)


@pytest.mark.parametrize(
    ("location", "expected_location"),
    [
        ("manifest", ("future_entry_semantics",)),
        ("file", ("files", 0, "future_entry_semantics")),
    ],
)
@pytest.mark.parametrize(
    "extra_type",
    [RaisingClassHiddenBackingExtra, SpoofedDictClassMapping],
)
def test_concrete_extra_dispatch_rejects_unknown_storage(
    location: str,
    expected_location: tuple[object, ...],
    extra_type: type[RaisingClassHiddenBackingExtra]
    | type[SpoofedDictClassMapping],
) -> None:
    manifest = validate_portable_manifest(valid_manifest()).model_copy(deep=True)
    node = _model_at_location(manifest, location)
    extra = extra_type({"future_entry_semantics": "deny"})
    object.__setattr__(node, "__pydantic_extra__", extra)

    with pytest.raises(ValidationError) as raised:
        validate_portable_manifest(manifest)

    _assert_concrete_extra_dispatch_counts(extra)
    errors = raised.value.errors()
    assert len(errors) == 1
    assert errors[0]["type"] == "extra_forbidden"
    assert errors[0]["loc"] == expected_location
    assert errors[0]["input"] == "deny"


@pytest.mark.parametrize(
    ("location", "field_name", "shadow_value"),
    [
        ("manifest", "manifest_id", "shadow-manifest"),
        ("file", "path", "canonical/shadow-records.jsonl"),
    ],
)
@pytest.mark.parametrize(
    "extra_type",
    [RaisingClassHiddenBackingExtra, SpoofedDictClassMapping],
)
def test_concrete_extra_dispatch_rejects_declared_overlap(
    location: str,
    field_name: str,
    shadow_value: str,
    extra_type: type[RaisingClassHiddenBackingExtra]
    | type[SpoofedDictClassMapping],
) -> None:
    manifest = validate_portable_manifest(valid_manifest()).model_copy(deep=True)
    node = _model_at_location(manifest, location)
    extra = extra_type({field_name: shadow_value})
    object.__setattr__(node, "__pydantic_extra__", extra)

    with pytest.raises(ValidationError) as raised:
        validate_portable_manifest(manifest)

    _assert_concrete_extra_dispatch_counts(extra)
    assert "contract model has conflicting stored and extra fields" in str(
        raised.value
    )


@pytest.mark.parametrize(
    ("location", "expected_location"),
    [
        ("manifest", ("future_entry_semantics",)),
        ("file", ("files", 0, "future_entry_semantics")),
    ],
)
def test_iteration_only_unknown_extra_is_not_silently_discarded(
    location: str,
    expected_location: tuple[object, ...],
) -> None:
    manifest = validate_portable_manifest(valid_manifest()).model_copy(deep=True)
    node = _model_at_location(manifest, location)
    inverse = InverseItemsExtra("future_entry_semantics", "deny")
    object.__setattr__(node, "__pydantic_extra__", inverse)

    with pytest.raises(ValidationError) as raised:
        validate_portable_manifest(manifest)

    assert inverse.iteration_calls == 1
    assert inverse.item_calls == 1
    errors = raised.value.errors()
    assert len(errors) == 1
    assert errors[0]["type"] == "extra_forbidden"
    assert errors[0]["loc"] == expected_location
    assert errors[0]["input"] is None


@pytest.mark.parametrize(
    ("location", "field_name", "shadow_value"),
    [
        ("manifest", "manifest_id", "shadow-manifest"),
        ("file", "path", "canonical/shadow-records.jsonl"),
    ],
)
def test_iteration_only_declared_conflict_is_rejected(
    location: str,
    field_name: str,
    shadow_value: str,
) -> None:
    manifest = validate_portable_manifest(valid_manifest()).model_copy(deep=True)
    node = _model_at_location(manifest, location)
    inverse = InverseItemsExtra(field_name, shadow_value)
    object.__setattr__(node, "__pydantic_extra__", inverse)

    with pytest.raises(ValidationError) as raised:
        validate_portable_manifest(manifest)

    assert inverse.iteration_calls == 1
    assert inverse.item_calls == 1
    assert "contract model has conflicting stored and extra fields" in str(
        raised.value
    )


@pytest.mark.parametrize("location", ["manifest", "file"])
def test_iteration_only_subclass_default_moved_to_extra_is_rejected(
    location: str,
) -> None:
    if location == "manifest":
        manifest = CompatibleDefaultManifest.model_validate(valid_manifest())
        node = manifest
    else:
        manifest = validate_portable_manifest(valid_manifest()).model_copy(deep=True)
        node = CompatibleDefaultManifestFile.model_validate(
            manifest.files[0].model_dump()
        )
        manifest.files[0] = node
    default_value = node.__dict__.pop("defaulted_contract_field")
    inverse = InverseItemsExtra("defaulted_contract_field", default_value)
    object.__setattr__(node, "__pydantic_extra__", inverse)

    with pytest.raises(ValidationError) as raised:
        validate_portable_manifest(manifest)

    assert inverse.iteration_calls == 1
    assert inverse.item_calls == 1
    assert "contract model has conflicting stored and extra fields" in str(
        raised.value
    )


@pytest.mark.parametrize(
    ("location", "field_name", "original_value"),
    [
        ("manifest", "manifest_id", "manifest-7"),
        ("file", "path", "canonical/records.jsonl"),
    ],
)
def test_iteration_only_required_field_moved_to_extra_remains_rejected(
    location: str,
    field_name: str,
    original_value: str,
) -> None:
    manifest = validate_portable_manifest(valid_manifest()).model_copy(deep=True)
    node = _model_at_location(manifest, location)
    assert node.__dict__.pop(field_name) == original_value
    inverse = InverseItemsExtra(field_name, original_value)
    object.__setattr__(node, "__pydantic_extra__", inverse)

    with pytest.raises(ValidationError):
        validate_portable_manifest(manifest)

    assert inverse.iteration_calls == 1
    assert inverse.item_calls == 1


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
