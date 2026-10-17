"""Behavioral and adversarial tests for portable manifest validation."""

import gc
import itertools
import json
import sys
import weakref
from collections.abc import ItemsView, Iterator, Mapping
from copy import deepcopy
from pathlib import PureWindowsPath
from types import MappingProxyType
from typing import Any

import pytest
from pydantic import BaseModel, Field, ValidationError

import contracts_portability
from contracts_limits import (
    DISCOVERY_BUDGET_MESSAGE,
    PORTABILITY_MAX_DEPTH,
    PORTABILITY_MAX_EDGES,
    PORTABILITY_MAX_NODES,
    DiscoveryBudget,
    DiscoveryTier,
)

from contracts_portability import (
    ChecksumDescriptor,
    EncryptionReference,
    ExportScope,
    ManifestFile,
    PortableManifest,
    ProducerReference,
    _model_discovery_values,
    _validated_model_discovery_snapshot,
    _validated_model_storage,
    _validated_native_container_items,
    normalize_bundle_path,
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


class ReplacingFilesFieldsSetIterable:
    def __init__(
        self,
        names: tuple[str, ...],
        files: list[ManifestFile],
        replacement: ManifestFile | None,
    ) -> None:
        self.names = names
        self.files = files
        self.replacement = replacement
        self.iteration_calls = 0
        self.mutated = False

    def __iter__(self) -> Iterator[str]:
        self.iteration_calls += 1
        if not self.mutated:
            if self.replacement is None:
                list.__delitem__(self.files, 0)
            else:
                list.__setitem__(self.files, 0, self.replacement)
            self.mutated = True
        return iter(self.names)


class ReplacingFilesProducerMapping(Mapping[str, str]):
    def __init__(
        self,
        files: list[ManifestFile],
        replacement: ManifestFile | None,
    ) -> None:
        self.entries = {
            "implementation": "reference-exporter",
            "version": "1.4.2",
        }
        self.files = files
        self.replacement = replacement
        self.item_calls = 0
        self.mutated = False

    def __getitem__(self, key: str) -> str:
        return self.entries[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.entries)

    def __len__(self) -> int:
        return len(self.entries)

    def items(self) -> tuple[tuple[str, str], ...]:
        self.item_calls += 1
        if not self.mutated:
            if self.replacement is None:
                list.__delitem__(self.files, 0)
            else:
                list.__setitem__(self.files, 0, self.replacement)
            self.mutated = True
        return tuple(self.entries.items())


class DivergentFilesList(list[ManifestFile]):
    def __init__(
        self,
        native_items: tuple[ManifestFile, ...],
        public_items: tuple[ManifestFile, ...],
    ) -> None:
        list.__init__(self, native_items)
        self.public_items = public_items
        self.iteration_calls = 0

    def __iter__(self) -> Iterator[ManifestFile]:
        self.iteration_calls += 1
        return iter(self.public_items)


class LateExactContainerFilesList(list[ManifestFile]):
    def __init__(
        self,
        public_items: tuple[ManifestFile, ...],
        late_container_kind: str,
    ) -> None:
        list.__init__(self, public_items)
        self.public_items = public_items
        self.late_container_kind = late_container_kind
        self.iteration_calls = 0

    def __iter__(self) -> Iterator[object]:
        self.iteration_calls += 1
        if self.iteration_calls == 1:
            return iter(self.public_items)
        if self.late_container_kind == "dict":
            late_container: object = self.public_items[0].model_dump(mode="python")
        else:
            late_container = [self.public_items[0]]
        return iter((late_container, *self.public_items[1:]))


class TransientExactContainerFilesList(list[ManifestFile]):
    def __init__(self, public_items: tuple[ManifestFile, ...]) -> None:
        list.__init__(self, public_items)
        self.public_items = public_items
        self.iteration_calls = 0
        self.first_id: int | None = None
        self.second_id: int | None = None

    def __iter__(self) -> Iterator[object]:
        self.iteration_calls += 1
        transient = self.public_items[0].model_dump(mode="python")
        if self.iteration_calls == 1:
            self.first_id = id(transient)
        else:
            transient["path"] = ""
            self.second_id = id(transient)
        return iter((transient, *self.public_items[1:]))


class TransientModelFilesList(list[ManifestFile]):
    def __init__(
        self,
        public_items: tuple[ManifestFile, ...],
        replacement_path: str,
    ) -> None:
        list.__init__(self, public_items)
        self.public_items = public_items
        self.replacement_path = replacement_path
        self.iteration_calls = 0
        self.first_reference: weakref.ReferenceType[ManifestFile] | None = None
        self.first_id: int | None = None
        self.second_id: int | None = None
        self.first_live_during_walk: bool | None = None
        self.first_live_during_rebuild: bool | None = None

    def __iter__(self) -> Iterator[ManifestFile]:
        self.iteration_calls += 1
        if self.iteration_calls == 1:
            transient = self.public_items[0].model_copy(deep=True)
            self.first_reference = weakref.ref(transient)
            self.first_id = id(transient)
            yield transient
            del transient
            yield self.public_items[1]
            gc.collect()
            self.first_live_during_walk = self.first_reference() is not None
            return

        gc.collect()
        self.first_live_during_rebuild = self.first_reference() is not None
        replacement = self.public_items[0].model_copy(
            deep=True,
            update={"path": self.replacement_path},
        )
        self.second_id = id(replacement)
        yield replacement
        yield self.public_items[1]


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


class InconsistentHashStoredName(str):
    def __new__(cls, value: str) -> "InconsistentHashStoredName":
        instance = super().__new__(cls, value)
        instance.armed = False
        instance.hash_calls = 0
        instance.equality_calls = 0
        instance.class_calls = 0
        return instance

    def __hash__(self) -> int:
        if self.armed:
            self.hash_calls += 1
        return str.__hash__(self) ^ 0x40000000

    def __eq__(self, other: object) -> bool:
        if self.armed:
            self.equality_calls += 1
        return str.__eq__(self, other)

    @property
    def __class__(self) -> type[str]:
        if self.armed:
            self.class_calls += 1
        return str


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


class ReplacingFilesFieldName(str):
    def __new__(
        cls,
        value: str,
        files: list[ManifestFile],
        replacement: ManifestFile | None,
    ) -> "ReplacingFilesFieldName":
        instance = super().__new__(cls, value)
        instance.files = files
        instance.replacement = replacement
        instance.hash_calls = 0
        instance.mutated = False
        return instance

    def __hash__(self) -> int:
        self.hash_calls += 1
        if not self.mutated:
            if self.replacement is None:
                list.__delitem__(self.files, 0)
            else:
                list.__setitem__(self.files, 0, self.replacement)
            self.mutated = True
        return str.__hash__(self)


class ReplacingFilesStoredName(str):
    def __new__(
        cls,
        value: str,
        files: list[ManifestFile],
        replacement: ManifestFile | None,
    ) -> "ReplacingFilesStoredName":
        instance = super().__new__(cls, value)
        instance.files = files
        instance.replacement = replacement
        instance.armed = False
        instance.hash_calls = 0
        instance.mutated = False
        return instance

    def __hash__(self) -> int:
        self.hash_calls += 1
        if self.armed and not self.mutated:
            if self.replacement is None:
                list.__delitem__(self.files, 0)
            else:
                list.__setitem__(self.files, 0, self.replacement)
            self.mutated = True
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
    assert repairing_producer.item_calls == 1  # a public view is observed once and replayed, never re-read


def test_nested_mapping_with_initially_valid_model_storage_remains_valid() -> None:
    manifest = validate_portable_manifest(valid_manifest()).model_copy(deep=True)
    repairing_producer = RepairingProducer(manifest)
    native_stored = object.__getattribute__(manifest, "__dict__")
    dict.__setitem__(native_stored, "producer", repairing_producer)

    revalidated = validate_portable_manifest(manifest)

    assert revalidated.manifest_id == "manifest-7"
    assert revalidated.producer.implementation == "reference-exporter"
    assert repairing_producer.item_calls == 1  # a public view is observed once and replayed, never re-read


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
        assert hook.item_calls == 1  # a public view is observed once and replayed, never re-read


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

    # The name is read as exact text when the fields-set is captured, so its hash never runs: nothing can repair
    # the storage (it ran at least once before).
    assert repairing_name.hash_calls == 0
    assert dict.__getitem__(native_target, field_name) == (valid_value if initially_valid else "")


@pytest.mark.parametrize("hook_kind", ["fields-set", "mapping", "stored-key"])
@pytest.mark.parametrize("edge_action", ["replace", "remove"])
@pytest.mark.parametrize("initially_valid", [False, True])
def test_supported_hook_cannot_change_frozen_native_list_edge(
    hook_kind: str,
    edge_action: str,
    initially_valid: bool,
) -> None:
    manifest = validate_portable_manifest(valid_manifest()).model_copy(deep=True)
    files = manifest.files
    valid_file = files[0]
    invalid_file = valid_file.model_copy(update={"path": ""})
    original = valid_file if initially_valid else invalid_file
    replacement = (
        invalid_file if initially_valid else valid_file
    )
    if edge_action == "remove":
        replacement = None
    list.__setitem__(files, 0, original)

    if hook_kind == "fields-set":
        native_fields_set = _PYDANTIC_FIELDS_SET_SLOT.__get__(
            manifest, type(manifest)
        )
        hook = ReplacingFilesFieldsSetIterable(
            tuple(set.__iter__(native_fields_set)),
            files,
            replacement,
        )
        _PYDANTIC_FIELDS_SET_SLOT.__set__(manifest, hook)
    elif hook_kind == "mapping":
        hook = ReplacingFilesProducerMapping(files, replacement)
        native_manifest = _PYDANTIC_DICT_DESCRIPTOR.__get__(manifest, BaseModel)
        dict.__setitem__(native_manifest, "producer", hook)
    else:
        native_fields_set = _PYDANTIC_FIELDS_SET_SLOT.__get__(
            manifest, type(manifest)
        )
        field_names = [
            name
            for name in set.__iter__(native_fields_set)
            if name != "manifest_id"
        ]
        hook = ReplacingFilesFieldName("manifest_id", files, replacement)
        field_names.append(hook)
        _PYDANTIC_FIELDS_SET_SLOT.__set__(manifest, field_names)

    if initially_valid:
        revalidated = validate_portable_manifest(manifest)
        assert revalidated.files[0].path == "canonical/records.jsonl"
    else:
        with pytest.raises(ValidationError) as raised:
            validate_portable_manifest(manifest)
        assert raised.value.errors()[0]["type"] == "string_too_short"
        assert raised.value.errors()[0]["loc"] == ("files", 0, "path")

    if hook_kind == "stored-key":
        # A fields-set name is read as exact text at capture: its hook never runs, so the list is untouched.
        assert len(files) == 2 and list.__getitem__(files, 0) is original
    elif replacement is None:
        assert len(files) == 1
        assert list.__getitem__(files, 0).path == "crypto/envelope.bin"
    else:
        assert list.__getitem__(files, 0) is replacement
    if hook_kind == "fields-set":
        assert hook.iteration_calls == 1
    elif hook_kind == "mapping":
        assert hook.item_calls == 1  # a public view is observed once and replayed, never re-read
    else:
        assert hook.hash_calls == 0


@pytest.mark.parametrize(
    "edge_case",
    ["invalid-repaired", "valid-corrupted", "valid-removed"],
)
def test_stored_name_callback_cannot_change_frozen_native_list_edge(
    edge_case: str,
) -> None:
    manifest = validate_portable_manifest(valid_manifest()).model_copy(deep=True)
    files = manifest.files
    valid_file = list.__getitem__(files, 0)
    invalid_file = valid_file.model_copy(update={"path": ""})
    if edge_case == "invalid-repaired":
        original, replacement = invalid_file, valid_file
    elif edge_case == "valid-corrupted":
        original, replacement = valid_file, invalid_file
    else:
        original, replacement = valid_file, None
    list.__setitem__(files, 0, original)

    hook = ReplacingFilesStoredName("schema_version", files, replacement)
    native_manifest = _PYDANTIC_DICT_DESCRIPTOR.__get__(manifest, BaseModel)
    stored_entries = tuple(dict.items(native_manifest))
    dict.clear(native_manifest)
    for name, item in stored_entries:
        dict.__setitem__(
            native_manifest,
            hook if name == "schema_version" else name,
            item,
        )
    hook.armed = True
    hash_calls_before_validation = hook.hash_calls  # the test's own insertion into the native storage

    if edge_case == "invalid-repaired":
        with pytest.raises(ValidationError) as raised:
            validate_portable_manifest(manifest)
        assert raised.value.errors()[0]["type"] == "string_too_short"
        assert raised.value.errors()[0]["loc"] == ("files", 0, "path")
    else:
        revalidated = validate_portable_manifest(manifest)
        assert len(revalidated.files) == 2
        assert revalidated.files[0].path == "canonical/records.jsonl"

    # Stored names are copied to exact text at capture, so the name's hash callback never runs (it ran four
    # times before): the frozen edge is judged, and the list is never touched.
    assert len(files) == 2
    assert list.__getitem__(files, 0) is original
    assert hook.hash_calls == hash_calls_before_validation


@pytest.mark.parametrize("invalid_view", ["native", "public"])
def test_list_subclass_public_view_remains_authoritative(
    invalid_view: str,
) -> None:
    manifest = validate_portable_manifest(valid_manifest()).model_copy(deep=True)
    valid_file = manifest.files[0]
    invalid_file = valid_file.model_copy(update={"path": "../hidden.jsonl"})
    if invalid_view == "native":
        native_first, public_first = invalid_file, valid_file
    else:
        native_first, public_first = valid_file, invalid_file
    files = DivergentFilesList(
        (native_first, manifest.files[1]),
        (public_first, manifest.files[1]),
    )
    native_manifest = _PYDANTIC_DICT_DESCRIPTOR.__get__(manifest, BaseModel)
    dict.__setitem__(native_manifest, "files", files)

    if invalid_view == "native":
        revalidated = validate_portable_manifest(manifest)
        assert revalidated.files[0].path == "canonical/records.jsonl"
    else:
        with pytest.raises(ValidationError) as raised:
            validate_portable_manifest(manifest)
        assert raised.value.errors()[0]["loc"] == ("files", 0, "path")

    assert files.iteration_calls == 1  # a public view is observed once and replayed, never re-read


@pytest.mark.parametrize("late_container_kind", ["dict", "list"])
def test_list_subclass_cannot_introduce_unvalidated_exact_container(
    late_container_kind: str,
) -> None:
    # A later traversal that would expose a different container is never made:
    # the first observation of a public view is the one validated and replayed.
    manifest = validate_portable_manifest(valid_manifest()).model_copy(deep=True)
    files = LateExactContainerFilesList(
        tuple(manifest.files),
        late_container_kind,
    )
    native_manifest = _PYDANTIC_DICT_DESCRIPTOR.__get__(manifest, BaseModel)
    dict.__setitem__(native_manifest, "files", files)

    revalidated = validate_portable_manifest(manifest)

    assert [item.path for item in revalidated.files] == [item.path for item in files.public_items]
    assert all(type(item) is ManifestFile for item in revalidated.files)
    assert files.iteration_calls == 1


def test_transient_exact_container_owner_survives_public_iterations() -> None:
    # The transient container from the first (only) observation is retained with
    # the observation, so its identity cannot be reused before reconstruction.
    manifest = validate_portable_manifest(valid_manifest()).model_copy(deep=True)
    files = TransientExactContainerFilesList(tuple(manifest.files))
    native_manifest = _PYDANTIC_DICT_DESCRIPTOR.__get__(manifest, BaseModel)
    dict.__setitem__(native_manifest, "files", files)

    revalidated = validate_portable_manifest(manifest)

    assert revalidated.files[0].path == files.public_items[0].path  # not the later blanked path
    assert files.iteration_calls == 1
    assert files.first_id is not None and files.second_id is None


@pytest.mark.parametrize(
    ("owner", "candidate"),
    [
        ({"owner": 1}, {"candidate": 1}),
        (["owner"], ["candidate"]),
        (("owner",), ("candidate",)),
        ({"owner"}, {"candidate"}),
        (frozenset({"owner"}), frozenset({"candidate"})),
    ],
)
def test_native_container_inventory_requires_exact_owner_identity(
    owner: object,
    candidate: object,
) -> None:
    items = ("frozen",)
    inventory = {id(candidate): (owner, type(candidate), items)}

    with pytest.raises(
        ValueError,
        match="contract container was not present in the validated graph",
    ):
        _validated_native_container_items(candidate, inventory)

    inventory[id(candidate)] = (candidate, type(candidate), items)
    assert _validated_native_container_items(candidate, inventory) is items


def test_model_discovery_queues_only_declared_exact_string_values() -> None:
    schema_version = object()
    files = object()
    future_semantics = object()
    stored_entries = (
        ("schema_version", schema_version),
        ("future_entry_semantics", future_semantics),
        ("files", files),
    )

    queued = _model_discovery_values(PortableManifest, stored_entries)

    assert queued == (files, schema_version)
    assert all(item is not future_semantics for item in queued)


def _hostile_field_mapping_manifest() -> tuple[PortableManifest, list[str]]:
    """A manifest whose class ``model_fields`` mutates its own state on use."""

    class HostileFields(dict):
        target: Any = None
        calls: list[str]

        def _strike(self, hook: str) -> None:
            self.calls.append(hook)
            if self.target is not None:
                self.target.clear()

        def __contains__(self, name: object) -> bool:
            self._strike("contains")
            return dict.__contains__(self, name)

        def __iter__(self) -> Iterator[str]:
            self._strike("iter")
            return dict.__iter__(self)

        def keys(self):  # type: ignore[override]
            self._strike("keys")
            return dict.keys(self)

        def __getitem__(self, name: str) -> Any:
            self._strike("getitem")
            return dict.__getitem__(self, name)

    class HostileManifest(PortableManifest):
        pass

    manifest = HostileManifest.model_validate(valid_manifest())
    hostile = HostileFields(PortableManifest.model_fields)
    hostile.calls = []
    hostile.target = manifest.canonical_ids
    type.__setattr__(HostileManifest, "model_fields", hostile)
    return manifest, hostile.calls


def test_hostile_model_fields_cannot_enable_silent_subset_acceptance() -> None:
    manifest, calls = _hostile_field_mapping_manifest()
    assert len(manifest.canonical_ids) == 3

    try:
        accepted = validate_portable_manifest(manifest)
    except ValidationError:
        accepted = None

    # Fail-closed: reject, or accept the complete original state; never the
    # sanitized subset produced by a callback-bearing field mapping.
    assert accepted is None or len(accepted.canonical_ids) == 3
    assert calls == []


@pytest.mark.parametrize("inventory_kind", ["discovery", "validated"])
def test_model_inventories_require_exact_owner_identity(
    inventory_kind: str,
) -> None:
    manifest = validate_portable_manifest(valid_manifest()).model_copy(deep=True)
    owner = manifest.files[0]
    candidate = owner.model_copy(deep=True)
    stored_entries = tuple(dict.items(owner.__dict__))

    if inventory_kind == "discovery":
        fields_set_storage = set(owner.model_fields_set)
        extra_snapshot = (True, ())
        inventory = {
            id(candidate): (
                owner,
                stored_entries,
                fields_set_storage,
                extra_snapshot,
            )
        }
        with pytest.raises(
            ValueError,
            match="contract model was not present in the validated graph",
        ):
            _validated_model_discovery_snapshot(candidate, inventory)

        inventory[id(candidate)] = (
            candidate,
            stored_entries,
            fields_set_storage,
            extra_snapshot,
        )
        assert _validated_model_discovery_snapshot(candidate, inventory) == (
            stored_entries,
            fields_set_storage,
            extra_snapshot,
        )
    else:
        extra_entries: tuple[tuple[object, object], ...] = ()
        inventory = {
            id(candidate): (owner, stored_entries, extra_entries)
        }
        with pytest.raises(
            ValueError,
            match="contract model was not present in the validated graph",
        ):
            _validated_model_storage(candidate, inventory)

        inventory[id(candidate)] = (
            candidate,
            stored_entries,
            extra_entries,
        )
        assert _validated_model_storage(candidate, inventory) == (
            stored_entries,
            extra_entries,
        )

    candidate_reference = weakref.ref(candidate)
    del candidate
    gc.collect()
    assert candidate_reference() is not None
    inventory.clear()
    gc.collect()
    assert candidate_reference() is None


@pytest.mark.parametrize(
    "replacement_path",
    ["", "canonical/records.jsonl"],
)
def test_transient_model_owner_survives_validation_and_reconstruction(
    replacement_path: str,
) -> None:
    # The transient model yielded by the only observation is kept alive by that
    # observation until reconstruction finishes, then released.
    manifest = validate_portable_manifest(valid_manifest()).model_copy(deep=True)
    files = TransientModelFilesList(tuple(manifest.files), replacement_path)
    native_manifest = _PYDANTIC_DICT_DESCRIPTOR.__get__(manifest, BaseModel)
    dict.__setitem__(native_manifest, "files", files)

    revalidated = validate_portable_manifest(manifest)

    assert revalidated.files[0].path == manifest.files[0].path
    assert files.iteration_calls == 1
    assert files.first_reference is not None
    assert files.first_live_during_walk is True
    assert files.second_id is None

    first_reference = files.first_reference
    del revalidated
    gc.collect()
    assert first_reference() is None


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

    # Bounded discovery (shared discovery-budget policy): unbounded recursion
    # became a stable, value-free budget rejection raised before any recursion.
    assert isinstance(raised.value.__cause__, ValueError)
    assert str(raised.value.__cause__) == DISCOVERY_BUDGET_MESSAGE


@pytest.mark.parametrize("storage_kind", ["stored", "extra"])
def test_deep_unknown_value_preserves_extra_forbidden_priority(
    storage_kind: str,
) -> None:
    manifest = validate_portable_manifest(valid_manifest()).model_copy(deep=True)
    deeply_nested: object = "leaf"
    for _ in range(sys.getrecursionlimit() + 100):
        deeply_nested = [deeply_nested]

    if storage_kind == "stored":
        native_manifest = _PYDANTIC_DICT_DESCRIPTOR.__get__(manifest, BaseModel)
        dict.__setitem__(native_manifest, "future_entry_semantics", deeply_nested)
    else:
        _PYDANTIC_EXTRA_SLOT.__set__(
            manifest,
            {"future_entry_semantics": deeply_nested},
        )

    with pytest.raises(ValidationError) as raised:
        validate_portable_manifest(manifest)

    errors = raised.value.errors()
    assert len(errors) == 1
    assert errors[0]["type"] == "extra_forbidden"
    assert errors[0]["loc"] == ("future_entry_semantics",)
    # The caller's object is not carried (rendering a list nested past the recursion
    # limit would itself fail); the refusal renders.
    assert errors[0]["input"] == contracts_portability._OMITTED_INPUT
    assert "future_entry_semantics" in str(raised.value)


@pytest.mark.parametrize("value_kind", ["shallow", "deep"])
def test_inconsistent_hash_stored_name_preserves_unknown_field_authority(
    value_kind: str,
) -> None:
    manifest = validate_portable_manifest(valid_manifest()).model_copy(deep=True)
    if value_kind == "deep":
        retained_value: object = "leaf"
        for _ in range(sys.getrecursionlimit() + 100):
            retained_value = [retained_value]
    else:
        retained_value = "unknown"

    stored_name = InconsistentHashStoredName("files")
    native_manifest = _PYDANTIC_DICT_DESCRIPTOR.__get__(manifest, BaseModel)
    dict.__delitem__(native_manifest, "files")
    dict.__setitem__(native_manifest, stored_name, retained_value)
    stored_name.hash_calls = 0
    stored_name.equality_calls = 0
    stored_name.class_calls = 0
    stored_name.armed = True

    with pytest.raises(ValidationError) as raised:
        validate_portable_manifest(manifest)

    if value_kind == "deep":
        # A conservatively queued (non-exact) name holding a graph deeper than
        # the discovery budget is rejected by the budget before any stored-name
        # callback runs.  This is the intentional bounded-input envelope.
        assert isinstance(raised.value.__cause__, ValueError)
        assert str(raised.value.__cause__) == DISCOVERY_BUDGET_MESSAGE
        assert (
            stored_name.hash_calls,
            stored_name.equality_calls,
            stored_name.class_calls,
        ) == (0, 0, 0)
        return

    # The name is read as its exact text at capture ("files", a declared field), never through its own hash:
    # the retained value is judged as that field (and refused), and no hook of the name runs.
    errors = raised.value.errors()
    assert len(errors) == 1
    assert errors[0]["type"] == "list_type"
    assert errors[0]["loc"] == ("files",)
    assert (
        stored_name.hash_calls,
        stored_name.equality_calls,
        stored_name.class_calls,
    ) == (0, 0, 0)


def test_unknown_fields_and_missing_schema_version_are_rejected() -> None:
    unknown = valid_manifest()
    unknown["authorization_proven"] = True
    with pytest.raises(ValidationError, match="authorization_proven"):
        validate_portable_manifest(unknown)

    unversioned = valid_manifest()
    del unversioned["schema_version"]
    with pytest.raises(ValidationError, match="schema_version"):
        validate_portable_manifest(unversioned)


def test_public_path_normalizer_matches_manifest_contract() -> None:
    assert normalize_bundle_path("a/./b//c.txt") == "a/b/c.txt"
    for bad in ("../x", "/abs", "a\\b", "CON", ""):
        with pytest.raises(ValueError):
            normalize_bundle_path(bad)
        manifest = valid_manifest()
        manifest["files"][0]["path"] = bad
        with pytest.raises(ValidationError):
            validate_portable_manifest(manifest)


# --- bounded native discovery (portability tier) ----------------------------


def _manifest_with_files(files: object) -> PortableManifest:
    return validate_portable_manifest(valid_manifest()).model_copy(
        update={"files": files}
    )


def _assert_budget_rejection(manifest: PortableManifest) -> None:
    with pytest.raises(ValidationError) as raised:
        validate_portable_manifest(manifest)
    assert isinstance(raised.value.__cause__, ValueError)
    assert str(raised.value.__cause__) == DISCOVERY_BUDGET_MESSAGE


def test_discovery_depth_beyond_budget_is_rejected_with_stable_error() -> None:
    nested: object = "leaf"
    for _ in range(PORTABILITY_MAX_DEPTH + 5):
        nested = [nested]
    _assert_budget_rejection(_manifest_with_files(nested))


def test_discovery_node_count_beyond_budget_is_rejected() -> None:
    wide = [[] for _ in range(PORTABILITY_MAX_NODES + 1)]
    _assert_budget_rejection(_manifest_with_files(wide))


def test_discovery_edges_are_charged_for_aliased_repeats() -> None:
    shared: list[object] = []
    aliased = [shared] * (PORTABILITY_MAX_EDGES + 1)  # two nodes, many edges
    _assert_budget_rejection(_manifest_with_files(aliased))


def test_discovery_budget_applies_under_conservative_non_exact_names() -> None:
    manifest = validate_portable_manifest(valid_manifest()).model_copy(deep=True)
    wide = [[] for _ in range(PORTABILITY_MAX_NODES + 1)]
    native = _PYDANTIC_DICT_DESCRIPTOR.__get__(manifest, BaseModel)
    dict.__delitem__(native, "files")
    dict.__setitem__(native, InconsistentHashStoredName("files"), wide)
    _assert_budget_rejection(manifest)


class _RecordingBudget(DiscoveryBudget):
    instances: list["_RecordingBudget"] = []

    def __init__(self, tier=None):
        super().__init__(tier) if tier is not None else super().__init__()
        type(self).instances.append(self)


def _measure_valid_manifest(monkeypatch) -> tuple[int, int, int]:
    _RecordingBudget.instances = []
    monkeypatch.setattr(contracts_portability, "DiscoveryBudget", _RecordingBudget)
    validate_portable_manifest(validate_portable_manifest(valid_manifest()))
    budget = _RecordingBudget.instances[-1]
    deepest = max(
        budget.authoritative_depth(obj) or 0 for obj in budget._seen.values()
    )
    return budget.nodes, budget.edges, deepest


def _tier(nodes: int, edges: int, depth: int) -> DiscoveryTier:
    return DiscoveryTier("test", nodes, edges, depth)


def test_budget_boundary_is_exact_for_nodes_edges_and_depth(monkeypatch) -> None:
    nodes, edges, depth = _measure_valid_manifest(monkeypatch)
    assert nodes > 1 and edges > 1 and depth > 0
    manifest = validate_portable_manifest(valid_manifest())

    def run(tier: DiscoveryTier) -> None:
        monkeypatch.setattr(
            contracts_portability,
            "PORTABILITY",
            tier,
        )
        validate_portable_manifest(manifest)

    run(_tier(nodes, edges, depth))  # exactly at every limit: accepted
    for tight in (
        _tier(nodes - 1, edges, depth),
        _tier(nodes, edges - 1, depth),
        _tier(nodes, edges, depth - 1),
    ):
        with pytest.raises(ValidationError) as raised:
            run(tight)
        assert str(raised.value.__cause__) == DISCOVERY_BUDGET_MESSAGE


def test_budget_accounting_runs_no_container_protocol(monkeypatch) -> None:
    calls: list[str] = []

    class Noisy(list):
        def __len__(self) -> int:
            calls.append("len")
            return list.__len__(self)

        def __iter__(self):
            calls.append("iter")
            return list.__iter__(self)

        def __bool__(self) -> bool:
            calls.append("bool")
            return True

    noisy = Noisy([[] for _ in range(10)])
    manifest = _manifest_with_files(noisy)  # built under the real tier: a mapping input is budgeted too
    monkeypatch.setattr(contracts_portability, "PORTABILITY", _tier(2, 4, 8))
    calls.clear()
    with pytest.raises(ValidationError) as raised:
        validate_portable_manifest(manifest)
    assert str(raised.value.__cause__) == DISCOVERY_BUDGET_MESSAGE
    assert calls == []


def test_endless_public_mapping_view_is_bounded() -> None:
    class Endless(Mapping):
        def __getitem__(self, key):
            return key

        def __iter__(self):
            index = 0
            while True:
                yield index
                index += 1

        def __len__(self) -> int:
            return 1

        def items(self):
            index = 0
            while True:
                yield (index, index)
                index += 1

    _assert_budget_rejection(_manifest_with_files(Endless()))


@pytest.mark.parametrize("kind", ["list", "dict"])
def test_discovery_edge_budget_rejects_before_any_stored_name_callback(
    monkeypatch, kind: str
) -> None:
    _, baseline_edges, _ = _measure_valid_manifest(monkeypatch)
    monkeypatch.setattr(
        contracts_portability,
        "PORTABILITY",
        _tier(10_000, baseline_edges + 20, 64),
    )
    shared: list[object] = []
    heavy: object = (
        [shared] * 200 if kind == "list" else {index: shared for index in range(200)}
    )
    manifest = validate_portable_manifest(valid_manifest()).model_copy(deep=True)
    stored_name = InconsistentHashStoredName("files")
    native = _PYDANTIC_DICT_DESCRIPTOR.__get__(manifest, BaseModel)
    dict.__delitem__(native, "files")
    dict.__setitem__(native, stored_name, heavy)
    stored_name.hash_calls = stored_name.equality_calls = stored_name.class_calls = 0
    stored_name.armed = True

    with pytest.raises(ValidationError) as raised:
        validate_portable_manifest(manifest)

    assert str(raised.value.__cause__) == DISCOVERY_BUDGET_MESSAGE
    assert (
        stored_name.hash_calls,
        stored_name.equality_calls,
        stored_name.class_calls,
    ) == (0, 0, 0)


# --- exact native containers are charged once ---------------------------------


def _budget_measure(monkeypatch, manifest: PortableManifest) -> tuple[int, int]:
    _RecordingBudget.instances = []
    monkeypatch.setattr(contracts_portability, "DiscoveryBudget", _RecordingBudget)
    with pytest.raises(ValidationError):
        validate_portable_manifest(manifest)  # the replaced field is invalid; accounting is what we read
    budget = _RecordingBudget.instances[-1]
    return budget.nodes, budget.edges


@pytest.mark.parametrize("kind", [list, tuple])
def test_exact_native_sequence_entries_are_charged_exactly_once(monkeypatch, kind) -> None:
    shared: list[object] = []
    _, empty_edges = _budget_measure(monkeypatch, _manifest_with_files(kind()))
    _, full_edges = _budget_measure(monkeypatch, _manifest_with_files(kind([shared] * 1000)))
    assert full_edges - empty_edges == 1000


@pytest.mark.parametrize("kind", [list, tuple])
def test_an_exact_sequence_within_the_adr_edge_limit_is_not_rejected_by_the_budget(kind) -> None:
    shared: list[object] = []
    within = kind([shared] * (PORTABILITY_MAX_EDGES // 2 + 1000))  # twice that is over the old, doubled limit
    manifest = _manifest_with_files(within)
    with pytest.raises(ValidationError) as raised:  # still invalid as a manifest, but not for the budget
        validate_portable_manifest(manifest)
    assert str(raised.value.__cause__) != DISCOVERY_BUDGET_MESSAGE


def test_the_adr_edge_limit_itself_still_rejects_an_exact_sequence() -> None:
    shared: list[object] = []
    _assert_budget_rejection(_manifest_with_files([shared] * (PORTABILITY_MAX_EDGES + 1)))


def test_exact_dict_and_public_view_charging_is_unchanged(monkeypatch) -> None:
    # Exact dict: two edges per entry (key and value), charged once at discovery.
    _, empty_edges = _budget_measure(monkeypatch, _manifest_with_files({}))
    _, full_edges = _budget_measure(monkeypatch, _manifest_with_files({i: 0 for i in range(500)}))
    assert full_edges - empty_edges == 1000


def test_endless_public_sequence_view_is_bounded() -> None:
    class EndlessList(list):
        def __iter__(self):  # the public view; the native backing is empty
            index = 0
            while True:
                yield index
                index += 1

    _assert_budget_rejection(_manifest_with_files(EndlessList()))


# --- exact-type tests use identity, never == (hostile metaclass) -----------------


def _hostile_equal_types():
    calls: list[str] = []

    class ClaimsEquality(type):
        def __eq__(cls, other):  # a class claiming to equal list/tuple/set/frozenset/dict
            calls.append(f"eq:{getattr(other, '__name__', other)}")
            return True

        def __hash__(cls):
            return type.__hash__(cls)

    return calls, ClaimsEquality


@pytest.mark.parametrize("base", [list, tuple])
def test_a_metaclass_claiming_equality_with_an_exact_type_is_not_taken_as_exact(base) -> None:
    calls, meta = _hostile_equal_types()
    declared = [{"path": "canonical/a.txt", "size_bytes": 1,
                 "checksum": {"algorithm": "sha256", "digest": "a" * 64}, "entry_type": "file"}]

    public = declared + [{"path": "../escape", "size_bytes": 1,
                          "checksum": {"algorithm": "sha256", "digest": "a" * 64}, "entry_type": "file"}]

    class Hostile(base, metaclass=meta):
        # The public view differs from the (empty) native backing: a subclass is
        # read through its public iteration, never through hidden native storage.
        def __iter__(self):
            return iter(public)

    manifest = _manifest_with_files(Hostile())
    calls.clear()
    with pytest.raises(ValidationError):
        validate_portable_manifest(manifest)
    assert calls == [], "a metaclass __eq__ ran during exact-type classification"


def test_the_public_view_of_a_spoofing_container_is_what_is_validated() -> None:
    calls, meta = _hostile_equal_types()

    stable = [{"path": "../escape", "size_bytes": 1,
               "checksum": {"algorithm": "sha256", "digest": "a" * 64}, "entry_type": "file"}]

    class Hostile(list, metaclass=meta):
        def __iter__(self):  # stable objects: the public view is observed the same way each time
            return iter(stable)

    with pytest.raises(ValidationError) as raised:
        validate_portable_manifest(_manifest_with_files(Hostile()))
    assert "traverse" in str(raised.value) or "relative" in str(raised.value) or "files" in str(raised.value)
    assert calls == []


def test_no_exact_type_test_in_the_module_uses_tuple_membership() -> None:
    import ast
    import pathlib

    tree = ast.parse(pathlib.Path(contracts_portability.__file__).read_text(encoding="utf-8"))
    offenders = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare) and any(isinstance(op, (ast.In, ast.NotIn)) for op in node.ops):
            for comparator in node.comparators:
                if isinstance(comparator, ast.Tuple) and any(
                    isinstance(element, ast.Name) and element.id in {"list", "tuple", "set", "frozenset", "dict", "str", "int"}
                    for element in comparator.elts
                ):
                    offenders.append(node.lineno)
    assert offenders == [], f"type membership via == at lines {offenders}"


# --- review findings: every public observation is bounded; reconstruction is linear ---


class _EndlessExtra(Mapping):
    """A custom extra-storage mapping whose public views never end."""

    def __init__(self, mode: str) -> None:
        self.mode = mode

    def __getitem__(self, key):
        return key

    def __len__(self) -> int:
        return 1

    def __iter__(self):
        if self.mode == "items":
            return iter(())  # only the item view is endless
        return (f"name_{index}" for index in iter(int, 1))

    def items(self):
        if self.mode == "names":
            return iter(())  # only the name view is endless
        return ((f"name_{index}", index) for index in iter(int, 1))


@pytest.mark.parametrize("mode", ["names", "items"])
def test_an_endless_custom_extra_mapping_is_stopped_by_the_budget(mode) -> None:
    manifest = validate_portable_manifest(valid_manifest()).model_copy(deep=True)
    _PYDANTIC_EXTRA_SLOT.__set__(manifest, _EndlessExtra(mode))
    _assert_budget_rejection(manifest)


def test_an_endless_fields_set_is_stopped_by_the_budget() -> None:
    import itertools

    manifest = validate_portable_manifest(valid_manifest()).model_copy(deep=True)
    _PYDANTIC_FIELDS_SET_SLOT.__set__(manifest, itertools.repeat("manifest_id"))
    _assert_budget_rejection(manifest)


def test_a_finite_non_set_fields_set_keeps_its_ordinary_outcome() -> None:
    manifest = validate_portable_manifest(valid_manifest()).model_copy(deep=True)
    names = list(_PYDANTIC_FIELDS_SET_SLOT.__get__(manifest, type(manifest)))
    _PYDANTIC_FIELDS_SET_SLOT.__set__(manifest, tuple(names))
    validate_portable_manifest(manifest)  # a tuple of declared names is still fine


def test_a_native_fields_set_is_not_charged_twice(monkeypatch) -> None:
    manifest = validate_portable_manifest(valid_manifest())
    _, edges_set = _budget_measure_ok(monkeypatch, manifest)
    copy = manifest.model_copy(deep=True)
    _PYDANTIC_FIELDS_SET_SLOT.__set__(copy, frozenset(_PYDANTIC_FIELDS_SET_SLOT.__get__(manifest, type(manifest))))
    _, edges_frozen = _budget_measure_ok(monkeypatch, copy)
    # a set is charged once at discovery; a non-set is charged once, per yielded name, at the walk
    assert edges_set == edges_frozen


def _budget_measure_ok(monkeypatch, manifest) -> tuple[int, int]:
    _RecordingBudget.instances = []
    monkeypatch.setattr(contracts_portability, "DiscoveryBudget", _RecordingBudget)
    validate_portable_manifest(manifest)
    budget = _RecordingBudget.instances[-1]
    monkeypatch.undo()
    return budget.nodes, budget.edges


def test_shared_native_substructure_is_reconstructed_once_per_object() -> None:
    import time

    child: object = []
    for _ in range(40):  # 2**40 paths, 40 nodes
        child = [child, child]
    manifest = _manifest_with_files(child)
    started = time.monotonic()
    with pytest.raises(ValidationError) as raised:
        validate_portable_manifest(manifest)
    assert time.monotonic() - started < 5
    assert str(raised.value.__cause__) != contracts_portability.DISCOVERY_BUDGET_MESSAGE if hasattr(
        contracts_portability, "DISCOVERY_BUDGET_MESSAGE") else True


def test_a_public_view_is_observed_once_and_replayed() -> None:
    """A public iterator that changes (here: becomes endless) after its first traversal."""

    entry = {"path": "canonical/a.txt", "size_bytes": 1,
             "checksum": {"algorithm": "sha256", "digest": "a" * 64}, "entry_type": "file"}

    class ChangesAfterFirstLook(list):
        traversals = 0

        def __iter__(self):
            type(self).traversals += 1
            if type(self).traversals == 1:
                return iter([entry])
            def forever():
                while True:
                    yield entry
            return forever()

    view = ChangesAfterFirstLook()
    manifest = _manifest_with_files(view)
    with pytest.raises(ValidationError):  # a list subclass of one entry: still invalid as a file list type
        validate_portable_manifest(manifest)
    assert ChangesAfterFirstLook.traversals == 1


def test_a_shared_public_view_reached_by_two_edges_is_replayed_identically() -> None:
    entry = {"path": "canonical/a.txt", "size_bytes": 1,
             "checksum": {"algorithm": "sha256", "digest": "a" * 64}, "entry_type": "file"}
    looks: list[int] = []

    class Counting(list):
        def __iter__(self):
            looks.append(1)
            return iter([entry])

    shared = Counting()
    manifest = _manifest_with_files([shared, shared])
    with pytest.raises(ValidationError):
        validate_portable_manifest(manifest)
    assert len(looks) == 1


def test_reconstruction_still_detects_cycles_after_memoization() -> None:
    cycle: list[object] = []
    cycle.append(cycle)
    with pytest.raises(ValidationError):
        validate_portable_manifest(_manifest_with_files([cycle, cycle]))


def test_a_valid_manifest_with_shared_models_still_validates() -> None:
    document = valid_manifest()
    manifest = validate_portable_manifest(document)
    shared_checksum = manifest.files[0].checksum
    copy = manifest.model_copy(deep=True)
    files = list(copy.files)
    files[1] = files[1].model_copy(update={"checksum": shared_checksum})
    copy = copy.model_copy(update={"files": files})
    validated = validate_portable_manifest(copy)
    assert validated.files[1].checksum.digest == shared_checksum.digest


# --- pinned shapes from a sweep of the receiving boundary ---------------------------

_SWEEP_FILE = {"path": "canonical/a.txt", "size_bytes": 1,
               "checksum": {"algorithm": "sha256", "digest": "a" * 64}, "entry_type": "file"}


class _TwinKey(str):
    """A key with the same text as a declared name but a different hash."""

    def __hash__(self) -> int:
        return str.__hash__(self) ^ 1

    def __eq__(self, other: object) -> bool:
        return str.__eq__(self, other)


def test_an_object_spoofing_its_class_as_list_is_not_read_as_a_list() -> None:
    class Spoof:
        @property
        def __class__(self):
            return list

        def __iter__(self):
            return iter([_SWEEP_FILE])

    with pytest.raises(ValidationError):
        validate_portable_manifest(_manifest_with_files(Spoof()))


@pytest.mark.parametrize("container", ["dict", "view"])
def test_a_twin_key_cannot_smuggle_a_second_value_for_a_declared_name(container) -> None:
    pairs = list(_SWEEP_FILE.items()) + [(_TwinKey("path"), "../escape")]
    if container == "dict":
        entry: object = dict(pairs)
    else:
        class View(Mapping):
            def __getitem__(self, key):
                return dict(pairs)[key]

            def __iter__(self):
                return iter(key for key, _ in pairs)

            def __len__(self) -> int:
                return len(pairs)

            def items(self):
                return iter(pairs)

        entry = View()
    with pytest.raises(ValidationError):
        validate_portable_manifest(_manifest_with_files([entry]))


def test_a_custom_mapping_pair_with_a_changing_iterator_is_read_once() -> None:
    reads: list[int] = []

    class Pair(tuple):
        def __iter__(self):
            reads.append(1)
            if len(reads) == 1:
                return iter(("path", "canonical/a.txt"))
            return iter(("path", "../escape"))

    pairs = [(key, value) for key, value in _SWEEP_FILE.items() if key != "path"]
    pairs.append(Pair(("path", "canonical/a.txt")))

    class View(Mapping):
        def __getitem__(self, key):
            raise KeyError(key)

        def __iter__(self):
            return iter(())

        def __len__(self) -> int:
            return len(pairs)

        def items(self):
            return iter(pairs)

    second = validate_portable_manifest(valid_manifest()).files[1]
    validated = validate_portable_manifest(_manifest_with_files([View(), second]))
    assert validated.files[0].path == "canonical/a.txt"
    assert len(reads) == 1


def test_the_receiving_boundary_never_reads_the_clock() -> None:
    import inspect

    source = inspect.getsource(contracts_portability)
    assert not [word for word in ("datetime.now", "time.time", "utcnow", "monotonic(") if word in source]


# ---- concrete types decide traversal and reconstruction; a spoofed ``__class__`` is not a container


def _spoofing(cls, items):
    class Spoof:
        @property
        def __class__(self):
            return cls

        def __iter__(self):
            return iter(items)

        def __len__(self):
            return len(items)

    return Spoof()


@pytest.mark.parametrize("claimed", [list, tuple])
def test_an_object_spoofing_a_container_class_is_refused_with_an_otherwise_valid_inventory(claimed) -> None:
    files = valid_manifest()["files"]  # the complete inventory, envelope included
    assert validate_portable_manifest(_manifest_with_files(list(files)))  # the same items as a real list validate
    with pytest.raises(ValidationError):
        validate_portable_manifest(_manifest_with_files(_spoofing(claimed, files)))


def test_an_object_spoofing_list_inside_a_nested_value_is_refused() -> None:
    document = valid_manifest()
    document["scope"]["project_ids"] = _spoofing(list, list(document["scope"]["project_ids"]))
    with pytest.raises(ValidationError):
        validate_portable_manifest(document)


def test_a_genuine_list_subclass_with_the_valid_inventory_still_validates() -> None:
    class Inventory(list):
        pass

    files = valid_manifest()["files"]
    validated = validate_portable_manifest(_manifest_with_files(Inventory(files)))
    assert [f.path for f in validated.files] == [f["path"] for f in files]


def test_an_object_spoofing_the_manifest_class_is_refused() -> None:
    real = validate_portable_manifest(valid_manifest())

    class Impostor:
        @property
        def __class__(self):
            return type(real)

        def __getattr__(self, name):
            return getattr(real, name)

    with pytest.raises(ValidationError):
        validate_portable_manifest(Impostor())


# ---- keys that are str subclasses: read as their text, and one text never names two values


class _SameTextKey(str):
    """A key whose text names a field but whose hash and equality differ from that text."""

    def __hash__(self):
        return 7

    def __eq__(self, other):
        return False


def test_a_second_key_with_the_same_text_is_refused_not_silently_resolved() -> None:
    document = valid_manifest()
    document[_SameTextKey("manifest_id")] = "manifest-other"
    with pytest.raises(ValidationError):
        validate_portable_manifest(document)


def test_a_second_key_with_the_same_text_is_refused_in_a_nested_mapping() -> None:
    document = valid_manifest()
    document["scope"] = {**document["scope"], _SameTextKey("tenant_id"): "tenant-other"}
    with pytest.raises(ValidationError):
        validate_portable_manifest(document)


def test_an_ordinary_str_subclass_key_is_read_as_its_text() -> None:
    class Plain(str):
        pass

    document = {Plain(key): value for key, value in valid_manifest().items()}
    assert validate_portable_manifest(document).manifest_id == valid_manifest()["manifest_id"]


def test_a_shared_native_subtree_in_a_mapping_input_is_normalized_once() -> None:
    node: list = []
    for _ in range(40):
        node = [node, node]  # 2**40 paths, 41 objects
    document = valid_manifest()
    document["omissions"] = node
    with pytest.raises(ValidationError):  # the shape is wrong for the field; the walk itself must stay bounded
        validate_portable_manifest(document)


# ---- a malformed pair inside a finite item view is unpacked while it is collected


class _OnePairExtra(Mapping):
    """A custom extra-storage mapping with a finite item view of one given pair."""

    def __init__(self, pair) -> None:
        self.pair = pair

    def __getitem__(self, key):
        raise KeyError(key)

    def __len__(self) -> int:
        return 1

    def __iter__(self):
        return iter(())

    def items(self):
        return iter((self.pair,))


def test_an_endless_iterator_posing_as_one_extra_pair_is_refused_not_consumed() -> None:
    import itertools

    manifest = validate_portable_manifest(valid_manifest()).model_copy(deep=True)
    _PYDANTIC_EXTRA_SLOT.__set__(manifest, _OnePairExtra(itertools.repeat("unbounded")))
    with pytest.raises(ValidationError):
        validate_portable_manifest(manifest)


def test_an_ordinary_pair_in_a_custom_extra_view_keeps_its_outcome() -> None:
    manifest = validate_portable_manifest(valid_manifest()).model_copy(deep=True)
    _PYDANTIC_EXTRA_SLOT.__set__(manifest, _OnePairExtra(("smuggled", "value")))
    with pytest.raises(ValidationError):  # an unknown field is still refused
        validate_portable_manifest(manifest)


# ---- rendering a refusal never re-reads caller objects: errors are value-free and bounded


class _CountingEndlessExtra(Mapping):
    """An extra-storage view of well-formed pairs that never ends; counts every read."""

    def __init__(self) -> None:
        self.reads = 0

    def __getitem__(self, key):
        return key

    def __len__(self) -> int:
        return 1

    def _next(self) -> int:
        self.reads += 1
        if self.reads > 2_000_000:  # a guard for the unfixed tree: never reached when bounded
            raise RuntimeError("kept reading")
        return self.reads

    def __iter__(self):
        while True:
            yield f"name_{self._next()}"

    def items(self):
        return ((f"name_{n}", n) for n in iter(self._next, -1))


def _render_everything(error: ValidationError) -> None:
    import time

    started = time.monotonic()
    for render in (str, repr):
        assert len(render(error)) < 20_000
    assert len(error.json()) < 20_000
    error.errors()
    assert time.monotonic() - started < 5


def test_rendering_a_refused_model_never_reads_its_extra_view_again() -> None:
    manifest = validate_portable_manifest(valid_manifest()).model_copy(deep=True)
    extra = _CountingEndlessExtra()
    _PYDANTIC_EXTRA_SLOT.__set__(manifest, extra)
    with pytest.raises(ValidationError) as raised:
        validate_portable_manifest(manifest)
    after_validation = extra.reads
    _render_everything(raised.value)
    assert extra.reads == after_validation, f"rendering read {extra.reads - after_validation} more entries"


def test_rendering_a_refused_mapping_never_reads_a_nested_models_extra_view() -> None:
    manifest = validate_portable_manifest(valid_manifest())
    scope = manifest.scope.model_copy(deep=True)
    extra = _CountingEndlessExtra()
    _PYDANTIC_EXTRA_SLOT.__set__(scope, extra)
    document = valid_manifest()
    document["scope"] = scope
    with pytest.raises(ValidationError) as raised:
        validate_portable_manifest(document)
    after_validation = extra.reads
    _render_everything(raised.value)
    assert extra.reads == after_validation, f"rendering read {extra.reads - after_validation} more entries"


def test_an_ordinary_refusal_still_names_the_field_and_the_error_type() -> None:
    document = valid_manifest()
    document["manifest_id"] = 5
    with pytest.raises(ValidationError) as raised:
        validate_portable_manifest(document)
    details = raised.value.errors()
    assert any(error["loc"][:1] == ("manifest_id",) for error in details)
    assert all("type" in error for error in details)
    assert "manifest_id" in str(raised.value)


def test_a_nested_model_instance_in_a_mapping_input_still_validates() -> None:
    manifest = validate_portable_manifest(valid_manifest())
    document = valid_manifest()
    document["scope"] = manifest.scope  # an ordinary validated instance, no retained extras
    assert validate_portable_manifest(document).scope == manifest.scope


def test_a_nested_model_instance_with_a_smuggled_extra_is_refused_in_a_mapping_input() -> None:
    manifest = validate_portable_manifest(valid_manifest())
    scope = manifest.scope.model_copy(deep=True)
    _PYDANTIC_EXTRA_SLOT.__set__(scope, MappingProxyType({"smuggled": "value"}))
    document = valid_manifest()
    document["scope"] = scope
    with pytest.raises(ValidationError):
        validate_portable_manifest(document)


def test_nested_model_instances_in_a_mapping_input_share_one_discovery_budget() -> None:
    """K instances that share one large retained field cost one budget, not K.

    Each instance's retained ``path`` is the same 300,000-element list (the wrong type, so the
    schema refuses it at the end).  Rebuilding every nested instance under its own budget and memo
    copied that list once per instance: 200 instances took minutes and gigabytes.
    """

    import time

    shared = list(range(300_000))
    manifest = validate_portable_manifest(valid_manifest())
    files = []
    for _ in range(200):
        nested = manifest.files[0].model_copy(deep=True)
        nested.__dict__["path"] = shared
        files.append(nested)
    document = valid_manifest()
    document["files"] = files
    started = time.monotonic()
    with pytest.raises(ValidationError) as raised:
        validate_portable_manifest(document)
    assert time.monotonic() - started < 5
    assert {error["loc"][:2] for error in raised.value.errors()} >= {("files", 0)}


def test_the_shared_budget_still_refuses_a_call_whose_nested_instances_exceed_it() -> None:
    """The budget is the call's: many nested instances that each fit alone are refused together."""

    manifest = validate_portable_manifest(valid_manifest())
    files = []
    for _ in range(40):
        nested = manifest.files[0].model_copy(deep=True)
        nested.__dict__["path"] = list(range(30_000))  # distinct lists: 1.2M entries in all
        files.append(nested)
    document = valid_manifest()
    document["files"] = files
    with pytest.raises(ValidationError, match="discovery budget"):
        validate_portable_manifest(document)


def test_a_refusal_by_the_schema_never_renders_a_caller_object() -> None:
    """An object the traversal leaves to the schema is not carried into the refusal and rendered."""

    class Loud:
        calls = 0

        def __repr__(self) -> str:
            Loud.calls += 1
            return "LOUD" * 100_000

    document = valid_manifest()
    document["manifest_id"] = Loud()
    with pytest.raises(ValidationError) as raised:
        validate_portable_manifest(document)
    Loud.calls = 0
    text = str(raised.value) + repr(raised.value) + raised.value.json()
    assert Loud.calls == 0 and "LOUD" not in text
    assert any(error["loc"][:1] == ("manifest_id",) for error in raised.value.errors())


# --- the first snapshot of a model or exact container is kept when a shallower path revisits it ---


def _files_with_first(manifest: PortableManifest, first: object) -> tuple[object, ...]:
    """The manifest's own files, the first replaced by ``first`` (so nothing else about the manifest is wrong)."""

    return (first, *manifest.files[1:])


def _file_dict(file: ManifestFile, checksum: object) -> dict[str, object]:
    return {"path": file.path, "size_bytes": file.size_bytes, "checksum": checksum, "entry_type": "file"}


class _RepairsOnIteration(list):
    """The native backing holds the invalid object four levels down; the public iterator runs ``repair`` and
    yields files that reach the same object one level up (a shallower revisit)."""

    def __init__(self, native_holder: object, repair, public_files: tuple[object, ...]) -> None:
        list.__init__(self, [[[[native_holder]]]])
        self.repair, self.public_files = repair, public_files
        self.iteration_calls = 0

    def __iter__(self):
        self.iteration_calls += 1
        self.repair()
        return iter(self.public_files)


def test_a_shallower_revisit_does_not_replace_the_first_snapshot_of_a_model() -> None:
    manifest = validate_portable_manifest(valid_manifest())
    first = manifest.files[0]
    checksum = first.checksum.model_copy(update={"digest": "not-a-digest"})  # unchecked: invalid on entry
    repaired = first.checksum.digest

    def repair() -> None:
        object.__setattr__(checksum, "digest", repaired)

    control = _RepairsOnIteration(checksum, lambda: None, _files_with_first(manifest, _file_dict(first, checksum)))
    with pytest.raises(ValidationError):
        validate_portable_manifest(_manifest_with_files(control))  # control: nothing repairs it
    public = _files_with_first(manifest, _file_dict(first, checksum))
    broken = _RepairsOnIteration(checksum, repair, public)
    with pytest.raises(ValidationError):
        validate_portable_manifest(_manifest_with_files(broken))
    assert checksum.digest == repaired  # the public iterator did run and repair the live model afterwards


def test_a_shallower_revisit_does_not_replace_the_first_snapshot_of_an_exact_container() -> None:
    manifest = validate_portable_manifest(valid_manifest())
    first = manifest.files[0]
    checksum = {"algorithm": first.checksum.algorithm, "digest": "not-a-digest"}  # exact dict, invalid

    def repair() -> None:
        checksum["digest"] = first.checksum.digest  # the live exact container is repaired

    public = _files_with_first(manifest, _file_dict(first, checksum))  # reaches the exact dict shallower
    control = _RepairsOnIteration(checksum, lambda: None, public)
    with pytest.raises(ValidationError):
        validate_portable_manifest(_manifest_with_files(control))
    with pytest.raises(ValidationError):
        validate_portable_manifest(_manifest_with_files(_RepairsOnIteration(checksum, repair, public)))
    assert checksum["digest"] == first.checksum.digest


def test_an_honest_shallower_revisit_still_validates() -> None:
    manifest = validate_portable_manifest(valid_manifest())
    first = manifest.files[0]
    public = _files_with_first(manifest, _file_dict(first, first.checksum))
    honest = _RepairsOnIteration(first.checksum, lambda: None, public)
    assert validate_portable_manifest(_manifest_with_files(honest)).files[0].path == first.path


# --- a mapping input is discovered whole, and its subclasses are observed, like a model graph ---


def _sibling_repair_document(producer_first: bool) -> tuple[dict, BaseModel]:
    """A mapping whose producer model's fields-set iterator repairs the scope model's invalid tenant."""

    manifest = validate_portable_manifest(valid_manifest())
    scope = manifest.scope.model_copy(deep=True)
    _PYDANTIC_DICT_DESCRIPTOR.__get__(scope, BaseModel)["tenant_id"] = ""  # invalid on entry
    producer = manifest.producer.model_copy(deep=True)
    _PYDANTIC_FIELDS_SET_SLOT.__set__(
        producer,
        RepairingFieldsSetIterable(tuple(type(producer).model_fields), scope, "tenant_id", "tenant-1"),
    )
    document = {key: value for key, value in valid_manifest().items() if key not in ("scope", "producer")}
    ordered = (("producer", producer), ("scope", scope)) if producer_first else (("scope", scope), ("producer", producer))
    for key, value in ordered:
        document[key] = value
    return document, scope


@pytest.mark.parametrize("producer_first", [True, False], ids=["producer-first", "scope-first"])
def test_a_sibling_models_callback_cannot_repair_another_model_before_it_is_snapshotted(producer_first) -> None:
    document, scope = _sibling_repair_document(producer_first)
    with pytest.raises(ValidationError) as raised:
        validate_portable_manifest(document)
    assert any(error["loc"][:2] == ("scope", "tenant_id") for error in raised.value.errors())
    assert _PYDANTIC_DICT_DESCRIPTOR.__get__(scope, BaseModel)["tenant_id"] == "tenant-1"  # the callback did run


def test_the_same_sibling_repair_is_refused_inside_a_manifest_model_too() -> None:
    document, scope = _sibling_repair_document(True)
    manifest = validate_portable_manifest(valid_manifest()).model_copy(deep=True)
    native = _PYDANTIC_DICT_DESCRIPTOR.__get__(manifest, BaseModel)
    dict.__setitem__(native, "producer", document["producer"])
    dict.__setitem__(native, "scope", scope)
    with pytest.raises(ValidationError):
        validate_portable_manifest(manifest)


class _DivergentList(list):
    """Valid native backing, but the public iterator yields something else."""

    def __init__(self, native, public) -> None:
        list.__init__(self, native)
        self.public = public
        self.iterations = 0

    def __iter__(self):
        self.iterations += 1
        return iter(self.public)


def _unsafe_file(manifest: PortableManifest) -> ManifestFile:
    return manifest.files[0].model_copy(update={"path": "../escape.txt"})


@pytest.mark.parametrize("form", ["model", "mapping"])
def test_a_list_subclass_is_validated_by_its_public_view_in_both_input_forms(form) -> None:
    manifest = validate_portable_manifest(valid_manifest())
    divergent = _DivergentList(list(manifest.files), [_unsafe_file(manifest), *manifest.files[1:]])
    if form == "model":
        subject: object = _manifest_with_files(divergent)
    else:
        subject = valid_manifest()
        subject["files"] = divergent  # type: ignore[index]
    with pytest.raises(ValidationError):
        validate_portable_manifest(subject)  # type: ignore[arg-type]
    assert divergent.iterations == 1  # observed once, never re-read


def test_an_honest_list_subclass_validates_in_both_input_forms() -> None:
    manifest = validate_portable_manifest(valid_manifest())
    honest = _DivergentList(list(manifest.files), list(manifest.files))
    document = valid_manifest()
    document["files"] = honest
    assert validate_portable_manifest(document).files[0].path == manifest.files[0].path
    assert validate_portable_manifest(_manifest_with_files(_DivergentList(list(manifest.files), list(manifest.files)))).files[0].path == manifest.files[0].path


class _DivergentDict(dict):
    """Valid native entries, but the public items() view differs."""

    def __init__(self, native, public) -> None:
        dict.__init__(self, native)
        self.public = public

    def items(self):
        return self.public.items()


def test_a_dict_subclass_in_a_mapping_input_is_validated_by_its_public_view() -> None:
    scope = valid_manifest()["scope"]
    document = valid_manifest()
    document["scope"] = _DivergentDict(scope, {**scope, "tenant_id": ""})
    with pytest.raises(ValidationError):
        validate_portable_manifest(document)


def test_an_endless_public_view_inside_a_mapping_input_is_observed_and_cut_off_by_the_budget() -> None:
    produced = [0]

    class Endless(list):
        def __iter__(self):
            def forever():
                while True:
                    produced[0] += 1
                    yield produced[0]

            return forever()

    document = valid_manifest()
    document["files"] = Endless()
    with pytest.raises(ValidationError):
        validate_portable_manifest(document)
    # the public view is observed (charged as it is read) and stopped by the discovery budget; the native,
    # empty backing used to be copied without ever asking the view
    assert 0 < produced[0] <= PORTABILITY_MAX_EDGES + 1


# --- a declared file cannot also be the directory of another declared file -------------------------


def _manifest_with_paths(paths: list[str]) -> dict:
    document = valid_manifest()
    base = document["files"][0]
    document["files"] = [
        {**base, "path": path, "checksum": dict(base["checksum"])} for path in [*paths, "crypto/envelope.bin"]
    ]
    return document


@pytest.mark.parametrize(
    "paths",
    [["node", "node/child"], ["Node", "node/child"], ["a", "a/b/c"], ["x/y", "x/y/z/w"], ["X/Y", "x/y/z"]],
    ids=["exact", "folded", "deep", "nested", "nested-folded"],
)
def test_a_declared_file_that_is_the_directory_of_another_is_refused(paths) -> None:
    with pytest.raises(ValidationError) as raised:
        validate_portable_manifest(_manifest_with_paths(paths))
    assert "also a directory" in str(raised.value.__cause__ or raised.value)


def test_declared_paths_that_only_share_a_prefix_are_accepted() -> None:
    assert validate_portable_manifest(_manifest_with_paths(["a/b", "a/c", "a2", "a/b2/c"]))


def test_the_same_conflict_is_refused_inside_a_manifest_model() -> None:
    manifest = validate_portable_manifest(_manifest_with_paths(["a/b", "a2"]))
    conflicting = manifest.files[0].model_copy(update={"path": "a"})
    with pytest.raises(ValidationError):
        validate_portable_manifest(manifest.model_copy(update={"files": [conflicting, *manifest.files]}))


def test_the_ancestor_check_does_work_linear_in_the_components_not_the_depth_squared() -> None:
    from contracts_portability import names_below_a_file

    calls = [0]

    def counting(component: str) -> str:
        calls[0] += 1
        return component

    paths = ["/".join(["a"] * 1_000) + f"/x{i}" for i in range(50)]
    assert names_below_a_file(paths, paths, counting) == []
    components = 50 * 1_001
    assert calls[0] <= 2 * components + 10, calls[0]  # once to build the trie, once to walk each name
    # and the answer is right: a file below a file is reported, a sibling is not
    assert names_below_a_file(["a", "a/b", "a2"], ["a", "a/b", "a2"]) == ["a/b"]
    assert names_below_a_file(["A/b"], ["a"], str.lower) == ["A/b"]


# --- a key that is no text cannot overwrite a text key's value ------------------------------------


class _AliasKey:
    """Not a string, but hashes and compares equal to ``"path"``."""

    def __hash__(self) -> int:
        return hash("path")

    def __eq__(self, other: object) -> bool:
        return other == "path" or other is self


class _TwoKeyItems(Mapping):
    """A custom mapping whose items() yields an unsafe value under "path", then an alias key with a safe one."""

    def __init__(self, base: dict) -> None:
        self.base = base

    def __getitem__(self, key):
        raise KeyError(key)

    def __iter__(self):
        return iter(())

    def __len__(self) -> int:
        return 0

    def items(self):
        rest = {key: value for key, value in self.base.items() if key != "path"}
        yield ("path", "../escape.txt")
        yield (_AliasKey(), "safe.txt")  # would replace the unsafe value and keep the first key
        yield from rest.items()


def test_an_alias_key_cannot_replace_an_unsafe_value_before_the_schema_sees_it() -> None:
    document = valid_manifest()
    document["files"] = [_TwoKeyItems(document["files"][0]), *document["files"][1:]]
    with pytest.raises(ValidationError):
        validate_portable_manifest(document)


def test_non_text_keys_are_refused_and_text_keys_still_work() -> None:
    document = valid_manifest()
    document["scope"] = {**document["scope"], 7: "x"}
    with pytest.raises(ValidationError):
        validate_portable_manifest(document)
    assert validate_portable_manifest(valid_manifest())


class _PathThatLies(str):
    """Its text is one path; its own string methods describe another."""

    def __contains__(self, item: object) -> bool:
        return False

    def split(self, *args: object, **kwargs: object) -> list[str]:
        return ["canonical", "a.txt"]

    def startswith(self, *args: object, **kwargs: object) -> bool:
        return False

    def __str__(self) -> str:
        return "canonical/a.txt"

    def __fspath__(self) -> str:
        return "canonical/a.txt"


@pytest.mark.parametrize("text", ["../../etc/passwd", "canonical/a\x00.txt", "/etc/passwd", "canonical\\a.txt"])
def test_the_public_path_normalizer_judges_a_str_subclass_by_its_own_text(text: str) -> None:
    with pytest.raises(ValueError):
        normalize_bundle_path(_PathThatLies(text))


def test_the_public_path_normalizer_returns_exact_text_for_a_str_subclass() -> None:
    class Plain(str):
        pass

    normalized = normalize_bundle_path(Plain("canonical/./a.txt"))
    assert normalized == "canonical/a.txt" and type(normalized) is str


# ---- model storage names: read as exact text at capture; a non-text name or a repeated text is refused


class _ShapeShiftingName:
    """Not text; unequal to ``"path"`` (and hashed apart) while it is stored, then equal with the same hash."""

    def __init__(self) -> None:
        self.equal = False
        self.calls = 0

    def __hash__(self) -> int:
        if self.equal:
            self.calls += 1
            return hash("path")
        return hash("path") ^ 0x5BD1E995

    def __eq__(self, other: object) -> bool:
        if self.equal:
            self.calls += 1
            return other == "path" or other is self
        return other is self


class _ShiftyName(str):
    """Equal to its text, hashed apart until ``settle()``; then it hashes like its text."""

    def __init__(self, value: str) -> None:
        self.settled = False
        self.calls = 0

    def __hash__(self) -> int:
        if self.settled:
            self.calls += 1
            return str.__hash__(self)
        return str.__hash__(self) ^ 0x5BD1E995

    def __eq__(self, other: object) -> bool:
        if self.settled:
            self.calls += 1
        return str.__eq__(self, other)


def _file_with_stored_twin(twin, *, unsafe_under_twin: bool) -> ManifestFile:
    """A ManifestFile whose native storage holds "path" and ``twin`` (stored while unequal), one value unsafe."""

    file = validate_portable_manifest(valid_manifest()).files[0].model_copy()
    native = _PYDANTIC_DICT_DESCRIPTOR.__get__(file, BaseModel)
    exact_value, twin_value = ("safe.txt", "../escape.txt") if unsafe_under_twin else ("../escape.txt", "safe.txt")
    dict.__setitem__(native, "path", exact_value)
    dict.__setitem__(native, twin, twin_value)
    assert dict.__len__(native) == len(ManifestFile.model_fields) + 1
    return file


def _settle(twin) -> None:
    if isinstance(twin, _ShapeShiftingName):
        twin.equal = True
    else:
        twin.settled = True


def _inputs_holding(file: ManifestFile):
    """The file inside a model graph and inside a mapping document (both routes reach model storage)."""

    manifest = validate_portable_manifest(valid_manifest()).model_copy(deep=True)
    files = list(manifest.files)
    files[0] = file
    dict.__setitem__(_PYDANTIC_DICT_DESCRIPTOR.__get__(manifest, BaseModel), "files", files)
    document = valid_manifest()
    document["files"] = [file, *document["files"][1:]]
    return {"model": manifest, "mapping": document}


@pytest.mark.parametrize("route", ["model", "mapping"])
@pytest.mark.parametrize("unsafe_under_twin", [False, True], ids=["unsafe-under-path", "unsafe-under-twin"])
@pytest.mark.parametrize("twin_kind", ["non-text", "str-subclass"])
def test_a_stored_twin_of_a_field_name_cannot_hide_an_unsafe_value(route, unsafe_under_twin, twin_kind) -> None:
    twin = _ShapeShiftingName() if twin_kind == "non-text" else _ShiftyName("path")
    file = _file_with_stored_twin(twin, unsafe_under_twin=unsafe_under_twin)
    target = _inputs_holding(file)[route]
    _settle(twin)
    with pytest.raises(ValidationError):
        validate_portable_manifest(target)
    assert twin.calls == 0  # the name is never hashed or compared: it is read once, natively, at capture


def test_a_str_subclass_stored_name_is_read_as_its_text_without_hooks() -> None:
    file = validate_portable_manifest(valid_manifest()).files[0].model_copy()
    native = _PYDANTIC_DICT_DESCRIPTOR.__get__(file, BaseModel)
    value = dict.__getitem__(native, "path")
    dict.__delitem__(native, "path")
    name = _ShiftyName("path")
    dict.__setitem__(native, name, value)
    _settle(name)  # it now hashes like its text: an ordinary-looking subclass name
    for target in _inputs_holding(file).values():
        name.calls = 0
        assert validate_portable_manifest(target).files[0].path == value
        assert name.calls == 0


def test_a_non_text_stored_name_alone_is_refused() -> None:
    file = validate_portable_manifest(valid_manifest()).files[0].model_copy()
    dict.__setitem__(_PYDANTIC_DICT_DESCRIPTOR.__get__(file, BaseModel), 7, "x")
    for target in _inputs_holding(file).values():
        with pytest.raises(ValidationError):
            validate_portable_manifest(target)


@pytest.mark.parametrize("twin_kind", ["non-text", "str-subclass"])
def test_extras_with_a_twin_name_are_refused_without_hooks(twin_kind) -> None:
    file = validate_portable_manifest(valid_manifest()).files[0].model_copy()
    twin = _ShapeShiftingName() if twin_kind == "non-text" else _ShiftyName("path")
    _PYDANTIC_EXTRA_SLOT.__set__(file, {"note": "a", twin: "b"})
    _settle(twin)
    for target in _inputs_holding(file).values():
        with pytest.raises(ValidationError):
            validate_portable_manifest(target)
    assert twin.calls == 0


class _ExtrasView(Mapping):
    """A custom extras mapping that yields its names and pairs without any key operation of its own."""

    def __init__(self, pairs, names=None) -> None:
        self.pairs = list(pairs)
        self.names = [name for name, _ in self.pairs] if names is None else list(names)

    def __getitem__(self, key):
        raise KeyError(key)

    def __iter__(self):
        return iter(self.names)

    def __len__(self) -> int:
        return len(self.pairs)

    def items(self):
        return iter(self.pairs)


@pytest.mark.parametrize("where", ["both", "items-only", "iteration-only"])
@pytest.mark.parametrize("twin_kind", ["non-text", "str-subclass"])
def test_custom_mapping_extras_with_a_twin_name_are_refused_without_hooks(twin_kind, where) -> None:
    """The names a custom extras mapping iterates and the names its items() yields are each read as text."""

    file = validate_portable_manifest(valid_manifest()).files[0].model_copy()
    twin = _ShapeShiftingName() if twin_kind == "non-text" else _ShiftyName("note")
    pairs = [("note", "a"), (twin, "b")] if where != "iteration-only" else [("note", "a")]
    names = ["note", twin] if where != "items-only" else ["note"]
    _PYDANTIC_EXTRA_SLOT.__set__(file, _ExtrasView(pairs, names))
    _settle(twin)
    for target in _inputs_holding(file).values():
        with pytest.raises(ValidationError):
            validate_portable_manifest(target)
    assert twin.calls == 0


def test_a_stored_name_is_copied_with_the_base_method_not_its_own_str() -> None:
    calls: list[str] = []

    class LyingName(str):
        def __str__(self) -> str:
            calls.append("__str__")
            return "something-else"

    file = validate_portable_manifest(valid_manifest()).files[0].model_copy()
    native = _PYDANTIC_DICT_DESCRIPTOR.__get__(file, BaseModel)
    value = dict.__getitem__(native, "path")
    dict.__delitem__(native, "path")
    dict.__setitem__(native, LyingName("path"), value)
    for target in _inputs_holding(file).values():
        assert validate_portable_manifest(target).files[0].path == value
    assert calls == []


def test_a_stored_name_that_only_claims_to_be_text_is_refused_as_no_text() -> None:
    class Claims:
        @property
        def __class__(self):  # type: ignore[override]
            return str

        def __hash__(self) -> int:
            return hash("note")

    file = validate_portable_manifest(valid_manifest()).files[0].model_copy()
    dict.__setitem__(_PYDANTIC_DICT_DESCRIPTOR.__get__(file, BaseModel), Claims(), "x")
    for target in _inputs_holding(file).values():
        with pytest.raises(ValidationError) as raised:
            validate_portable_manifest(target)
        assert "names must be text" in str(raised.value.__cause__)


# ---- fields-set names are read as exact text (the twin of a field name cannot make an unknown name disappear)


class _FieldsSetTwin(str):
    """Its text is ``phantom_only``; once settled it hashes and compares like ``path``."""

    def __init__(self, value: str) -> None:
        self.settled = False
        self.calls = 0

    def __hash__(self) -> int:
        if self.settled:
            self.calls += 1
            return hash("path")
        return str.__hash__(self)

    def __eq__(self, other: object) -> bool:
        if self.settled:
            self.calls += 1
            return other == "path" or other is self
        return str.__eq__(self, other)


@pytest.mark.parametrize("storage", ["native-set", "frozenset", "tuple", "list"])
def test_a_fields_set_twin_of_a_field_name_cannot_hide_an_unknown_name(storage) -> None:
    file = validate_portable_manifest(valid_manifest()).files[0].model_copy()
    twin = _FieldsSetTwin("phantom_only")
    names = set.copy(_PYDANTIC_FIELDS_SET_SLOT.__get__(file, type(file)))
    assert "path" in names
    if storage == "native-set":
        _PYDANTIC_FIELDS_SET_SLOT.__get__(file, type(file)).add(twin)  # stored while unequal to "path"
        twin.settled = True
    else:
        names.add(twin)
        twin.settled = True
        replacement = {"frozenset": frozenset, "tuple": tuple, "list": list}[storage](names)
        _PYDANTIC_FIELDS_SET_SLOT.__set__(file, replacement)
    manifest = validate_portable_manifest(valid_manifest()).model_copy(deep=True)
    list.__setitem__(manifest.files, 0, file)
    document = valid_manifest()
    document["files"] = [file, *document["files"][1:]]
    for target in (manifest, document):
        with pytest.raises(ValidationError) as raised:
            validate_portable_manifest(target)
        assert any(error["type"] == "extra_forbidden" and error["loc"][-1] == "phantom_only" for error in raised.value.errors())
    assert twin.calls == 0  # read once as text; its own hash and equality never run


def test_a_non_text_fields_set_name_is_refused_before_it_joins_a_set() -> None:
    manifest = validate_portable_manifest(valid_manifest()).model_copy(deep=True)
    _PYDANTIC_FIELDS_SET_SLOT.__get__(manifest, type(manifest)).add(7)
    with pytest.raises(ValidationError) as raised:
        validate_portable_manifest(manifest)
    assert "names must be text" in str(raised.value.__cause__) or "text" in str(raised.value)


def test_ordinary_fields_set_names_still_validate_as_text() -> None:
    class Plain(str):
        pass

    manifest = validate_portable_manifest(valid_manifest()).model_copy(deep=True)
    fields = _PYDANTIC_FIELDS_SET_SLOT.__get__(manifest, type(manifest))
    fields.discard("manifest_id")
    fields.add(Plain("manifest_id"))
    assert validate_portable_manifest(manifest).manifest_id == "manifest-7"


# ---- the declared fields of a model class are charged to the discovery budget for every instance scanned

def test_sparse_instances_of_a_class_with_many_fields_are_charged_for_the_field_names() -> None:
    from pydantic import create_model

    wide = create_model("Wide", __base__=ManifestFile, **{f"extra_{i}": (int | None, None) for i in range(8_000)})
    manifest = validate_portable_manifest(valid_manifest()).model_copy(deep=True)
    files = manifest.files
    for _ in range(100):  # unchecked instances: nothing stored, so only the field enumeration costs anything
        sparse = object.__new__(wide)
        _PYDANTIC_DICT_DESCRIPTOR.__set__(sparse, {})
        _PYDANTIC_FIELDS_SET_SLOT.__set__(sparse, set())
        _PYDANTIC_EXTRA_SLOT.__set__(sparse, None)
        list.append(files, sparse)
    with pytest.raises(ValidationError) as raised:
        validate_portable_manifest(manifest)
    assert str(raised.value.__cause__) == DISCOVERY_BUDGET_MESSAGE  # 100 x 8,000 names is over the edge budget


def test_a_few_instances_of_a_wide_class_stay_within_the_budget() -> None:
    from pydantic import create_model

    wide = create_model("Wide", __base__=ManifestFile, **{f"extra_{i}": (int | None, None) for i in range(8_000)})
    manifest = validate_portable_manifest(valid_manifest()).model_copy(deep=True)
    sparse = object.__new__(wide)
    _PYDANTIC_DICT_DESCRIPTOR.__set__(sparse, {})
    _PYDANTIC_FIELDS_SET_SLOT.__set__(sparse, set())
    _PYDANTIC_EXTRA_SLOT.__set__(sparse, None)
    list.append(manifest.files, sparse)
    with pytest.raises(ValidationError) as raised:
        validate_portable_manifest(manifest)
    assert str(raised.value.__cause__) != DISCOVERY_BUDGET_MESSAGE  # refused for its content, not its budget


# ---- the declared-field table is found by scanning the class namespace: a planted twin key never runs code

class _TwinNamespaceKey(str):
    """Hashes like ``__pydantic_fields__``; its equality is caller code (a hook, armed after the model exists)."""

    hook = None
    armed = False
    calls = 0

    def __hash__(self) -> int:
        return str.__hash__("__pydantic_fields__")

    def __eq__(self, other: object) -> bool:
        if type(self).armed:
            type(self).calls += 1
            if type(self).hook is not None:
                type(self).hook()
        return False


def _planted_manifest():
    """A manifest whose first file is an instance of a class with a planted twin namespace key, and whose
    second file is invalid until a hook (run by the twin's equality) repairs it."""

    twin_class = type("Twin", (_TwinNamespaceKey,), {"hook": None, "armed": False, "calls": 0})
    twin = twin_class("__pydantic_fields__")

    class Meta(type(ManifestFile)):
        @classmethod
        def __prepare__(mcls, name, bases, **kwargs):
            namespace = super().__prepare__(name, bases, **kwargs)
            namespace[twin] = None
            return namespace

    class Planted(ManifestFile, metaclass=Meta):  # a class statement, so the metaclass's __prepare__ runs
        pass

    manifest = validate_portable_manifest(valid_manifest()).model_copy(deep=True)
    good = manifest.files[0]
    planted = object.__new__(Planted)
    _PYDANTIC_DICT_DESCRIPTOR.__set__(planted, dict(_PYDANTIC_DICT_DESCRIPTOR.__get__(good, BaseModel)))
    _PYDANTIC_FIELDS_SET_SLOT.__set__(planted, set(_PYDANTIC_FIELDS_SET_SLOT.__get__(good, type(good))))
    _PYDANTIC_EXTRA_SLOT.__set__(planted, None)
    bad = good.model_copy(update={"path": "../escape"})
    list.__setitem__(manifest.files, 0, planted)
    list.append(manifest.files, bad)

    def repair() -> None:
        dict.__setitem__(_PYDANTIC_DICT_DESCRIPTOR.__get__(bad, BaseModel), "path", "sources/note.md")

    twin_class.hook = staticmethod(repair)
    return manifest, twin_class, bad


def test_a_planted_twin_namespace_key_cannot_run_code_or_repair_a_later_file() -> None:
    manifest, twin_class, bad = _planted_manifest()
    with pytest.raises(ValidationError):  # the unarmed control: the invalid second file is refused
        validate_portable_manifest(manifest)
    twin_class.armed = True
    with pytest.raises(ValidationError):  # armed, the hook would repair it if it ran
        validate_portable_manifest(manifest)
    assert twin_class.calls == 0  # the namespace is scanned, never probed by name, so the twin is never compared
    assert dict.__getitem__(_PYDANTIC_DICT_DESCRIPTOR.__get__(bad, BaseModel), "path") == "../escape"


def test_an_honest_manifest_runs_no_namespace_key_code() -> None:
    manifest, twin_class, bad = _planted_manifest()
    list.pop(manifest.files)  # the invalid file removed: only the planted instance remains
    twin_class.armed = True
    assert validate_portable_manifest(manifest).files[0].path == "canonical/records.jsonl"
    assert twin_class.calls == 0


def test_the_declared_field_table_is_charged_once_per_class_per_call_not_per_instance() -> None:
    manifest = validate_portable_manifest(valid_manifest()).model_copy(deep=True)
    for _ in range(50):
        list.append(manifest.files, manifest.files[0].model_copy(update={"path": f"canonical/extra-{_}.bin"}))
    assert len(validate_portable_manifest(manifest).files) == len(manifest.files)


@pytest.mark.parametrize(("instances", "over_budget"), [(40, False), (60, True)])
def test_the_declared_field_charge_counts_both_scans_of_every_instance(instances, over_budget) -> None:
    """One instance of an 8,000-field class costs 16,000 edges (the discovery scan and the walk's), so 40 fit
    under the 786,432-edge tier and 60 do not; half a charge (one scan) would let 60 through."""

    from pydantic import create_model

    wide = create_model("Wide", __base__=ManifestFile, **{f"extra_{i}": (int | None, None) for i in range(8_000)})
    manifest = validate_portable_manifest(valid_manifest()).model_copy(deep=True)
    for _ in range(instances):
        sparse = object.__new__(wide)
        _PYDANTIC_DICT_DESCRIPTOR.__set__(sparse, {})
        _PYDANTIC_FIELDS_SET_SLOT.__set__(sparse, set())
        _PYDANTIC_EXTRA_SLOT.__set__(sparse, None)
        list.append(manifest.files, sparse)
    with pytest.raises(ValidationError) as raised:
        validate_portable_manifest(manifest)
    assert (str(raised.value.__cause__) == DISCOVERY_BUDGET_MESSAGE) is over_budget


# ---- classification is by native class identity, and the depth outcome does not depend on sibling order

def _count_model_class_checks(monkeypatch):
    """Counts the ``isinstance``/``issubclass`` checks against model classes that ``contracts_portability`` itself
    makes: the checks dispatch through the model metaclass, which a caller can replace or extend."""

    import sys

    calls: list[str] = []
    metaclass = type(PortableManifest)
    original = {name: getattr(metaclass, name) for name in ("__instancecheck__", "__subclasscheck__")}

    def counting(name):
        def check(cls, other):
            if sys._getframe(1).f_globals.get("__name__") == "contracts_portability":
                calls.append(name)
            return original[name](cls, other)

        return check

    for name in original:
        monkeypatch.setattr(metaclass, name, counting(name))
    return calls


def test_discovery_and_dispatch_classify_by_native_class_identity(monkeypatch) -> None:
    calls = _count_model_class_checks(monkeypatch)
    manifest = validate_portable_manifest(valid_manifest()).model_copy(deep=True)
    assert validate_portable_manifest(manifest).manifest_id == "manifest-7"
    assert validate_portable_manifest(valid_manifest()).manifest_id == "manifest-7"  # the mapping route too
    assert calls == [], calls  # none of the classification went through the metaclass check hooks


def test_classification_reads_no_attribute_of_a_hooked_model_instance() -> None:
    reads: list[str] = []

    class Spoofing(PortableManifest):
        def __getattribute__(self, name):
            reads.append(name)
            return super().__getattribute__(name)

    manifest = validate_portable_manifest(valid_manifest())
    spoofing = object.__new__(Spoofing)
    _PYDANTIC_DICT_DESCRIPTOR.__set__(spoofing, dict(_PYDANTIC_DICT_DESCRIPTOR.__get__(manifest, BaseModel)))
    _PYDANTIC_FIELDS_SET_SLOT.__set__(spoofing, set(_PYDANTIC_FIELDS_SET_SLOT.__get__(manifest, type(manifest))))
    _PYDANTIC_EXTRA_SLOT.__set__(spoofing, None)
    del reads[:]
    validate_portable_manifest(spoofing)
    assert reads == [], reads


@pytest.mark.parametrize("shared_first", [True, False])
def test_the_depth_outcome_does_not_depend_on_sibling_order(monkeypatch, shared_first) -> None:
    """Depth is judged at each node's minimum depth: ``[shared, wrapper]`` and ``[wrapper, shared]`` agree."""

    from contracts_limits import DiscoveryTier

    monkeypatch.setattr(contracts_portability, "PORTABILITY", DiscoveryTier("tiny", 1_000, 100_000, 3))
    shared = [[]]
    wrapper = [shared]
    manifest = validate_portable_manifest(valid_manifest()).model_copy(deep=True)
    dict.__setitem__(_PYDANTIC_DICT_DESCRIPTOR.__get__(manifest, BaseModel), "files", [shared, wrapper] if shared_first else [wrapper, shared])
    with pytest.raises(ValidationError) as raised:
        validate_portable_manifest(manifest)
    assert str(raised.value.__cause__) != DISCOVERY_BUDGET_MESSAGE  # refused for its content, never for depth


def test_a_node_that_is_deep_on_every_path_is_still_refused_for_depth(monkeypatch) -> None:
    from contracts_limits import DiscoveryTier

    monkeypatch.setattr(contracts_portability, "PORTABILITY", DiscoveryTier("tiny", 1_000, 100_000, 3))
    deep = [[[[]]]]  # four levels below the field's list: over the limit on its only path
    manifest = validate_portable_manifest(valid_manifest()).model_copy(deep=True)
    dict.__setitem__(_PYDANTIC_DICT_DESCRIPTOR.__get__(manifest, BaseModel), "files", [deep])
    with pytest.raises(ValidationError) as raised:
        validate_portable_manifest(manifest)
    assert str(raised.value.__cause__) == DISCOVERY_BUDGET_MESSAGE


@pytest.mark.parametrize("limit", [3, 4, 5, 6])
def test_every_sibling_permutation_gets_the_same_depth_outcome(monkeypatch, limit) -> None:
    import itertools

    from contracts_limits import DiscoveryTier

    monkeypatch.setattr(contracts_portability, "PORTABILITY", DiscoveryTier("tiny", 1_000, 100_000, limit))
    chain = [[[]]]
    shared = [chain]
    siblings = {"shared": shared, "wrapper": [shared], "deeper": [[shared]], "chain": chain, "leaf": []}
    outcomes = set()
    for order in itertools.permutations(siblings):
        manifest = validate_portable_manifest(valid_manifest()).model_copy(deep=True)
        dict.__setitem__(_PYDANTIC_DICT_DESCRIPTOR.__get__(manifest, BaseModel), "files", [siblings[name] for name in order])
        with pytest.raises(ValidationError) as raised:
            validate_portable_manifest(manifest)
        outcomes.add(str(raised.value.__cause__) == DISCOVERY_BUDGET_MESSAGE)
    assert len(outcomes) == 1, outcomes  # budget refusal for all orders, or for none
