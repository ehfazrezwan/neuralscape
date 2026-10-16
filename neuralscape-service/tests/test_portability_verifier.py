"""Adversarial and positive tests for the offline portable-bundle verifier."""

import hashlib
import io
import json
import os
import stat
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

import portability_verifier as pv
from portability_verifier import verify_bundle


def _manifest(files: dict[str, bytes], **overrides) -> dict:
    document = {
        "schema_version": "candidate-v1",
        "format_version": "portable-candidate-v1",
        "manifest_id": "manifest-1",
        "producer": {"implementation": "reference-exporter", "version": "1.0"},
        "scope": {"kind": "tenant", "tenant_id": "tenant-a", "project_ids": []},
        "snapshot": {
            "snapshot_id": "snapshot-1",
            "snapshot_position": {"stream_id": "ledger", "position": "10"},
            "completion_barrier": {"stream_id": "ledger", "position": "10"},
            "replay_position": {"stream_id": "intents", "position": "4"},
        },
        "files": [
            {
                "path": path,
                "size_bytes": len(data),
                "checksum": {"algorithm": "sha256", "digest": hashlib.sha256(data).hexdigest()},
                "entry_type": "file",
            }
            for path, data in files.items()
        ],
        "canonical_ids": [{"kind": "record", "id": "record-1"}],
        "included_capabilities": ["canonical-records"],
        "omissions": [{"code": "model-specific-projections", "reason": "not selected"}],
        "identity_mappings": [],
        "encryption": {
            "mode": "plaintext_authorized_export",
            "protocol_reference": None,
            "key_envelope_paths": [],
        },
    }
    document.update(overrides)
    return document


PAYLOAD = {
    "canonical/records.jsonl": b'{"id":"record-1"}\n' * 50,
    "sources/note.md": "# original\ncontent\n".encode(),
}


def _write_zip(path: Path, files: dict[str, bytes], manifest: dict | None = None) -> Path:
    manifest = _manifest(PAYLOAD) if manifest is None else manifest
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", json.dumps(manifest))
        for name, data in files.items():
            archive.writestr(name, data)
    return path


def _write_dir(root: Path, files: dict[str, bytes], manifest: dict | None = None) -> Path:
    manifest = _manifest(PAYLOAD) if manifest is None else manifest
    root.mkdir()
    (root / "manifest.json").write_text(json.dumps(manifest))
    for name, data in files.items():
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    return root


def _codes(report) -> set[str]:
    return {finding.code for finding in report.findings}


@pytest.fixture(params=["zip", "dir"])
def build(request, tmp_path):
    def _build(files=None, manifest=None):
        files = PAYLOAD if files is None else files
        if request.param == "zip":
            return _write_zip(tmp_path / "bundle.zip", files, manifest)
        return _write_dir(tmp_path / "bundle", files, manifest)

    return _build


# --- positive control -------------------------------------------------------


def test_valid_bundle_verifies(build) -> None:
    report = verify_bundle(build())
    assert report.ok, report.findings
    assert report.files_declared == report.files_verified == 2
    assert report.omissions == ["model-specific-projections"]
    assert report.included_capabilities == ["canonical-records"]
    assert "not claimed" in report.to_dict()["scope_note"]


# --- content failures -------------------------------------------------------


def test_single_byte_flip_is_a_digest_mismatch_only(build) -> None:
    tampered = dict(PAYLOAD)
    data = bytearray(tampered["sources/note.md"])
    data[3] ^= 0x01
    tampered["sources/note.md"] = bytes(data)
    report = verify_bundle(build(tampered))
    assert [(f.code, f.path) for f in report.findings] == [
        (pv.DIGEST_MISMATCH, "sources/note.md")
    ]


def test_wrong_length_is_size_mismatch(build) -> None:
    tampered = dict(PAYLOAD)
    tampered["sources/note.md"] += b"extra"
    assert _codes(verify_bundle(build(tampered))) == {pv.SIZE_MISMATCH}


def test_missing_declared_file(build) -> None:
    files = {k: v for k, v in PAYLOAD.items() if k != "sources/note.md"}
    report = verify_bundle(build(files))
    assert [(f.code, f.path) for f in report.findings] == [
        (pv.FILE_MISSING, "sources/note.md")
    ]


def test_undeclared_extra_file(build) -> None:
    report = verify_bundle(build({**PAYLOAD, "stray/extra.txt": b"x"}))
    assert [(f.code, f.path) for f in report.findings] == [
        (pv.FILE_UNDECLARED, "stray/extra.txt")
    ]


def test_manifest_listing_itself_is_rejected(build) -> None:
    manifest = _manifest({**PAYLOAD, "manifest.json": b"{}"})
    report = verify_bundle(build(manifest=manifest))
    assert pv.MANIFEST_SELF_LISTED in _codes(report)


def test_manifest_missing(tmp_path) -> None:
    path = tmp_path / "b.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("a.txt", b"x")
    assert _codes(verify_bundle(path)) == {pv.MANIFEST_MISSING}


@pytest.mark.parametrize(
    "text",
    ["not json", "[]", '{"schema_version":"candidate-v1","schema_version":"candidate-v1"}', "{}"],
)
def test_invalid_manifest(tmp_path, text) -> None:
    path = tmp_path / "b.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("manifest.json", text)
    assert _codes(verify_bundle(path)) == {pv.MANIFEST_INVALID}


def test_manifest_with_unsafe_declared_path_is_invalid(tmp_path) -> None:
    manifest = _manifest(PAYLOAD)
    manifest["files"][0]["path"] = "../escape"
    path = _write_zip(tmp_path / "b.zip", PAYLOAD, manifest)
    assert _codes(verify_bundle(path)) == {pv.MANIFEST_INVALID}


# --- container failures -----------------------------------------------------


def test_truncated_zip(tmp_path) -> None:
    path = _write_zip(tmp_path / "b.zip", PAYLOAD)
    path.write_bytes(path.read_bytes()[:-40])
    assert _codes(verify_bundle(path)) == {pv.CONTAINER_UNREADABLE}


def test_garbage_and_nonexistent_input(tmp_path) -> None:
    junk = tmp_path / "junk"
    junk.write_bytes(b"\x00" * 64)
    assert _codes(verify_bundle(junk)) == {pv.CONTAINER_UNREADABLE}
    assert _codes(verify_bundle(tmp_path / "absent")) == {pv.CONTAINER_UNREADABLE}


def test_corrupt_payload_bytes_in_zip_are_entry_corrupt(tmp_path) -> None:
    path = _write_zip(tmp_path / "b.zip", PAYLOAD)
    raw = bytearray(path.read_bytes())
    with zipfile.ZipFile(path) as archive:
        info = archive.getinfo("canonical/records.jsonl")
        data_start = (
            info.header_offset
            + 30
            + len(info.filename.encode())
            + int.from_bytes(raw[info.header_offset + 28 : info.header_offset + 30], "little")
        )
    raw[data_start + 2] ^= 0xFF
    path.write_bytes(bytes(raw))
    report = verify_bundle(path)
    assert not report.ok
    assert _codes(report) <= {pv.ENTRY_CORRUPT, pv.DIGEST_MISMATCH}


@pytest.mark.parametrize(
    "name",
    [
        "../escape.txt",
        "/abs.txt",
        "a/../../b.txt",
        "back\\slash.txt",
        "nul\x00.txt",
        "C:/drive.txt",
        "dir/CON",
        "trailing./x",
    ],
)
def test_unsafe_names_in_zip(tmp_path, name) -> None:
    path = tmp_path / "b.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("manifest.json", json.dumps(_manifest({})))
        archive.writestr(zipfile.ZipInfo(name), b"x")
    assert pv.PATH_UNSAFE in _codes(verify_bundle(path))


def test_unsafe_name_in_directory(tmp_path) -> None:
    root = _write_dir(tmp_path / "b", {}, _manifest({}))
    (root / "back\\slash").write_bytes(b"x")
    assert pv.PATH_UNSAFE in _codes(verify_bundle(root))


def _zip_with_mode(tmp_path, mode: int) -> Path:
    path = tmp_path / "b.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("manifest.json", json.dumps(_manifest({})))
        info = zipfile.ZipInfo("entry")
        info.external_attr = mode << 16
        archive.writestr(info, b"target")
    return path


def test_symlink_entry_in_zip(tmp_path) -> None:
    path = _zip_with_mode(tmp_path, stat.S_IFLNK | 0o777)
    assert pv.SYMLINK_ENTRY in _codes(verify_bundle(path))


@pytest.mark.parametrize("kind", [stat.S_IFCHR, stat.S_IFBLK, stat.S_IFIFO, stat.S_IFSOCK])
def test_device_and_special_entries_in_zip(tmp_path, kind) -> None:
    path = _zip_with_mode(tmp_path, kind | 0o600)
    assert pv.SPECIAL_ENTRY in _codes(verify_bundle(path))


def test_symlink_in_directory(tmp_path) -> None:
    root = _write_dir(tmp_path / "b", PAYLOAD)
    os.symlink("/etc/passwd", root / "link")
    assert pv.SYMLINK_ENTRY in _codes(verify_bundle(root))


def test_symlinked_bundle_root_is_rejected(tmp_path) -> None:
    root = _write_dir(tmp_path / "real", PAYLOAD)
    os.symlink(root, tmp_path / "alias")
    assert _codes(verify_bundle(tmp_path / "alias")) == {pv.SYMLINK_ENTRY}


def test_hardlink_in_directory(tmp_path) -> None:
    root = _write_dir(tmp_path / "b", PAYLOAD)
    os.link(root / "sources/note.md", root / "sources/note-alias.md")
    codes = _codes(verify_bundle(root))
    assert pv.HARDLINK_ENTRY in codes


def test_fifo_in_directory(tmp_path) -> None:
    root = _write_dir(tmp_path / "b", PAYLOAD)
    os.mkfifo(root / "pipe")
    assert pv.SPECIAL_ENTRY in _codes(verify_bundle(root))


def test_encrypted_zip_entry_is_rejected(tmp_path) -> None:
    path = tmp_path / "b.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("manifest.json", json.dumps(_manifest({})))
        archive.writestr("secret", b"x")
        archive.infolist()[1].flag_bits |= 0x1
    assert pv.ENTRY_ENCRYPTED in _codes(verify_bundle(path))


def test_duplicate_zip_entries(tmp_path) -> None:
    path = tmp_path / "b.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("manifest.json", json.dumps(_manifest({})))
        with pytest.warns(UserWarning):
            archive.writestr("a.txt", b"one")
            archive.writestr("a.txt", b"two")
    assert pv.DUPLICATE_ENTRY in _codes(verify_bundle(path))


def test_dot_segment_alias_duplicates_normalize_together(tmp_path) -> None:
    path = tmp_path / "b.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("manifest.json", json.dumps(_manifest({})))
        archive.writestr("a/x.txt", b"one")
        archive.writestr("a/./x.txt", b"two")
    assert pv.DUPLICATE_ENTRY in _codes(verify_bundle(path))


@pytest.mark.parametrize(
    "names",
    [("a/File.txt", "a/file.txt"), ("caf\u00e9.txt", "cafe\u0301.txt"), ("Stra\u00dfe", "STRASSE")],
)
def test_case_fold_and_unicode_collisions_in_zip(tmp_path, names) -> None:
    path = tmp_path / "b.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("manifest.json", json.dumps(_manifest({})))
        for name in names:
            archive.writestr(name, b"x")
    assert pv.CASE_COLLISION in _codes(verify_bundle(path))


def test_case_collision_in_directory(tmp_path) -> None:
    root = _write_dir(tmp_path / "b", {}, _manifest({}))
    (root / "A.txt").write_bytes(b"1")
    (root / "a.txt").write_bytes(b"2")
    assert pv.CASE_COLLISION in _codes(verify_bundle(root))


# --- resource guards --------------------------------------------------------


def test_compression_bomb_ratio(tmp_path) -> None:
    bomb = b"\x00" * (4 * 1024**2)
    files = {**PAYLOAD, "bomb.bin": bomb}
    path = _write_zip(tmp_path / "b.zip", files, _manifest(files))
    report = verify_bundle(path)
    assert _codes(report) == {pv.RATIO_LIMIT}
    assert report.files_verified == 0


def test_entry_count_limit(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(pv, "MAX_BUNDLE_ENTRIES", 2)
    path = _write_zip(tmp_path / "b.zip", PAYLOAD)
    assert _codes(verify_bundle(path)) == {pv.ENTRY_COUNT_LIMIT}


def test_entry_and_total_size_limits(tmp_path, monkeypatch) -> None:
    path = _write_zip(tmp_path / "b.zip", PAYLOAD)
    monkeypatch.setattr(pv, "MAX_ENTRY_BYTES", 10)
    assert _codes(verify_bundle(path)) == {pv.SIZE_LIMIT}
    monkeypatch.undo()
    monkeypatch.setattr(pv, "MAX_TOTAL_BYTES", 10)
    assert _codes(verify_bundle(path)) == {pv.SIZE_LIMIT}


def test_directory_size_and_depth_limits(tmp_path, monkeypatch) -> None:
    root = _write_dir(tmp_path / "b", PAYLOAD)
    monkeypatch.setattr(pv, "MAX_TOTAL_BYTES", 10)
    assert _codes(verify_bundle(root)) == {pv.SIZE_LIMIT}
    monkeypatch.undo()
    monkeypatch.setattr(pv, "MAX_DIRECTORY_DEPTH", 0)
    assert _codes(verify_bundle(root)) == {pv.ENTRY_COUNT_LIMIT}


def test_overlapping_zip_entries(tmp_path) -> None:
    path = tmp_path / "b.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("manifest.json", json.dumps(_manifest({})))
        archive.writestr("a", b"a" * 100)
        archive.writestr("b", b"b" * 100)
        first, second = archive.infolist()[1:3]
        second.header_offset = first.header_offset + 5
    assert _codes(verify_bundle(path)) == {pv.OVERLAPPING_ENTRIES}


def test_header_understating_size_cannot_overrun_declared_length(tmp_path) -> None:
    files = {"a.bin": b"abcdefgh" * 100}
    manifest = _manifest(files)
    manifest["files"][0]["size_bytes"] = 100  # declared smaller than actual
    manifest["files"][0]["checksum"]["digest"] = hashlib.sha256(files["a.bin"][:100]).hexdigest()
    path = _write_zip(tmp_path / "b.zip", files, manifest)
    assert _codes(verify_bundle(path)) == {pv.SIZE_MISMATCH}


# --- "no payload read before structure" -------------------------------------


def test_payload_not_hashed_when_structure_is_bad(tmp_path, monkeypatch) -> None:
    path = _write_zip(tmp_path / "b.zip", {**PAYLOAD, "stray.txt": b"x"})
    hashed: list[str] = []
    original = pv._hash_entry
    monkeypatch.setattr(pv, "_hash_entry", lambda *a, **k: hashed.append("x") or original(*a, **k))
    assert pv.FILE_UNDECLARED in _codes(verify_bundle(path))
    assert hashed == []


def test_fatal_container_finding_reads_nothing(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(pv, "MAX_BUNDLE_ENTRIES", 1)
    monkeypatch.setattr(pv, "_read_chunks", lambda *a, **k: pytest.fail("payload read"))
    path = _write_zip(tmp_path / "b.zip", PAYLOAD)
    assert _codes(verify_bundle(path)) == {pv.ENTRY_COUNT_LIMIT}


# --- read-only / no extraction ----------------------------------------------


def _snapshot(root: Path) -> dict:
    return {
        str(p.relative_to(root)): p.read_bytes() if p.is_file() else None
        for p in sorted(root.rglob("*"))
    }


def test_verification_never_writes_or_extracts(tmp_path) -> None:
    work = tmp_path / "work"
    work.mkdir()
    _write_zip(work / "b.zip", PAYLOAD)
    _write_dir(work / "d", PAYLOAD)
    before = _snapshot(work)
    verify_bundle(work / "b.zip")
    verify_bundle(work / "d")
    assert _snapshot(work) == before
    assert sorted(os.listdir(work)) == ["b.zip", "d"]


def test_findings_do_not_echo_payload_content(tmp_path) -> None:
    secret = b"TOP-SECRET-CONTENT"
    files = {"a.txt": secret}
    manifest = _manifest(files)
    manifest["files"][0]["checksum"]["digest"] = "0" * 64
    report = verify_bundle(_write_zip(tmp_path / "b.zip", files, manifest))
    assert pv.DIGEST_MISMATCH in _codes(report)
    assert "TOP-SECRET" not in json.dumps(report.to_dict())


# --- CLI --------------------------------------------------------------------


def _run_cli(*args: str) -> subprocess.CompletedProcess:
    service_dir = Path(pv.__file__).resolve().parent
    return subprocess.run(
        [sys.executable, "-m", "portability_verifier", *args],
        cwd=service_dir,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_cli_exit_codes(tmp_path) -> None:
    good = _write_zip(tmp_path / "good.zip", PAYLOAD)
    bad = _write_zip(tmp_path / "bad.zip", {**PAYLOAD, "stray": b"x"})
    ok = _run_cli(str(good))
    assert ok.returncode == 0 and json.loads(ok.stdout)["ok"] is True
    failed = _run_cli(str(bad))
    assert failed.returncode == 1
    assert json.loads(failed.stdout)["findings"][0]["code"] == pv.FILE_UNDECLARED
    assert _run_cli().returncode == 2
    assert _run_cli("-x").returncode == 2
