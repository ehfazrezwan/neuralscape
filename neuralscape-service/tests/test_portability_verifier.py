"""Adversarial and positive tests for the offline portable-bundle verifier."""

import hashlib
import io
import json
import os
import stat
import struct
import subprocess
import sys
import zipfile
import zlib
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
    raw = bytearray(path.read_bytes())
    local = raw.index(b"secret") - 30
    raw[local + 6] |= 0x1  # a real encrypted member flags its local header too
    path.write_bytes(bytes(raw))
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


# --- archive framing and stream-vs-directory agreement ----------------------

_BIG = bytes(range(256)) * 32  # 8192 bytes


def _prefix_manifest(name: str, size: int) -> dict:
    manifest = _manifest({name: _BIG})
    manifest["files"][0]["size_bytes"] = size
    manifest["files"][0]["checksum"]["digest"] = hashlib.sha256(_BIG[:size]).hexdigest()
    return manifest


def _patch_member(
    raw: bytearray, name: str, *, local: bool, central: bool, **fields: int
) -> None:
    """Rewrite crc/csize/usize in the named member's local and/or central header."""

    encoded = name.encode()
    layouts = {"crc": (14, 16, "<I"), "csize": (18, 20, "<I"), "usize": (22, 24, "<I")}
    import struct

    if local:
        position = raw.index(b"PK\x03\x04" + b"\x00" * 0)
        while raw[position + 30 : position + 30 + len(encoded)] != encoded:
            position = raw.index(b"PK\x03\x04", position + 1)
        for key, value in fields.items():
            struct.pack_into(layouts[key][2], raw, position + layouts[key][0], value)
    if central:
        position = raw.index(b"PK\x01\x02")
        while raw[position + 46 : position + 46 + len(encoded)] != encoded:
            position = raw.index(b"PK\x01\x02", position + 1)
        for key, value in fields.items():
            struct.pack_into(layouts[key][2], raw, position + layouts[key][1], value)


@pytest.mark.parametrize("method", [zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED])
@pytest.mark.parametrize("patch_local", [False, True])
def test_central_size_lie_is_rejected(tmp_path, method, patch_local) -> None:
    path = tmp_path / "b.zip"
    with zipfile.ZipFile(path, "w", method) as archive:
        archive.writestr("manifest.json", json.dumps(_prefix_manifest("big.bin", 100)))
        archive.writestr("big.bin", _BIG)
    raw = bytearray(path.read_bytes())
    fields = {"usize": 100, "crc": zlib_crc(_BIG[:100])}
    if method == zipfile.ZIP_STORED:
        fields["csize"] = 100
    _patch_member(raw, "big.bin", local=patch_local, central=True, **fields)
    path.write_bytes(bytes(raw))
    report = verify_bundle(path)
    assert not report.ok
    assert _codes(report) <= {pv.ARCHIVE_FRAMING, pv.ENTRY_CORRUPT}
    assert report.files_verified == 0


def zlib_crc(data: bytes) -> int:
    import zlib

    return zlib.crc32(data)


def _good_zip_bytes(tmp_path) -> bytes:
    return _write_zip(tmp_path / "good.zip", PAYLOAD).read_bytes()


def test_junk_before_first_record_is_rejected(tmp_path) -> None:
    path = tmp_path / "b.zip"
    path.write_bytes(b"JUNKJUNK" + _good_zip_bytes(tmp_path))
    assert _codes(verify_bundle(path)) == {pv.ARCHIVE_FRAMING}


def test_trailing_bytes_after_end_record_are_rejected(tmp_path) -> None:
    path = tmp_path / "b.zip"
    path.write_bytes(_good_zip_bytes(tmp_path) + b"TRAILING")
    assert _codes(verify_bundle(path)) == {pv.ARCHIVE_FRAMING}


def test_archive_comment_is_rejected(tmp_path) -> None:
    path = _write_zip(tmp_path / "b.zip", PAYLOAD)
    with zipfile.ZipFile(path, "a") as archive:
        archive.comment = b"note"
    assert _codes(verify_bundle(path)) == {pv.ARCHIVE_FRAMING}


def test_comment_embedding_another_archive_is_rejected(tmp_path) -> None:
    good = _good_zip_bytes(tmp_path)
    path = _write_zip(tmp_path / "outer.zip", {**PAYLOAD, "stray": b"x"})
    with zipfile.ZipFile(path, "a") as archive:
        archive.comment = good
    report = verify_bundle(path)
    assert not report.ok
    assert pv.ARCHIVE_FRAMING in _codes(report)


def _plain_zip(path: Path, method: int = zipfile.ZIP_STORED) -> bytearray:
    with zipfile.ZipFile(path, "w", method) as archive:
        archive.writestr("manifest.json", json.dumps(_manifest(PAYLOAD)))
        for name, data in PAYLOAD.items():
            archive.writestr(name, data)
    return bytearray(path.read_bytes())


def _insert_bytes(raw: bytearray, position: int, junk: bytes) -> None:
    """Insert ``junk`` and repair every directory offset so zipfile still parses."""

    import struct

    raw[position:position] = junk
    eocd = raw.rindex(b"PK\x05\x06")
    cd_offset = struct.unpack_from("<I", raw, eocd + 16)[0]
    cursor = cd_offset + (len(junk) if cd_offset >= position else 0)
    while raw[cursor : cursor + 4] == b"PK\x01\x02":
        offset = struct.unpack_from("<I", raw, cursor + 42)[0]
        if offset >= position:
            struct.pack_into("<I", raw, cursor + 42, offset + len(junk))
        name_length, extra_length, comment_length = struct.unpack_from("<HHH", raw, cursor + 28)
        cursor += 46 + name_length + extra_length + comment_length
    if cd_offset >= position:
        struct.pack_into("<I", raw, eocd + 16, cd_offset + len(junk))


def _second_header_offset(path: Path) -> int:
    with zipfile.ZipFile(path) as archive:
        return archive.infolist()[1].header_offset


def test_gap_between_records_is_rejected(tmp_path) -> None:
    path = tmp_path / "b.zip"
    raw = _plain_zip(path)
    _insert_bytes(raw, _second_header_offset(path), b"HIDDEN-BYTES")
    path.write_bytes(bytes(raw))
    with zipfile.ZipFile(path) as archive:  # still a valid zip to Python
        assert archive.testzip() is None
    assert _codes(verify_bundle(path)) == {pv.ARCHIVE_FRAMING}


def test_gap_before_central_directory_is_rejected(tmp_path) -> None:
    import struct

    path = tmp_path / "b.zip"
    raw = _plain_zip(path)
    eocd = raw.rindex(b"PK\x05\x06")
    _insert_bytes(raw, struct.unpack_from("<I", raw, eocd + 16)[0], b"HIDDEN-BYTES")
    path.write_bytes(bytes(raw))
    with zipfile.ZipFile(path) as archive:
        assert archive.testzip() is None
    assert _codes(verify_bundle(path)) == {pv.ARCHIVE_FRAMING}


def test_junk_inside_deflate_span_is_rejected(tmp_path) -> None:
    """Directory sizes claim extra bytes after the deflate stream ends."""

    path = tmp_path / "b.zip"
    raw = _plain_zip(path, zipfile.ZIP_DEFLATED)
    with zipfile.ZipFile(path) as archive:
        victim = archive.getinfo("sources/note.md")
        end = victim.header_offset + 30 + len(victim.filename) + victim.compress_size
    _insert_bytes(raw, end, b"JUNK")
    _patch_member(raw, "sources/note.md", local=True, central=True, csize=victim.compress_size + 4)
    path.write_bytes(bytes(raw))
    report = verify_bundle(path)
    assert not report.ok
    assert _codes(report) <= {pv.ARCHIVE_FRAMING, pv.ENTRY_CORRUPT}


def test_stored_member_with_unequal_sizes_is_rejected(tmp_path) -> None:
    path = tmp_path / "b.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("manifest.json", json.dumps(_manifest(PAYLOAD)))
        for name, data in PAYLOAD.items():
            archive.writestr(name, data)
    raw = bytearray(path.read_bytes())
    _patch_member(raw, "sources/note.md", local=True, central=True, usize=3)
    path.write_bytes(bytes(raw))
    assert not verify_bundle(path).ok


# --- framing positive controls ----------------------------------------------


def test_nested_archive_payload_still_verifies(tmp_path) -> None:
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as nested:
        nested.writestr("x", b"inner")
    files = {**PAYLOAD, "attachments/inner.zip": inner.getvalue()}
    path = _write_zip(tmp_path / "b.zip", files, _manifest(files))
    assert verify_bundle(path).ok


def test_zip64_member_verifies(tmp_path) -> None:
    path = tmp_path / "b.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", json.dumps(_manifest(PAYLOAD)))
        for name, data in PAYLOAD.items():
            with archive.open(name, "w", force_zip64=True) as member:
                member.write(data)
    report = verify_bundle(path)
    assert report.ok, report.findings


class _Unseekable(io.RawIOBase):
    def __init__(self) -> None:
        self.buffer = io.BytesIO()

    def writable(self) -> bool:
        return True

    def write(self, data) -> int:
        return self.buffer.write(data)

    def flush(self) -> None:
        pass


def test_streamed_data_descriptor_archive_verifies(tmp_path) -> None:
    stream = _Unseekable()
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", json.dumps(_manifest(PAYLOAD)))
        for name, data in PAYLOAD.items():
            archive.writestr(name, data)
    path = tmp_path / "b.zip"
    path.write_bytes(stream.buffer.getvalue())
    report = verify_bundle(path)
    assert report.ok, report.findings


def test_member_with_incompressible_and_multichunk_payload_verifies(tmp_path) -> None:
    blob = os.urandom(300_000)
    files = {**PAYLOAD, "blob.bin": blob}
    path = _write_zip(tmp_path / "b.zip", files, _manifest(files))
    assert verify_bundle(path).ok


@pytest.mark.parametrize("field", ["usize", "crc", "csize"])
def test_local_header_disagreeing_with_central_is_rejected(tmp_path, field) -> None:
    path = tmp_path / "b.zip"
    raw = _plain_zip(path, zipfile.ZIP_DEFLATED)
    with zipfile.ZipFile(path) as archive:
        info = archive.getinfo("sources/note.md")
    value = {"usize": info.file_size + 1, "crc": info.CRC ^ 1, "csize": info.compress_size + 1}[field]
    _patch_member(raw, "sources/note.md", local=True, central=False, **{field: value})
    path.write_bytes(bytes(raw))
    assert pv.ARCHIVE_FRAMING in _codes(verify_bundle(path))


@pytest.mark.parametrize("kind", ["short", "crc"])
def test_deflate_output_must_match_declared_length_and_crc(tmp_path, kind) -> None:
    path = tmp_path / "b.zip"
    raw = _plain_zip(path, zipfile.ZIP_DEFLATED)
    with zipfile.ZipFile(path) as archive:
        info = archive.getinfo("sources/note.md")
    if kind == "short":
        fields = {"usize": info.file_size + 7}
    else:
        fields = {"crc": info.CRC ^ 0xFFFF}
    _patch_member(raw, "sources/note.md", local=True, central=True, **fields)
    path.write_bytes(bytes(raw))
    assert _codes(verify_bundle(path)) <= {pv.ENTRY_CORRUPT, pv.SIZE_MISMATCH}
    assert not verify_bundle(path).ok


# --- review findings: raw names, directory records, bounded enumeration -----------


def _rename_member_in_both_headers(raw: bytearray, old: bytes, new: bytes) -> None:
    """Rewrite a member name in its local and central headers (same length)."""

    assert len(old) == len(new)
    count = 0
    position = raw.find(old)
    while position != -1:
        raw[position : position + len(old)] = new
        count += 1
        position = raw.find(old, position + len(new))
    assert count == 2  # local header and central record


def test_a_nul_in_a_raw_member_name_is_rejected_not_truncated(tmp_path) -> None:
    # The archive library truncates names at the first NUL, so "safe\x00suffix" would
    # otherwise be checked (and matched to the manifest) as "safe".
    files = {"canonical/safe": b"payload\n"}
    path = tmp_path / "b.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("manifest.json", json.dumps(_manifest(files)))
        archive.writestr("canonical/safeXsuffix.txt", files["canonical/safe"])
    raw = bytearray(path.read_bytes())
    _rename_member_in_both_headers(raw, b"canonical/safeXsuffix.txt", b"canonical/safe\x00suffix.txt")
    path.write_bytes(bytes(raw))
    with zipfile.ZipFile(path) as archive:
        assert "canonical/safe" in archive.namelist()  # what a truncating reader sees
    report = verify_bundle(path)
    assert not report.ok
    assert pv.PATH_UNSAFE in _codes(report)


@pytest.mark.parametrize("method", [zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED])
def test_a_directory_record_carrying_a_payload_is_rejected(tmp_path, method) -> None:
    path = tmp_path / "b.zip"
    with zipfile.ZipFile(path, "w", method) as archive:
        archive.writestr("manifest.json", json.dumps(_manifest(PAYLOAD)))
        for name, data in PAYLOAD.items():
            archive.writestr(name, data)
        archive.writestr("hidden/", b"secret bytes")
    report = verify_bundle(path)
    assert not report.ok


def test_a_directory_record_whose_stream_lies_about_being_empty_is_rejected(tmp_path) -> None:
    import struct

    path = tmp_path / "b.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", json.dumps(_manifest(PAYLOAD)))
        for name, data in PAYLOAD.items():
            archive.writestr(name, data)
        archive.writestr("hidden/", b"secret bytes")
    raw = bytearray(path.read_bytes())
    # Claim an uncompressed size of 0 (and CRC 0) in both headers; the deflate stream still holds bytes.
    _patch_member(raw, "hidden/", local=True, central=True, usize=0, crc=0)
    path.write_bytes(bytes(raw))
    report = verify_bundle(path)
    assert not report.ok


def test_an_empty_directory_record_is_still_accepted(tmp_path) -> None:
    path = tmp_path / "b.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("manifest.json", json.dumps(_manifest(PAYLOAD)))
        for name, data in PAYLOAD.items():
            archive.writestr(name, data)
        archive.writestr("canonical/", b"")
    assert verify_bundle(path).ok


def test_directory_enumeration_counts_every_child_including_rejected_ones(tmp_path, monkeypatch) -> None:
    root = _write_dir(tmp_path / "b", PAYLOAD)
    for index in range(40):
        os.symlink("/etc/passwd", root / f"link-{index}")  # rejected entries still count
    monkeypatch.setattr(pv, "MAX_BUNDLE_ENTRIES", 20)
    report = verify_bundle(root)
    assert pv.ENTRY_COUNT_LIMIT in _codes(report)
    assert len(report.findings) <= 21  # bounded findings, not one per child


def test_a_wide_directory_is_not_buffered_and_sorted_before_the_limit(tmp_path, monkeypatch) -> None:
    root = _write_dir(tmp_path / "b", PAYLOAD)
    for index in range(50):
        (root / f"extra-{index}").write_bytes(b"x")
    monkeypatch.setattr(pv, "MAX_BUNDLE_ENTRIES", 10)
    real_sorted = sorted
    sizes: list[int] = []

    def watching_sorted(iterable, *args, **kwargs):
        result = real_sorted(iterable, *args, **kwargs)
        sizes.append(len(result))
        return result

    monkeypatch.setattr(pv, "sorted", watching_sorted, raising=False)
    assert pv.ENTRY_COUNT_LIMIT in _codes(verify_bundle(root))
    assert all(size <= 11 for size in sizes)


def test_central_directory_limits_apply_before_the_archive_is_parsed(tmp_path, monkeypatch) -> None:
    path = _write_zip(tmp_path / "b.zip", PAYLOAD)
    monkeypatch.setattr(pv, "MAX_BUNDLE_ENTRIES", 2)

    def must_not_parse(*args, **kwargs):
        raise AssertionError("the archive was parsed before the central-directory limits")

    monkeypatch.setattr(pv.zipfile, "ZipFile", must_not_parse)
    assert _codes(verify_bundle(path)) == {pv.ENTRY_COUNT_LIMIT}


def test_a_central_directory_larger_than_its_byte_limit_is_refused_unparsed(tmp_path, monkeypatch) -> None:
    path = _write_zip(tmp_path / "b.zip", PAYLOAD)
    monkeypatch.setattr(pv, "MAX_CENTRAL_DIRECTORY_BYTES", 64, raising=False)
    monkeypatch.setattr(pv.zipfile, "ZipFile", lambda *a, **k: pytest.fail("parsed"))
    assert _codes(verify_bundle(path)) == {pv.SIZE_LIMIT}


def test_a_lying_end_record_count_does_not_bypass_the_record_limit(tmp_path, monkeypatch) -> None:
    import struct

    path = _write_zip(tmp_path / "b.zip", PAYLOAD)
    raw = bytearray(path.read_bytes())
    eocd = raw.rindex(b"PK\x05\x06")
    struct.pack_into("<HH", raw, eocd + 8, 1, 1)  # claims one record; the directory holds three
    path.write_bytes(bytes(raw))
    monkeypatch.setattr(pv, "MAX_BUNDLE_ENTRIES", 2)
    monkeypatch.setattr(pv.zipfile, "ZipFile", lambda *a, **k: pytest.fail("parsed"))
    report = verify_bundle(path)
    assert not report.ok and _codes(report) & {pv.ENTRY_COUNT_LIMIT, pv.ARCHIVE_FRAMING}


def _three_record_zip(tmp_path) -> Path:
    return _write_zip(tmp_path / "b.zip", PAYLOAD)


@pytest.mark.parametrize("shape", ["prepended", "offset-zero"])
def test_the_record_count_precheck_cannot_be_moved_off_the_directory(tmp_path, monkeypatch, shape) -> None:
    """The archive library finds the directory at end-record position minus its size, not
    at the stored offset; a precheck that trusts the stored offset counts nothing."""

    import struct

    path = _three_record_zip(tmp_path)
    raw = bytearray(path.read_bytes())
    eocd = raw.rindex(b"PK\x05\x06")
    struct.pack_into("<HH", raw, eocd + 8, 1, 1)  # declared count 1: the declared-count check passes
    if shape == "prepended":
        raw[0:0] = b"\x00" * 4096
    else:
        struct.pack_into("<I", raw, eocd + 16, 0)
    path.write_bytes(bytes(raw))
    monkeypatch.setattr(pv, "MAX_BUNDLE_ENTRIES", 2)
    monkeypatch.setattr(pv.zipfile, "ZipFile", lambda *a, **k: pytest.fail("parsed before the limits"))
    report = verify_bundle(path)
    assert not report.ok and _codes(report) & {pv.ENTRY_COUNT_LIMIT, pv.ARCHIVE_FRAMING}


def test_a_prepended_stub_is_refused_before_parsing_even_within_limits(tmp_path, monkeypatch) -> None:
    path = _three_record_zip(tmp_path)
    path.write_bytes(b"#!/bin/sh\nexit 0\n" + path.read_bytes())
    monkeypatch.setattr(pv.zipfile, "ZipFile", lambda *a, **k: pytest.fail("parsed"))
    assert _codes(verify_bundle(path)) == {pv.ARCHIVE_FRAMING}


def _zip64_end_record_archive(tmp_path, monkeypatch) -> Path:
    path = tmp_path / "b.zip"
    with monkeypatch.context() as patch:
        patch.setattr(zipfile, "ZIP_FILECOUNT_LIMIT", 1)  # forces the zip64 end record
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("manifest.json", json.dumps(_manifest(PAYLOAD)))
            for name, data in PAYLOAD.items():
                archive.writestr(name, data)
    assert b"PK\x06\x06" in path.read_bytes() and b"PK\x06\x07" in path.read_bytes()
    return path


def test_a_zip64_end_record_archive_passes_the_precheck(tmp_path, monkeypatch) -> None:
    path = _zip64_end_record_archive(tmp_path, monkeypatch)
    report = pv.VerificationReport()
    assert pv._precheck_central_directory(str(path), report) is True and report.ok


def test_the_zip64_record_count_is_enforced_before_parsing(tmp_path, monkeypatch) -> None:
    path = _zip64_end_record_archive(tmp_path, monkeypatch)
    monkeypatch.setattr(pv, "MAX_BUNDLE_ENTRIES", 2)
    monkeypatch.setattr(pv.zipfile, "ZipFile", lambda *a, **k: pytest.fail("parsed"))
    assert pv.ENTRY_COUNT_LIMIT in _codes(verify_bundle(path))


def test_a_prepended_stub_before_a_zip64_archive_is_refused_before_parsing(tmp_path, monkeypatch) -> None:
    path = _zip64_end_record_archive(tmp_path, monkeypatch)
    path.write_bytes(b"\x00" * 512 + path.read_bytes())
    monkeypatch.setattr(pv.zipfile, "ZipFile", lambda *a, **k: pytest.fail("parsed"))
    assert _codes(verify_bundle(path)) == {pv.ARCHIVE_FRAMING}


# ---- the precheck reads the archive exactly as the parser will, or refuses it before parsing

_CENTRAL = __import__("struct").Struct("<4s6H3I5H2I")


def _central_records(count: int) -> bytes:
    return b"".join(
        _CENTRAL.pack(b"PK\x01\x02", 20, 20, 0, 0, 0, 0, 0, 0, 0, len(name), 0, 0, 0, 0, 0, 0) + name
        for name in (f"f{i}".encode() for i in range(count))
    )


def _end_record(entries: int, cd_size: int, cd_offset: int, comment: bytes = b"") -> bytes:
    import struct

    return struct.pack("<4s4H2IH", b"PK\x05\x06", 0, 0, entries, entries, cd_size, cd_offset, len(comment)) + comment


def _zip64_tail(cd: bytes, *, declared: int, extensible: int = 0) -> bytes:
    import struct

    record = struct.pack("<4sQ2H2I4Q", b"PK\x06\x06", 44 + extensible, 45, 45, 0, 0, declared, declared, len(cd), 0)
    locator = struct.pack("<4sIQI", b"PK\x06\x07", 0, len(cd), 1)
    return cd + record + b"\x00" * extensible + locator + _end_record(0xFFFF, _U32, _U32)


_U32 = 0xFFFFFFFF


def _refused_before_parsing(path, monkeypatch, *, limit: int = 2, codes=None, parser_may_refuse: bool = False) -> None:
    try:
        assert len(zipfile.ZipFile(path).infolist()) > limit  # the parser itself would read every record
    except zipfile.BadZipFile:
        # Some layouts are accepted by one interpreter's parser and refused by another's (its search for
        # the end record differs between patch releases); the precheck must refuse them on every one.
        assert parser_may_refuse, "the parser refused a layout the test expects it to read"
    monkeypatch.setattr(pv, "MAX_BUNDLE_ENTRIES", limit)
    monkeypatch.setattr(pv.zipfile, "ZipFile", lambda *a, **k: pytest.fail("parsed before the precheck refused it"))
    found = _codes(verify_bundle(path))
    assert found and (codes is None or found <= codes), found


def test_an_end_record_whose_own_fields_hold_the_signature_is_refused_before_parsing(tmp_path, monkeypatch) -> None:
    cd = _central_records(5)
    path = tmp_path / "b.zip"
    path.write_bytes(cd + _end_record(1, len(cd), 0x06054B50))  # the offset field's bytes are the signature
    _refused_before_parsing(path, monkeypatch, codes={pv.ARCHIVE_FRAMING})


def test_an_end_record_at_the_edge_of_the_parsers_search_window_is_refused_before_parsing(tmp_path, monkeypatch) -> None:
    cd = _central_records(5)
    path = tmp_path / "b.zip"
    path.write_bytes(cd + _end_record(5, len(cd), 0, b"c" * 0xFFFF) + b"x")
    _refused_before_parsing(path, monkeypatch, codes={pv.ARCHIVE_FRAMING}, parser_may_refuse=True)


def test_any_archive_comment_is_refused_before_parsing(tmp_path, monkeypatch) -> None:
    cd = _central_records(5)
    path = tmp_path / "b.zip"
    path.write_bytes(cd + _end_record(5, len(cd), 0, b"a comment"))
    _refused_before_parsing(path, monkeypatch, codes={pv.ARCHIVE_FRAMING})


def test_a_zip64_record_with_extensible_data_and_a_lying_count_is_refused_before_parsing(tmp_path, monkeypatch) -> None:
    path = tmp_path / "b.zip"
    path.write_bytes(_zip64_tail(_central_records(5), declared=1, extensible=32))
    _refused_before_parsing(path, monkeypatch, limit=4)


def test_a_zip64_directory_beyond_the_byte_limit_is_refused_before_parsing(tmp_path, monkeypatch) -> None:
    path = tmp_path / "b.zip"
    path.write_bytes(_zip64_tail(_central_records(50), declared=50, extensible=8))
    monkeypatch.setattr(pv, "MAX_CENTRAL_DIRECTORY_BYTES", 100)
    monkeypatch.setattr(pv.zipfile, "ZipFile", lambda *a, **k: pytest.fail("parsed past the byte limit"))
    assert pv.SIZE_LIMIT in _codes(verify_bundle(path))


def test_a_zip64_record_not_where_its_locator_points_is_refused(tmp_path, monkeypatch) -> None:
    import struct

    raw = bytearray(_zip64_tail(_central_records(3), declared=3))
    locator_at = raw.rindex(b"PK\x06\x07")
    struct.pack_into("<Q", raw, locator_at + 8, 1)  # points into the directory
    path = tmp_path / "b.zip"
    path.write_bytes(bytes(raw))
    report = pv.VerificationReport()
    assert pv._precheck_central_directory(str(path), report) is False and pv.ARCHIVE_FRAMING in _codes(report)


def test_saturated_end_record_fields_without_a_zip64_locator_are_refused(tmp_path) -> None:
    cd = _central_records(3)
    path = tmp_path / "b.zip"
    path.write_bytes(cd + _end_record(0xFFFF, len(cd), 0))
    report = pv.VerificationReport()
    assert pv._precheck_central_directory(str(path), report) is False and pv.ARCHIVE_FRAMING in _codes(report)


def test_a_directory_that_does_not_fill_its_declared_size_is_refused(tmp_path) -> None:
    cd = _central_records(3)
    path = tmp_path / "b.zip"
    path.write_bytes(cd + b"\x00" * 10 + _end_record(3, len(cd) + 10, 0))
    report = pv.VerificationReport()
    assert pv._precheck_central_directory(str(path), report) is False and pv.ARCHIVE_FRAMING in _codes(report)


def test_an_unreadable_file_is_refused_not_left_to_the_parser(tmp_path) -> None:
    report = pv.VerificationReport()
    assert pv._precheck_central_directory(str(tmp_path / "missing.zip"), report) is False and not report.ok


# ---- legitimate archives still pass the precheck
def _passes_precheck(path) -> None:
    report = pv.VerificationReport()
    assert pv._precheck_central_directory(str(path), report) is True, _codes(report)
    assert report.ok


def test_an_empty_archive_passes_the_precheck(tmp_path) -> None:
    path = tmp_path / "empty.zip"
    zipfile.ZipFile(path, "w").close()
    _passes_precheck(path)


def test_an_archive_with_many_entries_within_the_bounds_passes_the_precheck(tmp_path) -> None:
    path = tmp_path / "many.zip"
    with zipfile.ZipFile(path, "w") as archive:
        for i in range(70_000):  # more than 65,535: Python writes a real zip64 end record
            archive.writestr(f"d/{i}", b"")
    assert b"PK\x06\x06" in path.read_bytes()[-200:]
    _passes_precheck(path)


def test_a_consistent_zip64_record_with_extensible_data_passes_the_precheck(tmp_path) -> None:
    path = tmp_path / "b.zip"
    path.write_bytes(_zip64_tail(_central_records(5), declared=5, extensible=16))
    assert len(zipfile.ZipFile(path).infolist()) == 5
    _passes_precheck(path)


def test_a_python_written_bundle_still_verifies(tmp_path) -> None:
    assert verify_bundle(_write_zip(tmp_path / "b.zip", PAYLOAD)).ok


# ---- compressed bytes are bounded before streaming; directory type needs a directory name; zip64 by locator


class _PaddedCompressor:
    """A real deflate stream followed by many empty blocks: inflates to the same bytes, costs far more to read."""

    def __init__(self, pads: int) -> None:
        import zlib

        self._zlib = zlib
        self._inner = zlib.compressobj(6, zlib.DEFLATED, -15)
        self._pads = pads

    def compress(self, data: bytes) -> bytes:
        return self._inner.compress(data)

    def flush(self, *args) -> bytes:
        aligned = self._inner.flush(self._zlib.Z_SYNC_FLUSH)  # byte-aligned after a sync flush
        empty_stored_block = b"\x00\x00\x00\xff\xff"  # BFINAL=0, BTYPE=00, LEN=0, NLEN=0xFFFF
        return aligned + empty_stored_block * self._pads + self._inner.flush()


def _padded_zip(tmp_path, monkeypatch, *, padded: str, pads: int, extra_dir: bool = False) -> Path:
    import zipfile as zf

    path = tmp_path / "b.zip"
    real = zf._get_compressor
    with zf.ZipFile(path, "w", zf.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", json.dumps(_manifest(PAYLOAD)))
        for name, data in PAYLOAD.items():
            if name == padded:
                monkeypatch.setattr(zf, "_get_compressor", lambda *a, **k: _PaddedCompressor(pads))
            archive.writestr(name, data)
            monkeypatch.setattr(zf, "_get_compressor", real)
        if extra_dir:
            monkeypatch.setattr(zf, "_get_compressor", lambda *a, **k: _PaddedCompressor(pads))
            archive.writestr("canonical/", b"", compress_type=zf.ZIP_DEFLATED)
            monkeypatch.setattr(zf, "_get_compressor", real)
    return path


def test_a_member_padded_with_empty_deflate_blocks_is_refused_before_streaming(tmp_path, monkeypatch) -> None:
    path = _padded_zip(tmp_path, monkeypatch, padded="sources/note.md", pads=50_000)
    info = zipfile.ZipFile(path).getinfo("sources/note.md")
    assert zipfile.ZipFile(path).read("sources/note.md") == PAYLOAD["sources/note.md"]  # still a valid stream
    assert info.compress_size > 100 * info.file_size
    streamed = []
    real_chunks = pv._zip_member_chunks
    monkeypatch.setattr(pv, "_zip_member_chunks", lambda info, raw: streamed.append(info.filename) or real_chunks(info, raw))
    report = verify_bundle(path)
    assert not report.ok and pv.SIZE_LIMIT in _codes(report)
    assert streamed == []  # refused before any member was streamed


def test_an_empty_directory_record_padded_with_empty_blocks_is_refused(tmp_path, monkeypatch) -> None:
    path = _padded_zip(tmp_path, monkeypatch, padded="", pads=50_000, extra_dir=True)
    assert zipfile.ZipFile(path).getinfo("canonical/").compress_size > 100_000
    report = verify_bundle(path)
    assert not report.ok and pv.SIZE_LIMIT in _codes(report)


def test_compressed_bytes_across_the_archive_are_bounded(tmp_path, monkeypatch) -> None:
    import os as _os
    import zipfile as zf

    noisy = {**PAYLOAD, "canonical/random.bin": _os.urandom(20_000)}
    path = tmp_path / "b.zip"
    real = zf._get_compressor
    with zf.ZipFile(path, "w", zf.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", json.dumps(_manifest(noisy)))
        for name, data in noisy.items():
            if name == "canonical/random.bin":  # padded, but within its own per-entry bound
                monkeypatch.setattr(zf, "_get_compressor", lambda *a, **k: _PaddedCompressor(600))
            archive.writestr(name, data)
            monkeypatch.setattr(zf, "_get_compressor", real)
    infos = zipfile.ZipFile(path).infolist()
    uncompressed = sum(i.file_size for i in infos)
    compressed = sum(i.compress_size for i in infos)
    assert compressed > uncompressed
    monkeypatch.setattr(pv, "MAX_TOTAL_BYTES", uncompressed)  # the uncompressed total alone is within the limit
    assert pv.SIZE_LIMIT in _codes(verify_bundle(path))
    monkeypatch.setattr(pv, "MAX_TOTAL_BYTES", compressed)
    assert verify_bundle(path).ok, _codes(verify_bundle(path))


def test_a_lowered_entry_limit_bounds_compressed_bytes_too(tmp_path, monkeypatch) -> None:
    path = _padded_zip(tmp_path, monkeypatch, padded="sources/note.md", pads=2_000)
    info = zipfile.ZipFile(path).getinfo("sources/note.md")
    monkeypatch.setattr(pv, "MAX_ENTRY_BYTES", info.compress_size - 1)
    assert pv.SIZE_LIMIT in _codes(verify_bundle(path))


def test_ordinary_archives_stay_within_the_compressed_bounds(tmp_path) -> None:
    import os as _os

    noisy = {**PAYLOAD, "canonical/random.bin": _os.urandom(200_000)}  # incompressible: deflate slightly expands
    path = _write_zip(tmp_path / "b.zip", noisy, _manifest(noisy))
    assert verify_bundle(path).ok, _codes(verify_bundle(path))


def test_a_directory_type_member_without_a_directory_name_is_refused(tmp_path) -> None:
    import stat as _stat

    path = tmp_path / "b.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", json.dumps(_manifest(PAYLOAD)))
        for name, data in PAYLOAD.items():
            info = zipfile.ZipInfo(name)
            info.compress_type = zipfile.ZIP_DEFLATED
            if name == "sources/note.md":
                info.external_attr = (_stat.S_IFDIR | 0o755) << 16
            archive.writestr(info, data)
    report = verify_bundle(path)
    assert not report.ok and report.findings, _codes(report)


def test_a_python_written_directory_record_is_still_accepted(tmp_path) -> None:
    path = tmp_path / "b.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("manifest.json", json.dumps(_manifest(PAYLOAD)))
        for name, data in PAYLOAD.items():
            archive.writestr(name, data)
        archive.mkdir("canonical")  # S_IFDIR with a trailing slash
    assert verify_bundle(path).ok, _codes(verify_bundle(path))


def test_a_zip64_end_record_with_ordinary_end_fields_verifies_end_to_end(tmp_path, monkeypatch) -> None:
    import struct

    path = _zip64_end_record_archive(tmp_path, monkeypatch)
    raw = path.read_bytes()
    entries = struct.unpack_from("<H", raw, raw.rindex(b"PK\x05\x06") + 10)[0]
    assert entries != 0xFFFF  # the ordinary end record is not saturated
    report = verify_bundle(path)
    assert report.ok, _codes(report)


# ---- with a zip64 locator, each ordinary end-record field is its sentinel or the zip64 value

_ORDINARY_FIELDS = {  # offset in the end record, struct format, sentinel
    "disk": (4, "<H", 0xFFFF),
    "directory_disk": (6, "<H", 0xFFFF),
    "entries_this_disk": (8, "<H", 0xFFFF),
    "entries_total": (10, "<H", 0xFFFF),
    "directory_size": (12, "<I", 0xFFFFFFFF),
    "directory_offset": (16, "<I", 0xFFFFFFFF),
}


def _with_ordinary_field(path: Path, field: str, value=None, delta: int = 0) -> Path:
    import struct

    raw = bytearray(path.read_bytes())
    at = len(raw) - 22
    offset, fmt, _sentinel = _ORDINARY_FIELDS[field]
    current = struct.unpack_from(fmt, raw, at + offset)[0]
    struct.pack_into(fmt, raw, at + offset, current + delta if value is None else value)
    path.write_bytes(bytes(raw))
    return path


@pytest.mark.parametrize("field", sorted(_ORDINARY_FIELDS))
def test_an_ordinary_end_field_that_disagrees_with_the_zip64_record_is_refused_before_parsing(
    tmp_path, monkeypatch, field
) -> None:
    path = _with_ordinary_field(_zip64_end_record_archive(tmp_path, monkeypatch), field, delta=1)
    assert zipfile.ZipFile(path).namelist()  # the archive library reads the zip64 values and ignores the field
    monkeypatch.setattr(pv.zipfile, "ZipFile", lambda *a, **k: pytest.fail("parsed"))
    assert _codes(verify_bundle(path)) == {pv.ARCHIVE_FRAMING}


@pytest.mark.parametrize("field", sorted(_ORDINARY_FIELDS))
def test_the_framing_check_refuses_an_ordinary_end_field_that_disagrees_with_the_zip64_record(
    tmp_path, monkeypatch, field
) -> None:
    path = _with_ordinary_field(_zip64_end_record_archive(tmp_path, monkeypatch), field, delta=1)
    monkeypatch.setattr(pv, "_precheck_central_directory", lambda target, report: True)  # the second line of defence
    assert pv.ARCHIVE_FRAMING in _codes(verify_bundle(path))


def test_ordinary_end_fields_carrying_the_zip64_sentinels_verify(tmp_path, monkeypatch) -> None:
    path = _zip64_end_record_archive(tmp_path, monkeypatch)
    for field, (_offset, _fmt, sentinel) in _ORDINARY_FIELDS.items():
        _with_ordinary_field(path, field, value=sentinel)
    report = verify_bundle(path)
    assert report.ok, _codes(report)


def test_ordinary_end_fields_equal_to_the_zip64_record_verify(tmp_path, monkeypatch) -> None:
    path = _zip64_end_record_archive(tmp_path, monkeypatch)  # Python writes the real values where they fit
    report = verify_bundle(path)
    assert report.ok, _codes(report)


# ---- the precheck never depends on the parser's end-record search; one open file for every read


def test_the_precheck_never_consults_the_parsers_end_record_search(tmp_path, monkeypatch) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("the precheck used the archive library's end-record search")

    monkeypatch.setattr(zipfile, "_EndRecData", forbidden)
    monkeypatch.setattr(zipfile, "_EndRecData64", forbidden)
    cd = _central_records(5)
    layouts = {
        "edge of a search window": (cd + _end_record(5, len(cd), 0, b"c" * 0xFFFF) + b"x", False),
        "comment": (cd + _end_record(5, len(cd), 0, b"a comment"), False),
        "trailing byte": (cd + _end_record(5, len(cd), 0) + b"x", False),
        "strict layout": (cd + _end_record(5, len(cd), 0), True),
    }
    for name, (data, accepted) in layouts.items():
        path = tmp_path / f"{len(data)}.zip"
        path.write_bytes(data)
        report = pv.VerificationReport()
        assert pv._precheck_central_directory(str(path), report) is accepted, (name, _codes(report))
        if not accepted:
            assert _codes(report) == {pv.ARCHIVE_FRAMING}, (name, _codes(report))


def _many_entry_archive(path: Path, count: int) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        for i in range(count):
            archive.writestr(f"extra/{i}.txt", b"x")
    return path


def test_replacing_the_path_after_the_precheck_cannot_change_what_is_parsed(tmp_path, monkeypatch) -> None:
    path = _write_zip(tmp_path / "b.zip", PAYLOAD)  # a valid bundle of three entries
    replacement = _many_entry_archive(tmp_path / "other.zip", 6)
    monkeypatch.setattr(pv, "MAX_BUNDLE_ENTRIES", 3)
    real_precheck, real_zipfile = pv._precheck_central_directory, pv.zipfile.ZipFile
    parsed: list[int] = []

    def precheck_then_swap(source, report):
        accepted = real_precheck(source, report)
        os.replace(replacement, path)  # an attacker swaps the pathname right after the check
        return accepted

    def counting_zipfile(*args, **kwargs):
        archive = real_zipfile(*args, **kwargs)
        parsed.append(len(archive.infolist()))
        return archive

    monkeypatch.setattr(pv, "_precheck_central_directory", precheck_then_swap)
    monkeypatch.setattr(pv.zipfile, "ZipFile", counting_zipfile)
    report = verify_bundle(path)
    assert parsed == [3]  # the parser read the checked bytes, never the six-entry replacement
    assert report.ok, _codes(report)


def test_a_bundle_path_that_is_not_a_regular_file_when_opened_is_refused(tmp_path) -> None:
    fifo = tmp_path / "bundle.zip"
    os.mkfifo(fifo)
    report = verify_bundle(fifo)  # must not block waiting for a writer
    assert _codes(report) == {pv.CONTAINER_UNREADABLE}


# ---- member depth is bounded the same way for zip and directory bundles


def _deep_bundle(tmp_path, monkeypatch, directories: int, kind: str):
    monkeypatch.setattr(pv, "MAX_DIRECTORY_DEPTH", 3)
    deep = "/".join(f"d{i}" for i in range(directories)) + "/f.txt"
    files = {**PAYLOAD, deep: b"deep"}
    if kind == "zip":
        return _write_zip(tmp_path / "deep.zip", files, _manifest(files))
    return _write_dir(tmp_path / "deep", files, _manifest(files))


@pytest.mark.parametrize("directories,accepted", [(3, True), (4, False)])
def test_member_depth_gives_the_same_verdict_for_a_zip_and_a_directory(tmp_path, monkeypatch, directories, accepted) -> None:
    as_zip = verify_bundle(_deep_bundle(tmp_path, monkeypatch, directories, "zip"))
    as_dir = verify_bundle(_deep_bundle(tmp_path, monkeypatch, directories, "dir"))
    assert as_zip.ok is as_dir.ok is accepted, (_codes(as_zip), _codes(as_dir))
    if not accepted:
        assert pv.ENTRY_COUNT_LIMIT in _codes(as_zip)


def test_a_zip_directory_record_deeper_than_the_limit_is_refused(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(pv, "MAX_DIRECTORY_DEPTH", 3)
    path = _write_zip(tmp_path / "b.zip", PAYLOAD)
    with zipfile.ZipFile(path, "a") as archive:
        archive.mkdir("a/b/c/d")
    assert pv.ENTRY_COUNT_LIMIT in _codes(verify_bundle(path))


# ---- finding labels are printable: control characters are escaped, not passed through


@pytest.mark.parametrize("name", ["x\x1b[31mred\x07.txt", "tab\there.txt", "del\x7f.txt", "csi\x9b2J.txt"])
def test_a_member_name_with_control_characters_is_escaped_in_findings(tmp_path, name) -> None:
    path = tmp_path / "n.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("manifest.json", json.dumps(_manifest(PAYLOAD)))
        archive.writestr(name, b"x")
    labels = [finding.path or "" for finding in verify_bundle(path).findings]
    assert labels and all(all(32 <= ord(ch) < 127 for ch in label) for label in labels), labels


def test_an_ordinary_member_name_is_labelled_as_it_is() -> None:
    assert pv._safe_label("canonical/records.jsonl") == "canonical/records.jsonl"


# ---- a directory bundle is walked and read beneath one root, never through a swapped-in link


def test_a_directory_swapped_for_a_link_after_the_walk_is_never_followed(tmp_path, monkeypatch) -> None:
    bundle = _write_dir(tmp_path / "bundle", PAYLOAD)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "records.jsonl").write_bytes(PAYLOAD["canonical/records.jsonl"])  # identical bytes, outside
    real_walk = pv._walk_directory

    def walk_then_swap(*args, **kwargs):
        entries = real_walk(*args, **kwargs)
        os.rename(bundle / "canonical", tmp_path / "moved")
        os.symlink(outside, bundle / "canonical")
        return entries

    monkeypatch.setattr(pv, "_walk_directory", walk_then_swap)
    report = verify_bundle(bundle)
    assert not report.ok, "a file was read through a directory link swapped in after the walk"


def test_a_bundle_root_swapped_for_a_link_after_the_check_is_never_followed(tmp_path, monkeypatch) -> None:
    bundle = _write_dir(tmp_path / "bundle", PAYLOAD)
    decoy = _write_dir(tmp_path / "decoy", PAYLOAD)  # an identical bundle elsewhere
    real_islink = os.path.islink

    def check_then_swap(path):
        answer = real_islink(path)
        if os.fspath(path) == str(bundle) and not answer:
            os.rename(bundle, tmp_path / "bundle-original")
            os.symlink(decoy, bundle)
        return answer

    monkeypatch.setattr(pv.os.path, "islink", check_then_swap)
    report = verify_bundle(bundle)
    assert _codes(report) == {pv.SYMLINK_ENTRY}, _codes(report)


def test_an_untouched_directory_bundle_still_verifies(tmp_path) -> None:
    assert verify_bundle(_write_dir(tmp_path / "bundle", PAYLOAD)).ok


# --- a Unicode Path extra field (0x7075) names the member for the readers that honour it ----------


def _unicode_path_extra(alternate: str, *, version: int = 1, primary: str = "safe.txt", truncate: int = 0) -> bytes:
    name = alternate.encode()
    data = struct.pack("<BI", version, zlib.crc32(primary.encode())) + name
    return struct.pack("<HH", 0x7075, len(data) - truncate) + data[: len(data) - truncate]


def _zip_with_unicode_path(tmp_path, extra: bytes, where: str) -> Path:
    """A one-file bundle whose member ``safe.txt`` carries ``extra`` in the headers named by ``where``
    (``both``, ``local`` or ``central``): the block is written to both and its id is neutralised, same
    length, in the other so every offset and the framing checks stay intact."""

    path = tmp_path / "unicode.zip"
    payload = {"safe.txt": b"x" * 10}
    info = zipfile.ZipInfo("safe.txt")
    info.extra = extra
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("manifest.json", json.dumps(_manifest(payload)))
        archive.writestr(info, payload["safe.txt"])
    raw = bytearray(path.read_bytes())
    marker = struct.pack("<H", 0x7075)
    neutral = struct.pack("<H", 0x7076)
    central_start = raw.find(b"PK\x01\x02", raw.find(b"PK\x01\x02") + 1)  # second record: safe.txt
    if where == "local":
        at = raw.find(marker, central_start)
        raw[at : at + 2] = neutral
    elif where == "central":
        at = raw.find(marker)
        raw[at : at + 2] = neutral
    path.write_bytes(bytes(raw))
    return path


@pytest.mark.parametrize("where", ["both", "local", "central"])
@pytest.mark.parametrize("alternate", ["../escape.txt", "other.txt", "/abs.txt"])
def test_a_unicode_path_field_that_disagrees_with_the_name_is_refused(tmp_path, where, alternate) -> None:
    path = _zip_with_unicode_path(tmp_path, _unicode_path_extra(alternate), where)
    report = verify_bundle(path)
    assert not report.ok, "a field naming another path was reported clean"


@pytest.mark.parametrize("where", ["both", "local", "central"])
def test_a_unicode_path_field_that_agrees_with_the_name_is_accepted(tmp_path, where) -> None:
    report = verify_bundle(_zip_with_unicode_path(tmp_path, _unicode_path_extra("safe.txt"), where))
    assert report.ok, report.findings


@pytest.mark.parametrize(
    "extra",
    [
        _unicode_path_extra("safe.txt", version=2),
        _unicode_path_extra("safe.txt", truncate=len(b"safe.txt") + 2),
        struct.pack("<HH", 0x7075, 5) + struct.pack("<BI", 1, 0) + b"",  # a version and checksum, no name
        struct.pack("<HH", 0x7075, 9) + struct.pack("<BI", 1, 0) + b"\xff\xfe\xfd\xfc",  # not UTF-8
    ],
    ids=["unknown-version", "too-short", "empty-name", "not-utf8"],
)
def test_a_malformed_unicode_path_field_is_refused(tmp_path, extra) -> None:
    assert not verify_bundle(_zip_with_unicode_path(tmp_path, extra, "both")).ok


def test_other_extra_fields_are_left_alone(tmp_path) -> None:
    other = struct.pack("<HH", 0x5455, 5) + b"\x01" + struct.pack("<I", 1700000000)  # extended timestamp
    assert verify_bundle(_zip_with_unicode_path(tmp_path, other, "both")).ok


# --- root spellings, DOS directory bit, deflate chunk boundaries, file/directory conflicts ---------


def _bundle_zip(tmp_path, files: dict[str, bytes], *, infos=None, level=None, name="conflict.zip") -> Path:
    path = tmp_path / name
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", json.dumps(_manifest(files)))
        for member, data in files.items():
            info = (infos or {}).get(member) or zipfile.ZipInfo(member)
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data, compresslevel=level)
    return path


@pytest.mark.parametrize("spelling", ["alias", "alias/", "alias//", "alias/.", "alias/./", "alias/././."])
def test_a_symlinked_bundle_root_is_refused_whatever_its_trailing_spelling(tmp_path, spelling) -> None:
    real = _write_dir(tmp_path / "real", {"a.txt": b"x"}, _manifest({"a.txt": b"x"}))
    (tmp_path / "alias").symlink_to(real)
    report = verify_bundle(str(tmp_path) + "/" + spelling)
    assert pv.SYMLINK_ENTRY in _codes(report), report.findings


@pytest.mark.parametrize("spelling", ["real", "real/", "real//", "real/.", "real/./"])
def test_a_real_directory_root_still_verifies_in_every_spelling(tmp_path, spelling) -> None:
    _write_dir(tmp_path / "real", {"a.txt": b"x"}, _manifest({"a.txt": b"x"}))
    assert verify_bundle(str(tmp_path) + "/" + spelling).ok


def test_a_member_marked_as_a_dos_directory_without_a_directory_name_is_corrupt(tmp_path) -> None:
    info = zipfile.ZipInfo("dosdir")
    info.external_attr = 0x10  # the DOS directory bit; no Unix mode, no trailing slash
    info.create_system = 0
    assert pv.ENTRY_CORRUPT in _codes(verify_bundle(_bundle_zip(tmp_path, {"dosdir": b""}, infos={"dosdir": info})))


def test_a_directory_named_member_with_a_dos_directory_bit_is_still_a_directory(tmp_path) -> None:
    path = tmp_path / "d.zip"
    info = zipfile.ZipInfo("emptydir/")
    info.external_attr = 0x10
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("manifest.json", json.dumps(_manifest({})))
        archive.writestr(info, b"")
    assert verify_bundle(path).ok


@pytest.mark.parametrize("size", [65_535, 65_536, 65_537, 131_073, 200_001])
@pytest.mark.parametrize("level", [1, 6, 9])
def test_a_compressible_member_across_the_chunk_boundary_verifies(tmp_path, size, level) -> None:
    report = verify_bundle(_bundle_zip(tmp_path, {"big.bin": b"a" * size}, level=level))
    assert report.ok, (size, level, report.findings)


def test_a_truncated_deflate_stream_is_still_corrupt_after_the_drain(tmp_path) -> None:
    path = _bundle_zip(tmp_path, {"big.bin": b"a" * 70_000}, level=1)
    raw = bytearray(path.read_bytes())
    # cut the stored deflate stream short by lying about the compressed size in both headers is covered by
    # the central-size tests; here flip a byte inside the stream so the stream is invalid or short
    offset = raw.find(b"big.bin") + len("big.bin") + 8
    raw[offset] ^= 0xFF
    path.write_bytes(bytes(raw))
    assert not verify_bundle(path).ok


@pytest.mark.parametrize(
    "files",
    [
        {"node": b"1", "node/child.txt": b"2"},
        {"a": b"1", "a/b.txt": b"2"},
        {"A": b"1", "a/b.txt": b"2"},  # a folded alias of the parent
        {"a/b": b"1", "a/b/c/d.txt": b"2"},  # a deeper implicit parent
        {"manifest.json/data.txt": b"2"},  # beneath the manifest file itself
        {"manifest.json/payload.bin": b"zz"},
        {"Manifest.JSON/payload.bin": b"zz"},
    ],
    ids=["file-and-child", "short", "folded", "deep", "beneath-manifest", "beneath-manifest-payload", "folded-manifest"],
)
def test_a_file_that_is_also_a_directory_is_refused_before_hashing(tmp_path, files) -> None:
    report = verify_bundle(_bundle_zip(tmp_path, files))
    assert pv.PATH_CONFLICT in _codes(report), report.findings


def test_names_that_only_look_alike_are_not_a_conflict(tmp_path) -> None:
    assert verify_bundle(_bundle_zip(tmp_path, {"a/b.txt": b"1", "a2.txt": b"2", "a/c/d.txt": b"3"})).ok


def test_a_file_and_a_directory_of_one_name_are_refused_in_a_directory_bundle_too(tmp_path) -> None:
    root = _write_dir(tmp_path / "b", {"A": b"1"}, _manifest({"A": b"1"}))
    (root / "a").mkdir()
    (root / "a" / "b.txt").write_bytes(b"2")
    report = verify_bundle(root)
    assert not report.ok


def test_an_archive_named_with_a_trailing_slash_is_still_not_a_readable_archive(tmp_path) -> None:
    path = _bundle_zip(tmp_path, {"a.txt": b"x"}, name="plain.zip")
    assert verify_bundle(path).ok
    assert pv.CONTAINER_UNREADABLE in _codes(verify_bundle(str(path) + "/"))  # ENOTDIR semantics preserved


# --- streamed members' local CRC; an in-place rewrite after the precheck -------------------------


class _StreamSink(io.RawIOBase):
    def __init__(self) -> None:
        self.data = bytearray()

    def writable(self) -> bool:
        return True

    def write(self, chunk) -> int:
        self.data += chunk
        return len(chunk)


def _streamed_bundle(tmp_path, *, local_crc=None) -> Path:
    """A bundle whose members are streamed (bit 3, a trailing data descriptor); ``local_crc`` rewrites every
    local header's CRC field (None keeps the writer's zero placeholder)."""

    stream = _StreamSink()
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_STORED) as archive:
        archive.writestr("manifest.json", json.dumps(_manifest(PAYLOAD)))
        for name, data in PAYLOAD.items():
            archive.writestr(name, data)
    raw = bytearray(stream.data)
    assert raw[6] & 0x08  # the writer streamed them
    if local_crc is not None:
        offset = 0
        while (offset := raw.find(b"PK\x03\x04", offset)) != -1 and offset < raw.find(b"PK\x01\x02"):
            struct.pack_into("<I", raw, offset + 14, local_crc(raw, offset))
            offset += 4
    path = tmp_path / "streamed.zip"
    path.write_bytes(bytes(raw))
    return path


def _central_crc_of_local(raw: bytearray, local_offset: int) -> int:
    """The central directory's CRC for the member whose local header is at ``local_offset``."""

    cursor = raw.find(b"PK\x01\x02")
    while cursor != -1:
        if struct.unpack_from("<I", raw, cursor + 42)[0] == local_offset:
            return struct.unpack_from("<I", raw, cursor + 16)[0]
        cursor = raw.find(b"PK\x01\x02", cursor + 4)
    raise AssertionError("no central record")


def test_a_streamed_member_with_a_zero_or_the_real_local_crc_verifies(tmp_path) -> None:
    assert verify_bundle(_streamed_bundle(tmp_path)).ok  # the writer's placeholder
    assert verify_bundle(_streamed_bundle(tmp_path, local_crc=_central_crc_of_local)).ok  # the real value


def test_a_streamed_member_with_another_nonzero_local_crc_is_refused(tmp_path) -> None:
    report = verify_bundle(_streamed_bundle(tmp_path, local_crc=lambda raw, offset: 0xDEADBEEF))
    assert pv.ARCHIVE_FRAMING in _codes(report), report.findings


def test_rewriting_the_file_in_place_after_the_precheck_cannot_change_what_is_parsed(tmp_path, monkeypatch) -> None:
    path = _write_zip(tmp_path / "b.zip", PAYLOAD)  # a valid bundle of three entries
    replacement = _many_entry_archive(tmp_path / "other.zip", 6).read_bytes()
    monkeypatch.setattr(pv, "MAX_BUNDLE_ENTRIES", 3)
    real_precheck, real_zipfile = pv._precheck_central_directory, pv.zipfile.ZipFile
    parsed: list[int] = []

    def precheck_then_rewrite(source, report):
        accepted = real_precheck(source, report)
        with open(path, "r+b") as writer:  # another writer rewrites the same file, same inode, in place
            writer.seek(0)
            writer.write(replacement)
            writer.truncate()
        return accepted

    def counting_zipfile(*args, **kwargs):
        archive = real_zipfile(*args, **kwargs)
        parsed.append(len(archive.infolist()))
        return archive

    monkeypatch.setattr(pv, "_precheck_central_directory", precheck_then_rewrite)
    monkeypatch.setattr(pv.zipfile, "ZipFile", counting_zipfile)
    report = verify_bundle(path)
    assert parsed == [3]  # the parser got the checked directory, never the six-entry rewrite
    assert not report.ok  # and the change itself is refused


def test_bytes_appended_in_place_after_the_precheck_are_a_framing_change(tmp_path, monkeypatch) -> None:
    path = _write_zip(tmp_path / "b.zip", PAYLOAD)
    assert verify_bundle(path).ok
    real_precheck = pv._precheck_central_directory

    def precheck_then_append(source, report):
        accepted = real_precheck(source, report)
        with open(path, "ab") as writer:  # same inode, the checked bytes untouched, only the size changes
            writer.write(b"trailing bytes")
        return accepted

    monkeypatch.setattr(pv, "_precheck_central_directory", precheck_then_append)
    report = verify_bundle(path)
    assert not report.ok
    assert (pv.ARCHIVE_FRAMING, "file changed during verification") in [
        (item.code, item.detail) for item in report.findings
    ]


def test_a_file_touched_in_place_after_the_precheck_is_a_framing_change(tmp_path, monkeypatch) -> None:
    path = _write_zip(tmp_path / "b.zip", PAYLOAD)
    real_precheck = pv._precheck_central_directory

    def precheck_then_touch(source, report):
        accepted = real_precheck(source, report)
        stat = os.stat(path)
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 5_000_000_000))  # content and size untouched
        return accepted

    monkeypatch.setattr(pv, "_precheck_central_directory", precheck_then_touch)
    report = verify_bundle(path)
    assert [(item.code, item.detail) for item in report.findings] == [
        (pv.ARCHIVE_FRAMING, "file changed during verification")
    ]


def test_an_untouched_archive_is_unchanged_by_the_pinned_parse(tmp_path) -> None:
    assert verify_bundle(_write_zip(tmp_path / "b.zip", PAYLOAD)).ok


# --- the path argument is copied to exact text once; a str or bytes subclass cannot redirect the verifier ----


def _good_and_corrupt_bundles(tmp_path):
    good = _write_dir(tmp_path / "good", PAYLOAD)
    bad = _write_dir(tmp_path / "bad", PAYLOAD)
    (bad / "canonical" / "records.jsonl").write_bytes(b"tampered")  # size and digest no longer match
    assert verify_bundle(good).ok and not verify_bundle(bad).ok
    return good, bad


def test_a_bytes_subclass_path_cannot_make_the_verifier_read_another_bundle(tmp_path) -> None:
    good, bad = _good_and_corrupt_bundles(tmp_path)

    class Redirecting(bytes):
        def decode(self, *args, **kwargs):  # answers with the good bundle's path
            return str(good)

    assert not verify_bundle(Redirecting(os.fsencode(bad))).ok  # names the corrupt bundle: refused, not "ok"


def test_a_str_subclass_path_cannot_make_the_verifier_read_another_bundle(tmp_path) -> None:
    good, bad = _good_and_corrupt_bundles(tmp_path)

    class Redirecting(str):
        def endswith(self, *args, **kwargs):
            return True

        def rstrip(self, *args, **kwargs):
            return str(good)

    assert not verify_bundle(Redirecting(str(bad))).ok


def test_a_pathlike_is_asked_for_its_path_once_and_verified_by_that_text(tmp_path) -> None:
    good, bad = _good_and_corrupt_bundles(tmp_path)
    asked = []

    class Flipping:
        def __fspath__(self):
            asked.append(1)
            return str(bad) if len(asked) == 1 else str(good)

    assert not verify_bundle(Flipping()).ok
    assert len(asked) == 1


def test_ordinary_paths_still_verify_in_every_form(tmp_path) -> None:
    good, _ = _good_and_corrupt_bundles(tmp_path)
    zipped = _write_zip(tmp_path / "b.zip", PAYLOAD)
    assert verify_bundle(good).ok and verify_bundle(str(good)).ok and verify_bundle(os.fsencode(good)).ok
    assert verify_bundle(zipped).ok and verify_bundle(str(zipped)).ok and verify_bundle(os.fsencode(zipped)).ok


# --- a file that changes after the inventory is a failed report, never an exception ------------------------


def _noise(label: str, size: int) -> bytes:
    """Deterministic incompressible bytes."""

    out, block = bytearray(), label.encode()
    while len(out) < size:
        block = hashlib.sha256(block).digest()
        out += block
    return bytes(out[:size])


# several 100 KiB members: far more than the reader's buffer, so a read after the change reaches the file
BIG_PAYLOAD = {f"canonical/part{i}.bin": _noise(f"part{i}", 100_000) for i in range(3)}


def _big_zip(path: Path) -> Path:
    return _write_zip(path, BIG_PAYLOAD, manifest=_manifest(BIG_PAYLOAD))


def _after_inventory(monkeypatch, change) -> None:
    real_inventory = pv._inventory_zip

    def inventory_then_change(source, report):
        entries = real_inventory(source, report)
        change(entries)
        return entries

    monkeypatch.setattr(pv, "_inventory_zip", inventory_then_change)


def _assert_failed_report(report) -> None:
    assert not report.ok
    assert all(isinstance(item.code, str) for item in report.findings)
    json.dumps(report.to_dict())  # what the command line prints


def test_truncating_the_archive_right_after_the_inventory_is_a_failed_report(tmp_path, monkeypatch) -> None:
    path = _big_zip(tmp_path / "b.zip")

    def truncate(entries) -> None:
        os.truncate(path, 10)  # shorter than the first local header

    _after_inventory(monkeypatch, truncate)
    _assert_failed_report(verify_bundle(path))


@pytest.mark.parametrize("keep", [0, 1, 29, 30, 31, 45, 9_000, 100_000, 150_000, 299_999])
def test_truncating_to_any_length_after_the_inventory_is_a_failed_report(tmp_path, monkeypatch, keep) -> None:
    path = _big_zip(tmp_path / "b.zip")
    size = path.stat().st_size
    _after_inventory(monkeypatch, lambda entries: os.truncate(path, min(keep, size - 1)))
    _assert_failed_report(verify_bundle(path))


def test_a_member_whose_offset_cannot_be_seeked_is_a_failed_report(tmp_path, monkeypatch) -> None:
    path = _big_zip(tmp_path / "b.zip")

    def move_offsets(entries) -> None:
        for entry in entries:
            if entry.handle is not None:
                entry.handle.header_offset = 1 << 64  # beyond what a seek accepts

    _after_inventory(monkeypatch, move_offsets)
    _assert_failed_report(verify_bundle(path))


def test_every_prefix_and_byte_flip_of_a_valid_archive_is_a_report_not_a_traceback(tmp_path, monkeypatch) -> None:
    good = _big_zip(tmp_path / "good.zip").read_bytes()
    for keep in range(0, len(good), 9_973):
        target = tmp_path / f"cut{keep}.zip"
        target.write_bytes(good)
        _after_inventory(monkeypatch, lambda entries, target=target, keep=keep: os.truncate(target, keep))
        verify_bundle(target)  # must not raise
    for position in range(0, len(good), 7_919):
        target = tmp_path / f"flip{position}.zip"
        mutated = bytearray(good)
        mutated[position] ^= 0xFF
        target.write_bytes(bytes(mutated))
        monkeypatch.undo()
        verify_bundle(target)  # must not raise


def _forced_zip64_bundle(path: Path) -> bytes:
    """A valid bundle whose members carry their sizes in a local zip64 extra block (sizes 0xFFFFFFFF)."""

    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", json.dumps(_manifest(PAYLOAD)))
        for name, data in PAYLOAD.items():
            with archive.open(zipfile.ZipInfo(name), "w", force_zip64=True) as member:
                member.write(data)
    return path.read_bytes()


def _first_forced_zip64_extra(raw: bytes) -> int:
    """Offset of the first member's local zip64 block (a header whose sizes are 0xFFFFFFFF)."""

    position = 0
    while True:
        position = raw.index(b"PK\x03\x04", position)
        name_length, _extra_length = struct.unpack_from("<HH", raw, position + 26)
        if struct.unpack_from("<II", raw, position + 18) == (0xFFFFFFFF, 0xFFFFFFFF):
            return position + 30 + name_length
        position += 4


def test_a_forced_zip64_bundle_still_verifies(tmp_path) -> None:
    path = tmp_path / "b.zip"
    _forced_zip64_bundle(path)
    assert verify_bundle(path).ok


@pytest.mark.parametrize("declared", [17, 0x00EF, 0xFFFF])
def test_a_local_zip64_block_longer_than_its_extra_field_is_rejected(tmp_path, declared) -> None:
    """Its sizes are read from bytes another reader does not accept as the block: it cannot find the sizes."""

    path = tmp_path / "b.zip"
    raw = bytearray(_forced_zip64_bundle(path))
    block = _first_forced_zip64_extra(raw)
    assert struct.unpack_from("<HH", raw, block) == (0x0001, 16)
    struct.pack_into("<H", raw, block + 2, declared)
    path.write_bytes(bytes(raw))
    report = verify_bundle(path)
    assert not report.ok and pv.ARCHIVE_FRAMING in _codes(report)


def test_a_local_zip64_block_too_short_for_the_sizes_it_must_carry_is_rejected(tmp_path) -> None:
    path = tmp_path / "b.zip"
    raw = bytearray(_forced_zip64_bundle(path))
    block = _first_forced_zip64_extra(raw)
    struct.pack_into("<H", raw, block + 2, 8)  # room for one size; both are 0xFFFFFFFF in the header
    struct.pack_into("<HH", raw, block + 12, 0x9999, 4)  # the rest of the area still tiles as an unknown block
    path.write_bytes(bytes(raw))
    report = verify_bundle(path)
    assert not report.ok and pv.ARCHIVE_FRAMING in _codes(report)
    assert any("too short" in item.detail for item in report.findings)  # refused for this reason, not by accident


class _DescriptorSink(io.RawIOBase):
    """A write-only sink without seek, so ``zipfile`` streams members with data descriptors."""

    def __init__(self) -> None:
        self.data = bytearray()

    def writable(self) -> bool:
        return True

    def write(self, chunk) -> int:
        self.data += chunk
        return len(chunk)


def _streamed_zip64_bundle(path: Path) -> bytes:
    """A valid bundle whose members are streamed with a local zip64 block (sizes zero, then a descriptor)."""

    sink = _DescriptorSink()
    with zipfile.ZipFile(sink, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", json.dumps(_manifest(PAYLOAD)))
        for name, data in PAYLOAD.items():
            with archive.open(zipfile.ZipInfo(name), "w", force_zip64=True) as member:
                member.write(data)
    path.write_bytes(bytes(sink.data))
    return bytes(sink.data)


def _local_extra_of(raw: bytes, member: str) -> tuple[int, int]:
    """(offset, length) of ``member``'s local extra field."""

    position = 0
    while True:
        position = raw.index(b"PK\x03\x04", position)
        name_length, extra_length = struct.unpack_from("<HH", raw, position + 26)
        if raw[position + 30 : position + 30 + name_length] == member.encode():
            return position + 30 + name_length, extra_length
        position += 4


def test_a_streamed_zip64_bundle_still_verifies(tmp_path) -> None:
    path = tmp_path / "b.zip"
    _streamed_zip64_bundle(path)
    assert verify_bundle(path).ok


@pytest.mark.parametrize("declared", [17, 0x00EF, 0xFFFF])
def test_a_streamed_local_zip64_block_longer_than_its_extra_field_is_rejected(tmp_path, declared) -> None:
    """The block is never read for sizes (they follow in a descriptor), but readers still walk the field."""

    path = tmp_path / "b.zip"
    raw = bytearray(_streamed_zip64_bundle(path))
    offset, length = _local_extra_of(raw, next(iter(PAYLOAD)))  # a streamed zip64 member
    assert length >= 4 and struct.unpack_from("<H", raw, offset)[0] == 0x0001
    struct.pack_into("<H", raw, offset + 2, declared)
    path.write_bytes(bytes(raw))
    report = verify_bundle(path)
    assert not report.ok and pv.ARCHIVE_FRAMING in _codes(report)


def _bundle_with_local_extra(path: Path, local_extra: bytes) -> None:
    """A valid bundle whose first member carries ``local_extra`` in its local header only."""

    filler = struct.pack("<HH", 0x9999, len(local_extra) - 4) + b"\0" * (len(local_extra) - 4)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("manifest.json", json.dumps(_manifest(PAYLOAD)))
        for index, (name, data) in enumerate(PAYLOAD.items()):
            info = zipfile.ZipInfo(name)
            if index == 0:
                info.extra = filler
            archive.writestr(info, data)
    raw = bytearray(path.read_bytes())
    offset, length = _local_extra_of(raw, next(iter(PAYLOAD)))
    assert length == len(local_extra)
    raw[offset : offset + length] = local_extra
    path.write_bytes(bytes(raw))


def test_a_well_formed_unknown_local_extra_block_still_verifies(tmp_path) -> None:
    path = tmp_path / "b.zip"
    _bundle_with_local_extra(path, struct.pack("<HH", 0x9999, 4) + b"abcd")
    assert verify_bundle(path).ok


@pytest.mark.parametrize("declared", [0, 3, 5, 200])
def test_a_local_extra_block_that_does_not_tile_its_field_is_rejected(tmp_path, declared) -> None:
    """Any declared block length other than the one that fills the field leaves bytes readers parse differently."""

    path = tmp_path / "b.zip"
    _bundle_with_local_extra(path, struct.pack("<HH", 0x9999, declared) + b"abcd")
    report = verify_bundle(path)
    assert not report.ok and pv.ARCHIVE_FRAMING in _codes(report)


def test_a_partial_block_header_at_the_end_of_a_local_extra_field_is_rejected(tmp_path) -> None:
    path = tmp_path / "b.zip"
    _bundle_with_local_extra(path, struct.pack("<HH", 0x9999, 4) + b"abcd" + b"\x99\x99")
    report = verify_bundle(path)
    assert not report.ok and pv.ARCHIVE_FRAMING in _codes(report)
    assert any("partial block header" in item.detail for item in report.findings)  # refused for this reason


# --- the end records are rewritten in place while the precheck reads them ------------------------


class _RewriteBeforeRead:
    """A file view that rewrites ``data`` at ``at`` in the underlying file just before its ``rewrite_before``-th read.

    Reads go straight to the descriptor (``os.pread``), so no buffer can hide the rewrite from a later read."""

    def __init__(self, raw, path: Path, rewrite_before: int, at: int, data: bytes) -> None:
        self._raw, self._path, self._before, self._at, self._data = raw, path, rewrite_before, at, data
        self._position, self.reads = 0, 0

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        base = {os.SEEK_SET: 0, os.SEEK_CUR: self._position, os.SEEK_END: os.fstat(self._raw.fileno()).st_size}[whence]
        self._position = base + offset
        return self._position

    def tell(self) -> int:
        return self._position

    def read(self, count: int = -1) -> bytes:
        self.reads += 1
        if self.reads == self._before:
            with open(self._path, "r+b") as writer:  # another writer, same inode, in place
                writer.seek(self._at)
                writer.write(self._data)
        if count is None or count < 0:
            count = max(0, os.fstat(self._raw.fileno()).st_size - self._position)
        out = os.pread(self._raw.fileno(), count, self._position)
        self._position += len(out)
        return out


def _zip64_end(entries: int, cd_size: int, cd_offset: int) -> bytes:
    return struct.pack("<4sQ2H2I4Q", b"PK\x06\x06", 44, 45, 45, 0, 0, entries, entries, cd_size, cd_offset)


def _end_records_rewritten_during_the_precheck(tmp_path, monkeypatch, *, rewritten: str):
    """Six central records; the end records first bound only the last three, then a rewrite between the precheck's
    reads makes them declare all six, starting before the bounded slice."""

    cd = _central_records(6)
    first = len(_central_records(3))
    if rewritten in ("zip64-end-record", "zip64-locator"):
        bounded = _zip64_end(3, len(cd) - first, first)
        locator = struct.pack("<4sIQI", b"PK\x06\x07", 0, len(cd), 1)
        data = cd + bounded + locator + _end_record(0xFFFF, _U32, _U32)
        if rewritten == "zip64-end-record":
            rewrite = dict(rewrite_before=4, at=len(cd), data=_zip64_end(6, len(cd), 0))  # after the record is read
        else:  # the locator points elsewhere once the record it pointed to was read
            rewrite = dict(rewrite_before=4, at=len(cd) + len(bounded), data=struct.pack("<4sIQI", b"PK\x06\x07", 0, 0, 1))
    else:
        data = cd + _end_record(3, len(cd) - first, first)
        if rewritten == "end-record":
            rewrite = dict(rewrite_before=2, at=len(cd), data=_end_record(6, len(cd), 0))  # after the end record is read
        else:  # only the last byte (the comment length's high byte): a comment the parser would look past
            rewrite = dict(rewrite_before=2, at=len(data) - 1, data=b"\x01")
    path = tmp_path / "b.zip"
    path.write_bytes(data)
    monkeypatch.setattr(pv, "MAX_BUNDLE_ENTRIES", 3)
    real_precheck, real_zipfile = pv._precheck_central_directory, pv.zipfile.ZipFile
    views: list[_RewriteBeforeRead] = []
    parsed: list[int] = []

    def precheck_through_rewriting_view(source, report):
        view = _RewriteBeforeRead(source, path, **rewrite)
        views.append(view)
        return real_precheck(view, report)

    def counting_zipfile(*args, **kwargs):
        archive = real_zipfile(*args, **kwargs)
        parsed.append(len(archive.infolist()))
        return archive

    monkeypatch.setattr(pv, "_precheck_central_directory", precheck_through_rewriting_view)
    monkeypatch.setattr(pv.zipfile, "ZipFile", counting_zipfile)
    report = verify_bundle(path)
    assert views and views[0].reads >= rewrite["rewrite_before"]  # the rewrite happened inside the precheck
    return report, parsed


@pytest.mark.parametrize("rewritten", ["end-record", "end-record-last-byte", "zip64-end-record", "zip64-locator"])
def test_end_records_rewritten_while_the_precheck_reads_them_are_refused_before_parsing(tmp_path, monkeypatch, rewritten) -> None:
    report, parsed = _end_records_rewritten_during_the_precheck(tmp_path, monkeypatch, rewritten=rewritten)
    assert parsed == []  # never parsed: a rewritten end record could direct the parser past the entry limit
    assert (pv.ARCHIVE_FRAMING, "end records changed while they were read") in [
        (item.code, item.detail) for item in report.findings
    ]


def test_end_records_left_alone_during_the_precheck_still_verify(tmp_path, monkeypatch) -> None:
    path = _zip64_end_record_archive(tmp_path, monkeypatch)
    assert verify_bundle(path).ok
    assert verify_bundle(_write_zip(tmp_path / "plain.zip", PAYLOAD)).ok


def _zip64_pointer(entries: int, cd_size: int, record_at: int) -> bytes:
    """A zip64 end record (directory at offset 0) followed by a locator that points at it: 76 bytes."""

    return _zip64_end(entries, cd_size, 0) + struct.pack("<4sIQI", b"PK\x06\x07", 0, record_at, 1)


def test_a_zip64_locator_planted_in_the_directory_while_the_precheck_reads_is_refused(tmp_path, monkeypatch) -> None:
    """No locator when the end records are read, so the ordinary end record is bounded; then a locator and a zip64
    record are written into the last central record's name before the copy is taken.  The parser would follow them."""

    extra = _central_records(3)
    long_name = b"n" * 100
    last = _CENTRAL.pack(b"PK\x01\x02", 20, 20, 0, 0, 0, 0, 0, 0, 0, len(long_name), 0, 0, 0, 0, 0, 0) + long_name
    cd = _central_records(5) + last
    data = extra + cd + _end_record(6, len(cd), len(extra))
    path = tmp_path / "b.zip"
    path.write_bytes(data)
    eocd_at = len(data) - 22
    planted = _zip64_pointer(9, eocd_at - 76, eocd_at - 76)  # inside the long name: the record walk still passes
    monkeypatch.setattr(pv, "MAX_BUNDLE_ENTRIES", 6)
    real_precheck, real_zipfile = pv._precheck_central_directory, pv.zipfile.ZipFile
    parsed: list[int] = []

    def precheck_through_rewriting_view(source, report):
        return real_precheck(_RewriteBeforeRead(source, path, 3, eocd_at - 76, planted), report)  # before the copy

    def counting_zipfile(*args, **kwargs):
        archive = real_zipfile(*args, **kwargs)
        parsed.append(len(archive.infolist()))
        return archive

    monkeypatch.setattr(pv, "_precheck_central_directory", precheck_through_rewriting_view)
    monkeypatch.setattr(pv.zipfile, "ZipFile", counting_zipfile)
    report = verify_bundle(path)
    assert parsed == []
    assert (pv.ARCHIVE_FRAMING, "end records changed while they were read") in [
        (item.code, item.detail) for item in report.findings
    ]


def test_a_zip64_locator_written_before_an_empty_directory_after_the_precheck_cannot_be_parsed(tmp_path, monkeypatch) -> None:
    """An empty directory: the bytes the parser searches for a zip64 locator lie before the directory, outside a copy
    that starts at it.  A writer that plants a locator there after the precheck must not reach the parser."""

    prefix = _central_records(3) + b"\x00" * 76
    data = prefix + _end_record(0, 0, len(prefix))
    path = tmp_path / "b.zip"
    path.write_bytes(data)
    planted = _zip64_pointer(3, len(prefix) - 76, len(prefix) - 76)
    monkeypatch.setattr(pv, "MAX_BUNDLE_ENTRIES", 2)
    real_precheck, real_zipfile = pv._precheck_central_directory, pv.zipfile.ZipFile
    parsed: list[int] = []

    accepted: list[bool] = []

    def precheck_then_plant(source, report):
        accepted.append(real_precheck(source, report))
        with open(path, "r+b") as writer:  # same inode, in place, after the precheck
            writer.seek(len(prefix) - 76)
            writer.write(planted)
        return accepted[-1]

    def counting_zipfile(*args, **kwargs):
        archive = real_zipfile(*args, **kwargs)
        parsed.append(len(archive.infolist()))
        return archive

    monkeypatch.setattr(pv, "_precheck_central_directory", precheck_then_plant)
    monkeypatch.setattr(pv.zipfile, "ZipFile", counting_zipfile)
    verify_bundle(path)
    assert accepted == [True]  # the honest empty directory passes the precheck
    assert parsed == [0], parsed  # and the parser read the copy's empty directory, never the plant


# --- general-purpose flags the standard reader refuses are refused by the raw stream reader too -------------


def _with_flag_in_both_headers(path: Path, bit: int) -> None:
    data = bytearray(path.read_bytes())
    with zipfile.ZipFile(path) as archive:
        offsets = [info.header_offset for info in archive.infolist()]
    for offset in offsets:
        flags = struct.unpack_from("<H", data, offset + 6)[0]
        struct.pack_into("<H", data, offset + 6, flags | bit)  # local header
    position = data.rindex(b"PK\x01\x02")
    while position != -1 and position >= 0:
        flags = struct.unpack_from("<H", data, position + 8)[0]
        struct.pack_into("<H", data, position + 8, flags | bit)  # central record
        position = data.rfind(b"PK\x01\x02", 0, position)
    path.write_bytes(bytes(data))


@pytest.mark.parametrize(
    ("bit", "code"),
    [
        (0x20, pv.CONTAINER_UNSUPPORTED),  # compressed patched data
        (0x2000, pv.CONTAINER_UNSUPPORTED),  # masked local header values (encrypted central directory)
        (0x40, pv.ENTRY_ENCRYPTED),  # strong encryption
        (0x1, pv.ENTRY_ENCRYPTED),
    ],
)
def test_an_entry_with_a_flag_the_reader_refuses_is_not_a_clean_report(tmp_path, bit, code) -> None:
    path = _write_zip(tmp_path / "b.zip", PAYLOAD)
    _with_flag_in_both_headers(path, bit)
    report = verify_bundle(path)
    assert not report.ok and code in {item.code for item in report.findings}


@pytest.mark.parametrize("bit", [0x2, 0x4, 0x800])
def test_flags_that_only_describe_options_still_verify(tmp_path, bit) -> None:
    path = _write_zip(tmp_path / "b.zip", PAYLOAD)
    _with_flag_in_both_headers(path, bit)
    assert verify_bundle(path).ok


def test_the_flags_refused_here_are_the_ones_the_standard_reader_refuses(tmp_path) -> None:
    for bit in (0x20, 0x40):
        path = _write_zip(tmp_path / f"r{bit}.zip", PAYLOAD)
        _with_flag_in_both_headers(path, bit)
        with zipfile.ZipFile(path) as archive, pytest.raises(NotImplementedError):
            archive.open(archive.infolist()[0]).read()


# --- a member that starts on another disk is refused even when the end records say one disk -------------------


def _set_central_field(path: Path, offset: int, fmt: str, value: int) -> None:
    data = bytearray(path.read_bytes())
    position = data.find(b"PK\x01\x02")
    while position != -1:
        struct.pack_into(fmt, data, position + offset, value)
        position = data.find(b"PK\x01\x02", position + 4)
    path.write_bytes(bytes(data))


@pytest.mark.parametrize("disk", [1, 2, 0xFFFF])
def test_a_member_whose_start_disk_is_not_zero_is_refused(tmp_path, disk) -> None:
    path = _write_zip(tmp_path / "b.zip", PAYLOAD)
    _set_central_field(path, 34, "<H", disk)  # only the member's start-disk field; the end records stay at zero
    with zipfile.ZipFile(path) as archive:
        assert {info.volume for info in archive.infolist()} == {disk}
    report = verify_bundle(path)
    assert not report.ok
    assert [(item.code, item.detail) for item in report.findings] == [(pv.ARCHIVE_FRAMING, "member starts on another disk")]


def test_one_member_on_another_disk_is_enough(tmp_path) -> None:
    path = _write_zip(tmp_path / "b.zip", PAYLOAD)
    data = bytearray(path.read_bytes())
    struct.pack_into("<H", data, data.rindex(b"PK\x01\x02") + 34, 1)  # the last record only
    path.write_bytes(bytes(data))
    assert not verify_bundle(path).ok


@pytest.mark.parametrize(("offset", "fmt", "value"), [(36, "<H", 1), (36, "<H", 0xFFFF), (4, "<H", 0x0A63)])
def test_member_fields_that_no_reader_acts_on_still_verify(tmp_path, offset, fmt, value) -> None:
    path = _write_zip(tmp_path / "b.zip", PAYLOAD)
    _set_central_field(path, offset, fmt, value)  # internal attributes; version made by
    assert verify_bundle(path).ok
