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
