"""Standalone offline verifier for portable bundles.

Checks a bundle (a directory or a ``.zip`` file) against its
``manifest.json`` without extracting, writing, networking, or contacting any
service or subscription.  Every structural check on the container inventory
runs before any payload byte is read, and payload bytes are only hashed in
bounded chunks in memory.

A clean report means the *declared original bytes* are present and match the
manifest's lengths and SHA-256 digests, with no unsafe or undeclared entries.
It does NOT prove canonical-state completeness, source authenticity,
decryptability, authorization, or that an import will succeed.  A checksum only
detects mismatch with the manifest; authenticity belongs to the reviewed
cryptographic protocol.

Usage: ``python -m portability_verifier PATH`` prints a JSON report and exits
0 (clean), 1 (findings), or 2 (usage).
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
import unicodedata
import zipfile
import zlib
from dataclasses import dataclass, field
from typing import Any, Iterator

from pydantic import ValidationError

from contracts_portability import (
    PortableManifest,
    normalize_bundle_path,
    validate_portable_manifest,
)

MANIFEST_NAME = "manifest.json"

# Provisional resource limits.  They are named, module-level and NOT an
# advertised product promise; thresholds are to be replaced from reviewed
# measurements.
MAX_BUNDLE_ENTRIES = 100_000
MAX_ENTRY_BYTES = 8 * 1024**3
MAX_TOTAL_BYTES = 64 * 1024**3
MAX_MANIFEST_BYTES = 16 * 1024**2
MAX_COMPRESSION_RATIO = 200
RATIO_MIN_ENTRY_BYTES = 1024**2
MAX_DIRECTORY_DEPTH = 64
_CHUNK = 64 * 1024
_READ_ERRORS = (
    OSError,
    zipfile.BadZipFile,
    zlib.error,
    EOFError,
    ValueError,
    NotImplementedError,
)

# Stable finding codes.
CONTAINER_UNREADABLE = "container_unreadable"
CONTAINER_UNSUPPORTED = "container_unsupported"
ENTRY_COUNT_LIMIT = "entry_count_limit"
SIZE_LIMIT = "size_limit"
RATIO_LIMIT = "ratio_limit"
OVERLAPPING_ENTRIES = "overlapping_entries"
PATH_UNSAFE = "path_unsafe"
SYMLINK_ENTRY = "symlink_entry"
HARDLINK_ENTRY = "hardlink_entry"
SPECIAL_ENTRY = "special_entry"
ENTRY_ENCRYPTED = "entry_encrypted"
DUPLICATE_ENTRY = "duplicate_entry"
CASE_COLLISION = "case_collision"
MANIFEST_MISSING = "manifest_missing"
MANIFEST_INVALID = "manifest_invalid"
MANIFEST_SELF_LISTED = "manifest_self_listed"
FILE_MISSING = "file_missing"
FILE_UNDECLARED = "file_undeclared"
SIZE_MISMATCH = "size_mismatch"
DIGEST_MISMATCH = "digest_mismatch"
ENTRY_CORRUPT = "entry_corrupt"

# Findings after which no payload byte may be read.
_FATAL_CODES = frozenset(
    {
        CONTAINER_UNREADABLE,
        CONTAINER_UNSUPPORTED,
        ENTRY_COUNT_LIMIT,
        SIZE_LIMIT,
        RATIO_LIMIT,
        OVERLAPPING_ENTRIES,
    }
)


@dataclass(frozen=True)
class Finding:
    code: str
    path: str | None = None
    detail: str = ""


@dataclass
class VerificationReport:
    findings: list[Finding] = field(default_factory=list)
    files_declared: int = 0
    files_verified: int = 0
    omissions: list[str] = field(default_factory=list)
    included_capabilities: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.findings

    def add(self, code: str, path: str | None = None, detail: str = "") -> None:
        self.findings.append(Finding(code, path, detail))

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "files_declared": self.files_declared,
            "files_verified": self.files_verified,
            "declared_omissions": list(self.omissions),
            "included_capabilities": list(self.included_capabilities),
            "findings": [
                {"code": f.code, "path": f.path, "detail": f.detail}
                for f in self.findings
            ],
            "scope_note": (
                "original byte hashes against the manifest only; canonical "
                "completeness, authenticity and importability are not claimed"
            ),
        }


@dataclass(frozen=True)
class _Entry:
    """One container entry classified without reading payload bytes."""

    name: str  # normalized, relative, POSIX
    size: int
    kind: str  # "file" or "dir"
    handle: Any  # ZipInfo or absolute path


def _fold(name: str) -> str:
    return unicodedata.normalize("NFC", name).casefold()


def _safe_label(value: str) -> str:
    return value.encode("ascii", "backslashreplace").decode("ascii")[:512]


def _classify_name(raw: str, report: VerificationReport) -> str | None:
    try:
        raw.encode("utf-8")
        return normalize_bundle_path(raw)
    except (ValueError, UnicodeEncodeError):
        report.add(PATH_UNSAFE, _safe_label(raw))
        return None


def _mode_finding(mode: int) -> str | None:
    if stat.S_ISLNK(mode):
        return SYMLINK_ENTRY
    if stat.S_IFMT(mode) in (0, stat.S_IFREG, stat.S_IFDIR):
        return None  # type bits are absent in many zip writers: plain file
    return SPECIAL_ENTRY


def _inventory_zip(
    zf: zipfile.ZipFile, report: VerificationReport
) -> list[_Entry]:
    infos = zf.infolist()
    if len(infos) > MAX_BUNDLE_ENTRIES:
        report.add(ENTRY_COUNT_LIMIT)
        return []
    total = 0
    for info in infos:
        if info.file_size > MAX_ENTRY_BYTES:
            report.add(SIZE_LIMIT, _safe_label(info.filename))
            return []
        total += info.file_size
        if (
            info.file_size >= RATIO_MIN_ENTRY_BYTES
            and info.file_size > MAX_COMPRESSION_RATIO * max(info.compress_size, 1)
        ):
            report.add(RATIO_LIMIT, _safe_label(info.filename))
            return []
    if total > MAX_TOTAL_BYTES:
        report.add(SIZE_LIMIT)
        return []

    # Overlapping local records are the zip-quine/overlap bomb signature.
    ordered = sorted(infos, key=lambda item: item.header_offset)
    for current, following in zip(ordered, ordered[1:]):
        if following.header_offset < current.header_offset + current.compress_size:
            report.add(OVERLAPPING_ENTRIES)
            return []

    entries: list[_Entry] = []
    for info in infos:
        is_dir = info.is_dir()
        name = _classify_name(info.filename.rstrip("/") if is_dir else info.filename, report)
        if name is None:
            continue
        mode = info.external_attr >> 16
        problem = _mode_finding(mode) if mode else None
        if problem is not None:
            report.add(problem, _safe_label(name))
            continue
        if info.flag_bits & 0x1:
            report.add(ENTRY_ENCRYPTED, _safe_label(name))
            continue
        if info.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
            report.add(CONTAINER_UNSUPPORTED, _safe_label(name), "compression method")
            continue
        entries.append(_Entry(name, info.file_size, "dir" if is_dir else "file", info))
    return entries


def _walk_directory(root: str, report: VerificationReport) -> list[_Entry]:
    entries: list[_Entry] = []
    total = 0
    stack: list[tuple[str, str, int]] = [(root, "", 0)]
    while stack:
        directory, prefix, depth = stack.pop()
        if depth > MAX_DIRECTORY_DEPTH:
            report.add(ENTRY_COUNT_LIMIT, _safe_label(prefix), "directory depth")
            return []
        try:
            with os.scandir(directory) as scan:
                children = sorted(scan, key=lambda item: item.name)
        except OSError:
            report.add(CONTAINER_UNREADABLE, _safe_label(prefix))
            return []
        for child in children:
            relative = f"{prefix}{child.name}"
            if len(entries) >= MAX_BUNDLE_ENTRIES:
                report.add(ENTRY_COUNT_LIMIT)
                return []
            try:
                info = child.stat(follow_symlinks=False)
            except OSError:
                report.add(CONTAINER_UNREADABLE, _safe_label(relative))
                continue
            name = _classify_name(os.fsdecode(relative), report)
            if name is None:
                continue
            problem = _mode_finding(info.st_mode)
            if problem is not None:
                report.add(problem, _safe_label(name))
                continue
            if stat.S_ISDIR(info.st_mode):
                entries.append(_Entry(name, 0, "dir", child.path))
                stack.append((child.path, f"{relative}/", depth + 1))
                continue
            if info.st_nlink > 1:
                report.add(HARDLINK_ENTRY, _safe_label(name))
                continue
            if info.st_size > MAX_ENTRY_BYTES:
                report.add(SIZE_LIMIT, _safe_label(name))
                return []
            total += info.st_size
            if total > MAX_TOTAL_BYTES:
                report.add(SIZE_LIMIT)
                return []
            entries.append(_Entry(name, info.st_size, "file", child.path))
    return entries


def _collision_checks(entries: list[_Entry], report: VerificationReport) -> None:
    seen_exact: dict[str, int] = {}
    seen_folded: dict[str, str] = {}
    for entry in entries:
        seen_exact[entry.name] = seen_exact.get(entry.name, 0) + 1
        if seen_exact[entry.name] == 2:
            report.add(DUPLICATE_ENTRY, _safe_label(entry.name))
    for name in seen_exact:
        folded = _fold(name)
        other = seen_folded.setdefault(folded, name)
        if other != name:
            report.add(CASE_COLLISION, _safe_label(name))


def _read_chunks(entry: _Entry, zf: zipfile.ZipFile | None) -> Iterator[bytes]:
    if zf is not None:
        with zf.open(entry.handle) as stream:
            while chunk := stream.read(_CHUNK):
                yield chunk
        return
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    descriptor = os.open(entry.handle, flags)
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or opened.st_nlink > 1:
            raise OSError("entry changed type during verification")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            while chunk := stream.read(_CHUNK):
                yield chunk
    finally:
        os.close(descriptor)


def _hash_entry(
    entry: _Entry, zf: zipfile.ZipFile | None, limit: int
) -> tuple[int, str] | None:
    """Return ``(bytes_read, sha256)``; reading stops one byte past ``limit``."""

    digest = hashlib.sha256()
    count = 0
    try:
        for chunk in _read_chunks(entry, zf):
            count += len(chunk)
            if count > limit:
                return count, ""
            digest.update(chunk)
    except _READ_ERRORS:
        return None
    return count, digest.hexdigest()


def _load_manifest(
    entries: dict[str, _Entry], zf: zipfile.ZipFile | None, report: VerificationReport
) -> PortableManifest | None:
    entry = entries.get(MANIFEST_NAME)
    if entry is None or entry.kind != "file":
        report.add(MANIFEST_MISSING)
        return None
    if entry.size > MAX_MANIFEST_BYTES:
        report.add(MANIFEST_INVALID, MANIFEST_NAME, "manifest too large")
        return None
    data = bytearray()
    try:
        for chunk in _read_chunks(entry, zf):
            data += chunk
            if len(data) > MAX_MANIFEST_BYTES:
                report.add(MANIFEST_INVALID, MANIFEST_NAME, "manifest too large")
                return None
    except _READ_ERRORS:
        report.add(ENTRY_CORRUPT, MANIFEST_NAME)
        return None

    def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate key")
            result[key] = value
        return result

    try:
        document = json.loads(bytes(data).decode("utf-8"), object_pairs_hook=_no_duplicate_keys)
        return validate_portable_manifest(document)
    except (ValueError, ValidationError, RecursionError, UnicodeDecodeError):
        report.add(MANIFEST_INVALID, MANIFEST_NAME)
        return None


def _verify_open(
    entries: list[_Entry], zf: zipfile.ZipFile | None, report: VerificationReport
) -> None:
    _collision_checks(entries, report)
    if any(item.code in _FATAL_CODES for item in report.findings):
        return
    by_name = {entry.name: entry for entry in entries}
    manifest = _load_manifest(by_name, zf, report)
    if manifest is None:
        return

    declared = {item.path: item for item in manifest.files}
    report.files_declared = len(declared)
    report.omissions = [item.code for item in manifest.omissions]
    report.included_capabilities = list(manifest.included_capabilities)
    if MANIFEST_NAME in declared:
        report.add(MANIFEST_SELF_LISTED, MANIFEST_NAME)

    for entry in entries:
        if (
            entry.kind == "file"
            and entry.name != MANIFEST_NAME
            and entry.name not in declared
        ):
            report.add(FILE_UNDECLARED, _safe_label(entry.name))

    candidates: list[tuple[str, _Entry]] = []
    for path, descriptor in declared.items():
        entry = by_name.get(path)
        if entry is None or entry.kind != "file":
            report.add(FILE_MISSING, _safe_label(path))
        elif entry.size != descriptor.size_bytes:
            report.add(SIZE_MISMATCH, _safe_label(path), "container length")
        else:
            candidates.append((path, entry))

    # Payload bytes are hashed only when every structural check above is clean.
    if report.findings:
        return
    for path, entry in candidates:
        descriptor = declared[path]
        result = _hash_entry(entry, zf, descriptor.size_bytes)
        if result is None:
            report.add(ENTRY_CORRUPT, _safe_label(path))
            continue
        count, hexdigest = result
        if count != descriptor.size_bytes:
            report.add(SIZE_MISMATCH, _safe_label(path), "payload length")
        elif hexdigest != descriptor.checksum.digest:
            report.add(DIGEST_MISMATCH, _safe_label(path))
        else:
            report.files_verified += 1


def verify_bundle(path: str | os.PathLike[str]) -> VerificationReport:
    """Verify a bundle directory or zip file; never extracts or mutates it."""

    report = VerificationReport()
    target = os.fspath(path)
    if os.path.islink(target):
        report.add(SYMLINK_ENTRY, "<bundle>")
        return report
    if os.path.isdir(target):
        entries = _walk_directory(target, report)
        if not any(item.code in _FATAL_CODES for item in report.findings):
            _verify_open(entries, None, report)
        return report
    if not os.path.isfile(target):
        report.add(CONTAINER_UNREADABLE, "<bundle>")
        return report
    try:
        archive = zipfile.ZipFile(target)
    except (zipfile.BadZipFile, OSError, EOFError, ValueError, NotImplementedError):
        report.add(CONTAINER_UNREADABLE, "<bundle>", "not a readable zip archive")
        return report
    with archive:
        entries = _inventory_zip(archive, report)
        if not any(item.code in _FATAL_CODES for item in report.findings):
            _verify_open(entries, archive, report)
    return report


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    if len(arguments) != 1 or arguments[0].startswith("-"):
        print("usage: python -m portability_verifier PATH", file=sys.stderr)
        return 2
    report = verify_bundle(arguments[0])
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return 0 if report.ok else 1


__all__ = [
    "Finding",
    "VerificationReport",
    "verify_bundle",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
