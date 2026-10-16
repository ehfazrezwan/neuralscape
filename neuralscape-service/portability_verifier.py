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

Zip framing is strict so different parsers cannot read different archives: the
file must be exactly contiguous local records, the central directory, and an
end record with an empty comment (no leading or trailing bytes, gaps, or
embedded archives).  Each local header must agree with the central directory,
and each member's stored stream must end exactly where, and be exactly as long
as, the directory says (stored sizes equal; deflate ends with no leftover
bytes, inflates to the declared length and CRC).

Worst-case CPU is bounded by the provisional limits, not by input size: up to
``MAX_TOTAL_BYTES`` of payload can be inflated and hashed, at ratios below
``MAX_COMPRESSION_RATIO`` for members of at least ``RATIO_MIN_ENTRY_BYTES``
(smaller members are unconstrained by ratio), across at most
``MAX_BUNDLE_ENTRIES`` records.  A tiny archive can therefore still cost a
long single-threaded run; callers that need a tighter bound must lower the
constants.

Usage: ``python -m portability_verifier PATH`` prints a JSON report and exits
0 (clean), 1 (findings), or 2 (usage).
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import struct
import sys
import unicodedata
import zipfile
import zlib
from dataclasses import dataclass, field
from typing import Any, BinaryIO, Iterator, NamedTuple

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
MAX_CENTRAL_DIRECTORY_BYTES = 64 * 1024**2
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
ARCHIVE_FRAMING = "archive_framing"
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
        ARCHIVE_FRAMING,
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


class _ZipSource(NamedTuple):
    """An opened archive plus an independent raw handle on the same bytes."""

    archive: zipfile.ZipFile
    raw: BinaryIO


_LOCAL_SIGNATURE = b"PK\x03\x04"
_EOCD_SIGNATURE = b"PK\x05\x06"
_ZIP64_LOCATOR_SIGNATURE = b"PK\x06\x07"
_ZIP64_EOCD_SIGNATURE = b"PK\x06\x06"
_DESCRIPTOR_SIGNATURE = b"PK\x07\x08"
_FLAG_DESCRIPTOR = 0x8
_FLAG_UTF8 = 0x800
_LOCAL_HEADER = struct.Struct("<4sHHHHHIIIHH")
_EOCD = struct.Struct("<4sHHHHIIH")
_U32_MAX = 0xFFFFFFFF


class _Framing(Exception):
    """The archive bytes are not exactly the records the directory claims."""


def _zip64_local_sizes(extra: bytes, want_usize: bool, want_csize: bool) -> tuple[int, int]:
    position = 0
    while position + 4 <= len(extra):
        header_id, length = struct.unpack_from("<HH", extra, position)
        body = extra[position + 4 : position + 4 + length]
        if header_id == 0x0001:
            offset = 0
            usize = csize = -1
            if want_usize:
                usize = struct.unpack_from("<Q", body, offset)[0]
                offset += 8
            if want_csize:
                csize = struct.unpack_from("<Q", body, offset)[0]
            return usize, csize
        position += 4 + length
    raise _Framing("missing zip64 sizes")


def _local_record_end(raw: BinaryIO, info: zipfile.ZipInfo, start_dir: int) -> int:
    """Return the offset just past this member's local record.

    Compares the local header with the central-directory view of the member
    (name, flags, method, CRC and both sizes) and accounts for the data
    descriptor, so nothing the central directory does not describe can hide
    inside the record.  Payload bytes are not read here.
    """

    raw.seek(info.header_offset)
    head = raw.read(_LOCAL_HEADER.size)
    if len(head) != _LOCAL_HEADER.size:
        raise _Framing("short local header")
    (signature, _version, flags, method, _time, _date, crc, csize, usize, name_length, extra_length) = (
        _LOCAL_HEADER.unpack(head)
    )
    if signature != _LOCAL_SIGNATURE:
        raise _Framing("bad local header signature")
    name = raw.read(name_length)
    extra = raw.read(extra_length)
    if len(name) != name_length or len(extra) != extra_length:
        raise _Framing("short local header")
    try:
        decoded = name.decode("utf-8" if flags & _FLAG_UTF8 else "cp437")
    except UnicodeDecodeError:
        raise _Framing("undecodable local name") from None
    if decoded != info.orig_filename:
        raise _Framing("local and central names differ")
    if flags != info.flag_bits or method != info.compress_type:
        raise _Framing("local and central flags or method differ")

    if not flags & _FLAG_DESCRIPTOR:
        if csize == _U32_MAX or usize == _U32_MAX:
            zip64_usize, zip64_csize = _zip64_local_sizes(
                extra, usize == _U32_MAX, csize == _U32_MAX
            )
            if usize == _U32_MAX:
                usize = zip64_usize
            if csize == _U32_MAX:
                csize = zip64_csize
        if (crc, csize, usize) != (info.CRC, info.compress_size, info.file_size):
            raise _Framing("local and central CRC or sizes differ")
    elif (csize, usize) not in ((0, 0), (info.compress_size, info.file_size)) and not (
        csize == _U32_MAX or usize == _U32_MAX
    ):
        raise _Framing("local sizes conflict with the central directory")

    data_end = info.header_offset + _LOCAL_HEADER.size + name_length + extra_length + info.compress_size
    if data_end > start_dir:
        raise _Framing("member data overruns the central directory")
    if not flags & _FLAG_DESCRIPTOR:
        return data_end

    raw.seek(data_end)
    probe = raw.read(24)
    for with_signature in (True, False):
        for width, fmt in ((4, "<III"), (8, "<IQQ")):
            prefix = 4 if with_signature else 0
            size = prefix + struct.calcsize(fmt)
            if len(probe) < size:
                continue
            if with_signature and probe[:4] != _DESCRIPTOR_SIGNATURE:
                continue
            if struct.unpack_from(fmt, probe, prefix) == (info.CRC, info.compress_size, info.file_size):
                return data_end + size
    raise _Framing("data descriptor does not match the central directory")


def _check_framing(
    source: _ZipSource, infos: list[zipfile.ZipInfo], report: VerificationReport
) -> bool:
    """Require the file to be exactly: local records, central directory, EOCD.

    Rejects bytes before the first record or after the end record, a
    non-empty archive comment (which could embed another archive), unaccounted
    gaps between records, and any local/central disagreement, so different zip
    parsers cannot select different archives or different member bytes.
    """

    raw, archive = source.raw, source.archive
    try:
        raw.seek(0, os.SEEK_END)
        size = raw.tell()
        if size < _EOCD.size:
            raise _Framing("too short")
        raw.seek(size - _EOCD.size)
        eocd = _EOCD.unpack(raw.read(_EOCD.size))
        if eocd[0] != _EOCD_SIGNATURE or eocd[7] != 0:
            raise _Framing("trailing data or archive comment after the end record")
        eocd_position = size - _EOCD.size
        _sig, disk, cd_disk, _entries_here, entries, cd_size, cd_offset, _comment = eocd
        end_of_directory = eocd_position
        if (
            entries == 0xFFFF
            or cd_size == _U32_MAX
            or cd_offset == _U32_MAX
            or disk == 0xFFFF
            or cd_disk == 0xFFFF
        ):
            locator_position = eocd_position - 20
            if locator_position < 0:
                raise _Framing("missing zip64 locator")
            raw.seek(locator_position)
            locator = raw.read(20)
            if locator[:4] != _ZIP64_LOCATOR_SIGNATURE:
                raise _Framing("missing zip64 locator")
            record_offset = struct.unpack_from("<Q", locator, 8)[0]
            raw.seek(record_offset)
            record = raw.read(56)
            if len(record) != 56 or record[:4] != _ZIP64_EOCD_SIGNATURE:
                raise _Framing("bad zip64 end record")
            record_size = struct.unpack_from("<Q", record, 4)[0]
            if record_offset + 12 + record_size != locator_position:
                raise _Framing("gap before the zip64 locator")
            cd_size, cd_offset = struct.unpack_from("<QQ", record, 40)
            end_of_directory = record_offset
        if cd_offset + cd_size != end_of_directory:
            raise _Framing("gap between the central directory and the end record")
        if cd_offset != archive.start_dir:
            raise _Framing("data before the first record")

        ordered = sorted(infos, key=lambda item: item.header_offset)
        expected = 0
        for info in ordered:
            if info.header_offset != expected:
                raise _Framing("unaccounted bytes between records")
            expected = _local_record_end(raw, info, cd_offset)
        if expected != cd_offset:
            raise _Framing("unaccounted bytes before the central directory")
    except (_Framing, struct.error, OSError, ValueError) as exc:
        report.add(ARCHIVE_FRAMING, "<bundle>", str(exc)[:80])
        return False
    return True


def _inventory_zip(
    source: _ZipSource, report: VerificationReport
) -> list[_Entry]:
    infos = source.archive.infolist()
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
    if not _check_framing(source, infos, report):
        return []

    entries: list[_Entry] = []
    for info in infos:
        # ``filename`` is cut at the first NUL by the archive library; the raw name
        # is what other extractors use, so classify that one.
        raw_name = info.orig_filename
        is_dir = raw_name.endswith("/")
        name = _classify_name(raw_name.rstrip("/") if is_dir else raw_name, report)
        if name is None:
            continue
        if is_dir and info.file_size != 0:
            report.add(ENTRY_CORRUPT, _safe_label(name), "directory record carries data")
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
    observed = 0  # every child seen anywhere in the tree, accepted or not
    stack: list[tuple[str, str, int]] = [(root, "", 0)]
    while stack:
        directory, prefix, depth = stack.pop()
        if depth > MAX_DIRECTORY_DEPTH:
            report.add(ENTRY_COUNT_LIMIT, _safe_label(prefix), "directory depth")
            return []
        unsorted: list[os.DirEntry] = []
        try:
            with os.scandir(directory) as scan:
                for child in scan:
                    observed += 1
                    if observed > MAX_BUNDLE_ENTRIES:
                        report.add(ENTRY_COUNT_LIMIT)
                        return []
                    unsorted.append(child)
        except OSError:
            report.add(CONTAINER_UNREADABLE, _safe_label(prefix))
            return []
        children = sorted(unsorted, key=lambda item: item.name)  # bounded by the limit above
        for child in children:
            relative = f"{prefix}{child.name}"
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


def _zip_member_chunks(info: zipfile.ZipInfo, raw: BinaryIO) -> Iterator[bytes]:
    """Yield a member's bytes after proving the stored stream matches the directory.

    The compressed span is read directly (not through ``zipfile``, which stops
    at the central size) and must be consumed exactly: a stored member's two
    sizes agree; a deflate stream ends exactly at ``compress_size`` with no
    trailing bytes and inflates to exactly ``file_size`` with the central CRC.
    """

    raw.seek(info.header_offset)
    head = _LOCAL_HEADER.unpack(raw.read(_LOCAL_HEADER.size))
    raw.seek(info.header_offset + _LOCAL_HEADER.size + head[9] + head[10])
    remaining = info.compress_size
    produced = 0
    crc = 0
    if info.compress_type == zipfile.ZIP_STORED:
        if info.compress_size != info.file_size:
            raise ValueError("stored member sizes differ")
        decompressor = None
    else:
        decompressor = zlib.decompressobj(-15)
    pending = b""
    while True:
        if decompressor is None:
            chunk = raw.read(min(_CHUNK, remaining))
            if len(chunk) != min(_CHUNK, remaining):
                raise ValueError("truncated member")
            remaining -= len(chunk)
        else:
            if not pending and not decompressor.eof and remaining:
                pending = raw.read(min(_CHUNK, remaining))
                if not pending:
                    raise ValueError("truncated member")
                remaining -= len(pending)
            if not pending and not decompressor.eof:
                raise ValueError("deflate stream ended early")
            chunk = decompressor.decompress(pending, _CHUNK)
            pending = decompressor.unconsumed_tail
            if decompressor.eof:
                if decompressor.unused_data or pending or remaining:
                    raise ValueError("bytes after the deflate stream")
        produced += len(chunk)
        if produced > info.file_size:
            raise ValueError("member inflates past its declared size")
        crc = zlib.crc32(chunk, crc)
        if chunk:
            yield chunk
        finished = remaining == 0 if decompressor is None else decompressor.eof
        if finished and (decompressor is None or not pending):
            break
    if produced != info.file_size or crc != info.CRC:
        raise ValueError("member length or CRC differs from the directory")


def _read_chunks(entry: _Entry, zf: _ZipSource | None) -> Iterator[bytes]:
    if zf is not None:
        yield from _zip_member_chunks(entry.handle, zf.raw)
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
    entry: _Entry, zf: _ZipSource | None, limit: int
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
    entries: dict[str, _Entry], zf: _ZipSource | None, report: VerificationReport
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
    entries: list[_Entry], zf: _ZipSource | None, report: VerificationReport
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
    if zf is not None:
        # A directory record must hold an empty stream, whatever its headers claim.
        for entry in entries:
            if entry.kind == "dir":
                result = _hash_entry(entry, zf, 0)
                if result is None or result[0] != 0:
                    report.add(ENTRY_CORRUPT, _safe_label(entry.name), "directory record carries data")
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


def _precheck_central_directory(target: str, report: VerificationReport) -> bool:
    """Bound the central directory with small reads before the archive is parsed.

    The archive library reads the whole central directory and allocates one record
    object per entry before any limit can apply.  This reads only the end record
    (and the zip64 end record when present), refuses a directory larger than
    ``MAX_CENTRAL_DIRECTORY_BYTES`` or declaring more than ``MAX_BUNDLE_ENTRIES``
    records, then walks the directory itself counting records, so a lying end-record
    count cannot bypass the limit.  An archive without a locatable end record is left
    to the parser, which reports it unreadable.
    """

    try:
        with open(target, "rb") as raw:
            raw.seek(0, os.SEEK_END)
            size = raw.tell()
            window = min(size, _EOCD.size + 0xFFFF)
            raw.seek(size - window)
            tail = raw.read(window)
            position = tail.rfind(_EOCD_SIGNATURE)
            if position < 0 or len(tail) - position < _EOCD.size:
                return True
            fields = _EOCD.unpack(tail[position : position + _EOCD.size])
            entries, cd_size, cd_offset = fields[4], fields[5], fields[6]
            eocd_at = size - window + position
            directory_end = eocd_at  # where the directory must end, as the parser computes it
            # Like the parser, use a zip64 end record whenever a locator sits directly
            # before the end record, whatever the end record's own fields say.
            locator_at = eocd_at - 20
            locator = b""
            if locator_at >= 0:
                raw.seek(locator_at)
                locator = raw.read(20)
            saturated = entries == 0xFFFF or cd_size == _U32_MAX or cd_offset == _U32_MAX
            if locator[:4] != _ZIP64_LOCATOR_SIGNATURE and saturated:
                return True  # left to the parser, which reports the archive unreadable
            if locator[:4] == _ZIP64_LOCATOR_SIGNATURE:
                # Read the zip64 end record where the parser does: immediately before
                # the locator, not at the locator's stored offset.
                record_at = locator_at - 56
                if record_at < 0:
                    return True
                raw.seek(record_at)
                record = raw.read(56)
                if len(record) != 56 or record[:4] != _ZIP64_EOCD_SIGNATURE:
                    return True
                entries = struct.unpack_from("<Q", record, 32)[0]
                cd_size, cd_offset = struct.unpack_from("<QQ", record, 40)
                directory_end = record_at  # the zip64 end record follows the directory
            if cd_size > MAX_CENTRAL_DIRECTORY_BYTES:
                report.add(SIZE_LIMIT, "<bundle>", "central directory")
                return False
            if entries > MAX_BUNDLE_ENTRIES:
                report.add(ENTRY_COUNT_LIMIT)
                return False
            # The archive library locates the directory at ``directory_end - cd_size``
            # and treats any difference from the stored offset as prepended data.  A
            # bundle is never a self-extracting archive, so any such difference is
            # refused here, before parsing; the walk below reads exactly where the
            # parser would.
            start = directory_end - cd_size
            if start < 0 or start != cd_offset:
                report.add(ARCHIVE_FRAMING, "<bundle>", "data before the first record")
                return False
            raw.seek(start)
            directory = raw.read(cd_size)
    except OSError:
        return True
    records = 0
    cursor = 0
    while cursor + 46 <= len(directory) and directory[cursor : cursor + 4] == b"PK\x01\x02":
        records += 1
        if records > MAX_BUNDLE_ENTRIES:
            report.add(ENTRY_COUNT_LIMIT)
            return False
        name_length, extra_length, comment_length = struct.unpack_from("<HHH", directory, cursor + 28)
        cursor += 46 + name_length + extra_length + comment_length
    return True


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
    if not _precheck_central_directory(target, report):
        return report
    try:
        archive = zipfile.ZipFile(target)
    except (zipfile.BadZipFile, OSError, EOFError, ValueError, NotImplementedError):
        report.add(CONTAINER_UNREADABLE, "<bundle>", "not a readable zip archive")
        return report
    try:
        with archive, open(target, "rb") as raw:
            source = _ZipSource(archive, raw)
            entries = _inventory_zip(source, report)
            if not any(item.code in _FATAL_CODES for item in report.findings):
                _verify_open(entries, source, report)
    except OSError:
        report.add(CONTAINER_UNREADABLE, "<bundle>")
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
