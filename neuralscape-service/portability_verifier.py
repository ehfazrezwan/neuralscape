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

import errno
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
from typing import Any, BinaryIO, Iterator, NamedTuple, Sequence

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
    handle: Any  # ZipInfo, or the path relative to the bundle root


def _fold(name: str) -> str:
    return unicodedata.normalize("NFC", name).casefold()


def _safe_label(value: str) -> str:
    """A printable label for a report: non-ASCII and control characters are escaped."""

    ascii_only = value.encode("ascii", "backslashreplace").decode("ascii")
    return "".join(ch if 32 <= ord(ch) < 127 else f"\\x{ord(ch):02x}" for ch in ascii_only)[:512]


def _classify_name(raw: str, report: VerificationReport) -> str | None:
    try:
        raw.encode("utf-8")
        return normalize_bundle_path(raw)
    except (ValueError, UnicodeEncodeError):
        report.add(PATH_UNSAFE, _safe_label(raw))
        return None


_UNICODE_PATH_FIELD = 0x7075  # Info-ZIP Unicode Path: the name some readers use instead of the header's


def _unicode_path_problem(extra: bytes, declared: str) -> str | None:
    """Why an extra-field area is not acceptable for a member declared as ``declared``, or ``None``.

    A Unicode Path field names the member for every reader that honours it, whatever the header says, so a
    bundle whose bytes and manifest match the declared name could still extract elsewhere.  A field is
    accepted only if it is well formed and names exactly the declared name; any other is refused.  Blocks are
    read as extractors read them: an area that stops making sense ends the walk.
    """

    position = 0
    while position + 4 <= len(extra):
        identifier, size = struct.unpack_from("<HH", extra, position)
        data = extra[position + 4 : position + 4 + size]
        if len(data) != size:
            break  # a truncated block ends the area for every reader
        position += 4 + size
        if identifier != _UNICODE_PATH_FIELD:
            continue
        if size < 6 or data[0] != 1:
            return "malformed unicode path field"
        try:
            alternate = data[5:].decode("utf-8")
        except UnicodeDecodeError:
            return "undecodable unicode path field"
        if alternate != declared:
            return "unicode path field names another path"
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


_END_RECORD_SENTINELS = (0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, _U32_MAX, _U32_MAX)


def _end_record_agrees(ordinary: Sequence[int], zip64: Sequence[int]) -> bool:
    """Whether each ordinary end-record field is its zip64 sentinel or exactly the zip64 value.

    Both tuples are ``(disk, directory disk, entries on this disk, entries, directory size,
    directory offset)``.  The archive library takes the zip64 values whenever the locator is
    present, but readers that use the ordinary fields when they are not saturated would then
    read a different directory, so a disagreement is refused rather than resolved either way.
    """

    return all(
        field == sentinel or field == value
        for field, sentinel, value in zip(ordinary, _END_RECORD_SENTINELS, zip64)
    )


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
    problem = _unicode_path_problem(extra, info.orig_filename)
    if problem is not None:
        raise _Framing(f"local header: {problem}")
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
        _sig, disk, cd_disk, entries_here, entries, cd_size, cd_offset, _comment = eocd
        end_of_directory = eocd_position
        saturated = (
            entries == 0xFFFF
            or cd_size == _U32_MAX
            or cd_offset == _U32_MAX
            or disk == 0xFFFF
            or cd_disk == 0xFFFF
        )
        # Like the archive library (and the precheck), a zip64 locator directly before the end
        # record means the zip64 end record governs, whether or not the ordinary fields are
        # saturated; saturated fields without a locator are refused, and ordinary fields that
        # disagree with the zip64 record are refused too.
        locator_position = eocd_position - 20
        locator = b""
        if locator_position >= 0:
            raw.seek(locator_position)
            locator = raw.read(20)
        if locator[:4] != _ZIP64_LOCATOR_SIGNATURE and saturated:
            raise _Framing("missing zip64 locator")
        if locator[:4] == _ZIP64_LOCATOR_SIGNATURE:
            record_offset = struct.unpack_from("<Q", locator, 8)[0]
            raw.seek(record_offset)
            record = raw.read(56)
            if len(record) != 56 or record[:4] != _ZIP64_EOCD_SIGNATURE:
                raise _Framing("bad zip64 end record")
            record_size = struct.unpack_from("<Q", record, 4)[0]
            if record_offset + 12 + record_size != locator_position:
                raise _Framing("gap before the zip64 locator")
            zip64_fields = struct.unpack_from("<IIQQQQ", record, 16)
            if not _end_record_agrees((disk, cd_disk, entries_here, entries, cd_size, cd_offset), zip64_fields):
                raise _Framing("end record disagrees with the zip64 end record")
            cd_size, cd_offset = zip64_fields[4], zip64_fields[5]
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


def _compressed_bound(info: zipfile.ZipInfo) -> int:
    """The most compressed bytes a member of this uncompressed size may occupy.

    A stored member is exactly its size.  Deflate never needs more than its stored-block
    framing (5 bytes per 64 KiB) over the input; the bound below is far more generous than
    that, and it caps the bytes read for a member that inflates to little or nothing.
    """

    if info.compress_type == zipfile.ZIP_STORED:
        return info.file_size
    return info.file_size + info.file_size // 8 + 1024


def _inventory_zip(
    source: _ZipSource, report: VerificationReport
) -> list[_Entry]:
    infos = source.archive.infolist()
    if len(infos) > MAX_BUNDLE_ENTRIES:
        report.add(ENTRY_COUNT_LIMIT)
        return []
    total = 0
    total_compressed = 0
    for info in infos:
        if info.file_size > MAX_ENTRY_BYTES:
            report.add(SIZE_LIMIT, _safe_label(info.filename))
            return []
        # Compressed bytes are what is read: a deflate stream may hold any number of empty blocks
        # that inflate to nothing, so they are bounded per member and across the archive too.
        if info.compress_size > MAX_ENTRY_BYTES or info.compress_size > _compressed_bound(info):
            report.add(SIZE_LIMIT, _safe_label(info.filename), "compressed size")
            return []
        total += info.file_size
        total_compressed += info.compress_size
        if (
            info.file_size >= RATIO_MIN_ENTRY_BYTES
            and info.file_size > MAX_COMPRESSION_RATIO * max(info.compress_size, 1)
        ):
            report.add(RATIO_LIMIT, _safe_label(info.filename))
            return []
    if total > MAX_TOTAL_BYTES or total_compressed > MAX_TOTAL_BYTES:
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
        problem = _unicode_path_problem(info.extra, raw_name)
        if problem is not None:
            report.add(PATH_UNSAFE, _safe_label(name), problem)
            continue
        # The same depth rule as a directory bundle: a directory may sit at most
        # MAX_DIRECTORY_DEPTH levels down, so a bundle cannot verify as a zip and fail
        # once extracted.
        depth = name.count("/") + (1 if is_dir else 0)
        if depth > MAX_DIRECTORY_DEPTH:
            report.add(ENTRY_COUNT_LIMIT, _safe_label(name), "directory depth")
            return []
        if is_dir and info.file_size != 0:
            report.add(ENTRY_CORRUPT, _safe_label(name), "directory record carries data")
            continue
        mode = info.external_attr >> 16
        if stat.S_ISDIR(mode) and not is_dir:
            # Typed as a directory but named as a file: extractors disagree on what it is.
            report.add(ENTRY_CORRUPT, _safe_label(name), "directory type without a directory name")
            continue
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


class _DirSource(NamedTuple):
    """A directory bundle opened once: every walk and read happens beneath ``root``."""

    root: int  # descriptor of the bundle directory, opened without following links


_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_CLOEXEC = getattr(os, "O_CLOEXEC", 0)


def _open_beneath(root: int, relative: str, *, directory: bool) -> int:
    """Open ``relative`` under the root descriptor, refusing a link at every component.

    Each component is opened relative to its parent's descriptor with ``O_NOFOLLOW``,
    so a directory swapped for a link after the walk is refused, never followed out of
    the bundle.
    """

    parts = relative.split("/")
    current = root
    opened: list[int] = []
    try:
        for part in parts[:-1]:
            current = os.open(part, os.O_RDONLY | os.O_DIRECTORY | _NOFOLLOW | _CLOEXEC, dir_fd=current)
            opened.append(current)
        flags = os.O_RDONLY | _NOFOLLOW | _CLOEXEC | getattr(os, "O_NONBLOCK", 0)
        if directory:
            flags |= os.O_DIRECTORY
        return os.open(parts[-1], flags, dir_fd=current)
    finally:
        for descriptor in opened:
            os.close(descriptor)


def _walk_directory(root: int, report: VerificationReport) -> list[_Entry]:
    entries: list[_Entry] = []
    total = 0
    observed = 0  # every child seen anywhere in the tree, accepted or not
    stack: list[tuple[str, str, int]] = [("", "", 0)]
    while stack:
        directory, prefix, depth = stack.pop()
        if depth > MAX_DIRECTORY_DEPTH:
            report.add(ENTRY_COUNT_LIMIT, _safe_label(prefix), "directory depth")
            return []
        unsorted: list[os.DirEntry] = []
        try:
            descriptor = root if not directory else _open_beneath(root, directory, directory=True)
            try:
                with os.scandir(descriptor) as scan:
                    for child in scan:
                        observed += 1
                        if observed > MAX_BUNDLE_ENTRIES:
                            report.add(ENTRY_COUNT_LIMIT)
                            return []
                        unsorted.append(child)
                # stat while the directory is open: the entries refer to this descriptor
                children = sorted(
                    ((child.name, child.stat(follow_symlinks=False)) for child in unsorted),
                    key=lambda item: item[0],
                )  # bounded by the limit above
            finally:
                if descriptor != root:
                    os.close(descriptor)
        except OSError:
            report.add(CONTAINER_UNREADABLE, _safe_label(prefix))
            return []
        for child_name, info in children:
            relative = f"{prefix}{child_name}"
            name = _classify_name(os.fsdecode(relative), report)
            if name is None:
                continue
            problem = _mode_finding(info.st_mode)
            if problem is not None:
                report.add(problem, _safe_label(name))
                continue
            if stat.S_ISDIR(info.st_mode):
                entries.append(_Entry(name, 0, "dir", relative))
                stack.append((relative, f"{relative}/", depth + 1))
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
            entries.append(_Entry(name, info.st_size, "file", relative))
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


def _read_chunks(entry: _Entry, zf: "_ZipSource | _DirSource") -> Iterator[bytes]:
    if isinstance(zf, _ZipSource):
        yield from _zip_member_chunks(entry.handle, zf.raw)
        return
    descriptor = _open_beneath(zf.root, entry.handle, directory=False)
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
    entry: _Entry, zf: "_ZipSource | _DirSource", limit: int
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
    entries: dict[str, _Entry], zf: "_ZipSource | _DirSource", report: VerificationReport
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
    entries: list[_Entry], zf: "_ZipSource | _DirSource", report: VerificationReport
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
    if isinstance(zf, _ZipSource):
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


_ZIP64_RECORD = struct.Struct("<4sQHHIIQQQQ")
_ZIP64_LOCATOR = struct.Struct("<4sIQI")
_CENTRAL_RECORD_SIZE = 46


def _precheck_central_directory(target: str | os.PathLike[str] | BinaryIO, report: VerificationReport) -> bool:
    """Bound and locate the central directory before the archive is parsed, or refuse the archive.

    The archive library reads the whole central directory and allocates one record object per
    entry before any limit can apply, and it tolerates layouts (a comment, prepended data, a zip64
    record away from its locator) under which a different end record or directory than the one a
    simple reader finds is used.  So this reads the archive exactly as the library will and accepts
    it only if that reading is unambiguous; anything else is refused here, before parsing:

    * the last 22 bytes must be the end record with no comment (the library's first choice; a
      comment is refused anyway by the framing check).  Nothing here depends on how far a given
      interpreter's parser searches for an end record: any other layout is refused;
    * when a zip64 locator sits directly before it (the library then always uses zip64), the zip64
      end record must be where the locator points and end exactly at the locator, and each ordinary
      end-record field must be its zip64 sentinel or exactly the zip64 value (readers that use the
      ordinary fields would otherwise read a different directory);
    * end-record fields saturated for zip64 without a locator, other disks, a directory that does not
      end where the (zip64) end record begins, or one that starts anywhere but its stored offset (data
      before the first record) are refused;
    * the directory may hold at most ``MAX_CENTRAL_DIRECTORY_BYTES`` and ``MAX_BUNDLE_ENTRIES``
      records, and its records must fill it exactly and match the declared count.
    """

    def refuse(code: str, detail: str = "") -> bool:
        if code == ARCHIVE_FRAMING:
            report.add(ARCHIVE_FRAMING, "<bundle>", detail)
        elif code == SIZE_LIMIT:
            report.add(SIZE_LIMIT, "<bundle>", "central directory")
        else:
            report.add(code)
        return False

    if isinstance(target, (str, bytes, os.PathLike)):
        try:
            with open(target, "rb") as opened:
                return _precheck_central_directory(opened, report)
        except OSError:
            report.add(CONTAINER_UNREADABLE, "<bundle>")
            return False
    raw = target  # the same open file the parser and the raw reads use afterwards
    try:
        raw.seek(0, os.SEEK_END)
        size = raw.tell()
        # Whatever window an interpreter's parser searches, only the strict layout is accepted below;
        # this tail only decides how a refusal is reported (no end record anywhere: not an archive).
        window = min(size, _EOCD.size + (1 << 16))
        raw.seek(size - window)
        tail = raw.read(window)
        if _EOCD_SIGNATURE not in tail or size < _EOCD.size:
            report.add(CONTAINER_UNREADABLE, "<bundle>")
            return False
        eocd_at = size - _EOCD.size
        (signature, disk, cd_disk, entries_here, entries, cd_size, cd_offset, comment_length) = (
            _EOCD.unpack(tail[-_EOCD.size:])
        )
        if signature != _EOCD_SIGNATURE or comment_length != 0:
            return refuse(ARCHIVE_FRAMING, "trailing data or archive comment after the end record")
        ordinary = (disk, cd_disk, entries_here, entries, cd_size, cd_offset)
        locator = b""
        if eocd_at >= _ZIP64_LOCATOR.size:
            raw.seek(eocd_at - _ZIP64_LOCATOR.size)
            locator = raw.read(_ZIP64_LOCATOR.size)
        if locator[:4] == _ZIP64_LOCATOR_SIGNATURE:
            locator_at = eocd_at - _ZIP64_LOCATOR.size
            _sig, locator_disk, record_at, disks = _ZIP64_LOCATOR.unpack(locator)
            if locator_disk != 0 or disks != 1:
                return refuse(ARCHIVE_FRAMING, "multi-disk archive")
            if record_at + _ZIP64_RECORD.size > locator_at:
                return refuse(ARCHIVE_FRAMING, "bad zip64 end record")
            raw.seek(record_at)
            record = raw.read(_ZIP64_RECORD.size)
            (record_signature, record_size, _made, _needed, disk, cd_disk,
             entries_here, entries, cd_size, cd_offset) = _ZIP64_RECORD.unpack(record)
            # The record (with any extensible data it declares) must end exactly at the locator.
            if record_signature != _ZIP64_EOCD_SIGNATURE or record_at + 12 + record_size != locator_at:
                return refuse(ARCHIVE_FRAMING, "bad zip64 end record")
            if not _end_record_agrees(ordinary, (disk, cd_disk, entries_here, entries, cd_size, cd_offset)):
                return refuse(ARCHIVE_FRAMING, "end record disagrees with the zip64 end record")
            directory_end = record_at
        else:
            if entries == 0xFFFF or entries_here == 0xFFFF or cd_size == _U32_MAX or cd_offset == _U32_MAX:
                return refuse(ARCHIVE_FRAMING, "missing zip64 locator")
            directory_end = eocd_at
        if disk != 0 or cd_disk != 0 or entries_here != entries:
            return refuse(ARCHIVE_FRAMING, "multi-disk archive")
        if cd_size > MAX_CENTRAL_DIRECTORY_BYTES:
            return refuse(SIZE_LIMIT)
        if entries > MAX_BUNDLE_ENTRIES:
            return refuse(ENTRY_COUNT_LIMIT)
        # The library locates the directory at ``directory_end - cd_size`` and treats any difference
        # from the stored offset as prepended data; a bundle is never a self-extracting archive.
        if cd_offset + cd_size != directory_end:
            return refuse(ARCHIVE_FRAMING, "data before the first record")
        raw.seek(cd_offset)
        directory = raw.read(cd_size)
    except OSError:
        report.add(CONTAINER_UNREADABLE, "<bundle>")
        return False
    if len(directory) != cd_size:
        return refuse(ARCHIVE_FRAMING, "short central directory")
    records = 0
    cursor = 0
    while cursor < cd_size:
        if cursor + _CENTRAL_RECORD_SIZE > cd_size or directory[cursor : cursor + 4] != b"PK\x01\x02":
            return refuse(ARCHIVE_FRAMING, "central directory is not a sequence of records")
        records += 1
        if records > MAX_BUNDLE_ENTRIES:
            return refuse(ENTRY_COUNT_LIMIT)
        name_length, extra_length, comment_length = struct.unpack_from("<HHH", directory, cursor + 28)
        cursor += _CENTRAL_RECORD_SIZE + name_length + extra_length + comment_length
    if cursor != cd_size or records != entries:
        return refuse(ARCHIVE_FRAMING, "central directory does not match its declared size and count")
    return True


def verify_bundle(path: str | os.PathLike[str]) -> VerificationReport:
    """Verify a bundle directory or zip file; never extracts or mutates it."""

    report = VerificationReport()
    target = os.fspath(path)
    if os.path.islink(target):
        report.add(SYMLINK_ENTRY, "<bundle>")
        return report
    # A directory is opened once, without following a link, and walked and read beneath
    # that descriptor, so a root or directory swapped for a link later is never followed.
    try:
        root = os.open(target, os.O_RDONLY | os.O_DIRECTORY | _NOFOLLOW | _CLOEXEC)
    except OSError as exc:
        try:
            linked = exc.errno in (errno.ELOOP, errno.ENOTDIR) and stat.S_ISLNK(os.lstat(target).st_mode)
        except OSError:
            linked = False
        if linked:
            report.add(SYMLINK_ENTRY, "<bundle>")
            return report
        root = None  # not a directory: the archive path below decides
    if root is not None:
        try:
            entries = _walk_directory(root, report)
            if not any(item.code in _FATAL_CODES for item in report.findings):
                _verify_open(entries, _DirSource(root), report)
        finally:
            os.close(root)
        return report
    if not os.path.isfile(target):
        report.add(CONTAINER_UNREADABLE, "<bundle>")
        return report
    # One open file serves the precheck, the parser and every raw read, so replacing the pathname
    # after the precheck cannot put unchecked bytes in front of the parser.
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(target, flags)
    except OSError:
        report.add(CONTAINER_UNREADABLE, "<bundle>")
        return report
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            os.close(descriptor)
            report.add(CONTAINER_UNREADABLE, "<bundle>")
            return report
        raw = os.fdopen(descriptor, "rb")
    except OSError:
        os.close(descriptor)
        report.add(CONTAINER_UNREADABLE, "<bundle>")
        return report
    try:
        with raw:
            if not _precheck_central_directory(raw, report):
                return report
            try:
                archive = zipfile.ZipFile(raw)
            except (zipfile.BadZipFile, OSError, EOFError, ValueError, NotImplementedError):
                report.add(CONTAINER_UNREADABLE, "<bundle>", "not a readable zip archive")
                return report
            with archive:
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
