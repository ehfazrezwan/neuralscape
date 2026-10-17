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

import contextvars
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
    names_below_a_file,
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
# Everything reading a zip that another writer may truncate or change can raise: a short header unpacks to a
# struct.error, an offset beyond what a seek accepts to an OverflowError, a damaged stream to a zlib.error.
_READ_ERRORS = (
    OSError,
    zipfile.BadZipFile,
    zlib.error,
    EOFError,
    ValueError,
    NotImplementedError,
    struct.error,
    OverflowError,
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
UNSAFE_MODE = "unsafe_entry_mode"
ENTRY_ENCRYPTED = "entry_encrypted"
DUPLICATE_ENTRY = "duplicate_entry"
CASE_COLLISION = "case_collision"
PATH_CONFLICT = "path_conflict"
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
    """The key under which two names are aliases: canonical caseless matching (normalize, fold, normalize,
    fold, normalize).  Case folding can produce decomposed text, so a single normalization before it misses
    names that only become equal afterwards."""

    folded = unicodedata.normalize("NFD", name).casefold()
    return unicodedata.normalize("NFD", unicodedata.normalize("NFD", folded).casefold())


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


def _mode_finding(mode: int, *, declared: bool = False) -> str | None:
    """A refusal code for an entry's mode, or ``None``.

    A set-user-id, set-group-id or sticky bit is refused wherever the mode comes from.  A zip member's mode
    is a declaration the extractor honours, so there it must also be usable and, for a file, not writable by
    everyone: world-writable files and members with no permission bits at all (which cannot be read back)
    are refused.  A directory may be world-writable: ``ZipFile.mkdir`` writes every directory as 0o777.
    """

    if stat.S_ISLNK(mode):
        return SYMLINK_ENTRY
    if stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR):
        return SPECIAL_ENTRY  # type bits are absent in many zip writers: plain file
    if mode & (stat.S_ISUID | stat.S_ISGID | stat.S_ISVTX):
        return UNSAFE_MODE
    if declared and (not mode & 0o777 or (mode & stat.S_IWOTH and stat.S_IFMT(mode) != stat.S_IFDIR)):
        return UNSAFE_MODE
    return None


class _ZipSource(NamedTuple):
    """An opened archive plus an independent raw handle on the same bytes."""

    archive: zipfile.ZipFile
    raw: BinaryIO


_LOCAL_SIGNATURE = b"PK\x03\x04"
_EOCD_SIGNATURE = b"PK\x05\x06"
_ZIP64_LOCATOR_SIGNATURE = b"PK\x06\x07"
_ZIP64_EOCD_SIGNATURE = b"PK\x06\x06"
_DESCRIPTOR_SIGNATURE = b"PK\x07\x08"
_FLAG_ENCRYPTED = 0x1
_FLAG_DESCRIPTOR = 0x8
_FLAG_PATCHED_DATA = 0x20
_FLAG_STRONG_ENCRYPTION = 0x40
_FLAG_MASKED_HEADERS = 0x2000
_FLAG_UTF8 = 0x800
_DOS_DIRECTORY = 0x10  # low byte of external_attr: the MS-DOS directory attribute
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


def _extra_tiling_problem(extra: bytes) -> str | None:
    """Why an extra field's blocks do not tile it exactly, or ``None``.

    Readers walk the blocks by their declared lengths.  A partial block header, or a block that runs past the
    field, is a field each reader cuts differently (some stop, some read on into the next structure), so a
    member whose sizes or name come from that field is not read alike everywhere.
    """

    position = 0
    while position < len(extra):
        if position + 4 > len(extra):
            return "partial block header in the extra field"
        length = struct.unpack_from("<H", extra, position + 2)[0]
        if position + 4 + length > len(extra):
            return "an extra field block runs past the extra field"
        position += 4 + length
    return None


def _zip64_local_sizes(extra: bytes, want_usize: bool, want_csize: bool) -> tuple[int, int]:
    position = 0
    while position + 4 <= len(extra):
        header_id, length = struct.unpack_from("<HH", extra, position)
        body = extra[position + 4 : position + 4 + length]
        if header_id == 0x0001:
            # A block that runs past the extra field, or cannot hold the sizes the header defers to it, is
            # not one every reader parses alike: a reader that honours its length finds no sizes there.
            if len(body) != length:
                raise _Framing("zip64 extra field runs past the extra field")
            if length < 8 * (want_usize + want_csize):
                raise _Framing("zip64 extra field too short for its sizes")
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
    (signature, version_needed, flags, method, _time, _date, crc, csize, usize, name_length, extra_length) = (
        _LOCAL_HEADER.unpack(head)
    )
    if signature != _LOCAL_SIGNATURE:
        raise _Framing("bad local header signature")
    if version_needed > zipfile.MAX_EXTRACT_VERSION:
        raise _Framing("version needed to extract is not supported")
    if info.volume != 0:
        # The end records say one disk; a member whose own start-disk field says otherwise is expected
        # elsewhere by every reader that honours the field.
        raise _Framing("member starts on another disk")
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
    problem = _unicode_path_problem(extra, info.orig_filename) or _extra_tiling_problem(extra)
    if problem is not None:
        raise _Framing(f"local header: {problem}")
    if flags != info.flag_bits or method != info.compress_type:
        raise _Framing("local and central flags or method differ")

    # A size at its sentinel is resolved from the zip64 block before anything else is decided, streamed or not:
    # the block must exist and carry every size the header defers to it.  The sentinels also say which
    # descriptor follows a streamed member: wide for zip64, narrow otherwise.
    wide_descriptor = csize == _U32_MAX or usize == _U32_MAX
    if wide_descriptor:
        zip64_usize, zip64_csize = _zip64_local_sizes(extra, usize == _U32_MAX, csize == _U32_MAX)
        if usize == _U32_MAX:
            usize = zip64_usize
        if csize == _U32_MAX:
            csize = zip64_csize
    if not flags & _FLAG_DESCRIPTOR:
        if (crc, csize, usize) != (info.CRC, info.compress_size, info.file_size):
            raise _Framing("local and central CRC or sizes differ")
    elif (csize, usize) not in ((0, 0), (info.compress_size, info.file_size)):
        raise _Framing("local sizes conflict with the central directory")
    elif crc not in (0, info.CRC):
        # A streamed member's local CRC is a placeholder (zero) or the real one, never anything else.
        raise _Framing("local CRC disagrees with the central directory")

    data_end = info.header_offset + _LOCAL_HEADER.size + name_length + extra_length + info.compress_size
    if data_end > start_dir:
        raise _Framing("member data overruns the central directory")
    if not flags & _FLAG_DESCRIPTOR:
        return data_end

    raw.seek(data_end)
    probe = raw.read(24)
    # One format, chosen by the local header: a wide descriptor of an empty member (signature, CRC, two zero
    # sizes, then eight more zero bytes) begins exactly like a narrow one, so trying both would stop early.
    for with_signature in (True, False):
        for width, fmt in (((8, "<IQQ"),) if wide_descriptor else ((4, "<III"),)):
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
        if (stat.S_ISDIR(mode) or info.external_attr & _DOS_DIRECTORY) and not is_dir:
            # Typed as a directory (a Unix mode, or the DOS attribute a Windows writer sets) but named as a
            # file: extractors disagree on what it is.
            report.add(ENTRY_CORRUPT, _safe_label(name), "directory type without a directory name")
            continue
        problem = _mode_finding(mode, declared=True) if mode else None
        if problem is not None:
            report.add(problem, _safe_label(name))
            continue
        if not info.flag_bits & _FLAG_UTF8 and not info.orig_filename.isascii():
            # Read as cp437 here, but a reader that guesses UTF-8 reads the same bytes as other text: the
            # name a manifest declared for these bytes is not the name every extractor would write.
            report.add(CONTAINER_UNSUPPORTED, _safe_label(name), "non-ASCII name without the UTF-8 flag")
            continue
        if info.flag_bits & (_FLAG_ENCRYPTED | _FLAG_STRONG_ENCRYPTION):
            report.add(ENTRY_ENCRYPTED, _safe_label(name))
            continue
        if info.flag_bits & (_FLAG_PATCHED_DATA | _FLAG_MASKED_HEADERS):
            # The standard reader refuses patched data; masked header values mean an encrypted directory.
            report.add(CONTAINER_UNSUPPORTED, _safe_label(name), "unsupported general-purpose flag")
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
    # A regular file cannot also be a directory: every entry (an implicit parent too, and the manifest file
    # itself) is checked against the regular-file names under the same folding, component by component.
    for name in names_below_a_file(
        (entry.name for entry in entries),
        (entry.name for entry in entries if entry.kind == "file"),
        _fold,
    ):
        report.add(PATH_CONFLICT, _safe_label(name), "an ancestor is a file")
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
    header = raw.read(_LOCAL_HEADER.size)
    if len(header) != _LOCAL_HEADER.size:
        raise ValueError("truncated local header")
    head = _LOCAL_HEADER.unpack(header)
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
            # All the compressed input may be consumed while zlib still holds buffered output (a stream of
            # 65,537 identical bytes inflates past one chunk); ``decompress(b"", n)`` drains it.  The stream
            # is truncated only when that yields nothing and it has not ended.
            chunk = decompressor.decompress(pending, _CHUNK)
            if not pending and not chunk and not decompressor.eof:
                raise ValueError("deflate stream ended early")
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


# Where the precheck leaves the end records and central directory it read, for the parser (set by the verifier
# for the duration of one archive; a context slot rather than a parameter, so callers and tests that wrap
# the precheck keep its two-argument shape).
_PINNED_TAIL: "contextvars.ContextVar[list[tuple[int, bytes]] | None]" = contextvars.ContextVar(
    "portability_pinned_tail", default=None
)


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
            with open(_exact_path(target), "rb") as opened:
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
        end_record = tail[-_EOCD.size:]
        (signature, disk, cd_disk, entries_here, entries, cd_size, cd_offset, comment_length) = (
            _EOCD.unpack(end_record)
        )
        if signature != _EOCD_SIGNATURE or comment_length != 0:
            return refuse(ARCHIVE_FRAMING, "trailing data or archive comment after the end record")
        ordinary = (disk, cd_disk, entries_here, entries, cd_size, cd_offset)
        # Every end-record byte the bounds below are derived from, by file offset: the copy taken
        # after them must hold exactly these bytes, or the parser would read other end records.
        bounded_end_records = [(eocd_at, end_record)]
        locator = b""
        if eocd_at >= _ZIP64_LOCATOR.size:
            raw.seek(eocd_at - _ZIP64_LOCATOR.size)
            locator = raw.read(_ZIP64_LOCATOR.size)
            # Locator or not, the parser decides by these bytes: the copy must hold the ones judged here.
            bounded_end_records.append((eocd_at - _ZIP64_LOCATOR.size, locator))
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
            bounded_end_records.append((record_at, record))
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
        # The directory and every end record after it are read once, here, and the parser is later given
        # these bytes (see ``_PinnedTail``): a writer that rewrites the file in place after this check can
        # change the member data, which is hashed as it is read, but not what is parsed or counted.
        if size - cd_offset > cd_size + _MAX_END_RECORD_BYTES:
            return refuse(ARCHIVE_FRAMING, "oversized end records")
        # The copy also holds the bytes where the parser looks for a zip64 locator, even when the directory
        # is shorter than a locator (a zip64 record it would then read starts at or after the directory).
        capture_start = max(0, min(cd_offset, eocd_at - _ZIP64_LOCATOR.size))
        raw.seek(capture_start)
        tail_bytes = raw.read(size - capture_start)
        directory = tail_bytes[cd_offset - capture_start : cd_offset - capture_start + cd_size]
    except OSError:
        report.add(CONTAINER_UNREADABLE, "<bundle>")
        return False
    if len(directory) != cd_size or len(tail_bytes) != size - capture_start:
        return refuse(ARCHIVE_FRAMING, "short central directory")
    # The end records were read before the copy: a writer that rewrites them in place in between would
    # leave the bounded directory slice in the copy but hand the parser end records that were never
    # bounded (a directory starting before the copy, more records than the limit).
    if any(tail_bytes[at - capture_start : at - capture_start + len(data)] != data for at, data in bounded_end_records):
        return refuse(ARCHIVE_FRAMING, "end records changed while they were read")
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
    pin = _PINNED_TAIL.get()
    if pin is not None:
        pin.append((capture_start, tail_bytes))
    return True


_MAX_END_RECORD_BYTES = 1 << 20  # the zip64 end record, its locator and the end record, with their extensions


class _PinnedTail:
    """A read-only view of an open archive whose bytes from ``start`` on are a copy taken by the precheck.

    The archive parser reads the end records and the central directory from the tail of the file; handing
    it this view means it parses exactly the bytes that were bounded and checked, however the file is
    rewritten afterwards.  Reads before ``start`` go to the file.
    """

    def __init__(self, raw: BinaryIO, start: int, tail: bytes) -> None:
        self._raw, self._start, self._tail = raw, start, tail
        self._size = start + len(tail)
        self._position = 0

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._position

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        base = {os.SEEK_SET: 0, os.SEEK_CUR: self._position, os.SEEK_END: self._size}[whence]
        self._position = max(0, base + offset)
        return self._position

    def read(self, count: int = -1) -> bytes:
        if count is None or count < 0:
            count = max(0, self._size - self._position)
        end = min(self._position + count, self._size)
        out = b""
        if self._position < self._start and end > self._position:
            stop = min(end, self._start)
            self._raw.seek(self._position)
            out = self._raw.read(stop - self._position)
            if len(out) != stop - self._position:
                self._position += len(out)
                return out
        low = max(self._position + len(out), self._start)
        if end > low:
            out += self._tail[low - self._start : end - self._start]
        self._position += len(out)
        return out

    def close(self) -> None:  # the file belongs to the caller
        pass


def _exact_path(path: "str | bytes | os.PathLike[str] | os.PathLike[bytes]") -> str:
    """The path argument as exact text, copied once with the base types' own methods.

    ``os.fspath`` asks a path-like object once (its own ``__fspath__``); the result may be a ``str`` or
    ``bytes`` subclass whose ``endswith``, ``rstrip`` or ``decode`` is caller code that could answer with
    another bundle's path, so it is copied to an exact ``str`` without running any method of the
    subclass.  Every later use is of the copy.
    """

    target = os.fspath(path)
    if issubclass(type(target), str):
        return str.__str__(target)
    return bytes.decode(target, sys.getfilesystemencoding(), sys.getfilesystemencodeerrors())  # type: ignore[arg-type]


def _root_name(target: str | bytes) -> str:
    """The bundle path without trailing directory spellings (``alias/``, ``alias//``, ``alias/.``).

    ``os.path.islink("alias/")`` and an ``O_NOFOLLOW`` open of ``"alias/."`` follow the link on Linux, so a
    symlinked root spelled with a trailing slash or dot would verify clean.  Whatever names the root's last
    real component is checked and opened instead (``/`` and ``.`` themselves are left alone).
    """

    text = target if type(target) is str else _exact_path(target)
    while True:
        if len(text) > 1 and text.endswith("/"):
            text = text.rstrip("/") or "/"
        elif text.endswith("/."):
            text = text[:-2] or "/"
        else:
            return text


def _identity(info: os.stat_result) -> tuple[int, int, int]:
    return (info.st_ino, info.st_size, info.st_mtime_ns)


def verify_bundle(path: str | os.PathLike[str]) -> VerificationReport:
    """Verify a bundle directory or zip file; never extracts or mutates it.

    Threat model for a file that another writer can modify while it is verified: the pathname is bound
    once (one open handle, no re-resolution); the archive's end records and central directory are read
    once, bounded and checked, and that copy is what the parser parses; member bytes are hashed as they
    are read, and the file's inode, size and modification time are compared before and after, so an
    in-place rewrite during verification is refused (``archive_framing``).  A writer that restores the
    same modification time cannot be ruled out by a verifier that reads a file in place: a bundle from an
    untrusted writer should be copied somewhere that writer cannot reach before it is verified.
    """

    report = VerificationReport()
    target = _exact_path(path)  # exact text, copied once: every use below is of this copy
    # The root is judged and opened by the name of its last real component: a trailing ``/`` or ``/.``
    # makes the kernel (and ``islink``) follow a link there.  An archive keeps the path as spelled, so a
    # file named with a trailing slash is still not a readable archive.
    named = _root_name(target)
    if os.path.islink(named):
        report.add(SYMLINK_ENTRY, "<bundle>")
        return report
    # A directory is opened once, without following a link, and walked and read beneath
    # that descriptor, so a root or directory swapped for a link later is never followed.
    try:
        root = os.open(named, os.O_RDONLY | os.O_DIRECTORY | _NOFOLLOW | _CLOEXEC)
    except OSError as exc:
        try:
            linked = exc.errno in (errno.ELOOP, errno.ENOTDIR) and stat.S_ISLNK(os.lstat(named).st_mode)
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
            opened_as = _identity(os.fstat(raw.fileno()))
            pinned: list[tuple[int, bytes]] = []
            token = _PINNED_TAIL.set(pinned)
            try:
                accepted = _precheck_central_directory(raw, report)
            finally:
                _PINNED_TAIL.reset(token)
            if not accepted:
                return report
            try:
                # (the precheck always leaves its copy; a stand-in precheck that does not is parsed in place)
                archive = zipfile.ZipFile(_PinnedTail(raw, pinned[0][0], pinned[0][1]) if pinned else raw)
            except _READ_ERRORS:
                report.add(CONTAINER_UNREADABLE, "<bundle>", "not a readable zip archive")
                return report
            with archive:
                source = _ZipSource(archive, raw)
                entries = _inventory_zip(source, report)
                if not any(item.code in _FATAL_CODES for item in report.findings):
                    _verify_open(entries, source, report)
            if _identity(os.fstat(raw.fileno())) != opened_as:
                report.add(ARCHIVE_FRAMING, "<bundle>", "file changed during verification")
    except _READ_ERRORS:
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
