"""Expand an uploaded archive into its member files, safely.

A user can upload a ``.zip`` (e.g. a zipped folder) instead of individual
files. :func:`iter_archive` walks the archive with the stdlib ``zipfile`` and
yields ``(name, data)`` for each real file member, skipping directory entries,
macOS resource forks (``__MACOSX/``), dotfiles, and nested archives.

Guards bound the blast radius of a malicious archive (a "zip bomb" that expands
to gigabytes): each member is capped, the total uncompressed size is capped, and
the member count is capped. Exceeding a cap raises :class:`ArchiveTooLarge`
rather than silently truncating, so the caller can surface an honest error.
"""

from __future__ import annotations

import io
import logging
import os
import zipfile
from collections.abc import Iterator

logger = logging.getLogger(__name__)

# Extensions we treat as nested archives and skip (we don't recurse — a zip of
# zips is almost always a mistake, and recursing is where bombs hide).
_NESTED_ARCHIVE_EXTS = {".zip", ".gz", ".tar", ".tgz", ".bz2", ".xz", ".7z", ".rar"}


class ArchiveError(Exception):
    """Base error for archive handling."""


class ArchiveTooLarge(ArchiveError):
    """Raised when an archive exceeds a configured size/count cap."""


def is_zip(data: bytes) -> bool:
    """Cheap magic-byte check for a ZIP archive (PK\\x03\\x04 / empty / spanned)."""
    return data[:2] == b"PK"


def _skip_member(name: str) -> bool:
    """True for entries we never ingest: dirs, macOS forks, dotfiles, nested archives."""
    if name.endswith("/"):
        return True
    base = os.path.basename(name)
    if not base or base.startswith("."):
        return True
    if name.startswith("__MACOSX/") or "/__MACOSX/" in name:
        return True
    _, ext = os.path.splitext(base)
    if ext.lower() in _NESTED_ARCHIVE_EXTS:
        logger.info("Skipping nested archive in upload: %s", name)
        return True
    return False


def iter_archive(
    data: bytes,
    *,
    max_file_bytes: int,
    max_files: int,
    max_total_uncompressed_bytes: int,
) -> Iterator[tuple[str, bytes]]:
    """Yield ``(member_name, member_bytes)`` for each real file in a zip.

    Args:
        data: The raw ``.zip`` bytes.
        max_file_bytes: Reject any single member larger than this (uncompressed).
        max_files: Stop and raise after this many yielded members.
        max_total_uncompressed_bytes: Reject once cumulative uncompressed size
            crosses this ceiling.

    Raises:
        ArchiveError: If ``data`` is not a valid zip.
        ArchiveTooLarge: If any cap is exceeded.
    """
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as e:
        raise ArchiveError(f"Not a valid zip archive: {e}") from e

    with zf:
        yielded = 0
        total = 0
        for info in zf.infolist():
            name = info.filename
            if _skip_member(name):
                continue
            # Trust the declared uncompressed size for an up-front bomb check,
            # then verify against the bytes actually read.
            if info.file_size > max_file_bytes:
                raise ArchiveTooLarge(
                    f"Archive member '{name}' is {info.file_size} bytes, "
                    f"over the {max_file_bytes}-byte per-file limit"
                )
            total += info.file_size
            if total > max_total_uncompressed_bytes:
                raise ArchiveTooLarge(
                    f"Archive uncompressed size exceeds the "
                    f"{max_total_uncompressed_bytes}-byte limit"
                )
            if yielded >= max_files:
                raise ArchiveTooLarge(
                    f"Archive contains more than {max_files} ingestible files"
                )
            member = zf.read(info)
            # Defensive: a lying header could under-report file_size.
            if len(member) > max_file_bytes:
                raise ArchiveTooLarge(
                    f"Archive member '{name}' expanded past the per-file limit"
                )
            yielded += 1
            yield name, member
