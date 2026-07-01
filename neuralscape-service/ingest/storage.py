"""Persist uploaded files and manual context as on-disk artifacts.

Every file uploaded (or block of context pasted) is written to a mounted volume,
organized into ``{user}/{project}/{category}/`` subfolders, so each produced
memory can carry a ``source_ref`` that points back to a real, re-fetchable
artifact instead of being sourceless. The API writes the artifact at upload
time and hands the ingest worker a *relative path* (not the bytes), so large
files don't travel through Redis and both processes resolve the same file via
the shared volume + ``ingest_storage_dir``.

Layout (relative to ``ingest_storage_dir``):

    {user_id}/{project_id or _global}/{category}/{file_id}{ext}

``file_id`` is the content hash, so re-uploading identical bytes maps to the same
artifact (idempotent). Object storage (GCS/S3) is a later swap behind this same
small interface.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_SAFE_SEGMENT = re.compile(r"[^A-Za-z0-9._-]+")
_GLOBAL = "_global"


@dataclass(frozen=True)
class StoredArtifact:
    """A persisted file + how to reference it."""
    file_id: str
    rel_path: str  # relative to ingest_storage_dir; portable across API/worker
    abs_path: str
    filename: str  # original filename (for display / source_ref title)
    size: int


def _root(settings) -> Path:
    return Path(os.path.expanduser(settings.ingest_storage_dir)).resolve()


def _safe(segment: str, default: str = "unknown") -> str:
    """Sanitize a path segment (no traversal, no separators)."""
    seg = _SAFE_SEGMENT.sub("_", (segment or "").strip()).strip("._")
    return seg or default


def _safe_filename(filename: str, file_id: str) -> tuple[str, str]:
    """Return (basename_used_on_disk, original_extension)."""
    base = os.path.basename(filename or "")
    ext = os.path.splitext(base)[1].lower()
    ext = ext if re.fullmatch(r"\.[A-Za-z0-9]{1,12}", ext or "") else ""
    return f"{file_id}{ext}", ext


def store_artifact(
    data: bytes,
    filename: str,
    user_id: str,
    project_id: str | None,
    category: str,
    settings,
) -> StoredArtifact:
    """Write ``data`` to the volume under user/project/category. Returns metadata.

    Idempotent: identical bytes hash to the same ``file_id`` and overwrite the
    same path harmlessly.
    """
    file_id = hashlib.md5(data).hexdigest()[:16]
    disk_name, ext = _safe_filename(filename, file_id)
    rel = os.path.join(
        _safe(user_id, "anon"),
        _safe(project_id, _GLOBAL) if project_id else _GLOBAL,
        _safe(category, "uncategorized"),
        disk_name,
    )
    root = _root(settings)
    abs_path = (root / rel).resolve()
    # Defense-in-depth: the resolved path must stay under the storage root.
    if not str(abs_path).startswith(str(root) + os.sep):
        raise ValueError(f"Refusing to write artifact outside storage root: {rel}")
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    abs_path.write_bytes(data)
    logger.info("Stored ingest artifact %s (%d bytes) → %s", file_id, len(data), rel)
    return StoredArtifact(
        file_id=file_id,
        rel_path=rel,
        abs_path=str(abs_path),
        filename=os.path.basename(filename or disk_name),
        size=len(data),
    )


def read_artifact(rel_path: str, settings) -> bytes:
    """Read an artifact's bytes by its stored relative path (worker side)."""
    root = _root(settings)
    abs_path = (root / rel_path).resolve()
    if not str(abs_path).startswith(str(root) + os.sep):
        raise ValueError(f"Refusing to read artifact outside storage root: {rel_path}")
    return abs_path.read_bytes()


def find_artifact(file_id: str, user_id: str, settings) -> tuple[str, str] | None:
    """Locate a stored artifact by id, scoped to its owner. Returns (abs_path, filename).

    Owner-scoped by construction: only searches under the caller's ``{user_id}``
    subtree, so one user can't fetch another's artifact by guessing a hash.
    """
    safe_id = _safe(file_id)
    if not safe_id:
        return None
    user_root = (_root(settings) / _safe(user_id, "anon")).resolve()
    if not user_root.is_dir():
        return None
    for match in user_root.glob(f"**/{safe_id}*"):
        if match.is_file():
            return str(match), match.name
    return None


def artifact_source_ref(
    art: StoredArtifact,
    *,
    connector_type: str = "file_upload",
    extra: dict | None = None,
) -> dict:
    """Build a source_ref that references a stored artifact + a download handle."""
    ref = {
        "connector_id": connector_type,
        "connector_type": connector_type,
        "external_id": art.file_id,
        "parent_id": art.file_id,
        "title": art.filename,
        "url": f"/v1/ingest/artifacts/{art.file_id}",
        "stored_path": art.rel_path,
        "retrieval": {
            "mcp_server": "neuralscape",
            "tool": "fetch_artifact",
            "args": {"file_id": art.file_id},
        },
    }
    if extra:
        ref.update(extra)
    return ref
