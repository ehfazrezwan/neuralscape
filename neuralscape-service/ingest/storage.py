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
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_SAFE_SEGMENT = re.compile(r"[^A-Za-z0-9._-]+")
# Artifact ids are sha256(content)[:16] — exactly 16 lowercase hex chars.
_FILE_ID_RE = re.compile(r"[0-9a-f]{16}")
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
    file_id = hashlib.sha256(data).hexdigest()[:16]
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
    # Write to a unique temp file then atomically rename into place, so a
    # concurrent same-hash upload can never expose a partially-written file to a
    # reader (the ingest worker). os.replace is atomic within a filesystem.
    tmp_path = abs_path.with_name(f".{disk_name}.{os.getpid()}.tmp")
    tmp_path.write_bytes(data)
    os.replace(tmp_path, abs_path)
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

    ``file_id`` must be the exact 16-hex content hash and is matched against the
    filename *stem* (``{file_id}`` or ``{file_id}.<ext>``) — never a prefix — so a
    short/partial id can't resolve to an unrelated artifact.
    """
    if not _FILE_ID_RE.fullmatch(file_id or ""):
        return None
    user_root = (_root(settings) / _safe(user_id, "anon")).resolve()
    if not user_root.is_dir():
        return None
    # Sort for a deterministic result: identical bytes hash to the same file_id,
    # so if the same content was stored under different category/project folders
    # every match has identical bytes — sorting just makes which path we return
    # stable rather than filesystem-order-dependent.
    for match in sorted(user_root.rglob("*")):
        if match.is_file() and (match.stem == file_id or match.name == file_id):
            return str(match), match.name
    return None


def artifact_source_ref(
    art: StoredArtifact,
    *,
    connector_type: str = "file_upload",
    extra: dict | None = None,
) -> dict:
    """Build a source_ref that references a stored artifact + a download handle."""
    # Re-fetch mechanism is the REST download endpoint carried in `url`
    # (GET /v1/ingest/artifacts/{file_id}); we deliberately omit a `retrieval`
    # MCP handle since there is no MCP tool that streams artifact bytes.
    # NOTE: the internal stored_path is intentionally NOT persisted here — it
    # would leak the {user}/{project}/{category} layout into user-visible (and
    # possibly shared) memory metadata. The worker gets stored_path from the
    # queue payload; only file_id is needed to download.
    ref = {
        "connector_id": connector_type,
        "connector_type": connector_type,
        "external_id": art.file_id,
        "parent_id": art.file_id,
        "title": art.filename,
        "url": f"/v1/ingest/artifacts/{art.file_id}",
    }
    if extra:
        ref.update(extra)
    return ref


# ── Orphan sweep ────────────────────────────────────────────────────


def sweep_orphan_artifacts(
    settings,
    reference_count,
    *,
    grace_days: int,
    dry_run: bool = True,
    now: float | None = None,
    max_delete: int = 1000,
) -> dict:
    """Remove artifacts that no memory references and that are older than the grace period.

    Deleting user data is irreversible, so every guard fails closed:

    * only files named like an artifact (``{16 hex}`` or ``{16 hex}.<ext>``)
      under ``{user}/{project}/{category}/`` are considered; temp files and
      anything else is left alone;
    * a file younger than ``grace_days`` is never touched (an upload whose
      ingest job is still queued, or being retried, has no rows yet);
    * ``reference_count(user_id, file_id)`` must return exactly 0; any error
      from it, or a non-integer answer, counts as "referenced";
    * ``dry_run`` (the default) removes nothing and only reports;
    * at most ``max_delete`` files per run.

    The owner is the first path segment (the stored ``_safe`` user id), the
    same value the upload used, so the reference check is scoped to that user.

    Returns ``{"scanned", "referenced", "too_new", "orphans", "changed", "deleted",
    "errors", "dry_run", "orphan_paths"}`` (paths relative to the root, capped
    at 200 for the report).
    """
    root = _root(settings)
    out = {
        "scanned": 0, "referenced": 0, "too_new": 0, "orphans": 0, "changed": 0,
        "deleted": 0, "errors": 0, "dry_run": dry_run, "orphan_paths": [],
    }
    if not root.is_dir():
        return out
    cutoff = (time.time() if now is None else now) - grace_days * 86400
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name.startswith("."):
            continue
        rel = path.relative_to(root)
        if len(rel.parts) != 4:  # {user}/{project}/{category}/{file}
            continue
        file_id = path.stem if _FILE_ID_RE.fullmatch(path.stem) else (
            path.name if _FILE_ID_RE.fullmatch(path.name) else None
        )
        if file_id is None:
            continue
        out["scanned"] += 1
        try:
            before = path.stat()
            if before.st_mtime > cutoff:
                out["too_new"] += 1
                continue
            refs = reference_count(rel.parts[0], file_id)
        except Exception:  # noqa: BLE001 - cannot prove it is orphaned: keep it
            logger.warning("Artifact GC: reference check failed for %s; keeping it", rel, exc_info=True)
            out["errors"] += 1
            continue
        if not isinstance(refs, int) or isinstance(refs, bool) or refs != 0:
            out["referenced"] += 1
            continue
        out["orphans"] += 1
        if len(out["orphan_paths"]) < 200:
            out["orphan_paths"].append(str(rel))
        if dry_run or out["deleted"] >= max_delete:
            continue
        try:
            # The reference check can take a while; an identical re-upload in the
            # meantime rewrites (or replaces) the same path. Re-stat right before
            # the unlink and leave the file alone if anything about it moved.
            after = path.stat()
            if (after.st_mtime_ns, after.st_size, after.st_ino) != (
                before.st_mtime_ns, before.st_size, before.st_ino
            ):
                out["changed"] += 1
                continue
            path.unlink()
            out["deleted"] += 1
            logger.info("Artifact GC deleted %s", rel)
            # Drop folders the removal emptied (never the storage root itself).
            for parent in path.parents:
                if parent == root or root not in parent.parents:
                    break
                try:
                    parent.rmdir()
                except OSError:
                    break
        except OSError:
            logger.warning("Artifact GC: could not remove %s", rel, exc_info=True)
            out["errors"] += 1
    logger.info("Artifact GC %s: %s", "dry run" if dry_run else "sweep", {k: v for k, v in out.items() if k != "orphan_paths"})
    return out
