"""Checkpoint batch save (roadmap C4).

One call accepts up to 25 pre-categorized memories plus an optional
structured session note, runs a per-item content-hash dedup pre-check
(reusing ``store_raw``'s dedup key: user + md5(content) + scope +
visibility [+ project]), and hands the non-duplicates back as ONE batch
payload for ``enqueue_raw_batch`` — a single 202, a single task id, a
single tool card in MCP hosts. Writes stay async per NS convention: the
pre-check is a handful of Qdrant scrolls (milliseconds), extraction-free.

The session note is stored as a single ``task_context`` memory with
``observation_type="meeting_outcome"`` so the next session's context
injection surfaces it; ``store_raw``'s own dedup makes re-flushing the
same note idempotent.

Shared by the REST route (POST /v1/checkpoint) and the MCP ``checkpoint``
tool so their verdict/gating behavior can never drift.
"""

from __future__ import annotations

import hashlib
import logging

from config import settings
from schemas import (
    GLOBAL_CATEGORIES,
    CheckpointRequest,
    MemoryVisibility,
    default_scope_for_category,
    default_visibility_for_category,
    normalize_visibility,
)

logger = logging.getLogger(__name__)

CHECKPOINT_MAX_ITEMS = 25

# Session-note field → rendered label, in narrative order.
_NOTE_FIELDS = (
    ("request", "Request"),
    ("learned", "Learned"),
    ("completed", "Completed"),
    ("next_steps", "Next steps"),
)


def effective_storage_key(item: dict) -> tuple[str, str, str | None]:
    """Resolve the (scope, visibility, project_id) ``store_raw`` will use.

    The dedup pre-check must probe the SAME key store_raw's content-hash
    dedup uses, or the verdicts would disagree with what the worker later
    does. Mirrors store_raw: explicit scope wins; otherwise category default
    with a project override; ``standard`` visibility forces global scope
    with no project.
    """
    category = item["category"]
    project_id = item.get("project_id")
    scope = item.get("scope")
    if not scope:
        scope = default_scope_for_category(category).value
        if project_id and category not in GLOBAL_CATEGORIES:
            scope = "project"
    visibility = (
        normalize_visibility(item.get("visibility"))
        if item.get("visibility")
        else default_visibility_for_category(category).value
    )
    if visibility == MemoryVisibility.STANDARD.value:
        scope = "global"
        project_id = None
    return scope, visibility, project_id


def dedup_verdicts(service, items: list[dict]) -> list[dict]:
    """Per-item content-hash dedup verdicts (index-aligned with ``items``).

    ``{"index", "verdict": "new" | "duplicate", "existing_id"?}`` per item.
    Lookup failures degrade to "new" — store_raw re-checks the hash at
    store time, so a false "new" verdict can never create a duplicate row;
    it only costs the caller an unnecessary enqueue.
    """
    verdicts: list[dict] = []
    for idx, item in enumerate(items):
        scope, visibility, project_id = effective_storage_key(item)
        content_hash = hashlib.md5(item["content"].encode()).hexdigest()
        existing = service._find_by_content_hash(
            user_id=item["user_id"],
            content_hash=content_hash,
            scope=scope,
            project_id=project_id,
            visibility=visibility,
        )
        if existing is not None:
            verdicts.append(
                {"index": idx, "verdict": "duplicate", "existing_id": existing.id}
            )
        else:
            verdicts.append({"index": idx, "verdict": "new"})
    return verdicts


def render_session_note(note: dict) -> str:
    """Render the structured note as one readable memory body."""
    lines = ["Session note:"]
    for key, label in _NOTE_FIELDS:
        value = (note.get(key) or "").strip()
        if value:
            lines.append(f"{label}: {value}")
    return "\n".join(lines)


def session_note_item(note: dict, user_id: str, project_id: str | None) -> dict:
    """Build the batch item that persists the session note."""
    item: dict = {
        "content": render_session_note(note),
        "user_id": user_id,
        "category": "task_context",
        "scope": "project" if project_id else "global",
        "observation_type": "meeting_outcome",
        "source_type": "explicit",
        "tags": ["session_note"],
    }
    if project_id:
        item["project_id"] = project_id
    return item


def prepare_checkpoint(service, req: CheckpointRequest, user_id: str) -> dict:
    """Validate, gate, and dedup-precheck one checkpoint request.

    Synchronous (runs Qdrant scrolls) — call via a thread from async
    handlers. Returns ``{"verdicts", "to_enqueue", "duplicates",
    "session_note_included"}``.

    Raises:
        ValueError: an item claims a different user_id than the caller.
        PermissionError: a ``standard``-tier item from a non-dictator (the
            gate runs BEFORE enqueue so a rejected authoritative write
            fails fast instead of as a silent background-job failure).
    """
    items: list[dict] = []
    for idx, item in enumerate(req.memories):
        if item.user_id and item.user_id != user_id:
            raise ValueError(
                f"Item {idx}: user_id ({item.user_id!r}) does not match the caller ({user_id!r})"
            )
        d = item.model_dump(exclude_none=True)
        d["user_id"] = user_id
        # Serialize datetime so the batch survives JSON enqueue.
        if "expires_at" in d and hasattr(d["expires_at"], "isoformat"):
            d["expires_at"] = d["expires_at"].isoformat()
        # Persist the derived scope on the payload so the worker's store_raw
        # writes exactly the key the verdict below probed.
        scope, visibility, project_id = effective_storage_key(
            {**d, "scope": d.get("scope") if "scope" in item.model_fields_set else None}
        )
        d["scope"] = scope
        # Standard-tier write gate (mirrors _authorize_standard_write).
        if visibility == MemoryVisibility.STANDARD.value:
            if not settings.standards_enabled:
                raise PermissionError(
                    f"Item {idx}: the 'standard' visibility tier is disabled."
                )
            if not settings.is_dictator(user_id):
                raise PermissionError(
                    f"Item {idx}: user {user_id!r} is not authorized to write "
                    f"'standard'-tier memories."
                )
        items.append(d)

    verdicts = dedup_verdicts(service, items)
    to_enqueue = [d for d, v in zip(items, verdicts) if v["verdict"] == "new"]
    duplicates = sum(1 for v in verdicts if v["verdict"] == "duplicate")

    session_note_included = False
    if req.session_note is not None:
        to_enqueue.append(
            session_note_item(req.session_note.model_dump(), user_id, req.project_id)
        )
        session_note_included = True

    return {
        "verdicts": verdicts,
        "to_enqueue": to_enqueue,
        "duplicates": duplicates,
        "session_note_included": session_note_included,
    }
