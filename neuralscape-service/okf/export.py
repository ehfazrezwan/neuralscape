"""Build a spec-conformant OKF bundle zip from the memory store.

Serves ``GET /v1/export/okf``. One concept document per live memory
(passages excluded — they are verbatim source text, not knowledge), with
the full Neuralscape envelope riding as extension keys, per-folder
``index.md`` progressive disclosure, a bundle-root version marker, and a
``log.md`` creation/update history.

Visibility is enforced **by construction**: the Qdrant filters below are
the only place memories enter the bundle, and they admit exactly

- the caller's own rows (any visibility) — the personal pool, same
  ``user_id`` payload filter the recall read path uses; and
- team rows with explicit ``metadata.visibility=shared`` — the shared
  pool, same filter as ``_search_shared_pool``.

A ``visibility="shared"`` bundle skips the personal pool entirely, so a
private memory cannot appear in a shared bundle — there is no code path
that could include one. Tombstoned rows never export.
"""

from __future__ import annotations

import io
import logging
import re
import zipfile
from collections import defaultdict
from datetime import datetime, timezone

from okf import translate

logger = logging.getLogger(__name__)

#: Ceiling on rows per pool scroll — bounds memory for a pathological export.
#: It is never silent: a pool that hits it is reported as truncated, with the
#: exported and total counts (response headers, the bundle's ``log.md``).
MAX_EXPORT_MEMORIES = 50_000

_GLOBAL_DIR = "global"
_PROJECTS_DIR = "projects"


def _kebab(text: str) -> str:
    slug = re.sub(r"[^\w\s-]", "", (text or "").lower()).strip()
    slug = re.sub(r"[\s_]+", "-", slug)
    return re.sub(r"-+", "-", slug).strip("-")


def _one_line(text: str, limit: int) -> str:
    t = re.sub(r"\s+", " ", text or "").strip()
    return t if len(t) <= limit else t[: limit - 1].rstrip() + "…"


def _title_for(content: str) -> str:
    words = re.sub(r"[#*`>\[\]|]", " ", content or "").split()
    title = " ".join(words[:9])
    return _one_line(title, 80) or "Untitled"


# ── Collection (visibility by construction) ─────────────────────────


def _scroll(client, collection: str, must: list, limit: int) -> tuple[list, dict | None]:
    """Page one pool to completion (500 per request), up to ``limit`` rows.

    Returns ``(points, truncation)``; ``truncation`` is None when the whole
    pool was read, else ``{"exported": n, "total": exact pool size}``.
    """
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    must_not = [
        # Consolidated-away rows are historical record, not live knowledge.
        FieldCondition(key="metadata.dream_tombstoned", match=MatchValue(value=True)),
        # Verbatim passage chunks are source text; the distilled facts carry
        # the knowledge (and the E2E round-trip parity target).
        FieldCondition(key="metadata.memory_kind", match=MatchValue(value="passage")),
    ]
    points: list = []
    offset = None
    while len(points) < limit:
        page, offset = client.scroll(
            collection_name=collection,
            scroll_filter=Filter(must=must, must_not=must_not),
            limit=min(500, limit - len(points)),
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        points.extend(page or [])
        if offset is None or not page:
            return points, None
    # The loop only exits here when ``limit`` rows were read and Qdrant still
    # had a next page: the pool is larger than the cap.
    total = len(points)
    try:
        total = int(
            client.count(
                collection_name=collection,
                count_filter=Filter(must=must, must_not=must_not),
                exact=True,
            ).count
        )
    except Exception:  # noqa: BLE001 - the flag matters more than the exact number
        logger.warning("OKF export: exact pool count failed; reporting at-least counts", exc_info=True)
    if total <= len(points):  # exactly at the cap with nothing more: not truncated
        return points, None
    return points, {"exported": len(points), "total": total}


def _point_to_row(point) -> dict | None:
    payload = getattr(point, "payload", None) or {}
    meta = payload.get("metadata", {}) or {}
    if isinstance(meta.get("metadata"), dict):  # mem0 double-wrap
        meta = meta["metadata"]
    content = (payload.get("data") or "").strip()
    if not content:
        return None
    return {
        "memory_id": str(getattr(point, "id", "") or ""),
        "content": content,
        "category": meta.get("category"),
        "scope": meta.get("scope"),
        "project_id": meta.get("project_id"),
        "visibility": meta.get("visibility"),
        "owner_user_id": meta.get("owner_user_id") or payload.get("user_id"),
        "tags": meta.get("tags"),
        "created_at": payload.get("created_at"),
        "updated_at": payload.get("updated_at"),
        "confidence": meta.get("confidence"),
        "epistemic_level": meta.get("epistemic_level"),
        "derived_from": meta.get("derived_from"),
        "related_memory_ids": meta.get("related_memory_ids"),
        "times_derived": meta.get("times_derived"),
        "valid_at": meta.get("valid_at"),
        "invalid_at": meta.get("invalid_at"),
        "source_ref": meta.get("source_ref"),
        "memory_kind": meta.get("memory_kind"),
        "salience": meta.get("salience"),
    }


def collect_memories(
    service,
    *,
    user_id: str,
    project_id: str | None = None,
    scope: str | None = None,
    visibility: str | None = None,
    limit: int = MAX_EXPORT_MEMORIES,
) -> list[dict]:
    """Rows only; see :func:`collect_memories_with_info` for the truncation report."""
    return collect_memories_with_info(
        service, user_id=user_id, project_id=project_id, scope=scope,
        visibility=visibility, limit=limit,
    )[0]


def collect_memories_with_info(
    service,
    *,
    user_id: str,
    project_id: str | None = None,
    scope: str | None = None,
    visibility: str | None = None,
    limit: int = MAX_EXPORT_MEMORIES,
) -> tuple[list[dict], dict]:
    """Every memory the caller's identity can read, as plain row dicts, plus a report.

    The report is ``{"truncated": bool, "pools": {name: {"exported", "total"}}}``
    where ``pools`` lists only the pools that hit ``limit``.

    ``visibility="shared"`` builds a team bundle: ONLY the shared pool is
    scrolled, so private rows are excluded by construction (the personal-
    pool query never runs). ``visibility="private"`` scrolls only the
    caller's own rows. Default (None) is the union — exactly what the
    recall read path lets this identity see.
    """
    from qdrant_client.models import FieldCondition, MatchValue

    from config import settings

    m = service._get_memory()
    client = m.vector_store.client
    collection = settings.qdrant_collection

    common: list = []
    if scope:
        common.append(FieldCondition(key="metadata.scope", match=MatchValue(value=scope)))
    if project_id:
        common.append(
            FieldCondition(key="metadata.project_id", match=MatchValue(value=project_id))
        )

    rows: dict[str, dict] = {}
    truncated_pools: dict[str, dict] = {}

    # Personal pool — the caller's own memories, any visibility (you can
    # always read what you wrote). Mirrors _search_personal_pool's filter.
    if visibility != "shared":
        must = [FieldCondition(key="user_id", match=MatchValue(value=user_id))] + common
        if visibility:  # e.g. "private": only the caller's private rows
            must.append(
                FieldCondition(key="metadata.visibility", match=MatchValue(value=visibility))
            )
        points, cut = _scroll(client, collection, must, limit)
        if cut:
            truncated_pools["personal"] = cut
        for point in points:
            row = _point_to_row(point)
            if row:
                rows[row["memory_id"]] = row

    # Shared pool — cross-writer rows with explicit visibility=shared.
    # Mirrors _search_shared_pool's filter.
    if visibility in (None, "shared"):
        must = [
            FieldCondition(key="metadata.visibility", match=MatchValue(value="shared"))
        ] + common
        points, cut = _scroll(client, collection, must, limit)
        if cut:
            truncated_pools["shared"] = cut
        for point in points:
            row = _point_to_row(point)
            if row:
                rows.setdefault(row["memory_id"], row)

    info = {"truncated": bool(truncated_pools), "pools": truncated_pools}
    return list(rows.values()), info


# ── Bundle rendering ────────────────────────────────────────────────


def _concept_path(row: dict) -> str:
    category = _kebab(row.get("category") or "domain_knowledge") or "domain-knowledge"
    if row.get("project_id"):
        folder = f"{_PROJECTS_DIR}/{_kebab(row['project_id']) or 'project'}/{category}"
    else:
        folder = f"{_GLOBAL_DIR}/{category}"
    stem = _kebab(" ".join((row.get("content") or "").split()[:6]))[:60] or "memory"
    short = (row.get("memory_id") or "")[:8] or "concept"
    return f"{folder}/{stem}-{short}.md"


def _render_concept(row: dict, paths_by_id: dict[str, str]) -> str:
    ts = row.get("updated_at") or row.get("created_at")
    tags = [t for t in (row.get("tags") or []) if t]
    if row.get("category") and row["category"] not in tags:
        tags.append(row["category"])
    frontmatter = translate.concept_frontmatter(
        category=row.get("category"),
        title=_title_for(row["content"]),
        description=_one_line(row["content"], 180),
        tags=tags or None,
        timestamp=str(ts) if ts else None,
        extensions=translate.envelope_extensions(row),
    )
    parts = [frontmatter, "", row["content"].strip(), ""]

    # Cross-links (§5): derived_from / related ids that are themselves in
    # the bundle become bundle-relative links — provenance a consuming
    # agent (or our own re-ingest walker) can traverse as graph edges.
    related_ids: list[str] = []
    for key in ("derived_from", "related_memory_ids"):
        for rid in row.get(key) or []:
            rid = str(rid)
            if rid in paths_by_id and rid != row["memory_id"] and rid not in related_ids:
                related_ids.append(rid)
    if related_ids:
        parts += ["# Related", ""]
        for rid in related_ids:
            parts.append(f"* [{rid}](/{paths_by_id[rid]})")
        parts.append("")
    return "\n".join(parts)


def _folder_index_entries(files: dict[str, str], folder: str) -> list[str]:
    entries = []
    prefix = f"{folder}/"
    for path in sorted(files):
        if not path.startswith(prefix) or "/" in path[len(prefix):]:
            continue
        name = path.rsplit("/", 1)[-1]
        if translate.is_reserved_filename(name):
            continue
        fm, _ = translate.parse_document(files[path])
        entries.append(
            translate.index_entry(
                translate.concept_title(fm) or name[:-3],
                name.replace(" ", "%20"),
                translate.concept_description(fm),
            )
        )
    return entries


def _subdir_index_entries(child_dirs: list[str], labels: dict[str, str] | None = None) -> list[str]:
    labels = labels or {}
    return [
        translate.index_entry(
            labels.get(d, d.rsplit("/", 1)[-1]),
            f"{d.rsplit('/', 1)[-1]}/{translate.INDEX_FILENAME}",
            None,
        )
        for d in sorted(child_dirs)
    ]


def build_bundle(
    memories: list[dict],
    *,
    bundle_name: str = "neuralscape",
    generated_at: datetime | None = None,
    truncation: dict | None = None,
) -> dict[str, str]:
    """Render memories into ``{relative_path: content}`` bundle files.

    ``truncation`` (the report from :func:`collect_memories_with_info`, when
    ``truncated``) is written into ``log.md`` so a cut bundle says so itself.
    """
    now = generated_at or datetime.now(timezone.utc)
    files: dict[str, str] = {}

    paths_by_id: dict[str, str] = {}
    used: set[str] = set()
    ordered = sorted(memories, key=lambda r: (r.get("created_at") or "", r["memory_id"]))
    for row in ordered:
        path = _concept_path(row)
        while path in used:  # ultra-defensive: id-suffixed stems collide ~never
            path = path[:-3] + "x.md"
        used.add(path)
        paths_by_id[row["memory_id"]] = path

    for row in ordered:
        files[paths_by_id[row["memory_id"]]] = _render_concept(row, paths_by_id)

    # ── Per-folder index files (deepest first, so parents can list them) ──
    folders: set[str] = set()
    for path in list(files):
        parts = path.split("/")[:-1]
        for i in range(1, len(parts) + 1):
            folders.add("/".join(parts[:i]))
    for folder in sorted(folders, key=lambda f: -f.count("/")):
        entries = _folder_index_entries(files, folder)
        child_dirs = sorted(
            f for f in folders if f.startswith(f"{folder}/") and "/" not in f[len(folder) + 1:]
        )
        sections: list[tuple[str, list[str]]] = []
        label = folder.rsplit("/", 1)[-1].replace("-", " ").title()
        if entries:
            sections.append((label, entries))
        if child_dirs:
            sections.append((f"{label} — Subdirectories", _subdir_index_entries(child_dirs)))
        if sections:
            files[f"{folder}/{translate.INDEX_FILENAME}"] = translate.render_index(sections)

    # ── Root index (version marker) + top-level subdirectory listing ──
    top_dirs = sorted(f for f in folders if "/" not in f)
    root_sections: list[tuple[str, list[str]]] = []
    if top_dirs:
        root_sections.append(
            (
                f"{bundle_name} knowledge bundle",
                _subdir_index_entries(
                    top_dirs,
                    labels={
                        _GLOBAL_DIR: "Global knowledge",
                        _PROJECTS_DIR: "Project knowledge",
                    },
                ),
            )
        )
    else:
        root_sections.append(
            (f"{bundle_name} knowledge bundle", [translate.index_entry("(empty)", "#")])
        )
    files[translate.INDEX_FILENAME] = translate.render_index(root_sections, is_bundle_root=True)

    # ── log.md — creation/update history, newest date first (§7) ──
    by_date: dict[str, list[str]] = defaultdict(list)
    for row in ordered:
        path = paths_by_id[row["memory_id"]]
        title = _title_for(row["content"])
        created = str(row.get("created_at") or "")[:10]
        updated = str(row.get("updated_at") or "")[:10]
        if created:
            by_date[created].append(
                translate.log_entry("Creation", f"Stored [{title}](/{path}).")
            )
        if updated and updated != created:
            by_date[updated].append(
                translate.log_entry("Update", f"Revised [{title}](/{path}).")
            )
    export_day = now.date().isoformat()
    export_text = (
        f"Bundle generated from {len(memories)} memories "
        f"([index](/{translate.INDEX_FILENAME}))."
    )
    if truncation and truncation.get("truncated"):
        parts = ", ".join(
            f"{name} pool {c['exported']} of {c['total']}"
            for name, c in sorted(truncation.get("pools", {}).items())
        )
        export_text += f" INCOMPLETE: the export cap was reached ({parts})."
    by_date[export_day].insert(0, translate.log_entry("Export", export_text))
    dated = sorted(by_date.items(), key=lambda kv: kv[0], reverse=True)
    files[translate.LOG_FILENAME] = translate.render_log(
        f"{bundle_name} Update Log", dated
    )
    return files


def zip_bundle(files: dict[str, str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(files):
            zf.writestr(path, files[path])
    return buf.getvalue()


def export_bundle(
    service,
    *,
    user_id: str,
    project_id: str | None = None,
    scope: str | None = None,
    visibility: str | None = None,
    bundle_name: str = "neuralscape",
    limit: int | None = None,
) -> tuple[bytes, dict]:
    """Collect → render → zip. Returns ``(zip_bytes, stats)``."""
    memories, info = collect_memories_with_info(
        service,
        user_id=user_id,
        project_id=project_id,
        scope=scope,
        visibility=visibility,
        limit=MAX_EXPORT_MEMORIES if limit is None else limit,
    )
    files = build_bundle(memories, bundle_name=bundle_name, truncation=info)
    stats = {
        "concepts": len(memories),
        "files": len(files),
        "visibility": visibility or "all-readable",
        "truncated": info["truncated"],
        "truncated_pools": info["pools"],
    }
    if info["truncated"]:
        logger.warning("OKF export truncated at the cap: %s", info["pools"])
    logger.info(
        "OKF export for user=%s project=%s scope=%s visibility=%s → %d concepts / %d files",
        user_id, project_id, scope, visibility, stats["concepts"], stats["files"],
    )
    return zip_bundle(files), stats
