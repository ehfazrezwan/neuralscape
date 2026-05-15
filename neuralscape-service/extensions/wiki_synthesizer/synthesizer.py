"""Wiki synthesizer orchestrator.

The synthesizer is the third tier of the memory pipeline:

1. CAPTURE     — plugin hook buffers tool observations (existing)
2. COMPILE     — compile-observations skill stores structured memories  (existing)
3. SYNTHESIZE  — *this* module turns shared memories into topical wiki  pages

Per cron run it walks every NeuralScape category's shared Graphiti
group, enumerates communities inside, then for each (category, community)
pair:

* loads the community's source memories from Qdrant via MemoryService
* incrementally merges them into the existing wiki page via Gemini
* atomic-writes the page under ``{vault}/Wiki/{category_folder}/...``
* patches every contributing Graphiti node with a ``wiki_path`` back-reference

The synthesizer touches only ``visibility=shared`` memories in v1 —
private memories never reach the vault.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from memory_service import MemoryService, _build_group_id
from schemas import CATEGORY_VAULT_PATHS, MemoryVisibility

from .community_loader import Community, load_communities
from .config import SynthesizerSettings, synthesizer_settings
from .graph_patcher import patch_wiki_path
from .prompts import INCREMENTAL_MERGE_PROMPT, render_memories_block
from .wiki_renderer import (
    community_filename,
    render_page,
    split_existing_page,
    wiki_page_path,
    wikilink_path,
    write_page,
)

logger = logging.getLogger(__name__)


# Public result shapes (consumed by the API + cron return value).
@dataclass(slots=True)
class PageResult:
    category: str
    community_id: str
    wiki_path: str
    created: bool
    source_memory_count: int


@dataclass(slots=True)
class SynthesisResult:
    pages_created: int = 0
    pages_updated: int = 0
    memories_processed: int = 0
    communities_skipped_empty: int = 0
    errors: list[str] = None  # type: ignore[assignment]
    pages: list[PageResult] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.errors is None:
            self.errors = []
        if self.pages is None:
            self.pages = []


# In-process last-run state, surfaced by the admin `synthesize/status`
# endpoint. Mutated only by ``synthesize_all`` at the end of a pass.
# This is a process-local snapshot — when the API and worker are
# separate processes (the usual deploy shape) each tracks its own last
# run. For a single source of truth across processes, route status
# through Redis (follow-up).
@dataclass(slots=True)
class LastRunSnapshot:
    started_at: str | None = None
    finished_at: str | None = None
    duration_seconds: float = 0.0
    pages_created: int = 0
    pages_updated: int = 0
    memories_processed: int = 0
    communities_skipped_empty: int = 0
    errors: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.errors is None:
            self.errors = []


_LAST_RUN = LastRunSnapshot()


def get_last_run_snapshot() -> dict:
    """Return the most recent synthesis run state as a plain dict.

    Process-local. See LastRunSnapshot docstring for cross-process
    caveats.
    """
    return {
        "started_at": _LAST_RUN.started_at,
        "finished_at": _LAST_RUN.finished_at,
        "duration_seconds": _LAST_RUN.duration_seconds,
        "pages_created": _LAST_RUN.pages_created,
        "pages_updated": _LAST_RUN.pages_updated,
        "memories_processed": _LAST_RUN.memories_processed,
        "communities_skipped_empty": _LAST_RUN.communities_skipped_empty,
        "errors": list(_LAST_RUN.errors),
    }


async def synthesize_all(
    *,
    service: MemoryService,
    settings: SynthesizerSettings = synthesizer_settings,
    only_category: str | None = None,
    dry_run: bool = False,
) -> SynthesisResult:
    """Run one full synthesis pass over every shared group_id.

    Walks every group_id that starts with ``shared``: the team-wide
    ``shared`` pool plus every ``shared--project--<pid>`` pool. The
    earlier "one outer loop per category" design was wrong — categories
    are a per-memory property, not a per-group property, so iterating
    13 times per group produced 13 duplicate page attempts per community.
    Now we iterate group_id → community → infer category from the
    community's member memories. The ``only_category`` filter still works
    but is applied after the category is inferred.

    Args:
        service: The shared MemoryService instance. Used to access the
            Neo4j driver and to fetch source memory content from Qdrant.
        settings: Synthesizer settings (cadence, vault path, model).
        only_category: When set, only synthesize pages whose inferred
            category matches. Communities that resolve to a different
            category are skipped.
        dry_run: When True, do everything except write to the vault and
            patch Neo4j. The returned result still reports what would
            have happened.
    """
    started = datetime.now(timezone.utc)
    result = SynthesisResult()
    if not settings.enabled:
        logger.info("wiki synthesizer disabled — skipping run")
        _record_last_run(result, started)
        return result

    driver = _driver_from_service(service)
    if driver is None:
        result.errors.append("no Neo4j driver available on MemoryService")
        _record_last_run(result, started)
        return result

    shared_group_ids = await _list_shared_group_ids(driver)
    if not shared_group_ids:
        logger.info("synthesis skipped: no shared group_ids in the graph")
        return result

    # Make sure every group_id has at least one Community node before we
    # try to walk it. Graphiti's incremental `update_communities=True`
    # only updates EXISTING communities, so freshly-populated groups
    # often have zero communities until we explicitly call build.
    if settings.auto_build_communities:
        await _ensure_communities_built(
            service=service, group_ids=shared_group_ids, result=result
        )

    for group_id in shared_group_ids:
        try:
            communities = await load_communities(driver, group_id=group_id)
        except Exception as exc:
            logger.warning(
                "load_communities raised for group_id=%s: %s", group_id, exc
            )
            result.errors.append(f"{group_id}: load_communities failed")
            continue

        for community in communities:
            if not community.member_memory_ids:
                result.communities_skipped_empty += 1
                continue

            inferred_category = _infer_category(service, community)
            if inferred_category is None:
                result.communities_skipped_empty += 1
                continue
            if only_category and inferred_category != only_category:
                continue

            try:
                page = await _synthesize_community(
                    service=service,
                    settings=settings,
                    driver=driver,
                    category=inferred_category,
                    group_id=group_id,
                    community=community,
                    dry_run=dry_run,
                )
            except Exception as exc:
                logger.exception(
                    "synthesis failed for group_id=%s community=%s",
                    group_id,
                    community.uuid,
                )
                result.errors.append(
                    f"{group_id}/{community.uuid}: {exc.__class__.__name__}"
                )
                continue
            if page is None:
                continue
            if page.created:
                result.pages_created += 1
            else:
                result.pages_updated += 1
            result.memories_processed += page.source_memory_count
            result.pages.append(page)

    logger.info(
        "synthesis complete: created=%d updated=%d memories=%d errors=%d",
        result.pages_created,
        result.pages_updated,
        result.memories_processed,
        len(result.errors),
    )
    _record_last_run(result, started)
    return result


def _record_last_run(result: SynthesisResult, started: datetime) -> None:
    """Snapshot the just-finished synthesis into ``_LAST_RUN``."""
    finished = datetime.now(timezone.utc)
    _LAST_RUN.started_at = started.isoformat()
    _LAST_RUN.finished_at = finished.isoformat()
    _LAST_RUN.duration_seconds = (finished - started).total_seconds()
    _LAST_RUN.pages_created = result.pages_created
    _LAST_RUN.pages_updated = result.pages_updated
    _LAST_RUN.memories_processed = result.memories_processed
    _LAST_RUN.communities_skipped_empty = result.communities_skipped_empty
    _LAST_RUN.errors = list(result.errors)


async def _synthesize_community(
    *,
    service: MemoryService,
    settings: SynthesizerSettings,
    driver: Any,
    category: str,
    group_id: str,
    community: Community,
    dry_run: bool,
) -> PageResult | None:
    """Synthesize one community's wiki page. Returns the page result, or None on no-op."""

    # Cap the per-page memory count so a runaway community doesn't try
    # to stuff thousands of memories through a single LLM call.
    memory_ids = community.member_memory_ids[: settings.max_memories_per_page]

    memories = [m for m in (_load_memory(service, mid) for mid in memory_ids) if m]
    if not memories:
        logger.debug(
            "community %s has %d member memory_ids but none resolved to memories",
            community.uuid,
            len(memory_ids),
        )
        return None

    filename = community_filename(community.uuid, community.name)
    page_path = wiki_page_path(settings.wiki_dir, category, filename)
    rel_path = wikilink_path(category, filename)

    # Load existing content (if any) and pull out the previous body so
    # we can ask Gemini for an incremental merge instead of a full rewrite.
    existing_text = _safe_read_text(page_path)
    fm, existing_body = split_existing_page(existing_text)
    prior_count = int(fm.get("synthesis_count") or 0)

    title = community.name or f"Untitled topic ({community.uuid[:8]})"
    prompt = INCREMENTAL_MERGE_PROMPT.format(
        topic_title=title,
        category=category,
        existing_body=existing_body or "(empty — this is the first synthesis)",
        memories_block=render_memories_block(memories),
    )

    body = await _call_gemini(prompt, settings=settings)
    if not body:
        return None

    rendered = render_page(
        title=title,
        category=category,
        community_id=community.uuid,
        community_name=community.name,
        group_id=group_id,
        visibility=MemoryVisibility.SHARED.value,
        body=body,
        source_memory_ids=memory_ids,
        graph_node_uuids=community.member_node_uuids,
        synthesis_count=prior_count + 1,
        source_count=len(memories),
        now=datetime.now(timezone.utc),
    )

    if not dry_run:
        write_page(page_path, rendered)
        await patch_wiki_path(
            driver,
            node_uuids=community.member_node_uuids,
            wiki_path=rel_path,
            group_id=group_id,
        )

    return PageResult(
        category=category,
        community_id=community.uuid,
        wiki_path=rel_path,
        created=not existing_text,
        source_memory_count=len(memories),
    )


# ── Helpers ────────────────────────────────────────────────────────


async def _list_shared_group_ids(driver: Any) -> list[str]:
    """Return every group_id that starts with ``shared``.

    Covers the team-wide ``shared`` group plus per-project shared
    groups (``shared--project--<pid>``). Private groups (``user--...``,
    ``user--...--project--...``) and the legacy ``global`` namespace are
    intentionally excluded — v1 vault content is shared-only.
    """
    cypher = """
    MATCH (n)
    WHERE n.group_id STARTS WITH 'shared'
    RETURN DISTINCT n.group_id AS gid
    ORDER BY gid
    """
    try:
        async with driver.session() as session:
            result = await session.run(cypher)
            records = await result.data()
        return [r["gid"] for r in records if r.get("gid")]
    except Exception:
        logger.warning("_list_shared_group_ids failed", exc_info=True)
        return []


async def _ensure_communities_built(
    *,
    service: MemoryService,
    group_ids: list[str],
    result: SynthesisResult,
) -> None:
    """Call ``Graphiti.build_communities`` for any group_id with none yet.

    Graphiti's ``update_communities=True`` flag only refreshes EXISTING
    communities on episode add — a group with zero communities stays at
    zero. We check per-group_id and trigger build for the empty ones.
    Best-effort: failures are logged but don't fail the synthesis run.
    """
    driver = service._graphiti.driver  # type: ignore[attr-defined]
    graphiti = service._graphiti
    bridge_runner = service._run_on_bridge
    if not (graphiti and bridge_runner):
        return

    cypher = """
    UNWIND $gids AS gid
    OPTIONAL MATCH (c:Community {group_id: gid})
    WITH gid, count(c) AS n
    RETURN gid, n
    """
    needs_build: list[str] = []
    try:
        async with driver.session() as session:
            cursor = await session.run(cypher, gids=group_ids)
            records = await cursor.data()
        for r in records:
            if int(r.get("n") or 0) == 0:
                needs_build.append(r["gid"])
    except Exception:
        logger.warning("could not check community counts (skipping pre-build)", exc_info=True)
        return

    if not needs_build:
        return
    logger.info(
        "synthesizer pre-build: triggering build_communities for %d empty group(s): %s",
        len(needs_build),
        needs_build,
    )
    try:
        bridge_runner(
            graphiti.build_communities(group_ids=needs_build), timeout=600.0
        )
    except Exception as exc:
        logger.warning("build_communities failed (continuing): %s", exc, exc_info=True)
        result.errors.append(f"build_communities: {exc.__class__.__name__}")


def _infer_category(service: MemoryService, community: Community) -> str | None:
    """Return the most-common NeuralScape category among the community's members.

    Walks the community's ``member_memory_ids``, looks up each memory in
    Qdrant via ``service.get_memory``, and returns the modal category.
    Returns ``None`` if no member resolved to a memory with a known
    category — the synthesizer treats that as "skip this community".
    """
    if not community.member_memory_ids:
        return None
    counts: dict[str, int] = {}
    # Cap the lookup; on huge communities we don't need to read all members
    # just to pick a category.
    for mid in community.member_memory_ids[:25]:
        mem = _load_memory(service, mid)
        if not mem:
            continue
        cat = (mem.get("category") or "").strip()
        if cat and cat in CATEGORY_VAULT_PATHS:
            counts[cat] = counts.get(cat, 0) + 1
    if not counts:
        return None
    # Most common wins; ties broken by sort order (stable).
    return max(counts.items(), key=lambda kv: kv[1])[0]


def _driver_from_service(service: MemoryService) -> Any | None:
    """Reach through MemoryService to the Neo4j driver Graphiti is using.

    Mirrors how ``memory_service.py`` itself accesses the driver
    (``self._graphiti.driver``). Returns None if Graphiti isn't wired up
    in this MemoryService instance, in which case the synthesizer logs
    and skips the run.
    """
    graphiti = getattr(service, "_graphiti", None)
    if graphiti is None:
        return None
    return getattr(graphiti, "driver", None)


def _load_memory(service: MemoryService, memory_id: str) -> dict | None:
    """Best-effort fetch of a memory's content + v2 metadata."""
    try:
        mem = service.get_memory(memory_id)
    except Exception:
        logger.warning("get_memory failed for %s (non-fatal)", memory_id, exc_info=True)
        return None
    if mem is None:
        return None
    return {
        "memory_id": getattr(mem, "id", memory_id),
        "content": getattr(mem, "memory", "") or "",
        "category": getattr(mem, "category", "") or "",
        "domain": getattr(mem, "domain", None),
        "observation_type": getattr(mem, "observation_type", None),
        "confidence": getattr(mem, "confidence", None),
        "created_at": getattr(mem, "created_at", None),
        "visibility": getattr(mem, "visibility", None),
    }


def _safe_read_text(path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""
    except Exception:
        logger.warning("failed to read existing wiki page %s", path, exc_info=True)
        return ""


async def _call_gemini(prompt: str, *, settings: SynthesizerSettings) -> str:
    """Run the incremental-merge prompt through Gemini.

    Reuses the conversation_compiler's Gemini helper so model selection
    stays consistent across the two LLM-driven extensions. Adds:

    * ``WIKI_SYNTHESIZER_GEMINI_TIMEOUT_SECONDS`` hard timeout per
      attempt (default 5 minutes) — a hung Gemini call won't stall the
      whole cron.
    * ``WIKI_SYNTHESIZER_GEMINI_MAX_RETRIES`` exponential-backoff
      retries on transient failures or timeouts (default 2).
    """
    # Imported lazily to avoid a hard dependency between the two
    # extensions at import time.
    import asyncio

    from extensions.conversation_compiler.compile import _async_call_gemini

    last_exc: Exception | None = None
    attempts = max(1, settings.gemini_max_retries + 1)
    for attempt in range(attempts):
        try:
            return await asyncio.wait_for(
                _async_call_gemini(prompt),
                timeout=settings.gemini_timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            last_exc = exc
            logger.warning(
                "Gemini call timed out after %ds (attempt %d/%d)",
                settings.gemini_timeout_seconds,
                attempt + 1,
                attempts,
            )
        except Exception as exc:
            last_exc = exc
            logger.warning(
                "Gemini call failed (attempt %d/%d): %s",
                attempt + 1,
                attempts,
                exc.__class__.__name__,
            )
        if attempt + 1 < attempts:
            await asyncio.sleep(2 ** attempt)  # 1s, 2s, 4s, ...

    logger.error(
        "Gemini incremental-merge call exhausted %d attempt(s): %s",
        attempts,
        last_exc,
    )
    return ""
