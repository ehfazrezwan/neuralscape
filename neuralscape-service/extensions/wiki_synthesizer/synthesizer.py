"""Wiki synthesizer orchestrator (category-based).

The synthesizer is tier 3 of the memory pipeline:

1. CAPTURE     — plugin hook buffers tool observations  (existing)
2. COMPILE     — compile-observations skill stores structured memories (existing)
3. SYNTHESIZE  — *this* module turns shared memories into topical wiki pages

Per cron run, for every shared ``group_id`` and every NeuralScape category,
the synthesizer scrolls Qdrant for memories matching
``(visibility=shared, group_id, category)`` and incrementally merges them
into one wiki page via Gemini.

The grouping key is **(group_id × category)** — deterministic, derived
from the existing taxonomy. No Graphiti community detection, no LLM
clustering, no multi-hour cold-start build. Earlier versions of this
synthesizer used ``Graphiti.build_communities`` for topic discovery; the
LLM tournament summarization there was an order of magnitude too slow
to run on real data, so we replaced it with category-based bucketing.

The synthesizer touches only ``visibility=shared`` memories — private
memories never reach the vault.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from memory_service import MemoryService
from schemas import CATEGORY_VAULT_PATHS, MemoryVisibility

from .config import SynthesizerSettings, synthesizer_settings
from .graph_patcher import patch_wiki_path_by_memory_ids
from .prompts import INCREMENTAL_MERGE_PROMPT, render_memories_block
from .wiki_renderer import (
    category_filename,
    category_page_title,
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
    group_id: str
    wiki_path: str
    created: bool
    source_memory_count: int


@dataclass(slots=True)
class SynthesisResult:
    pages_created: int = 0
    pages_updated: int = 0
    memories_processed: int = 0
    pages_skipped_empty: int = 0
    errors: list[str] = None  # type: ignore[assignment]
    pages: list[PageResult] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.errors is None:
            self.errors = []
        if self.pages is None:
            self.pages = []


# In-process last-run state, surfaced by the admin `synthesize/status`
# endpoint. Mutated only by ``synthesize_all`` at the end of a pass.
# Process-local — when the API and worker are separate processes (the
# usual deploy shape) each tracks its own last run. For a single source
# of truth across processes, route status through Redis (follow-up).
@dataclass(slots=True)
class LastRunSnapshot:
    started_at: str | None = None
    finished_at: str | None = None
    duration_seconds: float = 0.0
    pages_created: int = 0
    pages_updated: int = 0
    memories_processed: int = 0
    pages_skipped_empty: int = 0
    errors: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.errors is None:
            self.errors = []


_LAST_RUN = LastRunSnapshot()


def get_last_run_snapshot() -> dict:
    """Return the most recent synthesis run state as a plain dict."""
    return {
        "started_at": _LAST_RUN.started_at,
        "finished_at": _LAST_RUN.finished_at,
        "duration_seconds": _LAST_RUN.duration_seconds,
        "pages_created": _LAST_RUN.pages_created,
        "pages_updated": _LAST_RUN.pages_updated,
        "memories_processed": _LAST_RUN.memories_processed,
        "pages_skipped_empty": _LAST_RUN.pages_skipped_empty,
        "errors": list(_LAST_RUN.errors),
    }


async def synthesize_all(
    *,
    service: MemoryService,
    settings: SynthesizerSettings = synthesizer_settings,
    only_category: str | None = None,
    dry_run: bool = False,
) -> SynthesisResult:
    """Run one full synthesis pass over every shared ``(group_id × category)``.

    Walks every group_id that starts with ``shared``: the team-wide
    ``shared`` pool plus every ``shared--project--<pid>`` pool. For each
    group, walks all 13 NeuralScape categories. Pages with zero matching
    memories are skipped.

    Args:
        service: Shared MemoryService instance. Used to access Qdrant
            (for the memory scroll) and Graphiti's bridge loop (for the
            wiki_path back-reference patch).
        settings: Synthesizer settings (cadence, vault path, model).
        only_category: When set, only synthesize pages for this category.
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

    shared_group_ids = await _list_shared_group_ids(service)
    if not shared_group_ids:
        logger.info("synthesis skipped: no shared group_ids in the graph")
        _record_last_run(result, started)
        return result

    categories = (
        [only_category] if only_category else list(CATEGORY_VAULT_PATHS.keys())
    )

    for group_id in shared_group_ids:
        for category in categories:
            if category not in CATEGORY_VAULT_PATHS:
                # Defensive: unknown category names produced by future
                # taxonomy changes should not crash the cron.
                logger.warning("unknown category %r — skipping", category)
                continue
            try:
                page = await _synthesize_category_page(
                    service=service,
                    settings=settings,
                    group_id=group_id,
                    category=category,
                    dry_run=dry_run,
                )
            except Exception as exc:
                logger.exception(
                    "synthesis failed for group_id=%s category=%s",
                    group_id,
                    category,
                )
                result.errors.append(
                    f"{group_id}/{category}: {exc.__class__.__name__}"
                )
                continue
            if page is None:
                result.pages_skipped_empty += 1
                continue
            if page.created:
                result.pages_created += 1
            else:
                result.pages_updated += 1
            result.memories_processed += page.source_memory_count
            result.pages.append(page)

    logger.info(
        "synthesis complete: created=%d updated=%d memories=%d skipped=%d errors=%d",
        result.pages_created,
        result.pages_updated,
        result.memories_processed,
        result.pages_skipped_empty,
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
    _LAST_RUN.pages_skipped_empty = result.pages_skipped_empty
    _LAST_RUN.errors = list(result.errors)


async def _synthesize_category_page(
    *,
    service: MemoryService,
    settings: SynthesizerSettings,
    group_id: str,
    category: str,
    dry_run: bool,
) -> PageResult | None:
    """Synthesize one ``(group_id, category)`` wiki page.

    Returns ``None`` when the bucket has zero matching memories (no page
    is written and the run reports it as ``pages_skipped_empty``).
    """
    memories = _scroll_memories(
        service=service,
        group_id=group_id,
        category=category,
        limit=settings.max_memories_per_page,
    )
    if not memories:
        return None

    memory_ids = [m["memory_id"] for m in memories if m.get("memory_id")]

    filename = category_filename(group_id)
    page_path = wiki_page_path(settings.wiki_dir, category, filename)
    rel_path = wikilink_path(category, filename)

    existing_text = _safe_read_text(page_path)
    fm, existing_body = split_existing_page(existing_text)
    prior_count = int(fm.get("synthesis_count") or 0)

    title = category_page_title(category, group_id)
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
        group_id=group_id,
        visibility=MemoryVisibility.SHARED.value,
        body=body,
        source_memory_ids=memory_ids,
        synthesis_count=prior_count + 1,
        source_count=len(memories),
        now=datetime.now(timezone.utc),
    )

    if not dry_run:
        write_page(page_path, rendered)
        # Best-effort back-reference: stamp ``wiki_path`` on every
        # Graphiti entity node whose ``memory_id`` is in this page.
        # Skipped silently if the bridge or driver isn't available.
        try:
            await patch_wiki_path_by_memory_ids(
                service,
                memory_ids=memory_ids,
                wiki_path=rel_path,
                group_id=group_id,
            )
        except Exception:
            logger.warning(
                "patch_wiki_path_by_memory_ids failed for %s/%s (non-fatal)",
                group_id,
                category,
                exc_info=True,
            )

    return PageResult(
        category=category,
        group_id=group_id,
        wiki_path=rel_path,
        created=not existing_text,
        source_memory_count=len(memories),
    )


# ── Helpers ────────────────────────────────────────────────────────


async def _list_shared_group_ids(service: MemoryService) -> list[str]:
    """Return every group_id that starts with ``shared``.

    Covers the team-wide ``shared`` group plus per-project shared groups
    (``shared--project--<pid>``). Private groups (``user--...``,
    ``user--...--project--...``) and the legacy ``global`` namespace are
    intentionally excluded — v1 vault content is shared-only.

    Runs on Graphiti's bridge event loop. The async driver is bound to
    that loop and any direct ``await driver.session()`` from another
    loop raises ``Future attached to a different loop``.
    """
    graphiti = getattr(service, "_graphiti", None)
    driver = getattr(graphiti, "driver", None) if graphiti else None
    if driver is None:
        return []
    cypher = """
    MATCH (n)
    WHERE n.group_id STARTS WITH 'shared'
    RETURN DISTINCT n.group_id AS gid
    ORDER BY gid
    """

    async def _inner() -> list[dict]:
        async with driver.session() as session:
            cursor = await session.run(cypher)
            return await cursor.data()

    try:
        records = await service._run_on_bridge_async(_inner(), timeout=30.0)
    except Exception:
        logger.warning("_list_shared_group_ids failed", exc_info=True)
        return []
    return [r["gid"] for r in records if r.get("gid")]


def _scroll_memories(
    *,
    service: MemoryService,
    group_id: str,
    category: str,
    limit: int,
) -> list[dict]:
    """Return shared memories for one ``(group_id, category)`` bucket.

    Scrolls Qdrant directly on the payload filter
    ``metadata.visibility=shared AND metadata.category=<cat> AND
    metadata.scope/project_id derived from group_id``. The visibility
    filter is belt-and-braces — the group_id parse already constrains
    us to shared content — but cheap insurance against a future
    metadata schema drift.

    Returns at most ``limit`` memories. Errors are logged and downgraded
    to an empty list so a single bad bucket doesn't break the cron.
    """
    visibility, project_id = _parse_shared_group_id(group_id)
    if visibility is None:
        logger.warning("unexpected group_id %r (not shared) — skipping", group_id)
        return []

    try:
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        client = service._memory.vector_store.client
        from config import settings as core_settings

        must: list = [
            FieldCondition(
                key="metadata.visibility",
                match=MatchValue(value=MemoryVisibility.SHARED.value),
            ),
            FieldCondition(
                key="metadata.category",
                match=MatchValue(value=category),
            ),
        ]
        if project_id:
            must.extend([
                FieldCondition(
                    key="metadata.scope", match=MatchValue(value="project")
                ),
                FieldCondition(
                    key="metadata.project_id",
                    match=MatchValue(value=project_id),
                ),
            ])
        else:
            # Team-wide shared (group_id == "shared"). Constrain to
            # scope=global so per-project shared content doesn't double-
            # surface on the team-wide page.
            must.append(
                FieldCondition(key="metadata.scope", match=MatchValue(value="global"))
            )

        points, _ = client.scroll(
            collection_name=core_settings.qdrant_collection,
            scroll_filter=Filter(must=must),
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
    except Exception:
        logger.warning(
            "Qdrant scroll failed for group_id=%s category=%s",
            group_id,
            category,
            exc_info=True,
        )
        return []

    memories: list[dict] = []
    for point in points or []:
        payload = getattr(point, "payload", None) or {}
        metadata = payload.get("metadata", {}) or {}
        memories.append(
            {
                "memory_id": str(getattr(point, "id", "") or ""),
                "content": payload.get("data", "") or "",
                "category": metadata.get("category", category),
                "domain": metadata.get("domain"),
                "observation_type": metadata.get("observation_type"),
                "confidence": metadata.get("confidence"),
                "created_at": payload.get("created_at"),
                "visibility": metadata.get("visibility"),
            }
        )
    return memories


def _parse_shared_group_id(group_id: str) -> tuple[str | None, str | None]:
    """Parse a shared group_id into ``(visibility, project_id)``.

    - ``"shared"`` → ``("shared", None)`` (team-wide)
    - ``"shared--project--<pid>"`` → ``("shared", "<pid>")``
    - anything else → ``(None, None)`` (caller should skip)
    """
    if group_id == "shared":
        return MemoryVisibility.SHARED.value, None
    prefix = "shared--project--"
    if group_id.startswith(prefix):
        pid = group_id[len(prefix):]
        return MemoryVisibility.SHARED.value, pid or None
    return None, None


def _safe_read_text(path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""
    except Exception:
        logger.warning("failed to read existing wiki page %s", path, exc_info=True)
        return ""


async def _call_gemini(prompt: str, *, settings: SynthesizerSettings) -> str:
    """Run the incremental-merge prompt through Gemini with a timeout + retries."""
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
            await asyncio.sleep(2 ** attempt)

    logger.error(
        "Gemini incremental-merge call exhausted %d attempt(s): %s",
        attempts,
        last_exc,
    )
    return ""
