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


async def synthesize_all(
    *,
    service: MemoryService,
    settings: SynthesizerSettings = synthesizer_settings,
    only_category: str | None = None,
    dry_run: bool = False,
) -> SynthesisResult:
    """Run one full synthesis pass over every shared category.

    Args:
        service: The shared MemoryService instance. Used to access the
            Neo4j driver and to fetch source memory content from Qdrant.
        settings: Synthesizer settings (cadence, vault path, model).
        only_category: When set, restrict synthesis to a single
            category (handy for manual triggers).
        dry_run: When True, do everything except write to the vault and
            patch Neo4j. The returned result still reports what would
            have happened.
    """
    result = SynthesisResult()
    if not settings.enabled:
        logger.info("wiki synthesizer disabled — skipping run")
        return result

    driver = _driver_from_service(service)
    if driver is None:
        result.errors.append("no Neo4j driver available on MemoryService")
        return result

    categories = [only_category] if only_category else list(CATEGORY_VAULT_PATHS.keys())

    for category in categories:
        # v1 synthesizes the team-wide shared pool only. Per-project
        # shared groups (`shared--project--{pid}`) come in a follow-up.
        group_id = _build_group_id(MemoryVisibility.SHARED.value, "", None)
        try:
            communities = await load_communities(driver, group_id=group_id)
        except Exception as exc:
            logger.warning(
                "load_communities raised for category=%s group_id=%s: %s",
                category,
                group_id,
                exc,
            )
            result.errors.append(f"{category}: load_communities failed")
            continue

        for community in communities:
            if not community.member_memory_ids:
                result.communities_skipped_empty += 1
                continue
            try:
                page = await _synthesize_community(
                    service=service,
                    settings=settings,
                    driver=driver,
                    category=category,
                    group_id=group_id,
                    community=community,
                    dry_run=dry_run,
                )
            except Exception as exc:
                logger.exception(
                    "synthesis failed for category=%s community=%s",
                    category,
                    community.uuid,
                )
                result.errors.append(
                    f"{category}/{community.uuid}: {exc.__class__.__name__}"
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
    return result


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
        )

    return PageResult(
        category=category,
        community_id=community.uuid,
        wiki_path=rel_path,
        created=not existing_text,
        source_memory_count=len(memories),
    )


# ── Helpers ────────────────────────────────────────────────────────


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

    Reuses the conversation_compiler's Gemini helper so model selection,
    retry, and error handling stay consistent across the two LLM-driven
    extensions. The synthesizer can override the model via its own
    ``WIKI_SYNTHESIZER_GEMINI_MODEL`` env var; an empty string inherits
    the conversation_compiler's default.
    """
    # Imported lazily to avoid a hard dependency between the two
    # extensions at import time.
    from extensions.conversation_compiler.compile import _async_call_gemini

    try:
        return await _async_call_gemini(prompt)
    except Exception:
        logger.exception("Gemini incremental-merge call failed")
        return ""
