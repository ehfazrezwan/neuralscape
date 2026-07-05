"""Strategy playbook synthesizer orchestrator.

Cumulative synthesis for trading strategies — the trading analogue of the
wiki_synthesizer, grouped by **strategy** instead of ``(group_id × category)``:

Per cron run, for every ``(owner, strategy)`` the synthesizer scrolls Qdrant for
that strategy's rule memories (trading categories, tagged ``strategy:<name>``)
and incrementally merges them into one canonical playbook page via Gemini.

Non-destructive by construction:
- source memories are immutable (never mutated — only read);
- playbook pages are versioned append-only (``version_number++``);
- contradictions between lessons resolve via Graphiti's bi-temporal edge
  invalidation, not deletion.

Idempotency: if a playbook's source memory-id set is unchanged since the last
synthesis, the (expensive) LLM merge is skipped.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

from memory_service import MemoryService

from .config import StrategySynthesizerSettings, strategy_synthesizer_settings
from .graph_patcher import patch_playbook_path_by_memory_ids
from .playbook_renderer import (
    playbook_page_path,
    playbook_rel_path,
    render_page,
    split_existing_page,
    write_page,
)
from .prompts import PLAYBOOK_MERGE_PROMPT, render_memories_block

logger = logging.getLogger(__name__)

# Tag prefix that marks a memory's strategy for grouping (set at ingest time,
# e.g. tags=["strategy:naked-forex-reversal"]).
STRATEGY_TAG_PREFIX = "strategy:"


@dataclass(slots=True)
class PlaybookResult:
    strategy_name: str
    owner: str | None
    playbook_path: str
    created: bool
    source_memory_count: int
    skipped_unchanged: bool = False


@dataclass(slots=True)
class SynthesisResult:
    playbooks_created: int = 0
    playbooks_updated: int = 0
    playbooks_skipped_unchanged: int = 0
    memories_processed: int = 0
    errors: list[str] = None  # type: ignore[assignment]
    playbooks: list[PlaybookResult] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.errors is None:
            self.errors = []
        if self.playbooks is None:
            self.playbooks = []


def _trading_categories() -> list[str]:
    """The trading adapter's category set (used to scope the memory scroll)."""
    from adapters.trading.profile import TRADING_CATEGORIES

    return list(TRADING_CATEGORIES.keys())


async def synthesize_all(
    *,
    service: MemoryService,
    settings: StrategySynthesizerSettings = strategy_synthesizer_settings,
    only_strategy: str | None = None,
    dry_run: bool = False,
    force: bool = False,
) -> SynthesisResult:
    """Run one full synthesis pass over every ``(owner, strategy)`` group."""
    result = SynthesisResult()
    if not settings.enabled:
        logger.info("strategy synthesizer disabled — skipping run")
        return result

    groups = _group_by_strategy(_scroll_trading_memories(service))
    if not groups:
        logger.info("strategy synthesis skipped: no trading memories found")
        return result

    for (owner, strategy_name), memories in groups.items():
        if only_strategy and strategy_name != only_strategy:
            continue
        # Per-playbook cap, applied after grouping (the scroll is global).
        if len(memories) > settings.max_memories_per_playbook:
            logger.warning(
                "strategy %s/%s has %d rule memories — capping at %d "
                "(oldest ids beyond the cap are excluded this run)",
                owner, strategy_name, len(memories), settings.max_memories_per_playbook,
            )
            memories = memories[: settings.max_memories_per_playbook]
        try:
            page = await _synthesize_playbook(
                service=service,
                settings=settings,
                owner=owner,
                strategy_name=strategy_name,
                memories=memories,
                dry_run=dry_run,
                force=force,
            )
        except Exception as exc:
            logger.exception(
                "strategy synthesis failed for owner=%s strategy=%s", owner, strategy_name
            )
            result.errors.append(f"{owner}/{strategy_name}: {exc.__class__.__name__}")
            continue
        if page is None:
            continue
        if page.skipped_unchanged:
            result.playbooks_skipped_unchanged += 1
            continue
        if page.created:
            result.playbooks_created += 1
        else:
            result.playbooks_updated += 1
        result.memories_processed += page.source_memory_count
        result.playbooks.append(page)

    logger.info(
        "strategy synthesis complete: created=%d updated=%d unchanged=%d memories=%d errors=%d",
        result.playbooks_created,
        result.playbooks_updated,
        result.playbooks_skipped_unchanged,
        result.memories_processed,
        len(result.errors),
    )
    return result


async def _synthesize_playbook(
    *,
    service: MemoryService,
    settings: StrategySynthesizerSettings,
    owner: str | None,
    strategy_name: str,
    memories: list[dict],
    dry_run: bool,
    force: bool,
) -> PlaybookResult | None:
    """Synthesize one strategy's playbook. Returns None when there's nothing to write."""
    if not memories:
        return None
    memory_ids = [m["memory_id"] for m in memories if m.get("memory_id")]

    page_path = playbook_page_path(settings.playbook_dir, owner, strategy_name)
    rel_path = playbook_rel_path(owner, strategy_name)
    if page_path is None or rel_path is None:
        logger.warning("strategy %r has no usable slug — skipping", strategy_name)
        return None

    existing_text = _safe_read_text(page_path)
    fm, existing_body = split_existing_page(existing_text)
    prior_version = int(fm.get("version_number") or 0)

    # ── Incremental skip ── unchanged source set ⇒ the merge reproduces the page.
    if existing_text and not force:
        prior_ids = _parse_id_list(fm.get("source_memory_ids", ""))
        if prior_ids and prior_ids == set(memory_ids):
            logger.debug(
                "strategy synthesis skipped (unchanged): %s/%s (%d memories)",
                owner, strategy_name, len(memory_ids),
            )
            return PlaybookResult(
                strategy_name=strategy_name,
                owner=owner,
                playbook_path=rel_path,
                created=False,
                source_memory_count=len(memories),
                skipped_unchanged=True,
            )

    prompt = PLAYBOOK_MERGE_PROMPT.format(
        strategy_name=strategy_name,
        existing_body=existing_body or "(empty — this is the first synthesis)",
        memories_block=render_memories_block(memories),
    )
    body = await _call_gemini(prompt, settings=settings)
    if not body:
        return None

    rendered = render_page(
        strategy_name=strategy_name,
        owner=owner,
        body=body,
        source_memory_ids=memory_ids,
        version_number=prior_version + 1,
        source_count=len(memories),
        now=datetime.now(timezone.utc),
    )

    if not dry_run:
        write_page(page_path, rendered)
        try:
            await patch_playbook_path_by_memory_ids(
                service, memory_ids=memory_ids, playbook_path=rel_path
            )
        except Exception:
            logger.warning(
                "patch_playbook_path failed for %s/%s (non-fatal)",
                owner, strategy_name, exc_info=True,
            )

    return PlaybookResult(
        strategy_name=strategy_name,
        owner=owner,
        playbook_path=rel_path,
        created=not existing_text,
        source_memory_count=len(memories),
    )


# ── Helpers ────────────────────────────────────────────────────────


def _parse_id_list(raw: str) -> set[str]:
    if not raw:
        return set()
    inner = raw.strip().lstrip("[").rstrip("]")
    return {part.strip() for part in inner.split(",") if part.strip()}


def _strategy_from_tags(tags: list[str] | None) -> str | None:
    """Return the strategy name from a ``strategy:<name>`` tag, or None."""
    for tag in tags or []:
        if isinstance(tag, str) and tag.startswith(STRATEGY_TAG_PREFIX):
            name = tag[len(STRATEGY_TAG_PREFIX):].strip()
            if name:
                return name
    return None


# Global safety ceiling on the full trading-memory scroll (across ALL strategies
# and owners). Distinct from max_memories_per_playbook, which caps one playbook's
# source set after grouping. 20k rules ≈ dozens of ingested books — far above any
# realistic v1 corpus, small enough to bound a runaway cron.
_SCROLL_TOTAL_MAX = 20_000
_SCROLL_PAGE_SIZE = 500


def _scroll_trading_memories(service: MemoryService) -> list[dict]:
    """Scroll Qdrant for ALL trading rule memories (facts in trading categories).

    Paginates with the scroll cursor until exhausted (or the global
    ``_SCROLL_TOTAL_MAX`` ceiling) — a single page would silently drop every
    memory beyond it and, worse, make the per-playbook idempotency check see a
    stable-but-incomplete source set. Passages are excluded (must_not
    memory_kind='passage') — playbooks synthesize from distilled rules, not
    verbatim chunks. Errors downgrade to an empty list.
    """
    try:
        from qdrant_client.models import (
            FieldCondition,
            Filter,
            MatchAny,
            MatchValue,
        )

        from config import settings as core_settings

        # _get_memory() lazily initializes; the raw attribute is None on a
        # service that hasn't served a request yet (e.g. admin/manual runs).
        client = service._get_memory().vector_store.client
        flt = Filter(
            must=[
                FieldCondition(
                    key="metadata.category",
                    match=MatchAny(any=_trading_categories()),
                )
            ],
            must_not=[
                FieldCondition(
                    key="metadata.memory_kind", match=MatchValue(value="passage")
                )
            ],
        )
        points: list = []
        offset = None
        while len(points) < _SCROLL_TOTAL_MAX:
            page, offset = client.scroll(
                collection_name=core_settings.qdrant_collection,
                scroll_filter=flt,
                limit=min(_SCROLL_PAGE_SIZE, _SCROLL_TOTAL_MAX - len(points)),
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            points.extend(page or [])
            if offset is None or not page:
                break
        if len(points) >= _SCROLL_TOTAL_MAX:
            logger.warning(
                "strategy synth: scroll hit the %d-memory ceiling — some rules "
                "may be excluded from playbooks this run",
                _SCROLL_TOTAL_MAX,
            )
    except Exception:
        logger.warning("strategy synth: Qdrant scroll failed", exc_info=True)
        return []

    memories: list[dict] = []
    for point in points:
        payload = getattr(point, "payload", None) or {}
        metadata = payload.get("metadata", {}) or {}
        memories.append(
            {
                "memory_id": str(getattr(point, "id", "") or ""),
                "content": payload.get("data", "") or "",
                "category": metadata.get("category"),
                "tags": metadata.get("tags"),
                "owner_user_id": metadata.get("owner_user_id"),
                "confidence": metadata.get("confidence"),
                "created_at": payload.get("created_at"),
            }
        )
    return memories


def _group_by_strategy(memories: list[dict]) -> dict[tuple[str | None, str], list[dict]]:
    """Group memories by ``(owner_user_id, strategy_name)``.

    Memories without a ``strategy:`` tag can't be attributed to a playbook and
    are skipped. Results are ordered by memory_id within a group for a stable,
    idempotent source-id set.
    """
    groups: dict[tuple[str | None, str], list[dict]] = {}
    for mem in memories:
        strategy = _strategy_from_tags(mem.get("tags"))
        if not strategy:
            continue
        key = (mem.get("owner_user_id"), strategy)
        groups.setdefault(key, []).append(mem)
    for key in groups:
        groups[key].sort(key=lambda m: m.get("memory_id") or "")
    return groups


def _safe_read_text(path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""
    except Exception:
        logger.warning("failed to read existing playbook %s", path, exc_info=True)
        return ""


async def _call_gemini(prompt: str, *, settings: StrategySynthesizerSettings) -> str:
    """Run the merge prompt through Gemini with a timeout + retries."""
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
                "Gemini merge timed out after %ds (attempt %d/%d)",
                settings.gemini_timeout_seconds, attempt + 1, attempts,
            )
        except Exception as exc:
            last_exc = exc
            logger.warning(
                "Gemini merge failed (attempt %d/%d): %s",
                attempt + 1, attempts, exc.__class__.__name__,
            )
        if attempt + 1 < attempts:
            await asyncio.sleep(2 ** attempt)

    logger.error("Gemini playbook-merge exhausted %d attempt(s): %s", attempts, last_exc)
    return ""
