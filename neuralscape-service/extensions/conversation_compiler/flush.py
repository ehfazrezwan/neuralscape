"""Flush engine: runs one conversation turn through the main extraction path.

A turn (user message + assistant response + metadata) is extracted and stored
by ``MemoryService.extract_and_store`` -- the same prompt, the same category
set, the same speaker / ``occurred_at`` / sensitivity handling and the same
content-hash dedup that ``remember_conversation`` uses. The facts that call
stored are then fed to the Obsidian dual-write (category folders + daily log),
so the vault sees exactly what the memory store holds: one extraction per turn,
one LLM call per turn, one shape of memory.

Until this module was unified it carried a private extraction prompt with its
own type vocabulary (decision / fact / pattern / gotcha / ...) mapped onto
categories by hand, and stored facts one by one with ``store_raw``. A turn
could therefore yield two differently-shaped memories depending on whether it
arrived through /flush or through remember_conversation.
"""

import asyncio
from datetime import datetime

import structlog

from memory_service import MemoryService
from schemas import (
    MemoryVisibility,
    default_visibility_for_category,
    normalize_visibility,
    validate_occurred_at,
)

from .obsidian_writer import ObsidianWriter
from .schemas import ExtractedFact, FlushResult

logger = structlog.get_logger(__name__)


def _turn_messages(user_message: str, assistant_response: str) -> list[dict]:
    """The turn as the ``[{role, content}]`` list the main extractor takes."""
    messages = [{"role": "user", "content": user_message}]
    if assistant_response and assistant_response.strip():
        messages.append({"role": "assistant", "content": assistant_response})
    return messages


def _event_time(timestamp: str | None) -> str | None:
    """The turn's event time for ``occurred_at``, or None when it will not validate.

    A client clock a little ahead of the server (or a malformed stamp) must not
    cost the turn its facts: an unusable timestamp means "event time unknown".
    """
    if not timestamp:
        return None
    try:
        return validate_occurred_at(timestamp)
    except (ValueError, TypeError):
        logger.warning("flush timestamp is not a usable event time; omitting occurred_at", timestamp=str(timestamp)[:40])
        return None


def _is_shared(memory) -> bool:
    """Whether a stored memory may reach the shared vault.

    Uses the visibility the store stamped (which already reflects the
    sensitivity gate); a row with no usable visibility falls back to the
    category default, the same rule the memory_stored handler applies.
    """
    category = getattr(memory, "category", "") or ""
    raw = getattr(memory, "visibility", None)
    try:
        normalized = normalize_visibility(raw) if raw is not None else None
        resolved = MemoryVisibility(normalized) if normalized else default_visibility_for_category(category)
    except (ValueError, TypeError):
        resolved = default_visibility_for_category(category)
    return resolved == MemoryVisibility.SHARED


async def flush_conversation_turn(
    user_message: str,
    assistant_response: str,
    session_id: str,
    channel: str,
    timestamp: str | None,
    project_id: str | None,
    user_id: str,
    service: MemoryService,
    writer: ObsidianWriter,
) -> FlushResult:
    """Extract facts from a conversation turn, store them, mirror shared ones to the vault.

    1. ``service.extract_and_store`` runs the main extraction path on the turn
       (one LLM call), stores the facts and returns them.
    2. Shared facts (private ones never reach the vault) are written to the
       category folders and the daily log, from the same stored facts.

    Args:
        user_message: The user's message text.
        assistant_response: The assistant's response text.
        session_id: Session identifier (the main path's ``run_id``).
        channel: Channel name (e.g. 'slack', 'telegram', 'api'); logged only.
        timestamp: ISO timestamp of the turn (defaults to now).
        project_id: Optional project context.
        user_id: The user's ID.
        service: MemoryService instance used for extraction and storage.
        writer: ObsidianWriter instance for the vault mirror.

    Returns:
        FlushResult with extraction details.
    """
    ts = timestamp or datetime.now().isoformat()
    date = ts[:10]  # YYYY-MM-DD
    time_str = ts[11:16] if len(ts) > 16 else datetime.now().strftime("%H:%M")

    logger.info(
        "Flushing conversation turn",
        session_id=session_id,
        channel=channel,
        user_id=user_id,
    )

    # Step 1: the main extraction path. It is synchronous (LLM call, embedding,
    # Qdrant and graph writes), so it runs off the event loop. Operator
    # extraction guidance (per-project + per-user) is resolved inside it, as
    # for every other conversation write.
    try:
        stored = await asyncio.to_thread(
            service.extract_and_store,
            messages=_turn_messages(user_message, assistant_response),
            user_id=user_id,
            project_id=project_id,
            run_id=session_id,
            occurred_at=_event_time(timestamp),
        )
    except Exception:
        logger.exception("Conversation extraction failed")
        return FlushResult(session_id=session_id, timestamp=ts)

    if not stored:
        logger.info("No facts extracted from conversation turn", session_id=session_id)
        return FlushResult(session_id=session_id, timestamp=ts)

    facts = [
        ExtractedFact(
            category=getattr(m, "category", None) or "personal_fact",
            content=m.memory,
            project_id=getattr(m, "project_id", None) or project_id,
            tags=list(getattr(m, "tags", None) or []),
        )
        for m in stored
    ]

    # Step 2: vault dual write (category folders) -- shared facts only.
    # Private memories never reach the vault (multi-user privacy).
    category_paths: list[str] = []
    shared_facts: list[ExtractedFact] = []
    for memory, fact in zip(stored, facts):
        if not _is_shared(memory):
            continue
        shared_facts.append(fact)
        try:
            cat_path = writer.append_category_entry(
                category=fact.category,
                content=fact.content,
                project_id=fact.project_id,
                session_id=session_id,
                timestamp=ts,
            )
            category_paths.append(cat_path)
        except Exception:
            logger.exception(
                "Failed to write category entry",
                fact=fact.content[:80],
                category=fact.category,
            )

    # Step 3: append shared facts to the daily log. The entry category is the
    # memory's main-taxonomy category; compile.py groups on it.
    daily_log_path: str | None = None
    if shared_facts:
        log_entries = [
            {
                "time": time_str,
                "category": f.category,
                "content": f.content,
                "session_id": session_id,
                "user_id": user_id,
            }
            for f in shared_facts
        ]
        daily_log_path = writer.append_daily_log(date, log_entries)

    # Step 4: update category index
    try:
        writer.update_category_index()
    except Exception:
        logger.exception("Failed to update category index")

    # Step 5: chronological log line -- what GET /status reads as "last flush".
    try:
        writer.append_log(
            f"Flush {session_id}: {len(stored)} memories stored, {len(shared_facts)} to the vault"
        )
    except Exception:
        logger.exception("Failed to append flush log entry")

    logger.info(
        "Flush complete",
        session_id=session_id,
        facts_extracted=len(facts),
        memories_stored=len(stored),
        category_paths=len(category_paths),
    )

    return FlushResult(
        session_id=session_id,
        timestamp=ts,
        facts_extracted=len(facts),
        facts=facts,
        daily_log_path=daily_log_path,
        category_paths=category_paths,
        memories_stored=len(stored),
    )
