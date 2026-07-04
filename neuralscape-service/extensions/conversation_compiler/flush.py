"""Flush engine — extracts facts from conversation turns via Gemini.

Takes a conversation turn (user message + assistant response + metadata)
and extracts structured facts, stores them in NeuralScape, and appends
a human-readable summary to the daily log.
"""

import json
import re
from datetime import datetime
from typing import Optional

import structlog

from memory_service import MemoryService
from schemas import MemoryVisibility, default_visibility_for_category

from .config import compiler_settings
from .obsidian_writer import ObsidianWriter
from .schemas import ExtractedFact, FlushResult

logger = structlog.get_logger(__name__)


# ──────────────────────────────────────────────
# Extraction prompt
# ──────────────────────────────────────────────

CONVERSATION_EXTRACTION_PROMPT = """\
You are an intelligent memory extraction engine. Analyze the conversation below and extract distinct, actionable pieces of knowledge.

For each fact, assign one of these types:
- decision — A choice made, with rationale (e.g. "Chose PostgreSQL over MySQL because...")
- preference — User preference or style choice
- fact — Factual information learned (personal, technical, domain)
- pattern — Technical pattern, convention, or architecture choice
- project — Project-specific context (tech stack, dependencies, structure)
- gotcha — Warning, pitfall, or "watch out for" insight
- action_item — Something the user needs to do or follow up on
- research — Investigation, comparison, or exploration of a topic

Rules:
1. Each fact must be a standalone sentence — useful without the conversation.
2. Be specific. "Uses Python" is too vague. "Uses Python 3.12 with FastAPI for backend services" is good.
3. Skip greetings, acknowledgments, and transient tool operations (file reads, git commands, etc.).
4. Skip information only meaningful in the current moment ("currently running tests").
5. Deduplicate — don't extract the same insight twice with different wording.
6. For decisions, always include the rationale ("chose X because Y").
7. If a project name is identifiable, include it.

Respond with a JSON object:
{
    "facts": [
        {"type": "decision", "content": "...", "project": "project-name-or-null", "tags": ["tag1"]},
        {"type": "preference", "content": "...", "project": null, "tags": []}
    ]
}

If no meaningful facts can be extracted, return: {"facts": []}

CONVERSATION:
"""


def _build_extraction_prompt(
    user_message: str,
    assistant_response: str,
    channel: str = "api",
) -> str:
    """Build the full extraction prompt for a conversation turn."""
    return (
        CONVERSATION_EXTRACTION_PROMPT
        + f"channel: {channel}\n"
        + f"user: {user_message}\n"
        + f"assistant: {assistant_response}\n"
    )


def _parse_extraction_response(response_text: str) -> list[ExtractedFact]:
    """Parse the LLM extraction response into ExtractedFact objects."""
    text = response_text.strip()

    # Strip markdown code block wrapper
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    try:
        data = json.loads(text)
        raw_facts = data.get("facts", [])
    except (json.JSONDecodeError, AttributeError):
        logger.warning("Failed to parse extraction response as JSON", response=text[:200])
        return []

    facts = []
    for item in raw_facts:
        if not isinstance(item, dict):
            continue
        content = item.get("content", "").strip()
        if not content:
            continue
        facts.append(
            ExtractedFact(
                category=item.get("type", "fact"),
                content=content,
                project_id=item.get("project"),
                tags=item.get("tags", []),
            )
        )
    return facts


# Map our extraction types to NeuralScape memory categories
_TYPE_TO_CATEGORY = {
    "decision": "decision",
    "preference": "preference",
    "fact": "personal_fact",
    "pattern": "convention",
    "project": "architecture",
    "gotcha": "domain_knowledge",
    "action_item": "task_context",
    "research": "domain_knowledge",
}


def _map_category(extraction_type: str) -> str:
    """Map an extraction type to a NeuralScape memory category."""
    return _TYPE_TO_CATEGORY.get(extraction_type, "personal_fact")


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
    """Extract facts from a conversation turn and store them.

    1. Calls Gemini to extract structured facts from the conversation.
    2. Stores each fact in NeuralScape via MemoryService.
    3. Appends a human-readable summary to the daily log.

    Args:
        user_message: The user's message text.
        assistant_response: The assistant's response text.
        session_id: Session identifier for grouping.
        channel: Channel name (e.g. 'slack', 'telegram', 'api').
        timestamp: ISO timestamp of the turn (defaults to now).
        project_id: Optional project context.
        user_id: The user's ID.
        service: MemoryService instance for storing facts.
        writer: ObsidianWriter instance for daily log writes.

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

    # Step 1: Call Gemini for extraction. E4: operator extraction
    # instructions (per-project + per-user) ride along as the OPERATOR
    # GUIDANCE addendum — same contract guard as the main extraction prompt.
    prompt = _build_extraction_prompt(user_message, assistant_response, channel)
    try:
        from extraction_settings import resolve_instructions
        from prompts import append_operator_guidance

        guidance = resolve_instructions(user_id, project_id)
        if guidance:
            prompt = append_operator_guidance(prompt, guidance)
    except Exception:  # noqa: BLE001 — guidance is best-effort, never blocks a flush
        logger.warning("operator-guidance resolve failed (non-fatal)", exc_info=True)

    try:
        from .compile import _async_call_gemini

        response_text = await _async_call_gemini(prompt)
    except Exception:
        logger.exception("Gemini extraction failed")
        return FlushResult(session_id=session_id, timestamp=ts)

    # Step 2: Parse extracted facts
    facts = _parse_extraction_response(response_text)
    if not facts:
        logger.info("No facts extracted from conversation turn", session_id=session_id)
        return FlushResult(session_id=session_id, timestamp=ts)

    # Step 3: Store each fact in NeuralScape
    memories_stored = 0
    for fact in facts:
        category = _map_category(fact.category)
        fact_project = fact.project_id or project_id
        scope = "project" if fact_project else "global"
        try:
            service.store_raw(
                content=fact.content,
                user_id=user_id,
                category=category,
                scope=scope,
                project_id=fact_project,
                tags=fact.tags or None,
            )
            memories_stored += 1
        except Exception:
            logger.exception(
                "Failed to store extracted fact",
                fact=fact.content[:80],
                category=category,
            )

    # Step 3.5: Write to category folders (dual write) — shared facts only.
    # Private memories never reach the vault (multi-user privacy).
    category_paths: list[str] = []
    shared_facts: list[ExtractedFact] = []
    for fact in facts:
        category = _map_category(fact.category)
        if default_visibility_for_category(category) != MemoryVisibility.SHARED:
            continue
        shared_facts.append(fact)
        fact_project = fact.project_id or project_id
        try:
            cat_path = writer.append_category_entry(
                category=category,
                content=fact.content,
                project_id=fact_project,
                session_id=session_id,
                timestamp=ts,
            )
            category_paths.append(cat_path)
        except Exception:
            logger.exception(
                "Failed to write category entry",
                fact=fact.content[:80],
                category=category,
            )

    # Step 4: Append shared facts to daily log
    daily_log_path: str | None = None
    if shared_facts:
        log_entries = [
            {
                "time": time_str,
                "category": f.category,
                "content": f.content,
                "session_id": session_id,
            }
            for f in shared_facts
        ]
        daily_log_path = writer.append_daily_log(date, log_entries)

    # Step 4.5: Update category index
    try:
        writer.update_category_index()
    except Exception:
        logger.exception("Failed to update category index")

    logger.info(
        "Flush complete",
        session_id=session_id,
        facts_extracted=len(facts),
        memories_stored=memories_stored,
        category_paths=len(category_paths),
    )

    return FlushResult(
        session_id=session_id,
        timestamp=ts,
        facts_extracted=len(facts),
        facts=facts,
        daily_log_path=daily_log_path,
        category_paths=category_paths,
        memories_stored=memories_stored,
    )
