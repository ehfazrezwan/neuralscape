"""Daily compiler — synthesizes daily log entries into structured articles.

Reads all daily log entries for a given date, groups them by project/topic/type,
calls Gemini to synthesize into structured articles, and writes them to the vault.
Idempotent: running twice on the same day updates rather than duplicates.
"""

import asyncio
import json
import re
from datetime import datetime
from typing import Optional

import structlog
from google import genai

from config import settings as core_settings
from memory_service import MemoryService

from .config import compiler_settings
from .obsidian_writer import ObsidianWriter, _slugify
from .schemas import CompileResult, CompiledArticle

logger = structlog.get_logger(__name__)


# ──────────────────────────────────────────────
# Compilation prompts
# ──────────────────────────────────────────────

SESSION_SUMMARY_PROMPT = """\
You are a knowledge compiler. Given the following daily log entries from a coding assistant session, write a concise session summary in Markdown.

Structure:
1. **Overview** — 2-3 sentence summary of the day's work
2. **Key Decisions** — Bulleted list of decisions made and their rationale
3. **Facts Learned** — New information discovered
4. **Action Items** — Outstanding tasks or follow-ups
5. **Technical Notes** — Patterns, gotchas, or technical insights worth remembering

Be concise. Use wikilinks ([[Page Name]]) where you reference topics that deserve their own page.
Omit sections that have no relevant entries.

DAILY LOG ENTRIES:
"""

PROJECT_SYNTHESIS_PROMPT = """\
You are a knowledge compiler. Given the following facts about a project, synthesize them into a structured project knowledge page in Markdown.

Structure:
1. **Overview** — What this project is and its purpose
2. **Tech Stack** — Technologies, frameworks, and tools used
3. **Architecture** — Key design decisions and patterns
4. **Conventions** — Coding conventions and standards
5. **Dependencies** — Key packages and version notes
6. **Gotchas** — Known issues, pitfalls, or warnings
7. **Recent Changes** — Notable recent work

Use wikilinks ([[Page Name]]) where relevant. Be specific and factual.
Omit sections with no relevant entries.

PROJECT: {project}

FACTS:
"""

DECISION_SYNTHESIS_PROMPT = """\
You are a knowledge compiler. Given the following decision-related entries, write a structured decision record in Markdown.

Structure:
1. **Decision** — Clear statement of what was decided
2. **Context** — Why this decision was needed
3. **Options Considered** — What alternatives were evaluated
4. **Rationale** — Why this option was chosen
5. **Consequences** — Known trade-offs or implications
6. **Related** — Links to related decisions or topics (use [[wikilinks]])

ENTRIES:
"""

RESEARCH_SYNTHESIS_PROMPT = """\
You are a knowledge compiler. Given the following research-related entries on a topic, write a structured research article in Markdown.

Structure:
1. **Summary** — Key findings in 2-3 sentences
2. **Details** — Full analysis and findings
3. **Comparisons** — If applicable, comparisons between options
4. **Conclusions** — What was concluded or recommended
5. **References** — Related pages (use [[wikilinks]])

TOPIC: {topic}

ENTRIES:
"""


def _call_gemini(prompt: str) -> str:
    """Call Gemini synchronously (for use via asyncio.to_thread)."""
    model = compiler_settings.get_llm_model(core_settings.gemini_llm_model)
    client = genai.Client(api_key=core_settings.google_api_key)
    response = client.models.generate_content(model=model, contents=prompt)
    return response.text or ""


async def _async_call_gemini(prompt: str) -> str:
    """Call Gemini without blocking the event loop."""
    return await asyncio.to_thread(_call_gemini, prompt)


def _group_entries(entries: list[dict]) -> dict[str, list[dict]]:
    """Group daily log entries by type for synthesis.

    Returns:
        Dict with keys: 'decisions', 'projects', 'research', 'general'.
        Each value is a list of entries, and 'projects' is a dict of
        project_name -> entries.
    """
    groups: dict[str, list[dict]] = {
        "decisions": [],
        "research": [],
        "general": [],
    }
    projects: dict[str, list[dict]] = {}

    for entry in entries:
        category = entry.get("category", "").lower()
        content = entry.get("content", "")

        # Check for project mentions
        project = _infer_project(content)

        if category == "decision":
            groups["decisions"].append(entry)
        elif category == "research":
            groups["research"].append(entry)
        elif project:
            projects.setdefault(project, []).append(entry)
        else:
            groups["general"].append(entry)

    return {**groups, "projects": projects}


def _infer_project(content: str) -> Optional[str]:
    """Try to infer a project name from content.

    Looks for known project patterns and explicit project mentions.
    """
    content_lower = content.lower()
    # Deployment-specific slugs come from KNOWN_PROJECT_SLUGS
    for slug in core_settings.known_projects:
        if slug in content_lower:
            return slug
    return None


def _extract_decision_slug(entries: list[dict]) -> str:
    """Generate a slug for a group of decision entries."""
    # Use the first decision's content to derive a slug
    if entries:
        content = entries[0].get("content", "decision")
        # Take first meaningful words
        words = re.sub(r"[^\w\s]", "", content).split()[:6]
        return "-".join(w.lower() for w in words)
    return "unnamed-decision"


def _extract_research_topic(entries: list[dict]) -> str:
    """Generate a topic name for a group of research entries."""
    if entries:
        content = entries[0].get("content", "research")
        words = re.sub(r"[^\w\s]", "", content).split()[:5]
        return "-".join(w.lower() for w in words)
    return "unnamed-research"


def _entries_to_text(entries: list[dict]) -> str:
    """Convert entry dicts to a text block for LLM prompts."""
    lines = []
    for e in entries:
        time = e.get("time", "")
        category = e.get("category", "")
        content = e.get("content", "")
        lines.append(f"- [{time}] ({category}) {content}")
    return "\n".join(lines)


async def compile_date(
    date: str,
    service: MemoryService,
    writer: ObsidianWriter,
    user_id: str = "ehfaz",
) -> CompileResult:
    """Compile all daily log entries for a given date into structured articles.

    This is idempotent — running twice on the same date will update existing
    articles rather than creating duplicates.

    Args:
        date: ISO date string (YYYY-MM-DD).
        service: MemoryService instance for dedup.
        writer: ObsidianWriter for vault I/O.
        user_id: User ID for post-compile dedup.

    Returns:
        CompileResult with details of what was compiled.
    """
    logger.info("Starting compilation", date=date)

    # Check if already compiled
    if writer.is_daily_log_compiled(date):
        logger.info("Daily log already compiled, re-compiling (idempotent)", date=date)

    # Read daily log entries
    entries = writer.get_daily_log_entries(date)
    if not entries:
        logger.info("No entries found for date", date=date)
        return CompileResult(date=date)

    # Group entries
    grouped = _group_entries(entries)
    articles: list[CompiledArticle] = []
    entries_compiled = 0

    # Each writer now returns the vault-root-relative path (including the
    # `_raw/` prefix) so the article paths surfaced in CompiledArticle
    # match the actual on-disk location and resolve as Obsidian wikilinks.
    # The is-new checks use `raw_file_exists` so callers don't have to
    # know about the `_raw/` prefix convention.

    # 1. Session summary (always produced if there are entries)
    all_text = _entries_to_text(entries)
    try:
        is_new = not writer.raw_file_exists(f"Sessions/{date}.md")
        summary_content = await _async_call_gemini(SESSION_SUMMARY_PROMPT + all_text)
        summary_path = writer.write_session_summary(date, summary_content)
        articles.append(
            CompiledArticle(
                path=summary_path,
                title=f"Session Summary — {date}",
                article_type="session",
                created=is_new,
            )
        )
        entries_compiled += len(entries)
    except Exception:
        logger.exception("Failed to compile session summary", date=date)

    # 2. Project pages
    projects: dict[str, list[dict]] = grouped.get("projects", {})
    for project_name, project_entries in projects.items():
        try:
            is_new = not writer.raw_file_exists(
                f"Projects/{_slugify(project_name)}/README.md"
            )
            prompt = PROJECT_SYNTHESIS_PROMPT.format(project=project_name)
            prompt += _entries_to_text(project_entries)
            project_content = await _async_call_gemini(prompt)
            project_path = writer.update_project_page(project_name, project_content)
            articles.append(
                CompiledArticle(
                    path=project_path,
                    title=project_name,
                    article_type="project",
                    created=is_new,
                )
            )
        except Exception:
            logger.exception("Failed to compile project page", project=project_name)

    # 3. Decisions
    decision_entries = grouped.get("decisions", [])
    if decision_entries:
        try:
            slug = _extract_decision_slug(decision_entries)
            is_new = not writer.raw_file_exists(f"Decisions/{_slugify(slug)}.md")
            prompt = DECISION_SYNTHESIS_PROMPT + _entries_to_text(decision_entries)
            decision_content = await _async_call_gemini(prompt)
            decision_path = writer.write_decision(slug, decision_content)
            articles.append(
                CompiledArticle(
                    path=decision_path,
                    title=slug.replace("-", " ").title(),
                    article_type="decision",
                    created=is_new,
                )
            )
        except Exception:
            logger.exception("Failed to compile decisions")

    # 4. Research
    research_entries = grouped.get("research", [])
    if research_entries:
        try:
            topic = _extract_research_topic(research_entries)
            is_new = not writer.raw_file_exists(f"Research/{_slugify(topic)}.md")
            prompt = RESEARCH_SYNTHESIS_PROMPT.format(topic=topic)
            prompt += _entries_to_text(research_entries)
            research_content = await _async_call_gemini(prompt)
            research_path = writer.write_research(topic, research_content)
            articles.append(
                CompiledArticle(
                    path=research_path,
                    title=topic.replace("-", " ").title(),
                    article_type="research",
                    created=is_new,
                )
            )
        except Exception:
            logger.exception("Failed to compile research")

    # 5. Update index with all new articles
    if articles:
        index_entries = [
            {"path": a.path, "title": a.title, "type": a.article_type}
            for a in articles
        ]
        writer.update_index(index_entries)

    # 6. Mark daily log as compiled
    writer.mark_daily_log_compiled(date)

    # 7. Append to chronological log
    writer.append_log(
        f"Compiled {date}: {len(articles)} articles from {entries_compiled} entries"
    )

    # 8. Trigger dedup
    # DEFENSIVE FIX: wrap sync dedup in asyncio.to_thread to prevent blocking
    # the event loop when compile_date is called inline (e.g., from session_end
    # fallback or when ARQ worker runs process_conversation_compile).
    dedup_triggered = False
    try:
        await asyncio.to_thread(service.dedup_memories, user_id)
        dedup_triggered = True
    except Exception:
        logger.warning("Post-compile dedup failed (non-critical)")

    result = CompileResult(
        date=date,
        articles=articles,
        entries_compiled=entries_compiled,
        dedup_triggered=dedup_triggered,
    )

    logger.info(
        "Compilation complete",
        date=date,
        articles=len(articles),
        entries_compiled=entries_compiled,
    )

    return result


async def compile_all_pending(
    service: MemoryService,
    writer: ObsidianWriter,
    user_id: str = "ehfaz",
) -> list[CompileResult]:
    """Compile all uncompiled daily logs.

    Returns:
        List of CompileResult for each date compiled.
    """
    uncompiled = writer.list_uncompiled_dates()
    if not uncompiled:
        logger.info("No uncompiled daily logs found")
        return []

    logger.info("Compiling pending daily logs", count=len(uncompiled))
    results = []
    for date in uncompiled:
        try:
            result = await compile_date(date, service, writer, user_id=user_id)
            results.append(result)
        except Exception:
            logger.exception("Failed to compile date", date=date)
    return results
