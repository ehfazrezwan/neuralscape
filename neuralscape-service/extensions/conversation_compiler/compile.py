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
    client = genai.Client(
        api_key=core_settings.google_api_key,
        http_options=core_settings.gemini_http_options(),
    )
    response = client.models.generate_content(model=model, contents=prompt)
    return response.text or ""


async def _async_call_gemini(prompt: str) -> str:
    """Call Gemini without blocking the event loop."""
    return await asyncio.to_thread(_call_gemini, prompt)


_RESEARCH_CATEGORIES = frozenset({"research", "domain_knowledge"})


def _semantic_dedup_enabled() -> bool:
    """Whether the lossy semantic dedup phase may run.

    Same rule as the dedup cron (``worker.dedup_all_memories``): when
    ``DREAMING_ENABLED`` is on, the dream sweep's MERGE action owns near-dup
    merges, so only the lossless exact-hash phase runs, unless
    ``DEDUP_SEMANTIC_TOMBSTONE_ENABLED`` makes the phase reversible.
    """
    from memory.delete import semantic_dedup_phase_enabled

    return semantic_dedup_phase_enabled()


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

        # Daily-log entries carry the memory's main-taxonomy category (the
        # flush path stores them through the main extractor). ``decision`` is
        # a core category; ``domain_knowledge`` is where research-style
        # findings land (the old flush mapped its ``research`` and ``gotcha``
        # types there), so it feeds the research article. ``research`` is kept
        # for daily logs written before the flush path was unified.
        if category == "decision":
            groups["decisions"].append(entry)
        elif category in _RESEARCH_CATEGORIES:
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


async def _dedup_compiled_users(
    service: MemoryService,
    entries: list[dict],
    fallback_user_id: str,
    include_requester: bool = False,
) -> tuple[bool, int, int]:
    """Run post-compile dedup for each user that owns a compiled entry.

    ``include_requester`` adds ``fallback_user_id`` (the user the compile was
    requested for) even when every entry belongs to someone else. The vault is
    one shared tree and only receives *shared* facts, so a requester whose
    memories are all private owns no entry on a day other users wrote to; their
    own explicit compile must still tidy their own memories.

    The semantic phase is skipped when dreaming is enabled, exactly like the
    dedup cron. ``dedup_memories`` is synchronous, so each call runs in a
    worker thread (issue #213: running it inline blocked the API event loop).
    One user's failure never stops the others. Returns True when every
    attempted dedup succeeded, plus the sibling graph jobs enqueued / not
    enqueued from the ``graph_jobs`` the dedup runs returned.
    """
    users = {e["user_id"] for e in entries if e.get("user_id")}
    if include_requester or not users or any(not e.get("user_id") for e in entries):
        users.add(fallback_user_id)
    semantic = _semantic_dedup_enabled()
    ok = True
    enqueued = failed = 0
    for uid in sorted(users):
        try:
            result = await asyncio.to_thread(service.dedup_memories, uid, semantic=semantic)
            jobs = (result or {}).get("graph_jobs") or []
            if jobs:
                e, f = await _enqueue_sibling_jobs(jobs)
                enqueued += e
                failed += f
        except Exception:
            ok = False
            logger.warning("Post-compile dedup failed (non-critical)", user_id=uid)
    return ok, enqueued, failed


async def _enqueue_sibling_jobs(jobs: list[dict]) -> tuple[int, int]:
    """Enqueue the sibling graph re-ingests a dedup run produced.

    Returns ``(enqueued, not_enqueued)``; never raises.
    """
    from task_manager import enqueue_graph_jobs_standalone

    res = await enqueue_graph_jobs_standalone(jobs)
    return res["graph_jobs_enqueued"], len(res["graph_jobs_not_enqueued_ids"])


async def compile_date(
    date: str,
    service: MemoryService,
    writer: ObsidianWriter,
    user_id: str | None = None,
    dedup_when_empty: bool = True,
) -> CompileResult:
    """Compile all daily log entries for a given date into structured articles.

    This is idempotent — running twice on the same date will update existing
    articles rather than creating duplicates.

    Args:
        date: ISO date string (YYYY-MM-DD).
        service: MemoryService instance for dedup.
        writer: ObsidianWriter for vault I/O.
        user_id: User the compile was requested for; post-compile dedup runs
            for the owners recorded on the entries and falls back to this user
            for entries without one. Defaults to ``DEFAULT_USER_ID``.
        dedup_when_empty: Marks an explicit compile request, which is also the
            user's "tidy my memories" trigger: the post-compile dedup always
            includes ``user_id`` itself, whether the date has no entries or
            only other users' entries (the vault only receives *shared* facts,
            so a user whose memories are all private owns no entry but still
            has near-duplicates).
            ``compile_all_pending`` passes False so a backlog of empty dates
            does not scan the pool once per date.

    Returns:
        CompileResult with details of what was compiled.
    """
    user_id = user_id or core_settings.default_user_id
    logger.info("Starting compilation", date=date)

    # Check if already compiled
    if writer.is_daily_log_compiled(date):
        logger.info("Daily log already compiled, re-compiling (idempotent)", date=date)

    # Read daily log entries
    entries = writer.get_daily_log_entries(date)
    if not entries:
        logger.info("No entries found for date", date=date)
        if not dedup_when_empty:
            return CompileResult(date=date)
        dedup_ok, sibling_enqueued, sibling_failed = await _dedup_compiled_users(service, [], user_id)
        return CompileResult(
            date=date,
            dedup_triggered=dedup_ok,
            dedup_graph_jobs_enqueued=sibling_enqueued,
            dedup_graph_jobs_not_enqueued=sibling_failed,
        )

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

    # 8. Trigger dedup for every user whose entries were compiled.
    # The vault is one shared tree, so one daily log can hold entries from
    # several users; each entry records its owner. Entries written before
    # owners were recorded fall back to the ``user_id`` the compile was
    # requested for. Sibling graph jobs the dedup returns are enqueued.
    dedup_triggered, sibling_enqueued, sibling_failed = await _dedup_compiled_users(
        service, entries, user_id, include_requester=dedup_when_empty
    )

    result = CompileResult(
        date=date,
        articles=articles,
        entries_compiled=entries_compiled,
        dedup_triggered=dedup_triggered,
        dedup_graph_jobs_enqueued=sibling_enqueued,
        dedup_graph_jobs_not_enqueued=sibling_failed,
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
    user_id: str | None = None,
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
            result = await compile_date(
                date, service, writer, user_id=user_id, dedup_when_empty=False
            )
            results.append(result)
        except Exception:
            logger.exception("Failed to compile date", date=date)
    return results
