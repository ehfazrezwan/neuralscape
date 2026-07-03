"""REM phase — reflection insights + the dream diary.

Reflection runs over the pool's post-consolidation memories and produces
NEW higher-order insights (pattern lens + failure lens). Each surviving
insight becomes a first-class memory:

- ``source_type="dream"``, ``observation_type="reflection"``
- ``related_memory_ids`` AND ``derived_from`` = the cited source memories
  (DERIVED_FROM provenance at both the vector and graph layers;
  ``derived_from`` is the first-class field ``get_reasoning_chain`` walks)
- ``epistemic_level`` self-labeled by the reflection prompt: ``deductive``
  (entailed by the cited memories) or ``inductive`` (pattern across >= 2);
  an invalid/missing label falls back to ``reflection``
- pool visibility/owner inherited (§4.2 — a private pool's insight is
  private; a shared pool's is shared)

so it is retrievable through the normal recall path — the property the
wiki_synthesizer never had. Dream-authored memories are excluded from the
next sweep's LIGHT intake (feedback-loop guard).

The diary is the human side: one markdown page per pool under
``{vault}/Dreams/``, versioned frontmatter with ``source_memory_ids`` for
the idempotent skip, sections per sweep. Per OpenAI's memory-summary-page
pattern it doubles as the user-facing "what the system knows" surface; it
is never a promotion source.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from schemas import MEMORY_CATEGORIES

from .consolidate import PoolBatch
from .prompts import REFLECTION_PROMPT, parse_json_response, render_memories_block

logger = logging.getLogger(__name__)


async def reflect(batch: PoolBatch, llm_call, *, max_insights: int) -> list[dict]:
    """Run the reflection pass; returns validated insight dicts."""
    # Reflection needs enough substrate to infer across. Rows consumed by
    # this sweep's own consolidation (marked via reconcile_batch) are not
    # substrate — an invalidated fact must not seed a fresh insight.
    candidates = [
        m for m in batch.memories
        if m.get("source_type") != "dream" and not m.get("dream_tombstoned")
    ]
    if len(candidates) < 3:
        return []
    now = datetime.now(timezone.utc)
    prompt = REFLECTION_PROMPT.format(
        today=now.date().isoformat(),
        categories=sorted(MEMORY_CATEGORIES.keys()),
        max_insights=max_insights,
        memories_block=render_memories_block(candidates, include_strength=False),
    )
    raw = await llm_call(prompt)
    insights = parse_json_response(raw, key="insights")
    known_ids = {m["memory_id"] for m in candidates}
    valid: list[dict] = []
    for ins in insights[:max_insights]:
        content = (ins.get("content") or "").strip()
        sources = [s for s in (ins.get("source_memory_ids") or []) if s in known_ids]
        if not content or len(sources) < 2:
            continue
        if ins.get("category") not in MEMORY_CATEGORIES:
            ins["category"] = "domain_knowledge"
        if ins.get("lens") not in ("pattern", "failure"):
            ins["lens"] = "pattern"
        # A1 epistemics: the prompt asks each insight to self-label deductive
        # (entailed by the cited memories) vs inductive (pattern across >= 2).
        # Anything else — missing, hallucinated value — falls back to the
        # honest catch-all "reflection".
        if ins.get("epistemic_level") not in ("deductive", "inductive"):
            ins["epistemic_level"] = "reflection"
        try:
            ins["confidence"] = max(0.0, min(1.0, float(ins.get("confidence", 0.0))))
        except (TypeError, ValueError):
            ins["confidence"] = 0.5
        ins["content"] = content
        ins["source_memory_ids"] = sources
        valid.append(ins)
    return valid


def store_insights(service, batch: PoolBatch, insights: list[dict], *, dry_run: bool) -> list[str]:
    """Store insights as first-class recallable memories. Returns new ids."""
    stored: list[str] = []
    for ins in insights:
        if dry_run:
            continue
        try:
            category = ins["category"]
            scope = "project" if batch.project_id else "global"
            # failure-lens lessons are procedures by shape
            observation_type = "reflection"
            responses = service.store_raw(
                content=ins["content"],
                user_id=batch.owner_user_id or "dreaming",
                category=category,
                scope=scope,
                project_id=batch.project_id,
                tags=["dream", ins["lens"]],
                observation_type=observation_type,
                concepts=["pattern"] if ins["lens"] == "pattern" else ["gotcha"],
                source_type="dream",
                related_memory_ids=ins["source_memory_ids"],
                derived_from=ins["source_memory_ids"],
                epistemic_level=ins.get("epistemic_level", "reflection"),
                confidence=ins["confidence"],
                visibility=batch.visibility if batch.visibility == "shared" else "private",
                add_to_graph=True,
            )
            if isinstance(responses, tuple):
                responses = responses[0]
            if responses:
                stored.append(responses[0].id)
        except Exception:
            logger.exception("failed to store dream insight")
    return stored


# ── Diary ───────────────────────────────────────────────────────────


def diary_page_path(dreams_dir: Path, pool: str) -> Path:
    from extensions.conversation_compiler.obsidian_writer import _slugify

    slug = _slugify(pool) or "pool"
    return dreams_dir / f"{slug}.md"


def render_diary_entry(
    *,
    pool: str,
    run_id: str,
    applied: list[dict],
    reported: list[dict],
    insights: list[dict],
    now: datetime | None = None,
) -> str:
    """One sweep's diary section (appended under the page header)."""
    ts = (now or datetime.now(timezone.utc)).strftime("%Y-%m-%d %H:%M UTC")
    lines = [f"## Dream {ts} ({run_id})", ""]
    if insights:
        lines.append("### Reflections")
        for ins in insights:
            cited = ", ".join(ins["source_memory_ids"][:4])
            lines.append(f"- **[{ins['lens']}]** {ins['content']} _(from {cited})_")
        lines.append("")
    if applied:
        lines.append("### Consolidated")
        for act in applied:
            ids = ", ".join(act.get("memory_ids", [])[:4])
            lines.append(f"- {act['type']}: {ids} — {act.get('reason', '')}")
        lines.append("")
    if reported:
        lines.append("### Proposed (not applied — review)")
        for act in reported:
            ids = ", ".join(act.get("memory_ids", [])[:4])
            lines.append(
                f"- {act['type']} (confidence {act.get('confidence', 0):.2f}): "
                f"{ids} — {act.get('reason', '')}"
            )
        lines.append("")
    if not (insights or applied or reported):
        lines.append("_Quiet night — nothing to consolidate._")
        lines.append("")
    return "\n".join(lines)


def write_diary(
    dreams_dir: Path,
    pool: str,
    entry: str,
    *,
    source_memory_ids: list[str],
    keep_entries: int = 20,
) -> str | None:
    """Prepend this sweep's entry to the pool's diary page (atomic write).

    Returns the vault-relative path, or None on failure. Old entries are
    trimmed beyond ``keep_entries`` — the diary is a review surface, not
    an archive (the DreamRun record in Redis is the machine log).
    """
    from extensions.conversation_compiler.obsidian_writer import _atomic_write

    path = diary_page_path(dreams_dir, pool)
    title = f"Dreams — {pool}"
    try:
        existing = path.read_text(encoding="utf-8") if path.exists() else ""
    except Exception:
        existing = ""
    # split off prior entries (everything from the first "## Dream")
    prior = ""
    if existing:
        idx = existing.find("## Dream")
        prior = existing[idx:] if idx != -1 else ""
    entries = [e for e in ("## Dream" + p for p in prior.split("## Dream")[1:]) if e.strip()]
    entries = entries[: keep_entries - 1]
    fm = "\n".join(
        [
            "---",
            f"title: {title}",
            f"pool: {pool}",
            f"source_memory_ids: [{', '.join(source_memory_ids)}]",
            f"last_dreamt: {datetime.now(timezone.utc).isoformat()}",
            "---",
            "",
            f"# {title}",
            "",
        ]
    )
    content = fm + entry + "\n" + "\n".join(entries)
    try:
        _atomic_write(path, content)
        return f"Dreams/{path.name}"
    except Exception:
        logger.warning("diary write failed for %s (non-fatal)", pool, exc_info=True)
        return None
