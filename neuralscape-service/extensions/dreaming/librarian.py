"""The vault librarian — human-readable topic pages from dreamt memories.

The successor to the wiki_synthesizer's vault output, reorganized around
*subjects* instead of memory categories. Where the old layout was

    Wiki/<project>/<TypeGroup>/<CategoryLeaf>.md      (taxonomy-first)

the librarian writes

    Projects/<project>/<Topic>.md    topic pages (narrative, wikilinked)
    Projects/<project>/<project>.md  hub page: overview + topic index
    Knowledge/<Topic>.md             team-wide shared pool topics
    Me/<Topic>.md                    the operator's private pool topics
    Home.md                          map of content (hubs + recents)

Design rules:

- **A page is a subject**, discovered by an LLM clustering pass over the
  pool's live memories — "TURN & ICE Connectivity", not "Architecture".
  The category taxonomy survives as frontmatter/tags, never as paths.
- **Dense wikilinks.** Topic pages link sibling topics and their hub;
  hubs link topics and Home; Home links hubs. Obsidian's graph view and
  backlinks become the navigation surface.
- **Idempotent per topic** — a topic whose ``source_memory_ids`` set is
  unchanged since its last write is skipped (no LLM merge).
- **Pool isolation** (spec §4.2): shared pools render under Projects/ or
  Knowledge/; ONLY the operator's own private pool renders under Me/ —
  other users' private pools never land in the operator's vault.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

from .consolidate import PoolBatch
from .prompts import parse_json_response

logger = logging.getLogger(__name__)


TOPIC_CLUSTER_PROMPT = """\
You are the librarian of a personal knowledge vault. Group the memories
below into a small set of SUBJECT topics a human would browse — think
wiki page titles, not classification labels. Emit STRICT JSON only.

Rules:
- 1-8 topics. Titles are concrete noun phrases ("TURN & ICE Connectivity",
  "LoRA Restyling Workflow"), Title Case, max 5 words. NEVER use taxonomy
  words (Architecture, Conventions, Procedures, Semantic, Episodic...).
- Every topic lists the memory ids it covers (>= 2 ids; ids from input only).
- A memory may appear in at most one topic; leave uncoverable strays out.
- summary: one browsable sentence for the hub page index.

Output schema:
{{"topics": [{{"title": "...", "summary": "...", "memory_ids": ["..."]}}]}}

MEMORIES (id | category | content):
{memories_block}
"""

TOPIC_MERGE_PROMPT = """\
You are updating one page of a personal knowledge vault. Merge the
memories below into the existing page body (may be empty) as a coherent,
readable narrative a human would enjoy browsing.

Rules:
- Short intro paragraph, then `##` sections as the material demands.
- Integrate every distinct fact; drop nothing that is in the page unless
  a memory contradicts it (newer memory wins — rewrite, don't append).
- Weave in [[wikilinks]] where OTHER PAGES below are genuinely related.
- No YAML frontmatter, no page title, no memory ids, no meta-commentary.
- Concrete and specific beats exhaustive. Keep it under ~600 words.

PAGE TITLE: {title}
OTHER PAGES you may [[link]] to: {siblings}

CURRENT PAGE BODY:
---
{existing_body}
---

MEMORIES:
{memories_block}

Output only the updated page body.
"""


# ── Paths ───────────────────────────────────────────────────────────


def _slug_title(title: str) -> str:
    """Filesystem-safe page name that keeps human casing ("TURN & ICE" → "TURN and ICE")."""
    t = title.replace("&", "and")
    t = re.sub(r"[^\w\s-]", "", t).strip()
    t = re.sub(r"\s+", " ", t)
    return t or "Untitled"


def pool_dir(vault: Path, batch: PoolBatch, operator_user_id: str) -> Path | None:
    """Vault directory for a pool — or None when this pool must not render.

    Only the operator's own private pool reaches the vault; other users'
    private pools are invisible here (pool isolation, spec §4.2).
    """
    if batch.visibility == "shared":
        if batch.project_id:
            safe = re.sub(r"[^\w-]", "-", batch.project_id).strip("-") or "project"
            return vault / "Projects" / safe
        return vault / "Knowledge"
    if batch.owner_user_id == operator_user_id:
        return vault / "Me"
    return None


# ── Frontmatter helpers (same conventions as the old renderer) ──────

_FM_RE = re.compile(r"^---\n(?P<fm>.*?)\n---\n(?P<body>.*)$", re.DOTALL)


def split_page(content: str) -> tuple[dict, str]:
    if not content:
        return {}, ""
    m = _FM_RE.match(content)
    if not m:
        return {}, content
    fm: dict[str, str] = {}
    for line in m.group("fm").splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            fm[k.strip()] = v.strip()
    return fm, m.group("body").lstrip("\n")


def _parse_id_list(raw: str) -> set[str]:
    if not raw:
        return set()
    inner = raw.strip().lstrip("[").rstrip("]")
    return {p.strip() for p in inner.split(",") if p.strip()}


def render_topic_page(
    *,
    title: str,
    pool: str,
    summary: str,
    body: str,
    memory_ids: list[str],
    categories: list[str],
    hub_link: str | None,
    version: int,
    now: datetime | None = None,
) -> str:
    ts = (now or datetime.now(timezone.utc)).isoformat()
    tags = " ".join(sorted({f"#{c}" for c in categories if c}))
    lines = [
        "---",
        f"title: {title}",
        f"summary: {summary}",
        f"pool: {pool}",
        f"source_memory_ids: [{', '.join(memory_ids)}]",
        f"last_dreamt: {ts}",
        f"version: {version}",
        "---",
        "",
        f"# {title}",
        "",
        body.strip(),
        "",
    ]
    footer = []
    if hub_link:
        footer.append(f"Part of [[{hub_link}]].")
    if tags:
        footer.append(tags)
    if footer:
        lines += ["---", " ".join(footer), ""]
    return "\n".join(lines)


# ── The librarian pass ──────────────────────────────────────────────


async def update_vault(
    batch: PoolBatch,
    llm_call,
    *,
    vault: Path,
    operator_user_id: str,
    dry_run: bool,
) -> dict:
    """Cluster the pool's staged memories into topics and update pages.

    Returns ``{pages_written, pages_skipped, topics}`` for the DreamRun
    report. Uses the *staged* batch (post-consolidation view of what's
    live and current) — tombstoned rows never reach the vault.
    """
    from extensions.conversation_compiler.obsidian_writer import _atomic_write

    out = {"pages_written": 0, "pages_skipped": 0, "topics": []}
    target = pool_dir(vault, batch, operator_user_id)
    if target is None:
        return out
    live = [
        m for m in batch.memories
        if not m.get("dream_tombstoned") and (m.get("content") or "").strip()
    ]
    if len(live) < 2:
        return out

    from .prompts import render_memories_block

    raw = await llm_call(TOPIC_CLUSTER_PROMPT.format(
        memories_block=render_memories_block(live, include_strength=False)
    ))
    topics = parse_json_response(raw, key="topics")
    known = {m["memory_id"]: m for m in live}
    hub_name = target.name if target.name not in ("Knowledge", "Me") else None

    sibling_titles: list[str] = [
        _slug_title(t.get("title", "")) for t in topics if t.get("title")
    ]

    for topic in topics:
        title = (topic.get("title") or "").strip()
        ids = [i for i in (topic.get("memory_ids") or []) if i in known]
        if not title or len(ids) < 2:
            continue
        page_name = _slug_title(title)
        path = target / f"{page_name}.md"
        existing = path.read_text(encoding="utf-8") if path.exists() else ""
        fm, existing_body = split_page(existing)

        if _parse_id_list(fm.get("source_memory_ids", "")) == set(ids):
            out["pages_skipped"] += 1
            continue

        mems = [known[i] for i in ids]
        siblings = [s for s in sibling_titles if s != page_name]
        if hub_name:
            siblings.append(hub_name)
        body = await llm_call(TOPIC_MERGE_PROMPT.format(
            title=title,
            siblings=", ".join(f"[[{s}]]" for s in siblings) or "(none)",
            existing_body=existing_body or "(empty — first synthesis)",
            memories_block=render_memories_block(mems, include_strength=False),
        ))
        if not body.strip():
            continue
        rendered = render_topic_page(
            title=title,
            pool=batch.pool,
            summary=(topic.get("summary") or "").strip(),
            body=body,
            memory_ids=sorted(ids),
            categories=sorted({m.get("category") or "" for m in mems} - {""}),
            hub_link=hub_name,
            version=int(fm.get("version") or 0) + 1,
        )
        if not dry_run:
            _atomic_write(path, rendered)
        out["pages_written"] += 1
        out["topics"].append({"title": title, "page": str(path), "memories": len(ids)})

    if not dry_run and out["pages_written"]:
        _write_hub(target, hub_name, _atomic_write)
        _write_home(vault, _atomic_write)
    return out


def _list_topic_pages(directory: Path) -> list[tuple[str, str]]:
    """(title, summary) for every topic page in a pool dir, newest first."""
    pages = []
    if not directory.exists():
        return pages
    for path in sorted(directory.glob("*.md")):
        if path.stem == directory.name:  # the hub itself
            continue
        fm, _ = split_page(path.read_text(encoding="utf-8"))
        pages.append((fm.get("title") or path.stem, fm.get("summary") or ""))
    return pages


def _write_hub(target: Path, hub_name: str | None, atomic_write) -> None:
    """Regenerate the pool's hub page (Projects/<pid>/<pid>.md)."""
    if not hub_name:
        return
    entries = _list_topic_pages(target)
    lines = [
        "---",
        f"title: {hub_name}",
        f"last_dreamt: {datetime.now(timezone.utc).isoformat()}",
        "---",
        "",
        f"# {hub_name}",
        "",
        "What the memory system knows about this project, by subject:",
        "",
    ]
    for title, summary in entries:
        lines.append(f"- [[{_slug_title(title)}|{title}]] — {summary}")
    lines += ["", "Back to [[Home]].", ""]
    atomic_write(target / f"{hub_name}.md", "\n".join(lines))


def _write_home(vault: Path, atomic_write) -> None:
    """Regenerate Home.md — the vault's map of content."""
    lines = [
        "---",
        "title: Home",
        f"last_dreamt: {datetime.now(timezone.utc).isoformat()}",
        "---",
        "",
        "# Home",
        "",
        "Everything the memory system knows, dreamt into browsable pages.",
        "",
    ]
    projects = sorted((vault / "Projects").glob("*/")) if (vault / "Projects").exists() else []
    if projects:
        lines += ["## Projects", ""]
        for pdir in projects:
            count = len([p for p in pdir.glob("*.md") if p.stem != pdir.name])
            lines.append(f"- [[{pdir.name}]] ({count} topics)")
        lines.append("")
    for section, dirname in (("Knowledge", "Knowledge"), ("Me", "Me")):
        d = vault / dirname
        entries = _list_topic_pages(d)
        if entries:
            lines += [f"## {section}", ""]
            for title, summary in entries:
                lines.append(f"- [[{_slug_title(title)}|{title}]] — {summary}")
            lines.append("")
    if (vault / "Dreams").exists():
        lines += ["## Dream journal", "", "Recent sweeps live in `Dreams/`.", ""]
    atomic_write(vault / "Home.md", "\n".join(lines))
