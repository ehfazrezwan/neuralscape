"""The vault librarian — human-readable topic pages from dreamt memories.

The successor to the wiki_synthesizer's vault output, reorganized around
*subjects* instead of memory categories. Where the old layout was

    Wiki/<project>/<TypeGroup>/<CategoryLeaf>.md      (taxonomy-first)

the librarian writes

    Projects/<project>/<Topic>.md    topic pages (fixed skeleton, wikilinked)
    Projects/<project>/<project>.md  hub page: overview + topic index
    Knowledge/<Topic>.md             team-wide shared pool topics
    Me/<Topic>.md                    the operator's private pool topics
    Home.md                          L0 identity + L1 Essential Story + MOC

Design rules:

- **A page is a subject**, discovered by an LLM clustering pass over the
  pool's live memories — "TURN & ICE Connectivity", not "Architecture".
  The category taxonomy survives as frontmatter/tags, never as paths.
- **Fixed page skeleton** (MemPalace "halls"): every topic page opens with
  a compact index-card table (`| What | Entities | Source |`), then the
  same five sections in the same order — Decisions & Facts / Events /
  Discoveries / Preferences / Advice. Empty sections are omitted, never
  rendered as placeholders. Predictability is the feature.
- **Home.md is a wake-up stack**: an L0 identity block (who the operator
  is, from their highest-salience personal facts/preferences), an L1
  "Essential Story" (budget-bounded ranked digest of the top memories by
  promotion score, each line wikilinked to its topic page), then the map
  of content with counts. Per-pool top-scorer lines persist in Redis
  (``dreaming:essential:{pool}``) so Home can rank across pools without
  re-scrolling them.
- **Dense wikilinks.** Topic pages link sibling topics and their hub;
  hubs link topics and Home; Home links hubs. Obsidian's graph view and
  backlinks become the navigation surface.
- **Idempotent per topic** — a topic whose ``source_memory_ids`` set is
  unchanged since its last write is skipped (no LLM merge).
- **Dim, don't delete** (B3b): memories whose retention strength fell
  below the prune threshold but survived consolidation drop out of the
  main sections and re-render as one-liners inside a collapsed
  ``> [!note]- Faded`` callout at the bottom of their topic page — they
  fade from view, never from the page.
- **Pool isolation** (spec §4.2): shared pools render under Projects/ or
  Knowledge/; ONLY the operator's own private pool renders under Me/ —
  other users' private pools never land in the operator's vault.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

from okf import translate as okf_translate

from .consolidate import PoolBatch
from .prompts import parse_json_object, parse_json_response

logger = logging.getLogger(__name__)


# ── Fixed skeleton (B2) ─────────────────────────────────────────────

#: The five halls, in the fixed order every topic page renders them.
SECTION_ORDER = ("Decisions & Facts", "Events", "Discoveries", "Preferences", "Advice")

#: How the 13 core categories map onto the five sections. Categories a
#: knowledge adapter registers later fall back to Discoveries.
CATEGORY_SECTION = {
    "decision": "Decisions & Facts",
    "architecture": "Decisions & Facts",
    "convention": "Decisions & Facts",
    "tech_stack": "Decisions & Facts",
    "dependency": "Decisions & Facts",
    "interaction": "Events",
    "task_context": "Events",
    "domain_knowledge": "Discoveries",
    "preference": "Preferences",
    "personal_fact": "Preferences",
    "procedure": "Advice",
    "workflow": "Advice",
    "technical_skill": "Advice",
}
_DEFAULT_SECTION = "Discoveries"
_INDEX_CARD_MAX_ROWS = 8

# ── Home.md budgets (B1) ────────────────────────────────────────────

HOME_STORY_BUDGET = 3200      # hard char ceiling on the Essential Story block
HOME_STORY_TOP_N = 15         # ~top memories by promotion score across pools
IDENTITY_MAX_LINES = 6

_ID_START = "<!-- ns:identity -->"
_ID_END = "<!-- /ns:identity -->"

_ESSENTIAL_KEY = "dreaming:essential:{pool}"
_ESSENTIAL_TTL = 60 * 60 * 24 * 45  # stale pools self-clean out of the story

_IDENTITY_CATEGORIES = ("personal_fact", "preference")

# ── Faded section (B3b) ─────────────────────────────────────────────

#: Markers bracketing the collapsed Faded callout so other passes (the
#: bridges patcher) can locate it without parsing markdown structure.
FADED_START = "<!-- ns:faded -->"
FADED_END = "<!-- /ns:faded -->"
_FADED_CALLOUT_TITLE = "> [!note]- Faded"


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
You are updating one page of a personal knowledge vault. Every page uses
the SAME fixed skeleton; distill the memories below (merged with the
existing page content, if any) into it. Emit STRICT JSON only — no prose,
no markdown fences.

The skeleton:
1. "index_card": 3-8 pointer rows a reader scans first. Each row:
   "what" (one clause, <= 12 words), "entities" (proper nouns / named
   things in the row, 0-4), "source" (the section holding the detail —
   exactly one of the five section names).
2. "sections": EXACTLY these five keys, in any order (rendering is fixed):
   "Decisions & Facts", "Events", "Discoveries", "Preferences", "Advice".
   Values are markdown (bullets or short paragraphs); use "" when nothing
   fits — empty sections are omitted at render time, never invent filler.

Where material goes:
- decisions, architecture, conventions, stack/dependency facts → "Decisions & Facts"
- things that happened (interactions, task context, episodes) → "Events"
- domain knowledge, learnings, findings                       → "Discoveries"
- the user's preferences and personal facts                   → "Preferences"
- procedures, workflows, how-tos, skills                      → "Advice"

Rules:
- Integrate every distinct fact; keep everything from the existing page
  unless a memory contradicts it (newer memory wins — rewrite, don't append).
- If the existing page uses an older structure, RESTRUCTURE it into this
  skeleton — do not preserve old headings.
- Weave [[wikilinks]] into section text where OTHER PAGES are genuinely
  related. Concrete and specific beats exhaustive; whole page under ~600 words.
- No YAML frontmatter, no page title, no memory ids, no meta-commentary.

PAGE TITLE: {title}
OTHER PAGES you may [[link]] to: {siblings}

CURRENT PAGE BODY:
---
{existing_body}
---

MEMORIES (id | category | content):
{memories_block}

Output schema:
{{"index_card": [{{"what": "...", "entities": ["..."], "source": "Decisions & Facts"}}],
 "sections": {{"Decisions & Facts": "...", "Events": "...", "Discoveries": "...",
   "Preferences": "...", "Advice": "..."}}}}
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
            v = v.strip()
            # The OKF-conformant renderer YAML-quotes values that need it
            # (timestamps, titles with colons); unwrap so line-oriented
            # readers keep seeing the raw value.
            if len(v) >= 2 and v[0] == v[-1] and v[0] in ("'", '"'):
                inner = v[1:-1]
                v = inner.replace("''", "'") if v[0] == "'" else inner
            fm[k.strip()] = v
    return fm, m.group("body").lstrip("\n")


def _parse_id_list(raw: str) -> set[str]:
    if not raw:
        return set()
    inner = raw.strip().lstrip("[").rstrip("]")
    return {p.strip() for p in inner.split(",") if p.strip()}


def _one_line(text: str, limit: int = 180) -> str:
    """Collapse to a single trimmed line, ellipsized at ``limit`` chars."""
    t = re.sub(r"\s+", " ", text or "").strip()
    return t if len(t) <= limit else t[: limit - 1].rstrip() + "…"


#: Render-schema salt folded into every content fingerprint. Bumping it
#: invalidates all stored ``content_hash`` values at once, forcing every
#: topic page to regenerate exactly ONCE under a new page layout and then
#: go stable again. "okf1" = the OKF-conformant frontmatter rollout (G1):
#: pages rendered before it lack the required ``type``/``description``/
#: ``timestamp`` fields, and the idempotent skip would otherwise preserve
#: them as non-conformant forever.
_RENDER_SCHEMA_SALT = b"okf1"


def _content_fingerprint(memories: list[dict]) -> str:
    """Order-independent digest of a topic's (id, content, faded) triples.

    The idempotent skip compares this alongside the id set: a memory
    rewritten in place (same id, new text — e.g. by this sweep's own
    REWRITE/MERGE reconciliation) must re-render its page even though the
    ``source_memory_ids`` set is unchanged. The faded flag participates
    too — a row crossing the retention threshold changes the rendering
    (main section → Faded callout) with identical ids and content.
    The render-schema salt participates deliberately — see its docstring.
    """
    digest = hashlib.sha256()
    digest.update(_RENDER_SCHEMA_SALT)
    digest.update(b"\x00")
    for mem in sorted(memories, key=lambda m: m.get("memory_id") or ""):
        digest.update((mem.get("memory_id") or "").encode())
        digest.update(b"\x00")
        digest.update((mem.get("content") or "").strip().encode())
        digest.update(b"F" if mem.get("dream_faded") else b"-")
        digest.update(b"\x01")
    return digest.hexdigest()[:16]


def _wikilink(title: str) -> str:
    """[[Page]] when the title already is its page name, else [[Page|Title]]."""
    page = _slug_title(title)
    return f"[[{page}]]" if page == title else f"[[{page}|{title}]]"


# ── Skeleton parsing / rendering (B2) ───────────────────────────────


def _canon_section(name: str) -> str | None:
    """Normalize an LLM-emitted section name onto the fixed five (or None)."""
    n = (name or "").strip().casefold().replace("&", "and")
    for section in SECTION_ORDER:
        if n == section.casefold().replace("&", "and"):
            return section
    return None


def parse_merge_response(raw: str) -> tuple[list[dict], dict[str, str]] | None:
    """Validate the TOPIC_MERGE JSON into (index_card rows, sections).

    Returns None when the response carries no usable structure — the
    caller falls back to deterministic category bucketing so the fixed
    skeleton survives LLM drift.
    """
    obj = parse_json_object(raw)
    raw_sections = obj.get("sections")
    if not isinstance(raw_sections, dict):
        return None
    sections: dict[str, str] = {s: "" for s in SECTION_ORDER}
    strays: list[str] = []
    for key, value in raw_sections.items():
        if not isinstance(value, str) or not value.strip():
            continue
        canon = _canon_section(str(key))
        if canon:
            sections[canon] = (sections[canon] + "\n\n" + value.strip()).strip()
        else:
            strays.append(value.strip())
    if strays:  # never drop content over a heading the model invented
        sections[_DEFAULT_SECTION] = (
            (sections[_DEFAULT_SECTION] + "\n\n" + "\n\n".join(strays)).strip()
        )
    if not any(sections.values()):
        return None

    first_filled = next((s for s in SECTION_ORDER if sections[s]), _DEFAULT_SECTION)
    cards: list[dict] = []
    for row in obj.get("index_card") or []:
        if not isinstance(row, dict):
            continue
        what = str(row.get("what") or "").strip()
        if not what:
            continue
        raw_entities = row.get("entities")
        entities = (
            [str(e).strip() for e in raw_entities if str(e).strip()]
            if isinstance(raw_entities, list)
            else []
        )
        cards.append(
            {
                "what": what,
                "entities": entities[:4],
                "source": _canon_section(str(row.get("source") or "")) or first_filled,
            }
        )
        if len(cards) >= _INDEX_CARD_MAX_ROWS:
            break
    return cards, sections


def fallback_structure(memories: list[dict]) -> tuple[list[dict], dict[str, str]]:
    """Deterministic skeleton when the merge LLM output is unusable.

    Buckets each memory into its section by category (unknown/adapter
    categories → Discoveries) and derives index-card rows from the
    highest-promotion-score memories.
    """
    buckets: dict[str, list[str]] = {s: [] for s in SECTION_ORDER}
    for mem in memories:
        content = (mem.get("content") or "").strip()
        if not content:
            continue
        section = CATEGORY_SECTION.get(mem.get("category") or "", _DEFAULT_SECTION)
        buckets[section].append(f"- {_one_line(content, 240)}")
    ranked = sorted(
        (m for m in memories if (m.get("content") or "").strip()),
        key=lambda m: float(m.get("promotion_score") or 0.0),
        reverse=True,
    )
    cards = [
        {
            "what": _one_line(m.get("content") or "", 100),
            "entities": [],
            "source": CATEGORY_SECTION.get(m.get("category") or "", _DEFAULT_SECTION),
        }
        for m in ranked[:_INDEX_CARD_MAX_ROWS]
    ]
    return cards, {s: "\n".join(rows) for s, rows in buckets.items()}


def _render_index_card(rows: list[dict], known_pages) -> list[str]:
    """The compact pointer table every topic page opens with."""
    known = {_slug_title(str(p)).casefold(): _slug_title(str(p)) for p in known_pages}

    def _entity(entity: str) -> str:
        page = known.get(_slug_title(entity).casefold())
        return f"[[{page}]]" if page else entity.replace("|", "\\|")

    lines = ["| What | Entities | Source |", "| --- | --- | --- |"]
    for row in rows[:_INDEX_CARD_MAX_ROWS]:
        what = _one_line(str(row.get("what") or ""), 120).replace("|", "\\|")
        if not what:
            continue
        entities = ", ".join(_entity(str(e)) for e in row.get("entities") or []) or "—"
        source = row.get("source")
        source_cell = f"[[#{source}]]" if source in SECTION_ORDER else "—"
        lines.append(f"| {what} | {entities} | {source_cell} |")
    return lines if len(lines) > 2 else []


def render_topic_page(
    *,
    title: str,
    pool: str,
    summary: str,
    memory_ids: list[str],
    categories: list[str],
    hub_link: str | None,
    version: int,
    index_card: list[dict] | None = None,
    sections: dict[str, str] | None = None,
    body: str = "",
    known_pages=(),
    content_hash: str = "",
    faded_lines: list[str] | None = None,
    now: datetime | None = None,
) -> str:
    """Render one topic page.

    The skeleton path (``index_card`` + ``sections``) renders the fixed
    layout: index-card table, then the five sections in ``SECTION_ORDER``
    with empty sections omitted. The legacy ``body`` path renders a raw
    markdown body verbatim (used by the wiki migration script; such pages
    are restructured into the skeleton on their next dream).

    ``faded_lines`` (B3b) render as a collapsed Obsidian callout at the
    bottom of the body — dimmed rows leave the main sections but never
    the page.
    """
    ts = (now or datetime.now(timezone.utc)).isoformat()
    tags = " ".join(sorted({f"#{c}" for c in categories if c}))
    body_lines: list[str] = []
    if index_card:
        card = _render_index_card(index_card, known_pages)
        if card:
            body_lines += card + [""]
    if sections:
        for section in SECTION_ORDER:
            text = (sections.get(section) or "").strip()
            if not text:
                continue
            body_lines += [f"## {section}", "", text, ""]
    if body.strip() and not sections:
        body_lines += [body.strip(), ""]
    if faded_lines:
        body_lines += [
            FADED_START,
            _FADED_CALLOUT_TITLE,
            "> Dimmed, not deleted — below the retention threshold.",
            *(f"> - {_one_line(line, 200)}" for line in faded_lines),
            FADED_END,
            "",
        ]
    while body_lines and not body_lines[-1]:
        body_lines.pop()

    # OKF-conformant frontmatter (G1): required type + recommended
    # title/description/tags/timestamp come from the translation module;
    # the NS page envelope (summary/pool/source ids/hash/version) rides
    # along as spec-permitted extension keys.
    frontmatter = okf_translate.concept_frontmatter(
        page_kind="topic",
        title=title,
        description=summary or f"Topic page dreamt from {len(memory_ids)} memories.",
        tags=categories,
        timestamp=ts,
        extensions={
            "summary": summary,
            "pool": pool,
            "source_memory_ids": f"[{', '.join(memory_ids)}]",
            "content_hash": content_hash or None,
            "last_dreamt": ts,
            "version": version,
        },
    )
    lines = [
        frontmatter,
        "",
        f"# {title}",
        "",
        *body_lines,
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
    redis=None,
    faded_threshold: float | None = None,
) -> dict:
    """Cluster the pool's staged memories into topics and update pages.

    Returns ``{pages_written, pages_skipped, topics}`` for the DreamRun
    report. Uses the *staged* batch (post-consolidation view of what's
    live and current) — tombstoned rows never reach the vault.

    When ``redis`` is provided, the pool's top promotion-scored lines are
    persisted under ``dreaming:essential:{pool}`` so Home.md's Essential
    Story can rank across pools without re-scrolling them.

    ``faded_threshold`` (B3b, normally the sweep's prune-strength
    threshold): live rows whose ``retention_strength`` sits below it are
    *faded* — they still cluster into topics but are excluded from the
    merge LLM, the main sections, the Essential Story and the identity
    block, rendering instead as one-liners in the collapsed Faded callout.
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
    if faded_threshold is not None:
        for mem in live:
            strength = mem.get("retention_strength")
            if isinstance(strength, (int, float)) and strength < faded_threshold:
                mem["dream_faded"] = True
    if len(live) < 2:
        return out

    from .prompts import render_memories_block

    raw = await llm_call(TOPIC_CLUSTER_PROMPT.format(
        memories_block=render_memories_block(live, include_strength=False)
    ))
    topics = parse_json_response(raw, key="topics")
    known = {m["memory_id"]: m for m in live}
    hub_name = target.name if target.name not in ("Knowledge", "Me") else None

    # (title, page_name, ids, summary) for every well-formed topic
    valid_topics: list[tuple[str, str, list[str], str]] = []
    for topic in topics:
        title = (topic.get("title") or "").strip()
        ids = [i for i in (topic.get("memory_ids") or []) if i in known]
        if not title or len(ids) < 2:
            continue
        valid_topics.append(
            (title, _slug_title(title), ids, (topic.get("summary") or "").strip())
        )

    sibling_titles = [page_name for _, page_name, _, _ in valid_topics]

    for title, page_name, ids, summary in valid_topics:
        path = target / f"{page_name}.md"
        existing = path.read_text(encoding="utf-8") if path.exists() else ""
        fm, existing_body = split_page(existing)

        mems = [known[i] for i in ids]
        fingerprint = _content_fingerprint(mems)
        # Idempotent skip: same id set AND same contents (pages predating
        # the fingerprint match on ids alone — they gain the hash on their
        # next id-set change).
        if _parse_id_list(fm.get("source_memory_ids", "")) == set(ids) and fm.get(
            "content_hash", fingerprint
        ) == fingerprint:
            out["pages_skipped"] += 1
            continue

        siblings = [s for s in sibling_titles if s != page_name]
        if hub_name:
            siblings.append(hub_name)
        # Faded rows (B3b) never reach the merge LLM or the main sections;
        # they become one-liners in the collapsed callout instead.
        active = [m for m in mems if not m.get("dream_faded")]
        faded = [m for m in mems if m.get("dream_faded")]
        faded_lines = [_one_line(m.get("content") or "", 200) for m in faded]
        if active:
            raw_merge = await llm_call(TOPIC_MERGE_PROMPT.format(
                title=title,
                siblings=", ".join(f"[[{s}]]" for s in siblings) or "(none)",
                existing_body=existing_body or "(empty — first synthesis)",
                memories_block=render_memories_block(active, include_strength=False),
            ))
            if not (raw_merge or "").strip():
                continue  # LLM exhausted its retries — leave the page alone
            parsed = parse_merge_response(raw_merge)
            if parsed is None:
                logger.warning(
                    "topic merge for %r returned no usable skeleton — falling back "
                    "to category bucketing", title,
                )
                parsed = fallback_structure(active)
        else:
            # every row faded — no LLM pass, the page is just the callout
            parsed = ([], {})
        index_card, sections = parsed
        rendered = render_topic_page(
            title=title,
            pool=batch.pool,
            summary=summary,
            index_card=index_card,
            sections=sections,
            memory_ids=sorted(ids),
            categories=sorted({m.get("category") or "" for m in mems} - {""}),
            hub_link=hub_name,
            version=int(fm.get("version") or 0) + 1,
            known_pages=siblings,
            content_hash=fingerprint,
            faded_lines=faded_lines,
        )
        if not dry_run:
            _atomic_write(path, rendered)
        out["pages_written"] += 1
        out["topics"].append({"title": title, "page": str(path), "memories": len(ids)})

    if not dry_run:
        if redis is not None and valid_topics:
            _store_essential(redis, batch.pool, _essential_candidates(valid_topics, known))
        if out["pages_written"]:
            identity_lines = None
            if batch.visibility == "private" and batch.owner_user_id == operator_user_id:
                identity_lines = _identity_lines(live) or None
            _write_hub(target, hub_name, _atomic_write)
            _write_home(vault, _atomic_write, redis=redis, identity_lines=identity_lines)
            # OKF bundle surface (G1): per-folder index.md + the root
            # version marker. Byte-idempotent — steady state writes nothing.
            try:
                from okf.vault import refresh_bundle_indexes

                refresh_bundle_indexes(vault)
            except Exception:
                logger.warning("okf bundle index refresh failed (non-fatal)", exc_info=True)
    return out


# ── Essential Story persistence (B1) ────────────────────────────────


def _essential_candidates(valid_topics, known: dict[str, dict]) -> list[dict]:
    """The pool's top one-liners, wikilink-ready.

    Ranked by ``salience`` (A4 dynamics-driven when the memory has recall
    state, retention fallback otherwise — B1: "top memories *by
    salience*"), with ``promotion_score`` as the fallback for staged rows
    scored before the salience field existed.

    Faded rows never make the story — Home is the wake-up stack, and a
    memory dimmed below the retention threshold has no business there.
    """

    def _rank(mem: dict) -> float:
        val = mem.get("salience")
        if val is None:
            val = mem.get("promotion_score") or 0.0
        try:
            return float(val)
        except (TypeError, ValueError):
            return 0.0

    out: list[dict] = []
    for title, page_name, ids, _summary in valid_topics:
        for mid in ids:
            mem = known[mid]
            if mem.get("dream_faded"):
                continue
            out.append(
                {
                    "score": round(_rank(mem), 4),
                    "text": _one_line(mem.get("content") or "", 200),
                    "page": page_name,
                    "title": title,
                }
            )
    out.sort(key=lambda e: e["score"], reverse=True)
    return out[:HOME_STORY_TOP_N]


def _store_essential(redis, pool: str, entries: list[dict]) -> None:
    key = _ESSENTIAL_KEY.format(pool=pool)
    try:
        if entries:
            redis.set(key, json.dumps(entries), ex=_ESSENTIAL_TTL)
        else:
            redis.delete(key)
    except Exception:
        logger.warning("essential-story write failed for %s (non-fatal)", pool, exc_info=True)


def _load_essential(redis) -> list[dict]:
    """All pools' persisted essential lines (best-effort, [] on any failure)."""
    if redis is None:
        return []
    entries: list[dict] = []
    try:
        pattern = _ESSENTIAL_KEY.format(pool="*")
        keys = (
            list(redis.scan_iter(match=pattern))
            if hasattr(redis, "scan_iter")
            else redis.keys(pattern)
        )
        for key in keys:
            raw = redis.get(key)
            if not raw:
                continue
            data = json.loads(raw)
            if isinstance(data, list):
                entries.extend(e for e in data if isinstance(e, dict))
    except Exception:
        logger.warning("essential-story read failed (non-fatal)", exc_info=True)
    return entries


# ── Home.md v2 (B1) ─────────────────────────────────────────────────


def _identity_lines(memories: list[dict]) -> list[str]:
    """L0: the operator's highest-salience personal facts/preferences."""
    candidates = [
        m for m in memories
        if (m.get("category") or "") in _IDENTITY_CATEGORIES
        and (m.get("content") or "").strip()
        and not m.get("dream_faded")
    ]
    candidates.sort(key=lambda m: float(m.get("promotion_score") or 0.0), reverse=True)
    return [f"- {_one_line(m['content'], 160)}" for m in candidates[:IDENTITY_MAX_LINES]]


def _parse_identity_block(home_text: str) -> list[str]:
    """Recover the previous Home.md identity block (cache tolerance)."""
    if not home_text or _ID_START not in home_text or _ID_END not in home_text:
        return []
    block = home_text.split(_ID_START, 1)[1].split(_ID_END, 1)[0]
    return [line for line in block.splitlines() if line.strip()]


def _render_story(entries: list[dict]) -> list[str]:
    """L1: the budget-bounded Essential Story block (≤ HOME_STORY_BUDGET chars)."""
    if not entries:
        return []
    ranked = sorted(
        entries, key=lambda e: float(e.get("score") or 0.0), reverse=True
    )[:HOME_STORY_TOP_N]
    # group lines by topic page; page order follows each page's best line
    groups: dict[str, list[dict]] = {}
    for entry in ranked:
        page = str(entry.get("page") or "").strip()
        text = str(entry.get("text") or "").strip()
        if page and text:
            groups.setdefault(page, []).append(entry)
    header = ["## Essential Story", ""]
    used = sum(len(line) + 1 for line in header)
    lines: list[str] = []
    for page, group in groups.items():
        for entry in group:
            title = str(entry.get("title") or page).strip()
            link = f"[[{page}]]" if title == page else f"[[{page}|{title}]]"
            line = f"- {_one_line(str(entry['text']), 200)} — {link}"
            if used + len(line) + 1 > HOME_STORY_BUDGET:
                return header + lines if lines else []
            lines.append(line)
            used += len(line) + 1
    return header + lines if lines else []


def _hub_stats(directory: Path) -> tuple[int, int, str]:
    """(pages, memories, last_dreamt) for one pool dir, from disk alone."""
    pages = memories = 0
    last = ""
    if not directory.exists():
        return 0, 0, "—"
    for path in sorted(directory.glob("*.md")):
        if path.stem == directory.name or path.stem == "Card":  # hub / identity card
            continue
        if okf_translate.is_reserved_filename(path.name):  # index.md / log.md
            continue
        try:
            fm, _ = split_page(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        pages += 1
        memories += len(_parse_id_list(fm.get("source_memory_ids", "")))
        stamp = (fm.get("last_dreamt") or "").strip()
        if not stamp:
            stamp = datetime.fromtimestamp(
                path.stat().st_mtime, tz=timezone.utc
            ).isoformat()
        last = max(last, stamp)
    return pages, memories, (last[:10] if last else "—")


def _render_moc(vault: Path) -> list[str]:
    """The map-of-content table: ``| Hub | Pages | Memories | Last dreamt |``."""
    rows: list[tuple[str, int, int, str]] = []
    projects_dir = vault / "Projects"
    if projects_dir.exists():
        for pdir in sorted(p for p in projects_dir.iterdir() if p.is_dir()):
            pages, memories, last = _hub_stats(pdir)
            if pages:
                rows.append((f"[[{pdir.name}]]", pages, memories, last))
    for name in ("Knowledge", "Me"):
        pages, memories, last = _hub_stats(vault / name)
        if pages:
            rows.append((name, pages, memories, last))
    if not rows:
        return []
    out = [
        "## Map of Content",
        "",
        "| Hub | Pages | Memories | Last dreamt |",
        "| --- | --- | --- | --- |",
    ]
    for hub, pages, memories, last in rows:
        out.append(f"| {hub} | {pages} | {memories} | {last} |")
    return out


def _list_topic_pages(directory: Path) -> list[dict]:
    """Per-topic-page metadata for a pool dir (title/summary/counts)."""
    pages: list[dict] = []
    if not directory.exists():
        return pages
    for path in sorted(directory.glob("*.md")):
        if path.stem == directory.name or path.stem == "Card":  # hub / identity card
            continue
        if okf_translate.is_reserved_filename(path.name):  # index.md / log.md
            continue
        fm, _ = split_page(path.read_text(encoding="utf-8"))
        pages.append(
            {
                "title": fm.get("title") or path.stem,
                "summary": fm.get("summary") or "",
                "memories": len(_parse_id_list(fm.get("source_memory_ids", ""))),
                "last_dreamt": (fm.get("last_dreamt") or "")[:10],
            }
        )
    return pages


def _write_hub(target: Path, hub_name: str | None, atomic_write) -> None:
    """Regenerate the pool's hub page (Projects/<pid>/<pid>.md)."""
    if not hub_name:
        return
    entries = _list_topic_pages(target)
    ts = datetime.now(timezone.utc).isoformat()
    lines = [
        okf_translate.concept_frontmatter(
            page_kind="hub",
            title=hub_name,
            description=f"What the memory system knows about {hub_name}, by subject.",
            timestamp=ts,
            extensions={
                "pages": len(entries),
                "memories": sum(e["memories"] for e in entries),
                "last_dreamt": ts,
            },
        ),
        "",
        f"# {hub_name}",
        "",
        "What the memory system knows about this project, by subject:",
        "",
    ]
    for e in entries:
        counts = f"{e['memories']} memories" if e["memories"] else "no memories yet"
        if e["last_dreamt"]:
            counts += f", dreamt {e['last_dreamt']}"
        lines.append(f"- {_wikilink(e['title'])} — {e['summary']} _({counts})_")
    lines += ["", "Back to [[Home]].", ""]
    atomic_write(target / f"{hub_name}.md", "\n".join(lines))


def _write_home(
    vault: Path,
    atomic_write,
    *,
    redis=None,
    identity_lines: list[str] | None = None,
) -> None:
    """Regenerate Home.md — L0 identity, L1 Essential Story, MOC with counts.

    Cache-tolerant: when no fresh operator identity lines are supplied
    (Home is regenerated after *any* pool's sweep, not just the
    operator's), the previous block is carried over from the existing
    Home.md between the ``ns:identity`` markers.
    """
    home_path = vault / "Home.md"
    previous = ""
    if home_path.exists():
        try:
            previous = home_path.read_text(encoding="utf-8")
        except Exception:
            previous = ""
    identity = [l for l in (identity_lines or []) if l.strip()]
    if not identity:
        identity = _parse_identity_block(previous)

    ts = datetime.now(timezone.utc).isoformat()
    lines = [
        okf_translate.concept_frontmatter(
            page_kind="home",
            title="Home",
            description="The wake-up stack: operator identity, essential story, and the map of content.",
            timestamp=ts,
            extensions={"last_dreamt": ts},
        ),
        "",
        "# Home",
        "",
    ]
    if identity:
        lines += [_ID_START, *identity[:IDENTITY_MAX_LINES], _ID_END, ""]
    lines += ["Everything the memory system knows, dreamt into browsable pages.", ""]
    story = _render_story(_load_essential(redis))
    if story:
        lines += story + [""]
    moc = _render_moc(vault)
    if moc:
        lines += moc + [""]
    for section, dirname in (("Knowledge", "Knowledge"), ("Me", "Me")):
        entries = _list_topic_pages(vault / dirname)
        if entries:
            lines += [f"## {section}", ""]
            for e in entries:
                lines.append(f"- {_wikilink(e['title'])} — {e['summary']}")
            lines.append("")
    if (vault / "Dreams").exists():
        lines += ["## Dream journal", "", "Recent sweeps live in `Dreams/`.", ""]
    atomic_write(home_path, "\n".join(lines))
