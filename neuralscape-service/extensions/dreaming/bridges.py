"""Bridges — cross-hub tunnels between topic pages (roadmap B3a).

When one subject surfaces under multiple hubs (two projects, or Knowledge
plus a project, or Me plus a project), every involved topic page gets a
``## Bridges`` section with labeled, path-qualified wikilinks to its
counterpart page(s). The section is a managed block bracketed by HTML
comments so the patcher can rewrite it in place without disturbing the
rest of the page (frontmatter, skeleton sections, the Faded callout).

Connection sources, cheapest first:

1. **Deterministic signals** (always available, even with a sparse graph):
   - identical topic slugs across hubs — the same page name under two
     hubs *is* the same subject;
   - shared ``source_memory_ids`` across hubs — two pages distilled from
     an overlapping memory set are two views of one subject.
2. **Graph enrichment** (best-effort): entities that co-occur across
   memories in different pools, from ONE bounded Cypher per sweep
   (``fetch_graph_rows``); the entity name becomes the bridge label.

Guarantees:

- **Reciprocal** — links are built pairwise in both directions.
- **Idempotent** — links are sorted and deduplicated; a page is only
  rewritten when its bridges block actually changed, and a stale block is
  removed when the connection disappears.
- **Dry-run safe** — planning happens unconditionally; writes don't.

The pass runs once per sweep AFTER every pool's librarian pass (it is
cross-pool by nature), scanning the vault from disk alone — the same
source of truth Home.md's MOC uses.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from okf.translate import is_reserved_filename

from .librarian import FADED_START, _parse_id_list, split_page

logger = logging.getLogger(__name__)

BRIDGES_START = "<!-- ns:bridges -->"
BRIDGES_END = "<!-- /ns:bridges -->"

#: Ceiling on bridge links rendered per page — a hub-spanning subject
#: should read as a short list of tunnels, not a second MOC.
MAX_LINKS_PER_PAGE = 8

#: One bounded query per sweep: entity names whose nodes span >= 2 graph
#: group_ids (pools), with the memory_ids they were extracted from. The
#: memory_ids map onto topic pages via frontmatter ``source_memory_ids``;
#: hub-spanning is re-checked on the vault side, so this only needs to be
#: a generous superset.
SHARED_ENTITY_CYPHER = """
MATCH (n)
WHERE n.memory_id IS NOT NULL AND n.name IS NOT NULL AND n.group_id IS NOT NULL
WITH toLower(trim(n.name)) AS key,
     head(collect(n.name)) AS name,
     collect(DISTINCT n.memory_id) AS memory_ids,
     collect(DISTINCT n.group_id) AS groups
WHERE size(memory_ids) >= 2 AND size(groups) >= 2
RETURN name, memory_ids
ORDER BY size(groups) DESC, size(memory_ids) DESC
LIMIT $limit
"""


@dataclass(slots=True)
class TopicPage:
    """One topic page's bridge-relevant identity, read from disk."""

    path: Path
    rel: str          # vault-relative link target without extension, e.g. "Projects/alpha/TURN"
    hub: str          # hub key, e.g. "Projects/alpha", "Knowledge", "Me"
    label: str        # human hub label for link text, e.g. "alpha", "Knowledge", "Me"
    title: str
    slug: str         # casefolded page stem
    source_ids: set[str] = field(default_factory=set)


def scan_topic_pages(vault: Path) -> list[TopicPage]:
    """Collect every topic page across all hubs (hub pages + cards excluded)."""
    pages: list[TopicPage] = []

    def _collect(directory: Path, hub: str, label: str) -> None:
        if not directory.exists():
            return
        for path in sorted(directory.glob("*.md")):
            if path.stem == directory.name or path.stem == "Card":
                continue
            if is_reserved_filename(path.name):  # OKF index.md / log.md
                continue
            try:
                fm, _ = split_page(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            pages.append(TopicPage(
                path=path,
                rel=f"{hub}/{path.stem}",
                hub=hub,
                label=label,
                title=fm.get("title") or path.stem,
                slug=path.stem.casefold(),
                source_ids=_parse_id_list(fm.get("source_memory_ids", "")),
            ))

    projects = vault / "Projects"
    if projects.exists():
        for pdir in sorted(p for p in projects.iterdir() if p.is_dir()):
            _collect(pdir, f"Projects/{pdir.name}", pdir.name)
    _collect(vault / "Knowledge", "Knowledge", "Knowledge")
    _collect(vault / "Me", "Me", "Me")
    return pages


# ── Connection discovery ────────────────────────────────────────────


def _pair_key(a: TopicPage, b: TopicPage) -> tuple[str, str]:
    return (a.rel, b.rel) if a.rel <= b.rel else (b.rel, a.rel)


def compute_bridges(
    pages: list[TopicPage],
    graph_rows: list[dict] | None = None,
) -> dict[str, list[tuple[TopicPage, str]]]:
    """Cross-hub connections as ``{page.rel: [(counterpart, reason), ...]}``.

    Reciprocal by construction: every pair contributes a link in both
    directions. Reasons for the same pair accumulate ("same subject;
    shares entity \"X\"").
    """
    reasons: dict[tuple[str, str], list[str]] = {}
    by_rel = {p.rel: p for p in pages}

    def _connect(a: TopicPage, b: TopicPage, reason: str) -> None:
        if a.hub == b.hub:
            return  # tunnels cross hubs; sibling links are the librarian's job
        bucket = reasons.setdefault(_pair_key(a, b), [])
        if reason not in bucket:
            bucket.append(reason)

    # 1a. identical slugs across hubs
    by_slug: dict[str, list[TopicPage]] = {}
    for page in pages:
        by_slug.setdefault(page.slug, []).append(page)
    for group in by_slug.values():
        for i, a in enumerate(group):
            for b in group[i + 1:]:
                _connect(a, b, "same subject")

    # 1b. shared source memories across hubs — indexed by memory_id so the
    # scan stays proportional to actual overlaps, not O(pages²)
    by_memory: dict[str, list[TopicPage]] = {}
    for page in pages:
        for mid in page.source_ids:
            by_memory.setdefault(mid, []).append(page)
    shared_counts: dict[tuple[str, str], int] = {}
    for holders in by_memory.values():
        for i, a in enumerate(holders):
            for b in holders[i + 1:]:
                if a.hub != b.hub:
                    key = _pair_key(a, b)
                    shared_counts[key] = shared_counts.get(key, 0) + 1
    for (rel_a, rel_b), n in shared_counts.items():
        _connect(by_rel[rel_a], by_rel[rel_b],
                 f"shares {n} source memor{'y' if n == 1 else 'ies'}")

    # 2. graph enrichment: entities spanning pools → the pages holding them
    for row in graph_rows or []:
        name = str(row.get("name") or "").strip()
        memory_ids = {str(m) for m in (row.get("memory_ids") or []) if m}
        if not name or len(memory_ids) < 2:
            continue
        holders = [p for p in pages if p.source_ids & memory_ids]
        for i, a in enumerate(holders):
            for b in holders[i + 1:]:
                _connect(a, b, f'shares entity "{name}"')

    links: dict[str, list[tuple[TopicPage, str]]] = {}
    for (rel_a, rel_b), why in reasons.items():
        a, b = by_rel[rel_a], by_rel[rel_b]
        reason = "; ".join(why)
        links.setdefault(a.rel, []).append((b, reason))
        links.setdefault(b.rel, []).append((a, reason))
    for rel in links:
        links[rel].sort(key=lambda pair: pair[0].rel)
        links[rel] = links[rel][:MAX_LINKS_PER_PAGE]
    return links


# ── Rendering + patching ────────────────────────────────────────────


def render_bridges_block(links: list[tuple[TopicPage, str]]) -> str:
    """The managed ``## Bridges`` block (empty string when no links)."""
    if not links:
        return ""
    lines = [BRIDGES_START, "## Bridges", ""]
    for page, reason in links:
        lines.append(f"- [[{page.rel}|{page.title} ({page.label})]] — {reason}")
    lines += [BRIDGES_END]
    return "\n".join(lines)


_BLOCK_RE = re.compile(
    re.escape(BRIDGES_START) + r".*?" + re.escape(BRIDGES_END) + r"\n?",
    re.DOTALL,
)


def _insertion_point(text: str) -> int:
    """Where a fresh bridges block goes: before the Faded callout when the
    page has one, else before the trailing footer separator, else the end."""
    idx = text.find(FADED_START)
    if idx != -1:
        return idx
    fm_end = 0
    if text.startswith("---\n"):
        m = re.search(r"\n---\n", text)
        if m:
            fm_end = m.end()
    idx = text.rfind("\n---\n")
    if idx != -1 and idx >= fm_end:
        return idx + 1  # keep the preceding newline with the body
    return len(text)


def patch_page(text: str, block: str) -> str:
    """Replace/insert/remove the managed bridges block in one page's text."""
    if BRIDGES_START in text and BRIDGES_END in text:
        replacement = block + "\n" if block else ""
        new_text = _BLOCK_RE.sub(lambda _: replacement, text, count=1)
        return re.sub(r"\n{3,}", "\n\n", new_text)
    if not block:
        return text
    at = _insertion_point(text)
    head, tail = text[:at], text[at:]
    if head and not head.endswith("\n\n"):
        head = head.rstrip("\n") + "\n\n"
    if tail and not tail.startswith("\n"):
        tail = "\n" + tail
    return head + block + "\n" + tail


def update_bridges(
    vault: Path,
    *,
    graph_rows: list[dict] | None = None,
    dry_run: bool = False,
) -> dict:
    """The full bridges pass: scan → connect → patch changed pages.

    Returns ``{"pages_bridged": n, "pages_unchanged": m, "links": total}``.
    """
    out = {"pages_bridged": 0, "pages_unchanged": 0, "links": 0}
    try:
        pages = scan_topic_pages(vault)
    except Exception:
        logger.warning("bridge scan failed for %s (non-fatal)", vault, exc_info=True)
        return out
    links = compute_bridges(pages, graph_rows)
    out["links"] = sum(len(v) for v in links.values())

    from extensions.conversation_compiler.obsidian_writer import _atomic_write

    for page in pages:
        block = render_bridges_block(links.get(page.rel, []))
        try:
            text = page.path.read_text(encoding="utf-8")
        except Exception:
            continue
        patched = patch_page(text, block)
        if patched == text:
            out["pages_unchanged"] += 1
            continue
        if not dry_run:
            try:
                _atomic_write(page.path, patched)
            except Exception:
                logger.warning("bridge write failed for %s (non-fatal)", page.path, exc_info=True)
                continue
        out["pages_bridged"] += 1
    return out


# ── Graph enrichment fetch (best-effort, one query per sweep) ───────


async def fetch_graph_rows(service, *, limit: int = 200) -> list[dict]:
    """Cross-pool shared-entity rows from Graphiti, ``[]`` on any failure.

    Runs :data:`SHARED_ENTITY_CYPHER` once on the service's bridge loop
    (the loop Graphiti's async driver was created on — same dispatch as
    ``consolidate._graph_invalidate``). Returns
    ``[{"name": str, "memory_ids": [str]}]``.
    """
    graphiti = getattr(service, "_graphiti", None)
    driver = getattr(graphiti, "driver", None) if graphiti else None
    if driver is None or not getattr(service, "_bridge", None):
        return []

    async def _inner() -> list[dict]:
        async with driver.session() as session:
            cursor = await session.run(SHARED_ENTITY_CYPHER, limit=int(limit))
            return await cursor.data()

    from extensions.dreaming.graph_patcher import _is_kuzu

    if _is_kuzu(driver):
        # Kuzu arm: fetch raw (name, memory_id, group_id) rows per node table
        # and run the aggregation pipeline (toLower/trim key, head-of-collect
        # name, DISTINCT collects, >=2/>=2 hub filter, ordering, limit) in
        # Python — Kuzu support for those Cypher aggregation forms is
        # unverified, and solo graphs are small enough to aggregate app-side.
        async def _inner_kuzu() -> list[dict]:
            raw: list[dict] = []
            for label in ("Entity", "Episodic", "Community", "Saga"):
                out, _, _ = await driver.execute_query(
                    f"MATCH (n:{label}) WHERE n.memory_id IS NOT NULL "
                    f"AND n.name IS NOT NULL AND n.group_id IS NOT NULL "
                    f"RETURN n.name AS name, n.memory_id AS memory_id, "
                    f"n.group_id AS group_id"
                )
                raw.extend(out)
            agg: dict[str, dict] = {}
            for r in raw:
                name = str(r.get("name") or "")
                key = name.strip().lower()
                if not key:
                    continue
                a = agg.setdefault(
                    key,
                    {"name": name, "memory_ids": [], "_mids": set(), "groups": set()},
                )
                mid = r.get("memory_id")
                if mid and mid not in a["_mids"]:
                    a["_mids"].add(mid)
                    a["memory_ids"].append(mid)
                a["groups"].add(r.get("group_id"))
            hubs = [
                a for a in agg.values()
                if len(a["memory_ids"]) >= 2 and len(a["groups"]) >= 2
            ]
            hubs.sort(key=lambda a: (-len(a["groups"]), -len(a["memory_ids"])))
            return [
                {"name": a["name"], "memory_ids": a["memory_ids"]}
                for a in hubs[: int(limit)]
            ]

        fetch = _inner_kuzu()
    else:
        fetch = _inner()

    try:
        rows = await service._run_on_bridge_async(fetch, timeout=30.0)
        return [
            {
                "name": str(r.get("name") or ""),
                "memory_ids": [str(m) for m in (r.get("memory_ids") or []) if m],
            }
            for r in rows or []
        ]
    except Exception:
        logger.warning("shared-entity bridge query failed (non-fatal)", exc_info=True)
        return []
