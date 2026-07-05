"""One-shot migration: legacy Wiki/ category pages → humane topic layout.

Converts the wiki_synthesizer's taxonomy-first tree

    Wiki/<scope>/<TypeGroup>/<CategoryLeaf>.md

into the dreaming librarian's subject-first layout

    Projects/<scope>/<Topic>.md (+ <scope>.md hub)   for project scopes
    Knowledge/<Topic>.md                             for the global scope
    Home.md                                          map of content

Per scope, ONE LLM call reads every legacy page and re-partitions the
material into browsable subject topics with narrative bodies and
[[wikilinks]]. Legacy pages are then archived (moved, never deleted) to
``_archive/Wiki-pre-dreaming-<date>/``.

Migrated pages carry ``migrated: true`` and the union of their scope's
legacy ``source_memory_ids`` — the first real dream sweep re-anchors
each topic to its precise live memory set.

Usage:
    uv run python scripts/migrate_wiki_to_topics.py --vault /path/to/vault           # dry run
    uv run python scripts/migrate_wiki_to_topics.py --vault /path/to/vault --apply
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MIGRATE_PROMPT = """\
You are the librarian of a personal knowledge vault, migrating old
category-bucketed wiki pages into browsable SUBJECT topics. Emit STRICT
JSON only — no prose, no markdown fences.

Rules:
- Re-partition ALL the material below into 2-10 subject topics a human
  would browse. Titles are concrete noun phrases in Title Case, max 5
  words. NEVER taxonomy words (Architecture, Conventions, Procedures,
  Workflows, Decisions, Semantic, Episodic).
- Each topic body: a coherent narrative in markdown — short intro, then
  `##` sections as needed. Preserve every concrete fact; merge overlaps;
  when pages contradict, the "Recent updates" bullets win.
- Weave [[wikilinks]] between the topics you create where genuinely
  related. Keep each body under ~700 words.
- summary: one browsable sentence for the hub index.

Output schema:
{{"topics": [{{"title": "...", "summary": "...", "body": "..."}}]}}

SCOPE: {scope}

LEGACY PAGES:
{pages_block}
"""


def split_page(content: str) -> tuple[dict, str]:
    from extensions.dreaming.librarian import split_page as _sp

    return _sp(content)


def collect_scopes(wiki_root: Path) -> dict[str, list[Path]]:
    scopes: dict[str, list[Path]] = {}
    for path in sorted(wiki_root.rglob("*.md")):
        scope = path.relative_to(wiki_root).parts[0]
        scopes.setdefault(scope, []).append(path)
    return scopes


async def migrate_scope(scope: str, pages: list[Path], vault: Path, *, apply: bool) -> dict:
    from extensions.conversation_compiler.obsidian_writer import _atomic_write
    from extensions.dreaming.librarian import _slug_title, render_topic_page
    from extensions.dreaming.sweep import _make_llm_call
    from extensions.dreaming.config import dreaming_settings

    blocks, all_ids = [], set()
    for path in pages:
        fm, body = split_page(path.read_text(encoding="utf-8"))
        all_ids |= {
            i.strip()
            for i in fm.get("source_memory_ids", "").strip("[]").split(",")
            if i.strip()
        }
        blocks.append(f"### LEGACY PAGE: {fm.get('title') or path.stem}\n{body}")

    llm = await _make_llm_call(dreaming_settings)
    raw = await llm(MIGRATE_PROMPT.format(scope=scope, pages_block="\n\n".join(blocks)))
    from extensions.dreaming.prompts import parse_json_response

    topics = [
        t for t in parse_json_response(raw, key="topics")
        if (t.get("title") or "").strip() and (t.get("body") or "").strip()
    ]
    if not topics:
        return {"scope": scope, "topics": 0, "error": "LLM returned no topics"}

    target = vault / "Knowledge" if scope == "global" else vault / "Projects" / scope
    hub = None if scope == "global" else scope
    written = []
    for t in topics:
        title = t["title"].strip()
        page = render_topic_page(
            title=title,
            pool="migrated",
            summary=(t.get("summary") or "").strip(),
            body=t["body"],
            memory_ids=sorted(all_ids),
            categories=[],
            hub_link=hub,
            version=1,
        ).replace("---\n\n# ", "migrated: true\n---\n\n# ", 1)
        out_path = target / f"{_slug_title(title)}.md"
        if apply:
            _atomic_write(out_path, page)
        written.append(str(out_path))
    if apply and hub:
        from extensions.dreaming.librarian import _write_hub

        _write_hub(target, hub, _atomic_write)
    return {"scope": scope, "topics": len(written), "pages": written}


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vault", required=True)
    ap.add_argument("--apply", action="store_true", help="write pages + archive legacy tree")
    ap.add_argument("--only-scope", default=None)
    args = ap.parse_args()

    vault = Path(args.vault).expanduser().resolve()
    wiki = vault / "Wiki"
    if not wiki.exists():
        print(f"no Wiki/ tree under {vault} — nothing to migrate")
        return 0

    scopes = collect_scopes(wiki)
    if args.only_scope:
        scopes = {k: v for k, v in scopes.items() if k == args.only_scope}
    print(f"{'APPLY' if args.apply else 'DRY RUN'}: {len(scopes)} scopes, "
          f"{sum(len(v) for v in scopes.values())} legacy pages\n")

    results = []
    for scope, pages in sorted(scopes.items()):
        print(f"→ {scope} ({len(pages)} pages) …")
        try:
            res = await migrate_scope(scope, pages, vault, apply=args.apply)
        except Exception as exc:
            res = {"scope": scope, "topics": 0, "error": f"{exc.__class__.__name__}: {exc}"}
        results.append(res)
        print(f"  {res.get('topics', 0)} topic pages"
              + (f" — ERROR {res['error']}" if res.get("error") else ""))

    ok = [r for r in results if not r.get("error")]
    if args.apply and ok:
        from extensions.conversation_compiler.obsidian_writer import _atomic_write
        from extensions.dreaming.librarian import _write_home

        stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
        archive = vault / "_archive" / f"Wiki-pre-dreaming-{stamp}"
        archive.mkdir(parents=True, exist_ok=True)
        # Archive ONLY the scopes that migrated — failed scopes keep their
        # legacy pages under Wiki/ so a retry can pick them up (a wholesale
        # move would silently orphan them inside _archive/).
        for r in ok:
            src = wiki / r["scope"]
            if src.exists():
                shutil.move(str(src), str(archive / r["scope"]))
        if not any(wiki.rglob("*")):
            wiki.rmdir()
        _write_home(vault, _atomic_write)
        print(f"\nmigrated scopes archived → {archive}")

    print(f"\nmigrated {len(ok)}/{len(results)} scopes, "
          f"{sum(r.get('topics', 0) for r in ok)} topic pages")
    failed = [r for r in results if r.get("error")]
    for r in failed:
        print(f"  FAILED {r['scope']}: {r['error']}")
    return 1 if failed and args.apply else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
